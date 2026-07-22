from __future__ import annotations

from dataclasses import dataclass, replace
import sqlite3
from time import perf_counter
from typing import Any, Mapping

from .kernel import (
    ActionPolicy,
    GoalDirectedPolicy,
    RegistryPolicyProvider,
    SolveBudget,
    TypedExperienceCollector,
    UnifiedTypedReasoner,
    UnifiedTypedResult,
)
from .prompt_api import (
    AnswerEnvelope,
    AnswerStatus,
    PromptRequest,
    ResourceMetrics,
    SemanticProposal,
)
from .prompt_compiler import PromptCompilation, PromptCompiler
from .semantic_experience import SemanticTraceStore


@dataclass(frozen=True)
class AssistantConfig:
    budget: SolveBudget = SolveBudget()
    execute_model_proposals: bool = True


class SemOpAssistant:
    """Prompt-first facade that keeps semantic models outside the trust boundary."""

    def __init__(
        self,
        *,
        compiler: PromptCompiler | None = None,
        reasoner: UnifiedTypedReasoner | None = None,
        policy: ActionPolicy | RegistryPolicyProvider | None = None,
        experience_collector: TypedExperienceCollector | None = None,
        semantic_trace_store: SemanticTraceStore | None = None,
        config: AssistantConfig | None = None,
    ) -> None:
        self.compiler = compiler or PromptCompiler()
        self.reasoner = reasoner or UnifiedTypedReasoner()
        self.policy = policy or GoalDirectedPolicy()
        self.experience_collector = experience_collector
        self.semantic_trace_store = semantic_trace_store
        self.config = config or AssistantConfig()

    def solve(self, request: PromptRequest | str) -> AnswerEnvelope:
        prompt = request if isinstance(request, PromptRequest) else PromptRequest(request)
        started = perf_counter()
        compilation = self.compiler.compile(prompt)
        results: list[UnifiedTypedResult] = []
        runtime_errors: list[str] = []

        for proposal in compilation.proposals:
            if proposal.request is None:
                continue
            if not proposal.deterministic and not self.config.execute_model_proposals:
                continue
            try:
                results.append(self._run_typed(proposal))
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                runtime_errors.append(
                    f"{proposal.domain}: {type(exc).__name__}: {exc}"
                )

        status = _answer_status(compilation, tuple(results))
        answer = _render_answer(compilation, tuple(results), status, runtime_errors)
        proof = "\n\n".join(result.proof_ko for result in results if result.proof_ko)
        dependencies = _assumption_dependencies(tuple(results))
        unresolved = tuple(
            dict.fromkeys(
                (
                    *runtime_errors,
                    *(
                        diagnostic
                        for proposal in compilation.proposals
                        for diagnostic in proposal.diagnostics
                        if proposal.domain == "unsupported"
                    ),
                )
            )
        )
        typed_results = tuple(
            result.typed_result
            for result in results
            if result.typed_result is not None
        )
        expansions = sum(result.expansions for result in typed_results)
        controller_used = any(result.policy_used for result in typed_results)
        backend = self.compiler.backend
        model_loaded = bool(backend is not None and backend.loaded)
        model_id = compilation.model_id or (
            backend.model_id if backend is not None and model_loaded else ""
        )
        context_dependent = bool(
            prompt.conversation
            or prompt.recalled_memories
            or prompt.recalled_episodes
            or prompt.task_contexts
        )
        episode_retriever_used = any(
            proposal.producer_id == "episode_context_retriever"
            for proposal in compilation.proposals
        )
        memory_retriever_used = any(
            proposal.producer_id == "explicit_memory_retriever"
            for proposal in compilation.proposals
        )
        memory_context_missing = any(
            proposal.producer_id == "memory_context_guard"
            for proposal in compilation.proposals
        )
        provenance: dict[str, Any] = {
            "semantic_authority": (
                "model_proposal"
                if compilation.used_model
                else (
                    "memory_context_guard"
                    if memory_context_missing
                    else (
                        "explicit_user_memory_context"
                        if memory_retriever_used
                        else (
                            "recalled_episode_context"
                            if episode_retriever_used
                            else "deterministic_input"
                        )
                    )
                )
            ),
            "semantic_verified": not compilation.used_model
            and not episode_retriever_used
            and not memory_retriever_used,
            "logical_replay_verified": bool(typed_results)
            and all(
                result.verified for result in results if result.typed_result is not None
            ),
            "proposal_fingerprints": [
                proposal.fingerprint for proposal in compilation.proposals
            ],
            "repair_attempts": compilation.repair_attempts,
            "judge_used": False,
            "judge_opt_in": prompt.judge_opt_in,
            "conversation_messages_provided": len(prompt.conversation),
            "conversation_used_by_model": bool(
                prompt.conversation and compilation.used_model
            ),
            "recalled_memories_provided": len(prompt.recalled_memories),
            "recalled_memory_ids": [
                item.memory_id for item in prompt.recalled_memories
            ],
            "recalled_memory_retrievals": [
                item.retrieval for item in prompt.recalled_memories
            ],
            "memory_context_used_by_model": bool(
                prompt.recalled_memories and compilation.used_model
            ),
            "memory_context_used_by_retriever": memory_retriever_used,
            "memory_context_missing": memory_context_missing,
            "recalled_episodes_provided": len(prompt.recalled_episodes),
            "recalled_episode_ids": [
                item.episode_id for item in prompt.recalled_episodes
            ],
            "recalled_episode_scopes": [
                item.scope for item in prompt.recalled_episodes
            ],
            "episode_context_used_by_model": bool(
                prompt.recalled_episodes and compilation.used_model
            ),
            "episode_context_used_by_retriever": episode_retriever_used,
            "task_contexts_provided": len(prompt.task_contexts),
            "task_context_ids": [item.task_id for item in prompt.task_contexts],
            "task_context_used_by_model": bool(
                prompt.task_contexts and compilation.used_model
            ),
            "task_context_used_by_retriever": any(
                proposal.producer_id == "task_context_retriever"
                for proposal in compilation.proposals
            ),
            "answer_source": _answer_source(
                compilation,
                tuple(results),
                status,
            ),
        }
        generation_metrics = getattr(backend, "last_generation_metrics", {})
        if compilation.used_model and generation_metrics:
            provenance["semantic_generation"] = dict(generation_metrics)
        envelope = AnswerEnvelope(
            answer=answer,
            status=status,
            proposals=compilation.proposals,
            results=tuple(results),
            proof=proof,
            assumption_dependencies=dependencies,
            unresolved=unresolved,
            provenance=provenance,
            metrics=ResourceMetrics(
                elapsed_seconds=perf_counter() - started,
                model_loaded=model_loaded,
                model_id=model_id,
                controller_used=controller_used,
                expansions=expansions,
                peak_vram_bytes=(
                    _peak_vram_bytes() if compilation.used_model else None
                ),
            ),
        )
        if self.semantic_trace_store is not None and not context_dependent:
            try:
                trace = self.semantic_trace_store.capture(prompt, envelope)
            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                provenance["semantic_trace_error"] = f"{type(exc).__name__}: {exc}"
            else:
                if trace is not None:
                    provenance["semantic_trace_id"] = trace.trace_id
            envelope = replace(envelope, provenance=provenance)
        elif self.semantic_trace_store is not None and context_dependent:
            provenance["semantic_trace_skipped"] = (
                "context-dependent traces are not yet eligible for text distillation"
            )
            envelope = replace(envelope, provenance=provenance)
        return envelope

    def _run_typed(self, proposal: SemanticProposal) -> UnifiedTypedResult:
        assert proposal.request is not None
        if self.experience_collector is None:
            return self.reasoner.run(
                proposal.request,
                policy=self.policy,
                budget=self.config.budget,
            )
        observed = self.experience_collector.run(
            proposal.request,
            source="semop-prompt-assistant",
            policy=self.policy,
            budget=self.config.budget,
        )
        if observed.result is None:
            detail = observed.error_message or "typed experience execution failed"
            raise RuntimeError(f"{observed.error_type}: {detail}")
        return observed.result


def _answer_status(
    compilation: PromptCompilation,
    results: tuple[UnifiedTypedResult, ...],
) -> AnswerStatus:
    proposals = compilation.proposals
    if all(proposal.domain == "unsupported" for proposal in proposals):
        if any(
            proposal.producer_id
            in {
                "task_context_retriever",
                "episode_context_retriever",
                "explicit_memory_retriever",
            }
            and proposal.candidate_answer
            for proposal in proposals
        ):
            return AnswerStatus.BEST_EFFORT
        if compilation.used_model and any(
            not proposal.deterministic and proposal.candidate_answer
            for proposal in proposals
        ):
            return AnswerStatus.BEST_EFFORT
        return AnswerStatus.UNSUPPORTED
    if compilation.used_model:
        return AnswerStatus.BEST_EFFORT
    solved = tuple(
        result.typed_result
        for result in results
        if result.typed_result is not None and result.success and result.verified
    )
    if not solved or len(solved) != len(proposals):
        return AnswerStatus.UNSUPPORTED
    if any(
        result.conditional or result.unverified_dependencies
        for result in solved
    ):
        return AnswerStatus.CONDITIONAL
    return AnswerStatus.VERIFIED


def _render_answer(
    compilation: PromptCompilation,
    results: tuple[UnifiedTypedResult, ...],
    status: AnswerStatus,
    runtime_errors: list[str],
) -> str:
    if not results:
        candidate = next(
            (
                proposal.candidate_answer
                for proposal in compilation.proposals
                if proposal.candidate_answer
            ),
            "",
        )
        if candidate:
            return candidate
        if runtime_errors:
            return "typed 실행을 완료하지 못했어. " + runtime_errors[0]
        return "이 입력은 아직 지원되는 typed 문제로 해석하지 못했어."

    rendered = [_render_domain_result(result) for result in results]
    text = "\n\n".join(item for item in rendered if item)
    if status is AnswerStatus.BEST_EFFORT:
        replayed = tuple(
            result
            for result in results
            if result.success
            and result.typed_result is not None
            and result.typed_result.verified
        )
        if not replayed:
            candidate = _candidate_answer(compilation)
            if candidate:
                return candidate
        prefix = (
            "아래 답은 모델이 제안한 입력 해석을 전제로 계산했어. "
            "연산자 실행은 재검증됐지만 원문 해석 자체는 아직 미검증이야."
        )
        return f"{prefix}\n\n{text}"
    if status is AnswerStatus.CONDITIONAL:
        prefix = (
            "입력에 적힌 사실과 가정을 참이라고 둘 때의 검증된 결론이야."
        )
        return f"{prefix}\n\n{text}"
    if status is AnswerStatus.UNSUPPORTED:
        return text or "현재 연산자만으로 목표를 증명하지 못했어."
    return text


def _candidate_answer(compilation: PromptCompilation) -> str:
    return next(
        (
            proposal.candidate_answer
            for proposal in compilation.proposals
            if proposal.candidate_answer
        ),
        "",
    )


def _answer_source(
    compilation: PromptCompilation,
    results: tuple[UnifiedTypedResult, ...],
    status: AnswerStatus,
) -> str:
    if any(
        proposal.producer_id == "memory_context_guard"
        for proposal in compilation.proposals
    ):
        return "memory_context_missing"
    if status in {AnswerStatus.VERIFIED, AnswerStatus.CONDITIONAL}:
        return "typed_executor"
    if status is AnswerStatus.BEST_EFFORT:
        if any(
            proposal.producer_id == "task_context_retriever"
            for proposal in compilation.proposals
        ):
            return "task_context"
        if any(
            proposal.producer_id == "explicit_memory_retriever"
            for proposal in compilation.proposals
        ):
            return "explicit_memory"
        if any(
            proposal.producer_id == "episode_context_retriever"
            for proposal in compilation.proposals
        ):
            return "episodic_context"
        replayed = any(
            result.success
            and result.typed_result is not None
            and result.typed_result.verified
            for result in results
        )
        if replayed:
            return "model_grounded_typed_executor"
        if _candidate_answer(compilation):
            return "model_candidate"
        return "model_interpretation"
    return "system_message"


def _render_domain_result(result: UnifiedTypedResult) -> str:
    metadata: Mapping[str, Any] = (
        result.instance.metadata if result.instance is not None else {}
    )
    if result.domain.value == "math":
        input_kind = str(metadata.get("input_kind", ""))
        if input_kind == "quadratic_equation":
            variable = str(metadata.get("variable", "x")).strip() or "x"
            answers = tuple(str(item) for item in metadata.get("answers", ()))
            if not answers:
                return f"검증 결과, {variable}에 대한 실수해가 없어."
            if len(answers) == 1:
                return f"검증된 실수해는 {variable} = {answers[0]}이야."
            rendered = ", ".join(f"{variable} = {item}" for item in answers)
            return f"검증된 실수해는 {rendered}이야."
        if input_kind == "numeric_comparison":
            truth = bool(metadata.get("truth_value"))
            return f"비교 결과는 {'참' if truth else '거짓'}이야."
        answer = str(metadata.get("answer", "")).strip()
        variable = str(metadata.get("variable", "")).strip()
        if answer and variable:
            return f"검증 결과는 {variable} = {answer}이야."
        if answer:
            return f"검증된 계산 결과는 {answer}이야."
    if result.domain.value == "language":
        predicate = _first_goal_predicate(result)
        if result.success and predicate == "READY":
            return "입력된 조건을 기준으로 목표를 진행할 수 있어."
        if result.success and predicate == "NOT_READY":
            return "입력된 조건을 기준으로 목표는 아직 준비되지 않았어."
        return "입력된 전제만으로는 목표 상태를 증명하지 못했어."
    if result.domain.value == "vision":
        if result.instance is not None:
            payload = result.instance.metadata.get("raster_reasoning", {})
            if isinstance(payload, Mapping):
                counts = dict(payload.get("color_counts", ()))
                total = int(payload.get("closed_world_component_count", 0))
                if result.success and "몇 개" in str(metadata.get("query", "")):
                    if counts:
                        detail = ", ".join(
                            f"{name} {count}개" for name, count in sorted(counts.items())
                        )
                        return f"검증된 객체는 모두 {total}개야 ({detail})."
                    return f"검증된 객체는 모두 {total}개야."
        return (
            "픽셀 측정으로 질문의 관계를 검증했어."
            if result.success
            else "픽셀 측정으로는 질문의 관계를 증명하지 못했어."
        )
    if result.domain.value == "coding":
        category = str(metadata.get("category", "알고리즘"))
        code = str(metadata.get("code", "")).strip()
        if result.success and code:
            return (
                f"{category} 풀이를 컴파일하고 등록된 테스트로 검증했어.\n\n"
                f"```cpp\n{code}\n```"
            )
        if code:
            return f"{category} 후보 코드를 만들었지만 검증을 통과하지 못했어."
        return "코딩 후보를 검증하지 못했어."
    return "목표를 검증했어." if result.success else "목표를 증명하지 못했어."


def _first_goal_predicate(result: UnifiedTypedResult) -> str:
    if result.typed_result is None or not result.typed_result.goals:
        return ""
    return result.typed_result.goals[0].goal.atom.predicate.name


def _assumption_dependencies(
    results: tuple[UnifiedTypedResult, ...],
) -> tuple[str, ...]:
    dependencies: list[str] = []
    for result in results:
        if result.typed_result is None:
            continue
        dependencies.extend(
            str(fact) for fact in result.typed_result.assumption_dependencies
        )
        dependencies.extend(
            str(fact) for fact in result.typed_result.unverified_dependencies
        )
    return tuple(dict.fromkeys(dependencies))


def _peak_vram_bytes() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())

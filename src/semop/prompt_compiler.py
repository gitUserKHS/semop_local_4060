from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .kernel import (
    CodingProblem,
    LanguageTextParser,
    LanguageTextProblem,
    MathInputAdapter,
    RasterImage,
    RasterVisionAdapter,
    RasterVisionProblem,
    SceneThresholdParser,
    SceneThresholdProblem,
    TypedDomainRequest,
    VisionAreaGoal,
    VisionCountGoal,
    VisionPropertyGoal,
    VisionRelationGoal,
)
from .prompt_api import (
    CognitiveOperator,
    ImageRegion,
    PromptImage,
    PromptRequest,
    RecalledEpisode,
    RecalledMemory,
    ResourceTier,
    SemanticProposal,
    SourceSpan,
)
from .task_memory import TaskQueryKind, task_query_kind


@runtime_checkable
class OperatorRetriever(Protocol):
    def retrieve(self, query: str, *, limit: int = 8) -> tuple[str, ...]: ...


@runtime_checkable
class SemanticMemoryRetriever(Protocol):
    def retrieve(
        self,
        request: PromptRequest,
        *,
        limit: int = 3,
    ) -> tuple[str, ...]: ...


@runtime_checkable
class SemanticProposalBackend(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def loaded(self) -> bool: ...

    def generate(
        self,
        request: PromptRequest,
        *,
        operator_hints: Sequence[str] = (),
        repair_hint: str = "",
    ) -> str | Mapping[str, Any]: ...


@runtime_checkable
class VisionSidecarBackend(Protocol):
    def analyze(
        self,
        image: PromptImage,
        *,
        tasks: Sequence[str] = ("objects", "ocr"),
    ) -> Sequence[Any]: ...


@dataclass(frozen=True)
class PromptCompilation:
    proposals: tuple[SemanticProposal, ...]
    used_model: bool = False
    model_id: str = ""
    repair_attempts: int = 0

    def __post_init__(self) -> None:
        if not self.proposals:
            raise ValueError("prompt compilation requires at least one proposal")


class PromptCompiler:
    """High-precision compiler with a strictly untrusted model fallback."""

    def __init__(
        self,
        *,
        backend: SemanticProposalBackend | None = None,
        retriever: OperatorRetriever | None = None,
        memory_retriever: SemanticMemoryRetriever | None = None,
        vision_sidecar: VisionSidecarBackend | None = None,
        repair_attempts: int = 2,
    ) -> None:
        if repair_attempts < 0 or repair_attempts > 2:
            raise ValueError("semantic repair_attempts must be between 0 and 2")
        if backend is not None and not isinstance(backend, SemanticProposalBackend):
            raise TypeError("semantic backend must implement generate")
        if retriever is not None and not isinstance(retriever, OperatorRetriever):
            raise TypeError("operator retriever must implement retrieve")
        if memory_retriever is not None and not isinstance(
            memory_retriever,
            SemanticMemoryRetriever,
        ):
            raise TypeError("semantic memory retriever must implement retrieve")
        if vision_sidecar is not None and not isinstance(
            vision_sidecar, VisionSidecarBackend
        ):
            raise TypeError("vision sidecar must implement analyze")
        self.backend = backend
        self.retriever = retriever
        self.memory_retriever = memory_retriever
        self.vision_sidecar = vision_sidecar
        self.repair_attempts = repair_attempts

    def compile(self, request: PromptRequest) -> PromptCompilation:
        task_query = task_query_kind(
            request.text,
            allow_implicit=bool(request.task_contexts),
        )
        if task_query is not None:
            if not request.task_contexts:
                return PromptCompilation((self._missing_task_context(request),))
            return PromptCompilation(
                (self._selected_task_context(request, task_query),)
            )
        if _is_direct_explicit_memory_recall_request(request):
            return PromptCompilation((self._selected_explicit_memory_answer(request),))
        if _is_direct_episode_recall_request(request):
            return PromptCompilation((self._selected_episode_answer(request),))
        if _is_missing_personal_memory_request(request):
            return PromptCompilation((self._missing_memory_context(request),))
        deterministic = self._compile_deterministic(request)
        if deterministic:
            return PromptCompilation(tuple(deterministic))

        if request.resource_tier is ResourceTier.SYMBOLIC or self.backend is None:
            return PromptCompilation((self._unsupported(request),))

        hints: tuple[str, ...] = ()
        if self.retriever is not None and request.text and not request.images:
            try:
                hints = self.retriever.retrieve(request.text, limit=8)
            except (ImportError, OSError, TypeError, ValueError, RuntimeError):
                # Retrieval is guidance only. A missing encoder must not widen the
                # model's authority or break the deterministic fallback.
                hints = ()
        if self.memory_retriever is not None:
            try:
                hints = tuple(
                    (*hints, *self.memory_retriever.retrieve(request, limit=3))
                )
            except (ImportError, OSError, TypeError, ValueError, RuntimeError):
                # Reviewed memory is optional guidance. It cannot make a request fail
                # or turn a model proposal into a verified fact.
                pass
        errors: list[str] = []
        last_candidate_fallback: SemanticProposal | None = None
        repair_limit = _semantic_repair_limit(request, self.repair_attempts)
        for attempt in range(repair_limit + 1):
            repair_hint = errors[-1] if errors else ""
            raw: str | Mapping[str, Any] | None = None
            try:
                raw = self.backend.generate(
                    request,
                    operator_hints=hints,
                    repair_hint=repair_hint,
                )
                proposal = self._decode_model_proposal(request, raw)
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                diagnostic = f"{type(exc).__name__}: {exc}"
                errors.append(diagnostic)
                fallback = self._unverified_candidate_fallback(
                    request,
                    raw,
                    diagnostic=diagnostic,
                )
                if fallback is not None:
                    contract_error = _candidate_answer_contract_error(
                        request,
                        fallback.candidate_answer,
                    )
                    fallback = replace(
                        fallback,
                        diagnostics=(
                            "typed model payload was rejected; the answer is "
                            "retained only as unverified chat output",
                            *fallback.diagnostics,
                            *((contract_error,) if contract_error else ()),
                        ),
                        fingerprint="",
                    )
                    last_candidate_fallback = fallback
                    if contract_error and attempt < repair_limit:
                        errors.append(contract_error)
                        continue
                    return PromptCompilation(
                        (fallback,),
                        used_model=True,
                        model_id=self.backend.model_id,
                        repair_attempts=attempt,
                    )
                continue
            contract_error = _candidate_answer_contract_error(
                request,
                proposal.candidate_answer,
            )
            if contract_error:
                errors.append(contract_error)
                last_candidate_fallback = replace(
                    proposal,
                    domain="unsupported",
                    request=None,
                    diagnostics=(*proposal.diagnostics, contract_error),
                    fingerprint="",
                )
                if attempt < repair_limit:
                    continue
                proposal = replace(
                    proposal,
                    diagnostics=(*proposal.diagnostics, contract_error),
                    fingerprint="",
                )
            if (
                proposal.domain == "vision"
                and proposal.request is None
                and not proposal.image_regions
                and request.images
                and self.vision_sidecar is not None
            ):
                proposal = self._attach_vision_sidecar(request, proposal)
            return PromptCompilation(
                (proposal,),
                used_model=True,
                model_id=self.backend.model_id,
                repair_attempts=attempt,
            )
        if last_candidate_fallback is not None:
            return PromptCompilation(
                (
                    replace(
                        last_candidate_fallback,
                        diagnostics=(
                            *last_candidate_fallback.diagnostics,
                            "later bounded repair attempts failed; preserved the last "
                            "usable answer as unverified best effort",
                            *errors,
                        ),
                        fingerprint="",
                    ),
                ),
                used_model=True,
                model_id=self.backend.model_id,
                repair_attempts=repair_limit,
            )
        return PromptCompilation(
            (
                self._unsupported(
                    request,
                    diagnostics=(
                        "semantic model output was rejected after bounded repair",
                        *errors,
                    ),
                ),
            ),
            used_model=True,
            model_id=self.backend.model_id,
            repair_attempts=repair_limit,
        )

    def _attach_vision_sidecar(
        self,
        request: PromptRequest,
        proposal: SemanticProposal,
    ) -> SemanticProposal:
        assert self.vision_sidecar is not None
        try:
            candidates = tuple(self.vision_sidecar.analyze(request.images[0]))
        except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
            return replace(
                proposal,
                diagnostics=(
                    *proposal.diagnostics,
                    f"vision sidecar unavailable: {type(exc).__name__}: {exc}",
                ),
                fingerprint="",
            )
        regions = _sidecar_image_regions(candidates)
        summaries = tuple(
            json.dumps(
                dict(getattr(candidate, "payload", {})),
                ensure_ascii=False,
                sort_keys=True,
            )[:4_000]
            for candidate in candidates
        )
        return replace(
            proposal,
            image_regions=tuple((*proposal.image_regions, *regions)),
            diagnostics=(
                *proposal.diagnostics,
                "Florence OCR/object regions remain unverified PROPOSED evidence",
                *summaries,
            ),
            fingerprint="",
        )

    def _compile_deterministic(
        self,
        request: PromptRequest,
    ) -> list[SemanticProposal]:
        text, explicit_domain = _strip_domain_prefix(request.text)
        if request.images:
            vision = _compile_raster_vision(text, request.images[0])
            if vision is not None:
                return [
                    _deterministic_proposal(
                        "vision",
                        vision,
                        request.text,
                        (
                            CognitiveOperator.DECOMPOSE,
                            CognitiveOperator.SELECT,
                            CognitiveOperator.INFER,
                            CognitiveOperator.VERIFY,
                            CognitiveOperator.EXPLAIN,
                        ),
                        confidence=1.0,
                    )
                ]
            if explicit_domain == "vision":
                return []

        math_expression = _normalize_math_prompt(text)
        if explicit_domain == "math" or (
            explicit_domain is None and math_expression is not None
        ):
            if math_expression is None:
                return []
            try:
                MathInputAdapter().adapt(math_expression)
            except (TypeError, ValueError):
                return []
            return [
                _deterministic_proposal(
                    "math",
                    TypedDomainRequest("math", math_expression, mode="shadow"),
                    request.text,
                    (
                        CognitiveOperator.DECOMPOSE,
                        CognitiveOperator.INFER,
                        CognitiveOperator.VERIFY,
                        CognitiveOperator.EXPLAIN,
                    ),
                    confidence=1.0,
                )
            ]

        if _looks_like_coding_problem(text):
            try:
                problem = CodingProblem(text)
            except ValueError:
                return []
            return [
                _deterministic_proposal(
                    "coding",
                    TypedDomainRequest("coding", problem, mode="shadow"),
                    request.text,
                    (
                        CognitiveOperator.DECOMPOSE,
                        CognitiveOperator.RETRIEVE,
                        CognitiveOperator.COMPOSE,
                        CognitiveOperator.VERIFY,
                        CognitiveOperator.REPAIR,
                        CognitiveOperator.EXPLAIN,
                    ),
                    confidence=0.99,
                )
            ]

        parsed = LanguageTextParser().parse(
            LanguageTextProblem(text, use_legacy_heuristics=False)
        )
        explicit_claims = tuple(claim for claim in parsed.claims if claim.verified)
        has_goal = any(claim.relation == "GOAL" for claim in explicit_claims)
        has_requirement = any(
            claim.relation == "REQUIRES" for claim in explicit_claims
        )
        if explicit_domain == "language" or (has_goal and has_requirement):
            if not has_goal or not has_requirement:
                return []
            problem = LanguageTextProblem(
                text,
                source_context="semop_prompt_compiler",
                use_legacy_heuristics=False,
            )
            return [
                _deterministic_proposal(
                    "language",
                    TypedDomainRequest("language", problem, mode="shadow"),
                    request.text,
                    (
                        CognitiveOperator.DECOMPOSE,
                        CognitiveOperator.ALIGN,
                        CognitiveOperator.INFER,
                        CognitiveOperator.VERIFY,
                        CognitiveOperator.EXPLAIN,
                    ),
                    confidence=1.0,
                )
            ]
        return []

    def _decode_model_proposal(
        self,
        prompt: PromptRequest,
        raw: str | Mapping[str, Any],
    ) -> SemanticProposal:
        payload = dict(raw) if isinstance(raw, Mapping) else _decode_json_object(raw)
        domain = str(payload.get("domain", "unsupported")).strip().lower()
        confidence = float(payload.get("confidence", 0.0))
        decoded_candidate_answer = _decode_candidate_answer(payload)
        augmented_candidate_answer = _augment_general_python_answer(
            prompt,
            payload,
            decoded_candidate_answer,
        )
        candidate_answer = _strip_unsolicited_python_blocks(
            prompt,
            augmented_candidate_answer,
        )
        assembled_python_example = (
            augmented_candidate_answer != decoded_candidate_answer
        )
        removed_unsolicited_python = candidate_answer != augmented_candidate_answer
        operator_program = _decode_operator_program(payload.get("operator_program"))
        source_spans = _decode_source_spans(payload.get("source_spans"), prompt.text)
        image_regions = _decode_image_regions(
            payload.get("image_regions"),
            len(prompt.images),
        )
        prompt_text, _explicit_domain = _strip_domain_prefix(prompt.text)
        general_coding = (
            _looks_like_general_coding_request(prompt_text)
            and not _looks_like_coding_problem(prompt_text)
            and domain in {"coding", "language", "unsupported"}
        )
        typed_request = (
            None
            if general_coding
            else _typed_request_from_model_payload(
                domain,
                payload.get("payload"),
                prompt,
            )
        )
        if typed_request is None and not candidate_answer:
            raise ValueError("model proposal needs a typed payload or candidate answer")
        proposal = SemanticProposal(
            domain=domain,
            operator_program=operator_program,
            confidence=confidence,
            producer_id="local_semantic_model",
            request=typed_request,
            candidate_answer=candidate_answer,
            source_spans=source_spans,
            image_regions=image_regions,
            diagnostics=(
                "model semantics remain proposed even when typed execution succeeds",
                *(
                    (
                        "assembled an AST-checked Python example from model-proposed "
                        "source without executing it",
                    )
                    if assembled_python_example
                    else ()
                ),
                *(
                    (
                        "removed an unsolicited Python block from a non-coding "
                        "model answer",
                    )
                    if removed_unsolicited_python
                    else ()
                ),
            ),
            model_id=self.backend.model_id if self.backend is not None else "",
            resource_tier=prompt.resource_tier,
            deterministic=False,
        )
        return _stage_general_coding_candidate(prompt, proposal)

    def _unverified_candidate_fallback(
        self,
        prompt: PromptRequest,
        raw: str | Mapping[str, Any] | None,
        *,
        diagnostic: str,
    ) -> SemanticProposal | None:
        """Keep useful chat text while discarding an invalid typed interpretation."""

        if raw is None:
            return None
        recovered_code = False
        try:
            payload = dict(raw) if isinstance(raw, Mapping) else _decode_json_object(raw)
            domain = str(payload.get("domain", "")).strip().lower()
            decoded_candidate_answer = _decode_candidate_answer(payload)
            augmented_candidate_answer = _augment_general_python_answer(
                prompt,
                payload,
                decoded_candidate_answer,
            )
            recovered_code = augmented_candidate_answer != decoded_candidate_answer
            candidate_answer = _strip_unsolicited_python_blocks(
                prompt,
                augmented_candidate_answer,
            )
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            if not isinstance(raw, str):
                return None
            domain = _recover_json_string_field(raw, "domain").strip().lower()
            candidate_answer = _recover_json_string_field(raw, "answer").strip()
            confidence = _recover_json_number_field(raw, "confidence")
            recovered_answer = _augment_general_python_answer_from_raw(
                prompt,
                raw,
                candidate_answer,
            )
            recovered_code = recovered_answer != candidate_answer
            candidate_answer = _strip_unsolicited_python_blocks(
                prompt,
                recovered_answer,
            )
        if domain not in {
            "math",
            "language",
            "coding",
            "vision",
            "composed",
            "unsupported",
        } or not candidate_answer:
            return None
        return SemanticProposal(
            domain="unsupported",
            operator_program=(CognitiveOperator.EXPLAIN,),
            confidence=min(1.0, max(0.0, confidence)),
            producer_id="local_semantic_model",
            candidate_answer=candidate_answer,
            source_spans=(
                (SourceSpan(0, len(prompt.text), prompt.text),) if prompt.text else ()
            ),
            diagnostics=(
                diagnostic,
                *(
                    (
                        "recovered an AST-valid Python example from the malformed "
                        "model payload without executing it",
                    )
                    if recovered_code
                    else ()
                ),
            ),
            model_id=self.backend.model_id if self.backend is not None else "",
            resource_tier=prompt.resource_tier,
            deterministic=False,
        )

    @staticmethod
    def _selected_task_context(
        request: PromptRequest,
        query_kind: TaskQueryKind,
    ) -> SemanticProposal:
        task = request.task_contexts[0]
        if query_kind is TaskQueryKind.SUMMARY:
            decisions = ", ".join(task.decisions) if task.decisions else "저장된 결정 없음"
            if task.next_actions:
                next_actions = ", ".join(task.next_actions)
            elif task.status == "completed":
                next_actions = "없음 (완료됨)"
            else:
                next_actions = "저장된 다음 행동 없음"
            answer = "\n".join(
                (
                    f"{task.title} (revision {task.revision}, {task.status})",
                    f"목표: {task.objective}",
                    f"진행: {task.progress or '저장된 진행 내용 없음'}",
                    f"결정: {decisions}",
                    f"다음 행동: {next_actions}",
                )
            )
        elif query_kind is TaskQueryKind.DECISIONS:
            answer = (
                "결정:\n" + "\n".join(f"- {item}" for item in task.decisions)
                if task.decisions
                else "선택한 장기 작업에는 저장된 결정이 아직 없어."
            )
        elif task.status == "completed":
            answer = "선택한 장기 작업은 이미 완료됐어."
        elif task.next_actions:
            answer = task.next_actions[0]
        else:
            answer = "선택한 장기 작업에는 저장된 다음 행동이 아직 없어."
        return SemanticProposal(
            domain="unsupported",
            operator_program=(
                CognitiveOperator.RETRIEVE,
                CognitiveOperator.SELECT,
                CognitiveOperator.EXPLAIN,
            ),
            confidence=1.0,
            producer_id="task_context_retriever",
            candidate_answer=answer,
            source_spans=(SourceSpan(0, len(request.text), request.text),),
            diagnostics=(
                f"selected task revision {task.revision} supplied {query_kind.value}",
            ),
            resource_tier=request.resource_tier,
            deterministic=True,
        )

    @staticmethod
    def _selected_episode_answer(request: PromptRequest) -> SemanticProposal:
        episode = max(request.recalled_episodes, key=lambda item: item.score)
        return SemanticProposal(
            domain="unsupported",
            operator_program=(
                CognitiveOperator.RETRIEVE,
                CognitiveOperator.SELECT,
                CognitiveOperator.EXPLAIN,
            ),
            confidence=episode.score,
            producer_id="episode_context_retriever",
            candidate_answer=_direct_episode_answer(request, episode),
            source_spans=(SourceSpan(0, len(request.text), request.text),),
            diagnostics=(
                f"selected {episode.scope} episode as unverified recall context",
            ),
            resource_tier=request.resource_tier,
            deterministic=True,
        )

    @staticmethod
    def _selected_explicit_memory_answer(request: PromptRequest) -> SemanticProposal:
        memory = max(request.recalled_memories, key=lambda item: item.score)
        return SemanticProposal(
            domain="unsupported",
            operator_program=(
                CognitiveOperator.RETRIEVE,
                CognitiveOperator.SELECT,
                CognitiveOperator.EXPLAIN,
            ),
            confidence=memory.score,
            producer_id="explicit_memory_retriever",
            candidate_answer=_direct_memory_answer(request, memory),
            source_spans=(SourceSpan(0, len(request.text), request.text),),
            diagnostics=(
                "selected an explicit user-saved memory as unverified recall context",
            ),
            resource_tier=request.resource_tier,
            deterministic=True,
        )

    @staticmethod
    def _missing_memory_context(request: PromptRequest) -> SemanticProposal:
        return SemanticProposal(
            domain="unsupported",
            operator_program=(
                CognitiveOperator.RETRIEVE,
                CognitiveOperator.ASK,
            ),
            confidence=1.0,
            producer_id="memory_context_guard",
            candidate_answer=(
                "관련된 저장 기억이나 과거 대화를 찾지 못했어. 계속 기억할 "
                "내용은 `기억해:`로 저장해 줘."
            ),
            source_spans=(SourceSpan(0, len(request.text), request.text),),
            diagnostics=(
                "personal recall requested without retrieved memory context",
            ),
            resource_tier=request.resource_tier,
            deterministic=True,
        )

    @staticmethod
    def _missing_task_context(request: PromptRequest) -> SemanticProposal:
        return SemanticProposal(
            domain="unsupported",
            operator_program=(CognitiveOperator.ASK,),
            confidence=1.0,
            producer_id="prompt_compiler",
            candidate_answer=(
                "현재 채팅에 선택된 장기 작업이 없어. 장기 작업 관리에서 이어갈 "
                "작업을 먼저 선택해 줘."
            ),
            source_spans=(
                (SourceSpan(0, len(request.text), request.text),)
                if request.text
                else ()
            ),
            diagnostics=("selected task context is required for task continuation",),
            resource_tier=request.resource_tier,
            deterministic=True,
        )

    @staticmethod
    def _unsupported(
        request: PromptRequest,
        *,
        diagnostics: tuple[str, ...] = (),
    ) -> SemanticProposal:
        return SemanticProposal(
            domain="unsupported",
            operator_program=(CognitiveOperator.ASK,),
            confidence=1.0,
            producer_id="prompt_compiler",
            candidate_answer=(
                "이 입력은 아직 안전한 typed 요청으로 해석하지 못했어. "
                "수학식, 명시적인 목표·조건, 코딩 문제 또는 작은 색상 이미지를 "
                "조금 더 구체적으로 적어 줘."
            ),
            source_spans=(
                (SourceSpan(0, len(request.text), request.text),)
                if request.text
                else ()
            ),
            diagnostics=diagnostics,
            resource_tier=request.resource_tier,
            deterministic=True,
        )


def _deterministic_proposal(
    domain: str,
    typed_request: TypedDomainRequest,
    source_text: str,
    operators: tuple[CognitiveOperator, ...],
    *,
    confidence: float,
) -> SemanticProposal:
    return SemanticProposal(
        domain=domain,
        operator_program=operators,
        confidence=confidence,
        producer_id="deterministic_prompt_compiler",
        request=typed_request,
        source_spans=(
            (SourceSpan(0, len(source_text), source_text),) if source_text else ()
        ),
        resource_tier=ResourceTier.SYMBOLIC,
        deterministic=True,
    )


_PREFIXES = {
    "math": "math",
    "수학": "math",
    "language": "language",
    "언어": "language",
    "logic": "language",
    "논리": "language",
    "coding": "coding",
    "code": "coding",
    "코딩": "coding",
    "vision": "vision",
    "비전": "vision",
}


def _strip_domain_prefix(text: str) -> tuple[str, str | None]:
    match = re.match(
        r"^\s*/?(?P<domain>math|수학|language|언어|logic|논리|coding|code|코딩|vision|비전)\s*[:：]?\s*(?P<body>.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return text.strip(), None
    return match.group("body").strip(), _PREFIXES[match.group("domain").lower()]


def _normalize_math_prompt(text: str) -> str | None:
    clean = _normalize_math_symbols(text.strip())
    if not clean:
        return None
    replacements = (
        (r"더하기", "+"),
        (r"빼기", "-"),
        (r"곱하기", "*"),
        (r"나누기", "/"),
        (r"같다", "="),
    )
    normalized = clean
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)

    colloquial = re.fullmatch(
        r"(?:어떤\s*수|미지수|(?P<variable>[A-Za-z]))에\s*"
        r"(?P<amount>[+-]?(?:\d+(?:\.\d+)?|\d+/\d+))\s*(?:을|를)?\s*"
        r"(?P<operation>더하면|빼면)\s*"
        r"(?P<answer>[+-]?(?:\d+(?:\.\d+)?|\d+/\d+))(?:야|이야|이다|가\s*돼|가\s*된다)?[?.!\s]*",
        clean,
    )
    if colloquial:
        variable = colloquial.group("variable") or "x"
        operator = "+" if colloquial.group("operation") == "더하면" else "-"
        return f"{variable} {operator} {colloquial.group('amount')} = {colloquial.group('answer')}"

    binary = re.fullmatch(
        r"(?P<left>[+-]?(?:\d+(?:\.\d+)?|\d+/\d+))\s*(?:와|과)\s*"
        r"(?P<right>[+-]?(?:\d+(?:\.\d+)?|\d+/\d+))\s*(?:을|를)?\s*"
        r"(?P<operation>더해|빼|곱해|나눠)(?:\s*줘)?[?.!\s]*",
        clean,
    )
    if binary:
        symbol = {"더해": "+", "빼": "-", "곱해": "*", "나눠": "/"}[
            binary.group("operation")
        ]
        return f"{binary.group('left')} {symbol} {binary.group('right')}"

    normalized = re.sub(
        r"^(?:계산|검산|풀어)(?:해|해줘|해\s*줘)?\s*[:：]?\s*",
        "",
        normalized,
    )
    normalized = re.sub(
        r"\s*(?:은|는)?\s*(?:얼마(?:야|인가요?)?|계산해\s*줘|풀어\s*줘)[?.!\s]*$",
        "",
        normalized,
    ).strip()
    if not re.fullmatch(r"[0-9A-Za-z+\-*/^().=<>!\s]+", normalized):
        return None
    if not re.search(r"\d", normalized):
        return None
    if not re.search(r"[+\-*/=<>]", normalized):
        return None
    return normalized


def _looks_like_coding_problem(text: str) -> bool:
    lowered = text.casefold()
    signals = (
        "input",
        "output",
        "constraints",
        "time limit",
        "입력",
        "출력",
        "제한",
        "알고리즘",
        "c++",
        "cpp",
    )
    return len(text) >= 40 and sum(signal in lowered for signal in signals) >= 2


def _looks_like_general_coding_request(text: str) -> bool:
    lowered = text.casefold()
    signals = (
        "python",
        "javascript",
        "typescript",
        "rust",
        "java ",
        "코딩",
        "프로그래밍",
        "함수",
        "리스트 컴프리헨션",
        "for 루프",
    )
    return any(signal in lowered for signal in signals) or bool(
        re.search(r"코드(?!명)", lowered)
    )


def _is_direct_episode_recall_request(request: PromptRequest) -> bool:
    if request.images or not request.recalled_episodes:
        return False
    if max(item.score for item in request.recalled_episodes) < 0.55:
        return False
    return _has_direct_recall_cue(request.text)


def _is_direct_explicit_memory_recall_request(request: PromptRequest) -> bool:
    if request.images or not request.recalled_memories:
        return False
    if max(item.score for item in request.recalled_memories) < 0.18:
        return False
    return _has_direct_recall_cue(request.text)


def _is_missing_personal_memory_request(request: PromptRequest) -> bool:
    if request.images or request.recalled_memories or request.recalled_episodes:
        return False
    if not _has_direct_recall_cue(request.text):
        return False
    text = request.text.casefold()
    korean_markers = (
        "내 ",
        "내가",
        "나의",
        "우리",
        "저장",
        "기억",
        "전에",
        "아까",
        "지난 대화",
        "이번 실험",
    )
    if any(marker in text for marker in korean_markers):
        return True
    return bool(
        re.search(
            r"\b(?:my|our|we|saved|remember|earlier|previous conversation)\b",
            text,
        )
    )


def _has_direct_recall_cue(text: str) -> bool:
    normalized = text.casefold()
    korean_cues = (
        "뭐였지",
        "무엇이었지",
        "뭐라고 했지",
        "뭐라고 말했지",
        "기억나?",
        "기억나니",
    )
    if any(cue in normalized for cue in korean_cues):
        return True
    return bool(
        re.search(r"\bdo you remember\b", normalized)
        or re.search(
            r"\bwhat (?:was|did) .{0,48}\b(?:again|before|earlier)\b",
            normalized,
        )
    )


def _direct_episode_answer(
    request: PromptRequest,
    episode: RecalledEpisode,
) -> str:
    code_name = _requested_code_name(
        request,
        episode.user_content,
        episode.assistant_content,
    )
    return code_name or episode.assistant_content


def _direct_memory_answer(
    request: PromptRequest,
    memory: RecalledMemory,
) -> str:
    code_name = _requested_code_name(request, memory.content)
    return code_name or memory.content


def _requested_code_name(
    request: PromptRequest,
    *sources: str,
) -> str:
    text = request.text.casefold()
    code_name_only = (
        ("코드명" in text and "만" in text)
        or bool(re.search(r"\b(?:only.*code name|code name.*only)\b", text))
    )
    if not code_name_only:
        return ""
    patterns = (
        re.compile(r"코드명(?:은|는)?\s*[:：]?\s*([^\s,.!?]+)", re.IGNORECASE),
        re.compile(
            r"\bcode\s+name(?:\s+is)?\s*[:：]?\s*([^\s,.!?]+)",
            re.IGNORECASE,
        ),
    )
    for source in sources:
        for pattern in patterns:
            match = pattern.search(source)
            if match is None:
                continue
            token = match.group(1).strip("'\"`()[]{}")
            for suffix in ("입니다", "이라고", "이야", "이다", "야"):
                if token.endswith(suffix) and len(token) > len(suffix):
                    token = token[: -len(suffix)]
                    break
            if token:
                return token
    return ""


def _semantic_repair_limit(request: PromptRequest, configured: int) -> int:
    text, _explicit_domain = _strip_domain_prefix(request.text)
    if _looks_like_general_coding_request(text) and not _looks_like_coding_problem(text):
        return min(configured, 1)
    return configured


def _stage_general_coding_candidate(
    prompt: PromptRequest,
    proposal: SemanticProposal,
) -> SemanticProposal:
    """Keep prose coding help outside the C++/controlled-logic verifier boundary."""

    text, _explicit_domain = _strip_domain_prefix(prompt.text)
    if (
        not _looks_like_general_coding_request(text)
        or _looks_like_coding_problem(text)
        or proposal.domain not in {"coding", "language"}
        or not proposal.candidate_answer
    ):
        return proposal
    return replace(
        proposal,
        domain="unsupported",
        request=None,
        operator_program=(
            CognitiveOperator.DECOMPOSE,
            CognitiveOperator.RETRIEVE,
            CognitiveOperator.COMPARE,
            CognitiveOperator.EXPLAIN,
        ),
        diagnostics=(
            *proposal.diagnostics,
            "general coding guidance remains an unverified model answer; only "
            "explicit C++ contest artifacts enter the current coding verifier",
        ),
        fingerprint="",
    )


_FENCED_BLOCK_PATTERN = re.compile(
    r"^[ \t]*```(?P<label>[A-Za-z0-9_+.-]*)[ \t]*\r?\n"
    r"(?P<code>.*?)^[ \t]*```[ \t]*$",
    flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def _python_fenced_blocks(text: str) -> tuple[tuple[re.Match[str], str], ...]:
    blocks: list[tuple[re.Match[str], str]] = []
    for match in _FENCED_BLOCK_PATTERN.finditer(text):
        if match.group("label").casefold() not in {"", "python", "py", "text"}:
            continue
        source = match.group("code").strip()
        if not source:
            continue
        try:
            ast.parse(source)
        except SyntaxError:
            continue
        blocks.append((match, source))
    return tuple(blocks)


def _remove_python_fenced_blocks(text: str) -> str:
    blocks = _python_fenced_blocks(text)
    if not blocks:
        return text.strip()
    parts: list[str] = []
    cursor = 0
    for match, _source in blocks:
        parts.append(text[cursor : match.start()])
        cursor = match.end()
    parts.append(text[cursor:])
    return "\n".join(part.strip() for part in parts if part.strip()).strip()


def _strip_unsolicited_python_blocks(
    request: PromptRequest,
    candidate_answer: str,
) -> str:
    text, _explicit_domain = _strip_domain_prefix(request.text)
    if _looks_like_general_coding_request(text):
        return candidate_answer
    stripped = _remove_python_fenced_blocks(candidate_answer)
    return stripped or candidate_answer


def _candidate_answer_contract_error(
    request: PromptRequest,
    candidate_answer: str,
) -> str:
    text, _explicit_domain = _strip_domain_prefix(request.text)
    lowered = text.casefold()
    asks_for_example = "예시" in text or "example" in lowered
    if (
        not candidate_answer
        or not asks_for_example
        or "python" not in lowered
        or not _looks_like_general_coding_request(text)
    ):
        return ""

    snippets = tuple(source for _match, source in _python_fenced_blocks(candidate_answer))
    if not snippets:
        return (
            "the user requested a Python example; include one complete fenced Python "
            "example with its input and result variables initialized"
        )
    parsed = [ast.parse(snippet) for snippet in snippets]
    asks_for_comparison = any(
        token in lowered for token in ("차이", "비교", "vs", "versus")
    )
    mentions_comprehension = "컴프리헨션" in text or "comprehension" in lowered
    mentions_for_loop = "for 루프" in lowered or "for loop" in lowered
    if asks_for_comparison and mentions_comprehension and mentions_for_loop:
        nodes = tuple(node for tree in parsed for node in ast.walk(tree))
        if not any(isinstance(node, ast.For) for node in nodes) or not any(
            isinstance(node, ast.ListComp) for node in nodes
        ):
            return (
                "the requested comparison needs one complete Python example containing "
                "both an explicit for loop and a list comprehension"
            )
    return ""


_COLOR_ALIASES = {
    "빨간": "red",
    "빨강": "red",
    "red": "red",
    "초록": "green",
    "녹색": "green",
    "green": "green",
    "파란": "blue",
    "파랑": "blue",
    "blue": "blue",
}


def _compile_raster_vision(
    question: str,
    image_value: PromptImage,
) -> TypedDomainRequest | None:
    try:
        image = _load_small_raster(image_value)
    except (ImportError, OSError, TypeError, ValueError):
        return None
    colors = _mentioned_colors(question)
    lowered = question.casefold()
    goals: list[Any] = []
    count_selectors: tuple[str, ...] = ()
    if "몇 개" in question or "how many" in lowered:
        selector = colors[0] if colors else "all"
        try:
            analysis = RasterVisionAdapter().analyze(image)
        except ValueError:
            return None
        expected = (
            len(analysis.objects)
            if selector == "all"
            else sum(item.color_name == selector for item in analysis.objects)
        )
        goals.append(VisionCountGoal(selector, expected, question))
        count_selectors = (selector,)
    else:
        explicit_count = re.search(r"(?P<count>\d+)\s*개", question)
        if explicit_count:
            selector = colors[0] if colors else "all"
            goals.append(
                VisionCountGoal(selector, int(explicit_count.group("count")), question)
            )
        elif len(colors) >= 2 and ("왼쪽" in question or "left" in lowered):
            goals.append(VisionRelationGoal("LEFT_OF", colors[0], colors[1], question))
        elif len(colors) >= 2 and ("위" in question or "above" in lowered):
            goals.append(VisionRelationGoal("ABOVE", colors[0], colors[1], question))
        elif len(colors) >= 2 and any(
            token in question for token in ("더 크", "더 넓", "면적")
        ):
            goals.append(VisionAreaGoal(colors[0], colors[1], question))
        elif colors and ("정사각형" in question or "square" in lowered):
            goals.append(VisionPropertyGoal("SQUARE", colors[0], question))
        elif colors and ("직사각형" in question or "rectangle" in lowered):
            goals.append(VisionPropertyGoal("FILLED_RECTANGLE", colors[0], question))
    if not goals:
        return None
    return TypedDomainRequest(
        "vision",
        RasterVisionProblem(
            image=image,
            goals=tuple(goals),
            query=question,
            count_selectors=count_selectors,
        ),
        mode="shadow",
    )


def _mentioned_colors(text: str) -> tuple[str, ...]:
    found: list[tuple[int, str]] = []
    lowered = text.casefold()
    for alias, canonical in _COLOR_ALIASES.items():
        index = lowered.find(alias)
        if index >= 0:
            found.append((index, canonical))
    return tuple(dict.fromkeys(color for _, color in sorted(found)))


def _load_small_raster(value: PromptImage) -> RasterImage:
    if not isinstance(value, bytes):
        path = Path(value)
        if path.suffix.casefold() in {".pbm", ".pgm", ".ppm", ".pnm"}:
            image = RasterImage.from_pnm(path)
            _validate_small_raster(image)
            return image
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("PNG/JPEG raster input requires Pillow") from exc
    source = BytesIO(value) if isinstance(value, bytes) else Path(value)
    with Image.open(source) as opened:
        converted = opened.convert("RGB")
        if converted.width * converted.height > 16_384:
            raise ValueError("deterministic raster verification is limited to 16,384 pixels")
        rows = tuple(
            tuple(converted.getpixel((x, y)) for x in range(converted.width))
            for y in range(converted.height)
        )
    raster = RasterImage.from_rows(
        rows,
        source="memory" if isinstance(value, bytes) else str(Path(value)),
    )
    _validate_small_raster(raster)
    return raster


def _validate_small_raster(image: RasterImage) -> None:
    height = len(image.rows)
    width = len(image.rows[0])
    if width > 128 or height > 128:
        raise ValueError("deterministic raster verification supports up to 128x128")
    unique = {pixel for row in image.rows for pixel in row}
    if len(unique) > 64:
        raise ValueError("natural images require the semantic vision model")


def _sidecar_image_regions(candidates: Sequence[Any]) -> tuple[ImageRegion, ...]:
    regions: list[ImageRegion] = []
    for candidate in candidates:
        width = int(getattr(candidate, "image_width", 0))
        height = int(getattr(candidate, "image_height", 0))
        payload = getattr(candidate, "payload", {})
        if width <= 0 or height <= 0 or not isinstance(payload, Mapping):
            continue
        for result in payload.values():
            if not isinstance(result, Mapping):
                continue
            boxes = result.get("bboxes") or result.get("quad_boxes") or ()
            labels = result.get("labels") or ()
            if not isinstance(boxes, Sequence) or isinstance(boxes, (str, bytes)):
                continue
            for index, box in enumerate(boxes):
                if not isinstance(box, Sequence) or isinstance(box, (str, bytes)):
                    continue
                try:
                    coordinates = tuple(float(value) for value in box)
                except (TypeError, ValueError):
                    continue
                if len(coordinates) == 4:
                    xs = coordinates[0::2]
                    ys = coordinates[1::2]
                elif len(coordinates) == 8:
                    xs = coordinates[0::2]
                    ys = coordinates[1::2]
                else:
                    continue
                label = (
                    str(labels[index])
                    if isinstance(labels, Sequence)
                    and not isinstance(labels, (str, bytes))
                    and index < len(labels)
                    else ""
                )
                regions.append(
                    ImageRegion(
                        image_index=0,
                        x1=max(0.0, min(1.0, min(xs) / width)),
                        y1=max(0.0, min(1.0, min(ys) / height)),
                        x2=max(0.0, min(1.0, max(xs) / width)),
                        y2=max(0.0, min(1.0, max(ys) / height)),
                        label=label,
                    )
                )
    return tuple(dict.fromkeys(regions))


def _decode_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError("semantic model output must be text or an object")
    stripped = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.I)
    start = stripped.find("{")
    if start < 0:
        raise ValueError("semantic model output contains no JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(stripped[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid semantic JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise TypeError("semantic model JSON must be an object")
    return value


def _recover_json_string_field(text: str, name: str) -> str:
    """Recover one valid leading JSON string without accepting the whole payload."""

    match = re.search(rf'"{re.escape(name)}"\s*:\s*', text)
    if match is None:
        return ""
    try:
        value, _end = json.JSONDecoder().raw_decode(text[match.end() :].lstrip())
    except json.JSONDecodeError:
        return ""
    return value if isinstance(value, str) else ""


def _recover_json_string_list_field(text: str, name: str) -> tuple[str, ...]:
    """Recover one valid leading JSON string list from an otherwise invalid object."""

    match = re.search(rf'"{re.escape(name)}"\s*:\s*', text)
    if match is None:
        return ()
    try:
        value, _end = json.JSONDecoder().raw_decode(text[match.end() :].lstrip())
    except json.JSONDecodeError:
        return ()
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 120
        or not all(isinstance(item, str) for item in value)
    ):
        return ()
    return tuple(value)


def _recover_json_number_field(text: str, name: str) -> float:
    match = re.search(
        rf'"{re.escape(name)}"\s*:\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))',
        text,
    )
    if match is None:
        return 0.0
    try:
        return float(match.group("value"))
    except ValueError:
        return 0.0


def _decode_candidate_answer(payload: Mapping[str, Any]) -> str:
    answer_value = payload.get("answer", "")
    if not isinstance(answer_value, str):
        raise TypeError("semantic answer must be a string")
    answer = answer_value.strip()
    raw_lines = payload.get("code_lines")
    if raw_lines is None:
        return answer
    if not isinstance(raw_lines, list) or not 1 <= len(raw_lines) <= 120:
        raise TypeError("code_lines must be a non-empty list with at most 120 lines")
    lines: list[str] = []
    for value in raw_lines:
        if not isinstance(value, str):
            raise TypeError("every code_lines entry must be a string")
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        for line in normalized.split("\n"):
            if len(line) > 240:
                raise ValueError("each decoded source line must be bounded")
            lines.append(line.rstrip())
    if len(lines) > 120:
        raise ValueError("decoded code_lines cannot exceed 120 source lines")
    if not any(line.strip() for line in lines):
        raise ValueError("code_lines cannot contain only blank lines")
    language = str(payload.get("code_language", "text")).strip().casefold()
    if language == "py":
        language = "python"
    if re.fullmatch(r"[a-z0-9+#.-]{1,20}", language) is None:
        raise ValueError("code_language must be a short code-fence label")
    source = "\n".join(lines)
    code_block = f"```{language}\n{source}\n```"
    return f"{answer}\n\n{code_block}" if answer else code_block


def _augment_general_python_answer_from_raw(
    request: PromptRequest,
    raw: str,
    candidate_answer: str,
) -> str:
    inner: dict[str, str] = {}
    for key in ("code", "controlled_text", "statement"):
        recovered = _recover_json_string_field(raw, key)
        if recovered:
            inner[key] = recovered
    payload: dict[str, Any] = {"payload": inner}
    code_lines = _recover_json_string_list_field(raw, "code_lines")
    if code_lines:
        payload["code_lines"] = list(code_lines)
    return _augment_general_python_answer(request, payload, candidate_answer)


def _augment_general_python_answer(
    request: PromptRequest,
    payload: Mapping[str, Any],
    candidate_answer: str,
) -> str:
    text, _explicit_domain = _strip_domain_prefix(request.text)
    if "python" not in text.casefold() or not _looks_like_general_coding_request(text):
        return candidate_answer

    candidates: list[str] = []
    raw_lines = payload.get("code_lines")
    if isinstance(raw_lines, list) and raw_lines:
        flattened = "\n".join(
            str(value).replace("\r\n", "\n").replace("\r", "\n")
            for value in raw_lines
            if isinstance(value, str)
        )
        if flattened.strip():
            candidates.append(flattened)
    inner = payload.get("payload")
    if isinstance(inner, Mapping):
        for key in ("code", "controlled_text", "statement"):
            value = inner.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value)

    best_source = ""
    best_score = (-1, -1)
    for value in candidates:
        source = _longest_parseable_python_prefix(value)
        if not source:
            continue
        tree = ast.parse(source)
        nodes = tuple(ast.walk(tree))
        structure = int(any(isinstance(node, ast.For) for node in nodes)) + int(
            any(isinstance(node, ast.ListComp) for node in nodes)
        )
        score = (structure, len(source))
        if score > best_score:
            best_source = source
            best_score = score
    if not best_source:
        return candidate_answer
    completed_source = _complete_list_comprehension_comparison(request, best_source)
    if completed_source != best_source:
        best_source = completed_source
        completed_tree = ast.parse(best_source)
        completed_nodes = tuple(ast.walk(completed_tree))
        best_score = (
            int(any(isinstance(node, ast.For) for node in completed_nodes))
            + int(any(isinstance(node, ast.ListComp) for node in completed_nodes)),
            len(best_source),
        )

    existing_blocks = tuple(
        source for _match, source in _python_fenced_blocks(candidate_answer)
    )
    existing_score = 0
    for source in existing_blocks:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        nodes = tuple(ast.walk(tree))
        existing_score = max(
            existing_score,
            int(any(isinstance(node, ast.For) for node in nodes))
            + int(any(isinstance(node, ast.ListComp) for node in nodes)),
        )
    if existing_score >= best_score[0]:
        return candidate_answer
    prose = _remove_python_fenced_blocks(candidate_answer)
    block = f"```python\n{best_source}\n```"
    return f"{prose}\n\n{block}" if prose else block


def _complete_list_comprehension_comparison(
    request: PromptRequest,
    source: str,
) -> str:
    """Expand one proposed list comprehension into a runnable side-by-side example."""

    text, _explicit_domain = _strip_domain_prefix(request.text)
    lowered = text.casefold()
    if not (
        any(token in lowered for token in ("차이", "비교", "vs", "versus"))
        and ("컴프리헨션" in text or "comprehension" in lowered)
        and ("for 루프" in lowered or "for loop" in lowered)
    ):
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    nodes = tuple(ast.walk(tree))
    if any(isinstance(node, ast.For) for node in nodes):
        return source

    selected: ast.Assign | None = None
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.ListComp)
            and len(statement.value.generators) == 1
            and not statement.value.generators[0].is_async
        ):
            selected = statement
            break
    if selected is None:
        return source

    comprehension = selected.value
    generator = comprehension.generators[0]
    used_names = {node.id for node in nodes if isinstance(node, ast.Name)}

    def fresh_name(preferred: str) -> str:
        if preferred not in used_names:
            used_names.add(preferred)
            return preferred
        index = 2
        while f"{preferred}_{index}" in used_names:
            index += 1
        value = f"{preferred}_{index}"
        used_names.add(value)
        return value

    prefix = [statement for statement in tree.body if statement is not selected]
    assigned_names = {
        node.id
        for statement in prefix
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    lines: list[str] = []
    if prefix:
        lines.extend(ast.unparse(ast.Module(body=prefix, type_ignores=[])).splitlines())
    if isinstance(generator.iter, ast.Name) and generator.iter.id not in assigned_names:
        lines.append(f"{generator.iter.id} = [1, 2, 3, 4, 5]")

    loop_result = fresh_name("loop_result")
    comprehension_result = fresh_name("comprehension_result")
    target = ast.unparse(generator.target)
    iterable = ast.unparse(generator.iter)
    element = ast.unparse(comprehension.elt)
    lines.extend((f"{loop_result} = []", f"for {target} in {iterable}:"))
    indent = "    "
    if generator.ifs:
        condition = " and ".join(f"({ast.unparse(item)})" for item in generator.ifs)
        lines.append(f"{indent}if {condition}:")
        indent += "    "
    lines.append(f"{indent}{loop_result}.append({element})")
    lines.append(f"{comprehension_result} = {ast.unparse(comprehension)}")
    completed = "\n".join(lines)
    try:
        completed_tree = ast.parse(completed)
    except SyntaxError:
        return source
    completed_nodes = tuple(ast.walk(completed_tree))
    if not any(isinstance(node, ast.For) for node in completed_nodes) or not any(
        isinstance(node, ast.ListComp) for node in completed_nodes
    ):
        return source
    return completed


def _longest_parseable_python_prefix(value: str) -> str:
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    for end in range(len(lines), 0, -1):
        source = "\n".join(lines[:end]).strip()
        if not source:
            continue
        try:
            ast.parse(source)
        except SyntaxError:
            continue
        return source
    return ""


def _decode_operator_program(value: Any) -> tuple[CognitiveOperator, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError("operator_program must be a non-empty list")
    if len(value) > 32:
        raise ValueError("operator_program cannot exceed 32 steps")
    return tuple(CognitiveOperator(str(item).strip().upper()) for item in value)


def _decode_source_spans(value: Any, text: str) -> tuple[SourceSpan, ...]:
    if value is None:
        return (SourceSpan(0, len(text), text),) if text else ()
    if not isinstance(value, list) or len(value) > 16:
        raise TypeError("source_spans must be a list with at most 16 entries")
    spans: list[SourceSpan] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("source span must be an object")
        supplied = str(item.get("text", ""))
        requested_start = _safe_int(item.get("start"), default=0)
        if supplied:
            matches = _substring_offsets(text, supplied)
            if not matches:
                continue
            start = min(matches, key=lambda value: (abs(value - requested_start), value))
            spans.append(SourceSpan(start, start + len(supplied), supplied))
            continue
        start = requested_start
        end = _safe_int(item.get("end"), default=start)
        if 0 <= start <= end <= len(text):
            spans.append(SourceSpan(start, end, text[start:end]))
    if spans:
        return tuple(dict.fromkeys(spans))
    return (SourceSpan(0, len(text), text),) if text else ()


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _substring_offsets(text: str, excerpt: str) -> tuple[int, ...]:
    offsets: list[int] = []
    start = 0
    while True:
        found = text.find(excerpt, start)
        if found < 0:
            return tuple(offsets)
        offsets.append(found)
        start = found + 1


def _decode_image_regions(value: Any, image_count: int) -> tuple[ImageRegion, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > 64:
        raise TypeError("image_regions must be a list with at most 64 entries")
    regions: list[ImageRegion] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("image region must be an object")
        region = ImageRegion(
            image_index=int(item.get("image_index", 0)),
            x1=float(item["x1"]),
            y1=float(item["y1"]),
            x2=float(item["x2"]),
            y2=float(item["y2"]),
            label=str(item.get("label", "")),
        )
        if region.image_index >= image_count:
            raise ValueError("image region refers to a missing image")
        regions.append(region)
    return tuple(regions)


def _typed_request_from_model_payload(
    domain: str,
    value: Any,
    prompt: PromptRequest,
) -> TypedDomainRequest | None:
    if domain == "unsupported":
        return None
    if not isinstance(value, Mapping):
        raise TypeError("semantic payload must be an object")
    if domain == "math":
        expression = _normalize_math_symbols(
            str(value.get("expression", "")).strip()
        )
        if not expression or len(expression) > 240:
            raise ValueError("math proposal needs a bounded expression")
        expression = _normalize_verified_constant_equality(expression)
        MathInputAdapter().adapt(expression)
        return TypedDomainRequest("math", expression, mode="shadow")
    if domain == "language":
        controlled_value = value.get("controlled_text", "")
        if not isinstance(controlled_value, str):
            raise TypeError("language controlled_text must be a string")
        controlled = controlled_value.strip()
        if not controlled or len(controlled) > 4_000:
            raise ValueError("language proposal needs controlled_text")
        problem = LanguageTextProblem(
            controlled,
            source_context="untrusted_semantic_model",
            use_legacy_heuristics=False,
        )
        parsed = LanguageTextParser().parse(problem)
        if not any(claim.relation == "GOAL" for claim in parsed.claims):
            raise ValueError("language controlled_text needs an explicit Goal")
        if not any(claim.relation == "REQUIRES" for claim in parsed.claims):
            raise ValueError("language controlled_text needs at least one Requires")
        return TypedDomainRequest("language", problem, mode="shadow")
    if domain == "coding":
        statement = str(value.get("statement", prompt.text)).strip()
        return TypedDomainRequest("coding", CodingProblem(statement), mode="shadow")
    if domain == "vision":
        question = str(value.get("question", prompt.text)).strip()
        if not prompt.images:
            raise ValueError("vision proposal requires an image")
        return _compile_raster_vision(question, prompt.images[0])
    if domain == "composed":
        if not prompt.images:
            raise ValueError("composed proposal requires an image")
        controlled_value = value.get("controlled_text", "")
        if not isinstance(controlled_value, str):
            raise TypeError("composed controlled_text must be a string")
        controlled = controlled_value.strip()
        if not controlled or len(controlled) > 4_000:
            raise ValueError("composed proposal needs bounded controlled_text")
        image = _load_small_raster(prompt.images[0])
        SceneThresholdParser().parse(controlled)
        return TypedDomainRequest(
            "composed",
            SceneThresholdProblem(image, controlled),
            mode="shadow",
        )
    raise ValueError(f"unsupported model proposal domain: {domain!r}")


def _normalize_verified_constant_equality(expression: str) -> str:
    """Accept a model's checked arithmetic assertion as the calculation it shows."""

    equality = re.fullmatch(r"(?P<left>[^<>=!]+)=(?P<right>[^<>=!]+)", expression)
    if equality is None:
        return expression
    left = equality.group("left").strip()
    right = equality.group("right").strip()
    if re.search(r"[A-Za-z]", left + right):
        return expression
    adapter = MathInputAdapter()
    left_instance = adapter.adapt(left)
    right_instance = adapter.adapt(right)
    if left_instance.metadata.get("answer") != right_instance.metadata.get("answer"):
        raise ValueError("constant equality does not replay to the same exact value")
    return left


def _normalize_math_symbols(text: str) -> str:
    return text.translate(
        str.maketrans(
            {
                "²": "^2",
                "×": "*",
                "÷": "/",
                "−": "-",
            }
        )
    )

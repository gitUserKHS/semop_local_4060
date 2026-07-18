from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import statistics
from time import process_time
from typing import Protocol, runtime_checkable
from uuid import uuid4

from .contracts import DomainInstance
from .engine import ActionPolicy, OperatorKernel
from .library import MacroOperatorCandidate, MdlMacroLibrary
from .model import SolveBudget, SolveResult
from .synthesis import SyntheticProblem
from .traces import (
    DecisionTrainingCase,
    TraceCorpus,
    build_decision_training_cases,
    hard_negative_records,
)


class LearningSplit(str, Enum):
    TRAIN = "train"
    HELDOUT = "heldout"


class SemanticLabelAuthority(str, Enum):
    """Authority behind an expected outcome used for semantic evaluation."""

    PROGRAMMATIC = "programmatic"
    HUMAN_REVIEWED = "human_reviewed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LearningTask:
    """One labeled problem at the verifier boundary.

    ``expected_solved`` is an evaluation label, never a fact supplied to the
    operator kernel. Negative controls therefore remain impossible to solve
    unless registered operators can genuinely derive their goals.
    """

    task_id: str
    instance: DomainInstance
    expected_solved: bool = True
    split: LearningSplit = LearningSplit.TRAIN
    source: str = "verifier"
    domain: str = ""
    capability: str = ""
    structure_key: str = ""
    difficulty: int = 1
    label_authority: SemanticLabelAuthority = SemanticLabelAuthority.UNKNOWN

    def __post_init__(self) -> None:
        task_id = self.task_id.strip()
        if not task_id:
            raise ValueError("learning task id must not be empty")
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "split", LearningSplit(self.split))
        domain = self.domain.strip() or self.instance.domain.strip()
        if not domain:
            raise ValueError("learning task domain must not be empty")
        object.__setattr__(self, "domain", domain)
        if self.difficulty <= 0:
            raise ValueError("learning task difficulty must be positive")
        if self.source not in TraceCorpus.VALID_SOURCES:
            raise ValueError(f"unknown learning task source: {self.source}")
        authority = SemanticLabelAuthority(self.label_authority)
        if authority is SemanticLabelAuthority.UNKNOWN:
            authority = (
                SemanticLabelAuthority.HUMAN_REVIEWED
                if self.source == "reviewed"
                else SemanticLabelAuthority.PROGRAMMATIC
            )
        object.__setattr__(self, "label_authority", authority)
        if not self.instance.goals:
            raise ValueError("learning tasks require at least one verifier goal")

    @classmethod
    def from_synthetic(
        cls,
        problem: SyntheticProblem,
        *,
        split: LearningSplit,
        namespace: str = "",
        expected_solved: bool = True,
    ) -> "LearningTask":
        prefix = f"{namespace.strip()}:" if namespace.strip() else ""
        return cls(
            task_id=f"{prefix}{problem.problem_id}",
            instance=problem.instance,
            expected_solved=expected_solved,
            split=split,
            source="synthetic",
            domain=problem.domain,
            capability=problem.capability,
            structure_key=problem.structure_key,
            difficulty=problem.difficulty,
            label_authority=SemanticLabelAuthority.PROGRAMMATIC,
        )


def learning_tasks_from_synthetic(
    problems: Sequence[SyntheticProblem],
    *,
    split: LearningSplit,
    namespace: str = "",
    expected_solved: bool = True,
) -> tuple[LearningTask, ...]:
    return tuple(
        LearningTask.from_synthetic(
            problem,
            split=split,
            namespace=namespace,
            expected_solved=expected_solved,
        )
        for problem in problems
    )


@dataclass(frozen=True)
class SelfLearningBudget:
    """Resource and promotion limits for a bounded learning run."""

    iterations: int = 1
    solve_budget: SolveBudget = field(default_factory=SolveBudget)
    max_training_tasks: int = 5_000
    max_heldout_tasks: int = 2_000
    min_training_traces: int = 1
    required_proof_soundness: float = 1.0
    max_false_positives: int = 0
    max_solve_rate_drop: float = 0.01
    min_expansion_reduction: float = 0.01
    max_policy_parameters: int = 15_000_000
    max_artifact_bytes: int = 64 * 1024 * 1024
    max_p95_cpu_seconds: float = 10.0

    def __post_init__(self) -> None:
        positive = (
            self.iterations,
            self.max_training_tasks,
            self.max_heldout_tasks,
            self.min_training_traces,
            self.max_policy_parameters,
            self.max_artifact_bytes,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("self-learning integer limits must be positive")
        if not 0.0 <= self.required_proof_soundness <= 1.0:
            raise ValueError("required proof soundness must be between 0 and 1")
        if self.max_false_positives < 0:
            raise ValueError("max false positives must not be negative")
        if not 0.0 <= self.max_solve_rate_drop <= 1.0:
            raise ValueError("max solve-rate drop must be between 0 and 1")
        if not 0.0 <= self.min_expansion_reduction <= 1.0:
            raise ValueError("minimum expansion reduction must be between 0 and 1")
        if self.max_p95_cpu_seconds <= 0:
            raise ValueError("p95 CPU limit must be positive")

    @property
    def required_replay_integrity(self) -> float:
        """Canonical name for the legacy constructor field."""

        return self.required_proof_soundness


@dataclass(frozen=True)
class PolicyCandidate:
    """A policy plus its portable, hashable inference artifact."""

    policy: ActionPolicy = field(compare=False, repr=False)
    kind: str
    artifact_suffix: str
    artifact: bytes = field(compare=False, repr=False)
    parameter_count: int
    training_updates: int = 0
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("policy candidate kind must not be empty")
        if not self.artifact_suffix.startswith("."):
            raise ValueError("policy artifact suffix must begin with '.'")
        if any(
            character not in ".-_abcdefghijklmnopqrstuvwxyz0123456789"
            for character in self.artifact_suffix.lower()
        ):
            raise ValueError("policy artifact suffix contains unsafe characters")
        if self.parameter_count < 0 or self.training_updates < 0:
            raise ValueError("policy counts must not be negative")

    @property
    def artifact_bytes(self) -> int:
        return len(self.artifact)

    @property
    def artifact_sha256(self) -> str:
        return sha256(self.artifact).hexdigest()


@runtime_checkable
class PolicyLearner(Protocol):
    """Pluggable learner; implementations cannot mutate verifier state."""

    name: str

    def train(
        self,
        cases: Sequence[DecisionTrainingCase],
        incumbent: ActionPolicy | None = None,
    ) -> PolicyCandidate: ...

    def restore(self, artifact: bytes) -> ActionPolicy: ...


@dataclass(frozen=True)
class TaskEvaluation:
    task_id: str
    domain: str
    expected_solved: bool
    success: bool
    verified: bool
    false_positive: bool
    expansions: int
    proof_steps: int
    inference_rounds: int
    cpu_seconds: float
    halt_reason: str
    policy_used: bool
    fallback_used: bool
    label_authority: SemanticLabelAuthority = SemanticLabelAuthority.PROGRAMMATIC

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "label_authority",
            SemanticLabelAuthority(self.label_authority),
        )

    @property
    def has_semantic_gold(self) -> bool:
        return self.label_authority is SemanticLabelAuthority.HUMAN_REVIEWED


@dataclass(frozen=True)
class DomainLearningMetrics:
    domain: str
    tasks: int
    positive_tasks: int
    replay_verified_goal_completion: float
    primitive_replay_integrity: float
    semantic_correctness: float | None
    semantic_gold_tasks: int
    false_positives: int
    positive_expansions: int
    median_positive_expansions: float
    p95_cpu_seconds: float

    @property
    def verified_solve_rate(self) -> float:
        """Deprecated alias for replay-verified goal completion."""

        return self.replay_verified_goal_completion

    @property
    def proof_soundness(self) -> float:
        """Deprecated alias for primitive proof-replay integrity."""

        return self.primitive_replay_integrity

    def to_dict(self, *, legacy_aliases: bool = True) -> dict[str, object]:
        payload = asdict(self)
        if legacy_aliases:
            payload["verified_solve_rate"] = self.verified_solve_rate
            payload["proof_soundness"] = self.proof_soundness
        return payload


@dataclass(frozen=True)
class LearningMetrics:
    tasks: int
    positive_tasks: int
    replay_verified_goal_completion: float
    programmatic_outcome_accuracy: float
    primitive_replay_integrity: float
    semantic_correctness: float | None
    semantic_gold_tasks: int
    false_positives: int
    positive_expansions: int
    total_expansions: int
    median_positive_expansions: float
    p50_cpu_seconds: float
    p95_cpu_seconds: float
    fallback_count: int
    by_domain: tuple[DomainLearningMetrics, ...]

    def for_domain(self, domain: str) -> DomainLearningMetrics:
        for item in self.by_domain:
            if item.domain == domain:
                return item
        raise KeyError(domain)

    @property
    def verified_solve_rate(self) -> float:
        """Deprecated alias for replay-verified goal completion."""

        return self.replay_verified_goal_completion

    @property
    def expected_outcome_accuracy(self) -> float:
        """Deprecated alias for programmatic outcome accuracy."""

        return self.programmatic_outcome_accuracy

    @property
    def proof_soundness(self) -> float:
        """Deprecated alias for primitive proof-replay integrity."""

        return self.primitive_replay_integrity

    def to_dict(self, *, legacy_aliases: bool = True) -> dict[str, object]:
        payload = asdict(self)
        payload["by_domain"] = tuple(
            item.to_dict(legacy_aliases=legacy_aliases) for item in self.by_domain
        )
        if legacy_aliases:
            payload["verified_solve_rate"] = self.verified_solve_rate
            payload["expected_outcome_accuracy"] = self.expected_outcome_accuracy
            payload["proof_soundness"] = self.proof_soundness
        return payload


@dataclass(frozen=True)
class SelfLearningIteration:
    iteration: int
    generation_before: int
    generation_after: int
    candidate_kind: str | None
    accepted: bool
    rejection_reasons: tuple[str, ...]
    verified_training_traces: int
    decision_cases: int
    training_label_conflicts: tuple[str, ...]
    parameter_count: int
    artifact_bytes: int
    training_updates: int
    expansion_reduction: float
    baseline_metrics: LearningMetrics
    candidate_metrics: LearningMetrics | None
    baseline_tasks: tuple[TaskEvaluation, ...]
    candidate_tasks: tuple[TaskEvaluation, ...]
    retained_macro_candidates: int


@dataclass(frozen=True)
class SelfLearningCheckpoint:
    format_version: int
    generation: int
    iteration: int
    policy_kind: str | None
    policy_artifact: str | None
    policy_sha256: str | None
    policy_parameters: int
    trace_artifact: str
    trace_sha256: str
    trace_count: int
    macro_artifact: str
    macro_sha256: str
    macro_count: int
    manifest_path: Path = field(compare=False)


@dataclass(frozen=True)
class SelfLearningResult:
    final_policy: ActionPolicy | None = field(compare=False, repr=False)
    corpus: TraceCorpus = field(compare=False, repr=False)
    macro_library: MdlMacroLibrary = field(compare=False, repr=False)
    training_metrics: LearningMetrics
    iterations: tuple[SelfLearningIteration, ...]
    accepted_generations: int
    checkpoint: SelfLearningCheckpoint | None = None
    active_candidate: PolicyCandidate | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    @property
    def promoted(self) -> bool:
        return self.accepted_generations > 0


class SelfLearningStore:
    """Atomic audit/checkpoint storage for verifier-gated learning."""

    FORMAT_VERSION = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def persist(
        self,
        *,
        iteration: SelfLearningIteration,
        generation: int,
        corpus: TraceCorpus,
        macro_library: MdlMacroLibrary,
        active_candidate: PolicyCandidate | None,
    ) -> SelfLearningCheckpoint:
        self.root.mkdir(parents=True, exist_ok=True)
        iteration_relative = f"iterations/iteration-{iteration.iteration:04d}.json"
        _atomic_write_json(self.root / iteration_relative, asdict(iteration))

        trace_relative = "traces/verified-traces.jsonl"
        trace_payload = _trace_jsonl(corpus)
        _atomic_write_bytes(self.root / trace_relative, trace_payload)

        macro_relative = "macros/retained-candidates.json"
        macro_payload = json.dumps(
            {
                "format_version": MdlMacroLibrary.FORMAT_VERSION,
                "active": False,
                "activation_supported": True,
                "records": [asdict(record) for record in macro_library.records],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        _atomic_write_bytes(self.root / macro_relative, macro_payload)

        policy_relative: str | None = None
        policy_hash: str | None = None
        policy_kind: str | None = None
        policy_parameters = 0
        if active_candidate is not None:
            policy_relative = (
                f"generations/generation-{generation:04d}/policy"
                f"{active_candidate.artifact_suffix}"
            )
            _atomic_write_bytes(
                self.root / policy_relative,
                active_candidate.artifact,
            )
            policy_hash = active_candidate.artifact_sha256
            policy_kind = active_candidate.kind
            policy_parameters = active_candidate.parameter_count

        payload = {
            "format_version": self.FORMAT_VERSION,
            "generation": generation,
            "iteration": iteration.iteration,
            "policy_kind": policy_kind,
            "policy_artifact": policy_relative,
            "policy_sha256": policy_hash,
            "policy_parameters": policy_parameters,
            "trace_artifact": trace_relative,
            "trace_sha256": sha256(trace_payload).hexdigest(),
            "trace_count": len(corpus.records),
            "macro_artifact": macro_relative,
            "macro_sha256": sha256(macro_payload).hexdigest(),
            "macro_count": len(macro_library.records),
        }
        _atomic_write_json(self.manifest_path, payload)
        return self.load_checkpoint()

    def load_checkpoint(self) -> SelfLearningCheckpoint:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("unsupported self-learning checkpoint format")
        checkpoint = SelfLearningCheckpoint(
            **payload,
            manifest_path=self.manifest_path.resolve(),
        )
        self._verify_artifact(checkpoint.trace_artifact, checkpoint.trace_sha256)
        self._verify_artifact(checkpoint.macro_artifact, checkpoint.macro_sha256)
        if checkpoint.policy_artifact is not None:
            if checkpoint.policy_sha256 is None or checkpoint.policy_kind is None:
                raise ValueError("checkpoint policy metadata is incomplete")
            self._verify_artifact(
                checkpoint.policy_artifact,
                checkpoint.policy_sha256,
            )
        return checkpoint

    def load_corpus(self, checkpoint: SelfLearningCheckpoint | None = None) -> TraceCorpus:
        resolved = checkpoint or self.load_checkpoint()
        return TraceCorpus.load_jsonl(self._artifact_path(resolved.trace_artifact))

    def restore_candidate(
        self,
        learner: PolicyLearner,
        checkpoint: SelfLearningCheckpoint | None = None,
    ) -> PolicyCandidate | None:
        resolved = checkpoint or self.load_checkpoint()
        if resolved.policy_artifact is None:
            return None
        if resolved.policy_kind != learner.name:
            raise ValueError(
                f"checkpoint policy kind {resolved.policy_kind!r} cannot be "
                f"restored by {learner.name!r}"
            )
        artifact = self._artifact_path(resolved.policy_artifact).read_bytes()
        return PolicyCandidate(
            policy=learner.restore(artifact),
            kind=resolved.policy_kind,
            artifact_suffix=Path(resolved.policy_artifact).suffix,
            artifact=artifact,
            parameter_count=resolved.policy_parameters,
            diagnostics=("restored from verifier-gated checkpoint",),
        )

    def _verify_artifact(self, relative: str, expected_hash: str) -> None:
        path = self._artifact_path(relative)
        if not path.is_file():
            raise ValueError(f"checkpoint artifact is missing: {relative}")
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != expected_hash:
            raise ValueError(f"checkpoint artifact hash mismatch: {relative}")

    def _artifact_path(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("checkpoint artifact escapes its storage root")
        return candidate


class SelfLearningLoop:
    """Learn action ranking, then promote it only through held-out verification."""

    def __init__(
        self,
        learner: PolicyLearner | None = None,
        *,
        budget: SelfLearningBudget | None = None,
        corpus: TraceCorpus | None = None,
        macro_library: MdlMacroLibrary | None = None,
        store: SelfLearningStore | str | Path | None = None,
    ) -> None:
        if learner is None:
            from semop.tiny_controller.linear_policy import StructuralPolicyLearner

            learner = StructuralPolicyLearner()
        self.learner = learner
        self.budget = budget or SelfLearningBudget()
        self.corpus = corpus or TraceCorpus()
        self.macro_library = macro_library or MdlMacroLibrary()
        self.store = (
            store
            if isinstance(store, SelfLearningStore) or store is None
            else SelfLearningStore(store)
        )

    def run(
        self,
        training_tasks: Sequence[LearningTask],
        heldout_tasks: Sequence[LearningTask],
        *,
        incumbent_policy: ActionPolicy | None = None,
        resume: bool = False,
    ) -> SelfLearningResult:
        training = tuple(training_tasks)
        heldout = tuple(heldout_tasks)
        self._validate_tasks(training, heldout)

        generation = 0
        iteration_offset = 0
        active_candidate: PolicyCandidate | None = None
        if resume:
            if self.store is None or not self.store.exists():
                raise ValueError("resume requires an existing self-learning store")
            checkpoint = self.store.load_checkpoint()
            generation = checkpoint.generation
            iteration_offset = checkpoint.iteration
            self.corpus = self.store.load_corpus(checkpoint)
            active_candidate = self.store.restore_candidate(self.learner, checkpoint)
            incumbent_policy = (
                active_candidate.policy
                if active_candidate is not None
                else incumbent_policy
            )
        elif self.store is not None and self.store.exists():
            raise ValueError(
                "self-learning store already contains a checkpoint; "
                "pass resume=True or use a new store"
            )

        (
            decision_cases,
            verified_results,
            training_conflicts,
            training_metrics,
        ) = self._collect_verified_training_data(training)

        incumbent = incumbent_policy
        iterations: list[SelfLearningIteration] = []
        accepted_generations = 0
        checkpoint: SelfLearningCheckpoint | None = None
        for local_iteration in range(1, self.budget.iterations + 1):
            iteration_number = iteration_offset + local_iteration
            baseline_metrics, baseline_tasks, baseline_results = self._evaluate(
                heldout,
                incumbent,
            )
            candidate: PolicyCandidate | None = None
            candidate_metrics: LearningMetrics | None = None
            candidate_tasks: tuple[TaskEvaluation, ...] = ()
            candidate_results: dict[str, SolveResult] = {}
            reasons: list[str] = []

            if len(verified_results) < self.budget.min_training_traces:
                reasons.append(
                    "insufficient_verified_training_traces: "
                    f"{len(verified_results)} < {self.budget.min_training_traces}"
                )
            if training_conflicts:
                reasons.append("training_labels_conflict_with_typed_verifier")
            if not decision_cases:
                reasons.append("no_replay_verified_decision_cases")

            if not reasons:
                try:
                    candidate = self.learner.train(decision_cases, incumbent)
                    if candidate.kind != self.learner.name:
                        reasons.append("learner_returned_mismatched_policy_kind")
                    if (
                        active_candidate is not None
                        and candidate.kind == active_candidate.kind
                        and candidate.artifact_sha256
                        == active_candidate.artifact_sha256
                    ):
                        reasons.append("candidate_is_identical_to_incumbent")
                    candidate_metrics, candidate_tasks, candidate_results = self._evaluate(
                        heldout,
                        candidate.policy,
                    )
                    reasons.extend(
                        self._promotion_rejections(
                            baseline_metrics,
                            candidate_metrics,
                            candidate,
                        )
                    )
                except Exception as exc:
                    reasons.append(
                        "policy_training_or_evaluation_failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

            reduction = (
                learning_expansion_reduction(baseline_metrics, candidate_metrics)
                if candidate_metrics is not None
                else 0.0
            )
            accepted = candidate is not None and not reasons
            generation_before = generation
            if accepted:
                incumbent = candidate.policy
                active_candidate = candidate
                generation += 1
                accepted_generations += 1
                self.macro_library = self._induce_passive_macros(
                    verified_results,
                    heldout,
                    candidate_results,
                )

            iteration = SelfLearningIteration(
                iteration=iteration_number,
                generation_before=generation_before,
                generation_after=generation,
                candidate_kind=candidate.kind if candidate is not None else None,
                accepted=accepted,
                rejection_reasons=tuple(reasons),
                verified_training_traces=len(verified_results),
                decision_cases=len(decision_cases),
                training_label_conflicts=training_conflicts,
                parameter_count=candidate.parameter_count if candidate else 0,
                artifact_bytes=candidate.artifact_bytes if candidate else 0,
                training_updates=candidate.training_updates if candidate else 0,
                expansion_reduction=reduction,
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
                baseline_tasks=baseline_tasks,
                candidate_tasks=candidate_tasks,
                retained_macro_candidates=len(self.macro_library.records),
            )
            iterations.append(iteration)
            if self.store is not None:
                checkpoint = self.store.persist(
                    iteration=iteration,
                    generation=generation,
                    corpus=self.corpus,
                    macro_library=self.macro_library,
                    active_candidate=active_candidate,
                )

        return SelfLearningResult(
            final_policy=incumbent,
            corpus=self.corpus,
            macro_library=self.macro_library,
            training_metrics=training_metrics,
            iterations=tuple(iterations),
            accepted_generations=accepted_generations,
            checkpoint=checkpoint,
            active_candidate=active_candidate,
        )

    def _validate_tasks(
        self,
        training: tuple[LearningTask, ...],
        heldout: tuple[LearningTask, ...],
    ) -> None:
        if not training:
            raise ValueError("self-learning requires training tasks")
        if not heldout:
            raise ValueError("self-learning requires held-out tasks")
        if len(training) > self.budget.max_training_tasks:
            raise ValueError("training task limit exceeded")
        if len(heldout) > self.budget.max_heldout_tasks:
            raise ValueError("held-out task limit exceeded")
        if any(task.split is not LearningSplit.TRAIN for task in training):
            raise ValueError("all training tasks must use the train split")
        if any(task.split is not LearningSplit.HELDOUT for task in heldout):
            raise ValueError("all held-out tasks must use the heldout split")
        identifiers = [task.task_id for task in training + heldout]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("learning task ids must be unique across splits")
        if not any(task.expected_solved for task in heldout):
            raise ValueError("held-out tasks require at least one positive problem")

    def _collect_verified_training_data(
        self,
        tasks: tuple[LearningTask, ...],
    ) -> tuple[
        tuple[DecisionTrainingCase, ...],
        dict[str, SolveResult],
        tuple[str, ...],
        LearningMetrics,
    ]:
        cases: list[DecisionTrainingCase] = []
        verified_results: dict[str, SolveResult] = {}
        conflicts: list[str] = []
        evaluations: list[TaskEvaluation] = []
        existing_ids = {record.trace_id for record in self.corpus.records}
        for task in tasks:
            kernel = OperatorKernel(task.instance.registry)
            started = process_time()
            result = kernel.solve(
                task.instance.state,
                task.instance.goals,
                budget=self.budget.solve_budget,
            )
            evaluations.append(
                task_evaluation_from_result(task, result, process_time() - started)
            )
            if result.success and not task.expected_solved:
                conflicts.append(task.task_id)
                continue
            if not (task.expected_solved and result.success and result.verified):
                continue
            task_cases = build_decision_training_cases(kernel, result)
            if task.task_id not in existing_ids:
                try:
                    self.corpus.add_result(
                        task.task_id,
                        task.domain,
                        result,
                        source=task.source,
                        hard_negatives=hard_negative_records(task_cases),
                        metadata=_trace_metadata(task),
                    )
                except ValueError:
                    continue
                existing_ids.add(task.task_id)
            cases.extend(task_cases)
            verified_results[task.task_id] = result
        return (
            tuple(cases),
            verified_results,
            tuple(sorted(conflicts)),
            summarize_task_evaluations(tuple(evaluations)),
        )

    def _evaluate(
        self,
        tasks: tuple[LearningTask, ...],
        policy: ActionPolicy | None,
    ) -> tuple[
        LearningMetrics,
        tuple[TaskEvaluation, ...],
        dict[str, SolveResult],
    ]:
        evaluations: list[TaskEvaluation] = []
        results: dict[str, SolveResult] = {}
        for task in tasks:
            started = process_time()
            result = OperatorKernel(task.instance.registry).solve(
                task.instance.state,
                task.instance.goals,
                policy=policy,
                budget=self.budget.solve_budget,
            )
            results[task.task_id] = result
            evaluations.append(
                task_evaluation_from_result(task, result, process_time() - started)
            )
        resolved = tuple(evaluations)
        return summarize_task_evaluations(resolved), resolved, results

    def _promotion_rejections(
        self,
        baseline: LearningMetrics,
        candidate: LearningMetrics,
        policy: PolicyCandidate,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if (
            candidate.primitive_replay_integrity
            < self.budget.required_replay_integrity
        ):
            reasons.append(
                "primitive_replay_integrity_below_gate: "
                f"{candidate.primitive_replay_integrity:.6f} < "
                f"{self.budget.required_replay_integrity:.6f}"
            )
        if candidate.false_positives > self.budget.max_false_positives:
            reasons.append(
                "false_positive_gate_failed: "
                f"{candidate.false_positives} > {self.budget.max_false_positives}"
            )
        minimum_rate = (
            baseline.replay_verified_goal_completion
            - self.budget.max_solve_rate_drop
        )
        if candidate.replay_verified_goal_completion < minimum_rate:
            reasons.append(
                "replay_verified_goal_completion_regressed: "
                f"{candidate.replay_verified_goal_completion:.6f} < "
                f"{minimum_rate:.6f}"
            )
        baseline_domains = {item.domain: item for item in baseline.by_domain}
        candidate_domains = {item.domain: item for item in candidate.by_domain}
        for domain in sorted(baseline_domains):
            baseline_domain = baseline_domains[domain]
            candidate_domain = candidate_domains[domain]
            minimum_domain_rate = (
                baseline_domain.replay_verified_goal_completion
                - self.budget.max_solve_rate_drop
            )
            if (
                candidate_domain.replay_verified_goal_completion
                < minimum_domain_rate
            ):
                reasons.append(
                    f"domain_solve_rate_regressed[{domain}]: "
                    f"{candidate_domain.replay_verified_goal_completion:.6f} < "
                    f"{minimum_domain_rate:.6f}"
                )
        reduction = learning_expansion_reduction(baseline, candidate)
        if reduction < self.budget.min_expansion_reduction:
            reasons.append(
                "expansion_reduction_below_gate: "
                f"{reduction:.6f} < {self.budget.min_expansion_reduction:.6f}"
            )
        if candidate.p95_cpu_seconds > self.budget.max_p95_cpu_seconds:
            reasons.append(
                "p95_cpu_limit_exceeded: "
                f"{candidate.p95_cpu_seconds:.6f} > "
                f"{self.budget.max_p95_cpu_seconds:.6f}"
            )
        if policy.parameter_count > self.budget.max_policy_parameters:
            reasons.append(
                "policy_parameter_limit_exceeded: "
                f"{policy.parameter_count} > {self.budget.max_policy_parameters}"
            )
        if policy.artifact_bytes > self.budget.max_artifact_bytes:
            reasons.append(
                "policy_artifact_limit_exceeded: "
                f"{policy.artifact_bytes} > {self.budget.max_artifact_bytes}"
            )
        return tuple(reasons)

    def _induce_passive_macros(
        self,
        verified_results: dict[str, SolveResult],
        heldout_tasks: tuple[LearningTask, ...],
        heldout_results: dict[str, SolveResult],
    ) -> MdlMacroLibrary:
        task_by_id = {task.task_id: task for task in heldout_tasks}

        def validator(candidate: MacroOperatorCandidate) -> bool:
            matched = False
            for task_id, result in sorted(heldout_results.items()):
                if not result.success or not result.verified:
                    continue
                sequence = tuple(
                    step.action.operator.name for step in result.proof
                )
                if not _contains_subsequence(sequence, candidate.operator_sequence):
                    continue
                matched = True
                task = task_by_id[task_id]
                replay = OperatorKernel(task.instance.registry).replay(
                    task.instance.state,
                    task.instance.goals,
                    result.proof,
                )
                if not replay.verified:
                    return False
            return matched

        induced = MdlMacroLibrary.induce(
            verified_results,
            heldout_validator=validator,
        )
        by_signature = {
            (record.operator_sequence, record.typed_pattern): record
            for record in self.macro_library.records
        }
        by_signature.update(
            {
                (record.operator_sequence, record.typed_pattern): record
                for record in induced.records
            }
        )
        return MdlMacroLibrary(
            by_signature[key] for key in sorted(by_signature, key=str)
        )


def task_evaluation_from_result(
    task: LearningTask,
    result: SolveResult,
    cpu_seconds: float,
) -> TaskEvaluation:
    return TaskEvaluation(
        task_id=task.task_id,
        domain=task.domain,
        expected_solved=task.expected_solved,
        success=result.success,
        verified=result.verified,
        false_positive=result.success and not task.expected_solved,
        expansions=result.expansions,
        proof_steps=len(result.proof),
        inference_rounds=result.inference_rounds,
        cpu_seconds=cpu_seconds,
        halt_reason=result.halt_reason,
        policy_used=result.policy_used,
        fallback_used=result.fallback_used,
        label_authority=task.label_authority,
    )


def summarize_task_evaluations(
    evaluations: tuple[TaskEvaluation, ...],
) -> LearningMetrics:
    positive = tuple(item for item in evaluations if item.expected_solved)
    successes = tuple(item for item in evaluations if item.success)
    semantic_gold = tuple(item for item in evaluations if item.has_semantic_gold)
    by_domain = tuple(
        _summarize_domain(
            domain,
            tuple(item for item in evaluations if item.domain == domain),
        )
        for domain in sorted({item.domain for item in evaluations})
    )
    return LearningMetrics(
        tasks=len(evaluations),
        positive_tasks=len(positive),
        replay_verified_goal_completion=_ratio(
            sum(item.success and item.verified for item in positive),
            len(positive),
        ),
        programmatic_outcome_accuracy=_ratio(
            sum(item.success == item.expected_solved for item in evaluations),
            len(evaluations),
        ),
        primitive_replay_integrity=_ratio(
            sum(item.verified for item in successes),
            len(successes),
            empty=1.0,
        ),
        semantic_correctness=(
            _ratio(
                sum(
                    item.success == item.expected_solved for item in semantic_gold
                ),
                len(semantic_gold),
            )
            if semantic_gold
            else None
        ),
        semantic_gold_tasks=len(semantic_gold),
        false_positives=sum(item.false_positive for item in evaluations),
        positive_expansions=sum(item.expansions for item in positive),
        total_expansions=sum(item.expansions for item in evaluations),
        median_positive_expansions=_median(
            [item.expansions for item in positive]
        ),
        p50_cpu_seconds=_percentile(
            [item.cpu_seconds for item in evaluations], 0.50
        ),
        p95_cpu_seconds=_percentile(
            [item.cpu_seconds for item in evaluations], 0.95
        ),
        fallback_count=sum(item.fallback_used for item in evaluations),
        by_domain=by_domain,
    )


def _summarize_domain(
    domain: str,
    evaluations: tuple[TaskEvaluation, ...],
) -> DomainLearningMetrics:
    positive = tuple(item for item in evaluations if item.expected_solved)
    successes = tuple(item for item in evaluations if item.success)
    semantic_gold = tuple(item for item in evaluations if item.has_semantic_gold)
    return DomainLearningMetrics(
        domain=domain,
        tasks=len(evaluations),
        positive_tasks=len(positive),
        replay_verified_goal_completion=_ratio(
            sum(item.success and item.verified for item in positive),
            len(positive),
        ),
        primitive_replay_integrity=_ratio(
            sum(item.verified for item in successes),
            len(successes),
            empty=1.0,
        ),
        semantic_correctness=(
            _ratio(
                sum(
                    item.success == item.expected_solved for item in semantic_gold
                ),
                len(semantic_gold),
            )
            if semantic_gold
            else None
        ),
        semantic_gold_tasks=len(semantic_gold),
        false_positives=sum(item.false_positive for item in evaluations),
        positive_expansions=sum(item.expansions for item in positive),
        median_positive_expansions=_median(
            [item.expansions for item in positive]
        ),
        p95_cpu_seconds=_percentile(
            [item.cpu_seconds for item in evaluations], 0.95
        ),
    )


def learning_expansion_reduction(
    baseline: LearningMetrics,
    candidate: LearningMetrics | None,
) -> float:
    if candidate is None:
        return 0.0
    if baseline.positive_expansions == 0:
        return 0.0 if candidate.positive_expansions == 0 else -1.0
    return (
        baseline.positive_expansions - candidate.positive_expansions
    ) / baseline.positive_expansions


def _trace_metadata(task: LearningTask) -> dict[str, str]:
    metadata = {
        "split": task.split.value,
        "label_authority": task.label_authority.value,
        "difficulty": str(task.difficulty),
    }
    if task.capability:
        metadata["capability"] = task.capability
    if task.structure_key:
        metadata["structure_key"] = task.structure_key
    for name, value in sorted(task.instance.metadata.items()):
        if not name.startswith(("discovery_", "experience_")):
            continue
        metadata[name] = (
            value
            if isinstance(value, str)
            else json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )
    return metadata


def _contains_subsequence(
    sequence: tuple[str, ...],
    candidate: tuple[str, ...],
) -> bool:
    if not candidate or len(candidate) > len(sequence):
        return False
    return any(
        sequence[start : start + len(candidate)] == candidate
        for start in range(len(sequence) - len(candidate) + 1)
    )


def _ratio(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    return numerator / denominator if denominator else empty


def _median(values: Sequence[int | float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _trace_jsonl(corpus: TraceCorpus) -> bytes:
    lines = (
        json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
        for record in corpus.records
    )
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    return payload.encode("utf-8")


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8"),
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()

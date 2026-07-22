from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import process_time

from .engine import ActionPolicy, OperatorKernel
from .library import (
    MacroOperatorCandidate,
    MdlMacroLibrary,
    operator_schema_fingerprint,
)
from .macro_policy import PrimitiveMacroPolicy
from .model import SolveBudget, SolveResult
from .self_learning import (
    LearningMetrics,
    LearningSplit,
    LearningTask,
    TaskEvaluation,
    learning_expansion_reduction,
    summarize_task_evaluations,
    task_evaluation_from_result,
)


@dataclass(frozen=True)
class MacroLearningBudget:
    solve_budget: SolveBudget = field(default_factory=SolveBudget)
    min_support: int = 3
    min_reduction: float = 0.10
    max_macros: int = 64
    max_solve_rate_drop: float = 0.01
    min_expansion_reduction: float = 0.10
    required_proof_soundness: float = 1.0
    max_false_positives: int = 0
    min_domains_improved: int = 2

    def __post_init__(self) -> None:
        if self.min_support < 3:
            raise ValueError("macro learning requires support of at least three")
        if self.max_macros <= 0:
            raise ValueError("macro limit must be positive")
        if not 0.0 <= self.min_reduction <= 1.0:
            raise ValueError("macro MDL reduction must be between zero and one")
        if not 0.0 <= self.max_solve_rate_drop <= 1.0:
            raise ValueError("solve-rate drop must be between zero and one")
        if not 0.0 <= self.min_expansion_reduction <= 1.0:
            raise ValueError("expansion reduction must be between zero and one")
        if not 0.0 <= self.required_proof_soundness <= 1.0:
            raise ValueError("proof soundness must be between zero and one")
        if self.max_false_positives < 0:
            raise ValueError("false-positive limit must not be negative")
        if not 1 <= self.min_domains_improved <= 3:
            raise ValueError("improved-domain gate must be between one and three")

    @property
    def required_replay_integrity(self) -> float:
        return self.required_proof_soundness


@dataclass(frozen=True)
class MacroActivationAudit:
    task_id: str
    domain: str
    active_programs: int
    rejected_programs: int


@dataclass(frozen=True)
class MacroLearningResult:
    promoted: bool
    rejection_reasons: tuple[str, ...]
    candidate_library: MdlMacroLibrary = field(compare=False, repr=False)
    active_library: MdlMacroLibrary = field(compare=False, repr=False)
    verified_training_traces: int
    verified_validation_traces: int
    baseline_metrics: LearningMetrics
    guided_metrics: LearningMetrics
    expansion_reduction: float
    improved_domains: tuple[str, ...]
    activation_audit: tuple[MacroActivationAudit, ...]
    baseline_tasks: tuple[TaskEvaluation, ...]
    guided_tasks: tuple[TaskEvaluation, ...]
    baseline_results: dict[str, SolveResult] = field(compare=False, repr=False)
    guided_results: dict[str, SolveResult] = field(compare=False, repr=False)


class VerifiedMacroLearningLoop:
    """Induce, validate, and promote primitive-expanded procedural memory."""

    def __init__(self, budget: MacroLearningBudget | None = None) -> None:
        self.budget = budget or MacroLearningBudget()

    def run(
        self,
        training_tasks: Sequence[LearningTask],
        validation_tasks: Sequence[LearningTask],
        heldout_tasks: Sequence[LearningTask],
        *,
        base_policy: ActionPolicy | None = None,
    ) -> MacroLearningResult:
        training = tuple(training_tasks)
        validation = tuple(validation_tasks)
        heldout = tuple(heldout_tasks)
        self._validate_splits(training, validation, heldout)

        training_results, training_failures = self._verified_positive_results(
            training
        )
        validation_results, validation_failures = self._verified_positive_results(
            validation
        )
        task_by_validation_id = {task.task_id: task for task in validation}

        def validator(candidate: MacroOperatorCandidate) -> bool:
            matched = False
            for task_id, result in sorted(validation_results.items()):
                task = task_by_validation_id[task_id]
                if not self._candidate_matches_result(candidate, result):
                    continue
                replay = OperatorKernel(task.instance.registry).replay(
                    task.instance.state,
                    task.instance.goals,
                    result.proof,
                )
                if not replay.verified:
                    return False
                matched = True
            return matched

        candidate_library = MdlMacroLibrary.induce(
            training_results,
            heldout_validator=validator,
            min_support=self.budget.min_support,
            min_reduction=self.budget.min_reduction,
        )
        if len(candidate_library.records) > self.budget.max_macros:
            retained = sorted(
                candidate_library.records,
                key=lambda item: (
                    -item.reduction_ratio,
                    -item.support,
                    item.name,
                ),
            )[: self.budget.max_macros]
            candidate_library = MdlMacroLibrary(retained)

        baseline_metrics, baseline_tasks, baseline_results, _baseline_audit = (
            self._evaluate(heldout, None, base_policy)
        )
        guided_metrics, guided_tasks, guided_results, activation_audit = (
            self._evaluate(heldout, candidate_library, base_policy)
        )
        expansion_reduction = learning_expansion_reduction(
            baseline_metrics,
            guided_metrics,
        )
        improved_domains = tuple(
            domain
            for domain in sorted(
                {item.domain for item in baseline_metrics.by_domain}
            )
            if guided_metrics.for_domain(domain).positive_expansions
            < baseline_metrics.for_domain(domain).positive_expansions
        )

        reasons = list(training_failures + validation_failures)
        if not candidate_library.records:
            reasons.append("no_validation_retained_macro_candidates")
        minimum_rate = (
            baseline_metrics.replay_verified_goal_completion
            - self.budget.max_solve_rate_drop
        )
        if guided_metrics.replay_verified_goal_completion < minimum_rate:
            reasons.append(
                "macro_replay_verified_goal_completion_regressed: "
                f"{guided_metrics.replay_verified_goal_completion:.6f} < "
                f"{minimum_rate:.6f}"
            )
        if (
            guided_metrics.primitive_replay_integrity
            < self.budget.required_replay_integrity
        ):
            reasons.append(
                "macro_primitive_replay_integrity_below_gate: "
                f"{guided_metrics.primitive_replay_integrity:.6f} < "
                f"{self.budget.required_replay_integrity:.6f}"
            )
        if guided_metrics.false_positives > self.budget.max_false_positives:
            reasons.append(
                "macro_guided_false_positive_limit_exceeded: "
                f"{guided_metrics.false_positives} > "
                f"{self.budget.max_false_positives}"
            )
        if expansion_reduction < self.budget.min_expansion_reduction:
            reasons.append(
                "macro_expansion_reduction_below_gate: "
                f"{expansion_reduction:.6f} < "
                f"{self.budget.min_expansion_reduction:.6f}"
            )
        if len(improved_domains) < self.budget.min_domains_improved:
            reasons.append(
                "macro_improved_domain_count_below_gate: "
                f"{len(improved_domains)} < {self.budget.min_domains_improved}"
            )
        active_positive_domains = {
            item.domain
            for item, task in zip(activation_audit, heldout, strict=True)
            if task.expected_solved and item.active_programs > 0
        }
        if len(active_positive_domains) < self.budget.min_domains_improved:
            reasons.append(
                "macro_active_domain_count_below_gate: "
                f"{len(active_positive_domains)} < "
                f"{self.budget.min_domains_improved}"
            )

        resolved_reasons = tuple(dict.fromkeys(reasons))
        promoted = not resolved_reasons
        return MacroLearningResult(
            promoted=promoted,
            rejection_reasons=resolved_reasons,
            candidate_library=candidate_library,
            active_library=(
                candidate_library if promoted else MdlMacroLibrary()
            ),
            verified_training_traces=len(training_results),
            verified_validation_traces=len(validation_results),
            baseline_metrics=baseline_metrics,
            guided_metrics=guided_metrics,
            expansion_reduction=expansion_reduction,
            improved_domains=improved_domains,
            activation_audit=activation_audit,
            baseline_tasks=baseline_tasks,
            guided_tasks=guided_tasks,
            baseline_results=baseline_results,
            guided_results=guided_results,
        )

    @staticmethod
    def persist_promoted(
        result: MacroLearningResult,
        path: str | Path,
    ) -> Path | None:
        """Atomically replace the active artifact only after promotion."""

        if not result.promoted:
            return None
        return result.active_library.save(path)

    def _verified_positive_results(
        self,
        tasks: tuple[LearningTask, ...],
    ) -> tuple[dict[str, SolveResult], tuple[str, ...]]:
        results: dict[str, SolveResult] = {}
        failures: list[str] = []
        for task in tasks:
            if not task.expected_solved:
                continue
            result = OperatorKernel(task.instance.registry).solve(
                task.instance.state,
                task.instance.goals,
                budget=self.budget.solve_budget,
            )
            if result.success and result.verified:
                results[task.task_id] = result
            else:
                failures.append(f"unverified_macro_seed:{task.task_id}")
        return results, tuple(failures)

    def _evaluate(
        self,
        tasks: tuple[LearningTask, ...],
        library: MdlMacroLibrary | None,
        base_policy: ActionPolicy | None,
    ) -> tuple[
        LearningMetrics,
        tuple[TaskEvaluation, ...],
        dict[str, SolveResult],
        tuple[MacroActivationAudit, ...],
    ]:
        evaluations: list[TaskEvaluation] = []
        results: dict[str, SolveResult] = {}
        audits: list[MacroActivationAudit] = []
        for task in tasks:
            policy = base_policy
            active_count = 0
            rejected_count = 0
            if library is not None:
                macro_policy = PrimitiveMacroPolicy(
                    task.instance.registry,
                    library,
                    base_policy=base_policy,
                )
                policy = macro_policy
                active_count = macro_policy.active_program_count
                rejected_count = macro_policy.rejected_program_count
            started = process_time()
            result = OperatorKernel(task.instance.registry).solve(
                task.instance.state,
                task.instance.goals,
                policy=policy,
                budget=self.budget.solve_budget,
            )
            elapsed = process_time() - started
            results[task.task_id] = result
            evaluations.append(
                task_evaluation_from_result(task, result, elapsed)
            )
            audits.append(
                MacroActivationAudit(
                    task_id=task.task_id,
                    domain=task.domain,
                    active_programs=active_count,
                    rejected_programs=rejected_count,
                )
            )
        resolved = tuple(evaluations)
        return (
            summarize_task_evaluations(resolved),
            resolved,
            results,
            tuple(audits),
        )

    @staticmethod
    def _candidate_matches_result(
        candidate: MacroOperatorCandidate,
        result: SolveResult,
    ) -> bool:
        length = len(candidate.operator_sequence)
        if length == 0 or length > len(result.proof):
            return False
        for start in range(len(result.proof) - length + 1):
            window = result.proof[start : start + length]
            names = tuple(step.action.operator.name for step in window)
            if names != candidate.operator_sequence:
                continue
            fingerprints = tuple(
                operator_schema_fingerprint(step.action.operator)
                for step in window
            )
            if fingerprints == candidate.operator_schema_fingerprints:
                return True
        return False

    @staticmethod
    def _validate_splits(
        training: tuple[LearningTask, ...],
        validation: tuple[LearningTask, ...],
        heldout: tuple[LearningTask, ...],
    ) -> None:
        if not training or not validation or not heldout:
            raise ValueError("macro learning requires train, validation, and heldout tasks")
        if any(task.split is not LearningSplit.TRAIN for task in training):
            raise ValueError("macro training tasks must use the train split")
        if any(
            task.split is not LearningSplit.HELDOUT
            for task in validation + heldout
        ):
            raise ValueError("macro validation and heldout tasks must use heldout split")
        identifiers = [
            task.task_id for task in training + validation + heldout
        ]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("macro learning task ids must be disjoint")
        if not any(task.expected_solved for task in heldout):
            raise ValueError("macro heldout tasks require positive examples")

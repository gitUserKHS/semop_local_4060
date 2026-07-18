from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import process_time

from .brain import HierarchicalOperatorBrain
from .engine import ActionPolicy, OperatorKernel, RegistryPolicyProvider
from .library import MdlMacroLibrary
from .macro_learning import (
    MacroLearningBudget,
    MacroLearningResult,
    VerifiedMacroLearningLoop,
)
from .macro_policy import PrimitiveMacroPolicy
from .model import SolveResult
from .self_learning import (
    LearningMetrics,
    LearningSplit,
    LearningTask,
    PolicyLearner,
    SelfLearningBudget,
    SelfLearningLoop,
    SelfLearningResult,
    TaskEvaluation,
    learning_expansion_reduction,
    summarize_task_evaluations,
    task_evaluation_from_result,
)


@dataclass(frozen=True)
class HierarchicalLearningBudget:
    controller: SelfLearningBudget = field(default_factory=SelfLearningBudget)
    macro: MacroLearningBudget = field(
        default_factory=lambda: MacroLearningBudget(min_domains_improved=3)
    )
    min_joint_expansion_reduction: float = 0.30
    max_joint_solve_rate_drop: float = 0.01
    required_proof_soundness: float = 1.0
    max_false_positives: int = 0
    min_active_domains: int = 3
    require_complementarity: bool = True
    max_parameters: int = 15_000_000
    max_artifact_bytes: int = 64 * 1024 * 1024
    max_p95_cpu_seconds: float = 10.0

    def __post_init__(self) -> None:
        bounded_rates = (
            self.min_joint_expansion_reduction,
            self.max_joint_solve_rate_drop,
            self.required_proof_soundness,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded_rates):
            raise ValueError("hierarchical learning rates must be between zero and one")
        if self.max_false_positives < 0:
            raise ValueError("hierarchical false-positive limit must not be negative")
        if not 1 <= self.min_active_domains <= 3:
            raise ValueError("hierarchical active-domain gate must be between one and three")
        if self.max_parameters <= 0 or self.max_artifact_bytes <= 0:
            raise ValueError("hierarchical resource limits must be positive")
        if self.max_p95_cpu_seconds <= 0:
            raise ValueError("hierarchical p95 CPU limit must be positive")


@dataclass(frozen=True)
class HierarchicalPolicyAblation:
    name: str
    metrics: LearningMetrics
    tasks: tuple[TaskEvaluation, ...]
    results: dict[str, SolveResult] = field(compare=False, repr=False)


@dataclass(frozen=True)
class HierarchicalLearningResult:
    promoted: bool
    rejection_reasons: tuple[str, ...]
    controller_learning: SelfLearningResult = field(compare=False, repr=False)
    macro_learning: MacroLearningResult = field(compare=False, repr=False)
    candidate_brain: HierarchicalOperatorBrain | None = field(
        compare=False,
        repr=False,
    )
    active_brain: HierarchicalOperatorBrain | None = field(
        compare=False,
        repr=False,
    )
    deterministic: HierarchicalPolicyAblation
    controller_only: HierarchicalPolicyAblation
    macro_only: HierarchicalPolicyAblation
    joint: HierarchicalPolicyAblation
    joint_expansion_reduction: float
    active_domains: tuple[str, ...]
    artifact_bytes: int


class HierarchicalSelfLearningLoop:
    """Promote a shared controller and procedural memory through separate gates."""

    def __init__(
        self,
        learner: PolicyLearner | None = None,
        *,
        budget: HierarchicalLearningBudget | None = None,
    ) -> None:
        if learner is None:
            from semop.tiny_controller.linear_policy import StructuralPolicyLearner

            learner = StructuralPolicyLearner()
        self.learner = learner
        self.budget = budget or HierarchicalLearningBudget()

    def run(
        self,
        controller_training_tasks: Sequence[LearningTask],
        controller_heldout_tasks: Sequence[LearningTask],
        macro_training_tasks: Sequence[LearningTask],
        macro_validation_tasks: Sequence[LearningTask],
        macro_heldout_tasks: Sequence[LearningTask],
        joint_heldout_tasks: Sequence[LearningTask],
        *,
        incumbent_policy: ActionPolicy | None = None,
    ) -> HierarchicalLearningResult:
        controller_training = tuple(controller_training_tasks)
        controller_heldout = tuple(controller_heldout_tasks)
        macro_training = tuple(macro_training_tasks)
        macro_validation = tuple(macro_validation_tasks)
        macro_heldout = tuple(macro_heldout_tasks)
        joint_heldout = tuple(joint_heldout_tasks)
        self._validate_splits(
            controller_training,
            controller_heldout,
            macro_training,
            macro_validation,
            macro_heldout,
            joint_heldout,
        )

        controller_learning = SelfLearningLoop(
            learner=self.learner,
            budget=self.budget.controller,
        ).run(
            controller_training,
            controller_heldout,
            incumbent_policy=incumbent_policy,
        )
        macro_learning = VerifiedMacroLearningLoop(
            self.budget.macro
        ).run(
            macro_training,
            macro_validation,
            macro_heldout,
        )

        active_candidate = controller_learning.active_candidate
        candidate_brain = None
        if active_candidate is not None and macro_learning.promoted:
            candidate_brain = HierarchicalOperatorBrain.from_candidate(
                active_candidate,
                macro_learning.active_library,
            )

        deterministic = self._evaluate(joint_heldout, name="deterministic")
        controller_only = self._evaluate(
            joint_heldout,
            name="controller_only",
            policy=controller_learning.final_policy,
        )
        macro_only = self._evaluate(
            joint_heldout,
            name="macro_only",
            macro_library=macro_learning.active_library,
        )
        joint = self._evaluate(
            joint_heldout,
            name="joint",
            policy=candidate_brain,
        )
        joint_reduction = learning_expansion_reduction(
            deterministic.metrics,
            joint.metrics,
        )
        active_domains = tuple(
            sorted(
                {
                    task.domain
                    for task in joint_heldout
                    if task.expected_solved
                    and macro_learning.active_library.activate(
                        task.instance.registry
                    ).programs
                }
            )
        )
        artifact_bytes = (
            len(candidate_brain.to_artifact())
            if candidate_brain is not None
            else 0
        )

        reasons: list[str] = []
        if active_candidate is None:
            reasons.append("controller_component_not_promoted")
            reasons.extend(
                f"controller:{reason}"
                for iteration in controller_learning.iterations
                for reason in iteration.rejection_reasons
            )
        if not macro_learning.promoted:
            reasons.append("macro_component_not_promoted")
            reasons.extend(
                f"macro:{reason}" for reason in macro_learning.rejection_reasons
            )
        if candidate_brain is None:
            reasons.append("hierarchical_brain_candidate_unavailable")

        best_component_rate = max(
            controller_only.metrics.verified_solve_rate,
            macro_only.metrics.verified_solve_rate,
        )
        if (
            joint.metrics.verified_solve_rate
            < best_component_rate - self.budget.max_joint_solve_rate_drop
        ):
            reasons.append("joint_solve_rate_regressed_against_component")
        if joint.metrics.proof_soundness < self.budget.required_proof_soundness:
            reasons.append("joint_proof_soundness_below_gate")
        if joint.metrics.false_positives > self.budget.max_false_positives:
            reasons.append("joint_false_positive_limit_exceeded")
        if joint_reduction < self.budget.min_joint_expansion_reduction:
            reasons.append("joint_expansion_reduction_below_gate")

        best_component_expansions = min(
            controller_only.metrics.positive_expansions,
            macro_only.metrics.positive_expansions,
        )
        if joint.metrics.positive_expansions > best_component_expansions:
            reasons.append("joint_expansions_exceed_best_component")
        if self.budget.require_complementarity and not (
            joint.metrics.positive_expansions
            < controller_only.metrics.positive_expansions
            and joint.metrics.positive_expansions
            < macro_only.metrics.positive_expansions
        ):
            reasons.append("controller_and_macro_are_not_complementary")
        if len(active_domains) < self.budget.min_active_domains:
            reasons.append("joint_active_domain_count_below_gate")
        if candidate_brain is not None:
            if candidate_brain.parameter_count > self.budget.max_parameters:
                reasons.append("hierarchical_parameter_limit_exceeded")
            if artifact_bytes > self.budget.max_artifact_bytes:
                reasons.append("hierarchical_artifact_limit_exceeded")
        if joint.metrics.p95_cpu_seconds > self.budget.max_p95_cpu_seconds:
            reasons.append("joint_p95_cpu_limit_exceeded")
        if not self._all_successes_replay_primitives(joint_heldout, joint):
            reasons.append("joint_primitive_replay_failed")

        resolved_reasons = tuple(dict.fromkeys(reasons))
        promoted = candidate_brain is not None and not resolved_reasons
        return HierarchicalLearningResult(
            promoted=promoted,
            rejection_reasons=resolved_reasons,
            controller_learning=controller_learning,
            macro_learning=macro_learning,
            candidate_brain=candidate_brain,
            active_brain=candidate_brain if promoted else None,
            deterministic=deterministic,
            controller_only=controller_only,
            macro_only=macro_only,
            joint=joint,
            joint_expansion_reduction=joint_reduction,
            active_domains=active_domains,
            artifact_bytes=artifact_bytes,
        )

    @staticmethod
    def persist_promoted(
        result: HierarchicalLearningResult,
        path: str | Path,
    ) -> Path | None:
        if not result.promoted or result.active_brain is None:
            return None
        return result.active_brain.save(path)

    def _evaluate(
        self,
        tasks: tuple[LearningTask, ...],
        *,
        name: str,
        policy: ActionPolicy | RegistryPolicyProvider | None = None,
        macro_library: MdlMacroLibrary | None = None,
    ) -> HierarchicalPolicyAblation:
        evaluations: list[TaskEvaluation] = []
        results: dict[str, SolveResult] = {}
        for task in tasks:
            active_policy = policy
            if macro_library is not None and macro_library.records:
                active_policy = PrimitiveMacroPolicy(
                    task.instance.registry,
                    macro_library,
                )
            started = process_time()
            result = OperatorKernel(task.instance.registry).solve(
                task.instance.state,
                task.instance.goals,
                policy=active_policy,
                budget=self.budget.controller.solve_budget,
            )
            results[task.task_id] = result
            evaluations.append(
                task_evaluation_from_result(
                    task,
                    result,
                    process_time() - started,
                )
            )
        resolved = tuple(evaluations)
        return HierarchicalPolicyAblation(
            name=name,
            metrics=summarize_task_evaluations(resolved),
            tasks=resolved,
            results=results,
        )

    @staticmethod
    def _all_successes_replay_primitives(
        tasks: tuple[LearningTask, ...],
        ablation: HierarchicalPolicyAblation,
    ) -> bool:
        for task in tasks:
            result = ablation.results[task.task_id]
            if not result.success:
                continue
            if any(
                step.action.operator.name.startswith("macro_")
                or step.action.operator.name
                not in task.instance.registry.operators
                for step in result.proof
            ):
                return False
            replay = OperatorKernel(task.instance.registry).replay(
                task.instance.state,
                task.instance.goals,
                result.proof,
            )
            if not replay.verified:
                return False
        return True

    @staticmethod
    def _validate_splits(
        controller_training: tuple[LearningTask, ...],
        controller_heldout: tuple[LearningTask, ...],
        macro_training: tuple[LearningTask, ...],
        macro_validation: tuple[LearningTask, ...],
        macro_heldout: tuple[LearningTask, ...],
        joint_heldout: tuple[LearningTask, ...],
    ) -> None:
        groups = (
            controller_training,
            controller_heldout,
            macro_training,
            macro_validation,
            macro_heldout,
            joint_heldout,
        )
        if any(not group for group in groups):
            raise ValueError("hierarchical self-learning requires every split")
        if any(
            task.split is not LearningSplit.TRAIN
            for task in controller_training + macro_training
        ):
            raise ValueError("hierarchical component training tasks must be train split")
        if any(
            task.split is not LearningSplit.HELDOUT
            for task in (
                controller_heldout
                + macro_validation
                + macro_heldout
                + joint_heldout
            )
        ):
            raise ValueError("hierarchical evaluation tasks must be heldout split")
        identifiers = [task.task_id for group in groups for task in group]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("hierarchical learning task ids must be disjoint")
        if not any(task.expected_solved for task in macro_heldout):
            raise ValueError("macro heldout split needs positive tasks")
        if not any(task.expected_solved for task in joint_heldout):
            raise ValueError("joint heldout split needs positive tasks")

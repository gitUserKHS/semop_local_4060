from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import math
from pathlib import Path
import re

from .engine import ActionPolicy, OperatorKernel, PolicyDecision
from .library import MdlMacroLibrary
from .model import SolveBudget, SolveResult
from .self_learning import (
    LearningSplit,
    LearningTask,
    PolicyCandidate,
    PolicyLearner,
    SelfLearningBudget,
    SelfLearningCheckpoint,
    SelfLearningIteration,
    SelfLearningLoop,
    SelfLearningStore,
)
from .traces import DecisionTrainingCase, TraceCorpus, build_decision_training_cases


@dataclass(frozen=True)
class TaskProfile:
    task_id: str
    domain: str
    capability: str
    structure_key: str
    program_signature: str
    expected_solved: bool
    outcome_matches_label: bool
    replay_verified: bool
    proof_depth: int
    operator_sequence: tuple[str, ...]
    operator_families: tuple[str, ...]
    goal_shapes: tuple[str, ...]
    state_predicates: tuple[str, ...]
    decision_cases: int
    policy_uncertainty: float
    difficulty: float


@dataclass(frozen=True)
class StructuralSplitAudit:
    valid: bool
    training_structures: tuple[str, ...]
    heldout_structures: tuple[str, ...]
    overlapping_structures: tuple[str, ...]
    training_programs: tuple[str, ...]
    heldout_programs: tuple[str, ...]
    overlapping_programs: tuple[str, ...]
    training_domains: tuple[str, ...]
    heldout_domains: tuple[str, ...]
    missing_heldout_domains: tuple[str, ...]
    training_outcomes_valid: bool
    heldout_outcomes_valid: bool
    training_profiles: tuple[TaskProfile, ...]
    heldout_profiles: tuple[TaskProfile, ...]


@dataclass(frozen=True)
class ActiveCurriculumConfig:
    max_tasks: int = 6
    min_tasks_per_domain: int = 1
    uncertainty_weight: float = 0.30
    novelty_weight: float = 0.45
    difficulty_weight: float = 0.10
    domain_balance_weight: float = 0.15

    def __post_init__(self) -> None:
        if self.max_tasks <= 0 or self.min_tasks_per_domain <= 0:
            raise ValueError("active curriculum task limits must be positive")
        weights = (
            self.uncertainty_weight,
            self.novelty_weight,
            self.difficulty_weight,
            self.domain_balance_weight,
        )
        if any(value < 0 or not math.isfinite(value) for value in weights):
            raise ValueError("active curriculum weights must be finite and non-negative")
        if sum(weights) <= 0:
            raise ValueError("active curriculum requires at least one positive weight")


@dataclass(frozen=True)
class CurriculumDecision:
    rank: int
    task_id: str
    domain: str
    capability: str
    structure_key: str
    priority: float
    uncertainty: float
    novelty: float
    difficulty: float
    domain_balance: float


@dataclass(frozen=True)
class CurriculumSelection:
    selected_tasks: tuple[LearningTask, ...]
    decisions: tuple[CurriculumDecision, ...]
    profiles: tuple[TaskProfile, ...]
    rejected_unverified: tuple[str, ...]
    rejected_no_supervision: tuple[str, ...]
    domain_counts: tuple[tuple[str, int], ...]
    seen_structure_keys_after: tuple[str, ...]


class ActiveCurriculumScheduler:
    """Select diverse, uncertain, verifier-solvable tasks on a fixed CPU budget."""

    def __init__(self, config: ActiveCurriculumConfig | None = None) -> None:
        self.config = config or ActiveCurriculumConfig()

    def select(
        self,
        tasks: Sequence[LearningTask],
        *,
        policy: ActionPolicy | None = None,
        seen_structure_keys: Sequence[str] = (),
        solve_budget: SolveBudget | None = None,
    ) -> CurriculumSelection:
        pool = tuple(tasks)
        if not pool:
            raise ValueError("active curriculum requires a non-empty task pool")
        if any(task.split is not LearningSplit.TRAIN for task in pool):
            raise ValueError("active curriculum accepts train-split tasks only")
        identifiers = [task.task_id for task in pool]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("active curriculum task ids must be unique")

        budget = solve_budget or SolveBudget()
        analyses = tuple(_analyze_task(task, policy, budget) for task in pool)
        profiles = tuple(profile for profile, _cases, _result in analyses)
        eligible = {
            profile.task_id: (task, profile)
            for task, (profile, _cases, _result) in zip(pool, analyses, strict=True)
            if (
                task.expected_solved
                and profile.replay_verified
                and profile.decision_cases > 0
            )
        }
        rejected = tuple(
            sorted(
                task.task_id
                for task, (profile, _cases, _result) in zip(
                    pool, analyses, strict=True
                )
                if task.expected_solved and not profile.replay_verified
            )
        )
        no_supervision = tuple(
            sorted(
                task.task_id
                for task, (profile, _cases, _result) in zip(
                    pool, analyses, strict=True
                )
                if (
                    task.expected_solved
                    and profile.replay_verified
                    and profile.decision_cases == 0
                )
            )
        )
        if not eligible:
            return CurriculumSelection(
                selected_tasks=(),
                decisions=(),
                profiles=profiles,
                rejected_unverified=rejected,
                rejected_no_supervision=no_supervision,
                domain_counts=(),
                seen_structure_keys_after=tuple(sorted(set(seen_structure_keys))),
            )

        domains = tuple(sorted({profile.domain for _task, profile in eligible.values()}))
        required_slots = len(domains) * self.config.min_tasks_per_domain
        if self.config.max_tasks < required_slots:
            raise ValueError(
                "max_tasks is too small for min_tasks_per_domain across all domains"
            )

        selected: list[LearningTask] = []
        decisions: list[CurriculumDecision] = []
        domain_counts: Counter[str] = Counter()
        dynamic_seen = set(seen_structure_keys)
        remaining = dict(eligible)
        limit = min(self.config.max_tasks, len(remaining))
        while remaining and len(selected) < limit:
            deficient = {
                domain
                for domain in domains
                if domain_counts[domain] < self.config.min_tasks_per_domain
            }
            candidates = tuple(
                value
                for value in remaining.values()
                if not deficient or value[1].domain in deficient
            )
            ranked = sorted(
                (
                    (
                        _priority(
                            profile,
                            dynamic_seen,
                            domain_counts,
                            self.config,
                        ),
                        task,
                        profile,
                    )
                    for task, profile in candidates
                ),
                key=lambda item: (-item[0][0], item[1].task_id),
            )
            components, task, profile = ranked[0]
            priority, novelty, difficulty, balance = components
            rank = len(selected) + 1
            selected.append(task)
            decisions.append(
                CurriculumDecision(
                    rank=rank,
                    task_id=task.task_id,
                    domain=profile.domain,
                    capability=profile.capability,
                    structure_key=profile.structure_key,
                    priority=priority,
                    uncertainty=profile.policy_uncertainty,
                    novelty=novelty,
                    difficulty=difficulty,
                    domain_balance=balance,
                )
            )
            domain_counts[profile.domain] += 1
            dynamic_seen.add(profile.structure_key)
            remaining.pop(task.task_id)

        return CurriculumSelection(
            selected_tasks=tuple(selected),
            decisions=tuple(decisions),
            profiles=profiles,
            rejected_unverified=rejected,
            rejected_no_supervision=no_supervision,
            domain_counts=tuple(sorted(domain_counts.items())),
            seen_structure_keys_after=tuple(sorted(dynamic_seen)),
        )


@dataclass(frozen=True)
class ActiveSelfLearningRound:
    round_index: int
    selection: CurriculumSelection
    learning_iteration: SelfLearningIteration


@dataclass(frozen=True)
class ActiveSelfLearningResult:
    final_policy: ActionPolicy | None = field(compare=False, repr=False)
    active_candidate: PolicyCandidate | None = field(compare=False, repr=False)
    corpus: TraceCorpus = field(compare=False, repr=False)
    macro_library: MdlMacroLibrary = field(compare=False, repr=False)
    split_audit: StructuralSplitAudit
    rounds: tuple[ActiveSelfLearningRound, ...]
    promoted_generations: int
    selected_task_ids: tuple[str, ...]
    checkpoint: SelfLearningCheckpoint | None = None


class ActiveSelfLearningLoop:
    """Run bounded continual generations selected from a capability task pool."""

    def __init__(
        self,
        learner: PolicyLearner | None = None,
        *,
        scheduler: ActiveCurriculumScheduler | None = None,
        self_learning_budget: SelfLearningBudget | None = None,
        generations: int = 1,
        store: SelfLearningStore | str | Path | None = None,
    ) -> None:
        if generations <= 0:
            raise ValueError("active self-learning generations must be positive")
        if learner is None:
            from semop.tiny_controller.linear_policy import StructuralPolicyLearner

            learner = StructuralPolicyLearner()
        self.learner = learner
        self.scheduler = scheduler or ActiveCurriculumScheduler()
        self.self_learning_budget = self_learning_budget or SelfLearningBudget()
        self.generations = generations
        self.store = (
            store
            if isinstance(store, SelfLearningStore) or store is None
            else SelfLearningStore(store)
        )

    def run(
        self,
        training_pool: Sequence[LearningTask],
        heldout_tasks: Sequence[LearningTask],
        *,
        incumbent_policy: ActionPolicy | None = None,
        seen_structure_keys: Sequence[str] = (),
    ) -> ActiveSelfLearningResult:
        pool = tuple(training_pool)
        heldout = tuple(heldout_tasks)
        audit = audit_structural_split(
            pool,
            heldout,
            solve_budget=self.self_learning_budget.solve_budget,
        )
        if not audit.valid:
            raise ValueError(
                "active self-learning requires a verified, non-overlapping "
                "structural split; "
                f"structure_overlap={list(audit.overlapping_structures)} "
                f"program_overlap={list(audit.overlapping_programs)}"
            )
        if self.store is not None and self.store.exists():
            raise ValueError(
                "active self-learning store already exists; use a new generation root"
            )

        remaining = {task.task_id: task for task in pool}
        seen = set(seen_structure_keys)
        corpus = TraceCorpus()
        macro_library = MdlMacroLibrary()
        policy = incumbent_policy
        active_candidate: PolicyCandidate | None = None
        promoted = 0
        rounds: list[ActiveSelfLearningRound] = []
        selected_ids: list[str] = []
        checkpoint: SelfLearningCheckpoint | None = None
        for round_index in range(1, self.generations + 1):
            if not remaining:
                break
            selection = self.scheduler.select(
                tuple(remaining.values()),
                policy=policy,
                seen_structure_keys=tuple(seen),
                solve_budget=self.self_learning_budget.solve_budget,
            )
            if not selection.selected_tasks:
                break
            inner_budget = replace(
                self.self_learning_budget,
                iterations=1,
                max_training_tasks=max(
                    self.self_learning_budget.max_training_tasks,
                    len(selection.selected_tasks),
                ),
            )
            learning = SelfLearningLoop(
                learner=self.learner,
                budget=inner_budget,
                corpus=corpus,
                macro_library=macro_library,
            ).run(
                selection.selected_tasks,
                heldout,
                incumbent_policy=policy,
            )
            raw_iteration = learning.iterations[0]
            generation_before = promoted
            if learning.promoted:
                promoted += 1
                policy = learning.final_policy
                active_candidate = learning.active_candidate
            corpus = learning.corpus
            macro_library = learning.macro_library
            adjusted = replace(
                raw_iteration,
                iteration=round_index,
                generation_before=generation_before,
                generation_after=promoted,
            )
            rounds.append(
                ActiveSelfLearningRound(
                    round_index=round_index,
                    selection=selection,
                    learning_iteration=adjusted,
                )
            )
            selected_ids.extend(task.task_id for task in selection.selected_tasks)
            for task in selection.selected_tasks:
                remaining.pop(task.task_id, None)
            seen.update(selection.seen_structure_keys_after)
            if self.store is not None:
                checkpoint = self.store.persist(
                    iteration=adjusted,
                    generation=promoted,
                    corpus=corpus,
                    macro_library=macro_library,
                    active_candidate=active_candidate,
                )

        return ActiveSelfLearningResult(
            final_policy=policy,
            active_candidate=active_candidate,
            corpus=corpus,
            macro_library=macro_library,
            split_audit=audit,
            rounds=tuple(rounds),
            promoted_generations=promoted,
            selected_task_ids=tuple(selected_ids),
            checkpoint=checkpoint,
        )


def profile_learning_task(
    task: LearningTask,
    *,
    policy: ActionPolicy | None = None,
    solve_budget: SolveBudget | None = None,
) -> TaskProfile:
    profile, _cases, _result = _analyze_task(
        task,
        policy,
        solve_budget or SolveBudget(),
    )
    return profile


def audit_structural_split(
    training_tasks: Sequence[LearningTask],
    heldout_tasks: Sequence[LearningTask],
    *,
    solve_budget: SolveBudget | None = None,
) -> StructuralSplitAudit:
    training = tuple(training_tasks)
    heldout = tuple(heldout_tasks)
    if not training or not heldout:
        raise ValueError("structural split audit requires both train and held-out tasks")
    budget = solve_budget or SolveBudget()
    train_profiles = tuple(
        profile_learning_task(task, solve_budget=budget) for task in training
    )
    heldout_profiles = tuple(
        profile_learning_task(task, solve_budget=budget) for task in heldout
    )
    train_structures = tuple(sorted({item.structure_key for item in train_profiles}))
    heldout_structures = tuple(sorted({item.structure_key for item in heldout_profiles}))
    overlap = tuple(sorted(set(train_structures) & set(heldout_structures)))
    train_programs = tuple(
        sorted(
            {
                item.program_signature
                for item in train_profiles
                if item.expected_solved
            }
        )
    )
    heldout_programs = tuple(
        sorted(
            {
                item.program_signature
                for item in heldout_profiles
                if item.expected_solved
            }
        )
    )
    program_overlap = tuple(
        sorted(set(train_programs) & set(heldout_programs))
    )
    train_domains = tuple(sorted({item.domain for item in train_profiles}))
    heldout_domains = tuple(sorted({item.domain for item in heldout_profiles}))
    missing_domains = tuple(sorted(set(train_domains) - set(heldout_domains)))
    train_valid = all(item.outcome_matches_label for item in train_profiles)
    heldout_valid = all(item.outcome_matches_label for item in heldout_profiles)
    return StructuralSplitAudit(
        valid=(
            not overlap
            and not program_overlap
            and not missing_domains
            and train_valid
            and heldout_valid
        ),
        training_structures=train_structures,
        heldout_structures=heldout_structures,
        overlapping_structures=overlap,
        training_programs=train_programs,
        heldout_programs=heldout_programs,
        overlapping_programs=program_overlap,
        training_domains=train_domains,
        heldout_domains=heldout_domains,
        missing_heldout_domains=missing_domains,
        training_outcomes_valid=train_valid,
        heldout_outcomes_valid=heldout_valid,
        training_profiles=train_profiles,
        heldout_profiles=heldout_profiles,
    )


def _analyze_task(
    task: LearningTask,
    policy: ActionPolicy | None,
    budget: SolveBudget,
) -> tuple[TaskProfile, tuple[DecisionTrainingCase, ...], SolveResult]:
    kernel = OperatorKernel(task.instance.registry)
    result = kernel.solve(
        task.instance.state,
        task.instance.goals,
        budget=budget,
    )
    cases = (
        build_decision_training_cases(kernel, result)
        if task.expected_solved and result.success and result.verified
        else ()
    )
    uncertainty = _policy_uncertainty(policy, cases)
    operator_sequence = tuple(step.action.operator.name for step in result.proof)
    operator_families = tuple(step.action.operator.family for step in result.proof)
    goal_shapes = tuple(
        f"{goal.atom.predicate.name}("
        + ",".join(argument.type.name for argument in goal.atom.arguments)
        + ")"
        for goal in task.instance.goals
    )
    state_predicates = tuple(
        sorted({fact.atom.predicate.name for fact in task.instance.state.facts})
    )
    structure_key = task.structure_key.strip() or _derived_structure_key(
        task.domain,
        goal_shapes,
        operator_families,
        len(result.proof),
    )
    outcome_matches = result.success == task.expected_solved
    program_signature = _program_signature(
        task.domain,
        goal_shapes,
        operator_sequence,
    )
    return (
        TaskProfile(
            task_id=task.task_id,
            domain=task.domain,
            capability=task.capability or structure_key,
            structure_key=structure_key,
            program_signature=program_signature,
            expected_solved=task.expected_solved,
            outcome_matches_label=outcome_matches,
            replay_verified=result.success and result.verified,
            proof_depth=len(result.proof),
            operator_sequence=operator_sequence,
            operator_families=operator_families,
            goal_shapes=goal_shapes,
            state_predicates=state_predicates,
            decision_cases=len(cases),
            policy_uncertainty=uncertainty,
            difficulty=min(1.0, max(task.difficulty, len(result.proof)) / 6.0),
        ),
        tuple(cases),
        result,
    )


def _policy_uncertainty(
    policy: ActionPolicy | None,
    cases: Sequence[DecisionTrainingCase],
) -> float:
    informative = tuple(case for case in cases if len(case.actions) > 1)
    if not informative:
        return 0.0
    if policy is None:
        return 1.0
    values: list[float] = []
    for case in informative:
        try:
            raw = policy.score_actions(case.state, case.goals, case.actions)
            scores = (
                raw.action_scores
                if isinstance(raw, PolicyDecision)
                else tuple(float(value) for value in raw)
            )
            if len(scores) != len(case.actions):
                raise ValueError("policy score count mismatch")
            gold = scores[case.target_action]
            strongest_negative = max(
                score
                for index, score in enumerate(scores)
                if index != case.target_action
            )
            margin = max(-30.0, min(30.0, gold - strongest_negative))
            values.append(1.0 / (1.0 + math.exp(margin)))
        except Exception:
            values.append(1.0)
    return sum(values) / len(values)


def _priority(
    profile: TaskProfile,
    seen: set[str],
    domain_counts: Counter[str],
    config: ActiveCurriculumConfig,
) -> tuple[float, float, float, float]:
    novelty = 0.0 if profile.structure_key in seen else 1.0
    difficulty = profile.difficulty
    balance = 1.0 / (1.0 + domain_counts[profile.domain])
    priority = (
        config.uncertainty_weight * profile.policy_uncertainty
        + config.novelty_weight * novelty
        + config.difficulty_weight * difficulty
        + config.domain_balance_weight * balance
    )
    return priority, novelty, difficulty, balance


def _derived_structure_key(
    domain: str,
    goal_shapes: tuple[str, ...],
    families: tuple[str, ...],
    proof_depth: int,
) -> str:
    family_shape = ">".join(families) or "observed"
    return (
        f"{domain}|goals:{'+'.join(goal_shapes)}|"
        f"families:{family_shape}|depth:{proof_depth}"
    )


def _program_signature(
    domain: str,
    goal_shapes: tuple[str, ...],
    operators: tuple[str, ...],
) -> str:
    normalized = tuple(
        re.sub(r"\d+", "#", operator).strip("_#")
        for operator in operators
    )
    operator_shape = ">".join(normalized) or "observed_or_unproved"
    return (
        f"{domain}|goals:{'+'.join(goal_shapes)}|operators:{operator_shape}"
    )

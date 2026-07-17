from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from itertools import combinations, product
from pathlib import Path

from .composition import (
    CompositionComponent,
    RegistryCompositionError,
    compose_domain_instances,
)
from .curriculum import (
    ActiveCurriculumConfig,
    ActiveCurriculumScheduler,
    ActiveSelfLearningLoop,
    ActiveSelfLearningResult,
    TaskProfile,
    profile_learning_task,
)
from .engine import ActionPolicy, OperatorKernel
from .model import FactStatus, SolveBudget, WorldState
from .self_learning import (
    LearningSplit,
    LearningTask,
    PolicyLearner,
    SelfLearningBudget,
    SelfLearningStore,
)


@dataclass(frozen=True)
class TaskDiscoveryBudget:
    """Hard limits for verifier-backed task mutation on an ordinary PC."""

    max_seed_tasks: int = 12
    max_seeds_per_domain: int = 4
    max_candidates: int = 64
    max_positive_tasks: int = 12
    max_negative_tasks: int = 12
    max_components: int = 3
    max_positive_proof_depth: int = 6
    max_candidate_proof_depth: int = 12
    counterfactuals_per_positive: int = 1
    max_counterfactual_attempts_per_positive: int = 8
    scaffold_long_proofs: bool = True
    solve_budget: SolveBudget = field(
        default_factory=lambda: SolveBudget(
            max_steps=32,
            max_expansions=20_000,
            timeout_seconds=10.0,
        )
    )

    def __post_init__(self) -> None:
        positive = (
            self.max_seed_tasks,
            self.max_seeds_per_domain,
            self.max_candidates,
            self.max_positive_tasks,
            self.max_components,
            self.max_positive_proof_depth,
            self.max_candidate_proof_depth,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("task-discovery limits must be positive")
        if (
            self.max_negative_tasks < 0
            or self.counterfactuals_per_positive < 0
            or self.max_counterfactual_attempts_per_positive <= 0
        ):
            raise ValueError(
                "task-discovery negative limits must be non-negative and "
                "the attempt cap must be positive"
            )
        if self.max_components < 2:
            raise ValueError("task discovery requires at least two components")
        if self.max_positive_proof_depth > self.max_candidate_proof_depth:
            raise ValueError(
                "positive proof depth cannot exceed candidate proof depth"
            )


@dataclass(frozen=True)
class DiscoveryCandidateRecord:
    candidate_id: str
    mutation: str
    parent_task_ids: tuple[str, ...]
    domain: str
    structure_key: str
    expected_solved: bool
    accepted: bool
    rejection_reasons: tuple[str, ...]
    proof_depth: int
    program_signature: str
    baseline_expansions: int
    policy_success: bool | None
    policy_expansions: int | None
    policy_uncertainty: float
    priority: float
    scaffolded_steps: int = 0
    removed_fact: str = ""


@dataclass(frozen=True)
class TaskDiscoveryResult:
    seed_task_ids: tuple[str, ...]
    positive_tasks: tuple[LearningTask, ...]
    negative_tasks: tuple[LearningTask, ...]
    records: tuple[DiscoveryCandidateRecord, ...]

    @property
    def accepted_records(self) -> tuple[DiscoveryCandidateRecord, ...]:
        return tuple(record for record in self.records if record.accepted)

    @property
    def rejected_records(self) -> tuple[DiscoveryCandidateRecord, ...]:
        return tuple(record for record in self.records if not record.accepted)

    @property
    def discovered_structures(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    task.structure_key
                    for task in self.positive_tasks + self.negative_tasks
                }
            )
        )


@dataclass(frozen=True)
class SelfDiscoveringLearningResult:
    training_discovery: TaskDiscoveryResult
    heldout_discovery: TaskDiscoveryResult
    expanded_training_tasks: tuple[LearningTask, ...]
    expanded_heldout_tasks: tuple[LearningTask, ...]
    learning: ActiveSelfLearningResult = field(compare=False, repr=False)


@dataclass(frozen=True)
class _VerifiedCandidate:
    task: LearningTask = field(compare=False, repr=False)
    profile: TaskProfile
    record: DiscoveryCandidateRecord


class VerifiedTaskDiscovery:
    """Discover typed composition tasks; retain only verifier-checked mutations."""

    def __init__(self, budget: TaskDiscoveryBudget | None = None) -> None:
        self.budget = budget or TaskDiscoveryBudget()

    def discover(
        self,
        seed_tasks: Sequence[LearningTask],
        *,
        policy: ActionPolicy | None = None,
    ) -> TaskDiscoveryResult:
        seeds = tuple(seed_tasks)
        if not seeds:
            raise ValueError("task discovery requires seed tasks")
        identifiers = tuple(task.task_id for task in seeds)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("task discovery seed ids must be unique")

        analyzed = tuple(
            (task, profile_learning_task(
                task,
                policy=policy,
                solve_budget=self.budget.solve_budget,
            ))
            for task in seeds
            if task.expected_solved
        )
        eligible = tuple(
            (task, profile)
            for task, profile in analyzed
            if profile.replay_verified and profile.decision_cases > 0
        )
        selected_seeds, seed_records = self._select_seeds(eligible)
        if len({task.domain for task, _profile in selected_seeds}) < 2:
            return TaskDiscoveryResult(identifiers, (), (), seed_records)

        source_structures = {profile.structure_key for _task, profile in eligible}
        source_programs = {profile.program_signature for _task, profile in eligible}
        candidates: list[_VerifiedCandidate] = []
        records: dict[str, DiscoveryCandidateRecord] = {
            record.candidate_id: record for record in seed_records
        }
        generated = 0
        grouped: dict[str, list[tuple[LearningTask, TaskProfile]]] = defaultdict(list)
        for item in selected_seeds:
            grouped[item[0].domain].append(item)
        domains = tuple(sorted(grouped))

        for component_count in range(
            2,
            min(self.budget.max_components, len(domains)) + 1,
        ):
            for domain_group in combinations(domains, component_count):
                pools = tuple(tuple(grouped[domain]) for domain in domain_group)
                for parent_group in product(*pools):
                    if generated >= self.budget.max_candidates:
                        break
                    generated += 1
                    candidate, record = self._compose_candidate(
                        parent_group,
                        policy=policy,
                    )
                    records[record.candidate_id] = record
                    if candidate is not None:
                        candidates.append(candidate)
                if generated >= self.budget.max_candidates:
                    break
            if generated >= self.budget.max_candidates:
                break

        ranked = sorted(
            candidates,
            key=lambda item: (
                -item.record.priority,
                item.record.candidate_id,
            ),
        )
        accepted: list[_VerifiedCandidate] = []
        seen_structures = set(source_structures)
        seen_programs = set(source_programs)
        for candidate in ranked:
            reasons: list[str] = []
            if candidate.profile.structure_key in seen_structures:
                reasons.append("duplicate_discovered_structure")
            if candidate.profile.program_signature in seen_programs:
                reasons.append("duplicate_discovered_program")
            if len(accepted) >= self.budget.max_positive_tasks:
                reasons.append("positive_capacity_reached")
            if reasons:
                records[candidate.record.candidate_id] = replace(
                    candidate.record,
                    accepted=False,
                    rejection_reasons=tuple(reasons),
                )
                continue
            accepted.append(candidate)
            seen_structures.add(candidate.profile.structure_key)
            seen_programs.add(candidate.profile.program_signature)

        negatives: list[LearningTask] = []
        for candidate in accepted:
            if len(negatives) >= self.budget.max_negative_tasks:
                break
            discovered, negative_records = self._counterfactuals(
                candidate,
                remaining=self.budget.max_negative_tasks - len(negatives),
            )
            negatives.extend(discovered)
            records.update(
                (record.candidate_id, record) for record in negative_records
            )

        return TaskDiscoveryResult(
            seed_task_ids=identifiers,
            positive_tasks=tuple(item.task for item in accepted),
            negative_tasks=tuple(negatives),
            records=tuple(records[key] for key in sorted(records)),
        )

    def _select_seeds(
        self,
        eligible: Sequence[tuple[LearningTask, TaskProfile]],
    ) -> tuple[
        tuple[tuple[LearningTask, TaskProfile], ...],
        tuple[DiscoveryCandidateRecord, ...],
    ]:
        grouped: dict[str, list[tuple[LearningTask, TaskProfile]]] = defaultdict(list)
        for task, profile in eligible:
            grouped[task.domain].append((task, profile))
        selected_by_domain: dict[
            str,
            list[tuple[LearningTask, TaskProfile]],
        ] = {}
        rejected: list[DiscoveryCandidateRecord] = []
        for domain in sorted(grouped):
            seen_structures: set[str] = set()
            domain_selected: list[tuple[LearningTask, TaskProfile]] = []
            ranked = sorted(
                grouped[domain],
                key=lambda item: (
                    -item[1].policy_uncertainty,
                    -item[1].difficulty,
                    item[0].task_id,
                ),
            )
            for item in ranked:
                if item[1].structure_key in seen_structures:
                    rejected.append(
                        _seed_rejection_record(
                            item[0],
                            item[1],
                            "duplicate_seed_structure",
                        )
                    )
                    continue
                if len(seen_structures) >= self.budget.max_seeds_per_domain:
                    rejected.append(
                        _seed_rejection_record(
                            item[0],
                            item[1],
                            "seed_domain_capacity_reached",
                        )
                    )
                    continue
                domain_selected.append(item)
                seen_structures.add(item[1].structure_key)
            selected_by_domain[domain] = domain_selected

        round_robin: list[tuple[LearningTask, TaskProfile]] = []
        max_domain_size = max(
            (len(items) for items in selected_by_domain.values()),
            default=0,
        )
        for offset in range(max_domain_size):
            for domain in sorted(selected_by_domain):
                items = selected_by_domain[domain]
                if offset < len(items):
                    round_robin.append(items[offset])
        retained = tuple(round_robin[: self.budget.max_seed_tasks])
        for task, profile in round_robin[self.budget.max_seed_tasks :]:
            rejected.append(
                _seed_rejection_record(
                    task,
                    profile,
                    "seed_global_capacity_reached",
                )
            )
        return retained, tuple(rejected)

    def _compose_candidate(
        self,
        parents: Sequence[tuple[LearningTask, TaskProfile]],
        *,
        policy: ActionPolicy | None,
    ) -> tuple[_VerifiedCandidate | None, DiscoveryCandidateRecord]:
        parent_tasks = tuple(item[0] for item in parents)
        parent_profiles = tuple(item[1] for item in parents)
        parent_ids = tuple(task.task_id for task in parent_tasks)
        digest = _stable_id("|".join(parent_ids))
        split = parent_tasks[0].split
        candidate_id = f"discover-{split.value}-compose-{digest}"
        structure_parts = tuple(profile.structure_key for profile in parent_profiles)
        structure_key = "multidomain:compose[" + "+".join(structure_parts) + "]"
        capability = "compose[" + "+".join(
            task.capability or task.domain for task in parent_tasks
        ) + "]"
        empty = DiscoveryCandidateRecord(
            candidate_id=candidate_id,
            mutation="domain_composition",
            parent_task_ids=parent_ids,
            domain="multidomain",
            structure_key=structure_key,
            expected_solved=True,
            accepted=False,
            rejection_reasons=(),
            proof_depth=0,
            program_signature="",
            baseline_expansions=0,
            policy_success=None,
            policy_expansions=None,
            policy_uncertainty=sum(
                profile.policy_uncertainty for profile in parent_profiles
            )
            / len(parent_profiles),
            priority=0.0,
        )
        if any(task.split is not split for task in parent_tasks):
            return None, _rejected_record(empty, "mixed_learning_splits")

        try:
            composition = compose_domain_instances(
                tuple(
                    CompositionComponent(
                        task.instance,
                        alias=f"{task.domain}_{index}",
                    )
                    for index, task in enumerate(parent_tasks)
                ),
                domain="multidomain",
            )
        except RegistryCompositionError as exc:
            return None, _rejected_record(
                empty,
                f"registry_composition_failed:{type(exc).__name__}",
            )

        instance = replace(
            composition.instance,
            metadata={
                **composition.instance.metadata,
                "discovery_mutation": "domain_composition",
                "discovery_parent_ids": parent_ids,
                "discovery_structure": structure_key,
            },
        )
        task = LearningTask(
            task_id=candidate_id,
            instance=instance,
            expected_solved=True,
            split=split,
            source="synthetic",
            domain="multidomain",
            capability=capability,
            structure_key=structure_key,
            difficulty=min(12, sum(task.difficulty for task in parent_tasks)),
        )
        kernel = OperatorKernel(instance.registry)
        baseline = kernel.solve(
            instance.state,
            instance.goals,
            budget=self.budget.solve_budget,
        )
        if not baseline.success or not baseline.verified:
            return None, _rejected_record(
                replace(empty, baseline_expansions=baseline.expansions),
                "composed_candidate_not_replay_verified",
            )
        if len(baseline.proof) > self.budget.max_candidate_proof_depth:
            return None, _rejected_record(
                replace(
                    empty,
                    proof_depth=len(baseline.proof),
                    baseline_expansions=baseline.expansions,
                ),
                "candidate_proof_depth_exceeded",
            )

        scaffolded = 0
        if len(baseline.proof) > self.budget.max_positive_proof_depth:
            if not self.budget.scaffold_long_proofs:
                return None, _rejected_record(
                    replace(
                        empty,
                        proof_depth=len(baseline.proof),
                        baseline_expansions=baseline.expansions,
                    ),
                    "positive_proof_depth_exceeded",
                )
            scaffolded = len(baseline.proof) - self.budget.max_positive_proof_depth
            state = instance.state
            for step in baseline.proof[:scaffolded]:
                state = kernel.execute_action(state, step.action)
            structure_key = f"{structure_key}|verified_suffix:{scaffolded}"
            instance = replace(
                instance,
                state=state,
                metadata={
                    **instance.metadata,
                    "discovery_scaffolded_steps": scaffolded,
                },
            )
            task = replace(
                task,
                instance=instance,
                structure_key=structure_key,
                difficulty=max(1, task.difficulty - scaffolded),
            )
            kernel = OperatorKernel(instance.registry)
            baseline = kernel.solve(
                instance.state,
                instance.goals,
                budget=self.budget.solve_budget,
            )
            if (
                not baseline.success
                or not baseline.verified
                or len(baseline.proof) > self.budget.max_positive_proof_depth
            ):
                return None, _rejected_record(
                    replace(
                        empty,
                        structure_key=structure_key,
                        proof_depth=len(baseline.proof),
                        baseline_expansions=baseline.expansions,
                        scaffolded_steps=scaffolded,
                    ),
                    "verified_suffix_scaffold_failed",
                )

        profile = profile_learning_task(
            task,
            policy=policy,
            solve_budget=self.budget.solve_budget,
        )
        policy_success: bool | None = None
        policy_expansions: int | None = None
        if policy is not None:
            guided = kernel.solve(
                instance.state,
                instance.goals,
                policy=policy,
                budget=self.budget.solve_budget,
            )
            policy_success = (
                guided.success
                and guided.verified
                and not guided.fallback_used
            )
            policy_expansions = guided.expansions
        failure_bonus = 1.0 if policy_success is False else 0.0
        priority = (
            failure_bonus
            + 0.55 * profile.policy_uncertainty
            + 0.30 * min(1.0, len(baseline.proof) / 6.0)
            + 0.15 * min(1.0, len(parent_tasks) / self.budget.max_components)
        )
        record = replace(
            empty,
            structure_key=structure_key,
            accepted=True,
            proof_depth=len(baseline.proof),
            program_signature=profile.program_signature,
            baseline_expansions=baseline.expansions,
            policy_success=policy_success,
            policy_expansions=policy_expansions,
            policy_uncertainty=profile.policy_uncertainty,
            priority=priority,
            scaffolded_steps=scaffolded,
        )
        return _VerifiedCandidate(task, profile, record), record

    def _counterfactuals(
        self,
        candidate: _VerifiedCandidate,
        *,
        remaining: int,
    ) -> tuple[tuple[LearningTask, ...], tuple[DiscoveryCandidateRecord, ...]]:
        limit = min(self.budget.counterfactuals_per_positive, remaining)
        if limit <= 0:
            return (), ()
        task = candidate.task
        kernel = OperatorKernel(task.instance.registry)
        solved = kernel.solve(
            task.instance.state,
            task.instance.goals,
            budget=self.budget.solve_budget,
        )
        initial_atoms = task.instance.state.eligible_atoms
        support_atoms = {
            premise
            for step in solved.proof
            for premise in step.premises
            if premise in initial_atoms
        }
        status_by_atom = {
            fact.atom: fact.status for fact in task.instance.state.eligible_facts
        }
        ordered = sorted(
            support_atoms,
            key=lambda atom: (
                _fact_status_priority(status_by_atom.get(atom)),
                atom.canonical_key(),
            ),
        )
        negatives: list[LearningTask] = []
        records: list[DiscoveryCandidateRecord] = []
        attempts = 0
        for atom in ordered:
            if (
                len(negatives) >= limit
                or attempts
                >= self.budget.max_counterfactual_attempts_per_positive
            ):
                break
            attempts += 1
            digest = _stable_id(
                candidate.record.candidate_id + "|" + repr(atom.canonical_key())
            )
            candidate_id = f"{candidate.record.candidate_id}-ablate-{digest}"
            state = WorldState(
                tuple(
                    fact
                    for fact in task.instance.state.facts
                    if fact.atom != atom
                ),
                depth=task.instance.state.depth,
                path_cost=task.instance.state.path_cost,
            )
            structure_key = (
                f"{task.structure_key}|counterfactual:{atom.predicate.name}"
            )
            instance = replace(
                task.instance,
                state=state,
                metadata={
                    **task.instance.metadata,
                    "discovery_mutation": "support_ablation",
                    "discovery_removed_fact": str(atom),
                },
            )
            negative = LearningTask(
                task_id=candidate_id,
                instance=instance,
                expected_solved=False,
                split=task.split,
                source="synthetic",
                domain=task.domain,
                capability=task.capability,
                structure_key=structure_key,
                difficulty=task.difficulty,
            )
            outcome = kernel.solve(
                state,
                instance.goals,
                budget=self.budget.solve_budget,
            )
            accepted = not outcome.success and not outcome.verified
            reasons = () if accepted else ("ablation_remained_solvable",)
            records.append(
                DiscoveryCandidateRecord(
                    candidate_id=candidate_id,
                    mutation="support_ablation",
                    parent_task_ids=(task.task_id,),
                    domain=task.domain,
                    structure_key=structure_key,
                    expected_solved=False,
                    accepted=accepted,
                    rejection_reasons=reasons,
                    proof_depth=len(outcome.proof),
                    program_signature="",
                    baseline_expansions=outcome.expansions,
                    policy_success=None,
                    policy_expansions=None,
                    policy_uncertainty=candidate.profile.policy_uncertainty,
                    priority=candidate.record.priority,
                    scaffolded_steps=candidate.record.scaffolded_steps,
                    removed_fact=str(atom),
                )
            )
            if accepted:
                negatives.append(negative)
        return tuple(negatives), tuple(records)


class SelfDiscoveringLearningLoop:
    """Discover new compositions independently on train/held-out, then learn."""

    def __init__(
        self,
        learner: PolicyLearner | None = None,
        *,
        training_discovery: VerifiedTaskDiscovery | None = None,
        heldout_discovery: VerifiedTaskDiscovery | None = None,
        scheduler: ActiveCurriculumScheduler | None = None,
        self_learning_budget: SelfLearningBudget | None = None,
        generations: int = 1,
        store: SelfLearningStore | str | Path | None = None,
    ) -> None:
        self.learner = learner
        self.training_discovery = training_discovery or VerifiedTaskDiscovery()
        self.heldout_discovery = heldout_discovery or VerifiedTaskDiscovery(
            TaskDiscoveryBudget(
                max_positive_tasks=8,
                max_negative_tasks=8,
                max_positive_proof_depth=12,
                max_candidate_proof_depth=12,
                scaffold_long_proofs=False,
            )
        )
        self.scheduler = scheduler or ActiveCurriculumScheduler(
            ActiveCurriculumConfig(max_tasks=8)
        )
        self.self_learning_budget = self_learning_budget or SelfLearningBudget()
        self.generations = generations
        self.store = store

    def run(
        self,
        training_seeds: Sequence[LearningTask],
        heldout_tasks: Sequence[LearningTask],
        *,
        incumbent_policy: ActionPolicy | None = None,
    ) -> SelfDiscoveringLearningResult:
        training = tuple(training_seeds)
        heldout = tuple(heldout_tasks)
        heldout_positive = tuple(task for task in heldout if task.expected_solved)
        training_discovery = self.training_discovery.discover(
            training,
            policy=incumbent_policy,
        )
        # Evaluation discovery stays policy-independent to avoid tailoring the gate.
        heldout_discovery = self.heldout_discovery.discover(heldout_positive)
        expanded_training = _unique_tasks(
            training + training_discovery.positive_tasks
        )
        training_safety_controls = tuple(
            replace(task, split=LearningSplit.HELDOUT)
            for task in training_discovery.negative_tasks
        )
        expanded_heldout = _unique_tasks(
            heldout
            + training_safety_controls
            + heldout_discovery.positive_tasks
            + heldout_discovery.negative_tasks
        )
        learning = ActiveSelfLearningLoop(
            learner=self.learner,
            scheduler=self.scheduler,
            self_learning_budget=self.self_learning_budget,
            generations=self.generations,
            store=self.store,
        ).run(
            expanded_training,
            expanded_heldout,
            incumbent_policy=incumbent_policy,
        )
        return SelfDiscoveringLearningResult(
            training_discovery=training_discovery,
            heldout_discovery=heldout_discovery,
            expanded_training_tasks=expanded_training,
            expanded_heldout_tasks=expanded_heldout,
            learning=learning,
        )


def _rejected_record(
    record: DiscoveryCandidateRecord,
    reason: str,
) -> DiscoveryCandidateRecord:
    return replace(
        record,
        accepted=False,
        rejection_reasons=(reason,),
    )


def _fact_status_priority(status: FactStatus | None) -> int:
    if status in {FactStatus.OBSERVED, FactStatus.ASSUMED}:
        return 0
    if status is FactStatus.DERIVED:
        return 1
    return 2


def _seed_rejection_record(
    task: LearningTask,
    profile: TaskProfile,
    reason: str,
) -> DiscoveryCandidateRecord:
    return DiscoveryCandidateRecord(
        candidate_id=f"discover-seed-filter-{_stable_id(task.task_id)}",
        mutation="seed_selection",
        parent_task_ids=(task.task_id,),
        domain=task.domain,
        structure_key=profile.structure_key,
        expected_solved=task.expected_solved,
        accepted=False,
        rejection_reasons=(reason,),
        proof_depth=profile.proof_depth,
        program_signature=profile.program_signature,
        baseline_expansions=0,
        policy_success=None,
        policy_expansions=None,
        policy_uncertainty=profile.policy_uncertainty,
        priority=profile.policy_uncertainty,
    )


def _stable_id(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:12]


def _unique_tasks(tasks: Sequence[LearningTask]) -> tuple[LearningTask, ...]:
    found: dict[str, LearningTask] = {}
    for task in tasks:
        if task.task_id in found and found[task.task_id] != task:
            raise ValueError(f"conflicting discovered task id: {task.task_id}")
        found[task.task_id] = task
    return tuple(found[key] for key in sorted(found))

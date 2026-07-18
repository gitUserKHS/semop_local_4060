from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import product
from time import perf_counter
from typing import Protocol, runtime_checkable

from .model import (
    AssertionStatus,
    Atom,
    EvidenceStatus,
    Fact,
    FactStatus,
    Goal,
    GoalOutcome,
    GroundAction,
    KernelError,
    OperatorSpec,
    ProofStep,
    SolveBudget,
    SolveResult,
    Term,
    TypeValidationError,
    WorldState,
    collect_proof_dependencies,
)
from .registry import KernelRegistry
from .unification import substitute_atom, unify_atom


def _requirement_key(
    requirement: tuple[str, tuple[Term | None, ...]],
) -> tuple:
    predicate, arguments = requirement
    return (
        predicate,
        tuple(
            ("wildcard",) if argument is None else argument.canonical_key()
            for argument in arguments
        ),
    )


@dataclass(frozen=True)
class PolicyDecision:
    action_scores: tuple[float, ...]
    halt_probability: float = 0.0
    state_value: float = 0.0
    action_limit: int | None = None


@runtime_checkable
class ActionPolicy(Protocol):
    """A controller may rank verified actions, but cannot mutate world state."""

    def score_actions(
        self,
        state: WorldState,
        goals: Sequence[Goal],
        actions: Sequence[GroundAction],
    ) -> PolicyDecision | Sequence[float]: ...


@runtime_checkable
class RegistryPolicyProvider(Protocol):
    """Resolve a registry-specific policy without gaining state mutation access."""

    def policy_for(self, registry: KernelRegistry) -> ActionPolicy | None: ...


@dataclass(frozen=True)
class ReplayResult:
    verified: bool
    final_state: WorldState
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SearchResult:
    success: bool
    state: WorldState
    proof: tuple[ProofStep, ...]
    expansions: int
    halt_reason: str
    policy_used: bool
    policy_failed: bool
    diagnostics: tuple[str, ...]
    inference_rounds: int


class OperatorKernel:
    """Typed forward search with a mandatory proof-replay verifier."""

    def __init__(self, registry: KernelRegistry) -> None:
        self.registry = registry

    def solve(
        self,
        state: WorldState,
        goals: Sequence[Goal],
        policy: ActionPolicy | RegistryPolicyProvider | None = None,
        budget: SolveBudget | None = None,
    ) -> SolveResult:
        solve_budget = budget or SolveBudget()
        normalized_goals = tuple(goals)
        if not normalized_goals:
            raise KernelError("solve requires at least one goal")
        started = perf_counter()

        provider_failed = False
        provider_diagnostics: tuple[str, ...] = ()
        resolved_policy: ActionPolicy | None
        if isinstance(policy, RegistryPolicyProvider):
            try:
                resolved_policy = policy.policy_for(self.registry)
                if resolved_policy is not None and not isinstance(
                    resolved_policy,
                    ActionPolicy,
                ):
                    raise TypeError("policy provider returned an invalid policy")
            except Exception as exc:
                resolved_policy = None
                provider_failed = True
                provider_diagnostics = (
                    "registry policy provider failed; deterministic search used: "
                    f"{type(exc).__name__}: {exc}",
                )
        else:
            resolved_policy = policy

        if resolved_policy is None:
            search = self._forward_chain(
                state, normalized_goals, None, solve_budget, started
            )
            if provider_failed:
                search = replace(
                    search,
                    policy_used=True,
                    policy_failed=True,
                    diagnostics=provider_diagnostics + search.diagnostics,
                )
            fallback_used = provider_failed
        else:
            guided_limit = max(1, int(solve_budget.max_expansions * 0.45))
            guided_budget = replace(solve_budget, max_expansions=guided_limit)
            search = self._forward_chain(
                state, normalized_goals, resolved_policy, guided_budget, started
            )
            fallback_used = search.policy_failed
            retryable = search.halt_reason in {
                "max_expansions",
                "max_steps",
                "timeout",
                "proof_construction_failed",
            }
            if not search.success and retryable:
                remaining_expansions = solve_budget.max_expansions - search.expansions
                remaining_time = solve_budget.timeout_seconds - (perf_counter() - started)
                if remaining_expansions > 0 and remaining_time > 0:
                    fallback_budget = replace(
                        solve_budget,
                        max_expansions=remaining_expansions,
                        timeout_seconds=remaining_time,
                    )
                    fallback = self._forward_chain(
                        state,
                        normalized_goals,
                        None,
                        fallback_budget,
                        perf_counter(),
                    )
                    search = replace(
                        fallback,
                        expansions=search.expansions + fallback.expansions,
                        policy_used=True,
                        policy_failed=search.policy_failed,
                        diagnostics=search.diagnostics
                        + ("guided search did not solve; deterministic fallback used",)
                        + fallback.diagnostics,
                    )
                    fallback_used = True

        replay = self.replay(state, normalized_goals, search.proof)
        verified = search.success and replay.verified
        final_state = replay.final_state if search.success else search.state
        diagnostics = search.diagnostics + replay.diagnostics
        if search.success and not replay.verified:
            diagnostics += ("search trace was rejected by proof replay",)

        outcomes = self._goal_outcomes(normalized_goals, final_state, search.proof)
        verified_proof = search.proof if verified else ()
        return SolveResult(
            success=verified and all(outcome.proven for outcome in outcomes),
            verified=verified,
            goals=outcomes,
            initial_state=state,
            final_state=final_state,
            proof=verified_proof,
            expansions=search.expansions,
            elapsed_seconds=perf_counter() - started,
            halt_reason=(search.halt_reason if verified or not search.success else "replay_failed"),
            policy_used=search.policy_used,
            fallback_used=fallback_used,
            diagnostics=diagnostics,
            inference_rounds=search.inference_rounds,
            dependencies=collect_proof_dependencies(
                state,
                normalized_goals,
                verified_proof,
            ),
        )

    def enumerate_actions(
        self,
        state: WorldState,
        budget: SolveBudget | None = None,
        *,
        operator_names: frozenset[str] | None = None,
    ) -> tuple[GroundAction, ...]:
        """Return all currently applicable, effectful actions deterministically."""

        solve_budget = budget or SolveBudget()
        facts_by_predicate: dict[str, list[Atom]] = defaultdict(list)
        for fact in state.eligible_facts:
            facts_by_predicate[fact.atom.predicate.name].append(fact.atom)
        for atoms in facts_by_predicate.values():
            atoms.sort(key=Atom.canonical_key)

        actions: dict[tuple, GroundAction] = {}
        for operator in sorted(
            self.registry.operators.values(), key=lambda item: item.name
        ):
            if operator_names is not None and operator.name not in operator_names:
                continue
            substitutions = self._match_preconditions(
                operator, facts_by_predicate
            )
            substitutions = self._complete_bindings(
                operator, substitutions, state, solve_budget.argument_candidates
            )
            for substitution in substitutions:
                if not self.registry.evaluate_guards(operator, substitution, state):
                    continue
                try:
                    preconditions = tuple(
                        substitute_atom(atom, substitution, self.registry)
                        for atom in operator.preconditions
                    )
                    effects = tuple(
                        substitute_atom(atom, substitution, self.registry)
                        for atom in operator.effects
                    )
                except (KeyError, TypeValidationError):
                    continue
                if not any(
                    effect.predicate.verified and not state.contains(effect)
                    for effect in effects
                ):
                    continue
                bindings = tuple(
                    (parameter.name, substitution[parameter.name])
                    for parameter in operator.parameters
                )
                action = GroundAction(
                    operator=operator,
                    bindings=bindings,
                    preconditions=preconditions,
                    effects=effects,
                    cost=operator.cost,
                )
                semantic_key = (
                    operator.name,
                    tuple(atom.canonical_key() for atom in preconditions),
                    tuple(atom.canonical_key() for atom in effects),
                )
                previous = actions.get(semantic_key)
                if previous is None or action.canonical_key() < previous.canonical_key():
                    actions[semantic_key] = action
        return tuple(actions[key] for key in sorted(actions))

    def policy_goals(
        self,
        state: WorldState,
        goals: Sequence[Goal],
        actions: Sequence[GroundAction] | None = None,
        budget: SolveBudget | None = None,
    ) -> tuple[Goal, ...]:
        """Return verifier goals plus bounded grounded frontiers for policy input."""

        solve_budget = budget or SolveBudget()
        normalized_goals = tuple(goals)
        available = (
            tuple(actions)
            if actions is not None
            else self.enumerate_actions(state, solve_budget)
        )
        requirements = self._backward_requirements(
            normalized_goals,
            max_depth=solve_budget.max_steps,
            max_requirements=min(
                solve_budget.max_facts,
                solve_budget.max_expansions,
            ),
        )
        return self._policy_frontier_goals(
            normalized_goals,
            available,
            requirements,
        )

    def execute_action(
        self, state: WorldState, action: GroundAction
    ) -> WorldState:
        """Validate and execute one registered ground action."""

        operator = self.registry.operators.get(action.operator.name)
        if operator is None or operator != action.operator:
            raise KernelError(f"unknown or changed operator: {action.operator.name}")
        bindings = action.binding_map()
        if set(bindings) != {parameter.name for parameter in operator.parameters}:
            raise KernelError(f"incomplete bindings for operator {operator.name}")
        for parameter in operator.parameters:
            if not self.registry.types.is_assignable(
                bindings[parameter.name].type, parameter.type
            ):
                raise TypeValidationError(
                    f"ill-typed binding ?{parameter.name} for {operator.name}"
                )
        preconditions = tuple(
            substitute_atom(atom, bindings, self.registry)
            for atom in operator.preconditions
        )
        effects = tuple(
            substitute_atom(atom, bindings, self.registry)
            for atom in operator.effects
        )
        if preconditions != action.preconditions or effects != action.effects:
            raise KernelError(f"ground action payload mismatch for {operator.name}")
        missing = [atom for atom in preconditions if not state.contains(atom)]
        if missing:
            raise KernelError(
                "action has unavailable premises: " + ", ".join(map(str, missing))
            )
        if not self.registry.evaluate_guards(operator, bindings, state):
            raise KernelError(f"guard rejected operator {operator.name}")
        return self._apply_action(state, action)

    def replay(
        self,
        initial_state: WorldState,
        goals: Sequence[Goal],
        proof: Sequence[ProofStep],
    ) -> ReplayResult:
        state = initial_state
        diagnostics: list[str] = []
        for expected_index, step in enumerate(proof, start=1):
            if step.index != expected_index:
                return ReplayResult(
                    False,
                    state,
                    (f"proof step index mismatch at {expected_index}",),
                )
            operator = self.registry.operators.get(step.action.operator.name)
            if operator is None or operator != step.action.operator:
                return ReplayResult(
                    False,
                    state,
                    (f"unknown or changed operator at step {expected_index}",),
                )
            bindings = step.action.binding_map()
            if set(bindings) != {parameter.name for parameter in operator.parameters}:
                return ReplayResult(
                    False,
                    state,
                    (f"incomplete bindings at step {expected_index}",),
                )
            for parameter in operator.parameters:
                term = bindings[parameter.name]
                if not self.registry.types.is_assignable(term.type, parameter.type):
                    return ReplayResult(
                        False,
                        state,
                        (f"ill-typed binding at step {expected_index}",),
                    )
            grounded_preconditions = tuple(
                substitute_atom(atom, bindings, self.registry)
                for atom in operator.preconditions
            )
            grounded_effects = tuple(
                substitute_atom(atom, bindings, self.registry)
                for atom in operator.effects
            )
            if grounded_preconditions != step.premises:
                return ReplayResult(
                    False,
                    state,
                    (f"premise trace mismatch at step {expected_index}",),
                )
            if grounded_effects != step.effects:
                return ReplayResult(
                    False,
                    state,
                    (f"effect trace mismatch at step {expected_index}",),
                )
            missing = [
                atom for atom in grounded_preconditions if not state.contains(atom)
            ]
            if missing:
                return ReplayResult(
                    False,
                    state,
                    (
                        f"step {expected_index} has unavailable premises: "
                        + ", ".join(map(str, missing)),
                    ),
                )
            if not self.registry.evaluate_guards(operator, bindings, state):
                return ReplayResult(
                    False,
                    state,
                    (f"guard rejected step {expected_index}",),
                )
            if state.digest() != step.before_digest:
                return ReplayResult(
                    False,
                    state,
                    (f"before-state digest mismatch at step {expected_index}",),
                )
            state = self.execute_action(state, step.action)
            if state.digest() != step.after_digest:
                return ReplayResult(
                    False,
                    state,
                    (f"after-state digest mismatch at step {expected_index}",),
                )
        missing_goals = [goal for goal in goals if not state.contains(goal.atom)]
        if missing_goals:
            diagnostics.append(
                "replay ended without goals: "
                + ", ".join(str(goal) for goal in missing_goals)
            )
        return ReplayResult(not missing_goals, state, tuple(diagnostics))

    def _forward_chain(
        self,
        initial_state: WorldState,
        goals: tuple[Goal, ...],
        policy: ActionPolicy | None,
        budget: SolveBudget,
        started: float,
    ) -> _SearchResult:
        policy_used = policy is not None
        if self._goals_met(initial_state, goals):
            return _SearchResult(
                success=True,
                state=initial_state,
                proof=(),
                expansions=0,
                halt_reason="goals_proven",
                policy_used=policy_used,
                policy_failed=False,
                diagnostics=(),
                inference_rounds=0,
            )

        state = initial_state
        active_policy = policy
        policy_failed = False
        diagnostics: list[str] = []
        expansions = 0
        inference_rounds = 0
        producers: dict[Atom, GroundAction] = {}
        relevant_operators = self._relevant_operator_names(goals)
        backward_requirements = self._backward_requirements(
            goals,
            max_depth=budget.max_steps,
            max_requirements=min(budget.max_facts, budget.max_expansions),
        )

        while True:
            if perf_counter() - started >= budget.timeout_seconds:
                return _SearchResult(
                    False,
                    state,
                    (),
                    expansions,
                    "timeout",
                    policy_used,
                    policy_failed,
                    tuple(diagnostics),
                    inference_rounds,
                )
            if inference_rounds >= budget.max_steps:
                return _SearchResult(
                    False,
                    state,
                    (),
                    expansions,
                    "max_steps",
                    policy_used,
                    policy_failed,
                    tuple(diagnostics),
                    inference_rounds,
                )

            actions = self.enumerate_actions(
                state,
                budget,
                operator_names=relevant_operators,
            )
            if not actions:
                return _SearchResult(
                    False,
                    state,
                    (),
                    expansions,
                    "no_solution",
                    policy_used,
                    policy_failed,
                    tuple(diagnostics),
                    inference_rounds,
                )

            selected = actions
            if active_policy is not None:
                try:
                    policy_goals = self._policy_frontier_goals(
                        goals,
                        actions,
                        backward_requirements,
                    )
                    decision = self._normalize_policy_decision(
                        active_policy.score_actions(state, policy_goals, actions),
                        len(actions),
                    )
                    if decision.halt_probability >= 0.5:
                        message = (
                            "policy halt was rejected because verifier goals were not met"
                        )
                        if message not in diagnostics:
                            diagnostics.append(message)
                    ranked = sorted(
                        zip(actions, decision.action_scores, strict=True),
                        key=lambda pair: (-pair[1], pair[0].canonical_key()),
                    )
                    allowed_operators: list[str] = []
                    for action, _score in ranked:
                        if action.operator.name not in allowed_operators:
                            allowed_operators.append(action.operator.name)
                        if len(allowed_operators) >= budget.top_operators:
                            break
                    selection_limit = budget.beam_width
                    if decision.action_limit is not None:
                        selection_limit = min(
                            selection_limit,
                            decision.action_limit,
                        )
                    selected = tuple(
                        action
                        for action, _score in ranked
                        if action.operator.name in allowed_operators
                    )[:selection_limit]
                except Exception as exc:  # a policy cannot break verified execution
                    policy_failed = True
                    active_policy = None
                    diagnostics.append(
                        "policy failure ignored; deterministic agenda fallback used: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    selected = actions

            working_state = state
            made_progress = False
            for action in selected:
                if expansions >= budget.max_expansions:
                    return _SearchResult(
                        False,
                        working_state,
                        (),
                        expansions,
                        "max_expansions",
                        policy_used,
                        policy_failed,
                        tuple(diagnostics),
                        inference_rounds + int(made_progress),
                    )
                before_atoms = working_state.eligible_atoms
                try:
                    candidate = self.execute_action(working_state, action)
                except KernelError as exc:
                    diagnostics.append(
                        f"agenda action invalidated after same-round facts: {exc}"
                    )
                    continue
                new_atoms = candidate.eligible_atoms - before_atoms
                if not new_atoms:
                    continue
                if len(candidate.facts) > budget.max_facts:
                    return _SearchResult(
                        False,
                        working_state,
                        (),
                        expansions,
                        "max_facts",
                        policy_used,
                        policy_failed,
                        tuple(diagnostics),
                        inference_rounds + int(made_progress),
                    )
                working_state = candidate
                expansions += 1
                made_progress = True
                for atom in sorted(new_atoms, key=Atom.canonical_key):
                    producers.setdefault(atom, action)

                if self._goals_met(working_state, goals):
                    inference_rounds += 1
                    try:
                        proof = self._build_supporting_proof(
                            initial_state, goals, producers
                        )
                    except KernelError as exc:
                        diagnostics.append(str(exc))
                        return _SearchResult(
                            False,
                            working_state,
                            (),
                            expansions,
                            "proof_construction_failed",
                            policy_used,
                            policy_failed,
                            tuple(diagnostics),
                            inference_rounds,
                        )
                    if len(proof) > budget.max_steps:
                        return _SearchResult(
                            False,
                            working_state,
                            (),
                            expansions,
                            "max_steps",
                            policy_used,
                            policy_failed,
                            tuple(diagnostics),
                            inference_rounds,
                        )
                    return _SearchResult(
                        True,
                        working_state,
                        proof,
                        expansions,
                        "goals_proven",
                        policy_used,
                        policy_failed,
                        tuple(diagnostics),
                        inference_rounds,
                    )

            if not made_progress:
                diagnostics.append("applicable agenda actions produced no eligible facts")
                return _SearchResult(
                    False,
                    state,
                    (),
                    expansions,
                    "no_solution",
                    policy_used,
                    policy_failed,
                    tuple(diagnostics),
                    inference_rounds,
                )
            state = working_state
            inference_rounds += 1

    def _relevant_operator_names(
        self, goals: Sequence[Goal]
    ) -> frozenset[str]:
        needed_predicates = {goal.atom.predicate.name for goal in goals}
        relevant: set[str] = set()
        changed = True
        while changed:
            changed = False
            for operator in sorted(
                self.registry.operators.values(), key=lambda item: item.name
            ):
                if operator.name in relevant:
                    continue
                if not any(
                    effect.predicate.name in needed_predicates
                    for effect in operator.effects
                ):
                    continue
                relevant.add(operator.name)
                before = len(needed_predicates)
                needed_predicates.update(
                    premise.predicate.name for premise in operator.preconditions
                )
                changed = changed or len(needed_predicates) != before
        return frozenset(relevant)

    def _backward_requirements(
        self,
        goals: Sequence[Goal],
        *,
        max_depth: int,
        max_requirements: int,
    ) -> tuple[tuple[str, tuple[Term | None, ...]], ...]:
        """Propagate grounded argument constraints backward through operator schemas."""

        requirements: dict[tuple, tuple[str, tuple[Term | None, ...]]] = {}
        queue: deque[
            tuple[tuple[str, tuple[Term | None, ...]], int]
        ] = deque()
        for goal in goals:
            requirement = self._requirement_from_atom(goal.atom, {})
            key = _requirement_key(requirement)
            if key not in requirements:
                requirements[key] = requirement
                queue.append((requirement, 0))

        requirement_limit = max(len(requirements), max_requirements)

        processed: set[tuple[str, int, tuple]] = set()
        operators = tuple(
            sorted(self.registry.operators.values(), key=lambda item: item.name)
        )
        while queue:
            requirement, depth = queue.popleft()
            if depth >= max_depth:
                continue
            requirement_key = _requirement_key(requirement)
            for operator in operators:
                for effect_index, effect in enumerate(operator.effects):
                    bindings = self._match_requirement(effect, requirement)
                    if bindings is None:
                        continue
                    marker = (operator.name, effect_index, requirement_key)
                    if marker in processed:
                        continue
                    processed.add(marker)
                    for precondition in operator.preconditions:
                        needed = self._requirement_from_atom(precondition, bindings)
                        key = _requirement_key(needed)
                        if key in requirements:
                            continue
                        if len(requirements) >= requirement_limit:
                            return tuple(
                                requirements[key]
                                for key in sorted(requirements, key=str)
                            )
                        requirements[key] = needed
                        queue.append((needed, depth + 1))
        return tuple(requirements[key] for key in sorted(requirements, key=str))

    def _policy_frontier_goals(
        self,
        goals: Sequence[Goal],
        actions: Sequence[GroundAction],
        requirements: Sequence[tuple[str, tuple[Term | None, ...]]],
    ) -> tuple[Goal, ...]:
        """Expose executable, dependency-matching effects as controller subgoals."""

        original_atoms = {goal.atom for goal in goals}
        frontier: dict[tuple, Goal] = {}
        for action in actions:
            for effect in action.effects:
                if effect in original_atoms:
                    continue
                if not any(
                    self._match_requirement(effect, requirement) is not None
                    for requirement in requirements
                ):
                    continue
                frontier[effect.canonical_key()] = Goal(
                    effect,
                    label="operator_frontier",
                )
        return tuple(goals) + tuple(frontier[key] for key in sorted(frontier))

    def _requirement_from_atom(
        self,
        atom: Atom,
        bindings: Mapping[str, Term],
    ) -> tuple[str, tuple[Term | None, ...]]:
        return (
            atom.predicate.name,
            tuple(
                self._resolve_requirement_term(argument, bindings)
                for argument in atom.arguments
            ),
        )

    def _resolve_requirement_term(
        self,
        term: Term,
        bindings: Mapping[str, Term],
    ) -> Term | None:
        from .model import TermApplication, Variable

        if isinstance(term, Variable):
            return bindings.get(term.name)
        if not isinstance(term, TermApplication):
            return term
        arguments = tuple(
            self._resolve_requirement_term(argument, bindings)
            for argument in term.arguments
        )
        if any(argument is None for argument in arguments):
            return None
        return self.registry.apply(
            term.function,
            *(argument for argument in arguments if argument is not None),
        )

    def _match_requirement(
        self,
        effect: Atom,
        requirement: tuple[str, tuple[Term | None, ...]],
    ) -> dict[str, Term] | None:
        predicate, required_arguments = requirement
        if effect.predicate.name != predicate:
            return None
        bindings: dict[str, Term] = {}
        for pattern, required in zip(
            effect.arguments,
            required_arguments,
            strict=True,
        ):
            if not self._match_requirement_term(pattern, required, bindings):
                return None
        return bindings

    def _match_requirement_term(
        self,
        pattern: Term,
        required: Term | None,
        bindings: dict[str, Term],
    ) -> bool:
        from .model import TermApplication, Variable

        if required is None:
            return True
        if isinstance(pattern, Variable):
            if not self.registry.types.is_assignable(required.type, pattern.type):
                return False
            existing = bindings.get(pattern.name)
            if existing is not None:
                return existing == required
            bindings[pattern.name] = required
            return True
        if isinstance(pattern, TermApplication):
            if not isinstance(required, TermApplication):
                return False
            if pattern.function.name != required.function.name:
                return False
            return all(
                self._match_requirement_term(left, right, bindings)
                for left, right in zip(
                    pattern.arguments,
                    required.arguments,
                    strict=True,
                )
            )
        return pattern == required

    def _build_supporting_proof(
        self,
        initial_state: WorldState,
        goals: Sequence[Goal],
        producers: Mapping[Atom, GroundAction],
    ) -> tuple[ProofStep, ...]:
        ordered_actions: list[GroundAction] = []
        resolved_actions: set[tuple] = set()
        visiting_atoms: set[Atom] = set()

        def require(atom: Atom) -> None:
            if initial_state.contains(atom):
                return
            action = producers.get(atom)
            if action is None:
                raise KernelError(f"proof producer is missing for {atom}")
            action_key = action.canonical_key()
            if action_key in resolved_actions:
                return
            if atom in visiting_atoms:
                raise KernelError(f"cyclic proof producer detected for {atom}")
            visiting_atoms.add(atom)
            for premise in action.preconditions:
                require(premise)
            visiting_atoms.remove(atom)
            resolved_actions.add(action_key)
            ordered_actions.append(action)

        for goal in goals:
            require(goal.atom)

        proof_state = initial_state
        proof: list[ProofStep] = []
        for index, action in enumerate(ordered_actions, start=1):
            before_digest = proof_state.digest()
            try:
                next_state = self.execute_action(proof_state, action)
            except KernelError as exc:
                raise KernelError(
                    f"supporting proof cannot replay {action.operator.name}: {exc}"
                ) from exc
            proof.append(
                ProofStep(
                    index=index,
                    action=action,
                    premises=action.preconditions,
                    effects=action.effects,
                    before_digest=before_digest,
                    after_digest=next_state.digest(),
                )
            )
            proof_state = next_state
        return tuple(proof)

    def _match_preconditions(
        self,
        operator: OperatorSpec,
        facts_by_predicate: Mapping[str, Sequence[Atom]],
    ) -> tuple[dict[str, Term], ...]:
        results: list[dict[str, Term]] = []

        def backtrack(index: int, substitution: dict[str, Term]) -> None:
            if index == len(operator.preconditions):
                results.append(substitution)
                return
            pattern = operator.preconditions[index]
            for fact in facts_by_predicate.get(pattern.predicate.name, ()):
                for unified in unify_atom(
                    pattern, fact, self.registry.types, substitution
                ):
                    backtrack(index + 1, unified)

        backtrack(0, {})
        unique: dict[tuple, dict[str, Term]] = {}
        for result in results:
            key = tuple(
                sorted(
                    (name, term.canonical_key()) for name, term in result.items()
                )
            )
            unique[key] = result
        return tuple(unique[key] for key in sorted(unique))

    def _complete_bindings(
        self,
        operator: OperatorSpec,
        substitutions: Sequence[dict[str, Term]],
        state: WorldState,
        candidate_limit: int,
    ) -> tuple[dict[str, Term], ...]:
        results: list[dict[str, Term]] = []
        terms = state.terms()
        for substitution in substitutions:
            missing = [
                parameter
                for parameter in operator.parameters
                if parameter.name not in substitution
            ]
            if not missing:
                results.append(dict(substitution))
                continue
            candidate_sets: list[tuple[Term, ...]] = []
            for parameter in missing:
                compatible = tuple(
                    term
                    for term in terms
                    if self.registry.types.is_assignable(term.type, parameter.type)
                )[:candidate_limit]
                if not compatible:
                    candidate_sets = []
                    break
                candidate_sets.append(compatible)
            for choices in product(*candidate_sets) if candidate_sets else ():
                completed = dict(substitution)
                completed.update(
                    (parameter.name, term)
                    for parameter, term in zip(missing, choices, strict=True)
                )
                results.append(completed)
        return tuple(results)

    @staticmethod
    def _normalize_policy_decision(
        decision: PolicyDecision | Sequence[float], action_count: int
    ) -> PolicyDecision:
        if isinstance(decision, PolicyDecision):
            normalized = decision
        else:
            normalized = PolicyDecision(tuple(float(value) for value in decision))
        if len(normalized.action_scores) != action_count:
            raise KernelError(
                f"policy returned {len(normalized.action_scores)} scores for "
                f"{action_count} actions"
            )
        if normalized.action_limit is not None and (
            isinstance(normalized.action_limit, bool)
            or not isinstance(normalized.action_limit, int)
            or normalized.action_limit <= 0
        ):
            raise KernelError("policy action_limit must be a positive integer or None")
        return normalized

    @staticmethod
    def _goals_met(state: WorldState, goals: Sequence[Goal]) -> bool:
        return all(state.contains(goal.atom) for goal in goals)

    @staticmethod
    def _apply_action(state: WorldState, action: GroundAction) -> WorldState:
        existing = state.eligible_atoms
        additions = tuple(
            Fact(
                atom=effect,
                status=FactStatus.DERIVED,
                source=action.operator.name,
                assertion_status=AssertionStatus.GENERATED,
                evidence_status=EvidenceStatus.DERIVED,
            )
            for effect in action.effects
            if effect not in existing
        )
        return WorldState(
            facts=state.facts + additions,
            depth=state.depth + 1,
            path_cost=state.path_cost + action.cost,
        )

    @staticmethod
    def _goal_outcomes(
        goals: Sequence[Goal], state: WorldState, proof: Sequence[ProofStep]
    ) -> tuple[GoalOutcome, ...]:
        outcomes: list[GoalOutcome] = []
        for goal in goals:
            step_index = next(
                (
                    step.index
                    for step in proof
                    if goal.atom in step.effects
                ),
                None,
            )
            outcomes.append(
                GoalOutcome(
                    goal=goal,
                    proven=state.contains(goal.atom),
                    proof_step=step_index,
                )
            )
        return tuple(outcomes)

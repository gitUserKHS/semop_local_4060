from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from typing import Mapping, Sequence

from .model import Atom, KernelError, OperatorSpec, Rule, Symbol, Term, Variable, WorldState
from .registry import KernelRegistry
from .unification import substitute_atom


@dataclass(frozen=True)
class TransitionOperator:
    name: str
    parameters: tuple[Variable, ...]
    preconditions: tuple[Atom, ...]
    add_effects: tuple[Atom, ...]
    delete_effects: tuple[Atom, ...] = ()
    duration: int = 1
    cost: float = 1.0
    description_ko: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("transition operator name cannot be empty")
        if self.duration <= 0 or self.cost <= 0:
            raise ValueError("transition duration and cost must be positive")
        if not self.add_effects and not self.delete_effects:
            raise ValueError("transition operator requires an add or delete effect")
        overlap = set(self.add_effects).intersection(self.delete_effects)
        if overlap:
            raise ValueError("transition cannot add and delete the same atom")


@dataclass(frozen=True)
class TransitionAction:
    operator: TransitionOperator
    bindings: tuple[tuple[str, Term], ...]

    @property
    def binding_map(self) -> dict[str, Term]:
        return dict(self.bindings)


@dataclass(frozen=True)
class TemporalFact:
    atom: Atom
    valid_from: int
    valid_until: int | None = None
    source_action: str = "initial"


@dataclass(frozen=True)
class TransitionWorldState:
    active: frozenset[Atom]
    time: int = 0
    history: tuple[TemporalFact, ...] = ()
    closed_world: bool = True

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError("transition time cannot be negative")
        object.__setattr__(self, "active", frozenset(self.active))
        if not self.history:
            object.__setattr__(
                self,
                "history",
                tuple(
                    TemporalFact(atom, 0)
                    for atom in sorted(self.active, key=lambda item: item.canonical_key())
                ),
            )

    def digest(self) -> str:
        payload = {
            "time": self.time,
            "closed_world": self.closed_world,
            "active": [
                atom.canonical_key()
                for atom in sorted(self.active, key=lambda item: item.canonical_key())
            ],
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def satisfies(self, goals: Sequence[Atom]) -> bool:
        return all(goal in self.active for goal in goals)


@dataclass(frozen=True)
class TransitionProofStep:
    index: int
    action: TransitionAction
    preconditions: tuple[Atom, ...]
    added: tuple[Atom, ...]
    deleted: tuple[Atom, ...]
    before_digest: str
    after_digest: str
    time_before: int
    time_after: int


@dataclass(frozen=True)
class TransitionPlanResult:
    success: bool
    verified: bool
    initial_state: TransitionWorldState
    final_state: TransitionWorldState
    goals: tuple[Atom, ...]
    proof: tuple[TransitionProofStep, ...]
    expansions: int
    halt_reason: str


class TransitionRegistry:
    """Separate non-monotonic library built on the kernel's typed vocabulary."""

    def __init__(self, kernel: KernelRegistry) -> None:
        self.kernel = kernel
        self.operators: dict[str, TransitionOperator] = {}

    def register(self, operator: TransitionOperator) -> TransitionOperator:
        if operator.name in self.operators or operator.name in self.kernel.operators:
            raise KernelError(f"operator already registered: {operator.name}")
        validation_rule = Rule(
            name=operator.name,
            parameters=operator.parameters,
            preconditions=operator.preconditions,
            effects=operator.add_effects + operator.delete_effects,
            cost=operator.cost,
            description_ko=operator.description_ko,
        )
        self.kernel._validate_operator(OperatorSpec(validation_rule))
        self.operators[operator.name] = operator
        return operator

    def ground_actions(
        self,
        state: TransitionWorldState,
        goals: Sequence[Atom] = (),
    ) -> tuple[TransitionAction, ...]:
        terms = _candidate_symbols((*state.active, *goals))
        actions: list[TransitionAction] = []
        for operator in sorted(self.operators.values(), key=lambda item: item.name):
            choices = [
                tuple(
                    term
                    for term in terms
                    if self.kernel.types.is_assignable(term.type, parameter.type)
                )
                for parameter in operator.parameters
            ]
            if any(not items for items in choices):
                continue
            combinations = product(*choices) if choices else ((),)
            for combination in combinations:
                bindings = tuple(
                    (parameter.name, term)
                    for parameter, term in zip(
                        operator.parameters,
                        combination,
                        strict=True,
                    )
                )
                action = TransitionAction(operator, bindings)
                grounded = _ground_atoms(
                    operator.preconditions,
                    action.binding_map,
                    self.kernel,
                )
                if all(atom in state.active for atom in grounded):
                    actions.append(action)
        return tuple(actions)


class TransitionExecutor:
    def __init__(self, registry: TransitionRegistry) -> None:
        self.registry = registry

    def apply(
        self,
        state: TransitionWorldState,
        action: TransitionAction,
        *,
        index: int = 1,
    ) -> tuple[TransitionWorldState, TransitionProofStep]:
        registered = self.registry.operators.get(action.operator.name)
        if registered != action.operator:
            raise KernelError("transition action references an unregistered operator")
        bindings = action.binding_map
        expected = {parameter.name for parameter in action.operator.parameters}
        if set(bindings) != expected:
            raise KernelError("transition action bindings do not match parameters")
        for parameter in action.operator.parameters:
            term = bindings[parameter.name]
            if not self.registry.kernel.types.is_assignable(term.type, parameter.type):
                raise KernelError(f"transition binding ?{parameter.name} has wrong type")
        preconditions = _ground_atoms(
            action.operator.preconditions,
            bindings,
            self.registry.kernel,
        )
        if not all(atom in state.active for atom in preconditions):
            raise KernelError("transition preconditions are not satisfied")
        added = _ground_atoms(
            action.operator.add_effects,
            bindings,
            self.registry.kernel,
        )
        deleted = _ground_atoms(
            action.operator.delete_effects,
            bindings,
            self.registry.kernel,
        )
        active = set(state.active)
        for atom in deleted:
            active.discard(atom)
        active.update(added)
        next_time = state.time + action.operator.duration
        history = list(state.history)
        deleted_set = set(deleted)
        for position, fact in enumerate(history):
            if fact.atom in deleted_set and fact.valid_until is None:
                history[position] = TemporalFact(
                    fact.atom,
                    fact.valid_from,
                    next_time,
                    fact.source_action,
                )
        history.extend(
            TemporalFact(atom, next_time, None, action.operator.name)
            for atom in added
            if atom not in state.active
        )
        next_state = TransitionWorldState(
            frozenset(active),
            time=next_time,
            history=tuple(history),
            closed_world=state.closed_world,
        )
        step = TransitionProofStep(
            index=index,
            action=action,
            preconditions=preconditions,
            added=added,
            deleted=deleted,
            before_digest=state.digest(),
            after_digest=next_state.digest(),
            time_before=state.time,
            time_after=next_time,
        )
        return next_state, step

    def replay(
        self,
        initial: TransitionWorldState,
        proof: Sequence[TransitionProofStep],
        goals: Sequence[Atom],
    ) -> tuple[bool, TransitionWorldState]:
        state = initial
        for expected_index, recorded in enumerate(proof, start=1):
            if recorded.index != expected_index or recorded.before_digest != state.digest():
                return False, state
            try:
                next_state, replayed = self.apply(
                    state,
                    recorded.action,
                    index=expected_index,
                )
            except KernelError:
                return False, state
            if replayed != recorded:
                return False, state
            state = next_state
        return state.satisfies(goals), state


class TransitionPlanner:
    def __init__(self, registry: TransitionRegistry) -> None:
        self.registry = registry
        self.executor = TransitionExecutor(registry)

    def solve(
        self,
        initial: TransitionWorldState,
        goals: Sequence[Atom],
        *,
        max_steps: int = 32,
        max_expansions: int = 20_000,
    ) -> TransitionPlanResult:
        resolved_goals = tuple(goals)
        if not resolved_goals:
            raise ValueError("transition planning requires at least one goal")
        if max_steps <= 0 or max_expansions <= 0:
            raise ValueError("transition planning limits must be positive")
        if initial.satisfies(resolved_goals):
            return TransitionPlanResult(
                True,
                True,
                initial,
                initial,
                resolved_goals,
                (),
                0,
                "goals_already_satisfied",
            )
        frontier = deque([(initial, tuple())])
        visited = {initial.digest()}
        expansions = 0
        halt_reason = "search_exhausted"
        while frontier:
            state, proof = frontier.popleft()
            if len(proof) >= max_steps:
                halt_reason = "max_steps"
                continue
            for action in self.registry.ground_actions(state, resolved_goals):
                if expansions >= max_expansions:
                    return TransitionPlanResult(
                        False,
                        False,
                        initial,
                        state,
                        resolved_goals,
                        proof,
                        expansions,
                        "max_expansions",
                    )
                expansions += 1
                next_state, step = self.executor.apply(
                    state,
                    action,
                    index=len(proof) + 1,
                )
                next_proof = proof + (step,)
                if next_state.satisfies(resolved_goals):
                    verified, replayed = self.executor.replay(
                        initial,
                        next_proof,
                        resolved_goals,
                    )
                    return TransitionPlanResult(
                        verified,
                        verified,
                        initial,
                        replayed,
                        resolved_goals,
                        next_proof,
                        expansions,
                        "goals_proven" if verified else "replay_failed",
                    )
                digest = next_state.digest()
                if digest not in visited:
                    visited.add(digest)
                    frontier.append((next_state, next_proof))
        return TransitionPlanResult(
            False,
            False,
            initial,
            initial,
            resolved_goals,
            (),
            expansions,
            halt_reason,
        )


def _ground_atoms(
    atoms: Sequence[Atom],
    bindings: Mapping[str, Term],
    registry: KernelRegistry,
) -> tuple[Atom, ...]:
    return tuple(substitute_atom(atom, bindings, registry) for atom in atoms)


def _candidate_symbols(atoms: Sequence[Atom]) -> tuple[Symbol, ...]:
    found: dict[tuple[object, ...], Symbol] = {}

    def visit(term: Term) -> None:
        if isinstance(term, Symbol):
            found[term.canonical_key()] = term
            return
        arguments = getattr(term, "arguments", ())
        for argument in arguments:
            visit(argument)

    for atom in atoms:
        for argument in atom.arguments:
            visit(argument)
    return tuple(found[key] for key in sorted(found))

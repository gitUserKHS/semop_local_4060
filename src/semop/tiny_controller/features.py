from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import blake2b
from typing import Sequence

from semop.kernel import Goal, GroundAction, Symbol, Term, TermApplication, WorldState


ACTION_STRUCTURAL_FEATURE_COUNT = 8


@dataclass(frozen=True)
class CanonicalRelation:
    name: str
    arguments: tuple[int, ...]
    role: str


@dataclass(frozen=True)
class CanonicalAction:
    operator: str
    operator_features: tuple[str, ...]
    argument_nodes: tuple[int, ...]
    argument_types: tuple[str, ...]
    structural_values: tuple[float, ...]


@dataclass(frozen=True)
class CanonicalProblemGraph:
    node_tokens: tuple[tuple[str, ...], ...]
    relations: tuple[CanonicalRelation, ...]
    goals: tuple[CanonicalRelation, ...]
    actions: tuple[CanonicalAction, ...]
    term_to_node: dict[Term, int]


def stable_bucket(value: str, buckets: int) -> int:
    digest = blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def canonicalize_problem(
    state: WorldState,
    goals: Sequence[Goal],
    actions: Sequence[GroundAction] = (),
) -> CanonicalProblemGraph:
    """Anonymize symbol names while preserving type and relational roles."""

    all_atoms = [fact.atom for fact in state.facts] + [goal.atom for goal in goals]
    symbols: set[Symbol] = set()

    def collect(term: Term) -> None:
        if isinstance(term, Symbol):
            symbols.add(term)
        elif isinstance(term, TermApplication):
            for argument in term.arguments:
                collect(argument)

    for atom in all_atoms:
        for argument in atom.arguments:
            collect(argument)
    for action in actions:
        for _, term in action.bindings:
            collect(term)

    role_signatures: dict[Symbol, Counter[str]] = defaultdict(Counter)
    for fact in state.facts:
        for index, argument in enumerate(fact.atom.arguments):
            for symbol in _symbols_in(argument):
                role_signatures[symbol][
                    f"fact:{fact.atom.predicate.name}:{index}:{fact.status.value}"
                ] += 1
    for goal in goals:
        role = _goal_role(goal)
        for index, argument in enumerate(goal.atom.arguments):
            for symbol in _symbols_in(argument):
                role_signatures[symbol][
                    f"{role}:{goal.atom.predicate.name}:{index}"
                ] += 1

    ordered_symbols = sorted(
        symbols,
        key=lambda symbol: (
            symbol.type.name,
            tuple(sorted(role_signatures[symbol].items())),
            symbol.canonical_key(),
        ),
    )
    term_to_node: dict[Term, int] = {}
    node_tokens: list[tuple[str, ...]] = []
    type_counts: Counter[str] = Counter()
    for symbol in ordered_symbols:
        anonymous_index = type_counts[symbol.type.name]
        type_counts[symbol.type.name] += 1
        term_to_node[symbol] = len(node_tokens)
        role_tokens = tuple(
            f"role:{name}:{count}"
            for name, count in sorted(role_signatures[symbol].items())
        )
        node_tokens.append(
            (f"type:{symbol.type.name}", f"anon:{symbol.type.name}:{anonymous_index}")
            + role_tokens
        )

    applications: set[TermApplication] = set()

    def collect_applications(term: Term) -> None:
        if isinstance(term, TermApplication):
            for child in term.arguments:
                collect_applications(child)
            applications.add(term)

    for atom in all_atoms:
        for argument in atom.arguments:
            collect_applications(argument)
    for action in actions:
        for _, term in action.bindings:
            collect_applications(term)
    pending = set(applications)
    while pending:
        ready = [
            term
            for term in pending
            if all(argument in term_to_node for argument in term.arguments)
        ]
        if not ready:
            raise ValueError("cyclic or incomplete term graph")
        ready.sort(
            key=lambda term: (
                term.function.name,
                tuple(term_to_node[arg] for arg in term.arguments),
                term.type.name,
            )
        )
        for term in ready:
            term_to_node[term] = len(node_tokens)
            node_tokens.append(
                (
                    f"type:{term.type.name}",
                    f"function:{term.function.name}",
                    *(
                        f"arg:{index}:node:{term_to_node[argument]}"
                        for index, argument in enumerate(term.arguments)
                    ),
                )
            )
            pending.remove(term)

    relations = tuple(
        sorted(
            (
                CanonicalRelation(
                    fact.atom.predicate.name,
                    tuple(term_to_node[term] for term in fact.atom.arguments),
                    fact.status.value,
                )
                for fact in state.facts
            ),
            key=lambda relation: (relation.name, relation.arguments, relation.role),
        )
    )
    goal_relations = tuple(
        sorted(
            (
                CanonicalRelation(
                    goal.atom.predicate.name,
                    tuple(term_to_node[term] for term in goal.atom.arguments),
                    _goal_role(goal),
                )
                for goal in goals
            ),
            key=lambda relation: (relation.name, relation.arguments),
        )
    )
    canonical_actions = tuple(
        CanonicalAction(
            operator=action.operator.name,
            operator_features=_operator_features(action, state, goals),
            argument_nodes=tuple(term_to_node[term] for _, term in action.bindings),
            argument_types=tuple(term.type.name for _, term in action.bindings),
            structural_values=_action_structural_values(action, state, goals),
        )
        for action in actions
    )
    return CanonicalProblemGraph(
        node_tokens=tuple(node_tokens),
        relations=relations,
        goals=goal_relations,
        actions=canonical_actions,
        term_to_node=term_to_node,
    )


def _symbols_in(term: Term) -> tuple[Symbol, ...]:
    if isinstance(term, Symbol):
        return (term,)
    if isinstance(term, TermApplication):
        return tuple(symbol for child in term.arguments for symbol in _symbols_in(child))
    return ()


def _goal_role(goal: Goal) -> str:
    return "frontier" if goal.label == "operator_frontier" else "goal"


def _operator_features(
    action: GroundAction,
    state: WorldState,
    goals: Sequence[Goal],
) -> tuple[str, ...]:
    features = {
        f"schema:{action.operator.name}",
        f"family:{action.operator.family}",
        f"cost:{action.operator.cost:.2f}",
        f"structure:binding_count:{min(len(action.bindings), 8)}",
        f"structure:precondition_count:{min(len(action.preconditions), 8)}",
        f"structure:effect_count:{min(len(action.effects), 8)}",
    }
    features.update(
        f"tag:{tag}"
        for tag in action.operator.tags
        if tag not in {"hard_negative", "synthetic"}
    )
    features.update(
        "precondition:"
        + atom.predicate.name
        + ":"
        + ",".join(term.type.name for term in atom.arguments)
        for atom in action.operator.preconditions
    )
    features.update(
        "effect:"
        + atom.predicate.name
        + ":"
        + ",".join(term.type.name for term in atom.arguments)
        for atom in action.operator.effects
    )
    verifier_atoms = tuple(
        goal.atom for goal in goals if _goal_role(goal) == "goal"
    )
    frontier_atoms = tuple(
        goal.atom for goal in goals if _goal_role(goal) == "frontier"
    )
    goal_atoms = verifier_atoms + frontier_atoms
    exact_match = any(effect in goal_atoms for effect in action.effects)
    predicate_match = any(
        effect.predicate.name == goal.predicate.name
        for effect in action.effects
        for goal in goal_atoms
    )
    type_match = any(
        tuple(term.type.name for term in effect.arguments)
        == tuple(term.type.name for term in goal.arguments)
        for effect in action.effects
        for goal in goal_atoms
    )
    max_argument_overlap = max(
        (
            len(set(effect.arguments) & set(goal.arguments))
            for effect in action.effects
            for goal in goal_atoms
        ),
        default=0,
    )
    novel_effects = sum(not state.contains(effect) for effect in action.effects)
    features.update(
        {
            f"structure:effect_goal_exact:{str(exact_match).lower()}",
            f"structure:effect_goal_predicate:{str(predicate_match).lower()}",
            f"structure:effect_goal_types:{str(type_match).lower()}",
            f"structure:effect_goal_argument_overlap:{min(max_argument_overlap, 8)}",
            f"structure:novel_effect_count:{min(novel_effects, 8)}",
            "structure:effect_verifier_exact:"
            + str(
                any(effect in verifier_atoms for effect in action.effects)
            ).lower(),
            "structure:effect_frontier_exact:"
            + str(
                any(effect in frontier_atoms for effect in action.effects)
            ).lower(),
        }
    )
    return tuple(sorted(features))


def _action_structural_values(
    action: GroundAction,
    state: WorldState,
    goals: Sequence[Goal],
) -> tuple[float, ...]:
    goal_atoms = tuple(goal.atom for goal in goals)
    exact_match = any(effect in goal_atoms for effect in action.effects)
    predicate_match = any(
        effect.predicate.name == goal.predicate.name
        for effect in action.effects
        for goal in goal_atoms
    )
    type_match = any(
        tuple(term.type.name for term in effect.arguments)
        == tuple(term.type.name for term in goal.arguments)
        for effect in action.effects
        for goal in goal_atoms
    )
    max_overlap = max(
        (
            len(set(effect.arguments) & set(goal.arguments))
            for effect in action.effects
            for goal in goal_atoms
        ),
        default=0,
    )
    max_goal_arity = max((len(goal.arguments) for goal in goal_atoms), default=1)
    novel_effects = sum(not state.contains(effect) for effect in action.effects)
    values = (
        float(exact_match),
        float(predicate_match),
        float(type_match),
        max_overlap / max(1, max_goal_arity),
        novel_effects / max(1, len(action.effects)),
        min(len(action.bindings), 8) / 8.0,
        min(len(action.preconditions), 8) / 8.0,
        min(len(action.effects), 8) / 8.0,
    )
    if len(values) != ACTION_STRUCTURAL_FEATURE_COUNT:
        raise AssertionError("action structural feature contract changed")
    return values

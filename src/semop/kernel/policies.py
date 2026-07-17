from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

from .engine import PolicyDecision
from .model import Goal, GroundAction, Term, TermApplication, WorldState


class GoalDirectedPolicy:
    """Non-learned relation-aware policy used as a controller-interface baseline."""

    def score_actions(
        self,
        state: WorldState,
        goals: Sequence[Goal],
        actions: Sequence[GroundAction],
    ) -> PolicyDecision:
        verifier_goals = tuple(
            goal for goal in goals if goal.label != "operator_frontier"
        )
        frontier_goals = tuple(
            goal for goal in goals if goal.label == "operator_frontier"
        )
        goal_atoms = {goal.atom for goal in verifier_goals}
        frontier_atoms = {goal.atom for goal in frontier_goals}
        goal_predicates = {goal.atom.predicate.name for goal in verifier_goals}
        frontier_predicates = {
            goal.atom.predicate.name for goal in frontier_goals
        }
        goal_terms = {
            term
            for goal in goals
            for argument in goal.atom.arguments
            for term in _flatten_terms(argument)
        }
        distances = _adjacency_distances(state, goal_terms)
        scores: list[float] = []
        for action in actions:
            score = -0.01 * action.cost
            for effect in action.effects:
                if effect in goal_atoms:
                    score += 1_000.0
                elif effect in frontier_atoms:
                    score += 100.0
                if effect.predicate.name in goal_predicates:
                    score += 100.0
                elif effect.predicate.name in frontier_predicates:
                    score += 10.0
                effect_terms = {
                    term
                    for argument in effect.arguments
                    for term in _flatten_terms(argument)
                }
                score += 2.0 * len(effect_terms.intersection(goal_terms))
                finite_distances = [
                    distances[term] for term in effect_terms if term in distances
                ]
                if finite_distances:
                    score += 5.0 / (1.0 + min(finite_distances))
            scores.append(score)
        ordered = sorted(scores, reverse=True)
        top_score = ordered[0] if ordered else 0.0
        competitive_actions = max(
            1,
            sum(top_score - score < 5.0 for score in ordered),
        )
        return PolicyDecision(
            tuple(scores),
            halt_probability=0.0,
            state_value=0.0,
            action_limit=competitive_actions,
        )


def _flatten_terms(term: Term) -> tuple[Term, ...]:
    if isinstance(term, TermApplication):
        return (term,) + tuple(
            nested
            for argument in term.arguments
            for nested in _flatten_terms(argument)
        )
    return (term,)


def _adjacency_distances(
    state: WorldState, targets: set[Term]
) -> dict[Term, int]:
    adjacency: dict[Term, set[Term]] = defaultdict(set)
    for fact in state.eligible_facts:
        if fact.atom.predicate.name != "ADJACENT" or len(fact.atom.arguments) != 2:
            continue
        left, right = fact.atom.arguments
        adjacency[left].add(right)
        adjacency[right].add(left)
    distances: dict[Term, int] = {}
    queue: deque[Term] = deque()
    for target in sorted(targets, key=lambda term: term.canonical_key()):
        if target in adjacency:
            distances[target] = 0
            queue.append(target)
    while queue:
        current = queue.popleft()
        for neighbor in sorted(
            adjacency[current], key=lambda term: term.canonical_key()
        ):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    return distances

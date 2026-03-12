from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List

from .commonsense_kb import concept_family
from .structures import AnalogicalMatch, StructuredMeaningGraph


@dataclass
class AnalogicalMemoryBuilder:
    max_matches: int = 3

    def build(self, graph: StructuredMeaningGraph, similar_graphs: List[StructuredMeaningGraph]) -> List[AnalogicalMatch]:
        scored: List[AnalogicalMatch] = []
        current_nodes = graph.node_ids()
        current_node_families = self._node_families(current_nodes)
        current_relations = {edge.relation for edge in graph.edges}
        current_goals = self._goal_patterns(graph.hidden_goals)
        current_required = set(graph.required_premises)
        current_missing = set(graph.missing_premises)
        current_families = {candidate.family for candidate in graph.induced_operators}

        for similar in similar_graphs:
            similar_nodes = similar.node_ids()
            similar_node_families = self._node_families(similar_nodes)
            similar_relations = {edge.relation for edge in similar.edges}
            similar_goals = self._goal_patterns(similar.hidden_goals)
            similar_required = set(similar.required_premises)
            similar_missing = set(similar.missing_premises)
            similar_families = {candidate.family for candidate in similar.induced_operators}

            node_overlap = max(
                self._overlap(current_nodes, similar_nodes),
                self._overlap(current_node_families, similar_node_families),
            )
            relation_overlap = self._overlap(current_relations, similar_relations)
            goal_overlap = self._overlap(current_goals, similar_goals)
            requirement_overlap = self._overlap(current_required, similar_required)
            missing_overlap = self._overlap(current_missing, similar_missing)
            family_overlap = self._overlap(current_families, similar_families)
            score = round(
                0.28 * goal_overlap
                + 0.22 * requirement_overlap
                + 0.16 * missing_overlap
                + 0.14 * relation_overlap
                + 0.12 * family_overlap
                + 0.08 * node_overlap,
                4,
            )
            if score <= 0.0:
                continue
            shared_bits = []
            if current_goals & similar_goals:
                shared_bits.append('goal_pattern')
            if current_required & similar_required:
                shared_bits.append('required_premise')
            if current_missing & similar_missing:
                shared_bits.append('missing_requirement')
            if current_relations & similar_relations:
                shared_bits.append('relation_pattern')
            if current_families & similar_families:
                shared_bits.append('operator_family')
            if current_node_families & similar_node_families:
                shared_bits.append('concept_family')
            scored.append(
                AnalogicalMatch(
                    query=similar.query,
                    score=score,
                    analogy_type=self._analogy_type(shared_bits),
                    shared_basis=shared_bits[:4],
                    shared_nodes=sorted((current_nodes & similar_nodes))[:4],
                    shared_goals=sorted((current_goals & similar_goals))[:3],
                    shared_requirements=sorted((current_required & similar_required))[:4],
                    shared_operator_families=sorted((current_families & similar_families))[:4],
                )
            )

        scored.sort(key=lambda item: (-item.score, item.query))
        selected: List[AnalogicalMatch] = []
        seen_queries: set[str] = set()
        for item in scored:
            if item.query in seen_queries:
                continue
            selected.append(item)
            seen_queries.add(item.query)
            if len(selected) >= self.max_matches:
                break
        return selected

    @staticmethod
    def _overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / float(len(left | right))

    @staticmethod
    def _node_families(nodes: set[str]) -> set[str]:
        families = {concept_family(node) for node in nodes}
        return {family for family in families if family and family != 'concept'}

    @classmethod
    def _goal_patterns(cls, goals: List[str]) -> set[str]:
        return {cls._goal_pattern(goal) for goal in goals if goal}

    @staticmethod
    def _goal_pattern(goal: str) -> str:
        if re.match(r'^retrieve_item_from_.+_goal$', goal):
            return 'retrieve_item_from_container_goal'
        if re.match(r'^store_.+_in_.+_goal$', goal):
            return 'store_item_in_container_goal'
        if re.match(r'^pass_through_.+_goal$', goal):
            return 'pass_through_barrier_goal'
        return goal

    @staticmethod
    def _analogy_type(shared_bits: List[str]) -> str:
        if 'goal_pattern' in shared_bits and 'required_premise' in shared_bits:
            return 'goal_premise_analogy'
        if 'missing_requirement' in shared_bits:
            return 'failure_analogy'
        if 'operator_family' in shared_bits:
            return 'operator_analogy'
        if 'relation_pattern' in shared_bits:
            return 'relation_analogy'
        return 'surface_analogy'

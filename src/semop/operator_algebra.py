from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .basis_operators import canonicalize_basis_signature
from .structures import FunctorHypothesis, OperatorCandidate, OperatorDecomposition, StructuredMeaningGraph


@dataclass
class OperatorAlgebraSummary:
    decompositions: List[OperatorDecomposition]
    functor_hypotheses: List[FunctorHypothesis]


class OperatorAlgebraLearner:
    VISUAL_DECOMPOSITIONS: Dict[str, List[str]] = {
        'CONTAINER_ACCESS_OPERATOR': ['CONTAINER_BODY_OPERATOR', 'ACCESS_PORT_OPERATOR'],
        'CONTROLLED_ACCESS_OPERATOR': ['ACCESS_PORT_OPERATOR', 'ACCESS_CONTROL_OPERATOR'],
        'MANIPULABLE_CONTAINER_OPERATOR': ['CONTAINER_BODY_OPERATOR', 'ATTACHED_GRASP_OPERATOR'],
        'CARRIABLE_CONTAINER_OPERATOR': ['CONTAINER_BODY_OPERATOR', 'ATTACHED_GRASP_OPERATOR'],
        'OPENING_CONTROL_OPERATOR': ['ACCESS_PORT_OPERATOR', 'ACCESS_CONTROL_OPERATOR'],
    }

    LANGUAGE_DECOMPOSITIONS: Dict[str, List[str]] = {
        'SERVICE_GOAL_OPERATOR': ['TYPICAL_FOR', 'REQUIRES'],
        'GOAL_PRESERVATION_OPERATOR': ['hidden_goal', 'REQUIRES', 'BLOCKED_BY'],
        'CONTAINMENT_GOAL_OPERATOR': ['TYPICAL_FOR', 'REQUIRES', 'CONTAINS'],
    }

    def enrich(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        summary = self.analyze(graph)
        graph.operator_decompositions = self._merge_decompositions(graph.operator_decompositions, summary.decompositions)
        graph.functor_hypotheses = self._merge_functors(graph.functor_hypotheses, summary.functor_hypotheses)
        if summary.decompositions:
            graph.audit_trace.append('operator algebra: induced operator decompositions')
        if summary.functor_hypotheses:
            graph.audit_trace.append('operator algebra: induced cross-category functor hypotheses')
        return graph

    def analyze(self, graph: StructuredMeaningGraph) -> OperatorAlgebraSummary:
        decompositions = self._induce_decompositions(graph)
        functors = self._induce_functors(graph, decompositions)
        return OperatorAlgebraSummary(decompositions=decompositions, functor_hypotheses=functors)

    def _induce_decompositions(self, graph: StructuredMeaningGraph) -> List[OperatorDecomposition]:
        decompositions: List[OperatorDecomposition] = []
        operator_names = {candidate.name for candidate in graph.induced_operators}
        relation_names = {edge.relation for edge in graph.edges}
        node_ids = graph.node_ids()

        if graph.hidden_goals and graph.required_premises:
            decompositions.append(
                OperatorDecomposition(
                    operator_name='GOAL_PRESERVATION_OPERATOR',
                    basis_operators=canonicalize_basis_signature(['HIDDEN_GOAL', 'REQUIRES'] + (['BLOCKED_BY'] if 'BLOCKED_BY' in relation_names else [])),
                    rationale='A hidden goal plus required premises and blockers acts like a higher-order goal-preservation operator.',
                    confidence=0.82,
                )
            )
        if self._has_service_goal_signal(graph, relation_names, node_ids):
            decompositions.append(
                OperatorDecomposition(
                    operator_name='SERVICE_GOAL_OPERATOR',
                    basis_operators=canonicalize_basis_signature(['TYPICAL_FOR', 'REQUIRES']),
                    rationale='Service-place reasoning decomposes into a typical-goal script plus enabling service preconditions.',
                    confidence=0.79,
                )
            )
        if 'bag' in node_ids and {'open_access', 'available_space'} <= set(graph.required_premises):
            decompositions.append(
                OperatorDecomposition(
                    operator_name='CONTAINMENT_GOAL_OPERATOR',
                    basis_operators=canonicalize_basis_signature(['TYPICAL_FOR', 'REQUIRES', 'CONTAINS']),
                    rationale='Containment reasoning decomposes into access, space, and interior-placement constraints.',
                    confidence=0.81,
                )
            )
        for name, basis in self.VISUAL_DECOMPOSITIONS.items():
            if name in operator_names:
                decompositions.append(
                    OperatorDecomposition(
                        operator_name=name,
                        basis_operators=canonicalize_basis_signature(basis),
                        rationale='A higher visual operator can be represented as a composition of simpler structural operators.',
                        confidence=0.76,
                    )
                )
        for name, basis in self.LANGUAGE_DECOMPOSITIONS.items():
            if any(item.operator_name == name for item in decompositions):
                continue
            if name == 'SERVICE_GOAL_OPERATOR' and 'TYPICAL_FOR' in relation_names and 'REQUIRES' in relation_names:
                decompositions.append(OperatorDecomposition(operator_name=name, basis_operators=canonicalize_basis_signature(basis), rationale='Language service reasoning exposes the same basis relations.', confidence=0.72))
        return self._merge_decompositions([], decompositions)

    @staticmethod
    def _has_service_goal_signal(
        graph: StructuredMeaningGraph,
        relation_names: set[str],
        node_ids: set[str],
    ) -> bool:
        hidden_goals = {str(item) for item in graph.hidden_goals}
        service_node_ids = {
            node.id
            for node in graph.nodes
            if (node.kind or '').lower() in {'service_place', 'service'}
        }
        if 'car_wash' in node_ids and hidden_goals:
            return True
        if not service_node_ids:
            return False
        if any(goal for goal in hidden_goals if any(token in goal for token in ('clean_car', 'booking', 'reservation', 'service'))):
            return True
        if 'vehicle_present' in graph.required_premises:
            return True
        if any(edge.relation == 'REQUIRES' and edge.target == 'vehicle_present' for edge in graph.edges):
            return True
        if any(edge.relation == 'TYPICAL_FOR' and edge.source in service_node_ids for edge in graph.edges):
            return True
        return 'TYPICAL_FOR' in relation_names and 'REQUIRES' in relation_names

    def _induce_functors(self, graph: StructuredMeaningGraph, decompositions: List[OperatorDecomposition]) -> List[FunctorHypothesis]:
        functors: List[FunctorHypothesis] = []
        operator_names = {candidate.name for candidate in graph.induced_operators}
        hidden_goal_present = bool(graph.hidden_goals)
        if hidden_goal_present or any(item.operator_name == 'GOAL_PRESERVATION_OPERATOR' for item in decompositions):
            functors.append(
                FunctorHypothesis(
                    name='ServiceGoalToConstraintFunctor',
                    source_category='service_script',
                    target_category='goal_preservation_logic',
                    object_map={
                        'service_place': 'hidden_goal',
                        'vehicle_present': 'required_premise',
                    },
                    morphism_map={
                        'TYPICAL_FOR': 'goal_projection',
                        'REQUIRES': 'constraint_binding',
                        'BLOCKED_BY': 'goal_risk',
                    },
                    confidence=0.83,
                )
            )
        if any(name in operator_names for name in self.VISUAL_DECOMPOSITIONS):
            functors.append(
                FunctorHypothesis(
                    name='VisualStructureToActionFunctor',
                    source_category='visual_structure',
                    target_category='action_logic',
                    object_map={
                        'container_body': 'container',
                        'access_port': 'open_access',
                        'control_part': 'access_control',
                        'grasp_part': 'manipulation_handle',
                    },
                    morphism_map={
                        'PART_OF': 'enables_action_on',
                        'STRUCTURAL_PART_OF': 'grounds_precondition',
                        'ACCESS_PORT_OPERATOR': 'open_access_requirement',
                        'ATTACHED_GRASP_OPERATOR': 'manipulation_affordance',
                    },
                    confidence=0.81,
                )
            )
        if any('geometry' in op.family.lower() for op in graph.induced_operators):
            functors.append(
                FunctorHypothesis(
                    name='GeometryToCpFrameFunctor',
                    source_category='visual_geometry',
                    target_category='computational_geometry',
                    object_map={
                        'point': 'point',
                        'segment': 'segment',
                        'polygon': 'polygon',
                    },
                    morphism_map={
                        'PARALLEL': 'parallel_constraint',
                        'PERPENDICULAR': 'orthogonality_constraint',
                        'EQUAL_LENGTH': 'length_invariant',
                    },
                    confidence=0.74,
                )
            )
        return self._merge_functors([], functors)

    @staticmethod
    def _merge_decompositions(existing: List[OperatorDecomposition], new_items: List[OperatorDecomposition]) -> List[OperatorDecomposition]:
        merged = list(existing)
        for item in new_items:
            if any(current.operator_name == item.operator_name for current in merged):
                continue
            merged.append(item)
        return merged

    @staticmethod
    def _merge_functors(existing: List[FunctorHypothesis], new_items: List[FunctorHypothesis]) -> List[FunctorHypothesis]:
        merged = list(existing)
        for item in new_items:
            if any(current.name == item.name for current in merged):
                continue
            merged.append(item)
        return merged


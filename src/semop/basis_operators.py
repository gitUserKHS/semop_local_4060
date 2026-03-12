from __future__ import annotations

from typing import Dict, Iterable, List

from .structures import StructuredMeaningGraph


BASIS_OPERATOR_AXES: Dict[str, str] = {
    "HIDDEN_GOAL": "goal",
    "TYPICAL_FOR": "goal",
    "REQUIRES": "constraint",
    "BLOCKED_BY": "constraint",
    "ALTERNATIVE": "constraint",
    "CONTAINS": "structure",
    "PART_OF": "structure",
    "STRUCTURAL_PART_OF": "structure",
    "ACCESS_PORT_OPERATOR": "structure",
    "ACCESS_CONTROL_OPERATOR": "structure",
    "ATTACHED_GRASP_OPERATOR": "structure",
    "CONTAINER_BODY_OPERATOR": "structure",
    "PARALLEL": "geometry",
    "PERPENDICULAR": "geometry",
    "EQUAL_LENGTH": "geometry",
    "SYMMETRIC_STRUCTURE": "geometry",
    "AXIS_ALIGNED_STRUCTURE": "geometry",
    "CLOSED_BOUNDARY_STRUCTURE": "topology",
    "RANGE_QUERY": "computation",
    "FEASIBILITY_CHECK": "computation",
    "STATE_TRANSITION": "computation",
    "CONNECTIVITY": "computation",
    "OPTIMIZE": "computation",
}

_ALIASES: Dict[str, str] = {
    "hidden_goal": "HIDDEN_GOAL",
    "goal": "HIDDEN_GOAL",
    "requires": "REQUIRES",
    "blocked_by": "BLOCKED_BY",
    "typical_for": "TYPICAL_FOR",
    "alternative": "ALTERNATIVE",
    "contains": "CONTAINS",
    "part_of": "PART_OF",
    "structural_part_of": "STRUCTURAL_PART_OF",
    "parallel": "PARALLEL",
    "perpendicular": "PERPENDICULAR",
    "equal_length": "EQUAL_LENGTH",
}

_COMPOSITE_HINTS: Dict[str, List[str]] = {
    "CONTAINER_ACCESS_OPERATOR": ["CONTAINER_BODY_OPERATOR", "ACCESS_PORT_OPERATOR"],
    "CONTROLLED_ACCESS_OPERATOR": ["ACCESS_PORT_OPERATOR", "ACCESS_CONTROL_OPERATOR"],
    "MANIPULABLE_CONTAINER_OPERATOR": ["CONTAINER_BODY_OPERATOR", "ATTACHED_GRASP_OPERATOR"],
    "CARRIABLE_CONTAINER_OPERATOR": ["CONTAINER_BODY_OPERATOR", "ATTACHED_GRASP_OPERATOR"],
    "OPENING_CONTROL_OPERATOR": ["ACCESS_PORT_OPERATOR", "ACCESS_CONTROL_OPERATOR"],
    "SERVICE_GOAL_OPERATOR": ["TYPICAL_FOR", "REQUIRES"],
    "GOAL_PRESERVATION_OPERATOR": ["HIDDEN_GOAL", "REQUIRES", "BLOCKED_BY"],
    "CONTAINMENT_GOAL_OPERATOR": ["TYPICAL_FOR", "REQUIRES", "CONTAINS"],
}

_AXIS_ORDER = {
    "goal": 0,
    "constraint": 1,
    "structure": 2,
    "geometry": 3,
    "topology": 4,
    "computation": 5,
    "unknown": 6,
}


def normalize_operator_symbol(value: str) -> str:
    if not value:
        return ""
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in _ALIASES:
        return _ALIASES[lowered]
    return stripped.upper()


def basis_operator_axis(symbol: str) -> str:
    return BASIS_OPERATOR_AXES.get(normalize_operator_symbol(symbol), "unknown")


def canonicalize_basis_signature(operators: Iterable[str]) -> List[str]:
    unique = {normalize_operator_symbol(item) for item in operators if item}
    return sorted(unique, key=lambda item: (_AXIS_ORDER.get(basis_operator_axis(item), 99), item))


def infer_basis_operators(graph: StructuredMeaningGraph) -> List[str]:
    hits: List[str] = []
    if graph.hidden_goals:
        hits.append("HIDDEN_GOAL")
    hits.extend(normalize_operator_symbol(edge.relation) for edge in graph.edges)
    for item in graph.premise_validations:
        if item.hidden_goal:
            hits.append("HIDDEN_GOAL")
        if item.premise:
            hits.append("REQUIRES")
        if item.status in {"unsupported", "contradicted"}:
            hits.append("BLOCKED_BY")
    for decomposition in graph.operator_decompositions:
        hits.extend(canonicalize_basis_signature(decomposition.basis_operators))
    for candidate in graph.induced_operators:
        hits.extend(_COMPOSITE_HINTS.get(candidate.name, []))
    return canonicalize_basis_signature(hits)

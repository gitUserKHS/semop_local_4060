from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Literal


Status = Literal["valid", "invalid", "warning"]


@dataclass
class Node:
    id: str
    label: str
    kind: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: List[str] = field(default_factory=list)


@dataclass
class Edge:
    source: str
    relation: str
    target: str
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: List[str] = field(default_factory=list)


@dataclass
class PlanStep:
    id: str
    action: str
    rationale: str
    requires: List[str] = field(default_factory=list)
    status: Status = "valid"


@dataclass
class OperatorCandidate:
    name: str
    family: str
    arity: int
    input_types: List[str]
    output_type: str
    description: str
    examples: List[str] = field(default_factory=list)
    confidence: float = 0.5
    provenance: List[str] = field(default_factory=list)
    abstract_parents: List[str] = field(default_factory=list)


@dataclass
class SymbolicResult:
    domain: str
    answer: str
    evidence: List[str] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)
    confidence: float = 0.6
    source: str = ""


@dataclass
class GoalPreservationCheck:
    action: str
    hidden_goal: str
    status: str
    rationale: str
    confidence: float = 0.6


@dataclass
class OperatorDecomposition:
    operator_name: str
    basis_operators: List[str] = field(default_factory=list)
    rationale: str = ""
    confidence: float = 0.6


@dataclass
class FunctorHypothesis:
    name: str
    source_category: str
    target_category: str
    object_map: Dict[str, str] = field(default_factory=dict)
    morphism_map: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.6


@dataclass
class StructuredMeaningGraph:
    query: str
    intent: str
    domain: str = "general"
    source_context: str = ""
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    inferred_scripts: List[str] = field(default_factory=list)
    semantic_operators: List[str] = field(default_factory=list)
    induced_operators: List[OperatorCandidate] = field(default_factory=list)
    grammar_hypotheses: List[str] = field(default_factory=list)
    symbolic_results: List[SymbolicResult] = field(default_factory=list)
    hidden_goals: List[str] = field(default_factory=list)
    hidden_assumptions: List[str] = field(default_factory=list)
    required_premises: List[str] = field(default_factory=list)
    optional_interpretations: List[str] = field(default_factory=list)
    goal_preservation_checks: List[GoalPreservationCheck] = field(default_factory=list)
    operator_decompositions: List[OperatorDecomposition] = field(default_factory=list)
    functor_hypotheses: List[FunctorHypothesis] = field(default_factory=list)
    clarification_needed: bool = False
    plan: List[PlanStep] = field(default_factory=list)
    candidate_actions: List[str] = field(default_factory=list)
    creative_alternatives: List[str] = field(default_factory=list)
    invalid_advice: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    audit_trace: List[str] = field(default_factory=list)

    def node_ids(self) -> set[str]:
        return {node.id for node in self.nodes}

    def relation_tuples(self) -> set[tuple[str, str, str]]:
        return {(edge.source, edge.relation, edge.target) for edge in self.edges}

    def add_node(self, node: Node) -> None:
        for existing in self.nodes:
            if existing.id == node.id:
                existing.provenance = _merge_unique(existing.provenance, node.provenance)
                if not existing.label and node.label:
                    existing.label = node.label
                if existing.kind == "concept" and node.kind != "concept":
                    existing.kind = node.kind
                existing.attributes.update(node.attributes)
                return
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        for existing in self.edges:
            if (existing.source, existing.relation, existing.target) == (edge.source, edge.relation, edge.target):
                existing.provenance = _merge_unique(existing.provenance, edge.provenance)
                existing.confidence = max(existing.confidence, edge.confidence)
                existing.attributes.update(edge.attributes)
                return
        self.edges.append(edge)

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)

    def model_dump_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=ensure_ascii)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StructuredMeaningGraph:
        graph = cls(
            query=data["query"],
            intent=data["intent"],
            domain=data.get("domain", "general"),
            source_context=data.get("source_context", ""),
        )
        graph.nodes = [Node(**item) for item in data.get("nodes", [])]
        graph.edges = [Edge(**item) for item in data.get("edges", [])]
        graph.inferred_scripts = list(data.get("inferred_scripts", []))
        graph.semantic_operators = list(data.get("semantic_operators", []))
        graph.induced_operators = [OperatorCandidate(**item) for item in data.get("induced_operators", [])]
        graph.grammar_hypotheses = list(data.get("grammar_hypotheses", []))
        graph.symbolic_results = [SymbolicResult(**item) for item in data.get("symbolic_results", [])]
        graph.hidden_goals = list(data.get("hidden_goals", []))
        graph.hidden_assumptions = list(data.get("hidden_assumptions", []))
        graph.required_premises = list(data.get("required_premises", []))
        graph.optional_interpretations = list(data.get("optional_interpretations", []))
        graph.goal_preservation_checks = [GoalPreservationCheck(**item) for item in data.get("goal_preservation_checks", [])]
        graph.operator_decompositions = [OperatorDecomposition(**item) for item in data.get("operator_decompositions", [])]
        graph.functor_hypotheses = [FunctorHypothesis(**item) for item in data.get("functor_hypotheses", [])]
        graph.clarification_needed = bool(data.get("clarification_needed", False))
        graph.plan = [PlanStep(**item) for item in data.get("plan", [])]
        graph.candidate_actions = list(data.get("candidate_actions", []))
        graph.creative_alternatives = list(data.get("creative_alternatives", []))
        graph.invalid_advice = list(data.get("invalid_advice", []))
        graph.warnings = list(data.get("warnings", []))
        graph.audit_trace = list(data.get("audit_trace", []))
        return graph


@dataclass
class ExtractedMeaning:
    intent: str
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    scripts: List[str] = field(default_factory=list)
    candidate_actions: List[str] = field(default_factory=list)
    missing_knowledge: List[str] = field(default_factory=list)


EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "entities": {"type": "array"},
        "relations": {"type": "array"},
        "constraints": {"type": "array"},
        "scripts": {"type": "array"},
        "candidate_actions": {"type": "array"},
        "missing_knowledge": {"type": "array"},
    },
    "required": [
        "intent",
        "entities",
        "relations",
        "constraints",
        "scripts",
        "candidate_actions",
        "missing_knowledge",
    ],
}


def _merge_unique(left: List[str], right: List[str]) -> List[str]:
    return list(dict.fromkeys(left + right))

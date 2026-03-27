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
class PremiseCandidate:
    premise: str
    hidden_goal: str = ""
    candidate_type: str = "required"
    source: str = "rule"
    support_score: float = 0.6
    evidence: List[str] = field(default_factory=list)


@dataclass
class PremiseValidation:
    premise: str
    hidden_goal: str = ""
    status: str = "supported"
    support_score: float = 0.6
    contradiction_score: float = 0.0
    goal_relevance: float = 0.6
    rationale: str = ""
    requirement_state: str = "unknown"


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
class AnalogicalMatch:
    query: str
    score: float = 0.0
    analogy_type: str = "surface_analogy"
    shared_basis: List[str] = field(default_factory=list)
    shared_nodes: List[str] = field(default_factory=list)
    shared_goals: List[str] = field(default_factory=list)
    shared_requirements: List[str] = field(default_factory=list)
    shared_operator_families: List[str] = field(default_factory=list)


@dataclass
class ContextFrame:
    frame_type: str = "generic_reasoning"
    reasoning_mode: str = "structure_first"
    primary_goal: str = ""
    focus_entities: List[str] = field(default_factory=list)
    active_constraints: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    satisfied_requirements: List[str] = field(default_factory=list)
    available_alternatives: List[str] = field(default_factory=list)
    risk_signals: List[str] = field(default_factory=list)
    basis_signature: List[str] = field(default_factory=list)
    operator_view: List[str] = field(default_factory=list)
    functor_view: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class OperatorInstruction:
    opcode: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.6


@dataclass
class ClaimGrounding:
    claim: str
    grounded: bool = False
    support_kind: str = "unsupported"
    supports: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class OperatorExecutionReport:
    satisfied_facts: List[str] = field(default_factory=list)
    missing_facts: List[str] = field(default_factory=list)
    derived_decisions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    support_trace: List[str] = field(default_factory=list)
    basis_operator_hits: List[str] = field(default_factory=list)
    basis_operator_axes: Dict[str, str] = field(default_factory=dict)
    compiler_alignment_score: float = 0.0
    compiler_findings: List[str] = field(default_factory=list)
    composition_score: float = 0.0
    counterexample_repairs: List[str] = field(default_factory=list)
    claim_groundings: List[ClaimGrounding] = field(default_factory=list)
    claim_grounding_score: float = 0.0

@dataclass
class StructuredMeaningGraph:
    query: str
    intent: str
    domain: str = "general"
    scenario: str = "qa"
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
    satisfied_premises: List[str] = field(default_factory=list)
    missing_premises: List[str] = field(default_factory=list)
    optional_interpretations: List[str] = field(default_factory=list)
    premise_candidates: List[PremiseCandidate] = field(default_factory=list)
    premise_validations: List[PremiseValidation] = field(default_factory=list)
    goal_preservation_checks: List[GoalPreservationCheck] = field(default_factory=list)
    operator_decompositions: List[OperatorDecomposition] = field(default_factory=list)
    functor_hypotheses: List[FunctorHypothesis] = field(default_factory=list)
    operator_instructions: List[OperatorInstruction] = field(default_factory=list)
    operator_execution: OperatorExecutionReport | None = None
    analogical_matches: List[AnalogicalMatch] = field(default_factory=list)
    context_frame: ContextFrame | None = None
    clarification_needed: bool = False
    clarification_score: float = 0.0
    clarification_reasons: List[str] = field(default_factory=list)
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
            scenario=data.get("scenario", "qa"),
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
        graph.satisfied_premises = list(data.get("satisfied_premises", []))
        graph.missing_premises = list(data.get("missing_premises", []))
        graph.optional_interpretations = list(data.get("optional_interpretations", []))
        graph.premise_candidates = [PremiseCandidate(**item) for item in data.get("premise_candidates", [])]
        graph.premise_validations = [PremiseValidation(**item) for item in data.get("premise_validations", [])]
        graph.goal_preservation_checks = [GoalPreservationCheck(**item) for item in data.get("goal_preservation_checks", [])]
        graph.operator_decompositions = [OperatorDecomposition(**item) for item in data.get("operator_decompositions", [])]
        graph.functor_hypotheses = [FunctorHypothesis(**item) for item in data.get("functor_hypotheses", [])]
        graph.operator_instructions = [OperatorInstruction(**item) for item in data.get("operator_instructions", [])]
        operator_execution = data.get("operator_execution")
        if operator_execution:
            payload = dict(operator_execution)
            payload["claim_groundings"] = [ClaimGrounding(**item) for item in payload.get("claim_groundings", [])]
            graph.operator_execution = OperatorExecutionReport(**payload)
        else:
            graph.operator_execution = None
        graph.analogical_matches = [AnalogicalMatch(**item) for item in data.get("analogical_matches", [])]
        context_frame = data.get("context_frame")
        graph.context_frame = ContextFrame(**context_frame) if context_frame else None
        graph.clarification_needed = bool(data.get("clarification_needed", False))
        graph.clarification_score = float(data.get("clarification_score", 0.0))
        graph.clarification_reasons = list(data.get("clarification_reasons", []))
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















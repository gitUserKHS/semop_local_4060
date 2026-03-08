from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List

from .commonsense_kb import concept_label
from .product_profiles import DomainProfile, resolve_domain_profile
from .structures import StructuredMeaningGraph


@dataclass
class OpsKpiReport:
    invalid_advice_rate: float
    plan_executability: float
    missing_prerequisite_rate: float
    context_misread_rate: float
    relation_recovery: float
    human_audit_usefulness: float
    clarification_need_rate: float
    notes: List[str] = field(default_factory=list)

    def model_dump(self) -> Dict[str, object]:
        return asdict(self)


class OpsKpiEvaluator:
    def evaluate(self, graph: StructuredMeaningGraph, domain: str = "general") -> OpsKpiReport:
        profile = resolve_domain_profile(domain)
        invalid_advice_rate = self._invalid_advice_rate(graph)
        missing_prerequisite_rate = self._missing_prerequisite_rate(graph)
        plan_executability = self._plan_executability(graph, missing_prerequisite_rate)
        context_misread_rate = self._context_misread_rate(graph, profile)
        relation_recovery = self._relation_recovery(graph, profile)
        human_audit_usefulness = self._human_audit_usefulness(graph)
        clarification_need_rate = self._clarification_need_rate(graph, profile)
        notes = self._notes(
            invalid_advice_rate,
            missing_prerequisite_rate,
            context_misread_rate,
            relation_recovery,
            human_audit_usefulness,
            clarification_need_rate,
        )
        return OpsKpiReport(
            invalid_advice_rate=invalid_advice_rate,
            plan_executability=plan_executability,
            missing_prerequisite_rate=missing_prerequisite_rate,
            context_misread_rate=context_misread_rate,
            relation_recovery=relation_recovery,
            human_audit_usefulness=human_audit_usefulness,
            clarification_need_rate=clarification_need_rate,
            notes=notes,
        )

    @staticmethod
    def _invalid_advice_rate(graph: StructuredMeaningGraph) -> float:
        denominator = max(1, len(graph.invalid_advice) + len(graph.plan))
        return round(len(graph.invalid_advice) / denominator, 2)

    def _missing_prerequisite_rate(self, graph: StructuredMeaningGraph) -> float:
        requires_edges = [edge for edge in graph.edges if edge.relation == "REQUIRES"]
        if not requires_edges:
            return 0.0
        plan_text = " ".join(step.action + " " + step.rationale + " " + " ".join(step.requires) for step in graph.plan).lower()
        covered = 0
        for edge in requires_edges:
            target_label = concept_label(edge.target).lower()
            if edge.target.lower() in plan_text or target_label in plan_text:
                covered += 1
        missing = max(0, len(requires_edges) - covered)
        return round(missing / len(requires_edges), 2)

    def _plan_executability(self, graph: StructuredMeaningGraph, missing_prerequisite_rate: float) -> float:
        valid_ratio = 1.0
        if graph.plan:
            valid_ratio = sum(1 for step in graph.plan if step.status == "valid") / len(graph.plan)
        blocker_edges = [edge for edge in graph.edges if edge.relation == "BLOCKED_BY"]
        if blocker_edges:
            plan_text = " ".join(step.action + " " + step.rationale for step in graph.plan).lower()
            blocker_hits = 0
            for edge in blocker_edges:
                blocker_label = concept_label(edge.target).lower()
                if edge.target.lower() in plan_text or blocker_label in plan_text or "우회" in plan_text or "hold" in plan_text:
                    blocker_hits += 1
            blocker_ratio = blocker_hits / len(blocker_edges)
        else:
            blocker_ratio = 1.0
        score = (valid_ratio + (1.0 - missing_prerequisite_rate) + blocker_ratio) / 3.0
        return round(score, 2)

    def _context_misread_rate(self, graph: StructuredMeaningGraph, profile: DomainProfile) -> float:
        text = f"{graph.query} {graph.source_context}".lower()
        expected = self._expected_concepts(text, profile.term_to_concept)
        if not expected:
            return 0.0
        recovered = graph.node_ids()
        missed = [concept for concept in expected if concept not in recovered]
        return round(len(missed) / len(expected), 2)

    def _relation_recovery(self, graph: StructuredMeaningGraph, profile: DomainProfile) -> float:
        text = f"{graph.query} {graph.source_context}".lower()
        expected_relations: List[str] = []
        if any(token in text for token in ["승인", "approval", "안전", "certification", "자격"]):
            expected_relations.append("REQUIRES")
        if any(token in text for token in ["막", "blocked", "불일치", "mismatch", "위험"]):
            expected_relations.append("BLOCKED_BY")
        if any(token in text for token in ["대체", "alternative", "우회", "hold", "staging", "보류"]):
            expected_relations.append("ALTERNATIVE")
        if not expected_relations and profile.name != "general":
            expected_relations = ["REQUIRES"]
        if not expected_relations:
            return 1.0
        recovered = {edge.relation for edge in graph.edges}
        score = sum(1 for relation in expected_relations if relation in recovered) / len(expected_relations)
        return round(score, 2)

    @staticmethod
    def _human_audit_usefulness(graph: StructuredMeaningGraph) -> float:
        edge_provenance = sum(1 for edge in graph.edges if edge.provenance)
        operator_provenance = sum(1 for operator in graph.induced_operators if operator.provenance)
        rationale_ratio = 0.0
        if graph.plan:
            rationale_ratio = sum(1 for step in graph.plan if step.rationale) / len(graph.plan)
        evidence_bonus = 1.0 if any(result.evidence or result.equations for result in graph.symbolic_results) else 0.0
        audit_bonus = 1.0 if graph.audit_trace else 0.0
        score = 0.2 * min(1.0, edge_provenance / max(1, len(graph.edges)))
        score += 0.2 * min(1.0, operator_provenance / max(1, len(graph.induced_operators) or 1))
        score += 0.3 * rationale_ratio
        score += 0.15 * evidence_bonus
        score += 0.15 * audit_bonus
        return round(min(1.0, score), 2)

    def _clarification_need_rate(self, graph: StructuredMeaningGraph, profile: DomainProfile) -> float:
        query = graph.query.lower()
        ambiguity_hits = sum(1 for trigger in profile.clarification_triggers if trigger in query)
        score = 0.0
        if profile.name != "general" and not graph.source_context.strip():
            score += 0.25
        if ambiguity_hits:
            score += min(0.5, 0.15 * ambiguity_hits)
        if graph.intent == "generic_reasoning" and not graph.symbolic_results:
            score += 0.25
        if any("missing knowledge" in warning.lower() for warning in graph.warnings):
            score += 0.25
        return round(min(1.0, score), 2)

    @staticmethod
    def _expected_concepts(text: str, term_to_concept: Dict[str, str]) -> List[str]:
        expected: List[str] = []
        for token, concept in term_to_concept.items():
            if token.lower() in text and concept not in expected:
                expected.append(concept)
        return expected

    @staticmethod
    def _notes(
        invalid_advice_rate: float,
        missing_prerequisite_rate: float,
        context_misread_rate: float,
        relation_recovery: float,
        human_audit_usefulness: float,
        clarification_need_rate: float,
    ) -> List[str]:
        notes: List[str] = []
        if invalid_advice_rate > 0.2:
            notes.append("invalid advice exposure is still too high for production-safe automation.")
        if missing_prerequisite_rate > 0.25:
            notes.append("prerequisite coverage is weak; add stronger SOP extraction or operator constraints.")
        if context_misread_rate > 0.25:
            notes.append("the graph is missing domain terms present in the source context.")
        if relation_recovery < 0.7:
            notes.append("important operational relations are not being recovered reliably enough.")
        if human_audit_usefulness < 0.7:
            notes.append("audit trace quality is below the level needed for supervisors or incident review.")
        if clarification_need_rate > 0.5:
            notes.append("the assistant should ask a clarifying question before giving an operational answer.")
        return notes

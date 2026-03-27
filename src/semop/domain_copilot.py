from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List

from .feedback_rules import FeedbackRuleSet
from .ops_kpi import OpsKpiEvaluator, OpsKpiReport
from .product_profiles import DomainProfile, resolve_domain_profile
from .response_synthesizer import ResponseSynthesizer
from .pipeline import StructuredMeaningPipeline
from .review_queue import ReviewQueueStore, infer_review_severity, review_reasons_from_graph_and_kpis
from .structures import StructuredMeaningGraph


@dataclass
class CopilotRequest:
    query: str
    context: str = ""
    domain: str = "general"
    scenario: str = "qa"


@dataclass
class AuditItem:
    stage: str
    detail: str


@dataclass
class CopilotResult:
    request: CopilotRequest
    graph: StructuredMeaningGraph
    answer_text: str
    kpis: OpsKpiReport
    audit_items: List[AuditItem]
    queued_for_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    review_severity: str = "medium"

    def model_dump(self) -> Dict[str, Any]:
        return {
            "request": asdict(self.request),
            "graph": self.graph.model_dump(),
            "answer_text": self.answer_text,
            "kpis": self.kpis.model_dump(),
            "audit_items": [asdict(item) for item in self.audit_items],
            "queued_for_review": self.queued_for_review,
            "review_reasons": self.review_reasons,
            "review_severity": self.review_severity,
        }

    def model_dump_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=ensure_ascii)

    def to_text(self) -> str:
        lines = [self.answer_text, "", f"도메인: {self.request.domain}"]
        lines.append("운영 KPI:")
        lines.append(f"- invalid advice rate: {self.kpis.invalid_advice_rate}")
        lines.append(f"- plan executability: {self.kpis.plan_executability}")
        lines.append(f"- missing prerequisite rate: {self.kpis.missing_prerequisite_rate}")
        lines.append(f"- context misread rate: {self.kpis.context_misread_rate}")
        lines.append(f"- relation recovery: {self.kpis.relation_recovery}")
        lines.append(f"- human audit usefulness: {self.kpis.human_audit_usefulness}")
        lines.append(f"- clarification need rate: {self.kpis.clarification_need_rate}")
        if self.queued_for_review:
            lines.append(f"- review queue: queued [{self.review_severity}] ({', '.join(self.review_reasons)})")
        if self.kpis.notes:
            lines.append("")
            lines.append("운영 메모:")
            lines.extend(f"- {note}" for note in self.kpis.notes)
        if self.audit_items:
            lines.append("")
            lines.append("감사 추적:")
            lines.extend(f"- [{item.stage}] {item.detail}" for item in self.audit_items)
        return "\n".join(lines)


class DomainCopilot:
    def __init__(
        self,
        mode: str = "heuristic",
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        memory_store_path: str | None = None,
        memory_source: str | None = None,
        review_queue_path: str | None = None,
        feedback_rules_path: str | None = None,
    ):
        self.pipeline = StructuredMeaningPipeline(
            mode=mode,
            model_id=model_id,
            memory_store_path=memory_store_path,
            memory_source=memory_source,
        )
        self.synthesizer = ResponseSynthesizer()
        self.kpi_evaluator = OpsKpiEvaluator()
        self.review_queue = ReviewQueueStore(review_queue_path) if review_queue_path else None
        self.feedback_rules = FeedbackRuleSet.from_path(feedback_rules_path)

    def run(self, request: CopilotRequest) -> CopilotResult:
        profile = resolve_domain_profile(request.domain)
        pipeline_input = self._compose_input(request)
        graph = self.pipeline.run(pipeline_input)
        graph.query = request.query
        graph.domain = profile.name
        graph.scenario = request.scenario
        graph.source_context = request.context
        self._apply_profile_guards(graph, profile)
        self._apply_feedback_rules(graph, request)
        audit_items = self._build_audit_items(graph, request, profile)
        graph.audit_trace = [f"{item.stage}: {item.detail}" for item in audit_items]
        answer_text = self.synthesizer.synthesize(graph).to_text()
        kpis = self.kpi_evaluator.evaluate(graph, domain=profile.name)
        result = CopilotResult(request=request, graph=graph, answer_text=answer_text, kpis=kpis, audit_items=audit_items)
        self._enqueue_review_if_needed(result)
        if result.queued_for_review:
            result.audit_items.append(AuditItem(stage="review", detail="queued_for_review=" + ",".join(result.review_reasons)))
            result.graph.audit_trace.append("review: queued_for_review=" + ",".join(result.review_reasons))
        return result

    @staticmethod
    def _compose_input(request: CopilotRequest) -> str:
        if not request.context.strip():
            return request.query
        return (
            f"[Domain: {request.domain}]\n"
            f"[Scenario: {request.scenario}]\n"
            f"SOP Context:\n{request.context.strip()}\n\n"
            f"Question:\n{request.query.strip()}"
        )

    @staticmethod
    def _apply_profile_guards(graph: StructuredMeaningGraph, profile: DomainProfile) -> None:
        text = f"{graph.query} {graph.source_context}".lower()
        missing_terms: List[str] = []
        for token, concept in profile.term_to_concept.items():
            if token.lower() in text and concept not in graph.node_ids() and token not in missing_terms:
                missing_terms.append(token)
        if missing_terms:
            graph.warnings.append("domain terms seen but weakly grounded: " + ", ".join(missing_terms[:5]))
        if profile.name != "general" and not graph.source_context.strip():
            graph.warnings.append("domain copilot ran without SOP context; answer quality is limited.")

    def _apply_feedback_rules(self, graph: StructuredMeaningGraph, request: CopilotRequest) -> None:
        if self.feedback_rules.is_empty():
            return
        text = f"{request.query} {request.context}".lower()
        matches = self.feedback_rules.matching_rules(domain=request.domain, scenario=request.scenario, text=text)
        if not matches:
            return
        required_step_text = " ".join(step.action + " " + step.rationale for step in graph.plan).lower()
        applied_rule_ids: List[str] = []
        for rule in matches:
            applied_rule_ids.append(rule.id)
            for phrase in rule.avoid_phrases:
                if phrase not in graph.invalid_advice:
                    graph.invalid_advice.append(phrase)
            for action in rule.recommended_actions:
                if action not in graph.creative_alternatives:
                    graph.creative_alternatives.insert(0, action)
            missing_requirements = [term for term in rule.require_terms if term.lower() not in required_step_text]
            if missing_requirements:
                graph.warnings.append("feedback rule requires attention to: " + ", ".join(missing_requirements[:4]))
            if rule.explanation:
                graph.warnings.append("feedback note: " + rule.explanation)
        graph.warnings.append("feedback rules applied: " + ", ".join(applied_rule_ids))

    @staticmethod
    def _build_audit_items(graph: StructuredMeaningGraph, request: CopilotRequest, profile: DomainProfile) -> List[AuditItem]:
        items = [
            AuditItem(stage="input", detail=f"domain={profile.name}, scenario={request.scenario}, context_chars={len(request.context)}"),
            AuditItem(stage="graph", detail=f"intent={graph.intent}, nodes={len(graph.nodes)}, edges={len(graph.edges)}"),
        ]
        if graph.induced_operators:
            families = ", ".join(candidate.family for candidate in graph.induced_operators[:3])
            items.append(AuditItem(stage="operators", detail=f"top_families={families}"))
        if graph.plan:
            valid_steps = sum(1 for step in graph.plan if step.status == "valid")
            items.append(AuditItem(stage="plan", detail=f"valid_steps={valid_steps}/{len(graph.plan)}"))
        if graph.symbolic_results:
            top_result = graph.symbolic_results[0]
            evidence = top_result.equations[0] if top_result.equations else (top_result.evidence[0] if top_result.evidence else top_result.answer)
            items.append(AuditItem(stage="evidence", detail=f"{top_result.domain}: {evidence}"))
        if graph.invalid_advice:
            items.append(AuditItem(stage="safety", detail=f"blocked_unsafe_patterns={len(graph.invalid_advice)}"))
        if graph.warnings:
            items.append(AuditItem(stage="warnings", detail=" | ".join(graph.warnings[:3])))
        return items

    def _enqueue_review_if_needed(self, result: CopilotResult) -> None:
        if self.review_queue is None:
            return
        reasons = review_reasons_from_graph_and_kpis(result.graph, result.kpis.model_dump())
        if not reasons:
            return
        severity = infer_review_severity(
            result.request.domain,
            result.request.scenario,
            reasons,
            result.kpis.model_dump(),
        )
        self.review_queue.enqueue(
            domain=result.request.domain,
            scenario=result.request.scenario,
            query=result.request.query,
            reasons=reasons,
            answer_text=result.answer_text,
            kpis=result.kpis.model_dump(),
            audit_items=[asdict(item) for item in result.audit_items],
            context_text=result.request.context,
            graph_payload=result.graph.model_dump(),
            severity=severity,
        )
        result.queued_for_review = True
        result.review_reasons = reasons
        result.review_severity = severity





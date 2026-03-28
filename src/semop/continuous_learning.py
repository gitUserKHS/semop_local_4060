from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .distillation import DistillationSftRecord, TeacherTraceRecord
from .graph_supervision import GraphSupervisionExporter
from .operating_policies import collect_review_promotion_decisions
from .review_queue import severity_weight
from .structures import StructuredMeaningGraph
from .unified_world_model import UnifiedWorldModelEngine


@dataclass
class ContinuousLearningBundleSummary:
    output_dir: str
    trace_count: int
    graph_supervision_count: int
    sft_record_count: int
    approved_review_count: int
    promoted_review_count: int
    filtered_review_count: int
    promoted_training_weight_total: float = 0.0
    operating_domain: str = 'general'
    integrated_trace_count: int = 0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class ContinuousLearningBundleBuilder:
    def build_from_graphs(
        self,
        graphs: Sequence[StructuredMeaningGraph],
        output_dir: str | Path,
        review_store_path: str | Path | None = None,
        operating_domain: str | None = None,
    ) -> ContinuousLearningBundleSummary:
        graphs = list(graphs)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        trace_rows = [self._trace_from_graph(graph) for graph in graphs]
        trace_path = output / 'teacher_traces.jsonl'
        self._write_jsonl(trace_path, trace_rows)

        integrated_trace_rows = [self._integrated_world_trace(graph) for graph in graphs]
        integrated_trace_path = output / 'integrated_world_traces.jsonl'
        self._write_jsonl(integrated_trace_path, integrated_trace_rows)

        supervision_summary = GraphSupervisionExporter().export_from_graphs(graphs, output / 'graph_supervision.jsonl')

        approved_review_count = 0
        promoted_review_count = 0
        filtered_review_count = 0
        promoted_training_weight_total = 0.0
        promoted_reviews: list[dict[str, Any]] = []
        promotion_manifest: list[dict[str, Any]] = []
        if review_store_path:
            review_rows = collect_review_promotion_decisions(review_store_path, domain=operating_domain)
            approved_review_count = len(review_rows)
            promoted_reviews = [detail for detail, decision in review_rows if decision.promotable]
            promoted_review_count = len(promoted_reviews)
            filtered_review_count = approved_review_count - promoted_review_count
            promotion_manifest = []
            for detail, decision in review_rows:
                training_weight = severity_weight(detail.get('severity'), decision.domain, str(detail.get('scenario', 'qa'))) if decision.promotable else 0.0
                if decision.promotable:
                    promoted_training_weight_total += training_weight
                promotion_manifest.append(
                    {
                        'review_id': decision.item_id,
                        'domain': decision.domain,
                        'scenario': str(detail.get('scenario', 'qa')),
                        'query': decision.query,
                        'severity': str(detail.get('severity', 'medium')),
                        'training_weight': round(training_weight, 4),
                        'promotable': decision.promotable,
                        'matched_reasons': decision.matched_reasons,
                        'blocked_reasons': decision.blocked_reasons,
                        'override_reasons': decision.override_reasons,
                        'rationale': decision.rationale,
                    }
                )

        sft_rows = [self._sft_from_graph(graph) for graph in graphs]
        sft_rows.extend(self._sft_from_review(item) for item in promoted_reviews)
        sft_path = output / 'continuous_learning_sft.jsonl'
        self._write_jsonl(sft_path, sft_rows)

        manifest = {
            'trace_path': str(trace_path),
            'integrated_trace_path': str(integrated_trace_path),
            'graph_supervision_path': str(output / 'graph_supervision.jsonl'),
            'sft_path': str(sft_path),
            'approved_review_count': approved_review_count,
            'promoted_review_count': promoted_review_count,
            'filtered_review_count': filtered_review_count,
            'promoted_training_weight_total': round(promoted_training_weight_total, 4),
            'operating_domain': str(operating_domain or 'general'),
            'review_promotion_manifest_path': str(output / 'review_promotion_manifest.json'),
        }
        (output / 'bundle_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        (output / 'review_promotion_manifest.json').write_text(json.dumps(promotion_manifest, ensure_ascii=False, indent=2), encoding='utf-8')

        return ContinuousLearningBundleSummary(
            output_dir=str(output),
            trace_count=len(trace_rows),
            graph_supervision_count=supervision_summary.exported_examples,
            sft_record_count=len(sft_rows),
            approved_review_count=approved_review_count,
            promoted_review_count=promoted_review_count,
            filtered_review_count=filtered_review_count,
            promoted_training_weight_total=round(promoted_training_weight_total, 4),
            operating_domain=str(operating_domain or 'general'),
            integrated_trace_count=len(integrated_trace_rows),
        )

    @staticmethod
    def _trace_from_graph(graph: StructuredMeaningGraph) -> TeacherTraceRecord:
        claim_groundings = ContinuousLearningBundleBuilder._claim_grounding_payload(graph)
        repair_programs = ContinuousLearningBundleBuilder._repair_program_payload(graph)
        engine = UnifiedWorldModelEngine()
        integrated_world = engine.from_graph(graph, source='continuous_learning')
        integrated_reasoning = engine.reason(integrated_world)
        unsupported_claim_count = sum(1 for item in claim_groundings if not item.get('grounded', False))
        return TeacherTraceRecord(
            task='continuous_runtime_trace',
            input_text=graph.query,
            input_payload={'query': graph.query, 'source_context': graph.source_context},
            teacher_trace={
                'intent': graph.intent,
                'domain': graph.domain,
                'hidden_goals': list(graph.hidden_goals),
                'required_premises': list(graph.required_premises),
                'missing_premises': list(graph.missing_premises),
                'operator_decompositions': [item.operator_name for item in graph.operator_decompositions[:8]],
                'compiled_decisions': list(graph.operator_execution.derived_decisions[:8]) if graph.operator_execution is not None else [],
                'grounding_edges': [f"{edge.source}:{edge.relation}:{edge.target}" for edge in graph.edges if edge.relation == 'GROUNDED_BY'][:8],
                'claim_groundings': claim_groundings,
                'repair_programs': repair_programs,
                'integrated_world': integrated_world.model_dump(),
                'integrated_reasoning': integrated_reasoning.model_dump(),
            },
            completion_payload={
                'hidden_goals': list(graph.hidden_goals),
                'required_premises': list(graph.required_premises),
                'operator_decompositions': [item.operator_name for item in graph.operator_decompositions[:8]],
                'claim_groundings': claim_groundings,
                'repair_programs_applied': repair_programs['applied_programs'],
                'repair_actions_applied': repair_programs['applied_actions'],
                'integrated_reasoning': integrated_reasoning.model_dump(),
            },
            metadata={
                'composition_score': float(graph.operator_execution.composition_score) if graph.operator_execution is not None else 0.0,
                'claim_grounding_score': float(graph.operator_execution.claim_grounding_score) if graph.operator_execution is not None else 0.0,
                'unsupported_claim_count': unsupported_claim_count,
                'repair_program_count': len(repair_programs['applied_programs']),
                'repair_rejection_count': len(repair_programs['rejected_actions']),
                'world_entity_count': len(integrated_world.entities),
                'world_relation_count': len(integrated_world.relations),
                'audit_trace': list(graph.audit_trace[:12]),
            },
        )

    @staticmethod
    def _sft_from_graph(graph: StructuredMeaningGraph) -> DistillationSftRecord:
        claim_groundings = ContinuousLearningBundleBuilder._claim_grounding_payload(graph)
        repair_programs = ContinuousLearningBundleBuilder._repair_program_payload(graph)
        engine = UnifiedWorldModelEngine()
        integrated_world = engine.from_graph(graph, source='continuous_learning')
        integrated_reasoning = engine.reason(integrated_world)
        completion = {
            'intent': graph.intent,
            'domain': graph.domain,
            'hidden_goals': list(graph.hidden_goals),
            'required_premises': list(graph.required_premises),
            'operator_decompositions': [item.operator_name for item in graph.operator_decompositions[:8]],
            'claim_groundings': claim_groundings,
            'repair_programs_applied': repair_programs['applied_programs'],
            'repair_actions_applied': repair_programs['applied_actions'],
            'integrated_reasoning': integrated_reasoning.model_dump(),
        }
        prompt = (
            'Read the query and return the operator-graph slots. '
            'Output JSON fields: intent, domain, hidden_goals, required_premises, operator_decompositions, claim_groundings, repair_programs_applied, repair_actions_applied, integrated_reasoning.\n\n'
            f'Query:\n{graph.query}\n\n'
            f'Source context:\n{graph.source_context[:400]}'
        )
        return DistillationSftRecord(
            prompt=prompt,
            completion=json.dumps(completion, ensure_ascii=False),
            task='continuous_graph_slots',
            metadata={
                'training_weight': 1.0,
                'source': 'runtime_graph',
                'unsupported_claim_count': sum(1 for item in claim_groundings if not item.get('grounded', False)),
                'repair_program_count': len(repair_programs['applied_programs']),
                'repair_rejection_count': len(repair_programs['rejected_actions']),
                'world_entity_count': len(integrated_world.entities),
                'world_relation_count': len(integrated_world.relations),
            },
        )

    @staticmethod
    def _integrated_world_trace(graph: StructuredMeaningGraph) -> dict[str, Any]:
        engine = UnifiedWorldModelEngine()
        world = engine.from_graph(graph, source='continuous_learning')
        reasoning = engine.reason(world)
        return {
            'query': graph.query,
            'intent': graph.intent,
            'domain': graph.domain,
            'integrated_world': world.model_dump(),
            'integrated_reasoning': reasoning.model_dump(),
            'integrated_reasoning_text': reasoning.summary,
        }

    @staticmethod
    def _sft_from_review(detail: dict[str, Any]) -> DistillationSftRecord:
        severity = str(detail.get('severity', 'medium'))
        training_weight = severity_weight(severity, str(detail.get('domain', 'general')), str(detail.get('scenario', 'qa')))
        review_claims = ContinuousLearningBundleBuilder._review_claim_grounding(detail)
        review_repairs = ContinuousLearningBundleBuilder._review_repair_programs(detail)
        prompt = (
            'Read the reviewed scenario and return a corrected structured answer. '
            'Output concise JSON or text consistent with the approved review.\n\n'
            f"Query:\n{detail.get('query', '')}\n\n"
            f"Reasons:\n{json.dumps(detail.get('reasons', []), ensure_ascii=False)}\n\n"
            f"Audit:\n{json.dumps(detail.get('audit_items', []), ensure_ascii=False)}"
        )
        if review_claims['unsupported_claims']:
            prompt += f"\n\nUnsupported claims:\n{json.dumps(review_claims['unsupported_claims'], ensure_ascii=False)}"
        if review_claims['grounded_claims']:
            prompt += f"\n\nGrounded claims:\n{json.dumps(review_claims['grounded_claims'], ensure_ascii=False)}"
        if review_repairs['applied_programs']:
            prompt += f"\n\nApplied repair programs:\n{json.dumps(review_repairs['applied_programs'], ensure_ascii=False)}"
        if review_repairs['rejected_actions']:
            prompt += f"\n\nRejected repairs:\n{json.dumps(review_repairs['rejected_actions'], ensure_ascii=False)}"
        completion = detail.get('answer_text', '') or json.dumps({'resolution_note': detail.get('resolution_note', '')}, ensure_ascii=False)
        return DistillationSftRecord(
            prompt=prompt,
            completion=completion,
            task='continuous_review_correction',
            metadata={
                'review_id': int(detail.get('id', 0) or 0),
                'domain': str(detail.get('domain', 'general')),
                'scenario': str(detail.get('scenario', 'qa')),
                'severity': severity,
                'training_weight': round(training_weight, 4),
                'claim_grounding_score': review_claims['claim_grounding_score'],
                'unsupported_claim_count': len(review_claims['unsupported_claims']),
                'repair_program_count': len(review_repairs['applied_programs']),
                'repair_rejection_count': len(review_repairs['rejected_actions']),
            },
        )

    @staticmethod
    def _claim_grounding_payload(graph: StructuredMeaningGraph) -> list[dict[str, Any]]:
        report = graph.operator_execution
        if report is None:
            return []
        return [
            {
                'claim': item.claim,
                'grounded': bool(item.grounded),
                'support_kind': item.support_kind,
                'supports': list(item.supports[:2]),
                'score': float(item.score),
            }
            for item in report.claim_groundings[:6]
        ]

    @staticmethod
    def _repair_program_payload(graph: StructuredMeaningGraph) -> dict[str, Any]:
        report = graph.operator_execution
        if report is None:
            return {'applied_programs': [], 'applied_actions': [], 'rejected_actions': []}
        decisions = list(report.derived_decisions)
        applied_programs = list(dict.fromkeys(item.split(':', 1)[1] for item in decisions if item.startswith('repair_program:')))
        applied_actions = list(dict.fromkeys(item.split(':', 1)[1] for item in decisions if item.startswith('repair_applied:')))
        rejected_actions = list(dict.fromkeys(item.split(':', 1)[1] for item in decisions if item.startswith('repair_rejected:')))
        return {
            'applied_programs': applied_programs[:6],
            'applied_actions': applied_actions[:8],
            'rejected_actions': rejected_actions[:6],
        }

    @staticmethod
    def _review_claim_grounding(detail: dict[str, Any]) -> dict[str, Any]:
        payload = detail.get('graph') if isinstance(detail.get('graph'), dict) else {}
        execution = payload.get('operator_execution', {}) if isinstance(payload.get('operator_execution'), dict) else {}
        claim_groundings = execution.get('claim_groundings', []) if isinstance(execution.get('claim_groundings'), list) else []
        grounded_claims = [
            str(item.get('claim', '')).strip()
            for item in claim_groundings
            if isinstance(item, dict) and bool(item.get('grounded')) and str(item.get('claim', '')).strip()
        ]
        unsupported_claims = [
            str(item.get('claim', '')).strip()
            for item in claim_groundings
            if isinstance(item, dict) and not bool(item.get('grounded')) and str(item.get('claim', '')).strip()
        ]
        return {
            'grounded_claims': grounded_claims[:4],
            'unsupported_claims': unsupported_claims[:4],
            'claim_grounding_score': round(float(execution.get('claim_grounding_score', 0.0) or 0.0), 4),
        }

    @staticmethod
    def _review_repair_programs(detail: dict[str, Any]) -> dict[str, Any]:
        payload = detail.get('graph') if isinstance(detail.get('graph'), dict) else {}
        execution = payload.get('operator_execution', {}) if isinstance(payload.get('operator_execution'), dict) else {}
        decisions = execution.get('derived_decisions', []) if isinstance(execution.get('derived_decisions'), list) else []
        applied_programs = [
            str(item).split(':', 1)[1]
            for item in decisions
            if isinstance(item, str) and item.startswith('repair_program:') and ':' in item
        ]
        rejected_actions = [
            str(item).split(':', 1)[1]
            for item in decisions
            if isinstance(item, str) and item.startswith('repair_rejected:') and ':' in item
        ]
        return {
            'applied_programs': list(dict.fromkeys(applied_programs))[:4],
            'rejected_actions': list(dict.fromkeys(rejected_actions))[:4],
        }

    @staticmethod
    def _write_jsonl(path: Path, rows: Sequence[Any]) -> None:
        with path.open('w', encoding='utf-8') as handle:
            for row in rows:
                payload = row.model_dump() if hasattr(row, 'model_dump') else row
                handle.write(json.dumps(payload, ensure_ascii=False) + '\n')


def build_continuous_learning_bundle(
    graphs: Sequence[StructuredMeaningGraph],
    output_dir: str | Path,
    review_store_path: str | Path | None = None,
    operating_domain: str | None = None,
) -> ContinuousLearningBundleSummary:
    return ContinuousLearningBundleBuilder().build_from_graphs(graphs, output_dir, review_store_path=review_store_path, operating_domain=operating_domain)

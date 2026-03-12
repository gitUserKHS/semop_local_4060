from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .distillation import DistillationSftRecord, TeacherTraceRecord
from .graph_supervision import GraphSupervisionExporter
from .review_queue import ReviewQueueStore
from .structures import StructuredMeaningGraph


@dataclass
class ContinuousLearningBundleSummary:
    output_dir: str
    trace_count: int
    graph_supervision_count: int
    sft_record_count: int
    approved_review_count: int

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class ContinuousLearningBundleBuilder:
    def build_from_graphs(
        self,
        graphs: Sequence[StructuredMeaningGraph],
        output_dir: str | Path,
        review_store_path: str | Path | None = None,
    ) -> ContinuousLearningBundleSummary:
        graphs = list(graphs)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        trace_rows = [self._trace_from_graph(graph) for graph in graphs]
        trace_path = output / 'teacher_traces.jsonl'
        self._write_jsonl(trace_path, trace_rows)

        supervision_summary = GraphSupervisionExporter().export_from_graphs(graphs, output / 'graph_supervision.jsonl')

        approved_reviews = self._approved_reviews(review_store_path)
        sft_rows = [self._sft_from_graph(graph) for graph in graphs]
        sft_rows.extend(self._sft_from_review(item) for item in approved_reviews)
        sft_path = output / 'continuous_learning_sft.jsonl'
        self._write_jsonl(sft_path, sft_rows)

        manifest = {
            'trace_path': str(trace_path),
            'graph_supervision_path': str(output / 'graph_supervision.jsonl'),
            'sft_path': str(sft_path),
            'approved_review_count': len(approved_reviews),
        }
        (output / 'bundle_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

        return ContinuousLearningBundleSummary(
            output_dir=str(output),
            trace_count=len(trace_rows),
            graph_supervision_count=supervision_summary.exported_examples,
            sft_record_count=len(sft_rows),
            approved_review_count=len(approved_reviews),
        )

    @staticmethod
    def _trace_from_graph(graph: StructuredMeaningGraph) -> TeacherTraceRecord:
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
            },
            completion_payload={
                'hidden_goals': list(graph.hidden_goals),
                'required_premises': list(graph.required_premises),
                'operator_decompositions': [item.operator_name for item in graph.operator_decompositions[:8]],
            },
            metadata={
                'composition_score': float(graph.operator_execution.composition_score) if graph.operator_execution is not None else 0.0,
                'audit_trace': list(graph.audit_trace[:12]),
            },
        )

    @staticmethod
    def _sft_from_graph(graph: StructuredMeaningGraph) -> DistillationSftRecord:
        completion = {
            'intent': graph.intent,
            'domain': graph.domain,
            'hidden_goals': list(graph.hidden_goals),
            'required_premises': list(graph.required_premises),
            'operator_decompositions': [item.operator_name for item in graph.operator_decompositions[:8]],
        }
        prompt = (
            'Read the query and return the operator-graph slots. '
            'Output JSON fields: intent, domain, hidden_goals, required_premises, operator_decompositions.\n\n'
            f'Query:\n{graph.query}\n\n'
            f'Source context:\n{graph.source_context[:400]}'
        )
        return DistillationSftRecord(prompt=prompt, completion=json.dumps(completion, ensure_ascii=False), task='continuous_graph_slots')

    @staticmethod
    def _approved_reviews(review_store_path: str | Path | None) -> list[dict[str, Any]]:
        if not review_store_path:
            return []
        store = ReviewQueueStore(review_store_path)
        items = store.fetch_items(status='approved', limit=200)
        return [store.fetch_item_detail(item.id) for item in items if store.fetch_item_detail(item.id) is not None]

    @staticmethod
    def _sft_from_review(detail: dict[str, Any]) -> DistillationSftRecord:
        prompt = (
            'Read the reviewed scenario and return a corrected structured answer. '
            'Output concise JSON or text consistent with the approved review.\n\n'
            f"Query:\n{detail.get('query', '')}\n\n"
            f"Reasons:\n{json.dumps(detail.get('reasons', []), ensure_ascii=False)}\n\n"
            f"Audit:\n{json.dumps(detail.get('audit_items', []), ensure_ascii=False)}"
        )
        completion = detail.get('answer_text', '') or json.dumps({'resolution_note': detail.get('resolution_note', '')}, ensure_ascii=False)
        return DistillationSftRecord(prompt=prompt, completion=completion, task='continuous_review_correction')

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
) -> ContinuousLearningBundleSummary:
    return ContinuousLearningBundleBuilder().build_from_graphs(graphs, output_dir, review_store_path=review_store_path)

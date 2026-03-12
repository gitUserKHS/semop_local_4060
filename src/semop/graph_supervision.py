from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .structures import StructuredMeaningGraph


@dataclass
class GraphSupervisionExample:
    query: str
    source_context: str
    intent: str
    domain: str
    hidden_goals: list[str]
    required_premises: list[str]
    satisfied_premises: list[str]
    missing_premises: list[str]
    relation_triples: list[tuple[str, str, str]]
    operator_decompositions: list[str]
    operator_families: list[str]
    functor_names: list[str]
    grounding_edge_count: int
    evidence_node_count: int
    compiler_score: float

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphSupervisionExportSummary:
    output_path: str
    exported_examples: int

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class GraphSupervisionExporter:
    def export_from_graphs(
        self,
        graphs: Sequence[StructuredMeaningGraph],
        output_path: str | Path,
    ) -> GraphSupervisionExportSummary:
        rows = [self._example_from_graph(graph).model_dump() for graph in graphs]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows)
        output.write_text(payload + ('\n' if payload else ''), encoding='utf-8')
        return GraphSupervisionExportSummary(output_path=str(output), exported_examples=len(rows))

    @staticmethod
    def _example_from_graph(graph: StructuredMeaningGraph) -> GraphSupervisionExample:
        relation_triples = [
            (edge.source, edge.relation, edge.target)
            for edge in graph.edges
            if edge.relation in {'REQUIRES', 'BLOCKED_BY', 'CONTAINS', 'GROUNDED_BY', 'USES_CONTEXT'}
        ]
        grounding_edge_count = sum(1 for edge in graph.edges if edge.source == 'question' and edge.relation == 'GROUNDED_BY')
        evidence_node_count = sum(1 for node in graph.nodes if node.kind == 'evidence')
        compiler_score = float(graph.operator_execution.composition_score) if graph.operator_execution is not None else 0.0
        return GraphSupervisionExample(
            query=graph.query,
            source_context=graph.source_context,
            intent=graph.intent,
            domain=graph.domain,
            hidden_goals=list(graph.hidden_goals),
            required_premises=list(graph.required_premises),
            satisfied_premises=list(graph.satisfied_premises),
            missing_premises=list(graph.missing_premises),
            relation_triples=relation_triples,
            operator_decompositions=[item.operator_name for item in graph.operator_decompositions],
            operator_families=[item.family for item in graph.induced_operators],
            functor_names=[item.name for item in graph.functor_hypotheses],
            grounding_edge_count=grounding_edge_count,
            evidence_node_count=evidence_node_count,
            compiler_score=compiler_score,
        )


def export_graph_supervision_from_graphs(
    graphs: Sequence[StructuredMeaningGraph],
    output_path: str | Path,
) -> GraphSupervisionExportSummary:
    return GraphSupervisionExporter().export_from_graphs(graphs, output_path)

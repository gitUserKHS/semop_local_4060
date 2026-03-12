from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .structures import FunctorHypothesis, StructuredMeaningGraph


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9a-z_]+", text.lower())


@dataclass
class MultimodalAlignmentRecord:
    alignment_id: str
    hidden_goals: list[str] = field(default_factory=list)
    required_premises: list[str] = field(default_factory=list)
    visual_terms: list[str] = field(default_factory=list)
    operator_families: list[str] = field(default_factory=list)
    support: int = 0
    confidence: float = 0.0
    functor_name: str = 'VisualEvidenceToGoalConstraintFunctor'
    rationale: str = ''

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MultimodalAlignmentModel:
    records: list[MultimodalAlignmentRecord] = field(default_factory=list)
    min_match_score: float = 0.42
    trained_on_graphs: int = 0

    def model_dump(self) -> dict[str, Any]:
        return {
            'records': [item.model_dump() for item in self.records],
            'min_match_score': self.min_match_score,
            'trained_on_graphs': self.trained_on_graphs,
        }

    @classmethod
    def from_path(cls, path: str | Path | None) -> "MultimodalAlignmentModel | None":
        if not path:
            return None
        payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if isinstance(payload, dict) and 'weights' in payload:
            payload = payload['weights']
        return cls(
            records=[MultimodalAlignmentRecord(**item) for item in payload.get('records', [])],
            min_match_score=float(payload.get('min_match_score', 0.42)),
            trained_on_graphs=int(payload.get('trained_on_graphs', 0)),
        )


@dataclass
class MultimodalAlignmentTrainingSummary:
    output_path: str
    trained_on_graphs: int
    retained_alignment_count: int
    model: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class MultimodalAlignmentTrainer:
    def train_from_graphs(
        self,
        graphs: Sequence[StructuredMeaningGraph],
        output_path: str | Path,
        min_support: int = 1,
    ) -> MultimodalAlignmentTrainingSummary:
        grouped: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]] = {}
        graphs = list(graphs)
        for graph in graphs:
            visual_terms = self._visual_terms(graph)
            if not visual_terms or not graph.hidden_goals:
                continue
            key = (tuple(sorted(graph.hidden_goals[:2])), tuple(sorted(graph.required_premises[:3])))
            bucket = grouped.setdefault(
                key,
                {
                    'hidden_goals': list(dict.fromkeys(graph.hidden_goals[:3])),
                    'required_premises': list(dict.fromkeys(graph.required_premises[:4])),
                    'visual_terms': {},
                    'operator_families': {},
                    'support': 0,
                    'confidence_sum': 0.0,
                    'rationales': [],
                },
            )
            for term in visual_terms:
                bucket['visual_terms'][term] = int(bucket['visual_terms'].get(term, 0)) + 1
            for candidate in graph.induced_operators:
                if any(tag.startswith('multimodal:') for tag in candidate.provenance):
                    bucket['operator_families'][candidate.family] = int(bucket['operator_families'].get(candidate.family, 0)) + 1
            bucket['support'] += 1
            confidence = float(graph.operator_execution.composition_score) if graph.operator_execution is not None else 0.7
            bucket['confidence_sum'] += max(0.5, confidence)
            bucket['rationales'].append(' '.join(graph.audit_trace[:2]))

        records: list[MultimodalAlignmentRecord] = []
        for index, bucket in enumerate(grouped.values(), start=1):
            support = int(bucket['support'])
            if support < min_support:
                continue
            records.append(
                MultimodalAlignmentRecord(
                    alignment_id=f'alignment_{index}',
                    hidden_goals=list(bucket['hidden_goals']),
                    required_premises=list(bucket['required_premises']),
                    visual_terms=self._top_terms(bucket['visual_terms']),
                    operator_families=self._top_terms(bucket['operator_families']),
                    support=support,
                    confidence=round(float(bucket['confidence_sum']) / float(max(1, support)), 4),
                    rationale=' '.join(dict.fromkeys(bucket['rationales']))[:240] or 'Retained multimodal alignment from verified visual-language traces.',
                )
            )
        records.sort(key=lambda item: (-item.support, -item.confidence, item.alignment_id))
        model = MultimodalAlignmentModel(records=records, trained_on_graphs=len(graphs))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'weights': model.model_dump()}, ensure_ascii=False, indent=2), encoding='utf-8')
        return MultimodalAlignmentTrainingSummary(
            output_path=str(output),
            trained_on_graphs=len(graphs),
            retained_alignment_count=len(records),
            model=model.model_dump(),
        )

    @staticmethod
    def _visual_terms(graph: StructuredMeaningGraph) -> list[str]:
        terms: list[str] = []
        for node in graph.nodes:
            if not any(tag.startswith('multimodal:vision') for tag in node.provenance):
                continue
            terms.extend(_tokenize(f"{node.id} {node.label} {' '.join(str(v) for v in node.attributes.values())}"))
        return list(dict.fromkeys(terms))[:20]

    @staticmethod
    def _top_terms(counts: dict[str, int], limit: int = 8) -> list[str]:
        items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [item[0] for item in items[:limit]]


class MultimodalAlignmentMemory:
    def __init__(self, model: MultimodalAlignmentModel | None = None, model_path: str | Path | None = None) -> None:
        self.model = model or MultimodalAlignmentModel.from_path(model_path) or MultimodalAlignmentModel()

    def enrich(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        visual_terms = set(MultimodalAlignmentTrainer._visual_terms(graph))
        if not visual_terms:
            return graph
        operator_families = {candidate.family for candidate in graph.induced_operators if any(tag.startswith('multimodal:') for tag in candidate.provenance)}
        applied: list[str] = []
        for record in self.model.records:
            term_overlap = len(visual_terms & set(record.visual_terms)) / float(len(record.visual_terms) or 1)
            operator_overlap = len(operator_families & set(record.operator_families)) / float(len(record.operator_families) or 1) if record.operator_families else 0.0
            score = (0.6 * term_overlap) + (0.25 * operator_overlap) + (0.15 * record.confidence)
            if score < self.model.min_match_score:
                continue
            for goal in record.hidden_goals:
                if goal not in graph.hidden_goals:
                    graph.hidden_goals.append(goal)
            for premise in record.required_premises:
                if premise not in graph.required_premises and premise not in graph.satisfied_premises:
                    graph.required_premises.append(premise)
            if not any(functor.name == record.functor_name for functor in graph.functor_hypotheses):
                graph.functor_hypotheses.append(
                    FunctorHypothesis(
                        name=record.functor_name,
                        source_category='visual_structure',
                        target_category='goal_preservation_logic',
                        object_map={'visual_scene': graph.hidden_goals[0] if graph.hidden_goals else 'hidden_goal'},
                        morphism_map={'HAS_EVIDENCE': 'constraint_binding', 'PART_OF': 'access_path'},
                        confidence=round(min(0.94, 0.48 + (0.2 * score)), 2),
                    )
                )
            applied.append(record.alignment_id)
        if applied:
            note = 'multimodal alignment memory: ' + ', '.join(applied[:3])
            if note not in graph.audit_trace:
                graph.audit_trace.append(note)
        return graph


def train_multimodal_alignment_from_graphs(
    graphs: Sequence[StructuredMeaningGraph],
    output_path: str | Path,
) -> MultimodalAlignmentTrainingSummary:
    return MultimodalAlignmentTrainer().train_from_graphs(graphs, output_path)

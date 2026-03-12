from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

from .basis_operators import canonicalize_basis_signature, infer_basis_operators
from .structures import FunctorHypothesis, OperatorCandidate, OperatorDecomposition, StructuredMeaningGraph


@dataclass
class RetainedOperatorRecord:
    operator_name: str
    basis_signature: list[str]
    support: int
    domain_support: int
    average_confidence: float
    verification_rate: float
    utility_score: float
    activated_support: int = 0
    activation_success_rate: float = 0.0
    retired: bool = False
    retirement_reason: str = ''
    input_types: list[str] = field(default_factory=list)
    output_type: str = 'relation_frame'
    rationale: str = ''
    source_domains: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetainedFunctorRecord:
    name: str
    source_category: str
    target_category: str
    support: int
    confidence: float
    object_map: dict[str, str] = field(default_factory=dict)
    morphism_map: dict[str, str] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetainedOperatorModel:
    records: list[RetainedOperatorRecord] = field(default_factory=list)
    functors: list[RetainedFunctorRecord] = field(default_factory=list)
    min_activation_score: float = 0.67
    trained_on_graphs: int = 0

    def model_dump(self) -> dict[str, Any]:
        return {
            'records': [item.model_dump() for item in self.records],
            'functors': [item.model_dump() for item in self.functors],
            'min_activation_score': self.min_activation_score,
            'trained_on_graphs': self.trained_on_graphs,
        }

    @classmethod
    def from_path(cls, path: str | Path | None) -> RetainedOperatorModel | None:
        if not path:
            return None
        payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if isinstance(payload, dict) and 'weights' in payload:
            payload = payload['weights']
        return cls(
            records=[RetainedOperatorRecord(**item) for item in payload.get('records', [])],
            functors=[RetainedFunctorRecord(**item) for item in payload.get('functors', [])],
            min_activation_score=float(payload.get('min_activation_score', 0.67)),
            trained_on_graphs=int(payload.get('trained_on_graphs', 0)),
        )


@dataclass
class RetainedOperatorTrainingSummary:
    output_path: str
    trained_on_graphs: int
    retained_operator_count: int
    retained_functor_count: int
    model: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class RetainedOperatorTrainer:
    def train_from_graphs(
        self,
        graphs: Sequence[StructuredMeaningGraph],
        output_path: str | Path,
        min_support: int = 2,
        min_verification_rate: float = 0.55,
    ) -> RetainedOperatorTrainingSummary:
        grouped: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        functor_groups: dict[tuple[str, str, str], dict[str, Any]] = {}
        graphs = list(graphs)
        for graph in graphs:
            operator_lookup = {item.name: item for item in graph.induced_operators}
            verification_score = 1.0
            if graph.operator_execution is not None and graph.operator_execution.composition_score > 0.0:
                verification_score = float(graph.operator_execution.composition_score)
            retained_activation_names = {
                item.name for item in graph.induced_operators if 'retained_operator_algebra' in item.provenance
            }
            for item in graph.operator_decompositions:
                basis = canonicalize_basis_signature(item.basis_operators)
                if not basis:
                    continue
                key = (item.operator_name, tuple(basis))
                bucket = grouped.setdefault(
                    key,
                    {
                        'operator_name': item.operator_name,
                        'basis_signature': basis,
                        'support': 0,
                        'domains': set(),
                        'confidence_sum': 0.0,
                        'verification_sum': 0.0,
                        'input_types': set(),
                        'output_type': 'relation_frame',
                        'rationales': [],
                        'activated_support': 0,
                        'activated_success_sum': 0.0,
                    },
                )
                bucket['support'] += 1
                bucket['domains'].add(graph.domain or 'general')
                bucket['confidence_sum'] += float(item.confidence)
                bucket['verification_sum'] += verification_score
                candidate = operator_lookup.get(item.operator_name)
                if candidate is not None:
                    bucket['input_types'].update(candidate.input_types)
                    if candidate.output_type:
                        bucket['output_type'] = candidate.output_type
                if item.rationale:
                    bucket['rationales'].append(item.rationale)
                if item.operator_name in retained_activation_names or str(item.rationale).startswith('retained algebra:'):
                    bucket['activated_support'] += 1
                    bucket['activated_success_sum'] += verification_score
            for functor in graph.functor_hypotheses:
                key = (functor.name, functor.source_category, functor.target_category)
                bucket = functor_groups.setdefault(
                    key,
                    {
                        'name': functor.name,
                        'source_category': functor.source_category,
                        'target_category': functor.target_category,
                        'support': 0,
                        'confidence_sum': 0.0,
                        'object_map': dict(functor.object_map),
                        'morphism_map': dict(functor.morphism_map),
                    },
                )
                bucket['support'] += 1
                bucket['confidence_sum'] += float(functor.confidence)
                bucket['object_map'].update(functor.object_map)
                bucket['morphism_map'].update(functor.morphism_map)

        max_support = max((item['support'] for item in grouped.values()), default=1)
        records: list[RetainedOperatorRecord] = []
        for bucket in grouped.values():
            support = int(bucket['support'])
            verification_rate = round(float(bucket['verification_sum']) / float(max(1, support)), 4)
            if support < min_support or verification_rate < min_verification_rate:
                continue
            average_confidence = round(float(bucket['confidence_sum']) / float(max(1, support)), 4)
            domain_support = len(bucket['domains'])
            activated_support = int(bucket.get('activated_support', 0))
            activation_success_rate = round(float(bucket.get('activated_success_sum', 0.0)) / float(max(1, activated_support)), 4) if activated_support else verification_rate
            utility = self._utility_score(
                support=support,
                max_support=max_support,
                domain_support=domain_support,
                average_confidence=average_confidence,
                verification_rate=verification_rate,
                activation_success_rate=activation_success_rate,
            )
            retired = activated_support > 0 and activation_success_rate < 0.52
            retirement_reason = 'low_activation_success' if retired else ''
            records.append(
                RetainedOperatorRecord(
                    operator_name=str(bucket['operator_name']),
                    basis_signature=list(bucket['basis_signature']),
                    support=support,
                    domain_support=domain_support,
                    average_confidence=average_confidence,
                    verification_rate=verification_rate,
                    utility_score=utility,
                    activated_support=activated_support,
                    activation_success_rate=activation_success_rate,
                    retired=retired,
                    retirement_reason=retirement_reason,
                    input_types=sorted(bucket['input_types']),
                    output_type=str(bucket['output_type']),
                    rationale=' '.join(dict.fromkeys(bucket['rationales']))[:240] or 'Retained from repeatedly verified operator decompositions.',
                    source_domains=sorted(bucket['domains']),
                )
            )

        functors: list[RetainedFunctorRecord] = []
        for bucket in functor_groups.values():
            support = int(bucket['support'])
            if support < min_support:
                continue
            confidence = round(float(bucket['confidence_sum']) / float(max(1, support)), 4)
            functors.append(
                RetainedFunctorRecord(
                    name=str(bucket['name']),
                    source_category=str(bucket['source_category']),
                    target_category=str(bucket['target_category']),
                    support=support,
                    confidence=confidence,
                    object_map=dict(bucket['object_map']),
                    morphism_map=dict(bucket['morphism_map']),
                )
            )

        records.sort(key=lambda item: (-item.utility_score, -item.support, item.operator_name))
        functors.sort(key=lambda item: (-item.support, -item.confidence, item.name))
        model = RetainedOperatorModel(records=records, functors=functors, trained_on_graphs=len(graphs))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'weights': model.model_dump()}, ensure_ascii=False, indent=2), encoding='utf-8')
        return RetainedOperatorTrainingSummary(
            output_path=str(output),
            trained_on_graphs=len(graphs),
            retained_operator_count=len(records),
            retained_functor_count=len(functors),
            model=model.model_dump(),
        )

    @staticmethod
    def _utility_score(
        support: int,
        max_support: int,
        domain_support: int,
        average_confidence: float,
        verification_rate: float,
        activation_success_rate: float,
    ) -> float:
        support_score = support / float(max(1, max_support))
        transfer_score = min(1.0, domain_support / 3.0)
        return round((0.25 * support_score) + (0.2 * transfer_score) + (0.15 * average_confidence) + (0.2 * verification_rate) + (0.2 * activation_success_rate), 4)


class RetainedOperatorAlgebra:
    def __init__(self, model: RetainedOperatorModel | None = None, model_path: str | Path | None = None) -> None:
        self.model = model or RetainedOperatorModel.from_path(model_path) or RetainedOperatorModel()

    def enrich(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        available = set(infer_basis_operators(graph))
        if graph.source_context.strip():
            available.add('DOCUMENT_CONTEXT')
        if any(any(tag.startswith('multimodal:vision') for tag in node.provenance) for node in graph.nodes):
            available.add('VISUAL_STRUCTURE')
        existing_names = {item.operator_name for item in graph.operator_decompositions}
        existing_candidates = {item.name for item in graph.induced_operators}
        retired_skips = 0
        for record in self.model.records:
            if record.retired:
                retired_skips += 1
                continue
            overlap = len(set(record.basis_signature) & available) / float(len(record.basis_signature) or 1)
            if overlap < self.model.min_activation_score:
                continue
            if record.operator_name not in existing_names:
                graph.operator_decompositions.append(
                    OperatorDecomposition(
                        operator_name=record.operator_name,
                        basis_operators=list(record.basis_signature),
                        rationale=f"retained algebra: {record.rationale}",
                        confidence=round(min(0.96, 0.48 + (0.2 * overlap) + (0.2 * record.utility_score)), 2),
                    )
                )
                existing_names.add(record.operator_name)
            if record.operator_name not in existing_candidates:
                graph.induced_operators.append(
                    OperatorCandidate(
                        name=record.operator_name,
                        family='RETAINED_OPERATOR',
                        arity=max(1, len(record.input_types) or 1),
                        input_types=list(record.input_types) or [graph.intent],
                        output_type=record.output_type,
                        description='Retained higher-order operator recovered from verified operator-algebra training traces.',
                        examples=[', '.join(record.basis_signature[:3])],
                        confidence=round(min(0.92, 0.45 + (0.18 * record.utility_score)), 2),
                        provenance=['retained_operator_algebra'],
                    )
                )
                existing_candidates.add(record.operator_name)
        functor_names = {item.name for item in graph.functor_hypotheses}
        for functor in self.model.functors:
            if functor.name in functor_names:
                continue
            graph.functor_hypotheses.append(
                FunctorHypothesis(
                    name=functor.name,
                    source_category=functor.source_category,
                    target_category=functor.target_category,
                    object_map=dict(functor.object_map),
                    morphism_map=dict(functor.morphism_map),
                    confidence=round(min(0.95, functor.confidence + 0.05), 2),
                )
            )
            functor_names.add(functor.name)
        if self.model.records:
            note = f"retained algebra: activated {len([item for item in self.model.records if item.operator_name in existing_names])} retained operator priors"
            if note not in graph.audit_trace:
                graph.audit_trace.append(note)
            if retired_skips:
                retired_note = f"retained algebra: skipped {retired_skips} retired operator priors"
                if retired_note not in graph.audit_trace:
                    graph.audit_trace.append(retired_note)
        return graph

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .pipeline import StructuredMeaningPipeline


@dataclass
class OperatorAlgebraEvalCase:
    query: str
    expected_decompositions: list[str]
    expected_functors: list[str]


@dataclass
class OperatorAlgebraEvalSummary:
    num_cases: int
    decomposition_recall: float
    functor_recall: float
    results: list[dict[str, Any]]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class OperatorAlgebraEvaluator:
    def __init__(self, pipeline: StructuredMeaningPipeline | None = None) -> None:
        self.pipeline = pipeline or StructuredMeaningPipeline(mode='heuristic')

    @staticmethod
    def load_cases(path: str | Path) -> list[OperatorAlgebraEvalCase]:
        rows: list[OperatorAlgebraEvalCase] = []
        for raw in Path(path).read_text(encoding='utf-8-sig').splitlines():
            if not raw.strip():
                continue
            payload = json.loads(raw)
            rows.append(
                OperatorAlgebraEvalCase(
                    query=str(payload.get('query', '')),
                    expected_decompositions=[str(item) for item in payload.get('expected_decompositions', [])],
                    expected_functors=[str(item) for item in payload.get('expected_functors', [])],
                )
            )
        return rows

    def evaluate(self, cases: Sequence[OperatorAlgebraEvalCase]) -> OperatorAlgebraEvalSummary:
        if not cases:
            return OperatorAlgebraEvalSummary(0, 0.0, 0.0, [])
        decomposition_scores: list[float] = []
        functor_scores: list[float] = []
        results: list[dict[str, Any]] = []
        for case in cases:
            graph = self.pipeline.run(case.query)
            decomposition_names = {item.operator_name for item in graph.operator_decompositions}
            functor_names = {item.name for item in graph.functor_hypotheses}
            decomposition_recall = self._recall(decomposition_names, case.expected_decompositions)
            functor_recall = self._recall(functor_names, case.expected_functors)
            decomposition_scores.append(decomposition_recall)
            functor_scores.append(functor_recall)
            results.append({
                'query': case.query,
                'decomposition_recall': decomposition_recall,
                'functor_recall': functor_recall,
                'predicted_decompositions': sorted(decomposition_names),
                'predicted_functors': sorted(functor_names),
            })
        total = float(len(cases))
        return OperatorAlgebraEvalSummary(
            num_cases=len(cases),
            decomposition_recall=round(sum(decomposition_scores) / total, 4),
            functor_recall=round(sum(functor_scores) / total, 4),
            results=results,
        )

    @staticmethod
    def _recall(predicted: set[str], expected: Sequence[str]) -> float:
        gold = {str(item) for item in expected}
        if not gold:
            return 1.0
        return len(predicted & gold) / float(len(gold))

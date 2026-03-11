from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .eval import VlsoGroundedEvaluator, VlsoEvalSummary
from .reasoner import VLSOReasoner


@dataclass
class VlsoStoreComparisonSummary:
    input_path: str
    primary_concept_store: str
    primary_operator_store: str
    compare_concept_store: str
    compare_operator_store: str
    primary_summary: dict[str, Any]
    compare_summary: dict[str, Any]
    deltas: dict[str, float]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class VlsoReviewImpactEvaluator:
    def __init__(
        self,
        mode: str = 'heuristic',
        answer_mode: str = 'structured',
        affordance_weights_path: str | None = None,
    ) -> None:
        self.mode = mode
        self.answer_mode = answer_mode
        self.affordance_weights_path = affordance_weights_path

    def compare_stores(
        self,
        input_path: str | Path,
        primary_concept_store: str | Path | None,
        primary_operator_store: str | Path | None,
        compare_concept_store: str | Path | None,
        compare_operator_store: str | Path | None,
    ) -> VlsoStoreComparisonSummary:
        cases = VlsoGroundedEvaluator.load_cases(input_path)
        primary = self._evaluate(cases, primary_concept_store, primary_operator_store)
        compare = self._evaluate(cases, compare_concept_store, compare_operator_store)
        deltas = {
            'object_recall_delta': round(compare.object_recall - primary.object_recall, 4),
            'relation_recall_delta': round(compare.relation_recall - primary.relation_recall, 4),
            'answer_term_recall_delta': round(compare.answer_term_recall - primary.answer_term_recall, 4),
            'grounded_answer_accuracy_delta': round(compare.grounded_answer_accuracy - primary.grounded_answer_accuracy, 4),
        }
        return VlsoStoreComparisonSummary(
            input_path=str(input_path),
            primary_concept_store=str(primary_concept_store or ''),
            primary_operator_store=str(primary_operator_store or ''),
            compare_concept_store=str(compare_concept_store or ''),
            compare_operator_store=str(compare_operator_store or ''),
            primary_summary=primary.model_dump(),
            compare_summary=compare.model_dump(),
            deltas=deltas,
        )

    def _evaluate(
        self,
        cases: list,
        concept_store_path: str | Path | None,
        operator_store_path: str | Path | None,
    ) -> VlsoEvalSummary:
        reasoner = VLSOReasoner(
            mode=self.mode,
            answer_mode=self.answer_mode,
            affordance_weights_path=self.affordance_weights_path,
            concept_store_path=str(concept_store_path) if concept_store_path else None,
            operator_store_path=str(operator_store_path) if operator_store_path else None,
        )
        return VlsoGroundedEvaluator(reasoner).evaluate_cases(cases)

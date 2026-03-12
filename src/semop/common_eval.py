from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .cp_parser_eval import CpLearnedParser, CpParserEvaluator
from .pipeline import StructuredMeaningPipeline
from .premise_eval import HiddenPremiseEvaluator
from .vlso.eval import VlsoGroundedEvaluator
from .vlso.reasoner import VLSOReasoner


@dataclass
class SemOpEvalSnapshot:
    hidden_premise: dict[str, Any] | None = None
    cp_parser: dict[str, Any] | None = None
    cp_hidden_constraints: dict[str, Any] | None = None
    vlso_grounded: dict[str, Any] | None = None
    vlso_real_image: dict[str, Any] | None = None
    operator_premise_support: dict[str, Any] | None = None
    operator_compiler: dict[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class SemOpCommonEvaluator:
    def __init__(self) -> None:
        self.hidden_premise_evaluator = HiddenPremiseEvaluator(StructuredMeaningPipeline(mode='heuristic'))
        self.cp_parser_evaluator = CpParserEvaluator()
        self.vlso_reasoner = VLSOReasoner()

    def evaluate(
        self,
        hidden_premise_cases: list | None = None,
        cp_examples: list | None = None,
        cp_hidden_examples: list | None = None,
        cp_model: CpLearnedParser | None = None,
        vlso_cases: list | None = None,
        vlso_real_image_cases: list | None = None,
        cp_mode: str = 'heuristic',
    ) -> SemOpEvalSnapshot:
        snapshot = SemOpEvalSnapshot()
        if hidden_premise_cases is not None:
            summary = self.hidden_premise_evaluator.evaluate(hidden_premise_cases).__dict__
            snapshot.hidden_premise = summary
            snapshot.operator_premise_support = {
                'operator_supported_premise_recall': summary.get('operator_supported_premise_recall', 0.0),
                'critical_premise_recall': summary.get('critical_premise_recall', 0.0),
                'unsupported_premise_precision': summary.get('unsupported_premise_precision', 0.0),
            }
            snapshot.operator_compiler = {
                'compiler_alignment_score': summary.get('compiler_alignment_score', 0.0),
                'basis_operator_nonempty_rate': summary.get('basis_operator_nonempty_rate', 0.0),
            }
        if cp_examples is not None:
            if cp_mode == 'compare':
                snapshot.cp_parser = self.cp_parser_evaluator.compare_examples(cp_examples, model=cp_model).model_dump()
            else:
                predictor = 'model' if cp_mode == 'model' else 'heuristic'
                snapshot.cp_parser = self.cp_parser_evaluator.evaluate_examples(cp_examples, predictor=predictor, model=cp_model).model_dump()
        if cp_hidden_examples is not None:
            if cp_mode == 'compare':
                snapshot.cp_hidden_constraints = self.cp_parser_evaluator.compare_examples(cp_hidden_examples, model=cp_model).model_dump()
            else:
                predictor = 'model' if cp_mode == 'model' else 'heuristic'
                snapshot.cp_hidden_constraints = self.cp_parser_evaluator.evaluate_examples(cp_hidden_examples, predictor=predictor, model=cp_model).model_dump()
        if vlso_cases is not None:
            vlso_summary = VlsoGroundedEvaluator(self.vlso_reasoner).evaluate_cases(vlso_cases).model_dump()
            snapshot.vlso_grounded = vlso_summary
            if snapshot.operator_premise_support is None:
                snapshot.operator_premise_support = {}
            snapshot.operator_premise_support['vlso_operator_recall'] = vlso_summary.get('operator_recall', 0.0)
            snapshot.operator_premise_support['vlso_operator_binding_recall'] = vlso_summary.get('operator_binding_recall', 0.0)
            snapshot.operator_premise_support['vlso_operator_premise_support'] = vlso_summary.get('operator_premise_support', 0.0)
        if vlso_real_image_cases is not None:
            vlso_real_summary = VlsoGroundedEvaluator(self.vlso_reasoner).evaluate_cases(vlso_real_image_cases).model_dump()
            snapshot.vlso_real_image = vlso_real_summary
            if snapshot.operator_premise_support is None:
                snapshot.operator_premise_support = {}
            snapshot.operator_premise_support['vlso_real_image_operator_recall'] = vlso_real_summary.get('operator_recall', 0.0)
            snapshot.operator_premise_support['vlso_real_image_operator_binding_recall'] = vlso_real_summary.get('operator_binding_recall', 0.0)
            snapshot.operator_premise_support['vlso_real_image_operator_premise_support'] = vlso_real_summary.get('operator_premise_support', 0.0)
        return snapshot

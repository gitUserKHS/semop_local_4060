from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .pipeline import StructuredMeaningPipeline
from .review_queue import severity_weight


@dataclass
class HiddenPremiseEvalCase:
    query: str
    expected_hidden_goals: List[str]
    expected_required_premises: List[str]
    expected_satisfied_premises: List[str]
    expected_missing_premises: List[str]
    expected_risky_actions: List[str]
    forbidden_premises: List[str]
    expected_clarification_needed: bool
    expected_clarification_score: float | None = None
    expected_support_operators: List[str] | None = None
    domain: str = 'general'
    scenario: str = 'qa'
    severity: str = 'medium'
    case_weight: float = 0.0


@dataclass
class HiddenPremiseEvalSummary:
    num_cases: int
    critical_premise_recall: float
    hidden_goal_recall: float
    goal_preservation_accuracy: float
    unsupported_premise_precision: float
    clarification_accuracy: float
    requirement_state_accuracy: float
    clarification_score_mae: float
    operator_supported_premise_recall: float
    compiler_alignment_score: float
    basis_operator_nonempty_rate: float


class HiddenPremiseEvaluator:
    def __init__(self, pipeline: StructuredMeaningPipeline | None = None) -> None:
        self.pipeline = pipeline or StructuredMeaningPipeline(mode='heuristic')

    def evaluate(self, cases: Sequence[HiddenPremiseEvalCase]) -> HiddenPremiseEvalSummary:
        if not cases:
            return HiddenPremiseEvalSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        premise_scores: List[float] = []
        goal_scores: List[float] = []
        preservation_scores: List[float] = []
        unsupported_scores: List[float] = []
        clarification_scores: List[float] = []
        requirement_scores: List[float] = []
        clarification_mae_terms: List[float] = []
        clarification_mae_weights: List[float] = []
        operator_support_scores: List[float] = []
        operator_support_weights: List[float] = []
        compiler_scores: List[float] = []
        basis_nonempty_scores: List[float] = []
        weights: List[float] = []
        for case in cases:
            graph = self.pipeline.run(case.query)
            weight = self._case_weight(case)
            weights.append(weight)
            premise_scores.append(self._recall(graph.required_premises, case.expected_required_premises))
            goal_scores.append(self._recall(graph.hidden_goals, case.expected_hidden_goals))
            predicted_risky = [check.action for check in graph.goal_preservation_checks if check.status in {'risk_high', 'invalid'}]
            preservation_scores.append(self._recall(predicted_risky, case.expected_risky_actions))
            unsupported_scores.append(self._unsupported_precision(graph.required_premises, case.forbidden_premises))
            clarification_scores.append(1.0 if bool(graph.clarification_needed) == bool(case.expected_clarification_needed) else 0.0)
            requirement_scores.append(self._requirement_state_accuracy(graph.satisfied_premises, graph.missing_premises, case.expected_satisfied_premises, case.expected_missing_premises))
            if case.expected_clarification_score is not None:
                clarification_mae_terms.append(abs(float(graph.clarification_score) - float(case.expected_clarification_score)))
                clarification_mae_weights.append(weight)
            if case.expected_support_operators is not None:
                supported = {item.name for item in graph.induced_operators} | {item.operator_name for item in graph.operator_decompositions}
                operator_support_scores.append(self._recall(supported, case.expected_support_operators))
                operator_support_weights.append(weight)
            report = graph.operator_execution
            compiler_scores.append(float(report.compiler_alignment_score) if report is not None else 0.0)
            basis_nonempty_scores.append(1.0 if report is not None and report.basis_operator_hits else 0.0)
        clarification_score_mae = round(self._weighted_average(clarification_mae_terms, clarification_mae_weights), 4) if clarification_mae_terms else 0.0
        operator_supported_premise_recall = round(self._weighted_average(operator_support_scores, operator_support_weights), 4) if operator_support_scores else 0.0
        return HiddenPremiseEvalSummary(
            num_cases=len(cases),
            critical_premise_recall=round(self._weighted_average(premise_scores, weights), 4),
            hidden_goal_recall=round(self._weighted_average(goal_scores, weights), 4),
            goal_preservation_accuracy=round(self._weighted_average(preservation_scores, weights), 4),
            unsupported_premise_precision=round(self._weighted_average(unsupported_scores, weights), 4),
            clarification_accuracy=round(self._weighted_average(clarification_scores, weights), 4),
            requirement_state_accuracy=round(self._weighted_average(requirement_scores, weights), 4),
            clarification_score_mae=clarification_score_mae,
            operator_supported_premise_recall=operator_supported_premise_recall,
            compiler_alignment_score=round(self._weighted_average(compiler_scores, weights), 4),
            basis_operator_nonempty_rate=round(self._weighted_average(basis_nonempty_scores, weights), 4),
        )

    @staticmethod
    def _case_weight(case: HiddenPremiseEvalCase) -> float:
        base = float(case.case_weight or 0.0)
        if base > 0.0:
            return base
        return severity_weight(case.severity, case.domain, case.scenario)

    @staticmethod
    def _weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
        if not values:
            return 0.0
        total_weight = sum(float(weight) for weight in weights) if weights else 0.0
        if total_weight <= 0.0:
            return sum(values) / float(len(values))
        return sum(float(value) * float(weight) for value, weight in zip(values, weights)) / float(total_weight)

    @staticmethod
    def _recall(predicted: Iterable[str], gold: Iterable[str]) -> float:
        predicted_set = set(predicted)
        gold_list = list(gold)
        if not gold_list:
            return 1.0
        gold_set = set(gold_list)
        return len(predicted_set & gold_set) / float(len(gold_set))

    @staticmethod
    def _unsupported_precision(predicted: Iterable[str], forbidden: Iterable[str]) -> float:
        predicted_set = set(predicted)
        forbidden_set = set(forbidden)
        if not predicted_set:
            return 1.0
        unsupported = len(predicted_set & forbidden_set)
        return max(0.0, (len(predicted_set) - unsupported) / float(len(predicted_set)))

    @staticmethod
    def _requirement_state_accuracy(predicted_satisfied: Iterable[str], predicted_missing: Iterable[str], gold_satisfied: Iterable[str], gold_missing: Iterable[str]) -> float:
        predicted = {('satisfied', item) for item in predicted_satisfied} | {('missing', item) for item in predicted_missing}
        gold = {('satisfied', item) for item in gold_satisfied} | {('missing', item) for item in gold_missing}
        if not gold:
            return 1.0 if not predicted else 0.0
        return len(predicted & gold) / float(len(gold))



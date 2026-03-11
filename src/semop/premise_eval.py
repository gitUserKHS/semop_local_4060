from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .pipeline import StructuredMeaningPipeline


@dataclass
class HiddenPremiseEvalCase:
    query: str
    expected_hidden_goals: List[str]
    expected_required_premises: List[str]
    expected_risky_actions: List[str]


@dataclass
class HiddenPremiseEvalSummary:
    num_cases: int
    critical_premise_recall: float
    hidden_goal_recall: float
    goal_preservation_accuracy: float


class HiddenPremiseEvaluator:
    def __init__(self, pipeline: StructuredMeaningPipeline | None = None) -> None:
        self.pipeline = pipeline or StructuredMeaningPipeline(mode='heuristic')

    def evaluate(self, cases: Sequence[HiddenPremiseEvalCase]) -> HiddenPremiseEvalSummary:
        if not cases:
            return HiddenPremiseEvalSummary(0, 0.0, 0.0, 0.0)
        premise_scores: List[float] = []
        goal_scores: List[float] = []
        preservation_scores: List[float] = []
        for case in cases:
            graph = self.pipeline.run(case.query)
            premise_scores.append(self._recall(graph.required_premises, case.expected_required_premises))
            goal_scores.append(self._recall(graph.hidden_goals, case.expected_hidden_goals))
            predicted_risky = [check.action for check in graph.goal_preservation_checks if check.status in {'risk_high', 'invalid'}]
            preservation_scores.append(self._recall(predicted_risky, case.expected_risky_actions))
        total = float(len(cases))
        return HiddenPremiseEvalSummary(
            num_cases=len(cases),
            critical_premise_recall=round(sum(premise_scores) / total, 4),
            hidden_goal_recall=round(sum(goal_scores) / total, 4),
            goal_preservation_accuracy=round(sum(preservation_scores) / total, 4),
        )

    @staticmethod
    def _recall(predicted: Iterable[str], gold: Iterable[str]) -> float:
        predicted_set = set(predicted)
        gold_list = list(gold)
        if not gold_list:
            return 1.0
        gold_set = set(gold_list)
        return len(predicted_set & gold_set) / float(len(gold_set))

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List

from .baseline_runner import BaselineRunner
from .domain_copilot import CopilotRequest, DomainCopilot


@dataclass
class LabeledOpsCase:
    id: str
    domain: str
    scenario: str
    query: str
    context: str
    expected_relations: List[str]
    expected_answer_terms: List[str]
    forbidden_phrases: List[str]
    expected_clarification: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LabeledOpsCaseResult:
    case_id: str
    system: str
    relation_recall: float
    answer_term_recall: float
    forbidden_phrase_hit: bool
    clarification_alignment: float
    answer_text: str

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


class LabeledOpsEvaluator:
    def __init__(self, copilot: DomainCopilot | None = None, baseline_name: str = "lexical_rag", baseline_config_path: str | None = None):
        self.copilot = copilot or DomainCopilot(mode="heuristic")
        self.baseline = BaselineRunner(baseline_name, config_path=baseline_config_path)
        self.baseline_name = baseline_name

    def evaluate_case(self, case: LabeledOpsCase) -> Dict[str, LabeledOpsCaseResult]:
        request = CopilotRequest(
            query=case.query,
            context=case.context,
            domain=case.domain,
            scenario=case.scenario,
        )
        semop = self.copilot.run(request)
        baseline = self.baseline.answer(case.query, case.context, domain=case.domain)
        return {
            "semop": self._score_semop(case, semop),
            self.baseline_name: self._score_baseline(case, baseline),
        }

    @staticmethod
    def _score_semop(case: LabeledOpsCase, result) -> LabeledOpsCaseResult:
        relations = {edge.relation for edge in result.graph.edges}
        relation_recall = _recall(case.expected_relations, relations)
        answer_text = result.answer_text.lower()
        answer_term_recall = _recall(case.expected_answer_terms, answer_text)
        blocked_text = " ".join(result.graph.invalid_advice).lower()
        forbidden_phrase_hit = any(
            phrase.lower() in answer_text and phrase.lower() not in blocked_text
            for phrase in case.forbidden_phrases
        )
        clarification_alignment = _clarification_alignment(case.expected_clarification, result.kpis.clarification_need_rate)
        return LabeledOpsCaseResult(
            case_id=case.id,
            system="semop",
            relation_recall=relation_recall,
            answer_term_recall=answer_term_recall,
            forbidden_phrase_hit=forbidden_phrase_hit,
            clarification_alignment=clarification_alignment,
            answer_text=result.answer_text,
        )

    def _score_baseline(self, case: LabeledOpsCase, result) -> LabeledOpsCaseResult:
        answer_text = result.answer_text.lower()
        relation_recall = _recall(case.expected_relations, set())
        answer_term_recall = _recall(case.expected_answer_terms, answer_text)
        forbidden_phrase_hit = any(phrase.lower() in answer_text for phrase in case.forbidden_phrases)
        clarification_alignment = _clarification_alignment(case.expected_clarification, 0.0)
        return LabeledOpsCaseResult(
            case_id=case.id,
            system=self.baseline_name,
            relation_recall=relation_recall,
            answer_term_recall=answer_term_recall,
            forbidden_phrase_hit=forbidden_phrase_hit,
            clarification_alignment=clarification_alignment,
            answer_text=result.answer_text,
        )



def load_labeled_ops_cases(path: str) -> List[LabeledOpsCase]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return [LabeledOpsCase(**json.loads(line)) for line in handle if line.strip()]


def _recall(expected: List[str], observed: Any) -> float:
    if not expected:
        return 1.0
    if isinstance(observed, set):
        score = sum(1 for item in expected if item in observed) / len(expected)
        return round(score, 2)
    observed_text = str(observed).lower()
    score = sum(1 for item in expected if item.lower() in observed_text) / len(expected)
    return round(score, 2)


def _clarification_alignment(expected: bool, observed_rate: float) -> float:
    if expected:
        return 1.0 if observed_rate >= 0.4 else 0.0
    return 1.0 if observed_rate < 0.4 else 0.0

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .common_eval import SemOpEvalSnapshot


@dataclass
class ProgressAxis:
    name: str
    score: float
    rationale: list[str]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorIntelligenceProgress:
    operator_architecture: ProgressAxis
    premise_reasoning: ProgressAxis
    shared_world_model: ProgressAxis
    cp_structuring: ProgressAxis
    raw_visual_reasoning: ProgressAxis
    general_operator_transfer: ProgressAxis
    research_architecture_overall: float
    robust_general_intelligence_overall: float

    def model_dump(self) -> dict[str, Any]:
        return {
            'operator_architecture': self.operator_architecture.model_dump(),
            'premise_reasoning': self.premise_reasoning.model_dump(),
            'shared_world_model': self.shared_world_model.model_dump(),
            'cp_structuring': self.cp_structuring.model_dump(),
            'raw_visual_reasoning': self.raw_visual_reasoning.model_dump(),
            'general_operator_transfer': self.general_operator_transfer.model_dump(),
            'research_architecture_overall': self.research_architecture_overall,
            'robust_general_intelligence_overall': self.robust_general_intelligence_overall,
        }


class OperatorIntelligenceProgressEstimator:
    """Estimate current progress toward the operator-intelligence target."""

    def estimate(self, snapshot: SemOpEvalSnapshot) -> OperatorIntelligenceProgress:
        hidden = snapshot.hidden_premise or {}
        cp = snapshot.cp_parser or {}
        vlso = snapshot.vlso_grounded or {}
        support = snapshot.operator_premise_support or {}
        compiler = snapshot.operator_compiler or {}

        hidden_coverage = self._coverage(hidden.get('num_cases', 0), target=120)
        cp_coverage = self._coverage(cp.get('num_examples', 0), target=120)
        vlso_coverage = self._coverage(vlso.get('num_cases', 0), target=80)

        operator_architecture = self._axis(
            'operator_architecture',
            [
                0.22 * self._value(support, 'operator_supported_premise_recall'),
                0.18 * self._value(compiler, 'compiler_alignment_score'),
                0.10 * self._value(compiler, 'basis_operator_nonempty_rate'),
                0.18 * self._value(vlso, 'operator_recall'),
                0.17 * self._value(vlso, 'operator_binding_recall'),
                0.075 * self._value(cp, 'frame_jaccard'),
                0.075 * self._value(cp, 'operator_jaccard'),
            ],
            coverage=max(hidden_coverage, cp_coverage, vlso_coverage),
            rationale=[
                f"hidden premise support={self._value(support, 'operator_supported_premise_recall'):.3f}",
                f"compiler alignment={self._value(compiler, 'compiler_alignment_score'):.3f}",
                f"vlso operator recall={self._value(vlso, 'operator_recall'):.3f}",
            ],
        )

        premise_reasoning = self._axis(
            'premise_reasoning',
            [
                0.22 * self._value(hidden, 'critical_premise_recall'),
                0.18 * self._value(hidden, 'hidden_goal_recall'),
                0.22 * self._value(hidden, 'goal_preservation_accuracy'),
                0.12 * self._value(hidden, 'unsupported_premise_precision'),
                0.10 * self._value(hidden, 'clarification_accuracy'),
                0.08 * self._inverse_mae(hidden, 'clarification_score_mae'),
                0.08 * self._value(hidden, 'compiler_alignment_score'),
            ],
            coverage=hidden_coverage,
            rationale=[
                f"critical premise recall={self._value(hidden, 'critical_premise_recall'):.3f}",
                f"goal preservation={self._value(hidden, 'goal_preservation_accuracy'):.3f}",
                f"unsupported premise precision={self._value(hidden, 'unsupported_premise_precision'):.3f}",
            ],
        )

        shared_world_model = self._axis(
            'shared_world_model',
            [
                0.16 * self._value(vlso, 'relation_recall'),
                0.16 * self._value(vlso, 'operator_recall'),
                0.16 * self._value(vlso, 'operator_binding_recall'),
                0.16 * self._value(vlso, 'operator_premise_support'),
                0.12 * self._value(hidden, 'operator_supported_premise_recall'),
                0.12 * self._value(hidden, 'goal_preservation_accuracy'),
                0.12 * self._value(compiler, 'compiler_alignment_score'),
            ],
            coverage=min(hidden_coverage, vlso_coverage) if hidden and vlso else max(hidden_coverage, vlso_coverage),
            rationale=[
                f"vlso relation recall={self._value(vlso, 'relation_recall'):.3f}",
                f"vlso operator-premise support={self._value(vlso, 'operator_premise_support'):.3f}",
                f"compiler alignment={self._value(compiler, 'compiler_alignment_score'):.3f}",
            ],
        )

        cp_structuring = self._axis(
            'cp_structuring',
            [
                0.24 * self._value(cp, 'frame_jaccard'),
                0.2 * self._value(cp, 'operator_jaccard'),
                0.18 * self._value(cp, 'domain_jaccard'),
                0.18 * self._value(cp, 'goal_jaccard'),
                0.1 * self._value(cp, 'algorithm_exact_match'),
                0.1 * self._value(cp, 'reasoning_nonempty_rate'),
            ],
            coverage=cp_coverage,
            rationale=[
                f"cp frame jaccard={self._value(cp, 'frame_jaccard'):.3f}",
                f"cp operator jaccard={self._value(cp, 'operator_jaccard'):.3f}",
                f"cp algorithm match={self._value(cp, 'algorithm_exact_match'):.3f}",
            ],
        )

        raw_visual_reasoning = self._axis(
            'raw_visual_reasoning',
            [
                0.18 * self._value(vlso, 'object_recall'),
                0.18 * self._value(vlso, 'relation_recall'),
                0.2 * self._value(vlso, 'grounded_answer_accuracy'),
                0.16 * self._value(vlso, 'operator_recall'),
                0.14 * self._value(vlso, 'operator_binding_recall'),
                0.14 * self._value(vlso, 'operator_premise_support'),
            ],
            coverage=vlso_coverage,
            rationale=[
                f"vlso object recall={self._value(vlso, 'object_recall'):.3f}",
                f"vlso grounded accuracy={self._value(vlso, 'grounded_answer_accuracy'):.3f}",
                f"vlso operator binding={self._value(vlso, 'operator_binding_recall'):.3f}",
            ],
        )

        general_operator_transfer = self._axis(
            'general_operator_transfer',
            [
                0.24 * operator_architecture.score,
                0.22 * premise_reasoning.score,
                0.2 * shared_world_model.score,
                0.18 * cp_structuring.score,
                0.16 * raw_visual_reasoning.score,
            ],
            coverage=min(hidden_coverage, cp_coverage, vlso_coverage),
            rationale=[
                f"operator architecture={operator_architecture.score:.3f}",
                f"premise reasoning={premise_reasoning.score:.3f}",
                f"shared world model={shared_world_model.score:.3f}",
            ],
        )

        research_overall = round(
            0.22 * operator_architecture.score
            + 0.24 * premise_reasoning.score
            + 0.2 * shared_world_model.score
            + 0.18 * cp_structuring.score
            + 0.16 * raw_visual_reasoning.score,
            4,
        )
        robust_overall = round(
            0.18 * operator_architecture.score
            + 0.24 * premise_reasoning.score
            + 0.22 * shared_world_model.score
            + 0.12 * cp_structuring.score
            + 0.24 * raw_visual_reasoning.score,
            4,
        )
        robust_overall = round(min(research_overall, robust_overall * 0.85 + general_operator_transfer.score * 0.15), 4)

        return OperatorIntelligenceProgress(
            operator_architecture=operator_architecture,
            premise_reasoning=premise_reasoning,
            shared_world_model=shared_world_model,
            cp_structuring=cp_structuring,
            raw_visual_reasoning=raw_visual_reasoning,
            general_operator_transfer=general_operator_transfer,
            research_architecture_overall=research_overall,
            robust_general_intelligence_overall=robust_overall,
        )

    @staticmethod
    def _value(payload: dict[str, Any], key: str) -> float:
        value = payload.get(key, 0.0)
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _inverse_mae(payload: dict[str, Any], key: str) -> float:
        value = payload.get(key, 1.0)
        try:
            return max(0.0, min(1.0, 1.0 - float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _coverage(count: Any, target: int) -> float:
        try:
            value = max(0.0, float(count))
        except (TypeError, ValueError):
            value = 0.0
        return max(0.2, min(1.0, value / float(max(1, target))))

    @staticmethod
    def _axis(name: str, weighted_terms: list[float], coverage: float, rationale: list[str]) -> ProgressAxis:
        base = round(sum(weighted_terms), 4)
        factor = round(0.55 + 0.45 * coverage, 4)
        return ProgressAxis(name=name, score=round(base * factor, 4), rationale=rationale + [f'benchmark coverage factor={factor:.3f}'])

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .common_eval import SemOpCommonEvaluator
from .progress_report import OperatorIntelligenceProgressEstimator


@dataclass
class UnderstandingEvalSummary:
    snapshot: dict[str, Any]
    progress: dict[str, Any]
    interpretation: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class SemOpUnderstandingEvaluator:
    """Run a broad understanding benchmark and return human-readable interpretation."""

    def __init__(self) -> None:
        self.common = SemOpCommonEvaluator()
        self.progress = OperatorIntelligenceProgressEstimator()

    def evaluate(
        self,
        hidden_premise_cases: list | None = None,
        cp_examples: list | None = None,
        cp_hidden_examples: list | None = None,
        cp_model: object | None = None,
        vlso_cases: list | None = None,
        vlso_real_image_cases: list | None = None,
        cp_mode: str = "heuristic",
    ) -> UnderstandingEvalSummary:
        snapshot = self.common.evaluate(
            hidden_premise_cases=hidden_premise_cases,
            cp_examples=cp_examples,
            cp_hidden_examples=cp_hidden_examples,
            cp_model=cp_model,
            vlso_cases=vlso_cases,
            vlso_real_image_cases=vlso_real_image_cases,
            cp_mode=cp_mode,
        )
        progress = self.progress.estimate(snapshot)
        interpretation = self._interpret(snapshot.model_dump(), progress.model_dump())
        return UnderstandingEvalSummary(
            snapshot=snapshot.model_dump(),
            progress=progress.model_dump(),
            interpretation=interpretation,
        )

    @staticmethod
    def _interpret(snapshot: dict[str, Any], progress: dict[str, Any]) -> dict[str, Any]:
        hidden = snapshot.get("hidden_premise") or {}
        cp = snapshot.get("cp_parser") or {}
        cp_hidden = snapshot.get("cp_hidden_constraints") or {}
        vlso = snapshot.get("vlso_grounded") or {}
        vlso_real = snapshot.get("vlso_real_image") or {}

        strengths: list[str] = []
        risks: list[str] = []
        next_steps: list[str] = []

        if float(hidden.get("goal_preservation_accuracy", 0.0) or 0.0) >= 0.9:
            strengths.append("Hidden-goal and goal-preservation reasoning is strong on the current benchmark.")
        else:
            risks.append("Goal-preservation judgments are still unstable on the current benchmark.")

        if float(hidden.get("unsupported_premise_precision", 0.0) or 0.0) >= 0.85:
            strengths.append("The system is relatively conservative about inventing unsupported premises.")
        else:
            risks.append("Unsupported premises still appear too often; premise validation needs tightening.")

        if float(cp.get("frame_jaccard", 0.0) or 0.0) >= 0.75:
            strengths.append("CP problem structuring is recovering logical frames reliably.")
        else:
            risks.append("CP frame extraction is still too weak for reliable problem structuring.")

        if cp_hidden:
            if float(cp_hidden.get("algorithm_exact_match", 0.0) or 0.0) < 0.5:
                risks.append("Hidden-constraint CP cases remain a bottleneck; parser and verifier need more training data.")
                next_steps.append("Expand CP hidden-constraint training and evaluation examples.")

        if float(vlso.get("operator_binding_recall", 0.0) or 0.0) >= 0.75:
            strengths.append("Structured visual operator binding works on the starter VLSO benchmark.")
        else:
            risks.append("Visual operator bindings are still weak even on the starter VLSO benchmark.")

        if vlso_real:
            if float(vlso_real.get("grounded_answer_accuracy", 0.0) or 0.0) < 0.7:
                risks.append("Real-image grounding is still the biggest visual bottleneck.")
                next_steps.append("Grow the reviewed real-image VLSO benchmark and improve detector/segmentation grounding.")
            else:
                strengths.append("Real-image grounded QA is reaching useful quality on the current reviewed set.")

        transfer_score = float((progress.get("general_operator_transfer") or {}).get("score", 0.0) or 0.0)
        if transfer_score < 0.5:
            risks.append("General operator transfer is still limited across domains.")
            next_steps.append("Train a stronger script-compatibility scorer and expand transfer benchmarks.")
        else:
            strengths.append("Operator transfer is beginning to generalize across domains.")

        if not next_steps:
            next_steps.append("Expand held-out benchmarks to avoid overfitting to the starter sets.")

        return {
            "headline": (
                "SemOp currently understands structured hidden premises and benchmarked symbolic structure better "
                "than it understands arbitrary real-world visual scenes."
            ),
            "strengths": strengths,
            "risks": risks,
            "next_steps": next_steps,
        }

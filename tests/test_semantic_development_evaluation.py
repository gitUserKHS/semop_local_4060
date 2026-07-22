from __future__ import annotations

import json

import pytest

from semop.semantic_development_evaluation import (
    development_candidate_passes,
    evaluate_generated_semantic_model,
)
from semop.semantic_distillation import DistillationDecision
from semop.semantic_student_evaluation import SemanticRuntimeObservation
from semop.verified_semantic_curriculum import build_verified_arithmetic_curriculum


def _observation(record, expression: str, model_id: str) -> SemanticRuntimeObservation:
    return SemanticRuntimeObservation(
        record_id=record.record_id,
        completion_json=json.dumps(
            {
                "domain": "math",
                "payload": {"expression": expression},
            }
        ),
        model_id=model_id,
        model_used=True,
        proof_eligible=True,
        replay_verified=True,
        elapsed_seconds=1.0,
        peak_vram_bytes=100,
    )


def test_generated_development_comparison_detects_real_improvement() -> None:
    corpus = build_verified_arithmetic_curriculum(
        train_examples_per_template=1,
        evaluation_examples_per_template=1,
    ).corpus("validation")
    positives = tuple(
        record
        for record in corpus.records
        if record.decision is not DistillationDecision.HARD_NEGATIVE
    )
    baseline_observations = tuple(
        _observation(record, "1 + 1", "base") for record in positives
    )
    candidate_observations = tuple(
        _observation(
            record,
            json.loads(record.completion)["payload"]["expression"],
            "candidate",
        )
        for record in positives
    )

    baseline = evaluate_generated_semantic_model(
        corpus,
        baseline_observations,
        model_id="base",
    )
    candidate = evaluate_generated_semantic_model(
        corpus,
        candidate_observations,
        model_id="candidate",
    )

    assert baseline.completion_rate == 0.0
    assert candidate.completion_rate == 1.0
    assert candidate.false_acceptances == 0
    assert development_candidate_passes(
        baseline,
        candidate,
        observed_probe_match=True,
    )
    assert candidate.to_dict()["promotion_eligible"] is False
    assert type(candidate).from_dict(candidate.to_dict()) == candidate


def test_generated_development_eval_rejects_training_score() -> None:
    corpus = build_verified_arithmetic_curriculum(
        train_examples_per_template=1,
        evaluation_examples_per_template=1,
    ).corpus("train")
    positives = tuple(
        record
        for record in corpus.records
        if record.decision is not DistillationDecision.HARD_NEGATIVE
    )

    with pytest.raises(ValueError, match="validation or sealed"):
        evaluate_generated_semantic_model(
            corpus,
            tuple(_observation(record, "1 + 1", "base") for record in positives),
            model_id="base",
        )

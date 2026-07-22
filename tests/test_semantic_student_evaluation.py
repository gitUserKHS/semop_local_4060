from __future__ import annotations

import json

import pytest

from semop.semantic_distillation import (
    DistillationDecision,
    SemanticDistillationCorpus,
    SemanticDistillationRecord,
)
from semop.semantic_student_evaluation import (
    SemanticRuntimeObservation,
    evaluate_semantic_student,
)


def _record(
    name: str,
    expression: str,
    *,
    decision: DistillationDecision = DistillationDecision.ACCEPTED,
    data_split: str = "sealed",
) -> SemanticDistillationRecord:
    completion = json.dumps(
        {
            "domain": "math",
            "confidence": 0.8,
            "operator_program": ["DECOMPOSE", "VERIFY", "EXPLAIN"],
            "payload": {"expression": expression},
            "answer": "ignored prose",
        },
        sort_keys=True,
    )
    return SemanticDistillationRecord(
        record_id="",
        domain="math",
        prompt=name,
        completion=completion,
        decision=decision,
        proposal_fingerprint=(name[0].encode().hex()[0] * 64),
        replay_verified=decision is not DistillationDecision.HARD_NEGATIVE,
        semantic_review_digest=(name[-1].encode().hex()[0] * 64),
        data_split=data_split,
        rejection_reason=(
            "reviewed semantics were wrong"
            if decision is DistillationDecision.HARD_NEGATIVE
            else ""
        ),
        source="local_semantic_trace_review",
    )


def _observation(
    record: SemanticDistillationRecord,
    expression: str,
    *,
    model_id: str,
    elapsed: float,
    vram: int,
) -> SemanticRuntimeObservation:
    completion = json.dumps(
        {
            "domain": "math",
            "confidence": 0.1,
            "operator_program": ["INFER"],
            "payload": {"expression": expression},
            "answer": "different prose",
        },
        sort_keys=True,
    )
    return SemanticRuntimeObservation(
        record_id=record.record_id,
        completion_json=completion,
        model_id=model_id,
        model_used=True,
        proof_eligible=True,
        replay_verified=True,
        elapsed_seconds=elapsed,
        peak_vram_bytes=vram,
    )


def test_sealed_student_gate_passes_only_replayed_reviewed_semantics() -> None:
    positive_a = _record("alpha1", "12 - 5")
    positive_b = _record("beta2", "2+3")
    rejected = _record(
        "gamma3",
        "12-5=8",
        decision=DistillationDecision.HARD_NEGATIVE,
    )
    corpus = SemanticDistillationCorpus((positive_a, positive_b, rejected))
    student = (
        _observation(
            positive_a,
            "12-5",
            model_id="Qwen/Qwen3.5-0.8B#adapter",
            elapsed=1.0,
            vram=100,
        ),
        _observation(
            positive_b,
            "2+3",
            model_id="Qwen/Qwen3.5-0.8B#adapter",
            elapsed=1.0,
            vram=100,
        ),
        _observation(
            rejected,
            "12-5",
            model_id="Qwen/Qwen3.5-0.8B#adapter",
            elapsed=1.0,
            vram=100,
        ),
    )
    teacher = (
        _observation(
            positive_a,
            "12-5",
            model_id="Qwen/Qwen3.5-2B",
            elapsed=2.0,
            vram=200,
        ),
        _observation(
            positive_b,
            "2+3",
            model_id="Qwen/Qwen3.5-2B",
            elapsed=2.0,
            vram=200,
        ),
    )

    evaluation = evaluate_semantic_student(
        corpus,
        student,
        teacher,
        corpus_sha256="a" * 64,
        candidate_sha256="b" * 64,
        candidate_model_id="Qwen/Qwen3.5-0.8B#adapter",
        teacher_model_id="Qwen/Qwen3.5-2B",
        artifact_bytes=1024,
    )

    assert evaluation.passed is True
    assert evaluation.report.replay_integrity == 1.0
    assert evaluation.report.false_acceptances == 0
    assert evaluation.report.teacher_retention == 1.0
    assert evaluation.report.latency_reduction == 0.5
    assert evaluation.report.vram_reduction == 0.5
    assert evaluation.semantic_matches == 2
    assert evaluation.teacher_live_matches == 2


def test_sealed_student_gate_rejects_repeated_human_negative() -> None:
    positive = _record("delta4", "3+4")
    rejected = _record(
        "epsilon5",
        "3+4=9",
        decision=DistillationDecision.HARD_NEGATIVE,
    )
    corpus = SemanticDistillationCorpus((positive, rejected))
    student = (
        _observation(
            positive,
            "3+4",
            model_id="student",
            elapsed=1.0,
            vram=100,
        ),
        _observation(
            rejected,
            "3+4=9",
            model_id="student",
            elapsed=1.0,
            vram=100,
        ),
    )
    teacher = (
        _observation(
            positive,
            "3+4",
            model_id="teacher",
            elapsed=2.0,
            vram=200,
        ),
    )

    evaluation = evaluate_semantic_student(
        corpus,
        student,
        teacher,
        corpus_sha256="c" * 64,
        candidate_sha256="d" * 64,
        candidate_model_id="student",
        teacher_model_id="teacher",
        artifact_bytes=1024,
    )

    assert evaluation.passed is False
    assert evaluation.report.false_acceptances == 1
    assert evaluation.outcomes[1].repeated_rejected_semantics is True
    assert "proof_eligible_false_acceptance_detected" in (
        evaluation.report.rejection_reasons("semantic_model")
    )


def test_sealed_student_gate_rejects_training_split() -> None:
    training_record = _record("zeta6", "8-2", data_split="train")
    observation = _observation(
        training_record,
        "8-2",
        model_id="student",
        elapsed=1.0,
        vram=100,
    )

    with pytest.raises(ValueError, match="outside the sealed split"):
        evaluate_semantic_student(
            SemanticDistillationCorpus((training_record,)),
            (observation,),
            (observation,),
            corpus_sha256="e" * 64,
            candidate_sha256="f" * 64,
            candidate_model_id="student",
            teacher_model_id="teacher",
            artifact_bytes=1024,
        )


def test_distillation_record_identity_binds_precommitted_split() -> None:
    sealed = _record("eta7", "5*5")

    with pytest.raises(ValueError, match="record id"):
        SemanticDistillationRecord(
            **{
                **sealed.to_dict(),
                "data_split": "train",
            }
        )

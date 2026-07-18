from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import pytest

from semop.kernel import (
    GroundingAuthority,
    GroundingLabel,
    LearningSplit,
    SemanticGroundingCorpus,
    SemanticGroundingReviewDecision,
    SemanticReviewDecision,
    build_semantic_grounding_target,
    create_semantic_grounding_review,
    create_semantic_review,
    generate_controlled_semantic_benchmark,
    load_semantic_benchmark,
    load_semantic_grounding_reviews,
    programmatic_semantic_learning_examples,
    write_semantic_grounding_review,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "semantic_benchmark" / "v1" / "cases.jsonl"


def _benchmark():
    return load_semantic_benchmark(CASES)


def test_compiles_exact_candidate_targets_without_promoting_case_labels() -> None:
    corpus = SemanticGroundingCorpus.compile(_benchmark())

    assert len(corpus.targets) == 20
    assert corpus.learning_examples() == ()
    assert corpus.audit().review_coverage == 0.0
    assert corpus.audit().clean
    assert dict(corpus.audit().domain_target_counts) == {
        "language": 6,
        "math": 6,
        "vision": 8,
    }
    assert all(target.candidate.sensor_features for target in corpus.targets)
    assert all(
        "expected_solved" not in target.candidate.statement
        for target in corpus.targets
    )


def test_case_level_review_cannot_be_reused_as_candidate_level_gold() -> None:
    benchmark = _benchmark()
    target = build_semantic_grounding_target(benchmark.cases[0])
    case_review = create_semantic_review(
        benchmark.cases[0],
        reviewer="reviewer",
        decision=SemanticReviewDecision.APPROVED,
        reviewed_at="2026-07-18T00:00:00Z",
    )

    with pytest.raises(TypeError, match="reviews have the wrong type"):
        SemanticGroundingCorpus((target,), (case_review,))  # type: ignore[arg-type]


def test_exact_human_review_becomes_one_training_example() -> None:
    target = build_semantic_grounding_target(_benchmark().cases[0])
    review = create_semantic_grounding_review(
        target,
        reviewed_label=GroundingLabel.ACCEPT,
        reviewer="human:unit-test",
        reviewed_at="2026-07-18T00:00:00Z",
        notes="checked the raw text and exact typed goal",
    )
    corpus = SemanticGroundingCorpus((target,), (review,))

    examples = corpus.learning_examples()

    assert len(examples) == 1
    assert examples[0].candidate is target.candidate
    assert examples[0].label is GroundingLabel.ACCEPT
    assert examples[0].authority is GroundingAuthority.HUMAN_REVIEW
    assert examples[0].record_digest == review.review_digest
    assert corpus.audit().approved_targets == (target.target_id,)


def test_review_may_correct_proposed_label_without_mutating_candidate() -> None:
    target = build_semantic_grounding_target(_benchmark().cases[0])
    corrected = GroundingLabel(
        GroundingLabel.REJECT
        if target.proposed_label is GroundingLabel.ACCEPT
        else GroundingLabel.ACCEPT
    )
    review = create_semantic_grounding_review(
        target,
        reviewed_label=corrected,
        reviewer="human:unit-test",
        reviewed_at="2026-07-18T00:00:00Z",
    )
    corpus = SemanticGroundingCorpus((target,), (review,))

    assert corpus.learning_examples()[0].label is corrected
    assert corpus.audit().corrected_label_targets == (target.target_id,)


def test_stale_or_rejected_review_never_creates_training_data() -> None:
    target = build_semantic_grounding_target(_benchmark().cases[0])
    review = create_semantic_grounding_review(
        target,
        reviewed_label=target.proposed_label,
        reviewer="human:unit-test",
        reviewed_at="2026-07-18T00:00:00Z",
    )
    stale = replace(review, candidate_digest="0" * 64)
    stale_corpus = SemanticGroundingCorpus((target,), (stale,))
    rejected = replace(
        review,
        decision=SemanticGroundingReviewDecision.REJECTED,
    )
    rejected_corpus = SemanticGroundingCorpus((target,), (rejected,))

    assert stale_corpus.learning_examples() == ()
    assert stale_corpus.audit().stale_review_targets == (target.target_id,)
    assert rejected_corpus.learning_examples() == ()
    assert rejected_corpus.audit().rejected_targets == (target.target_id,)


def test_review_round_trip_preserves_digest_binding() -> None:
    target = build_semantic_grounding_target(_benchmark().cases[0])
    review = create_semantic_grounding_review(
        target,
        reviewed_label=target.proposed_label,
        reviewer="human:unit-test",
        reviewed_at="2026-07-18T00:00:00Z",
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "reviews.jsonl"
        write_semantic_grounding_review(path, review)

        loaded = load_semantic_grounding_reviews(path)

    assert loaded == (review,)
    assert loaded[0].review_digest == review.review_digest


def test_programmatic_raw_lmv_cases_produce_balanced_verified_examples() -> None:
    benchmark = generate_controlled_semantic_benchmark(
        per_domain=8,
        split=LearningSplit.TRAIN,
        seed=13,
        namespace="unit",
    )

    examples = programmatic_semantic_learning_examples(benchmark)

    assert len(examples) == 24
    assert {example.candidate.domain for example in examples} == {
        "language",
        "math",
        "vision",
    }
    assert {example.label for example in examples} == {
        GroundingLabel.ACCEPT,
        GroundingLabel.REJECT,
    }
    assert all(
        example.authority is GroundingAuthority.EXTERNAL_VERIFIER
        for example in examples
    )
    assert all(example.candidate.sensor_features for example in examples)


def test_programmatic_oracle_contract_is_required() -> None:
    with pytest.raises(ValueError, match="lacks the programmatic oracle contract"):
        programmatic_semantic_learning_examples(_benchmark())

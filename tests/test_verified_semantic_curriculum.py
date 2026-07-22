from __future__ import annotations

import json
from pathlib import Path

from semop.kernel import OperatorKernel
from semop.kernel.domains.arithmetic import ArithmeticExpressionAdapter
from semop.semantic_distillation import DistillationDecision
from semop.verified_semantic_curriculum import (
    GENERATOR_VERSION,
    build_verified_arithmetic_curriculum,
)


def test_verified_arithmetic_curriculum_has_disjoint_template_splits() -> None:
    curriculum = build_verified_arithmetic_curriculum(
        train_examples_per_template=2,
        evaluation_examples_per_template=1,
    )

    train = set(curriculum.template_ids_by_split["train"])
    validation = set(curriculum.template_ids_by_split["validation"])
    sealed = set(curriculum.template_ids_by_split["sealed"])

    assert len(train) == 6
    assert len(validation) == 3
    assert len(sealed) == 3
    assert not train & validation
    assert not train & sealed
    assert not validation & sealed


def test_generated_positives_and_hard_negatives_are_executor_replayable() -> None:
    curriculum = build_verified_arithmetic_curriculum(
        train_examples_per_template=1,
        evaluation_examples_per_template=1,
    )

    for split in ("train", "validation", "sealed"):
        records = curriculum.records_by_split[split]
        positives = [
            item
            for item in records
            if item.decision is DistillationDecision.ACCEPTED
        ]
        negatives = [
            item
            for item in records
            if item.decision is DistillationDecision.HARD_NEGATIVE
        ]
        assert len(positives) == len(negatives)
        for positive, negative in zip(positives, negatives, strict=True):
            assert positive.prompt == negative.prompt
            positive_expression = json.loads(positive.completion)["payload"]["expression"]
            negative_expression = json.loads(negative.completion)["payload"]["expression"]
            positive_instance = ArithmeticExpressionAdapter().adapt(positive_expression)
            negative_instance = ArithmeticExpressionAdapter().adapt(negative_expression)
            positive_result = OperatorKernel(positive_instance.registry).solve(
                positive_instance.state,
                positive_instance.goals,
            )
            negative_result = OperatorKernel(negative_instance.registry).solve(
                negative_instance.state,
                negative_instance.goals,
            )
            assert positive_result.success and positive_result.verified
            assert negative_result.success and negative_result.verified
            assert positive_instance.metadata["answer"] != negative_instance.metadata["answer"]


def test_curriculum_is_seed_deterministic_and_writes_digest_manifest(
    tmp_path: Path,
) -> None:
    first = build_verified_arithmetic_curriculum(
        seed=17,
        train_examples_per_template=2,
        evaluation_examples_per_template=1,
    )
    second = build_verified_arithmetic_curriculum(
        seed=17,
        train_examples_per_template=2,
        evaluation_examples_per_template=1,
    )

    assert first.records_by_split == second.records_by_split
    manifest = first.save(tmp_path)

    assert manifest["schema_version"] == GENERATOR_VERSION
    assert manifest["promotion_eligible"] is False
    assert manifest["files"]["train"]["positives"] == 12
    for split in ("train", "validation", "sealed"):
        assert (tmp_path / f"{split}.jsonl").is_file()
        assert len(manifest["files"][split]["sha256"]) == 64

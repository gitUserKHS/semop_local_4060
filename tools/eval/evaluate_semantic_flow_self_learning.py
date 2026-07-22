from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_low_resource_transfer import _peak_rss_bytes
from semop.kernel import (
    ActiveCurriculumConfig,
    ActiveCurriculumScheduler,
    ActiveSelfLearningLoop,
    LearningSplit,
    OperatorKernel,
    SelfLearningBudget,
    SolveBudget,
    generate_semantic_flow_transfer_split,
    learning_tasks_from_synthetic,
)


def evaluate_semantic_flow_self_learning(
    *,
    examples_per_structure: int = 2,
    seed: int = 17,
    min_expansion_reduction: float = 0.10,
    max_expansions: int = 5_000,
) -> dict[str, Any]:
    split = generate_semantic_flow_transfer_split(
        examples_per_structure,
        seed=seed,
    )
    training = learning_tasks_from_synthetic(
        split.training,
        split=LearningSplit.TRAIN,
        namespace=f"semantic-flow-train-{seed}",
    )
    heldout_positive = learning_tasks_from_synthetic(
        split.heldout,
        split=LearningSplit.HELDOUT,
        namespace=f"semantic-flow-heldout-{seed}",
    )
    heldout_negative = learning_tasks_from_synthetic(
        split.negative_controls,
        split=LearningSplit.HELDOUT,
        namespace=f"semantic-flow-negative-{seed}",
        expected_solved=False,
    )
    heldout = heldout_positive + heldout_negative

    peak_before = _peak_rss_bytes()
    started = perf_counter()
    result = ActiveSelfLearningLoop(
        scheduler=ActiveCurriculumScheduler(
            ActiveCurriculumConfig(max_tasks=len(training))
        ),
        self_learning_budget=SelfLearningBudget(
            solve_budget=SolveBudget(
                max_expansions=max_expansions,
                timeout_seconds=10.0,
            ),
            min_expansion_reduction=min_expansion_reduction,
        ),
        generations=1,
    ).run(training, heldout)
    wall_seconds = perf_counter() - started
    peak_after = _peak_rss_bytes()

    learning_round = result.rounds[0]
    iteration = learning_round.learning_iteration
    candidate = iteration.candidate_metrics
    audit = result.split_audit
    positive_profiles = tuple(
        profile
        for profile in audit.heldout_profiles
        if profile.expected_solved
    )
    all_domain_tags = all(
        _has_verified_lmv_trace(task)
        for task in training + heldout_positive
    )
    training_counts = tuple(
        sum(count for _color, count in task.instance.metadata["object_counts"])
        for task in training
    )
    heldout_counts = tuple(
        sum(count for _color, count in task.instance.metadata["object_counts"])
        for task in heldout_positive
    )
    heldout_texts = tuple(
        str(task.instance.metadata["text"]) for task in heldout_positive
    )

    gates = {
        "structural_and_program_overlap_zero": (
            audit.valid
            and not audit.overlapping_structures
            and not audit.overlapping_programs
        ),
        "semantic_flow_uses_language_math_vision": all_domain_tags,
        "heldout_has_deeper_operator_program": (
            max(profile.proof_depth for profile in positive_profiles)
            > max(profile.proof_depth for profile in audit.training_profiles)
        ),
        "heldout_has_larger_images": max(heldout_counts) > max(training_counts),
        "heldout_has_new_english_and_korean_phrasing": (
            any("number of" in text for text in heldout_texts)
            and any("물체의 개수" in text for text in heldout_texts)
        ),
        "candidate_promoted": iteration.accepted,
        "proof_soundness_100_percent": (
            candidate is not None and candidate.proof_soundness == 1.0
        ),
        "false_positives_zero": (
            candidate is not None and candidate.false_positives == 0
        ),
        "solve_rate_drop_within_1pp": (
            candidate is not None
            and candidate.verified_solve_rate
            >= iteration.baseline_metrics.verified_solve_rate - 0.01
        ),
        "minimum_expansion_reduction": (
            iteration.expansion_reduction >= min_expansion_reduction
        ),
        "policy_parameters_under_15m": iteration.parameter_count <= 15_000_000,
        "artifact_under_64mb": iteration.artifact_bytes <= 64 * 1024 * 1024,
        "heldout_p95_under_10s": (
            candidate is not None and candidate.p95_cpu_seconds <= 10.0
        ),
        "additional_peak_rss_under_512mb": (
            max(0, peak_after - peak_before) <= 512 * 1024 * 1024
        ),
    }
    return {
        "schema_version": 1,
        "suite": "typed-semantic-flow-self-learning",
        "seed": seed,
        "split": {
            "training_structures": audit.training_structures,
            "heldout_structures": audit.heldout_structures,
            "overlapping_structures": audit.overlapping_structures,
            "training_programs": audit.training_programs,
            "heldout_programs": audit.heldout_programs,
            "overlapping_programs": audit.overlapping_programs,
            "training_proof_depths": tuple(
                profile.proof_depth for profile in audit.training_profiles
            ),
            "heldout_positive_proof_depths": tuple(
                profile.proof_depth for profile in positive_profiles
            ),
        },
        "data": {
            "training_tasks": len(training),
            "heldout_positive_tasks": len(heldout_positive),
            "heldout_negative_controls": len(heldout_negative),
            "selected_training_tasks": len(learning_round.selection.selected_tasks),
            "verified_training_traces": iteration.verified_training_traces,
            "decision_cases": iteration.decision_cases,
            "training_object_counts": training_counts,
            "heldout_object_counts": heldout_counts,
        },
        "policy": {
            "kind": iteration.candidate_kind,
            "promoted": iteration.accepted,
            "rejection_reasons": iteration.rejection_reasons,
            "parameters": iteration.parameter_count,
            "artifact_bytes": iteration.artifact_bytes,
            "training_updates": iteration.training_updates,
        },
        "before": iteration.baseline_metrics.to_dict(),
        "after": candidate.to_dict() if candidate is not None else None,
        "ab": {
            "positive_expansions_before": (
                iteration.baseline_metrics.positive_expansions
            ),
            "positive_expansions_after": (
                candidate.positive_expansions if candidate is not None else None
            ),
            "expansion_reduction": iteration.expansion_reduction,
        },
        "resources": {
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": peak_after,
            "additional_peak_rss_bytes": max(0, peak_after - peak_before),
        },
        "gates": {
            **gates,
            "all_passed": all(gates.values()),
        },
    }


def _has_verified_lmv_trace(task) -> bool:
    result = OperatorKernel(task.instance.registry).solve(
        task.instance.state,
        task.instance.goals,
    )
    tags = {
        tag
        for step in result.proof
        for tag in step.action.operator.tags
        if tag in {"language", "math", "vision"}
    }
    return result.success and result.verified and tags == {
        "language",
        "math",
        "vision",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate self-learning on held-out vision-to-math-to-language flows"
        )
    )
    parser.add_argument("--examples-per-structure", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-expansion-reduction", type=float, default=0.10)
    parser.add_argument("--max-expansions", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_semantic_flow_self_learning(
        examples_per_structure=args.examples_per_structure,
        seed=args.seed,
        min_expansion_reduction=args.min_expansion_reduction,
        max_expansions=args.max_expansions,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["gates"]["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

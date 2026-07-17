from __future__ import annotations

import argparse
from dataclasses import asdict
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
    LearningSplit,
    LearningTask,
    OperatorKernel,
    SelfDiscoveringLearningLoop,
    SelfLearningBudget,
    SolveBudget,
    generate_lmv_structural_transfer_split,
    learning_tasks_from_synthetic,
    profile_learning_task,
)


def evaluate_self_discovery(
    *,
    examples_per_structure: int = 1,
    seed: int = 41,
    min_expansion_reduction: float = 0.10,
    max_expansions: int = 5_000,
) -> dict[str, Any]:
    split = generate_lmv_structural_transfer_split(
        examples_per_structure,
        seed=seed,
    )
    training = learning_tasks_from_synthetic(
        split.training,
        split=LearningSplit.TRAIN,
        namespace=f"discovery-train-{seed}",
    )
    heldout = learning_tasks_from_synthetic(
        split.heldout,
        split=LearningSplit.HELDOUT,
        namespace=f"discovery-heldout-{seed}",
    ) + learning_tasks_from_synthetic(
        split.negative_controls,
        split=LearningSplit.HELDOUT,
        namespace=f"discovery-negative-{seed}",
        expected_solved=False,
    )

    peak_before = _peak_rss_bytes()
    started = perf_counter()
    result = SelfDiscoveringLearningLoop(
        self_learning_budget=SelfLearningBudget(
            solve_budget=SolveBudget(
                max_expansions=max_expansions,
                timeout_seconds=10.0,
            ),
            min_expansion_reduction=min_expansion_reduction,
        )
    ).run(training, heldout)
    wall_seconds = perf_counter() - started
    peak_after = _peak_rss_bytes()

    learning_round = result.learning.rounds[0]
    iteration = learning_round.learning_iteration
    candidate = iteration.candidate_metrics
    audit = result.learning.split_audit
    train_positive = result.training_discovery.positive_tasks
    train_negative = result.training_discovery.negative_tasks
    heldout_positive = result.heldout_discovery.positive_tasks
    heldout_negative = result.heldout_discovery.negative_tasks
    train_profiles = tuple(profile_learning_task(task) for task in train_positive)
    heldout_profiles = tuple(profile_learning_task(task) for task in heldout_positive)
    positive_replay = all(
        _is_verified_success(task) for task in train_positive + heldout_positive
    )
    negative_sound = all(
        _is_verified_failure(task) for task in train_negative + heldout_negative
    )
    train_programs = tuple(profile.program_signature for profile in train_profiles)
    heldout_programs = tuple(profile.program_signature for profile in heldout_profiles)
    max_training_depth = max(
        (profile.proof_depth for profile in train_profiles),
        default=0,
    )
    max_heldout_depth = max(
        (profile.proof_depth for profile in heldout_profiles),
        default=0,
    )

    gates = {
        "discovered_positives_replay_verified": positive_replay,
        "counterfactual_negatives_unproved": negative_sound,
        "training_programs_unique": len(train_programs) == len(set(train_programs)),
        "train_heldout_structure_overlap_zero": not audit.overlapping_structures,
        "train_heldout_program_overlap_zero": not audit.overlapping_programs,
        "training_synthetic_depth_at_most_6": max_training_depth <= 6,
        "heldout_contains_depth_extrapolation": max_heldout_depth > 6,
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
        "suite": "verifier-backed-language-math-vision-task-discovery",
        "seed": seed,
        "seeds": {
            "training": len(training),
            "heldout_positive": len(split.heldout),
            "heldout_negative": len(split.negative_controls),
        },
        "discovery": {
            "training_positive": len(train_positive),
            "training_negative": len(train_negative),
            "heldout_positive": len(heldout_positive),
            "heldout_negative": len(heldout_negative),
            "training_accepted_records": len(
                result.training_discovery.accepted_records
            ),
            "training_rejected_records": len(
                result.training_discovery.rejected_records
            ),
            "heldout_accepted_records": len(
                result.heldout_discovery.accepted_records
            ),
            "heldout_rejected_records": len(
                result.heldout_discovery.rejected_records
            ),
            "training_structures": tuple(
                task.structure_key for task in train_positive
            ),
            "heldout_structures": tuple(
                task.structure_key for task in heldout_positive
            ),
            "max_training_proof_depth": max_training_depth,
            "max_heldout_proof_depth": max_heldout_depth,
            "records": tuple(
                asdict(record) for record in result.training_discovery.records
            ),
        },
        "expanded_split": {
            "training_tasks": len(result.expanded_training_tasks),
            "heldout_tasks": len(result.expanded_heldout_tasks),
            "training_domains": audit.training_domains,
            "heldout_domains": audit.heldout_domains,
            "overlapping_structures": audit.overlapping_structures,
            "overlapping_programs": audit.overlapping_programs,
        },
        "curriculum": {
            "selected_tasks": len(learning_round.selection.selected_tasks),
            "domain_counts": dict(learning_round.selection.domain_counts),
            "decisions": tuple(
                asdict(item) for item in learning_round.selection.decisions
            ),
        },
        "policy": {
            "kind": iteration.candidate_kind,
            "promoted": iteration.accepted,
            "rejection_reasons": iteration.rejection_reasons,
            "parameters": iteration.parameter_count,
            "artifact_bytes": iteration.artifact_bytes,
        },
        "before": asdict(iteration.baseline_metrics),
        "after": asdict(candidate) if candidate is not None else None,
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


def _is_verified_success(task: LearningTask) -> bool:
    result = OperatorKernel(task.instance.registry).solve(
        task.instance.state,
        task.instance.goals,
    )
    return result.success and result.verified


def _is_verified_failure(task: LearningTask) -> bool:
    result = OperatorKernel(task.instance.registry).solve(
        task.instance.state,
        task.instance.goals,
    )
    return not result.success and not result.verified


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate verifier-backed task discovery and self-learning on CPU"
        )
    )
    parser.add_argument("--examples-per-structure", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--min-expansion-reduction", type=float, default=0.10)
    parser.add_argument("--max-expansions", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_self_discovery(
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

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
    ActiveCurriculumConfig,
    ActiveCurriculumScheduler,
    ActiveSelfLearningLoop,
    LearningSplit,
    SelfLearningBudget,
    SolveBudget,
    generate_lmv_structural_transfer_split,
    learning_tasks_from_synthetic,
)


def evaluate_self_learning(
    *,
    examples_per_domain: int = 3,
    seed: int = 11,
    min_expansion_reduction: float = 0.10,
    max_expansions: int = 2_000,
) -> dict[str, Any]:
    split = generate_lmv_structural_transfer_split(
        examples_per_domain,
        seed=seed,
    )
    training = learning_tasks_from_synthetic(
        split.training,
        split=LearningSplit.TRAIN,
        namespace=f"train-seed-{seed}",
    )
    heldout = learning_tasks_from_synthetic(
        split.heldout,
        split=LearningSplit.HELDOUT,
        namespace=f"heldout-seed-{seed}",
    ) + learning_tasks_from_synthetic(
        split.negative_controls,
        split=LearningSplit.HELDOUT,
        namespace=f"control-seed-{seed}",
        expected_solved=False,
    )

    budget = SelfLearningBudget(
        solve_budget=SolveBudget(
            max_expansions=max_expansions,
            timeout_seconds=5.0,
        ),
        min_expansion_reduction=min_expansion_reduction,
    )
    peak_before = _peak_rss_bytes()
    started = perf_counter()
    result = ActiveSelfLearningLoop(
        scheduler=ActiveCurriculumScheduler(
            ActiveCurriculumConfig(max_tasks=6)
        ),
        self_learning_budget=budget,
        generations=1,
    ).run(training, heldout)
    elapsed = perf_counter() - started
    peak_after = _peak_rss_bytes()
    learning_round = result.rounds[0]
    iteration = learning_round.learning_iteration
    candidate = iteration.candidate_metrics
    audit = result.split_audit

    gates = {
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
        "policy_parameters_under_15m": (
            iteration.parameter_count <= 15_000_000
        ),
        "artifact_under_64mb": (
            iteration.artifact_bytes <= 64 * 1024 * 1024
        ),
        "heldout_p95_under_10s": (
            candidate is not None and candidate.p95_cpu_seconds <= 10.0
        ),
        "additional_peak_rss_under_512mb": (
            max(0, peak_after - peak_before) <= 512 * 1024 * 1024
        ),
        "structural_split_non_overlapping": (
            audit.valid
            and not audit.overlapping_structures
            and not audit.overlapping_programs
        ),
        "language_math_vision_covered": (
            audit.training_domains == ("language", "math", "vision")
            and audit.training_domains == audit.heldout_domains
        ),
    }
    return {
        "schema_version": 2,
        "suite": "structural-transfer-language-math-vision-self-learning",
        "seed": seed,
        "split": {
            "kind": "held-out-capability-composition",
            "training_structures": audit.training_structures,
            "heldout_structures": audit.heldout_structures,
            "overlapping_structures": audit.overlapping_structures,
            "training_programs": audit.training_programs,
            "heldout_programs": audit.heldout_programs,
            "overlapping_programs": audit.overlapping_programs,
            "training_domains": audit.training_domains,
            "heldout_domains": audit.heldout_domains,
            "training_outcomes_valid": audit.training_outcomes_valid,
            "heldout_outcomes_valid": audit.heldout_outcomes_valid,
        },
        "data": {
            "training_pool_tasks": len(training),
            "selected_training_tasks": len(learning_round.selection.selected_tasks),
            "heldout_positive_tasks": len(split.heldout),
            "heldout_negative_controls": len(split.negative_controls),
            "verified_training_traces": iteration.verified_training_traces,
            "decision_cases": iteration.decision_cases,
        },
        "curriculum": {
            "decisions": [
                asdict(item) for item in learning_round.selection.decisions
            ],
            "domain_counts": dict(learning_round.selection.domain_counts),
            "rejected_unverified": learning_round.selection.rejected_unverified,
            "rejected_no_supervision": (
                learning_round.selection.rejected_no_supervision
            ),
        },
        "policy": {
            "kind": iteration.candidate_kind,
            "parameters": iteration.parameter_count,
            "artifact_bytes": iteration.artifact_bytes,
            "training_updates": iteration.training_updates,
            "generation": iteration.generation_after,
            "promoted": iteration.accepted,
            "rejection_reasons": iteration.rejection_reasons,
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
        "macros": {
            "retained_candidates": len(result.macro_library.records),
            "active": False,
        },
        "resources": {
            "wall_seconds": elapsed,
            "peak_rss_bytes": peak_after,
            "additional_peak_rss_bytes": max(0, peak_after - peak_before),
        },
        "gates": {
            **gates,
            "all_passed": all(gates.values()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate verifier-gated typed self-learning on an ordinary CPU"
    )
    parser.add_argument(
        "--examples-per-structure",
        "--examples-per-domain",
        dest="examples_per_domain",
        type=int,
        default=3,
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--min-expansion-reduction", type=float, default=0.10)
    parser.add_argument("--max-expansions", type=int, default=2_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_self_learning(
        examples_per_domain=args.examples_per_domain,
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

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_low_resource_transfer import _peak_rss_bytes
from semop.kernel import (
    LearningSplit,
    MacroLearningBudget,
    MdlMacroLibrary,
    OperatorKernel,
    SolveBudget,
    VerifiedMacroLearningLoop,
    generate_macro_reuse_transfer_split,
    learning_tasks_from_synthetic,
)


def evaluate_active_macro_learning(
    *,
    examples_per_domain: int = 3,
    validation_per_domain: int = 1,
    heldout_per_domain: int = 1,
    seed: int = 23,
    min_expansion_reduction: float = 0.50,
    max_expansions: int = 5_000,
) -> dict[str, Any]:
    split = generate_macro_reuse_transfer_split(
        examples_per_domain,
        validation_per_domain=validation_per_domain,
        heldout_per_domain=heldout_per_domain,
        seed=seed,
    )
    training = learning_tasks_from_synthetic(
        split.training,
        split=LearningSplit.TRAIN,
        namespace=f"active-macro-train-{seed}",
    )
    validation = learning_tasks_from_synthetic(
        split.validation,
        split=LearningSplit.HELDOUT,
        namespace=f"active-macro-validation-{seed}",
    )
    heldout_positive = learning_tasks_from_synthetic(
        split.heldout,
        split=LearningSplit.HELDOUT,
        namespace=f"active-macro-heldout-{seed}",
    )
    heldout_negative = learning_tasks_from_synthetic(
        split.negative_controls,
        split=LearningSplit.HELDOUT,
        namespace=f"active-macro-negative-{seed}",
        expected_solved=False,
    )
    heldout = heldout_positive + heldout_negative

    peak_before = _peak_rss_bytes()
    started = perf_counter()
    result = VerifiedMacroLearningLoop(
        MacroLearningBudget(
            solve_budget=SolveBudget(
                max_expansions=max_expansions,
                timeout_seconds=5.0,
            ),
            min_expansion_reduction=min_expansion_reduction,
            min_domains_improved=3,
        )
    ).run(training, validation, heldout)
    wall_seconds = perf_counter() - started
    peak_after = _peak_rss_bytes()

    primitive_proofs_only = True
    full_replay_verified = True
    for task in heldout_positive:
        solved = result.guided_results[task.task_id]
        primitive_proofs_only = primitive_proofs_only and all(
            step.action.operator.name in task.instance.registry.operators
            and not step.action.operator.name.startswith("macro_")
            for step in solved.proof
        )
        replay = OperatorKernel(task.instance.registry).replay(
            task.instance.state,
            task.instance.goals,
            solved.proof,
        )
        full_replay_verified = full_replay_verified and replay.verified

    artifact_bytes = 0
    artifact_round_trip = False
    with tempfile.TemporaryDirectory() as directory:
        artifact = Path(directory) / "active-macros.json"
        persisted = VerifiedMacroLearningLoop.persist_promoted(result, artifact)
        if persisted is not None:
            artifact_bytes = artifact.stat().st_size
            restored = MdlMacroLibrary.load(artifact)
            artifact_round_trip = restored.records == result.active_library.records

    all_ids = [
        problem.problem_id
        for group in (split.training, split.validation, split.heldout)
        for problem in group
    ]
    vision_digest_sets = tuple(
        {
            problem.instance.metadata["image_digest"]
            for problem in group
            if problem.domain == "vision"
        }
        for group in (split.training, split.validation, split.heldout)
    )
    vision_inputs_disjoint = (
        vision_digest_sets[0].isdisjoint(vision_digest_sets[1])
        and vision_digest_sets[0].isdisjoint(vision_digest_sets[2])
        and vision_digest_sets[1].isdisjoint(vision_digest_sets[2])
    )
    activation_by_domain = {
        domain: max(
            audit.active_programs
            for audit in result.activation_audit
            if audit.domain == domain
        )
        for domain in ("language", "math", "vision")
    }
    expansion_by_domain = {
        domain: {
            "before": result.baseline_metrics.for_domain(
                domain
            ).positive_expansions,
            "after": result.guided_metrics.for_domain(
                domain
            ).positive_expansions,
        }
        for domain in ("language", "math", "vision")
    }
    for values in expansion_by_domain.values():
        before = values["before"]
        values["reduction"] = (before - values["after"]) / before

    gates = {
        "three_way_task_ids_disjoint": len(all_ids) == len(set(all_ids)),
        "vision_inputs_disjoint": vision_inputs_disjoint,
        "verified_training_support": result.verified_training_traces >= 9,
        "independent_validation_support": result.verified_validation_traces >= 3,
        "candidate_promoted": result.promoted,
        "all_three_domains_active": set(result.improved_domains)
        == {"language", "math", "vision"}
        and all(value > 0 for value in activation_by_domain.values()),
        "proofs_use_registered_primitives_only": primitive_proofs_only,
        "full_proof_replay_verified": full_replay_verified,
        "proof_soundness_100_percent": result.guided_metrics.proof_soundness
        == 1.0,
        "false_positives_zero": result.guided_metrics.false_positives == 0,
        "solve_rate_drop_within_1pp": (
            result.guided_metrics.verified_solve_rate
            >= result.baseline_metrics.verified_solve_rate - 0.01
        ),
        "minimum_expansion_reduction": result.expansion_reduction
        >= min_expansion_reduction,
        "promoted_artifact_round_trip": artifact_round_trip,
        "artifact_under_64mb": artifact_bytes <= 64 * 1024 * 1024,
        "heldout_p95_under_10s": result.guided_metrics.p95_cpu_seconds <= 10.0,
        "additional_peak_rss_under_512mb": max(0, peak_after - peak_before)
        <= 512 * 1024 * 1024,
    }
    return {
        "schema_version": 1,
        "suite": "typed-active-macro-self-learning",
        "seed": seed,
        "split": {
            "training_tasks": len(training),
            "validation_tasks": len(validation),
            "heldout_positive_tasks": len(heldout_positive),
            "heldout_negative_controls": len(heldout_negative),
            "vision_image_digests": tuple(
                tuple(sorted(values)) for values in vision_digest_sets
            ),
        },
        "learning": {
            "promoted": result.promoted,
            "rejection_reasons": result.rejection_reasons,
            "verified_training_traces": result.verified_training_traces,
            "verified_validation_traces": result.verified_validation_traces,
            "retained_macros": len(result.candidate_library.records),
            "active_macros": len(result.active_library.records),
            "improved_domains": result.improved_domains,
            "activation_by_domain": activation_by_domain,
        },
        "before": result.baseline_metrics.to_dict(),
        "after": result.guided_metrics.to_dict(),
        "ab": {
            "positive_expansions_before": (
                result.baseline_metrics.positive_expansions
            ),
            "positive_expansions_after": result.guided_metrics.positive_expansions,
            "expansion_reduction": result.expansion_reduction,
            "by_domain": expansion_by_domain,
        },
        "artifact": {
            "format_version": MdlMacroLibrary.FORMAT_VERSION,
            "bytes": artifact_bytes,
            "round_trip_verified": artifact_round_trip,
        },
        "resources": {
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": peak_after,
            "additional_peak_rss_bytes": max(0, peak_after - peak_before),
        },
        "gates": {**gates, "all_passed": all(gates.values())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate verifier-gated active primitive macro learning"
    )
    parser.add_argument("--examples-per-domain", type=int, default=3)
    parser.add_argument("--validation-per-domain", type=int, default=1)
    parser.add_argument("--heldout-per-domain", type=int, default=1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--min-expansion-reduction", type=float, default=0.50)
    parser.add_argument("--max-expansions", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_active_macro_learning(
        examples_per_domain=args.examples_per_domain,
        validation_per_domain=args.validation_per_domain,
        heldout_per_domain=args.heldout_per_domain,
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

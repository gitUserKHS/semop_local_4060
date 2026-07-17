from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TRAIN = ROOT / "tools" / "train"
EVAL = ROOT / "tools" / "eval"
for path in (SRC, TRAIN, EVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_low_resource_transfer import evaluate
from train_tiny_controller import train


LMV_DOMAINS = ("language", "math", "vision")


def run_lodo_experiment(
    output_dir: str | Path,
    *,
    held_out_domains: Sequence[str] = LMV_DOMAINS,
    examples_per_domain: int = 20,
    epochs: int = 5,
    seed: int = 0,
    device: str = "cpu",
    debug_small: bool = True,
    max_expansions: int = 20_000,
) -> dict[str, Any]:
    selected = tuple(held_out_domains)
    if not selected:
        raise ValueError("LODO experiment requires at least one held-out domain")
    if len(set(selected)) != len(selected):
        raise ValueError("held-out domains must be unique")
    unknown = sorted(set(selected) - set(LMV_DOMAINS))
    if unknown:
        raise ValueError(f"unknown LODO domains: {unknown}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    runs: dict[str, Any] = {}
    for held_out in selected:
        trained_domains = tuple(
            domain for domain in LMV_DOMAINS if domain != held_out
        )
        artifact = destination / f"without_{held_out}.npz"
        training = train(
            artifact,
            examples_per_domain=examples_per_domain,
            epochs=epochs,
            seed=seed,
            device=device,
            debug_small=debug_small,
            curriculum="language-math-vision",
            domains=trained_domains,
        )
        evaluation = evaluate(
            str(artifact),
            max_expansions=max_expansions,
            suite="language-math-vision",
        )
        composed_evaluation = evaluate(
            str(artifact),
            max_expansions=max_expansions,
            suite="composed-v4",
        )
        heldout_metrics = evaluation["leave_one_domain_out"][held_out]
        if not heldout_metrics.get("available"):
            raise RuntimeError(
                f"evaluator did not verify {held_out} as held out: "
                f"{heldout_metrics.get('evaluation_mode')}"
            )
        baseline = evaluation["unguided"]["by_domain"][held_out]
        guided = evaluation["guided"]["by_domain"][held_out]
        baseline_expansions = baseline["median_expansions"]
        guided_expansions = guided["median_expansions"]
        median_reduction = (
            0.0
            if baseline_expansions == 0
            else (baseline_expansions - guided_expansions) / baseline_expansions
        )
        case_domains = {
            item["case_id"]: item["domain"]
            for item in evaluation["guided"]["cases"]
        }
        heldout_case_reductions = {
            case_id: reduction
            for case_id, reduction in evaluation["ab_comparison"][
                "expansion_reduction_by_case"
            ].items()
            if case_domains.get(case_id) == held_out
        }
        runs[held_out] = {
            "held_out_domain": held_out,
            "trained_domains": list(trained_domains),
            "evaluation_mode": heldout_metrics["evaluation_mode"],
            "artifact": str(artifact),
            "summary": training["summary_path"],
            "parameter_count": training["parameter_count"],
            "artifact_bytes": training["artifact_bytes"],
            "verified_synthetic_traces": training["verified_synthetic_traces"],
            "training_seconds": training["elapsed_seconds"],
            "verified_solve_rate": heldout_metrics["verified_solve_rate"],
            "proof_soundness": heldout_metrics["proof_soundness"],
            "baseline_median_expansions": baseline_expansions,
            "guided_median_expansions": guided_expansions,
            "median_expansion_reduction": median_reduction,
            "case_expansion_reductions": heldout_case_reductions,
            "p95_cpu_seconds": heldout_metrics["p95_cpu_seconds"],
            "overall_guided_expansions": evaluation["guided"]["expansions"],
            "overall_unguided_expansions": evaluation["unguided"]["expansions"],
            "composed_v4": {
                "verified_solve_rate": composed_evaluation["guided"][
                    "verified_solve_rate"
                ],
                "proof_soundness": composed_evaluation["guided"][
                    "proof_soundness"
                ],
                "false_positives": composed_evaluation["guided"][
                    "false_positives"
                ],
                "guided_expansions": composed_evaluation["guided"]["expansions"],
                "unguided_expansions": composed_evaluation["unguided"][
                    "expansions"
                ],
            },
        }

    sound = all(item["proof_soundness"] == 1.0 for item in runs.values())
    no_solve_drop = all(item["verified_solve_rate"] >= 0.99 for item in runs.values())
    improved_holdouts = sum(
        any(reduction > 0 for reduction in item["case_expansion_reductions"].values())
        for item in runs.values()
    )
    median_gate_holdouts = sum(
        item["median_expansion_reduction"] >= 0.30
        for item in runs.values()
    )
    resource_caps_pass = all(
        item["parameter_count"] <= 15_000_000
        and item["artifact_bytes"] <= 64 * 1024 * 1024
        and item["p95_cpu_seconds"] <= 10.0
        for item in runs.values()
    )
    composed_gate = all(
        item["composed_v4"]["verified_solve_rate"] == 1.0
        and item["composed_v4"]["proof_soundness"] == 1.0
        and item["composed_v4"]["false_positives"] == 0
        for item in runs.values()
    )
    lodo_gate_pass = (
        not debug_small
        and len(selected) == len(LMV_DOMAINS)
        and sound
        and no_solve_drop
        and resource_caps_pass
        and composed_gate
        and median_gate_holdouts >= 2
    )
    report = {
        "schema_version": 1,
        "experiment": "language-math-vision-leave-one-domain-out",
        "evaluation_status": "smoke_only" if debug_small else "full_model",
        "config": {
            "held_out_domains": list(selected),
            "examples_per_domain": examples_per_domain,
            "epochs": epochs,
            "seed": seed,
            "device": device,
            "debug_small": debug_small,
            "max_expansions": max_expansions,
        },
        "runs": runs,
        "aggregate": {
            "all_holdouts_metadata_verified": len(runs) == len(selected),
            "proof_soundness_100_percent": sound,
            "solve_rate_drop_within_1pp": no_solve_drop,
            "model_artifact_latency_caps_pass": resource_caps_pass,
            "composed_v4_verified_for_all_holdouts": composed_gate,
            "holdouts_with_any_case_improvement": improved_holdouts,
            "holdouts_with_30_percent_median_reduction": median_gate_holdouts,
            "lodo_transfer_gate_pass": lodo_gate_pass,
            "promotion_ready": False,
            "promotion_blockers": [
                "20/100-shot reviewed-data gate is not evaluated by LODO"
            ],
        },
        "elapsed_seconds": perf_counter() - started,
    }
    report_path = destination / "lodo_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and verify leave-one-domain-out SemOp controllers"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--held-out",
        action="append",
        choices=LMV_DOMAINS,
        help="evaluate only this holdout; repeat for multiple domains",
    )
    parser.add_argument("--examples-per-domain", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-expansions", type=int, default=20_000)
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="use the 5.84M controller; default is a smoke-only 29K model",
    )
    args = parser.parse_args()
    report = run_lodo_experiment(
        args.output_dir,
        held_out_domains=args.held_out or LMV_DOMAINS,
        examples_per_domain=args.examples_per_domain,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        debug_small=not args.full_model,
        max_expansions=args.max_expansions,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["aggregate"]["proof_soundness_100_percent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

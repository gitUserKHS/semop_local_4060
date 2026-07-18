from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (  # noqa: E402
    GroundingLabel,
    LearningSplit,
    SemanticGroundingFeatureProfile,
    generate_controlled_semantic_benchmark,
    generate_structural_semantic_benchmark,
    profile_semantic_grounding_examples,
    programmatic_semantic_learning_examples,
)
from semop.tiny_controller import (  # noqa: E402
    GroundingLearningBudget,
    GroundingPolicyOutcome,
    SparseGroundingPolicy,
    VerifiedGroundingLearningLoop,
    audit_grounding_split,
    evaluate_grounding_policy,
)
from semop.tiny_controller.grounding_learning import (  # noqa: E402
    GroundingReplayBuffer,
)


DOMAINS = ("language", "math", "vision")
DEFAULT_PROFILES = tuple(SemanticGroundingFeatureProfile)


def evaluate_semantic_grounding_ablation(
    *,
    profiles: Sequence[SemanticGroundingFeatureProfile | str] = DEFAULT_PROFILES,
    label_budgets: Sequence[int] = (20, 100),
    validation_per_domain: int = 24,
    seen_test_per_domain: int = 48,
    structural_test_per_domain: int = 40,
    seed: int = 173,
) -> dict[str, Any]:
    selected_profiles = tuple(SemanticGroundingFeatureProfile(item) for item in profiles)
    budgets = tuple(int(item) for item in label_budgets)
    if not selected_profiles or len(set(selected_profiles)) != len(selected_profiles):
        raise ValueError("semantic grounding ablation profiles must be unique and non-empty")
    if not budgets or any(item <= 0 for item in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("semantic grounding ablation budgets must be unique and positive")
    if validation_per_domain < 6 or seen_test_per_domain < 6:
        raise ValueError("controlled validation and seen test sizes must be at least six")
    if structural_test_per_domain < 10:
        raise ValueError("structural test size must be at least ten per domain")

    max_budget = max(budgets)
    training_base = programmatic_semantic_learning_examples(
        generate_controlled_semantic_benchmark(
            per_domain=max_budget,
            split=LearningSplit.TRAIN,
            seed=seed,
            namespace="semantic-ablation-train",
        )
    )
    validation_base = programmatic_semantic_learning_examples(
        generate_controlled_semantic_benchmark(
            per_domain=validation_per_domain,
            split=LearningSplit.HELDOUT,
            seed=seed + 10_000,
            namespace="semantic-ablation-validation",
        )
    )
    seen_test_base = programmatic_semantic_learning_examples(
        generate_controlled_semantic_benchmark(
            per_domain=seen_test_per_domain,
            split=LearningSplit.HELDOUT,
            seed=seed + 20_000,
            namespace="semantic-ablation-seen-test",
        )
    )
    structural_base = programmatic_semantic_learning_examples(
        generate_structural_semantic_benchmark(
            per_domain=structural_test_per_domain,
            split=LearningSplit.HELDOUT,
            seed=seed + 30_000,
            namespace="semantic-ablation-structural-test",
        )
    )

    runs: list[dict[str, Any]] = []
    started_all = perf_counter()
    for profile in selected_profiles:
        validation = profile_semantic_grounding_examples(validation_base, profile)
        seen_test = profile_semantic_grounding_examples(seen_test_base, profile)
        structural_test = profile_semantic_grounding_examples(structural_base, profile)
        validation_buffer = GroundingReplayBuffer.from_examples(validation)
        seen_test_buffer = GroundingReplayBuffer.from_examples(seen_test)
        structural_buffer = GroundingReplayBuffer.from_examples(structural_test)
        validation_seen_audit = audit_grounding_split(
            validation_buffer, seen_test_buffer
        )
        validation_structural_audit = audit_grounding_split(
            validation_buffer, structural_buffer
        )

        for label_budget in budgets:
            started = perf_counter()
            training_base_budget = _select_per_domain(training_base, label_budget)
            training = profile_semantic_grounding_examples(
                training_base_budget, profile
            )
            training_buffer = GroundingReplayBuffer.from_examples(training)
            train_validation_audit = audit_grounding_split(
                training_buffer, validation_buffer
            )
            train_seen_audit = audit_grounding_split(
                training_buffer, seen_test_buffer
            )
            train_structural_audit = audit_grounding_split(
                training_buffer, structural_buffer
            )
            label_counts = {
                label: sum(item.label is label for item in training)
                for label in GroundingLabel
            }
            result = VerifiedGroundingLearningLoop(
                budget=GroundingLearningBudget(
                    min_training_examples=len(training),
                    min_validation_examples=len(validation),
                    min_examples_per_label=max(
                        1,
                        min(min(label_counts.values()), 3),
                    ),
                    required_selective_accuracy=0.90,
                    min_validation_coverage=0.50,
                    min_domain_coverage=0.40,
                    max_false_accepts=0,
                    required_domains=DOMAINS,
                )
            ).run(training_buffer, validation_buffer)

            candidate_policy = (
                result.candidate.policy
                if result.candidate is not None
                else SparseGroundingPolicy()
            )
            candidate_seen = evaluate_grounding_policy(candidate_policy, seen_test)
            candidate_structural = evaluate_grounding_policy(
                candidate_policy, structural_test
            )
            structural_rejections = _structural_rejections(
                promoted=result.promoted,
                metrics=candidate_structural,
                policy=candidate_policy,
            )
            retained = not structural_rejections
            deployment_policy = (
                candidate_policy if retained else SparseGroundingPolicy()
            )
            deployment_structural = evaluate_grounding_policy(
                deployment_policy, structural_test
            )
            split_gates = {
                "train_validation_disjoint": train_validation_audit.valid,
                "train_seen_test_disjoint": train_seen_audit.valid,
                "train_structural_test_disjoint": train_structural_audit.valid,
                "validation_seen_test_disjoint": validation_seen_audit.valid,
                "validation_structural_test_disjoint": (
                    validation_structural_audit.valid
                ),
            }
            run_gates = {
                **split_gates,
                "validation_promoted": result.promoted,
                "structural_false_accepts_zero": (
                    candidate_structural.false_accepts == 0
                ),
                "structural_selective_accuracy_at_least_90pct": (
                    candidate_structural.selective_accuracy >= 0.90
                ),
                "structural_coverage_at_least_40pct": (
                    candidate_structural.coverage >= 0.40
                ),
                "structural_each_domain_coverage_at_least_30pct": all(
                    item.coverage >= 0.30
                    for item in candidate_structural.by_domain
                ),
                "artifact_under_64mb": candidate_policy.artifact_bytes <= 64 * 1024 * 1024,
                "parameters_under_15m": candidate_policy.parameter_count <= 15_000_000,
            }
            run_gates["all_passed"] = all(run_gates.values())
            if retained != run_gates["all_passed"]:
                raise RuntimeError("structural retention and gate result diverged")
            runs.append(
                {
                    "profile": profile.value,
                    "programmatic_labels_per_domain": label_budget,
                    "programmatic_verified_labels": len(training),
                    "accept_labels": label_counts[GroundingLabel.ACCEPT],
                    "hard_negative_labels": label_counts[GroundingLabel.REJECT],
                    "validation_promoted": result.promoted,
                    "retained_after_structural_gate": retained,
                    "validation_rejection_reasons": list(result.rejection_reasons),
                    "structural_rejection_reasons": structural_rejections,
                    "candidate_parameters": candidate_policy.parameter_count,
                    "candidate_artifact_bytes": candidate_policy.artifact_bytes,
                    "deployment_parameters": deployment_policy.parameter_count,
                    "validation_candidate": (
                        None
                        if result.candidate_metrics is None
                        else _metrics_payload(result.candidate_metrics)
                    ),
                    "seen_test_candidate": _metrics_payload(candidate_seen),
                    "structural_test_candidate": _metrics_payload(
                        candidate_structural
                    ),
                    "structural_candidate_errors": _prediction_errors(
                        candidate_policy, structural_test
                    ),
                    "structural_test_deployment": _metrics_payload(
                        deployment_structural
                    ),
                    "gates": run_gates,
                    "checkpoint": None,
                    "elapsed_cpu_seconds": perf_counter() - started,
                }
            )

    shortcut_audit = _shortcut_audit(runs, budgets)
    retained_runs = [run for run in runs if run["retained_after_structural_gate"]]
    fail_closed = all(
        run["retained_after_structural_gate"] == run["gates"]["all_passed"]
        and (
            run["retained_after_structural_gate"]
            or run["deployment_parameters"] == 0
        )
        for run in runs
    )
    full_runs = [run for run in runs if run["profile"] == "full"]
    return {
        "schema_version": 1,
        "benchmark": "typed-semantic-grounding-shortcut-ablation-v1",
        "scope": (
            "controlled training and selection plus unseen language compositions, "
            "deeper exact math, and larger deterministic raster structures"
        ),
        "profiles": tuple(profile.value for profile in selected_profiles),
        "label_budgets_per_domain": budgets,
        "validation_examples_per_domain": validation_per_domain,
        "seen_test_examples_per_domain": seen_test_per_domain,
        "structural_test_examples_per_domain": structural_test_per_domain,
        "seed": seed,
        "runs": runs,
        "shortcut_audit": shortcut_audit,
        "gates": {
            "fail_closed_retention": fail_closed,
            "no_checkpoints_written": all(run["checkpoint"] is None for run in runs),
            "retained_runs_have_zero_structural_false_accepts": all(
                run["structural_test_candidate"]["false_accepts"] == 0
                for run in retained_runs
            ),
            "full_profile_all_budgets_retained": bool(full_runs)
            and all(run["retained_after_structural_gate"] for run in full_runs),
            "shortcut_robustness": shortcut_audit["all_budgets_robust"],
        },
        "elapsed_cpu_seconds": perf_counter() - started_all,
    }


def _select_per_domain(examples, per_domain: int):
    selected = []
    for domain in DOMAINS:
        domain_examples = [
            example for example in examples if example.candidate.domain == domain
        ]
        if len(domain_examples) < per_domain:
            raise ValueError(f"semantic grounding training pool lacks {domain} examples")
        selected.extend(domain_examples[:per_domain])
    return tuple(selected)


def _metrics_payload(metrics) -> dict[str, Any]:
    payload = asdict(metrics)
    payload["by_domain"] = [asdict(item) for item in metrics.by_domain]
    payload["oracle_correct_completion_rate"] = (
        metrics.correct / metrics.examples if metrics.examples else 0.0
    )
    return payload


def _structural_rejections(*, promoted: bool, metrics, policy) -> list[str]:
    reasons: list[str] = []
    if not promoted:
        reasons.append("validation_candidate_not_promoted")
    if metrics.false_accepts:
        reasons.append("structural_false_accepts")
    if metrics.selective_accuracy < 0.90:
        reasons.append("structural_selective_accuracy_below_threshold")
    if metrics.coverage < 0.40:
        reasons.append("structural_coverage_below_threshold")
    if any(item.coverage < 0.30 for item in metrics.by_domain):
        reasons.append("structural_domain_coverage_below_threshold")
    if policy.artifact_bytes > 64 * 1024 * 1024:
        reasons.append("artifact_limit_exceeded")
    if policy.parameter_count > 15_000_000:
        reasons.append("parameter_limit_exceeded")
    return reasons


def _prediction_errors(policy, examples) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for example in examples:
        prediction = policy.predict(example.candidate)
        if prediction.outcome is GroundingPolicyOutcome.ABSTAIN:
            continue
        predicted_label = (
            GroundingLabel.ACCEPT
            if prediction.outcome is GroundingPolicyOutcome.ACCEPT
            else GroundingLabel.REJECT
        )
        if predicted_label is example.label:
            continue
        errors.append(
            {
                "candidate_id": example.candidate.candidate_id,
                "domain": example.candidate.domain,
                "atom": str(example.candidate.atom),
                "expected_label": example.label.value,
                "outcome": prediction.outcome.value,
                "confidence": prediction.confidence,
                "input_digest": example.candidate.input_digest,
                "sensor_features": [
                    list(item) for item in example.candidate.sensor_features
                ],
            }
        )
    return errors


def _shortcut_audit(runs: Sequence[dict[str, Any]], budgets: Sequence[int]):
    by_key = {
        (run["profile"], int(run["programmatic_labels_per_domain"])): run
        for run in runs
    }
    comparisons = []
    robust_values = []
    for budget in budgets:
        full = by_key.get(("full", budget))
        if full is None:
            continue
        full_rate = float(
            full["structural_test_candidate"]["oracle_correct_completion_rate"]
        )
        row: dict[str, Any] = {
            "programmatic_labels_per_domain": budget,
            "full_structural_completion_rate": full_rate,
        }
        budget_robust = True
        for profile, name in (
            ("domain_local_margin", "cross_domain_margin_namespace"),
            ("no_shared_margin", "shared_support_margin"),
            ("no_margin", "all_margin"),
            ("no_surface", "surface"),
            ("primitives_only", "surface_and_margin"),
        ):
            run = by_key.get((profile, budget))
            if run is None:
                row[f"{name}_dependence"] = None
                row[f"{name}_retained_ratio"] = None
                continue
            rate = float(
                run["structural_test_candidate"]["oracle_correct_completion_rate"]
            )
            ratio = rate / full_rate if full_rate else None
            row[f"{profile}_structural_completion_rate"] = rate
            row[f"{name}_dependence"] = full_rate - rate
            row[f"{name}_retained_ratio"] = ratio
            if profile != "no_margin":
                budget_robust = budget_robust and (
                    ratio is not None
                    and ratio >= 0.75
                    and run["structural_test_candidate"]["false_accepts"] == 0
                )
        row["robust_without_shortcuts"] = budget_robust
        robust_values.append(budget_robust)
        comparisons.append(row)
    return {
        "comparisons": comparisons,
        "minimum_retained_completion_ratio": 0.75,
        "all_budgets_robust": bool(robust_values) and all(robust_values),
        "interpretation": (
            "A large positive dependence means the full policy relied on the removed "
            "surface or target-relative margin signal."
        ),
    }


def _parse_profiles(value: str) -> tuple[SemanticGroundingFeatureProfile, ...]:
    try:
        return tuple(
            SemanticGroundingFeatureProfile(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_budgets(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("budgets must be comma-separated integers") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit sparse semantic grounding shortcuts on structural holdouts"
    )
    parser.add_argument(
        "--profiles",
        type=_parse_profiles,
        default=DEFAULT_PROFILES,
        help="comma-separated feature profiles",
    )
    parser.add_argument("--label-budgets", type=_parse_budgets, default=(20, 100))
    parser.add_argument("--validation-per-domain", type=int, default=24)
    parser.add_argument("--seen-test-per-domain", type=int, default=48)
    parser.add_argument("--structural-test-per-domain", type=int, default=40)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate_semantic_grounding_ablation(
        profiles=args.profiles,
        label_budgets=args.label_budgets,
        validation_per_domain=args.validation_per_domain,
        seen_test_per_domain=args.seen_test_per_domain,
        structural_test_per_domain=args.structural_test_per_domain,
        seed=args.seed,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output)
    if args.require_pass and not all(report["gates"].values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

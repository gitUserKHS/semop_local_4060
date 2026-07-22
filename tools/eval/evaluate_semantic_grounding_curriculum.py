from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
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
    generate_extrapolation_semantic_benchmark,
    generate_operator_boundary_semantic_benchmark,
    profile_semantic_grounding_examples,
    programmatic_semantic_learning_examples,
)
from semop.tiny_controller import (  # noqa: E402
    GroundingLearningBudget,
    GroundingPolicyOutcome,
    SparseGroundingPolicy,
    VerifiedGroundingLearningLoop,
    audit_grounding_curriculum_selection,
    audit_grounding_split,
    evaluate_grounding_policy,
    select_feature_novel_grounding_examples,
)
from semop.tiny_controller.grounding_learning import (  # noqa: E402
    GroundingReplayBuffer,
)


DOMAINS = ("language", "math", "vision")
STRATEGIES = ("prefix", "feature_novel")


def evaluate_semantic_grounding_curriculum(
    *,
    label_budgets: Sequence[int] = (5, 20),
    training_pool_per_domain: int = 96,
    validation_per_domain: int = 32,
    extrapolation_per_domain: int = 36,
    profile: SemanticGroundingFeatureProfile | str = (
        SemanticGroundingFeatureProfile.PRIMITIVES_ONLY
    ),
    seed: int = 257,
) -> dict[str, Any]:
    budgets = tuple(int(item) for item in label_budgets)
    selected_profile = SemanticGroundingFeatureProfile(profile)
    if not budgets or any(item < 2 for item in budgets):
        raise ValueError("curriculum label budgets must contain values of at least two")
    if len(set(budgets)) != len(budgets):
        raise ValueError("curriculum label budgets must be unique")
    if training_pool_per_domain < max(budgets):
        raise ValueError("curriculum training pool is smaller than a label budget")
    if validation_per_domain < 6:
        raise ValueError("curriculum validation requires at least six cases per domain")
    if extrapolation_per_domain < 12:
        raise ValueError("curriculum extrapolation requires at least twelve cases per domain")

    training_benchmark = generate_operator_boundary_semantic_benchmark(
        per_domain=training_pool_per_domain,
        split=LearningSplit.TRAIN,
        seed=seed,
        namespace="semantic-curriculum-train",
    )
    validation_benchmark = generate_operator_boundary_semantic_benchmark(
        per_domain=validation_per_domain,
        split=LearningSplit.HELDOUT,
        seed=seed + 10_000,
        namespace="semantic-curriculum-validation",
    )
    extrapolation_benchmark = generate_extrapolation_semantic_benchmark(
        per_domain=extrapolation_per_domain,
        split=LearningSplit.HELDOUT,
        seed=seed + 20_000,
        namespace="semantic-curriculum-final",
    )
    training_pool = _profile_examples(training_benchmark, selected_profile)
    validation = _profile_examples(validation_benchmark, selected_profile)
    extrapolation = _profile_examples(extrapolation_benchmark, selected_profile)
    validation_buffer = GroundingReplayBuffer.from_examples(validation)
    extrapolation_buffer = GroundingReplayBuffer.from_examples(extrapolation)
    validation_final_audit = audit_grounding_split(
        validation_buffer,
        extrapolation_buffer,
    )

    runs: list[dict[str, Any]] = []
    started_all = perf_counter()
    for label_budget in budgets:
        prefix = _select_prefix(training_pool, label_budget)
        selections = {
            "prefix": (
                prefix,
                audit_grounding_curriculum_selection(prefix, training_pool),
                _selection_digest(prefix),
            ),
            "feature_novel": _novel_selection(training_pool, label_budget),
        }
        for strategy in STRATEGIES:
            started = perf_counter()
            training, curriculum_audit, selection_digest = selections[strategy]
            training_buffer = GroundingReplayBuffer.from_examples(training)
            train_validation_audit = audit_grounding_split(
                training_buffer,
                validation_buffer,
            )
            train_final_audit = audit_grounding_split(
                training_buffer,
                extrapolation_buffer,
            )
            label_counts = {
                label: sum(item.label is label for item in training)
                for label in GroundingLabel
            }
            learning = VerifiedGroundingLearningLoop(
                budget=GroundingLearningBudget(
                    min_training_examples=len(training),
                    min_validation_examples=len(validation),
                    min_examples_per_label=max(
                        1,
                        min(min(label_counts.values()), 3),
                    ),
                    required_selective_accuracy=0.90,
                    min_validation_coverage=0.40,
                    min_domain_coverage=0.30,
                    max_false_accepts=0,
                    required_domains=DOMAINS,
                )
            ).run(training_buffer, validation_buffer)
            candidate_policy = (
                learning.candidate.policy
                if learning.candidate is not None
                else SparseGroundingPolicy()
            )
            final_metrics = evaluate_grounding_policy(
                candidate_policy,
                extrapolation_buffer,
            )
            run_gates = {
                "train_validation_disjoint": train_validation_audit.valid,
                "train_final_disjoint": train_final_audit.valid,
                "validation_final_disjoint": validation_final_audit.valid,
                "validation_promoted": learning.promoted,
                "final_false_accepts_zero": final_metrics.false_accepts == 0,
                "final_selective_accuracy_at_least_90pct": (
                    final_metrics.selective_accuracy >= 0.90
                ),
                "final_coverage_at_least_40pct": final_metrics.coverage >= 0.40,
                "final_each_domain_coverage_at_least_30pct": all(
                    item.coverage >= 0.30 for item in final_metrics.by_domain
                ),
                "artifact_under_64mb": (
                    candidate_policy.artifact_bytes <= 64 * 1024 * 1024
                ),
                "parameters_under_15m": (
                    candidate_policy.parameter_count <= 15_000_000
                ),
            }
            run_gates["all_passed"] = all(run_gates.values())
            retained = run_gates["all_passed"]
            deployment_policy = (
                candidate_policy if retained else SparseGroundingPolicy()
            )
            runs.append(
                {
                    "strategy": strategy,
                    "programmatic_labels_per_domain": label_budget,
                    "programmatic_verified_labels": len(training),
                    "accept_labels": label_counts[GroundingLabel.ACCEPT],
                    "hard_negative_labels": label_counts[GroundingLabel.REJECT],
                    "selection_digest": selection_digest,
                    "curriculum_audit": asdict(curriculum_audit),
                    "validation_promoted": learning.promoted,
                    "retained_after_final_gate": retained,
                    "validation_rejection_reasons": list(
                        learning.rejection_reasons
                    ),
                    "final_rejection_reasons": [
                        name for name, passed in run_gates.items() if not passed
                    ],
                    "candidate_parameters": candidate_policy.parameter_count,
                    "candidate_artifact_bytes": candidate_policy.artifact_bytes,
                    "deployment_parameters": deployment_policy.parameter_count,
                    "validation_candidate": (
                        None
                        if learning.candidate_metrics is None
                        else _metrics_payload(learning.candidate_metrics)
                    ),
                    "final_extrapolation_candidate": _metrics_payload(
                        final_metrics
                    ),
                    "final_candidate_errors": _prediction_errors(
                        candidate_policy,
                        extrapolation,
                    ),
                    "final_extrapolation_deployment": _metrics_payload(
                        evaluate_grounding_policy(
                            deployment_policy,
                            extrapolation_buffer,
                        )
                    ),
                    "gates": run_gates,
                    "checkpoint": None,
                    "elapsed_cpu_seconds": perf_counter() - started,
                }
            )

    comparisons = _compare_strategies(runs, budgets)
    split_structure = {
        "training_validation_case_digests_disjoint": _case_digests_disjoint(
            training_benchmark,
            validation_benchmark,
        ),
        "training_final_case_digests_disjoint": _case_digests_disjoint(
            training_benchmark,
            extrapolation_benchmark,
        ),
        "validation_final_case_digests_disjoint": _case_digests_disjoint(
            validation_benchmark,
            extrapolation_benchmark,
        ),
        "training_final_phenomena_disjoint": _phenomena_disjoint(
            training_benchmark,
            extrapolation_benchmark,
        ),
        "validation_final_phenomena_disjoint": _phenomena_disjoint(
            validation_benchmark,
            extrapolation_benchmark,
        ),
    }
    retained_runs = [run for run in runs if run["retained_after_final_gate"]]
    gates = {
        "split_structure_clean": all(split_structure.values()),
        "fail_closed_retention": all(
            run["retained_after_final_gate"] == run["gates"]["all_passed"]
            and (
                run["retained_after_final_gate"]
                or run["deployment_parameters"] == 0
            )
            for run in runs
        ),
        "no_checkpoints_written": all(run["checkpoint"] is None for run in runs),
        "retained_runs_have_zero_final_false_accepts": all(
            run["final_extrapolation_candidate"]["false_accepts"] == 0
            for run in retained_runs
        ),
        "feature_state_coverage_nonregression": all(
            comparison["feature_state_coverage_nonregression"]
            for comparison in comparisons
        ),
        "supported_feature_state_coverage_nonregression": all(
            comparison["supported_feature_state_coverage_nonregression"]
            for comparison in comparisons
        ),
        "final_deployment_completion_nonregression": all(
            comparison["final_deployment_completion_nonregression"]
            for comparison in comparisons
        ),
        "final_candidate_false_accept_nonregression": all(
            comparison["final_candidate_false_accept_nonregression"]
            for comparison in comparisons
        ),
        "development_split_not_used_for_promotion": True,
    }
    return {
        "schema_version": 1,
        "benchmark": "typed-semantic-grounding-feature-novel-development-v1",
        "evaluation_status": "development_extrapolation_not_sealed",
        "sealed_final_evaluated": False,
        "scope": (
            "verified operator-boundary curriculum selection and a development "
            "language, exact-math, deterministic-raster extrapolation split"
        ),
        "profile": selected_profile.value,
        "domains": DOMAINS,
        "strategies": STRATEGIES,
        "label_budgets_per_domain": budgets,
        "training_pool_examples_per_domain": training_pool_per_domain,
        "validation_examples_per_domain": validation_per_domain,
        "final_extrapolation_examples_per_domain": extrapolation_per_domain,
        "seed": seed,
        "split_structure": split_structure,
        "runs": runs,
        "comparisons": comparisons,
        "evidence": {
            "at_least_one_strict_feature_state_gain": any(
                comparison["strict_feature_state_gain"]
                for comparison in comparisons
            ),
            "at_least_one_strict_final_completion_gain": any(
                comparison["strict_final_completion_gain"]
                for comparison in comparisons
            ),
            "all_candidate_final_completion_nonregressing": all(
                comparison["final_candidate_completion_nonregression"]
                for comparison in comparisons
            ),
            "nonpromoted_candidate_regressions": [
                comparison["programmatic_labels_per_domain"]
                for comparison in comparisons
                if not comparison["final_candidate_completion_nonregression"]
            ],
            "development_split_opened_during_feature_design": True,
        },
        "gates": gates,
        "elapsed_cpu_seconds": perf_counter() - started_all,
    }


def _profile_examples(benchmark, profile):
    return profile_semantic_grounding_examples(
        programmatic_semantic_learning_examples(benchmark),
        profile,
    )


def _select_prefix(examples, per_domain: int):
    selected = []
    for domain in DOMAINS:
        candidates = [
            example for example in examples if example.candidate.domain == domain
        ]
        if len(candidates) < per_domain:
            raise ValueError(f"curriculum pool lacks {domain} examples")
        selected.extend(candidates[:per_domain])
    return tuple(selected)


def _novel_selection(examples, per_domain: int):
    selection = select_feature_novel_grounding_examples(
        examples,
        per_domain=per_domain,
    )
    return selection.examples, selection.audit, selection.selection_digest


def _selection_digest(examples) -> str:
    return sha256(
        json.dumps(
            sorted(example.record_digest for example in examples),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _metrics_payload(metrics) -> dict[str, Any]:
    payload = asdict(metrics)
    payload["oracle_correct_completion_rate"] = (
        metrics.correct / metrics.examples if metrics.examples else 0.0
    )
    return payload


def _prediction_errors(policy, examples) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for example in examples:
        prediction = policy.predict(example.candidate)
        if prediction.outcome is GroundingPolicyOutcome.ABSTAIN:
            continue
        predicted = (
            GroundingLabel.ACCEPT
            if prediction.outcome is GroundingPolicyOutcome.ACCEPT
            else GroundingLabel.REJECT
        )
        if predicted is example.label:
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
            }
        )
    return errors


def _compare_strategies(runs, budgets):
    by_key = {
        (run["strategy"], run["programmatic_labels_per_domain"]): run
        for run in runs
    }
    comparisons = []
    for budget in budgets:
        prefix = by_key[("prefix", budget)]
        novel = by_key[("feature_novel", budget)]
        prefix_coverage = prefix["curriculum_audit"]["feature_state_coverage"]
        novel_coverage = novel["curriculum_audit"]["feature_state_coverage"]
        prefix_supported = prefix["curriculum_audit"][
            "supported_feature_state_coverage"
        ]
        novel_supported = novel["curriculum_audit"][
            "supported_feature_state_coverage"
        ]
        prefix_metrics = prefix["final_extrapolation_candidate"]
        novel_metrics = novel["final_extrapolation_candidate"]
        prefix_deployment = prefix["final_extrapolation_deployment"]
        novel_deployment = novel["final_extrapolation_deployment"]
        prefix_completion = prefix_metrics["oracle_correct_completion_rate"]
        novel_completion = novel_metrics["oracle_correct_completion_rate"]
        prefix_deployment_completion = prefix_deployment[
            "oracle_correct_completion_rate"
        ]
        novel_deployment_completion = novel_deployment[
            "oracle_correct_completion_rate"
        ]
        comparisons.append(
            {
                "programmatic_labels_per_domain": budget,
                "prefix_feature_state_coverage": prefix_coverage,
                "feature_novel_feature_state_coverage": novel_coverage,
                "feature_state_coverage_delta": novel_coverage - prefix_coverage,
                "prefix_supported_feature_state_coverage": prefix_supported,
                "feature_novel_supported_feature_state_coverage": novel_supported,
                "supported_feature_state_coverage_delta": (
                    novel_supported - prefix_supported
                ),
                "prefix_final_completion_rate": prefix_completion,
                "feature_novel_final_completion_rate": novel_completion,
                "final_completion_delta": novel_completion - prefix_completion,
                "prefix_final_deployment_completion_rate": (
                    prefix_deployment_completion
                ),
                "feature_novel_final_deployment_completion_rate": (
                    novel_deployment_completion
                ),
                "final_deployment_completion_delta": (
                    novel_deployment_completion - prefix_deployment_completion
                ),
                "prefix_final_false_accepts": prefix_metrics["false_accepts"],
                "feature_novel_final_false_accepts": novel_metrics["false_accepts"],
                "feature_state_coverage_nonregression": (
                    novel_coverage >= prefix_coverage
                ),
                "supported_feature_state_coverage_nonregression": (
                    novel_supported >= prefix_supported
                ),
                "final_candidate_completion_nonregression": (
                    novel_completion >= prefix_completion
                ),
                "final_deployment_completion_nonregression": (
                    novel_deployment_completion >= prefix_deployment_completion
                ),
                "final_candidate_false_accept_nonregression": (
                    novel_metrics["false_accepts"]
                    <= prefix_metrics["false_accepts"]
                ),
                "strict_feature_state_gain": novel_coverage > prefix_coverage,
                "strict_final_completion_gain": novel_completion > prefix_completion,
            }
        )
    return comparisons


def _case_digests_disjoint(first, second) -> bool:
    return {case.digest for case in first.cases}.isdisjoint(
        {case.digest for case in second.cases}
    )


def _phenomena_disjoint(first, second) -> bool:
    return {case.phenomenon for case in first.cases}.isdisjoint(
        {case.phenomenon for case in second.cases}
    )


def _parse_budgets(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "budgets must be comma-separated integers"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare prefix and feature-novel verified grounding curricula"
    )
    parser.add_argument("--label-budgets", type=_parse_budgets, default=(5, 20))
    parser.add_argument("--training-pool-per-domain", type=int, default=96)
    parser.add_argument("--validation-per-domain", type=int, default=32)
    parser.add_argument("--extrapolation-per-domain", type=int, default=36)
    parser.add_argument(
        "--profile",
        choices=tuple(item.value for item in SemanticGroundingFeatureProfile),
        default=SemanticGroundingFeatureProfile.PRIMITIVES_ONLY.value,
    )
    parser.add_argument("--seed", type=int, default=257)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate_semantic_grounding_curriculum(
        label_budgets=args.label_budgets,
        training_pool_per_domain=args.training_pool_per_domain,
        validation_per_domain=args.validation_per_domain,
        extrapolation_per_domain=args.extrapolation_per_domain,
        profile=args.profile,
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

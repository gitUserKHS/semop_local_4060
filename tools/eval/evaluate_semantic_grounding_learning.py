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
    SemanticGroundingCorpus,
    generate_controlled_semantic_benchmark,
    load_semantic_benchmark,
    load_semantic_grounding_reviews,
    programmatic_semantic_learning_examples,
)
from semop.tiny_controller import (  # noqa: E402
    GroundingLearningBudget,
    GroundingPolicyStore,
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
DEFAULT_CASES = ROOT / "data" / "semantic_benchmark" / "v1" / "cases.jsonl"
DEFAULT_REVIEWS = ROOT / "data" / "semantic_grounding" / "v1" / "reviews.jsonl"


def evaluate_semantic_grounding_learning(
    *,
    label_budgets: Sequence[int] = (0, 5, 20, 100),
    validation_per_domain: int = 24,
    test_per_domain: int = 48,
    seed: int = 73,
    semantic_cases_path: str | Path = DEFAULT_CASES,
    semantic_reviews_path: str | Path = DEFAULT_REVIEWS,
    checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    budgets = tuple(int(value) for value in label_budgets)
    if not budgets or any(value < 0 for value in budgets):
        raise ValueError("semantic grounding label budgets must be non-negative")
    if len(set(budgets)) != len(budgets):
        raise ValueError("semantic grounding label budgets must be unique")
    if validation_per_domain < 6 or test_per_domain < 6:
        raise ValueError("semantic grounding validation and test sizes must be at least six")

    max_budget = max(budgets)
    training_pool = (
        ()
        if max_budget == 0
        else programmatic_semantic_learning_examples(
            generate_controlled_semantic_benchmark(
                per_domain=max_budget,
                split=LearningSplit.TRAIN,
                seed=seed,
                namespace="semantic-grounding-train",
            )
        )
    )
    validation = programmatic_semantic_learning_examples(
        generate_controlled_semantic_benchmark(
            per_domain=validation_per_domain,
            split=LearningSplit.HELDOUT,
            seed=seed + 10_000,
            namespace="semantic-grounding-validation",
        )
    )
    untouched_test = programmatic_semantic_learning_examples(
        generate_controlled_semantic_benchmark(
            per_domain=test_per_domain,
            split=LearningSplit.HELDOUT,
            seed=seed + 20_000,
            namespace="semantic-grounding-untouched-test",
        )
    )
    validation_buffer = GroundingReplayBuffer.from_examples(validation)
    test_buffer = GroundingReplayBuffer.from_examples(untouched_test)

    semantic_benchmark = load_semantic_benchmark(semantic_cases_path)
    semantic_reviews = load_semantic_grounding_reviews(semantic_reviews_path)
    semantic_corpus = SemanticGroundingCorpus.compile(
        semantic_benchmark,
        semantic_reviews,
    )
    semantic_audit = semantic_corpus.audit()

    runs: list[dict[str, Any]] = []
    started_all = perf_counter()
    for label_budget in budgets:
        started = perf_counter()
        if label_budget == 0:
            policy = SparseGroundingPolicy()
            validation_metrics = evaluate_grounding_policy(policy, validation)
            test_metrics = evaluate_grounding_policy(policy, untouched_test)
            runs.append(
                {
                    "programmatic_labels_per_domain": 0,
                    "human_reviewed_labels": 0,
                    "promoted": False,
                    "rejection_reasons": ["zero_label_baseline"],
                    "parameters": policy.parameter_count,
                    "artifact_bytes": policy.artifact_bytes,
                    "validation": _metrics_payload(validation_metrics),
                    "untouched_test": _metrics_payload(test_metrics),
                    "semantic_test": _semantic_target_report(policy, semantic_corpus),
                    "elapsed_cpu_seconds": perf_counter() - started,
                }
            )
            continue

        training = _select_per_domain(training_pool, label_budget)
        training_buffer = GroundingReplayBuffer.from_examples(training)
        train_validation_audit = audit_grounding_split(
            training_buffer,
            validation_buffer,
        )
        train_test_audit = audit_grounding_split(training_buffer, test_buffer)
        validation_test_audit = audit_grounding_split(
            validation_buffer,
            test_buffer,
        )
        label_counts = {
            label: sum(item.label is label for item in training)
            for label in GroundingLabel
        }
        store = None
        if checkpoint_root is not None and label_budget == max_budget:
            store = GroundingPolicyStore(
                Path(checkpoint_root) / f"programmatic-{label_budget}-per-domain"
            )
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
            ),
            store=store,
        ).run(training_buffer, validation_buffer)
        test_metrics = evaluate_grounding_policy(
            result.active_policy,
            untouched_test,
        )
        gates = {
            "candidate_promoted": result.promoted,
            "train_validation_lineage_disjoint": train_validation_audit.valid,
            "train_test_lineage_disjoint": train_test_audit.valid,
            "validation_test_lineage_disjoint": validation_test_audit.valid,
            "validation_false_accepts_zero": (
                result.candidate_metrics is not None
                and result.candidate_metrics.false_accepts == 0
            ),
            "untouched_test_false_accepts_zero": test_metrics.false_accepts == 0,
            "untouched_test_selective_accuracy": (
                test_metrics.selective_accuracy >= 0.90
            ),
            "untouched_test_lmv_coverage": all(
                metrics.coverage >= 0.40 for metrics in test_metrics.by_domain
            ),
            "artifact_under_64mb": (
                result.active_policy.artifact_bytes <= 64 * 1024 * 1024
            ),
            "parameters_under_15m": (
                result.active_policy.parameter_count <= 15_000_000
            ),
        }
        gates["all_passed"] = all(gates.values())
        runs.append(
            {
                "programmatic_labels_per_domain": label_budget,
                "programmatic_verified_labels": len(training),
                "accept_labels": label_counts[GroundingLabel.ACCEPT],
                "hard_negative_labels": label_counts[GroundingLabel.REJECT],
                "human_reviewed_labels": 0,
                "promoted": result.promoted,
                "rejection_reasons": list(result.rejection_reasons),
                "parameters": result.active_policy.parameter_count,
                "artifact_bytes": result.active_policy.artifact_bytes,
                "training_updates": (
                    result.candidate.training_updates if result.candidate else 0
                ),
                "validation": (
                    None
                    if result.candidate_metrics is None
                    else _metrics_payload(result.candidate_metrics)
                ),
                "untouched_test": _metrics_payload(test_metrics),
                "semantic_test": _semantic_target_report(
                    result.active_policy,
                    semantic_corpus,
                ),
                "gates": gates,
                "checkpoint": (
                    None if result.checkpoint is None else str(result.checkpoint)
                ),
                "elapsed_cpu_seconds": perf_counter() - started,
            }
        )

    learned_runs = [
        run for run in runs if run["programmatic_labels_per_domain"] > 0
    ]
    run_by_budget = {
        int(run["programmatic_labels_per_domain"]): run for run in runs
    }
    twenty_rate = _completion_rate(run_by_budget.get(20))
    hundred_rate = _completion_rate(run_by_budget.get(100))
    twenty_to_hundred = (
        twenty_rate / hundred_rate
        if twenty_rate is not None and hundred_rate not in {None, 0.0}
        else None
    )
    approved_by_domain = dict(semantic_audit.domain_approved_counts)
    human_curve = tuple(
        _human_budget_status(budget, approved_by_domain)
        for budget in budgets
    )
    return {
        "schema_version": 1,
        "benchmark": "typed-semantic-grounding-transfer-v1",
        "benchmark_scope": (
            "controlled raw language, exact math, and deterministic raster inputs "
            "through production adapters; not open-domain language or natural vision"
        ),
        "domains": DOMAINS,
        "programmatic_label_budgets_per_domain": budgets,
        "validation_examples_per_domain": validation_per_domain,
        "untouched_test_examples_per_domain": test_per_domain,
        "seed": seed,
        "runs": runs,
        "candidate_review_audit": semantic_audit.to_dict(),
        "human_review_curve": human_curve,
        "human_semantic_gold_evaluated": bool(semantic_audit.approved_targets),
        "data_efficiency": {
            "twenty_shot_completion_rate": twenty_rate,
            "hundred_shot_completion_rate": hundred_rate,
            "twenty_to_hundred_ratio": twenty_to_hundred,
        },
        "gates": {
            "all_programmatic_learned_budgets_pass": bool(learned_runs)
            and all(run["gates"]["all_passed"] for run in learned_runs),
            "zero_shot_abstains": (
                bool(runs)
                and runs[0]["programmatic_labels_per_domain"] == 0
                and runs[0]["untouched_test"]["coverage"] == 0.0
            ),
            "candidate_reviews_digest_clean": semantic_audit.clean,
            "human_semantic_gate": (
                "evaluated" if semantic_audit.approved_targets else "not_evaluated"
            ),
            "curated_labels_excluded_from_promotion": True,
            "twenty_shot_at_least_90_percent_of_100_shot": (
                "not_evaluated"
                if twenty_to_hundred is None
                else twenty_to_hundred >= 0.90
            ),
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
            raise ValueError(
                f"semantic grounding training pool lacks {domain} examples"
            )
        selected.extend(domain_examples[:per_domain])
    return tuple(selected)


def _metrics_payload(metrics) -> dict[str, Any]:
    payload = asdict(metrics)
    payload["by_domain"] = [asdict(item) for item in metrics.by_domain]
    payload["oracle_correct_completion_rate"] = (
        metrics.correct / metrics.examples if metrics.examples else 0.0
    )
    return payload


def _semantic_target_report(
    policy: SparseGroundingPolicy,
    corpus: SemanticGroundingCorpus,
) -> dict[str, Any]:
    reviews = corpus.review_by_id
    approved_targets = set(corpus.audit().approved_targets)
    total = decided = correct = false_accepts = abstained = 0
    reviewed_total = reviewed_decided = reviewed_correct = 0
    outcomes = []
    for target in corpus.targets:
        review = reviews.get(target.target_id)
        reviewed = bool(
            review is not None
            and target.target_id in approved_targets
        )
        label = review.reviewed_label if reviewed and review else target.proposed_label
        prediction = policy.predict(target.candidate)
        is_decided = prediction.outcome is not GroundingPolicyOutcome.ABSTAIN
        predicted_label = (
            GroundingLabel.ACCEPT
            if prediction.outcome is GroundingPolicyOutcome.ACCEPT
            else GroundingLabel.REJECT
        )
        is_correct = is_decided and predicted_label is label
        total += 1
        decided += int(is_decided)
        correct += int(is_correct)
        false_accepts += int(
            prediction.outcome is GroundingPolicyOutcome.ACCEPT
            and label is GroundingLabel.REJECT
        )
        abstained += int(not is_decided)
        if reviewed:
            reviewed_total += 1
            reviewed_decided += int(is_decided)
            reviewed_correct += int(is_correct)
        outcomes.append(
            {
                "target_id": target.target_id,
                "domain": target.domain.value,
                "phenomenon": target.phenomenon,
                "label": label.value,
                "label_authority": (
                    "human_review" if reviewed else "curated_unreviewed"
                ),
                "outcome": prediction.outcome.value,
                "confidence": prediction.confidence,
                "correct_if_label_is_valid": is_correct,
            }
        )
    return {
        "targets": total,
        "human_reviewed_targets": reviewed_total,
        "curated_unreviewed_targets": total - reviewed_total,
        "coverage": decided / total if total else 0.0,
        "selective_accuracy_if_labels_are_valid": (
            correct / decided if decided else 0.0
        ),
        "false_accepts_if_labels_are_valid": false_accepts,
        "abstained": abstained,
        "human_reviewed_coverage": (
            reviewed_decided / reviewed_total if reviewed_total else None
        ),
        "human_reviewed_selective_accuracy": (
            reviewed_correct / reviewed_decided if reviewed_decided else None
        ),
        "used_for_promotion": False,
        "outcomes": outcomes,
    }


def _human_budget_status(
    budget: int,
    available_by_domain: dict[str, int],
) -> dict[str, Any]:
    available = {domain: available_by_domain.get(domain, 0) for domain in DOMAINS}
    enough = budget > 0 and all(count >= budget for count in available.values())
    return {
        "human_labels_per_domain": budget,
        "available_by_domain": available,
        "status": "ready_for_split_design" if enough else "not_evaluated",
        "reason": (
            "digest-bound labels exist, but train/calibration/test roles must remain "
            "separate before evaluation"
            if enough
            else "insufficient digest-bound candidate reviews in one or more domains"
        ),
    }


def _completion_rate(run: dict[str, Any] | None) -> float | None:
    if run is None:
        return None
    return float(run["untouched_test"]["oracle_correct_completion_rate"])


def _parse_budgets(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("label budgets must be comma-separated integers") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate sparse grounding learning on raw controlled language, math, "
            "and raster inputs"
        )
    )
    parser.add_argument("--label-budgets", type=_parse_budgets, default=(0, 5, 20, 100))
    parser.add_argument("--validation-per-domain", type=int, default=24)
    parser.add_argument("--test-per-domain", type=int, default=48)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--semantic-cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--semantic-reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = evaluate_semantic_grounding_learning(
        label_budgets=args.label_budgets,
        validation_per_domain=args.validation_per_domain,
        test_per_domain=args.test_per_domain,
        seed=args.seed,
        semantic_cases_path=args.semantic_cases,
        semantic_reviews_path=args.semantic_reviews,
        checkpoint_root=args.checkpoint_root,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

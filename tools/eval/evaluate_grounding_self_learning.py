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

from semop.kernel import (
    AssertionStatus,
    GroundingAuthority,
    GroundingLabel,
    GroundingTrace,
    KernelRegistry,
    LanguageTextAdapter,
    RasterImage,
    RasterVisionAdapter,
    RasterVisionProblem,
    grounding_payload_digest,
    make_grounding_candidate,
    parse_arithmetic_expression,
    promote_grounding_proposal,
    stage_grounding_proposal,
)
from semop.tiny_controller import (
    GroundingLearningBudget,
    GroundingPolicyStore,
    SparseGroundingPolicy,
    VerifiedGroundingLearningLoop,
    audit_grounding_split,
    evaluate_grounding_policy,
)
from semop.tiny_controller.grounding_learning import GroundingReplayBuffer


DOMAINS = ("language", "math", "vision")


def evaluate_grounding_self_learning(
    *,
    label_budgets: Sequence[int] = (0, 5, 20, 100),
    validation_per_domain: int = 25,
    test_per_domain: int = 50,
    seed: int = 41,
    checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    budgets = tuple(int(value) for value in label_budgets)
    if not budgets or any(value < 0 for value in budgets):
        raise ValueError("grounding label budgets must be non-negative")
    if len(set(budgets)) != len(budgets):
        raise ValueError("grounding label budgets must be unique")
    if validation_per_domain < 5 or test_per_domain < 5:
        raise ValueError("grounding validation and test sizes must be at least five")

    validation = _generate_examples(
        "validation",
        validation_per_domain,
        seed=seed + 10_000,
    )
    untouched_test = _generate_examples(
        "untouched-test",
        test_per_domain,
        seed=seed + 20_000,
        novel_sensor_every=10,
    )
    test_buffer = GroundingReplayBuffer.from_examples(untouched_test)
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
                    "verified_labels_per_domain": 0,
                    "synthetic_verified_labels": 0,
                    "human_reviewed_labels": 0,
                    "promoted": False,
                    "rejection_reasons": ["zero_label_baseline"],
                    "parameters": policy.parameter_count,
                    "artifact_bytes": policy.artifact_bytes,
                    "validation": _metrics_payload(validation_metrics),
                    "untouched_test": _metrics_payload(test_metrics),
                    "elapsed_cpu_seconds": perf_counter() - started,
                }
            )
            continue

        training = _generate_examples(
            f"training-{label_budget}",
            label_budget,
            seed=seed + label_budget,
        )
        training_buffer = GroundingReplayBuffer.from_examples(training)
        validation_buffer = GroundingReplayBuffer.from_examples(validation)
        train_validation_audit = audit_grounding_split(
            training_buffer,
            validation_buffer,
        )
        train_test_audit = audit_grounding_split(training_buffer, test_buffer)
        validation_test_audit = audit_grounding_split(validation_buffer, test_buffer)
        store = None
        if checkpoint_root is not None and label_budget == max(budgets):
            store = GroundingPolicyStore(
                Path(checkpoint_root) / f"verified-{label_budget}-per-domain"
            )
        positives = sum(item.label is GroundingLabel.ACCEPT for item in training)
        negatives = len(training) - positives
        result = VerifiedGroundingLearningLoop(
            budget=GroundingLearningBudget(
                min_training_examples=len(training),
                min_validation_examples=len(validation),
                min_examples_per_label=max(1, min(positives, negatives, 3)),
                required_selective_accuracy=0.95,
                min_validation_coverage=0.80,
                min_domain_coverage=0.75,
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
                test_metrics.selective_accuracy >= 0.95
            ),
            "untouched_test_lmv_coverage": all(
                metrics.coverage >= 0.75 for metrics in test_metrics.by_domain
            ),
            "artifact_under_64mb": result.active_policy.artifact_bytes <= 64 * 1024 * 1024,
            "parameters_under_15m": result.active_policy.parameter_count <= 15_000_000,
        }
        gates["all_passed"] = all(gates.values())
        runs.append(
            {
                "verified_labels_per_domain": label_budget,
                "synthetic_verified_labels": len(training),
                "accept_labels": positives,
                "hard_negative_labels": negatives,
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
                "gates": gates,
                "checkpoint": (
                    None if result.checkpoint is None else str(result.checkpoint)
                ),
                "elapsed_cpu_seconds": perf_counter() - started,
            }
        )

    learned_runs = [item for item in runs if item["verified_labels_per_domain"] > 0]
    adapter_probe = _adapter_sensor_probe()
    return {
        "schema_version": 1,
        "benchmark": "typed-grounding-selective-contract-v1",
        "benchmark_scope": (
            "contract-level synthetic sensor consistency; not open-domain semantic "
            "language or natural-image understanding"
        ),
        "label_authority": GroundingAuthority.EXTERNAL_VERIFIER.value,
        "human_semantic_gold_evaluated": False,
        "domains": DOMAINS,
        "label_budgets_per_domain": budgets,
        "validation_examples_per_domain": validation_per_domain,
        "untouched_test_examples_per_domain": test_per_domain,
        "hard_negative_ratio": 4,
        "seed": seed,
        "runs": runs,
        "adapter_probe": adapter_probe,
        "gates": {
            "all_learned_budgets_pass": bool(learned_runs)
            and all(item["gates"]["all_passed"] for item in learned_runs),
            "zero_shot_abstains": (
                not runs[0]["promoted"]
                if runs and runs[0]["verified_labels_per_domain"] == 0
                else True
            ),
            "human_semantic_claim_withheld": True,
            "production_adapter_contracts_present": all(
                item["all_records_have_sensor_features"]
                for item in adapter_probe.values()
            ),
            "production_adapter_grounding_coverage": all(
                item["grounding_coverage_complete"]
                for item in adapter_probe.values()
            ),
        },
        "elapsed_cpu_seconds": perf_counter() - started_all,
    }


def _adapter_sensor_probe() -> dict[str, dict[str, object]]:
    white = (255, 255, 255)
    red = (255, 0, 0)
    instances = {
        "language": LanguageTextAdapter().adapt(
            "Goal: deploy; Requires: tests; Satisfied: tests"
        ),
        "math": parse_arithmetic_expression("2 * (3 + 4)"),
        "vision": RasterVisionAdapter().adapt(
            RasterVisionProblem(
                RasterImage.from_rows(
                    (
                        (white, white, white, white),
                        (white, red, red, white),
                        (white, red, red, white),
                        (white, white, white, white),
                    )
                )
            )
        ),
    }
    report: dict[str, dict[str, object]] = {}
    for domain, instance in instances.items():
        trace = instance.grounding_trace
        report[domain] = {
            "records": len(trace.records),
            "materialized_facts": len(trace.facts),
            "verified_learning_examples": len(trace.learning_examples),
            "all_records_have_sensor_features": bool(trace.records)
            and all(record.candidate.sensor_features for record in trace.records),
            "grounding_coverage_complete": trace.audit(
                instance.state.facts
            ).coverage_complete,
        }
    return report


def _generate_examples(
    split: str,
    per_domain: int,
    *,
    seed: int,
    novel_sensor_every: int = 0,
):
    examples = []
    for domain in DOMAINS:
        for index in range(per_domain):
            accepted = index % 5 == 0
            # A five-example/domain fixture must support both common negative
            # magnitude states at least twice across the shared LMV policy.
            magnitude = 0.25 if index % 3 == 2 else 1.0
            signal = magnitude if accepted else -magnitude
            sensor_name = (
                f"novel.{domain}.consistency"
                if novel_sensor_every and (index + 1) % novel_sensor_every == 0
                else "measurement.consistency"
            )
            examples.append(
                _verified_example(
                    domain,
                    f"{split}:{domain}:{seed}:{index}",
                    accepted=accepted,
                    signal=signal,
                    sensor_name=sensor_name,
                )
            )
    return tuple(examples)


def _verified_example(
    domain: str,
    input_key: str,
    *,
    accepted: bool,
    signal: float,
    sensor_name: str,
):
    registry = KernelRegistry()
    entity = registry.types.register("Entity")
    registry.register_predicate("SUPPORTED", (entity,))
    symbol = registry.symbol(f"entity_{grounding_payload_digest(input_key)[:10]}", entity)
    candidate = make_grounding_candidate(
        domain=domain,
        statement=f"{symbol.name} has typed support",
        atom=registry.atom("SUPPORTED", symbol),
        producer_id=f"{domain}_sensor_candidate_v1",
        source=f"{domain}_sensor_candidate",
        input_digest=grounding_payload_digest(input_key),
        assertion_status=AssertionStatus.INFERRED,
        confidence=0.8,
        sensor_features=((sensor_name, signal), ("sensor.quality", 1.0)),
    )
    proposed = stage_grounding_proposal(
        candidate,
        authority=GroundingAuthority.MODEL_PROPOSAL,
        verifier_id="synthetic-contract-grounder",
        rationale="synthetic sensor proposed a typed concept",
        confidence=0.8,
    )
    decided = promote_grounding_proposal(
        proposed,
        lambda _atom: accepted,
        authority=GroundingAuthority.EXTERNAL_VERIFIER,
        verifier_id="synthetic-contract-verifier",
        rationale="bounded fixture independently evaluated sensor consistency",
    )
    return GroundingTrace((decided,)).learning_examples[0]


def _metrics_payload(metrics) -> dict[str, Any]:
    payload = asdict(metrics)
    payload["by_domain"] = [asdict(item) for item in metrics.by_domain]
    return payload


def _parse_budgets(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("label budgets must be comma-separated integers") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate verifier-gated sparse grounding self-learning."
    )
    parser.add_argument("--label-budgets", type=_parse_budgets, default=(0, 5, 20, 100))
    parser.add_argument("--validation-per-domain", type=int, default=25)
    parser.add_argument("--test-per-domain", type=int, default=50)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = evaluate_grounding_self_learning(
        label_budgets=args.label_budgets,
        validation_per_domain=args.validation_per_domain,
        test_per_domain=args.test_per_domain,
        seed=args.seed,
        checkpoint_root=args.checkpoint_root,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["gates"]["all_learned_budgets_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

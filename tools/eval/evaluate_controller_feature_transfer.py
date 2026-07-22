from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (  # noqa: E402
    OperatorKernel,
    SolveBudget,
    build_decision_training_cases,
    generate_symbolic_curriculum,
)
from semop.tiny_controller import (  # noqa: E402
    ControllerFeatureProfile,
    StructuralPolicyLearner,
    canonicalize_problem,
)


LMV_DOMAINS = ("language", "math", "vision")
FEATURE_PROFILES = (
    ControllerFeatureProfile.FULL,
    ControllerFeatureProfile.TYPED_STRUCTURE,
)


def evaluate_controller_feature_transfer(
    *,
    training_per_domain: int = 8,
    test_per_domain: int = 12,
    seed: int = 317,
    max_expansions: int = 20_000,
) -> dict[str, Any]:
    """Measure whether action ranking transfers without domain identity tokens."""

    if training_per_domain <= 0 or test_per_domain <= 0:
        raise ValueError("controller transfer fixture counts must be positive")
    if max_expansions <= 0:
        raise ValueError("controller transfer expansion limit must be positive")

    started = perf_counter()
    training = {
        domain: _verified_cases(
            domain,
            training_per_domain,
            seed=seed + index * 1_000,
            max_expansions=max_expansions,
        )
        for index, domain in enumerate(LMV_DOMAINS)
    }
    heldout = {
        domain: _verified_cases(
            domain,
            test_per_domain,
            seed=seed + 10_000 + index * 1_000,
            max_expansions=max_expansions,
        )
        for index, domain in enumerate(LMV_DOMAINS)
    }

    profile_reports = {
        profile.value: _evaluate_profile(
            profile,
            training,
            heldout,
            max_expansions=max_expansions,
        )
        for profile in FEATURE_PROFILES
    }
    full = profile_reports[ControllerFeatureProfile.FULL.value]
    typed = profile_reports[ControllerFeatureProfile.TYPED_STRUCTURE.value]
    gates = {
        "typed_profile_hides_identity_tokens": (
            typed["feature_audit"]["identity_token_count"] == 0
        ),
        "full_profile_control_exposes_identity_tokens": (
            full["feature_audit"]["identity_token_count"] > 0
        ),
        "typed_profile_compacts_feature_vocabulary": (
            typed["feature_audit"]["union_feature_count"]
            < full["feature_audit"]["union_feature_count"]
        ),
        "heldout_domains_absent_from_training": all(
            run["held_out_domain"] not in run["trained_domains"]
            for run in typed["leave_one_domain_out"].values()
        ),
        "typed_profile_replay_integrity_100_percent": all(
            run["replay_integrity"] == 1.0
            for run in typed["leave_one_domain_out"].values()
        ),
        "typed_profile_no_solve_rate_drop": all(
            run["guided_verified_solve_rate"]
            >= run["unguided_verified_solve_rate"]
            for run in typed["leave_one_domain_out"].values()
        ),
        "typed_profile_action_top1_at_least_50_percent": all(
            run["action_top1_accuracy"] >= 0.50
            for run in typed["leave_one_domain_out"].values()
        ),
        "typed_sparse_artifacts_under_limits": all(
            run["parameter_count"] <= 15_000_000
            and run["artifact_bytes"] <= 64 * 1024 * 1024
            for run in typed["leave_one_domain_out"].values()
        ),
    }
    gates["all_passed"] = all(gates.values())
    return {
        "schema_version": 1,
        "benchmark": "lmv-controller-feature-transfer-v1",
        "claim_scope": (
            "verified symbolic LMV operator curricula with synthetic distractors; "
            "not open-domain language, advanced mathematics, or natural vision"
        ),
        "training_examples_per_domain": training_per_domain,
        "heldout_examples_per_domain": test_per_domain,
        "seed": seed,
        "profiles": profile_reports,
        "gates": gates,
        "elapsed_cpu_seconds": perf_counter() - started,
    }


def _verified_cases(
    domain: str,
    count: int,
    *,
    seed: int,
    max_expansions: int,
):
    problems = generate_symbolic_curriculum(
        count,
        seed=seed,
        curriculum="language-math-vision",
        domains=(domain,),
    )
    prepared = []
    for problem in problems:
        kernel = OperatorKernel(problem.instance.registry)
        result = kernel.solve(
            problem.instance.state,
            problem.instance.goals,
            budget=SolveBudget(max_steps=6, max_expansions=max_expansions),
        )
        if not result.success or not result.verified:
            raise RuntimeError(
                f"controller transfer fixture was not replay verified: {problem.problem_id}"
            )
        cases = build_decision_training_cases(
            kernel,
            result,
            negatives_per_positive=4,
        )
        if not cases:
            raise RuntimeError(
                f"controller transfer fixture has no action decisions: {problem.problem_id}"
            )
        prepared.append((problem, result, cases))
    if len(prepared) != count:
        raise RuntimeError(f"controller transfer fixture count mismatch for {domain}")
    return tuple(prepared)


def _evaluate_profile(
    profile: ControllerFeatureProfile,
    training,
    heldout,
    *,
    max_expansions: int,
) -> dict[str, Any]:
    domain_features = {
        domain: _domain_feature_set(items, profile)
        for domain, items in training.items()
    }
    identity_tokens = tuple(
        sorted(
            {
                token
                for items in training.values()
                for _problem, _result, cases in items
                for case in cases
                for token in _identity_tokens(
                    canonicalize_problem(
                        case.state,
                        case.goals,
                        case.actions,
                        feature_profile=profile,
                    )
                )
            }
        )
    )
    overlaps = {
        f"{left}-{right}": _jaccard(domain_features[left], domain_features[right])
        for index, left in enumerate(LMV_DOMAINS)
        for right in LMV_DOMAINS[index + 1 :]
    }
    union_features = frozenset().union(*domain_features.values())
    lodo = {}
    for held_out_domain in LMV_DOMAINS:
        trained_domains = tuple(
            domain for domain in LMV_DOMAINS if domain != held_out_domain
        )
        training_cases = tuple(
            case
            for domain in trained_domains
            for _problem, _result, cases in training[domain]
            for case in cases
        )
        candidate = StructuralPolicyLearner(
            feature_profile=profile,
            action_limit_score_margin=100.0,
        ).train(training_cases)
        policy = candidate.policy
        test_items = heldout[held_out_domain]
        decisions = tuple(
            case for _problem, _result, cases in test_items for case in cases
        )
        top1_correct = sum(_top1_correct(policy, case) for case in decisions)
        unguided_success = guided_success = replay_verified = 0
        unguided_expansions: list[int] = []
        guided_expansions: list[int] = []
        for problem, baseline, _cases in test_items:
            kernel = OperatorKernel(problem.instance.registry)
            guided = kernel.solve(
                problem.instance.state,
                problem.instance.goals,
                policy=policy,
                budget=SolveBudget(max_steps=6, max_expansions=max_expansions),
            )
            unguided_ok = baseline.success and baseline.verified
            guided_ok = guided.success and guided.verified
            unguided_success += int(unguided_ok)
            guided_success += int(guided_ok)
            unguided_expansions.append(baseline.expansions)
            guided_expansions.append(guided.expansions)
            if guided_ok:
                replay = kernel.replay(
                    guided.initial_state,
                    problem.instance.goals,
                    guided.proof,
                )
                replay_verified += int(replay.verified)
        lodo[held_out_domain] = {
            "held_out_domain": held_out_domain,
            "trained_domains": list(trained_domains),
            "training_decisions": len(training_cases),
            "heldout_decisions": len(decisions),
            "action_top1_accuracy": top1_correct / len(decisions),
            "unguided_verified_solve_rate": unguided_success / len(test_items),
            "guided_verified_solve_rate": guided_success / len(test_items),
            "replay_integrity": (
                replay_verified / guided_success if guided_success else 0.0
            ),
            "unguided_median_expansions": statistics.median(unguided_expansions),
            "guided_median_expansions": statistics.median(guided_expansions),
            "parameter_count": candidate.parameter_count,
            "artifact_bytes": len(candidate.artifact),
            "training_updates": candidate.training_updates,
        }
    return {
        "feature_profile": profile.value,
        "feature_audit": {
            "identity_token_count": len(identity_tokens),
            "identity_token_examples": list(identity_tokens[:20]),
            "feature_count_by_domain": {
                domain: len(features)
                for domain, features in domain_features.items()
            },
            "union_feature_count": len(union_features),
            "pairwise_jaccard": overlaps,
            "mean_pairwise_jaccard": statistics.fmean(overlaps.values()),
        },
        "leave_one_domain_out": lodo,
    }


def _domain_feature_set(items, profile: ControllerFeatureProfile) -> frozenset[str]:
    return frozenset(
        feature
        for _problem, _result, cases in items
        for case in cases
        for action in canonicalize_problem(
            case.state,
            case.goals,
            case.actions,
            feature_profile=profile,
        ).actions
        for feature in action.operator_features
    )


def _identity_tokens(graph) -> tuple[str, ...]:
    exposed: set[str] = set()
    for action in graph.actions:
        if not action.operator.startswith("family:"):
            exposed.add(f"action:{action.operator}")
        for feature in action.operator_features:
            if feature.startswith(("schema:", "tag:")):
                exposed.add(feature)
            elif feature.startswith(("precondition:", "effect:")) and not feature.startswith(
                ("precondition:typed:", "effect:typed:")
            ):
                exposed.add(feature)
    for relation in graph.relations + graph.goals:
        if not relation.name.startswith("typed:"):
            exposed.add(f"relation:{relation.name}")
    for tokens in graph.node_tokens:
        for token in tokens:
            if token.startswith("anon:") or ":node:" in token:
                exposed.add(token)
            elif token.startswith("role:") and ":typed:" not in token:
                exposed.add(token)
            elif token.startswith("function:") and not token.startswith(
                "function:typed:"
            ):
                exposed.add(token)
    return tuple(sorted(exposed))


def _top1_correct(policy, case) -> int:
    decision = policy.score_actions(case.state, case.goals, case.actions)
    predicted = max(
        range(len(decision.action_scores)),
        key=lambda index: (decision.action_scores[index], -index),
    )
    return int(predicted == case.target_action)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit domain-identity shortcuts and run sparse LMV leave-one-domain-out "
            "operator-policy transfer."
        )
    )
    parser.add_argument("--training-per-domain", type=int, default=8)
    parser.add_argument("--test-per-domain", type=int, default=12)
    parser.add_argument("--seed", type=int, default=317)
    parser.add_argument("--max-expansions", type=int, default=20_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = evaluate_controller_feature_transfer(
        training_per_domain=args.training_per_domain,
        test_per_domain=args.test_per_domain,
        seed=args.seed,
        max_expansions=args.max_expansions,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.require_pass and not report["gates"]["all_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

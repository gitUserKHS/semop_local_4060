from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (  # noqa: E402
    SelfLearningBudget,
    SolveBudget,
    evaluate_semantic_benchmark,
    load_semantic_benchmark,
    semantic_promotion_rejections,
)


DEFAULT_CASES = ROOT / "data" / "semantic_benchmark" / "v1" / "cases.jsonl"
DEFAULT_REVIEWS = ROOT / "data" / "semantic_benchmark" / "v1" / "reviews.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate digest-bound language, math, and vision semantic cases "
            "through the replay-verified typed kernel."
        )
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--reviewed-only", action="store_true")
    parser.add_argument("--require-all-reviewed", action="store_true")
    parser.add_argument("--gate-semantic-correctness", type=float)
    parser.add_argument("--gate-min-gold", type=int, default=0)
    parser.add_argument("--gate-min-gold-per-domain", type=int, default=0)
    parser.add_argument(
        "--gate-domains",
        default="",
        help="Comma-separated domains that must each contain reviewed gold.",
    )
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--max-expansions", type=int, default=20_000)
    parser.add_argument("--max-facts", type=int, default=10_000)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        benchmark = load_semantic_benchmark(args.cases, args.reviews)
        evaluation = evaluate_semantic_benchmark(
            benchmark,
            reviewed_only=args.reviewed_only,
            require_all_reviewed=args.require_all_reviewed,
            budget=SolveBudget(
                max_steps=args.max_steps,
                max_expansions=args.max_expansions,
                max_facts=args.max_facts,
                timeout_seconds=args.timeout_seconds,
            ),
        )
        gate_budget = _gate_budget(args)
        gate_rejections = (
            semantic_promotion_rejections(evaluation.metrics, gate_budget)
            if gate_budget is not None
            else ()
        )
    except Exception as exc:
        if args.format == "json":
            print(
                json.dumps(
                    {"ok": False, "error_type": type(exc).__name__, "error": str(exc)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            print(f"semantic benchmark error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        payload = evaluation.to_dict()
        payload["promotion_gate"] = {
            "enabled": gate_budget is not None,
            "passed": not gate_rejections,
            "rejection_reasons": list(gate_rejections),
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(_render_text(evaluation, gate_budget is not None, gate_rejections))
    return 0 if evaluation.passed and not gate_rejections else 2


def _render_text(evaluation, gate_enabled: bool, gate_rejections) -> str:
    audit = evaluation.audit
    metrics = evaluation.metrics
    semantic = (
        "n/a"
        if metrics.semantic_correctness is None
        else f"{metrics.semantic_correctness:.3f}"
    )
    lines = [
        "SemOp LMV semantic benchmark",
        f"cases: {audit.cases}",
        (
            "human-reviewed: "
            f"{len(audit.approved_cases)}/{audit.cases} "
            f"({audit.review_coverage:.1%})"
        ),
        f"labeled outcome accuracy: {metrics.labeled_outcome_accuracy:.3f}",
        f"semantic correctness: {semantic}",
        f"primitive replay integrity: {metrics.primitive_replay_integrity:.3f}",
        f"benchmark passed: {str(evaluation.passed).lower()}",
        "",
        "By domain",
    ]
    for domain in metrics.by_domain:
        domain_semantic = (
            "n/a"
            if domain.semantic_correctness is None
            else f"{domain.semantic_correctness:.3f}"
        )
        lines.append(
            f"- {domain.domain}: tasks={domain.tasks}, "
            f"labeled={domain.labeled_outcome_accuracy:.3f}, "
            f"semantic={domain_semantic}, reviewed={domain.semantic_gold_tasks}"
        )
    mismatches = tuple(outcome for outcome in evaluation.outcomes if not outcome.correct)
    if mismatches:
        lines.extend(("", "Mismatches"))
        lines.extend(
            f"- {item.case_id}: expected={item.expected_solved}, actual={item.success}"
            for item in mismatches
        )
    if audit.pending_cases:
        lines.extend(
            (
                "",
                "Review status",
                "- Seed cases are curated_unreviewed, not semantic gold.",
                f"- Pending approvals: {len(audit.pending_cases)}",
            )
        )
    if gate_enabled:
        lines.extend(
            (
                "",
                "Self-learning promotion gate",
                f"- Passed: {str(not gate_rejections).lower()}",
            )
        )
        lines.extend(f"- {reason}" for reason in gate_rejections)
    return "\n".join(lines)


def _gate_budget(args) -> SelfLearningBudget | None:
    domains = tuple(
        item.strip() for item in args.gate_domains.split(",") if item.strip()
    )
    enabled = (
        args.gate_semantic_correctness is not None
        or args.gate_min_gold != 0
        or args.gate_min_gold_per_domain != 0
        or bool(domains)
    )
    if not enabled:
        return None
    return SelfLearningBudget(
        required_semantic_correctness=args.gate_semantic_correctness,
        min_semantic_gold_tasks=args.gate_min_gold,
        min_semantic_gold_tasks_per_domain=args.gate_min_gold_per_domain,
        required_semantic_domains=domains,
    )


if __name__ == "__main__":
    raise SystemExit(main())

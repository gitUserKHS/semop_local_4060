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
    SolveBudget,
    evaluate_semantic_benchmark,
    load_semantic_benchmark,
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
        print(json.dumps(evaluation.to_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(_render_text(evaluation))
    return 0 if evaluation.passed else 2


def _render_text(evaluation) -> str:
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
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

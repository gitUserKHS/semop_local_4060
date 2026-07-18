from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (  # noqa: E402
    HUMAN_REVIEW_ATTESTATION,
    SemanticReviewDecision,
    create_semantic_review,
    load_semantic_benchmark,
    write_semantic_review,
)


DEFAULT_CASES = ROOT / "data" / "semantic_benchmark" / "v1" / "cases.jsonl"
DEFAULT_REVIEWS = ROOT / "data" / "semantic_benchmark" / "v1" / "reviews.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a digest-bound independent review of one semantic case."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--case-id")
    parser.add_argument("--reviewer")
    parser.add_argument(
        "--decision",
        choices=tuple(item.value for item in SemanticReviewDecision),
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--list-pending", action="store_true")
    parser.add_argument(
        "--attest-human-review",
        action="store_true",
        help=(
            "Attest that a person inspected both the raw payload and expected "
            "outcome. This is required before a review record is written."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        benchmark = load_semantic_benchmark(args.cases, args.reviews)
    except Exception as exc:
        print(f"review load error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.list_pending:
        audit = benchmark.audit()
        for case_id in audit.pending_cases:
            case = benchmark.case_by_id[case_id]
            print(
                f"{case.case_id}\t{case.domain.value}\t"
                f"expected={case.expected_solved}\t{case.phenomenon}\t{case.digest}"
            )
        return 0

    missing = tuple(
        name
        for name, value in (
            ("--case-id", args.case_id),
            ("--reviewer", args.reviewer),
            ("--decision", args.decision),
        )
        if not value
    )
    if missing:
        print("review error: required arguments: " + ", ".join(missing), file=sys.stderr)
        return 1
    if not args.attest_human_review:
        print(
            "review error: --attest-human-review is required; automated execution "
            "must not manufacture HUMAN_REVIEWED labels",
            file=sys.stderr,
        )
        return 1
    case = benchmark.case_by_id.get(args.case_id)
    if case is None:
        print(f"review error: unknown case id: {args.case_id}", file=sys.stderr)
        return 1

    try:
        review = create_semantic_review(
            case,
            reviewer=args.reviewer,
            decision=args.decision,
            notes=args.notes,
        )
        write_semantic_review(args.reviews, review)
    except Exception as exc:
        print(f"review write error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"case: {case.case_id}")
    print(f"digest: {case.digest}")
    print(f"decision: {review.decision.value}")
    print(f"reviewer: {review.reviewer}")
    print(f"attestation: {HUMAN_REVIEW_ATTESTATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

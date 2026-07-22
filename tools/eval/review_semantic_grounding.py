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
    GroundingLabel,
    SemanticGroundingCorpus,
    SemanticGroundingReviewDecision,
    create_semantic_grounding_review,
    load_semantic_benchmark,
    load_semantic_grounding_reviews,
    write_semantic_grounding_review,
)


DEFAULT_CASES = ROOT / "data" / "semantic_benchmark" / "v1" / "cases.jsonl"
DEFAULT_REVIEWS = ROOT / "data" / "semantic_grounding" / "v1" / "reviews.jsonl"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or record an exact candidate-level semantic review"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--case-id")
    parser.add_argument("--reviewer")
    parser.add_argument("--label", choices=tuple(label.value for label in GroundingLabel))
    parser.add_argument(
        "--decision",
        choices=tuple(decision.value for decision in SemanticGroundingReviewDecision),
        default=SemanticGroundingReviewDecision.APPROVED.value,
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    benchmark = load_semantic_benchmark(args.cases)
    corpus = SemanticGroundingCorpus.compile(
        benchmark,
        load_semantic_grounding_reviews(args.reviews),
    )
    if args.case_id is None:
        print(
            json.dumps(
                {
                    "audit": corpus.audit().to_dict(),
                    "targets": [target.to_dict() for target in corpus.targets],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    target = next(
        (
            item
            for item in corpus.targets
            if item.case_id == args.case_id or item.target_id == args.case_id
        ),
        None,
    )
    if target is None:
        parser.error(f"unknown semantic grounding case or target: {args.case_id}")
    if args.reviewer is None and args.label is None:
        print(json.dumps(target.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.reviewer is None or args.label is None:
        parser.error("recording a review requires both --reviewer and --label")

    review = create_semantic_grounding_review(
        target,
        reviewed_label=args.label,
        reviewer=args.reviewer,
        decision=args.decision,
        notes=args.notes,
    )
    write_semantic_grounding_review(args.reviews, review)
    print(json.dumps(review.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (  # noqa: E402
    HUMAN_REVIEW_ATTESTATION,
    ExperienceQueueItem,
    ExperienceQueueStatus,
    SemanticReviewDecision,
    TypedExperienceStore,
    canonical_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and independently review typed LMV experience."
    )
    parser.add_argument("--db", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    stats = commands.add_parser("stats", help="Show queue counts.")
    stats.add_argument("--format", choices=("text", "json"), default="text")

    listing = commands.add_parser("list", help="List prioritized queue items.")
    listing.add_argument(
        "--status",
        choices=tuple(item.value for item in ExperienceQueueStatus),
    )
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--format", choices=("text", "json"), default="text")

    show = commands.add_parser("show", help="Show one exact request and review.")
    show.add_argument("request_digest")

    review = commands.add_parser("review", help="Record one human review revision.")
    review.add_argument("request_digest")
    review.add_argument(
        "--expected",
        choices=("solved", "unsolved"),
        required=True,
    )
    review.add_argument("--phenomenon", required=True)
    review.add_argument("--rationale", required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument(
        "--decision",
        choices=tuple(item.value for item in SemanticReviewDecision),
        required=True,
    )
    review.add_argument("--notes", default="")
    review.add_argument("--difficulty", type=int, default=1)
    review.add_argument("--tag", action="append", default=[])
    review.add_argument(
        "--attest-human-review",
        action="store_true",
        help=(
            "Attest that a person inspected both the exact raw request and "
            "the expected outcome. Required before writing."
        ),
    )

    export = commands.add_parser("export", help="Export approved review records.")
    export.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        store = TypedExperienceStore.open_existing(args.db)
        if args.command == "stats":
            _print_stats(store, args.format)
        elif args.command == "list":
            _print_items(store, args.status, args.limit, args.format)
        elif args.command == "show":
            print(canonical_json(_item_dict(store.get_item(args.request_digest))))
        elif args.command == "review":
            if not args.attest_human_review:
                raise ValueError(
                    "--attest-human-review is required; automated execution "
                    "must not manufacture HUMAN_REVIEWED labels"
                )
            record = store.review(
                args.request_digest,
                expected_solved=args.expected == "solved",
                phenomenon=args.phenomenon,
                rationale=args.rationale,
                reviewer=args.reviewer,
                decision=args.decision,
                notes=args.notes,
                difficulty=args.difficulty,
                tags=tuple(args.tag),
            )
            print(canonical_json(record.to_dict()))
            print(f"attestation: {HUMAN_REVIEW_ATTESTATION}")
        elif args.command == "export":
            corpus = store.export_reviewed()
            artifact = canonical_json(
                {
                    "format_version": 1,
                    "partition_fingerprint": corpus.partition_fingerprint,
                    "records": [record.to_dict() for record in corpus.records],
                }
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(artifact + "\n", encoding="utf-8")
            print(f"exported: {len(corpus.records)}")
            print(f"output: {args.output.resolve()}")
    except Exception as exc:
        print(
            f"typed experience review error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


def _print_stats(store: TypedExperienceStore, output_format: str) -> None:
    stats = store.stats()
    if output_format == "json":
        print(canonical_json(asdict(stats)))
        return
    print("SemOp typed experience queue")
    print(f"requests: {stats.requests}")
    print(f"observations: {stats.observations}")
    print(f"pending: {stats.pending}")
    print(f"conflicted: {stats.conflicted}")
    print(f"approved: {stats.approved}")
    print(f"rejected: {stats.rejected}")
    for domain, count in stats.by_domain:
        print(f"{domain}: {count}")


def _print_items(
    store: TypedExperienceStore,
    status: str | None,
    limit: int,
    output_format: str,
) -> None:
    items = store.list_items(status=status, limit=limit)
    if output_format == "json":
        print(canonical_json([_item_dict(item) for item in items]))
        return
    for item in items:
        print(
            f"{item.request_digest}\t{item.domain}\t{item.status.value}\t"
            f"priority={item.priority}\toccurrences={item.occurrences}\t"
            f"triggers={','.join(item.triggers)}"
        )


def _item_dict(item: ExperienceQueueItem) -> dict[str, Any]:
    return {
        "request_digest": item.request_digest,
        "domain": item.domain,
        "payload": json.loads(item.payload_json),
        "occurrences": item.occurrences,
        "observed_failures": item.observed_failures,
        "unverified_results": item.unverified_results,
        "proposed_positive": item.proposed_positive,
        "proposed_negative": item.proposed_negative,
        "triggers": list(item.triggers),
        "status": item.status.value,
        "priority": item.priority,
        "latest_review": (
            item.latest_review.to_dict()
            if item.latest_review is not None
            else None
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import CorpusBuilder, PublicDatasetAdapter


def parse_csv_like(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize public QA/reasoning datasets into the local corpus format")
    parser.add_argument("--inputs", nargs="+", required=True, help="input dataset files: jsonl/json/csv/txt")
    parser.add_argument("--output", required=True, help="output corpus jsonl path")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--query-fields", help="comma-separated query field names")
    parser.add_argument("--context-fields", help="comma-separated context field names")
    parser.add_argument("--answer-fields", help="comma-separated answer field names")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    adapter = PublicDatasetAdapter(
        query_fields=parse_csv_like(args.query_fields),
        context_fields=parse_csv_like(args.context_fields),
        answer_fields=parse_csv_like(args.answer_fields),
    )
    examples = adapter.load_paths(args.inputs)
    builder = CorpusBuilder(train_ratio=args.train_ratio)
    records = builder.build_from_queries([(example.query, example.source) for example in examples], augment=not args.no_augment)
    builder.save_jsonl(records, args.output)
    print(f"normalized_examples={len(examples)}")
    print(f"built_records={len(records)}")


if __name__ == "__main__":
    main()

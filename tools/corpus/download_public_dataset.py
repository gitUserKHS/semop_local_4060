from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import CorpusBuilder, PublicDatasetAdapter, RemoteDatasetDownloader


DATASET_SUFFIXES = {".jsonl", ".json", ".csv", ".txt"}


def parse_csv_like(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def collect_dataset_files(paths: list[Path]) -> list[Path]:
    collected: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in DATASET_SUFFIXES:
            collected.append(path)
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in DATASET_SUFFIXES:
                    collected.append(child)
    return list(dict.fromkeys(collected))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public datasets and optionally normalize them into the local corpus format")
    parser.add_argument("--urls", nargs="+", required=True, help="dataset URLs to download")
    parser.add_argument("--output-dir", required=True, help="directory to save downloaded files")
    parser.add_argument("--extract", action="store_true", help="extract zip files after download")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing downloads")
    parser.add_argument("--normalize-output", help="optional output path for normalized corpus jsonl")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--query-fields", help="comma-separated query field names")
    parser.add_argument("--context-fields", help="comma-separated context field names")
    parser.add_argument("--answer-fields", help="comma-separated answer field names")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    downloader = RemoteDatasetDownloader()
    artifacts = downloader.download_many(args.urls, output_dir=args.output_dir, extract=args.extract, overwrite=args.overwrite)
    downloaded_files = [artifact.saved_path for artifact in artifacts]
    extracted_files = [path for artifact in artifacts for path in artifact.extracted_paths]

    print(f"downloaded_files={len(downloaded_files)}")
    print(f"extracted_files={len(extracted_files)}")

    if not args.normalize_output:
        return

    dataset_files = collect_dataset_files(extracted_files or downloaded_files)
    adapter = PublicDatasetAdapter(
        query_fields=parse_csv_like(args.query_fields),
        context_fields=parse_csv_like(args.context_fields),
        answer_fields=parse_csv_like(args.answer_fields),
    )
    examples = adapter.load_paths(dataset_files)
    builder = CorpusBuilder(train_ratio=args.train_ratio)
    records = builder.build_from_queries([(example.query, example.source) for example in examples], augment=not args.no_augment)
    builder.save_jsonl(records, args.normalize_output)
    print(f"normalized_examples={len(examples)}")
    print(f"built_records={len(records)}")


if __name__ == "__main__":
    main()


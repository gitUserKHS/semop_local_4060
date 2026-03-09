from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop.corpus_builder import CorpusBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a normalized and expanded reasoning corpus")
    parser.add_argument("--inputs", nargs="+", required=True, help="input jsonl/txt files")
    parser.add_argument("--output", required=True, help="output jsonl path")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    builder = CorpusBuilder(train_ratio=args.train_ratio)
    records = builder.build_from_paths(args.inputs, augment=not args.no_augment)
    builder.save_jsonl(records, args.output)
    print(f"built_records={len(records)}")


if __name__ == "__main__":
    main()

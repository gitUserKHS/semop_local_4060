from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import CpCorpusBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize mixed-format competitive-programming corpora into a single JSONL file")
    parser.add_argument("--inputs", nargs="+", required=True, help="input files or directories with raw CP statement data")
    parser.add_argument("--output", required=True, help="output normalized corpus JSONL path")
    parser.add_argument("--no-augment", action="store_true", help="disable paraphrase augmentation")
    args = parser.parse_args()

    builder = CpCorpusBuilder()
    records = builder.build_from_inputs(args.inputs, augment=not args.no_augment)
    builder.save_jsonl(args.output, records)
    summary = {
        "num_records": len(records),
        "source_kinds": sorted({record.source_kind for record in records}),
        "augmented_records": sum(1 for record in records if record.augmented),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


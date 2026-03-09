from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import CpCorpusBuilder, CpDslDatasetBuilder, CpTrainingPlanner


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CP DSL labeled dataset and optional SFT dataset from contest statements")
    parser.add_argument("--inputs", nargs="+", required=True, help="input files or directories with contest statements (.txt/.md/.jsonl/.json/.csv)")
    parser.add_argument("--output", required=True, help="output labeled CP DSL jsonl path")
    parser.add_argument("--sft-output", help="optional output path for prompt/completion SFT jsonl")
    parser.add_argument("--corpus-output", help="optional output path for the normalized statement corpus jsonl")
    parser.add_argument("--no-augment", action="store_true", help="disable simple statement paraphrase augmentation")
    args = parser.parse_args()

    corpus_builder = CpCorpusBuilder()
    corpus_records = corpus_builder.build_from_inputs(args.inputs, augment=not args.no_augment)
    statements = [record.statement for record in corpus_records]
    examples = CpDslDatasetBuilder().build_from_statements(statements)
    CpDslDatasetBuilder.save(args.output, examples)
    summary = CpTrainingPlanner.summarize_examples(examples)
    summary["num_corpus_records"] = len(corpus_records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.corpus_output:
        corpus_builder.save_jsonl(args.corpus_output, corpus_records)
        print(f"corpus_records={len(corpus_records)}")

    if args.sft_output:
        records = CpTrainingPlanner.build_sft_records(examples)
        CpTrainingPlanner.save_sft_records(args.sft_output, records)
        print(f"sft_records={len(records)}")


if __name__ == "__main__":
    main()


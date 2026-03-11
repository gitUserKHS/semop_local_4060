from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import CpLearnedParser, CpParserEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate heuristic or learned CP parsers on statement -> DSL/frame/sketch labels")
    parser.add_argument("--input", required=True, help="CP DSL labeled jsonl path")
    parser.add_argument("--mode", choices=["heuristic", "model", "compare"], default="heuristic")
    parser.add_argument("--model", help="local model path or Hugging Face model id for --mode model or --mode compare")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, help="optional maximum number of examples")
    args = parser.parse_args()

    examples = CpParserEvaluator.load_examples(args.input)
    if args.limit:
        examples = examples[: args.limit]
    evaluator = CpParserEvaluator()
    model = None
    if args.mode in {"model", "compare"}:
        if not args.model:
            parser.error("--model is required for --mode model or --mode compare")
        model = CpLearnedParser(args.model, local_files_only=args.local_files_only)
    if args.mode == 'compare':
        summary = evaluator.compare_examples(examples, model=model)
    else:
        summary = evaluator.evaluate_examples(examples, predictor='model' if args.mode == 'model' else 'heuristic', model=model)
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


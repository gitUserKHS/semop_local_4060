from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import MemoryPriorEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs memory-prior reasoning on a memory store")
    parser.add_argument("--store", required=True, help="sqlite memory store path")
    parser.add_argument("--source", help="optional source label filter")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-test-queries", type=int, default=100)
    parser.add_argument("--sample-queries", type=int, default=10)
    parser.add_argument("--output", help="optional output json path")
    args = parser.parse_args()

    evaluator = MemoryPriorEvaluator(mode=args.mode, model_id=args.model_id)
    result = evaluator.evaluate_store(
        store_path=args.store,
        source=args.source,
        max_test_queries=args.max_test_queries,
        sample_queries=args.sample_queries,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.to_json(indent=2), encoding="utf-8")
    print(result.to_json(indent=2))


if __name__ == "__main__":
    main()


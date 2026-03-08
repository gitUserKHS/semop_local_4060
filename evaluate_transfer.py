from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from semop.transfer_eval import TransferEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate operator family transfer on a held-out corpus split")
    parser.add_argument("--input", required=True, help="jsonl corpus path")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    args = parser.parse_args()

    evaluator = TransferEvaluator(mode=args.mode, model_id=args.model_id)
    result = evaluator.evaluate_jsonl(args.input, train_ratio=args.train_ratio)
    print(result.to_json(indent=2))


if __name__ == "__main__":
    main()
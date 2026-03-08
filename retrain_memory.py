from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from semop import CorpusMemoryStore, CorpusReasoningLearner, TransferEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain operator families from the memory store and evaluate held-out transfer")
    parser.add_argument("--store", required=True, help="sqlite memory store path")
    parser.add_argument("--source", default="ingest", help="source label")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", help="optional learning result json path")
    args = parser.parse_args()

    store = CorpusMemoryStore(args.store)
    train_queries = store.fetch_queries(split="train", source=args.source)
    test_queries = store.fetch_queries(split="test", source=args.source)

    learner = CorpusReasoningLearner(mode=args.mode, model_id=args.model_id)
    train_result = learner.learn_from_queries(train_queries)
    if args.output:
        learner.save_result(train_result, args.output)
    store.store_learning_result(train_result, source=f"{args.source}:retrain")

    print(train_result.to_json(indent=2))

    if test_queries:
        evaluator = TransferEvaluator(mode=args.mode, model_id=args.model_id)
        ratio = len(train_queries) / max(1, len(train_queries) + len(test_queries))
        transfer = evaluator.evaluate_queries(train_queries + test_queries, train_ratio=ratio)
        print(transfer.to_json(indent=2))


if __name__ == "__main__":
    main()

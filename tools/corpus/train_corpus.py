from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop.corpus_learning import CorpusReasoningLearner
from semop.corpus_store import CorpusMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn reusable reasoning operators from a corpus")
    parser.add_argument("--input", required=True, help="jsonl corpus path")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", help="optional output json path")
    parser.add_argument("--store", help="optional sqlite memory store path")
    parser.add_argument("--source", default="corpus", help="source label for stored examples")
    args = parser.parse_args()

    learner = CorpusReasoningLearner(mode=args.mode, model_id=args.model_id)
    result = learner.learn_from_jsonl(args.input)

    if args.output:
        learner.save_result(result, args.output)

    if args.store:
        store = CorpusMemoryStore(args.store)
        graphs = store.fetch_graphs(source=args.source)
        existing = {graph.query for graph in graphs}
        fresh_result = learner.learn_from_jsonl(args.input)
        for example in fresh_result.examples:
            if example.query in existing:
                continue
            graph = learner.pipeline.run(example.query)
            split = "train"
            store.upsert_graph(graph, source=args.source, split=split)
        store.store_learning_result(result, source=args.source)

    print(result.to_json(indent=2))


if __name__ == "__main__":
    main()

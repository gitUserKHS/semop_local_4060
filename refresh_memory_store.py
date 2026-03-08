from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from semop import CorpusMemoryStore, StructuredMeaningPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run the current pipeline over queries stored in SQLite and refresh stored graphs")
    parser.add_argument("--store", required=True, help="sqlite memory store path")
    parser.add_argument("--source", help="optional source filter")
    parser.add_argument("--split", default="train", help="split to refresh")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-queries", type=int, default=0, help="optional limit for refresh runs")
    args = parser.parse_args()

    store = CorpusMemoryStore(args.store)
    queries = store.fetch_queries(split=args.split, source=args.source)
    if args.max_queries > 0:
        queries = queries[:args.max_queries]
    if not queries:
        raise SystemExit(f"No queries found for split={args.split!r} source={args.source!r}")

    pipeline = StructuredMeaningPipeline(mode=args.mode, model_id=args.model_id)
    for index, query in enumerate(queries, start=1):
        graph = pipeline.run(query)
        store.upsert_graph(graph, source=args.source or "manual", split=args.split)
        if index % 250 == 0 or index == len(queries):
            print(f"refreshed={index}/{len(queries)}")


if __name__ == "__main__":
    main()

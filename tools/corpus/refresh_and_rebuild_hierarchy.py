from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import CorpusMemoryStore, OperatorHierarchyLearner, StructuredMeaningPipeline


def _iter_sources(store: CorpusMemoryStore, split: str) -> list[str]:
    import sqlite3

    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute(
            "select distinct source from examples where split = ? order by source",
            (split,),
        ).fetchall()
    return [row[0] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh stored graphs for every source in a SQLite corpus and optionally rebuild hierarchy")
    parser.add_argument("--store", required=True, help="sqlite memory store path")
    parser.add_argument("--split", default="train", help="split to refresh")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-queries-per-source", type=int, default=0)
    parser.add_argument("--rebuild-hierarchy", action="store_true")
    parser.add_argument("--global-hierarchy", action="store_true")
    args = parser.parse_args()

    store = CorpusMemoryStore(args.store)
    sources = _iter_sources(store, args.split)
    if not sources:
        raise SystemExit(f"No sources found for split={args.split!r}")

    pipeline = StructuredMeaningPipeline(mode=args.mode, model_id=args.model_id)
    learner = OperatorHierarchyLearner(mode=args.mode, model_id=args.model_id) if args.rebuild_hierarchy else None
    totals = defaultdict(int)

    for source in sources:
        queries = store.fetch_queries(split=args.split, source=source)
        if args.max_queries_per_source > 0:
            queries = queries[: args.max_queries_per_source]
        if not queries:
            continue
        print(f"refresh_source={source} total={len(queries)}")
        for index, query in enumerate(queries, start=1):
            graph = pipeline.run(query)
            store.upsert_graph(graph, source=source, split=args.split)
            totals[source] += 1
            if index % 500 == 0 or index == len(queries):
                print(f"refreshed_source={source} progress={index}/{len(queries)}")
        if learner is not None:
            graphs = store.fetch_graphs(split=args.split, source=source)
            result = learner.learn_from_graphs(graphs)
            store.store_hierarchy_result(result, source=source, split=args.split)
            print(f"rebuilt_hierarchy_source={source} corpus={result.corpus_size} families={len(result.family_nodes)} abstracts={len(result.abstract_nodes)}")

    if args.global_hierarchy and learner is not None:
        graphs = store.fetch_graphs(split=args.split)
        result = learner.learn_from_graphs(graphs)
        store.store_hierarchy_result(result, source="manual", split=args.split)
        print(f"rebuilt_hierarchy_source=manual corpus={result.corpus_size} families={len(result.family_nodes)} abstracts={len(result.abstract_nodes)}")

    print("refresh_complete")
    for source in sources:
        if totals[source]:
            print(f"source={source} refreshed={totals[source]}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from semop import CorpusBuilder, CorpusMemoryStore, StructuredMeaningPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw question files into the corpus memory store")
    parser.add_argument("--inputs", nargs="+", required=True, help="input jsonl/txt files")
    parser.add_argument("--store", required=True, help="sqlite memory store path")
    parser.add_argument("--source", default="ingest", help="source label")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    builder = CorpusBuilder(train_ratio=args.train_ratio)
    records = builder.build_from_paths(args.inputs, augment=not args.no_augment)
    pipeline = StructuredMeaningPipeline(mode=args.mode, model_id=args.model_id)
    store = CorpusMemoryStore(args.store)

    inserted = 0
    for record in records:
        graph = pipeline.run(record.query)
        store.upsert_graph(graph, source=args.source, split=record.split)
        inserted += 1

    print(f"ingested_records={inserted}")
    print(f"stored_examples={store.count_examples(source=args.source)}")


if __name__ == "__main__":
    main()

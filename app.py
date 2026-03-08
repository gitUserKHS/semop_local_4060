from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from semop import ResponseSynthesizer, StructuredMeaningPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic operator + structured meaning planner")
    parser.add_argument("--query", required=True, help="user query")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--memory-store", help="optional sqlite memory store path")
    parser.add_argument("--memory-source", help="optional memory source filter")
    args = parser.parse_args()

    pipeline = StructuredMeaningPipeline(mode=args.mode, model_id=args.model_id, memory_store_path=args.memory_store, memory_source=args.memory_source)
    graph = pipeline.run(args.query)

    if args.format == "json":
        print(graph.model_dump_json(indent=2, ensure_ascii=False))
        return

    response = ResponseSynthesizer().synthesize(graph)
    print(response.to_text())


if __name__ == "__main__":
    main()
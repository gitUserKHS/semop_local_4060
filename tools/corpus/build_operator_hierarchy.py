from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import CorpusMemoryStore, OperatorHierarchyLearner


def _print_json(payload: str) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    print(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn corpus-scale operator hierarchy from stored graphs")
    parser.add_argument("--store", required=True, help="sqlite memory store path")
    parser.add_argument("--source", help="optional source filter")
    parser.add_argument("--split", default="train", help="split to learn from")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", help="optional output json path")
    parser.add_argument("--persist", action="store_true", help="store hierarchy summary into sqlite")
    args = parser.parse_args()

    store = CorpusMemoryStore(args.store)
    graphs = store.fetch_graphs(split=args.split, source=args.source)
    if not graphs:
        raise SystemExit(f"No graphs found for split={args.split!r} source={args.source!r}")
    learner = OperatorHierarchyLearner(mode=args.mode, model_id=args.model_id)
    result = learner.learn_from_graphs(graphs)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        learner.save_result(result, output_path)

    if args.persist:
        store.store_hierarchy_result(result, source=args.source or "manual", split=args.split)

    _print_json(result.to_json(indent=2))


if __name__ == "__main__":
    main()


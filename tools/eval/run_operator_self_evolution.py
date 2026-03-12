from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import (
    CorpusMemoryStore,
    OperatorSelfEvolutionLoop,
    OperatorTransferEvalCase,
    StructuredMeaningPipeline,
)


def load_cases(path: str | Path) -> list[OperatorTransferEvalCase]:
    rows: list[OperatorTransferEvalCase] = []
    for raw in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        rows.append(OperatorTransferEvalCase(
            query=str(payload.get('query', '')),
            domain=str(payload.get('domain', 'general')),
            expected_operator_names=[str(item) for item in payload.get('expected_operator_names', [])],
            split=str(payload.get('split', 'train')),
        ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Run iterative operator self-evolution and persist retained operators.')
    parser.add_argument('--input', required=True, help='JSONL with train/test operator-transfer cases.')
    parser.add_argument('--memory-store', default='data/semop_memory.db')
    parser.add_argument('--source', default='operator_self_evolution')
    parser.add_argument('--split', default='train')
    parser.add_argument('--mode', default='heuristic')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--min-support', type=int, default=2)
    parser.add_argument('--utility-threshold', type=float, default=0.45)
    parser.add_argument('--output', default='data/operator_self_evolution_summary.json')
    args = parser.parse_args()

    cases = load_cases(args.input)
    train_queries = [case.query for case in cases if case.split == 'train']
    if not train_queries:
        raise SystemExit('No train cases found in input JSONL.')

    store = CorpusMemoryStore(args.memory_store)
    loop = OperatorSelfEvolutionLoop(
        StructuredMeaningPipeline(mode=args.mode, memory_store_path=args.memory_store, memory_source=args.source),
        store,
    )
    run_results = loop.run(
        queries=train_queries,
        source=args.source,
        split=args.split,
        iterations=args.iterations,
        min_support=args.min_support,
        utility_threshold=args.utility_threshold,
        transfer_cases=cases,
    )
    payload = {
        'input': str(args.input),
        'memory_store': str(args.memory_store),
        'source': args.source,
        'split': args.split,
        'iterations': args.iterations,
        'results': [item.model_dump() for item in run_results],
        'latest_evolution_summary': store.fetch_latest_operator_evolution_summary(source=args.source, split=args.split),
        'latest_transfer_summary': store.fetch_latest_operator_transfer_summary(source=args.source, split=args.split),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

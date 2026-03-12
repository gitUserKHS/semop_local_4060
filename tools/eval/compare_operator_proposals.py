from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import OperatorProposalComparator, StructuredMeaningPipeline


def load_queries(path: str | Path) -> list[str]:
    rows: list[str] = []
    for raw in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        rows.append(str(payload.get('query', '')))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare heuristic and LLM-backed operator proposal engines.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--mode', default='heuristic')
    parser.add_argument('--llm-model-id', required=True)
    args = parser.parse_args()

    pipeline = StructuredMeaningPipeline(mode=args.mode)
    graphs = [pipeline.run(query) for query in load_queries(args.input)]
    payload = OperatorProposalComparator().compare_with_hybrid(graphs, llm_model_id=args.llm_model_id)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

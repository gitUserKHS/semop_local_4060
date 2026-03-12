from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import OperatorTransferEvalCase, StructuredMeaningPipeline, VisualSignalImpactEvaluator


def load_queries(path: str | Path) -> list[str]:
    rows: list[str] = []
    for raw in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        if str(payload.get('split', 'train')) == 'train':
            rows.append(str(payload.get('query', '')))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate impact of symmetry/closure/axis-alignment signals on retained operators.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--mode', default='heuristic')
    args = parser.parse_args()

    pipeline = StructuredMeaningPipeline(mode=args.mode)
    graphs = [pipeline.run(query) for query in load_queries(args.input)]
    summary = VisualSignalImpactEvaluator().evaluate(graphs)
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

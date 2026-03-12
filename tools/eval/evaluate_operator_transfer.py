
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import OperatorTransferEvalCase, OperatorTransferEvaluator, StructuredMeaningPipeline


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
    parser = argparse.ArgumentParser(description='Evaluate operator transfer across train/test domains.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--mode', default='heuristic')
    args = parser.parse_args()

    evaluator = OperatorTransferEvaluator(StructuredMeaningPipeline(mode=args.mode))
    summary = evaluator.evaluate(load_cases(args.input))
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

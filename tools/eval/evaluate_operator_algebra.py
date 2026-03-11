from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import OperatorAlgebraEvaluator, StructuredMeaningPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate operator decomposition and functor recovery")
    parser.add_argument('--input', required=True, help='operator algebra eval jsonl path')
    parser.add_argument('--mode', choices=['heuristic', 'llm'], default='heuristic')
    args = parser.parse_args()

    evaluator = OperatorAlgebraEvaluator(StructuredMeaningPipeline(mode=args.mode))
    cases = evaluator.load_cases(args.input)
    summary = evaluator.evaluate(cases)
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

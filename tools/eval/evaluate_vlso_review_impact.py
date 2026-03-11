from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import VlsoReviewImpactEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare VLSO grounded QA before and after approved-cluster retraining')
    parser.add_argument('--input', required=True, help='VLSO eval jsonl path')
    parser.add_argument('--primary-concept-store', required=True)
    parser.add_argument('--primary-operator-store', help='optional primary operator DB path')
    parser.add_argument('--compare-concept-store', required=True)
    parser.add_argument('--compare-operator-store', help='optional comparison operator DB path')
    parser.add_argument('--mode', choices=['heuristic', 'hybrid', 'deep'], default='heuristic')
    parser.add_argument('--answer-mode', choices=['structured', 'llm'], default='structured')
    parser.add_argument('--weights', help='optional affordance weights JSON path')
    args = parser.parse_args()

    summary = VlsoReviewImpactEvaluator(
        mode=args.mode,
        answer_mode=args.answer_mode,
        affordance_weights_path=args.weights,
    ).compare_stores(
        input_path=args.input,
        primary_concept_store=args.primary_concept_store,
        primary_operator_store=args.primary_operator_store,
        compare_concept_store=args.compare_concept_store,
        compare_operator_store=args.compare_operator_store,
    )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

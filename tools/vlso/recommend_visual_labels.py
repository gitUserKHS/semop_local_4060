from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import VisualConceptLabelRecommender


def load_rows(path: str) -> list[dict]:
    rows = []
    with Path(path).open('r', encoding='utf-8-sig') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Rank the next most informative visual targets to label')
    parser.add_argument('--candidates', required=True, help='candidate JSONL produced by build_visual_concept_candidates.py')
    parser.add_argument('--concept-store', help='optional concept-memory SQLite path')
    parser.add_argument('--weights', help='optional classifier weights JSON path')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--output', help='optional JSON output file')
    args = parser.parse_args()

    ranked = VisualConceptLabelRecommender(
        concept_store_path=args.concept_store,
        weights_path=args.weights,
    ).rank_candidate_rows(load_rows(args.candidates), limit=args.limit)
    payload = json.dumps({'recommended': ranked}, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()

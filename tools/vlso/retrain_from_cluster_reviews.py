from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import VisualApprovedReviewRetrainer


def main() -> None:
    parser = argparse.ArgumentParser(description='Retrain VLSO concept/operator stores from approved cluster reviews')
    parser.add_argument('--summary', required=True, help='cluster summary JSON path')
    parser.add_argument('--reviews', required=True, help='cluster review decisions JSON path')
    parser.add_argument('--workspace', default='', help='output workspace directory; defaults to summary parent')
    parser.add_argument('--semop-memory-store', default='', help='optional SemOp SQLite memory DB path to seed premise/operator memories')
    parser.add_argument('--semop-memory-source', default='vlso_review', help='source tag for seeded SemOp memories')
    args = parser.parse_args()

    workspace = Path(args.workspace) if args.workspace else Path(args.summary).resolve().parent
    workspace.mkdir(parents=True, exist_ok=True)
    summary = VisualApprovedReviewRetrainer().export_and_retrain(
        summary_path=args.summary,
        review_path=args.reviews,
        labels_path=workspace / 'approved_review_labels.jsonl',
        concept_store_path=workspace / 'approved_review_concepts.db',
        operator_store_path=workspace / 'approved_review_operators.db',
        operator_summary_output=workspace / 'approved_review_operator_summary.json',
        semop_memory_store_path=Path(args.semop_memory_store) if args.semop_memory_store else None,
        semop_memory_source=args.semop_memory_source,
    )
    out_path = workspace / 'approved_review_retrain_summary.json'
    out_path.write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

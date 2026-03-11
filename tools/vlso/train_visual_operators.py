from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import VisualOperatorPrototypeTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description='Train few-shot visual operator prototypes from labeled JSONL')
    parser.add_argument('--labels', required=True, help='concept labels JSONL path')
    parser.add_argument('--store', required=True, help='operator prototype SQLite path')
    parser.add_argument('--summary-output', help='optional JSON summary output path')
    args = parser.parse_args()

    summary = VisualOperatorPrototypeTrainer().train_jsonl(args.labels, args.store, summary_output=args.summary_output)
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

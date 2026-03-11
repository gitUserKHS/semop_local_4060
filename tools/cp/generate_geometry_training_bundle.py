from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import CpGeometryTemplateGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate starter geometry-only CP train/val datasets from built-in templates')
    parser.add_argument('--train-output', default='examples/cp_geometry_train.jsonl', help='train jsonl path')
    parser.add_argument('--val-output', default='examples/cp_geometry_val.jsonl', help='validation jsonl path')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='train split ratio')
    args = parser.parse_args()

    summary = CpGeometryTemplateGenerator().build_train_val_split(args.train_output, args.val_output, train_ratio=args.train_ratio)
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import OperatorCurriculumBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a generic operator-learning curriculum bundle from teacher traces.')
    parser.add_argument('--teacher-traces', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--max-per-task', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = OperatorCurriculumBuilder().build_bundle(
        teacher_trace_path=args.teacher_traces,
        workspace=args.workspace,
        val_ratio=args.val_ratio,
        max_per_task=args.max_per_task,
    )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop.cp_geometry_eval import CpGeometryEvalBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a geometry-focused CP parser eval set from normalized labeled corpus JSONL')
    parser.add_argument('--input', required=True, help='normalized labeled corpus jsonl')
    parser.add_argument('--output', required=True, help='output CP parser eval jsonl')
    parser.add_argument('--limit', type=int, help='optional maximum number of examples')
    args = parser.parse_args()

    summary = CpGeometryEvalBuilder().build_from_normalized_jsonl(args.input, args.output, limit=args.limit)
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

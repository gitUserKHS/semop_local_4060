from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import CpGeometryTemplateGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate a starter geometry-only CP parser eval set from built-in templates')
    parser.add_argument('--output', default='examples/cp_geometry_parser_eval.jsonl', help='output eval jsonl path')
    args = parser.parse_args()

    summary = CpGeometryTemplateGenerator().build_eval_set(args.output)
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop.vlso.real_image_eval import RealImageEvalBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description='Finalize reviewed real-image VLSO candidates into a gold eval jsonl')
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--auto-approve-limit', type=int, default=0)
    parser.add_argument('--reject-terms', nargs='*', default=[])
    args = parser.parse_args()

    builder = RealImageEvalBuilder()
    if args.auto_approve_limit > 0:
        summary = builder.finalize_auto_selected(
            candidates_path=args.candidates,
            output_path=args.output,
            auto_approve_limit=args.auto_approve_limit,
            reject_terms=args.reject_terms,
        )
    else:
        summary = builder.finalize_reviewed(
            candidates_path=args.candidates,
            output_path=args.output,
        )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

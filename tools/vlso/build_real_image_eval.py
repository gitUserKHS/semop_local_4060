from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop.vlso.real_image_eval import RealImageEvalBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description='Build candidate and starter real-image VLSO eval sets from downloaded public images')
    parser.add_argument('--records', required=True, help='family_records.jsonl path')
    parser.add_argument('--manifest', required=True, help='family_download_manifest.jsonl path')
    parser.add_argument('--candidate-output', required=True, help='candidate eval jsonl output path')
    parser.add_argument('--seed-output', required=True, help='starter real-image eval jsonl output path')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    summary = RealImageEvalBuilder().build(
        records_path=args.records,
        manifest_path=args.manifest,
        candidate_output=args.candidate_output,
        seed_output=args.seed_output,
        limit=args.limit,
    )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

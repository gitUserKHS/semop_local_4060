from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import CpLabeledDatasetDownloader


def main() -> None:
    parser = argparse.ArgumentParser(description='Bootstrap a geometry-focused labeled CP corpus from a manifest')
    parser.add_argument('--manifest', default='examples/cp_geometry_labeled_manifest.json', help='geometry manifest path')
    parser.add_argument('--download-root', default='data/cp_geometry_downloads', help='download/cache directory')
    parser.add_argument('--output', default='data/cp_geometry_labeled.jsonl', help='normalized geometry JSONL output')
    parser.add_argument('--overwrite', action='store_true', help='overwrite downloaded artifacts')
    args = parser.parse_args()

    summary = CpLabeledDatasetDownloader().download_and_normalize(
        manifest_path=args.manifest,
        download_root=args.download_root,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

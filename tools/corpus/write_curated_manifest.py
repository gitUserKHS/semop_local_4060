from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import CURATED_PRESETS, CURATED_PUBLIC_DATASETS, curated_manifest, preset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a curated public reasoning dataset manifest")
    parser.add_argument("--output", required=True, help="manifest output path")
    parser.add_argument("--preset", choices=sorted(CURATED_PRESETS.keys()), help="named curated preset")
    parser.add_argument("--datasets", nargs="+", help="specific curated dataset aliases")
    args = parser.parse_args()

    if not args.preset and not args.datasets:
        parser.error("one of --preset or --datasets is required")

    manifest = preset_manifest(args.preset) if args.preset else curated_manifest(args.datasets)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"written_datasets={len(manifest)}")
    print(f"available_presets={','.join(sorted(CURATED_PRESETS))}")
    print(f"available_datasets={','.join(sorted(CURATED_PUBLIC_DATASETS))}")


if __name__ == "__main__":
    main()


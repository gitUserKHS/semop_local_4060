from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import VisualDataCollector


def _load_ids(path: str | None) -> list[str] | None:
    if not path:
        return None
    rows = []
    with Path(path).open('r', encoding='utf-8-sig') as handle:
        for line in handle:
            value = line.strip()
            if value:
                rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Approve collected visual records and prepare a download manifest')
    parser.add_argument('--records', required=True, help='normalized records JSONL from collect_visual_data.py --execute or curated input')
    parser.add_argument('--approved-output', required=True, help='approved records JSONL path')
    parser.add_argument('--manifest-output', required=True, help='download manifest JSONL path')
    parser.add_argument('--download-root', required=True, help='local download root')
    parser.add_argument('--allow-providers', nargs='*', help='provider whitelist such as wikimedia_commons openverse')
    parser.add_argument('--allow-licenses', nargs='*', help='license whitelist substrings such as cc0 by by-sa')
    parser.add_argument('--approved-ids', help='optional text file of source ids or media URLs to accept explicitly')
    parser.add_argument('--accept-all', action='store_true', help='approve every record with a media URL')
    parser.add_argument('--execute', action='store_true', help='perform actual downloads from the built manifest')
    args = parser.parse_args()

    summary = VisualDataCollector().prepare_downloads(
        records_path=args.records,
        approved_output=args.approved_output,
        manifest_output=args.manifest_output,
        download_root=args.download_root,
        allow_providers=args.allow_providers,
        allow_licenses=args.allow_licenses,
        approved_ids=_load_ids(args.approved_ids),
        accept_all=args.accept_all,
        execute=args.execute,
    )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

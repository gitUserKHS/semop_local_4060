from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument('--request-delay-seconds', type=float, default=None, help='override delay between download requests')
    parser.add_argument('--max-retries', type=int, default=None, help='override retry count for transient download failures')
    parser.add_argument('--progress', action='store_true', help='print per-download progress to stderr')
    args = parser.parse_args()

    if args.request_delay_seconds is not None:
        os.environ['SEMOP_VLSO_REQUEST_DELAY_SECONDS'] = str(args.request_delay_seconds)
    if args.max_retries is not None:
        os.environ['SEMOP_VLSO_MAX_RETRIES'] = str(args.max_retries)

    def _progress(event: str, payload: dict[str, object]) -> None:
        if not args.progress:
            return
        idx = payload.get('index', '?')
        total = payload.get('total', '?')
        provider = payload.get('provider', '')
        title = payload.get('title', '')
        if event == 'download_start':
            print(f'[download {idx}/{total}] {provider} :: {title}', file=sys.stderr)
        elif event == 'download_succeeded':
            print(f'[ok       {idx}/{total}] {provider} :: {title}', file=sys.stderr)
        elif event == 'download_failed':
            print(f'[fail     {idx}/{total}] {provider} :: {title} -> {payload.get("error", "error")}', file=sys.stderr)

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
        progress_callback=_progress if args.progress else None,
    )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

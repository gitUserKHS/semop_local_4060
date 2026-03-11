from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import VisualDataCollector


def main() -> None:
    parser = argparse.ArgumentParser(description='Plan or execute manifest-driven visual data collection for VLSO research')
    parser.add_argument('--manifest', required=True, help='JSON manifest path')
    parser.add_argument('--output', required=True, help='JSONL output path')
    parser.add_argument('--execute', action='store_true', help='perform live requests instead of dry-run planning')
    parser.add_argument('--summary-output', help='optional JSON summary path')
    parser.add_argument('--request-delay-seconds', type=float, default=None, help='override delay between API requests and downloads')
    parser.add_argument('--max-retries', type=int, default=None, help='override retry count for transient HTTP failures')
    parser.add_argument('--progress', action='store_true', help='print per-request progress to stderr')
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
        query = payload.get('query', '')
        if event == 'plan':
            print(f'[plan {idx}/{total}] {provider} :: {query}', file=sys.stderr)
        elif event == 'request_succeeded':
            print(f'[ok   {idx}/{total}] {provider} :: {query} -> {payload.get("records", 0)} records', file=sys.stderr)
        elif event == 'request_failed':
            print(f'[fail {idx}/{total}] {provider} :: {query} -> {payload.get("error", "error")}', file=sys.stderr)

    collector = VisualDataCollector()
    summary = collector.run_manifest(args.manifest, args.output, dry_run=not args.execute, progress_callback=_progress if args.progress else None)
    if args.summary_output:
        Path(args.summary_output).write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

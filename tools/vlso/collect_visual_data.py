from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args()

    collector = VisualDataCollector()
    summary = collector.run_manifest(args.manifest, args.output, dry_run=not args.execute)
    if args.summary_output:
        Path(args.summary_output).write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

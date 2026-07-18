from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop.data import (  # noqa: E402
    DatasetAcquisitionError,
    fetch_dataset_source,
    load_dataset_source_manifest,
    write_dataset_receipts,
)


DEFAULT_MANIFEST = ROOT / "data" / "sources" / "semantic_sources.v1.json"


def _source_summary(source) -> dict[str, object]:
    return {
        "source_id": source.source_id,
        "domain": source.domain.value,
        "title": source.title,
        "revision": source.revision,
        "license_spdx": source.license_spdx,
        "max_bytes": source.max_bytes,
        "integrity_locked": source.integrity_locked,
        "label_authority": source.label_authority.value,
        "use_policy": source.use_policy.value,
        "direct_training_allowed": source.direct_training_allowed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List or fetch pinned language, math, and vision sources"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source", action="append", default=[], help="source id; repeatable")
    parser.add_argument("--list", action="store_true", help="list catalog entries")
    parser.add_argument("--fetch", action="store_true", help="perform network downloads")
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="confirm that the listed source licenses were reviewed",
    )
    parser.add_argument(
        "--bootstrap-lock",
        action="store_true",
        help="allow an immutable source without a committed expected SHA-256",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "datasets")
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--max-total-mb", type=float, default=128.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = load_dataset_source_manifest(args.manifest)
    try:
        selected = (
            tuple(manifest.get(source_id) for source_id in args.source)
            if args.source
            else manifest.sources
        )
    except KeyError as error:
        parser.error(str(error))

    plan = {
        "schema_version": manifest.schema_version,
        "manifest_digest": manifest.digest,
        "sources": [_source_summary(source) for source in selected],
    }
    if args.list or not args.fetch:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if not args.source:
        parser.error("--fetch requires at least one explicit --source")
    if not args.accept_license:
        parser.error("--fetch requires --accept-license")
    if args.max_total_mb <= 0:
        parser.error("--max-total-mb must be positive")
    declared_total = sum(source.max_bytes for source in selected)
    if declared_total > int(args.max_total_mb * 1024 * 1024):
        parser.error("selected sources exceed --max-total-mb by declared limits")

    fetched = []
    try:
        for source in selected:
            fetched.append(
                fetch_dataset_source(
                    source,
                    args.output_dir,
                    manifest_digest=manifest.digest,
                    overwrite=args.overwrite,
                    allow_unpinned=args.bootstrap_lock,
                    timeout=args.timeout,
                )
            )
    except DatasetAcquisitionError as error:
        print(f"dataset acquisition failed: {error}", file=sys.stderr)
        return 1

    receipt_path = args.receipts or args.output_dir / "receipts.jsonl"
    write_dataset_receipts(receipt_path, [item.receipt for item in fetched])
    output = {
        "manifest_digest": manifest.digest,
        "receipt_path": str(receipt_path),
        "artifacts": [
            {
                "source_id": item.receipt.source_id,
                "path": str(item.path),
                "sha256": item.receipt.sha256,
                "byte_count": item.receipt.byte_count,
                "integrity_locked": item.receipt.integrity_locked,
                "downloaded": item.downloaded,
            }
            for item in fetched
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

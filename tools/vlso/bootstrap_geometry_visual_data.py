from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import VisualDataCollector
from semop.vlso.data_collection import build_geometry_seed_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap VLSO geometry/access visual data collection with a preset manifest")
    parser.add_argument("--workspace", default="data/vlso_geometry_bootstrap", help="output workspace directory")
    parser.add_argument("--manifest", help="override manifest path")
    parser.add_argument("--execute-collect", action="store_true", help="perform API fetches instead of dry-run planning")
    parser.add_argument("--execute-downloads", action="store_true", help="download approved media files after manifest creation")
    parser.add_argument("--accept-all", action="store_true", help="approve all collected records")
    parser.add_argument("--allow-providers", nargs="*", default=["wikimedia_commons", "openverse"], help="providers allowed during approval")
    parser.add_argument("--allow-licenses", nargs="*", default=["cc0", "by", "by-sa"], help="license tokens allowed during approval")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else workspace / "geometry_collection_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(build_geometry_seed_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")

    collector = VisualDataCollector()
    plan_path = workspace / ("geometry_records.jsonl" if args.execute_collect else "geometry_plan.jsonl")
    summary_path = workspace / ("geometry_records_summary.json" if args.execute_collect else "geometry_plan_summary.json")
    summary = collector.run_manifest(manifest_path, plan_path, dry_run=not args.execute_collect)
    summary_path.write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {"collection": summary.model_dump()}
    if args.execute_collect:
        approved_path = workspace / "geometry_approved.jsonl"
        download_manifest_path = workspace / "geometry_download_manifest.jsonl"
        download_root = workspace / "downloads"
        download_summary = collector.prepare_downloads(
            records_path=plan_path,
            approved_output=approved_path,
            manifest_output=download_manifest_path,
            download_root=download_root,
            allow_providers=args.allow_providers,
            allow_licenses=args.allow_licenses,
            accept_all=args.accept_all,
            execute=args.execute_downloads,
        )
        payload["downloads"] = download_summary.model_dump()

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import PublicCorpusIngestor


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, normalize, and ingest multiple public datasets from a manifest")
    parser.add_argument("--manifest", required=True, help="manifest json path")
    parser.add_argument("--download-root", default="data/public_downloads", help="download directory root")
    parser.add_argument("--normalized-root", default="data/normalized_corpora", help="normalized corpus output directory")
    parser.add_argument("--store", required=True, help="sqlite memory store path")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ingestor = PublicCorpusIngestor(mode=args.mode, model_id=args.model_id)
    summaries = ingestor.run_manifest(
        manifest_path=args.manifest,
        download_root=args.download_root,
        normalized_root=args.normalized_root,
        store_path=args.store,
        overwrite=args.overwrite,
    )
    print(json.dumps([summary.to_dict() for summary in summaries], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


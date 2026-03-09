from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import EmbeddingModelCache


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache a sentence-transformers embedding model locally for offline retrieval")
    parser.add_argument("--model-id", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", help="optional output directory for the cached model")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache = EmbeddingModelCache()
    model_path = cache.cache(model_id=args.model_id, output_dir=args.output_dir, force=args.force)
    print(f"cached_model={Path(model_path).resolve()}")
    print(cache.export_env(model_path))


if __name__ == "__main__":
    main()


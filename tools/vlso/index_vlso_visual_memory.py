from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import VisualEmbeddingRecord, VisualEmbeddingStore, VisionEmbeddingExtractor


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def image_payload(path: Path) -> dict:
    return {
        "label": path.stem,
        "image_path": str(path),
        "metadata": {
            "image_path": str(path),
            "source_path": str(path),
            "source_type": "image",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Index structured visual observations or local image paths into the VLSO embedding store")
    parser.add_argument("--inputs", nargs="*", default=[], help="JSON observation files")
    parser.add_argument("--image-inputs", nargs="*", default=[], help="local image files for a vision backbone pass")
    parser.add_argument("--store", required=True)
    parser.add_argument("--model-id", default="token_geometry_v1", choices=["token_geometry_v1", "dinov2_adapter", "openclip_adapter"])
    parser.add_argument("--model-path", help="local model checkpoint path for dinov2_adapter or openclip_adapter")
    args = parser.parse_args()

    if not args.inputs and not args.image_inputs:
        parser.error("Provide at least one --inputs or --image-inputs path.")

    extractor = VisionEmbeddingExtractor(model_id=args.model_id, local_model_path=args.model_path)
    store = VisualEmbeddingStore(args.store)
    stored = 0

    for item in args.inputs:
        path = Path(item)
        payload = load_payload(path)
        vector = extractor.embed_observation(payload)
        store.upsert(
            VisualEmbeddingRecord(
                key=path.stem,
                label=payload.get("label") or path.stem,
                vector=vector,
                metadata={
                    "source_path": str(path),
                    "source_type": "json",
                    "objects": len(payload.get("objects", [])),
                    "relations": len(payload.get("relations", [])),
                },
            )
        )
        stored += 1

    for item in args.image_inputs:
        path = Path(item)
        payload = image_payload(path)
        vector = extractor.embed_observation(payload)
        store.upsert(
            VisualEmbeddingRecord(
                key=path.stem,
                label=path.stem,
                vector=vector,
                metadata=payload["metadata"],
            )
        )
        stored += 1

    print(json.dumps({"stored": stored, "count": store.count(), "backend": extractor.backend_summary()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

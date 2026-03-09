from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import OpenImagesAnnotationAdapter


def _load_image_ids(path: str | None) -> list[str] | None:
    if not path:
        return None
    values = []
    with Path(path).open('r', encoding='utf-8-sig') as handle:
        for line in handle:
            value = line.strip()
            if value:
                values.append(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert Open Images box annotations into VLSO detector JSONL payloads')
    parser.add_argument('--boxes', required=True, help='Open Images boxes CSV')
    parser.add_argument('--labels', required=True, help='Open Images class description CSV')
    parser.add_argument('--output', required=True, help='output JSONL path')
    parser.add_argument('--segmentations', help='optional segmentation CSV for mask-path attachment')
    parser.add_argument('--image-ids', help='optional text file with image ids to keep')
    parser.add_argument('--limit-images', type=int, help='optional image limit')
    args = parser.parse_args()

    adapter = OpenImagesAnnotationAdapter()
    summary = adapter.build_and_save(
        boxes_path=args.boxes,
        labels_path=args.labels,
        output_path=args.output,
        segmentation_path=args.segmentations,
        image_ids=_load_image_ids(args.image_ids),
        limit_images=args.limit_images,
    )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

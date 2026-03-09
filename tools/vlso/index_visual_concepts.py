from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import RawImageObservationParser, VisualAffordanceFeatureExtractor, VisualConceptDataset, VisualConceptMemory, VisualConceptRecord


def main() -> None:
    parser = argparse.ArgumentParser(description='Index few-shot visual concept exemplars into the VLSO concept memory store')
    parser.add_argument('--labels', required=True, help='JSONL file with image_path and targets')
    parser.add_argument('--store', required=True, help='SQLite output path')
    args = parser.parse_args()

    dataset = VisualConceptDataset().load_jsonl(args.labels)
    image_parser = RawImageObservationParser()
    extractor = VisualAffordanceFeatureExtractor()
    store = VisualConceptMemory(args.store)
    indexed = 0

    for example in dataset:
        observation = image_parser.parse_image(example.image_path).observation
        candidates = extractor.extract(observation)
        by_subject = {item.subject: item for item in candidates}
        for target in example.targets:
            candidate = by_subject.get(target.subject_id)
            if candidate is None and target.subject_id:
                continue
            for label in target.positive_labels:
                chosen = candidate or next(iter(candidates), None)
                if chosen is None:
                    continue
                key = f"{Path(example.image_path).stem}:{chosen.subject}:{label}"
                store.upsert(
                    VisualConceptRecord(
                        key=key,
                        label=label,
                        feature_vector=chosen.features,
                        metadata={
                            'image_path': example.image_path,
                            'subject_id': chosen.subject,
                            'notes': target.notes,
                            'metadata': example.metadata,
                        },
                    )
                )
                indexed += 1
    print(json.dumps({'indexed': indexed, 'count': store.count(), 'store': args.store}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

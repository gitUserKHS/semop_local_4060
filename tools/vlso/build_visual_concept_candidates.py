from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from semop import RawImageObservationParser, VisualAffordanceFeatureExtractor, WeakAffordanceClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description='Build generic visual concept candidate rows for few-shot labeling')
    parser.add_argument('--inputs', nargs='+', required=True, help='image paths')
    parser.add_argument('--output', required=True, help='output JSONL path')
    parser.add_argument('--weights', help='optional affordance/classifier weights JSON')
    args = parser.parse_args()

    image_parser = RawImageObservationParser()
    extractor = VisualAffordanceFeatureExtractor()
    classifier = WeakAffordanceClassifier(args.weights) if args.weights else WeakAffordanceClassifier()
    rows = []
    for image_path in args.inputs:
        observation = image_parser.parse_image(image_path).observation
        candidates = extractor.extract(observation)
        rows.append(
            {
                'image_path': image_path,
                'metadata': observation.metadata,
                'targets': [
                    {
                        'subject_id': item.subject,
                        'parent': item.parent,
                        'bbox': item.object_data.get('bbox'),
                        'shape_hint': item.object_data.get('shape_hint'),
                        'feature_vector': {key: round(float(value), 6) for key, value in item.features.items()},
                        'prediction_details': [
                            {'label': pred.label, 'confidence': pred.confidence, 'score': pred.score}
                            for pred in classifier.predict(item.features, limit=5, threshold=0.0)
                        ],
                        'suggested_labels': [pred.label for pred in classifier.predict(item.features, limit=5, threshold=0.45)],
                        'positive_labels': [],
                        'negative_labels': [],
                        'notes': '',
                    }
                    for item in candidates
                ],
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f'wrote {output}')


if __name__ == '__main__':
    main()

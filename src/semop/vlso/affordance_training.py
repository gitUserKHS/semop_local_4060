from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .affordance_classifier import DEFAULT_AFFORDANCE_CLASSIFIER_PATH, WeakAffordanceClassifier
from .affordance_features import VisualAffordanceFeatureExtractor
from .image_parser import RawImageObservationParser


@dataclass
class AffordanceLabelTarget:
    subject_id: str
    positive_labels: List[str]
    negative_labels: List[str]


@dataclass
class AffordanceLabelExample:
    image_path: str
    targets: List[AffordanceLabelTarget]
    notes: str = ""


@dataclass
class AffordanceTrainingSummary:
    input_path: str
    output_path: str
    examples: int
    targets: int
    classes_updated: List[str]


class AffordanceLabelDataset:
    def load_jsonl(self, path: str | Path) -> List[AffordanceLabelExample]:
        rows: list[AffordanceLabelExample] = []
        for raw in Path(path).read_text(encoding='utf-8-sig').splitlines():
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            rows.append(self._coerce(item))
        return rows

    def _coerce(self, item: Dict[str, Any]) -> AffordanceLabelExample:
        targets: list[AffordanceLabelTarget] = []
        if item.get('targets'):
            for target in item['targets']:
                targets.append(
                    AffordanceLabelTarget(
                        subject_id=str(target.get('subject_id') or ''),
                        positive_labels=list(target.get('positive_labels', [])),
                        negative_labels=list(target.get('negative_labels', [])),
                    )
                )
        else:
            subject_id = str(item.get('subject_id') or '')
            targets.append(
                AffordanceLabelTarget(
                    subject_id=subject_id,
                    positive_labels=list(item.get('positive_labels', [])),
                    negative_labels=list(item.get('negative_labels', [])),
                )
            )
        return AffordanceLabelExample(
            image_path=str(item['image_path']),
            targets=targets,
            notes=str(item.get('notes', '')),
        )


class AffordanceWeightTrainer:
    def __init__(
        self,
        weights_path: str | Path | None = None,
        learning_rate: float = 0.18,
        epochs: int = 10,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_AFFORDANCE_CLASSIFIER_PATH
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset = AffordanceLabelDataset()
        self.image_parser = RawImageObservationParser()
        self.feature_extractor = VisualAffordanceFeatureExtractor()
        self.classifier = WeakAffordanceClassifier(self.weights_path)

    def train_jsonl(self, labels_path: str | Path, output_path: str | Path | None = None) -> AffordanceTrainingSummary:
        labels = self.dataset.load_jsonl(labels_path)
        specs = json.loads(self.weights_path.read_text(encoding='utf-8')) if self.weights_path.exists() else {'classes': {}}
        updated_classes: set[str] = set()
        target_count = sum(len(example.targets) for example in labels)
        for _ in range(self.epochs):
            for example in labels:
                observation = self.image_parser.parse_image(example.image_path).observation
                candidates = self.feature_extractor.extract(observation)
                candidate_map = {item.subject: item for item in candidates}
                for target in example.targets:
                    for label in target.positive_labels:
                        candidate = self._resolve_candidate(candidates, candidate_map, target.subject_id, label)
                        if candidate is None:
                            continue
                        updated_classes.add(label)
                        self._apply_example(specs, label, candidate.features, expected=1.0)
                    for label in target.negative_labels:
                        candidate = self._resolve_candidate(candidates, candidate_map, target.subject_id, label)
                        if candidate is None:
                            continue
                        updated_classes.add(label)
                        self._apply_example(specs, label, candidate.features, expected=0.0)
        destination = Path(output_path) if output_path else self.weights_path
        destination.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding='utf-8')
        return AffordanceTrainingSummary(
            input_path=str(labels_path),
            output_path=str(destination),
            examples=len(labels),
            targets=target_count,
            classes_updated=sorted(updated_classes),
        )

    def _resolve_candidate(self, candidates, candidate_map, subject_id: str, label: str):
        if subject_id and subject_id in candidate_map:
            return candidate_map[subject_id]
        chosen = self.feature_extractor.choose_default_subject(candidates, label)
        return candidate_map.get(chosen) if chosen else None

    def _apply_example(self, specs: Dict[str, Any], label: str, features: Dict[str, float], expected: float) -> None:
        classes = specs.setdefault('classes', {})
        weights = classes.setdefault(label, {'bias': 0.0})
        score = float(weights.get('bias', 0.0))
        for name, value in features.items():
            score += float(weights.get(name, 0.0)) * float(value)
        predicted = 1.0 / (1.0 + math.exp(-score))
        error = expected - predicted
        weights['bias'] = round(float(weights.get('bias', 0.0)) + self.learning_rate * error, 6)
        for name, value in features.items():
            weights[name] = round(float(weights.get(name, 0.0)) + self.learning_rate * error * float(value), 6)

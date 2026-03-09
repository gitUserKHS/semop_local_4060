from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .affordance_classifier import WeakAffordanceClassifier
from .affordance_features import VisualAffordanceFeatureExtractor
from .concept_dataset import VisualConceptDataset
from .concept_memory import VisualConceptMemory, VisualConceptRecord
from .image_parser import RawImageObservationParser


@dataclass
class VisualConceptLearningSummary:
    input_path: str
    store_path: str
    labels_trained: List[str]
    prototype_count: int
    examples: int
    targets: int

    def model_dump(self) -> dict[str, Any]:
        return {
            'input_path': self.input_path,
            'store_path': self.store_path,
            'labels_trained': list(self.labels_trained),
            'prototype_count': self.prototype_count,
            'examples': self.examples,
            'targets': self.targets,
        }


class VisualConceptPrototypeTrainer:
    def __init__(self) -> None:
        self.dataset = VisualConceptDataset()
        self.image_parser = RawImageObservationParser()
        self.feature_extractor = VisualAffordanceFeatureExtractor()

    def train_jsonl(self, labels_path: str | Path, store_path: str | Path, summary_output: str | Path | None = None) -> VisualConceptLearningSummary:
        examples = self.dataset.load_jsonl(labels_path)
        label_vectors: dict[str, list[dict[str, float]]] = {}
        label_metadata: dict[str, list[dict[str, Any]]] = {}
        co_occurrence: dict[str, dict[str, int]] = {}
        target_count = 0
        for example in examples:
            observation = self.image_parser.parse_image(example.image_path).observation
            candidates = self.feature_extractor.extract(observation)
            candidate_map = {item.subject: item for item in candidates}
            example_labels: list[str] = []
            for target in example.targets:
                target_count += 1
                for label in target.positive_labels:
                    candidate = self._resolve_candidate(candidates, candidate_map, target.subject_id, label)
                    if candidate is None:
                        continue
                    label_vectors.setdefault(label, []).append(dict(candidate.features))
                    label_metadata.setdefault(label, []).append(
                        {
                            'image_path': example.image_path,
                            'subject_id': candidate.subject,
                            'notes': target.notes,
                            'source_metadata': example.metadata,
                        }
                    )
                    example_labels.append(label)
            unique_labels = sorted(set(example_labels))
            for left in unique_labels:
                co_occurrence.setdefault(left, {})
                for right in unique_labels:
                    if left == right:
                        continue
                    co_occurrence[left][right] = co_occurrence[left].get(right, 0) + 1
        store = VisualConceptMemory(store_path)
        trained_labels = sorted(label_vectors)
        for label in trained_labels:
            prototype = self._mean_vector(label_vectors[label])
            support = len(label_vectors[label])
            co_labels = [
                {'label': other, 'support': count, 'confidence': round(count / support, 4)}
                for other, count in sorted(co_occurrence.get(label, {}).items(), key=lambda item: (-item[1], item[0]))
                if count > 0
            ]
            store.upsert(
                VisualConceptRecord(
                    key=f'prototype:{label}',
                    label=label,
                    feature_vector=prototype,
                    metadata={
                        'record_type': 'prototype',
                        'support': support,
                        'co_labels': co_labels,
                        'examples': label_metadata.get(label, [])[:8],
                    },
                )
            )
        summary = VisualConceptLearningSummary(
            input_path=str(labels_path),
            store_path=str(store_path),
            labels_trained=trained_labels,
            prototype_count=len(trained_labels),
            examples=len(examples),
            targets=target_count,
        )
        if summary_output:
            Path(summary_output).write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

    def _resolve_candidate(self, candidates, candidate_map, subject_id: str, label: str):
        if subject_id and subject_id in candidate_map:
            return candidate_map[subject_id]
        chosen = self.feature_extractor.choose_default_subject(candidates, label)
        return candidate_map.get(chosen) if chosen else None

    @staticmethod
    def _mean_vector(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {}
        keys = sorted({key for row in rows for key in row})
        return {
            key: round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 6)
            for key in keys
        }


class VisualConceptLabelRecommender:
    def __init__(self, concept_store_path: str | Path | None = None, weights_path: str | Path | None = None) -> None:
        self.concept_memory = VisualConceptMemory(concept_store_path) if concept_store_path else None
        self.classifier = WeakAffordanceClassifier(weights_path) if weights_path else WeakAffordanceClassifier()

    def rank_candidate_rows(self, candidate_rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for row in candidate_rows:
            image_path = str(row.get('image_path', ''))
            for target in row.get('targets', []):
                features = {key: float(value) for key, value in dict(target.get('feature_vector', {})).items()}
                if not features:
                    continue
                predictions = self.classifier.predict(features, limit=3, threshold=0.0)
                uncertainty = self._uncertainty(predictions)
                novelty = self._novelty(features)
                score = round(0.55 * novelty + 0.45 * uncertainty, 6)
                ranked.append(
                    {
                        'image_path': image_path,
                        'subject_id': str(target.get('subject_id', '')),
                        'score': score,
                        'novelty': round(novelty, 6),
                        'uncertainty': round(uncertainty, 6),
                        'suggested_labels': list(target.get('suggested_labels', [])),
                        'bbox': target.get('bbox'),
                        'shape_hint': target.get('shape_hint'),
                    }
                )
        ranked.sort(key=lambda item: (-item['score'], -item['novelty'], item['image_path'], item['subject_id']))
        return ranked[:limit]

    def _novelty(self, features: dict[str, float]) -> float:
        if self.concept_memory is None or self.concept_memory.count() == 0:
            return 1.0
        matches = self.concept_memory.search(features, limit=1)
        if not matches:
            return 1.0
        return max(0.0, 1.0 - matches[0].score)

    @staticmethod
    def _uncertainty(predictions) -> float:
        if not predictions:
            return 1.0
        best = max(item.confidence for item in predictions)
        return 1.0 - abs(best - 0.5) * 2.0

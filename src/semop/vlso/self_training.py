from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .affordance_classifier import WeakAffordanceClassifier
from .concept_memory import VisualConceptMemory, VisualConceptRecord


@dataclass
class PseudoLabelAcceptanceConfig:
    cluster_similarity_threshold: float = 0.9
    pseudo_confidence_threshold: float = 0.72
    cluster_consensus_threshold: float = 0.55
    concept_match_threshold: float = 0.86
    min_cluster_size: int = 2


@dataclass
class VisualPseudoCluster:
    cluster_id: str
    support: int
    accepted_labels: List[dict[str, Any]]
    suggested_labels: List[dict[str, Any]]
    centroid: Dict[str, float]
    review_priority: float = 0.0
    review_reason: str = ''
    members: List[dict[str, Any]] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            'cluster_id': self.cluster_id,
            'support': self.support,
            'accepted_labels': list(self.accepted_labels),
            'suggested_labels': list(self.suggested_labels),
            'centroid': dict(self.centroid),
            'review_priority': self.review_priority,
            'review_reason': self.review_reason,
            'members': list(self.members),
        }


@dataclass
class VisualSelfTrainingSummary:
    input_path: str
    store_path: str
    cluster_count: int
    candidate_targets: int
    accepted_prototypes: int
    accepted_labels: List[str]
    unlabeled_clusters: int

    def model_dump(self) -> dict[str, Any]:
        return {
            'input_path': self.input_path,
            'store_path': self.store_path,
            'cluster_count': self.cluster_count,
            'candidate_targets': self.candidate_targets,
            'accepted_prototypes': self.accepted_prototypes,
            'accepted_labels': list(self.accepted_labels),
            'unlabeled_clusters': self.unlabeled_clusters,
        }


class VisualConceptSelfTrainer:
    def __init__(self, concept_store_path: str | Path | None = None, weights_path: str | Path | None = None) -> None:
        self.classifier = WeakAffordanceClassifier(weights_path) if weights_path else WeakAffordanceClassifier()
        self.reference_memory = VisualConceptMemory(concept_store_path) if concept_store_path else None

    def train_candidates_jsonl(
        self,
        candidates_path: str | Path,
        store_path: str | Path,
        summary_output: str | Path | None = None,
        config: PseudoLabelAcceptanceConfig | None = None,
    ) -> VisualSelfTrainingSummary:
        cfg = config or PseudoLabelAcceptanceConfig()
        targets = self._load_targets(candidates_path)
        clusters = self._cluster_targets(targets, cfg.cluster_similarity_threshold)
        store = VisualConceptMemory(store_path)
        accepted_labels: list[str] = []
        accepted_prototypes = 0
        unlabeled_clusters = 0
        cluster_rows: list[VisualPseudoCluster] = []

        for index, cluster in enumerate(clusters, start=1):
            cluster_id = f'cluster_{index:04d}'
            suggested = self._aggregate_cluster_labels(cluster, cfg)
            accepted = [row for row in suggested if row['accept']]
            if len(cluster) < cfg.min_cluster_size:
                accepted = []
            if not accepted:
                unlabeled_clusters += 1
            centroid = self._mean_vector([item['feature_vector'] for item in cluster])
            members = [
                {
                    'image_path': item['image_path'],
                    'subject_id': item['subject_id'],
                    'shape_hint': item.get('shape_hint', ''),
                    'suggested_labels': list(item.get('suggested_labels', [])),
                }
                for item in cluster[:8]
            ]
            review_priority, review_reason = self._review_priority(cluster, suggested, accepted)
            cluster_rows.append(
                VisualPseudoCluster(
                    cluster_id=cluster_id,
                    support=len(cluster),
                    accepted_labels=accepted,
                    suggested_labels=suggested,
                    centroid=centroid,
                    review_priority=review_priority,
                    review_reason=review_reason,
                    members=members,
                )
            )
            for label_row in accepted:
                label = str(label_row['label'])
                accepted_labels.append(label)
                accepted_prototypes += 1
                store.upsert(
                    VisualConceptRecord(
                        key=f'pseudo:{label}:{cluster_id}',
                        label=label,
                        feature_vector=centroid,
                        metadata={
                            'record_type': 'pseudo_prototype',
                            'cluster_id': cluster_id,
                            'support': len(cluster),
                            'support_ratio': label_row['support_ratio'],
                            'mean_confidence': label_row['mean_confidence'],
                            'source': 'self_training',
                            'members': members,
                        },
                    )
                )

        summary = VisualSelfTrainingSummary(
            input_path=str(candidates_path),
            store_path=str(store_path),
            cluster_count=len(clusters),
            candidate_targets=len(targets),
            accepted_prototypes=accepted_prototypes,
            accepted_labels=sorted(set(accepted_labels)),
            unlabeled_clusters=unlabeled_clusters,
        )
        if summary_output:
            payload = {
                'summary': summary.model_dump(),
                'clusters': [item.model_dump() for item in sorted(cluster_rows, key=lambda row: (-row.review_priority, row.cluster_id))],
            }
            Path(summary_output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

    def _load_targets(self, candidates_path: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with Path(candidates_path).open('r', encoding='utf-8-sig') as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                image_path = str(payload.get('image_path', ''))
                for target in payload.get('targets', []):
                    feature_vector = {key: float(value) for key, value in dict(target.get('feature_vector', {})).items()}
                    if not feature_vector:
                        continue
                    prediction_details = target.get('prediction_details') or []
                    if not prediction_details:
                        prediction_details = [
                            {'label': item.label, 'confidence': item.confidence, 'score': item.score}
                            for item in self.classifier.predict(feature_vector, limit=5, threshold=0.0)
                        ]
                    rows.append(
                        {
                            'image_path': image_path,
                            'subject_id': str(target.get('subject_id', '')),
                            'shape_hint': str(target.get('shape_hint', '')),
                            'bbox': target.get('bbox'),
                            'feature_vector': feature_vector,
                            'prediction_details': prediction_details,
                            'suggested_labels': list(target.get('suggested_labels', [])),
                        }
                    )
        return rows

    def _cluster_targets(self, targets: list[dict[str, Any]], threshold: float) -> list[list[dict[str, Any]]]:
        clusters: list[list[dict[str, Any]]] = []
        centroids: list[dict[str, float]] = []
        for target in targets:
            best_index = -1
            best_score = -1.0
            for index, centroid in enumerate(centroids):
                score = self._cosine(target['feature_vector'], centroid)
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index >= 0 and best_score >= threshold:
                clusters[best_index].append(target)
                centroids[best_index] = self._mean_vector([item['feature_vector'] for item in clusters[best_index]])
            else:
                clusters.append([target])
                centroids.append(dict(target['feature_vector']))
        return clusters

    def _aggregate_cluster_labels(self, cluster: list[dict[str, Any]], cfg: PseudoLabelAcceptanceConfig) -> list[dict[str, Any]]:
        label_scores: dict[str, float] = {}
        label_support: dict[str, int] = {}
        for item in cluster:
            seen_labels: set[str] = set()
            for pred in item.get('prediction_details', []):
                label = str(pred.get('label', '')).strip()
                if not label:
                    continue
                confidence = float(pred.get('confidence', 0.0) or 0.0)
                label_scores[label] = label_scores.get(label, 0.0) + confidence
                if label not in seen_labels:
                    label_support[label] = label_support.get(label, 0) + 1
                    seen_labels.add(label)
            if self.reference_memory is not None:
                matches = self.reference_memory.search(item['feature_vector'], limit=2)
                for match in matches:
                    if match.score < cfg.concept_match_threshold:
                        continue
                    label_scores[match.label] = label_scores.get(match.label, 0.0) + match.score
                    if match.label not in seen_labels:
                        label_support[match.label] = label_support.get(match.label, 0) + 1
                        seen_labels.add(match.label)
        rows: list[dict[str, Any]] = []
        cluster_size = max(1, len(cluster))
        for label, total_score in sorted(label_scores.items(), key=lambda item: (-item[1], item[0])):
            support = label_support.get(label, 0)
            support_ratio = support / cluster_size
            mean_confidence = total_score / cluster_size
            accept = mean_confidence >= cfg.pseudo_confidence_threshold and support_ratio >= cfg.cluster_consensus_threshold
            rows.append(
                {
                    'label': label,
                    'support': support,
                    'support_ratio': round(support_ratio, 4),
                    'mean_confidence': round(mean_confidence, 4),
                    'accept': accept,
                }
            )
        return rows

    @staticmethod
    def _review_priority(cluster: list[dict[str, Any]], suggested: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> tuple[float, str]:
        reasons: list[str] = []
        priority = 0.0
        ordered = sorted(suggested, key=lambda item: (-float(item.get('mean_confidence', 0.0)), item.get('label', '')))
        if not accepted:
            priority += 1.0
            reasons.append('no accepted labels')
        if len(ordered) >= 2:
            gap = float(ordered[0].get('mean_confidence', 0.0)) - float(ordered[1].get('mean_confidence', 0.0))
            if gap < 0.12:
                priority += round(0.12 - gap, 4) * 4.0
                reasons.append('low margin between top labels')
        mixed_support = [row for row in ordered[:3] if float(row.get('support_ratio', 0.0)) >= 0.35]
        if len(mixed_support) >= 2:
            priority += 0.35
            reasons.append('mixed cluster labels')
        if len(cluster) <= 2:
            priority += 0.15
            reasons.append('small cluster')
        return round(priority, 4), '; '.join(reasons) if reasons else 'stable cluster'

    @staticmethod
    def _mean_vector(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {}
        keys = sorted({key for row in rows for key in row})
        return {
            key: round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 6)
            for key in keys
        }

    @staticmethod
    def _cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        keys = set(left) | set(right)
        dot = sum(float(left.get(key, 0.0)) * float(right.get(key, 0.0)) for key in keys)
        left_norm = math.sqrt(sum(float(left.get(key, 0.0)) ** 2 for key in keys)) or 1.0
        right_norm = math.sqrt(sum(float(right.get(key, 0.0)) ** 2 for key in keys)) or 1.0
        return dot / (left_norm * right_norm)

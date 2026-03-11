from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .cluster_review import VisualClusterReviewStore
from .concept_memory import VisualConceptMemory, VisualConceptRecord
from .operator_learning import VisualOperatorPrototypeTrainer


@dataclass
class VisualReviewRetrainSummary:
    summary_path: str
    review_path: str
    labels_path: str
    concept_store_path: str
    operator_store_path: str
    approved_clusters: int
    approved_targets: int
    approved_labels: list[str]
    operator_summary: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class VisualApprovedReviewRetrainer:
    def export_and_retrain(
        self,
        summary_path: str | Path,
        review_path: str | Path,
        labels_path: str | Path,
        concept_store_path: str | Path,
        operator_store_path: str | Path,
        operator_summary_output: str | Path | None = None,
    ) -> VisualReviewRetrainSummary:
        summary_payload = json.loads(Path(summary_path).read_text(encoding='utf-8-sig'))
        decisions = VisualClusterReviewStore(review_path).load_decisions()
        rows_by_image: dict[str, dict[str, Any]] = {}
        concept_store = VisualConceptMemory(concept_store_path)
        approved_clusters = 0
        approved_targets = 0
        approved_labels: set[str] = set()

        for cluster in summary_payload.get('clusters', []):
            if not isinstance(cluster, dict):
                continue
            cluster_id = str(cluster.get('cluster_id', ''))
            decision = decisions.get(cluster_id, {})
            if str(decision.get('status', 'pending')) != 'approved':
                continue
            labels = [str(item).strip() for item in decision.get('approved_labels', []) if str(item).strip()]
            if not labels:
                labels = [
                    str(item.get('label', '')).strip()
                    for item in cluster.get('accepted_labels', [])
                    if isinstance(item, dict) and str(item.get('label', '')).strip()
                ]
            labels = list(dict.fromkeys(labels))
            if not labels:
                continue
            approved_clusters += 1
            approved_labels.update(labels)
            centroid = {str(k): float(v) for k, v in dict(cluster.get('centroid', {})).items()}
            support = int(cluster.get('support', 0) or 0)
            for label in labels:
                concept_store.upsert(
                    VisualConceptRecord(
                        key=f"approved:{label}:{cluster_id}",
                        label=label,
                        feature_vector=centroid,
                        metadata={
                            'record_type': 'approved_cluster_prototype',
                            'cluster_id': cluster_id,
                            'support': support,
                            'source': 'cluster_review',
                        },
                    )
                )
            for member in cluster.get('members', []):
                if not isinstance(member, dict):
                    continue
                image_path = str(member.get('image_path', '')).strip()
                subject_id = str(member.get('subject_id', '')).strip()
                if not image_path or not subject_id:
                    continue
                row = rows_by_image.setdefault(image_path, {'image_path': image_path, 'targets': []})
                existing = None
                for target in row['targets']:
                    if str(target.get('subject_id', '')) == subject_id:
                        existing = target
                        break
                if existing is None:
                    existing = {'subject_id': subject_id, 'positive_labels': [], 'negative_labels': []}
                    row['targets'].append(existing)
                    approved_targets += 1
                merged = list(dict.fromkeys([*existing['positive_labels'], *labels]))
                existing['positive_labels'] = merged

        labels_out = Path(labels_path)
        labels_out.parent.mkdir(parents=True, exist_ok=True)
        with labels_out.open('w', encoding='utf-8') as handle:
            for row in rows_by_image.values():
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')

        operator_summary = VisualOperatorPrototypeTrainer().train_jsonl(labels_out, operator_store_path, summary_output=operator_summary_output).model_dump()
        return VisualReviewRetrainSummary(
            summary_path=str(summary_path),
            review_path=str(review_path),
            labels_path=str(labels_path),
            concept_store_path=str(concept_store_path),
            operator_store_path=str(operator_store_path),
            approved_clusters=approved_clusters,
            approved_targets=approved_targets,
            approved_labels=sorted(approved_labels),
            operator_summary=operator_summary,
        )

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class VisualClusterReviewDecision:
    cluster_id: str
    status: str
    note: str = ''
    approved_labels: List[str] | None = None

    def model_dump(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['approved_labels'] = list(self.approved_labels or [])
        return payload


class VisualClusterReviewStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_summary(self, summary_path: str | Path) -> dict[str, Any]:
        return json.loads(Path(summary_path).read_text(encoding='utf-8-sig'))

    def load_decisions(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding='utf-8-sig'))
        if isinstance(payload, dict):
            return {str(key): dict(value) for key, value in payload.items() if isinstance(value, dict)}
        return {}

    def save_decision(self, decision: VisualClusterReviewDecision) -> None:
        payload = self.load_decisions()
        payload[decision.cluster_id] = decision.model_dump()
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def list_clusters(self, summary_path: str | Path) -> list[dict[str, Any]]:
        payload = self.load_summary(summary_path)
        decisions = self.load_decisions()
        clusters = []
        for cluster in payload.get('clusters', []):
            if not isinstance(cluster, dict):
                continue
            cluster_id = str(cluster.get('cluster_id', ''))
            merged = dict(cluster)
            merged['review'] = decisions.get(cluster_id, {})
            clusters.append(merged)
        return clusters

    def stats(self, summary_path: str | Path) -> dict[str, int]:
        decisions = self.load_decisions()
        clusters = self.list_clusters(summary_path)
        summary = {'total': len(clusters), 'approved': 0, 'rejected': 0, 'pending': 0}
        for cluster in clusters:
            status = str(cluster.get('review', {}).get('status', 'pending'))
            if status == 'approved':
                summary['approved'] += 1
            elif status == 'rejected':
                summary['rejected'] += 1
            else:
                summary['pending'] += 1
        return summary

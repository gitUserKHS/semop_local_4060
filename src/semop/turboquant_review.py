from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class QuantizedReviewAssignment:
    label: str
    domain: str
    scenario: str
    cluster_id: int
    distortion: float
    novelty_score: float
    text_preview: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurboQuantReviewSummary:
    bit_width: int
    dimension: int
    codebook_size: int
    used_clusters: int
    average_distortion: float
    average_novelty: float
    coverage_score: float
    domain_scenario_pairs: list[str] = field(default_factory=list)
    cluster_loads: dict[str, int] = field(default_factory=dict)
    assignments: list[QuantizedReviewAssignment] = field(default_factory=list)
    output_path: str = ''

    def model_dump(self) -> dict[str, Any]:
        return {
            'bit_width': self.bit_width,
            'dimension': self.dimension,
            'codebook_size': self.codebook_size,
            'used_clusters': self.used_clusters,
            'average_distortion': self.average_distortion,
            'average_novelty': self.average_novelty,
            'coverage_score': self.coverage_score,
            'domain_scenario_pairs': list(self.domain_scenario_pairs),
            'cluster_loads': dict(self.cluster_loads),
            'assignments': [item.model_dump() for item in self.assignments],
            'output_path': self.output_path,
        }


class TurboQuantReviewPlanner:
    def __init__(
        self,
        *,
        dimension: int = 64,
        bit_width: int = 8,
        max_clusters: int = 24,
        spawn_threshold: float = 0.8,
        update_rate: float = 0.22,
    ) -> None:
        self.dimension = max(8, int(dimension))
        self.bit_width = max(2, int(bit_width))
        self.max_clusters = max(4, min(int(max_clusters), 2 ** self.bit_width))
        self.spawn_threshold = max(0.05, float(spawn_threshold))
        self.update_rate = max(0.01, min(1.0, float(update_rate)))
        self._centroids: list[list[float]] = []
        self._counts: list[int] = []
        self._assignments: list[QuantizedReviewAssignment] = []

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_가-힣]+", str(text).lower())

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = self._tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha1(token.encode('utf-8')).hexdigest()
            index = int(digest[:8], 16) % self.dimension
            sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    @staticmethod
    def _distance(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))

    def score_text(self, text: str) -> float:
        vector = self._vectorize(text)
        if not self._centroids:
            return 1.0
        distance = min(self._distance(vector, centroid) for centroid in self._centroids)
        return round(min(1.0, distance / self.spawn_threshold), 4)

    def add(self, *, label: str, text: str, domain: str, scenario: str) -> QuantizedReviewAssignment:
        vector = self._vectorize(text)
        if not self._centroids:
            self._centroids.append(list(vector))
            self._counts.append(1)
            assignment = QuantizedReviewAssignment(label=label, domain=domain, scenario=scenario, cluster_id=0, distortion=0.0, novelty_score=1.0, text_preview=str(text)[:120])
            self._assignments.append(assignment)
            return assignment
        distances = [self._distance(vector, centroid) for centroid in self._centroids]
        cluster_id = min(range(len(distances)), key=lambda index: distances[index])
        distortion = float(distances[cluster_id])
        novelty_score = round(min(1.0, distortion / self.spawn_threshold), 4)
        if distortion > self.spawn_threshold and len(self._centroids) < self.max_clusters:
            cluster_id = len(self._centroids)
            self._centroids.append(list(vector))
            self._counts.append(1)
            distortion = 0.0
            novelty_score = 1.0
        else:
            centroid = self._centroids[cluster_id]
            rate = self.update_rate / float(max(1, self._counts[cluster_id]))
            for index, value in enumerate(vector):
                centroid[index] = float(centroid[index]) + (float(value) - float(centroid[index])) * rate
            self._counts[cluster_id] += 1
        assignment = QuantizedReviewAssignment(label=label, domain=domain, scenario=scenario, cluster_id=cluster_id, distortion=round(distortion, 4), novelty_score=novelty_score, text_preview=str(text)[:120])
        self._assignments.append(assignment)
        return assignment

    def summarize(self, output_path: str | Path | None = None) -> TurboQuantReviewSummary:
        used_clusters = len({item.cluster_id for item in self._assignments})
        average_distortion = round(sum(item.distortion for item in self._assignments) / float(len(self._assignments) or 1), 4)
        average_novelty = round(sum(item.novelty_score for item in self._assignments) / float(len(self._assignments) or 1), 4)
        coverage_score = round((used_clusters / float(max(1, self.max_clusters)) + min(1.0, len({(item.domain, item.scenario) for item in self._assignments}) / 8.0)) / 2.0, 4)
        cluster_loads: dict[str, int] = {}
        for item in self._assignments:
            key = str(item.cluster_id)
            cluster_loads[key] = cluster_loads.get(key, 0) + 1
        summary = TurboQuantReviewSummary(
            bit_width=self.bit_width,
            dimension=self.dimension,
            codebook_size=self.max_clusters,
            used_clusters=used_clusters,
            average_distortion=average_distortion,
            average_novelty=average_novelty,
            coverage_score=coverage_score,
            domain_scenario_pairs=sorted({f'{item.domain}/{item.scenario}' for item in self._assignments}),
            cluster_loads=cluster_loads,
            assignments=list(self._assignments),
            output_path=str(output_path or ''),
        )
        if output_path:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

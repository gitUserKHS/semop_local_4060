from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from .concept_memory import VisualConceptMemory, VisualConceptMatch
from .operator_learning import VisualOperatorMatch, VisualOperatorMemory


@dataclass
class HybridMemoryMatch:
    label: str
    score: float
    source: str
    metadata: Dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VisualHybridMemoryResult:
    local_matches: List[HybridMemoryMatch]
    global_matches: List[HybridMemoryMatch]
    fused_labels: List[dict[str, Any]]

    def model_dump(self) -> dict[str, Any]:
        return {
            'local_matches': [item.model_dump() for item in self.local_matches],
            'global_matches': [item.model_dump() for item in self.global_matches],
            'fused_labels': list(self.fused_labels),
        }


class VisualHybridMemory:
    def __init__(
        self,
        concept_memory: VisualConceptMemory | None = None,
        operator_memory: VisualOperatorMemory | None = None,
        concept_threshold: float = 0.84,
        operator_threshold: float = 0.68,
    ) -> None:
        self.concept_memory = concept_memory
        self.operator_memory = operator_memory
        self.concept_threshold = concept_threshold
        self.operator_threshold = operator_threshold

    def retrieve(self, feature_vector: Dict[str, float], signature: List[str], limit: int = 3) -> VisualHybridMemoryResult:
        local_matches: list[HybridMemoryMatch] = []
        global_matches: list[HybridMemoryMatch] = []
        fused_scores: dict[str, float] = {}
        fused_sources: dict[str, set[str]] = {}

        if self.concept_memory is not None:
            for match in self.concept_memory.search(feature_vector, limit=limit):
                if match.score < self.concept_threshold:
                    continue
                local_matches.append(HybridMemoryMatch(label=match.label, score=match.score, source='concept_memory', metadata=match.metadata))
                fused_scores[match.label] = fused_scores.get(match.label, 0.0) + 0.65 * match.score
                fused_sources.setdefault(match.label, set()).add('concept_memory')
                for row in match.metadata.get('co_labels', []):
                    if not isinstance(row, dict):
                        continue
                    label = str(row.get('label', ''))
                    confidence = float(row.get('confidence', 0.0) or 0.0)
                    if not label or confidence < 0.45:
                        continue
                    fused_scores[label] = fused_scores.get(label, 0.0) + 0.35 * confidence
                    fused_sources.setdefault(label, set()).add('concept_co_label')

        if self.operator_memory is not None:
            for match in self.operator_memory.search(feature_vector, signature, limit=limit):
                if match.score < self.operator_threshold:
                    continue
                global_matches.append(HybridMemoryMatch(label=match.operator_name, score=match.score, source='operator_memory', metadata=match.metadata))
                for row in match.metadata.get('implied_labels', []):
                    if not isinstance(row, dict):
                        continue
                    label = str(row.get('label', ''))
                    confidence = float(row.get('confidence', 0.0) or 0.0)
                    if not label or confidence < 0.4:
                        continue
                    fused_scores[label] = fused_scores.get(label, 0.0) + 0.55 * match.score * confidence
                    fused_sources.setdefault(label, set()).add(match.operator_name)

        fused_labels = [
            {
                'label': label,
                'score': round(score, 6),
                'sources': sorted(fused_sources.get(label, set())),
            }
            for label, score in sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
        ]
        return VisualHybridMemoryResult(local_matches=local_matches, global_matches=global_matches, fused_labels=fused_labels)

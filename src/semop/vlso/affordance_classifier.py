from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_AFFORDANCE_CLASSIFIER_PATH = Path(__file__).resolve().parents[3] / 'data' / 'knowledge' / 'vlso_affordance_classifier.json'


@dataclass
class AffordancePrediction:
    label: str
    score: float
    confidence: float
    feature_values: Dict[str, float]


class WeakAffordanceClassifier:
    def __init__(self, weights_path: str | Path | None = None) -> None:
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_AFFORDANCE_CLASSIFIER_PATH
        self.spec = self._load_spec(self.weights_path)

    def predict(self, features: Dict[str, float], limit: int = 3, threshold: float = 0.55) -> List[AffordancePrediction]:
        outputs: list[AffordancePrediction] = []
        for label, weights in self.spec.get('classes', {}).items():
            score = float(weights.get('bias', 0.0))
            for name, value in features.items():
                score += float(weights.get(name, 0.0)) * float(value)
            confidence = 1.0 / (1.0 + math.exp(-score))
            if confidence >= threshold:
                outputs.append(
                    AffordancePrediction(
                        label=label,
                        score=round(score, 4),
                        confidence=round(confidence, 4),
                        feature_values={key: round(float(value), 4) for key, value in features.items() if abs(value) > 1e-9},
                    )
                )
        outputs.sort(key=lambda item: item.confidence, reverse=True)
        return outputs[:limit]

    def predict_many(self, feature_rows: Iterable[Dict[str, float]], limit: int = 3, threshold: float = 0.55) -> List[List[AffordancePrediction]]:
        return [self.predict(features, limit=limit, threshold=threshold) for features in feature_rows]

    @staticmethod
    def _load_spec(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {'classes': {}}
        return json.loads(path.read_text(encoding='utf-8'))

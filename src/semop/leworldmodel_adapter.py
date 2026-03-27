from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class LeWorldModelSequenceRecord:
    sequence_id: str
    domain: str
    steps: list[list[str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LeWorldModelArtifact:
    artifact_path: str
    domain: str
    latent_dim: int
    example_count: int
    step_count: int
    sigreg_score: float
    predictive_consistency: float
    straightness_score: float
    goal_prototype: list[float] = field(default_factory=list)
    transition_delta: list[float] = field(default_factory=list)
    top_tokens: list[str] = field(default_factory=list)
    token_frequencies: dict[str, int] = field(default_factory=dict)
    source_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'LeWorldModelArtifact':
        return cls(
            artifact_path=str(payload.get('artifact_path') or ''),
            domain=str(payload.get('domain') or 'generic'),
            latent_dim=int(payload.get('latent_dim') or 0),
            example_count=int(payload.get('example_count') or 0),
            step_count=int(payload.get('step_count') or 0),
            sigreg_score=float(payload.get('sigreg_score') or 0.0),
            predictive_consistency=float(payload.get('predictive_consistency') or 0.0),
            straightness_score=float(payload.get('straightness_score') or 0.0),
            goal_prototype=[float(item) for item in payload.get('goal_prototype', []) or []],
            transition_delta=[float(item) for item in payload.get('transition_delta', []) or []],
            top_tokens=[str(item) for item in payload.get('top_tokens', []) or []],
            token_frequencies={str(k): int(v) for k, v in (payload.get('token_frequencies') or {}).items()},
            source_summary=dict(payload.get('source_summary') or {}),
            notes=[str(item) for item in payload.get('notes', []) or []],
        )


@dataclass
class LeWorldModelAlignment:
    alignment_score: float
    goal_similarity: float
    predicted_goal_similarity: float
    transition_similarity: float
    straightness_estimate: float
    token_overlap: float
    notes: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class LeWorldModelAdapter:
    def __init__(self, *, latent_dim: int = 48, projection_count: int = 8) -> None:
        self.latent_dim = max(8, int(latent_dim))
        self.projection_count = max(4, int(projection_count))

    def fit(
        self,
        records: Sequence[LeWorldModelSequenceRecord],
        *,
        output_path: str,
        domain: str,
        source_summary: dict[str, Any] | None = None,
    ) -> LeWorldModelArtifact:
        flattened_steps: list[list[float]] = []
        deltas: list[list[float]] = []
        goal_vectors: list[list[float]] = []
        token_frequencies: dict[str, int] = {}
        straightness_values: list[float] = []
        predictive_values: list[float] = []
        step_count = 0
        for record in records:
            vectors = [self._encode_tokens(step) for step in record.steps if step]
            if not vectors:
                continue
            flattened_steps.extend(vectors)
            goal_vectors.append(vectors[-1])
            step_count += len(vectors)
            for step in record.steps:
                for token in step:
                    token_frequencies[token] = token_frequencies.get(token, 0) + 1
            if len(vectors) > 1:
                path_length = 0.0
                for current, nxt in zip(vectors, vectors[1:]):
                    delta = [b - a for a, b in zip(current, nxt)]
                    deltas.append(delta)
                    path_length += self._norm(delta)
                direct = self._norm([b - a for a, b in zip(vectors[0], vectors[-1])])
                straightness_values.append(direct / path_length if path_length > 1e-8 else 1.0)
        transition_delta = self._mean_vector(deltas, self.latent_dim)
        goal_prototype = self._mean_vector(goal_vectors, self.latent_dim)
        for record in records:
            vectors = [self._encode_tokens(step) for step in record.steps if step]
            if len(vectors) > 1 and transition_delta:
                for current, nxt in zip(vectors, vectors[1:]):
                    predicted = [a + b for a, b in zip(current, transition_delta)]
                    predictive_values.append(max(0.0, self._cosine(predicted, nxt)))
        artifact = LeWorldModelArtifact(
            artifact_path=str(output_path),
            domain=domain,
            latent_dim=self.latent_dim,
            example_count=len(records),
            step_count=step_count,
            sigreg_score=round(self._sigreg_score(flattened_steps), 4),
            predictive_consistency=round(sum(predictive_values) / float(len(predictive_values) or 1), 4),
            straightness_score=round(sum(straightness_values) / float(len(straightness_values) or 1), 4),
            goal_prototype=[round(item, 6) for item in goal_prototype],
            transition_delta=[round(item, 6) for item in transition_delta],
            top_tokens=[item for item, _ in sorted(token_frequencies.items(), key=lambda kv: (-kv[1], kv[0]))[:24]],
            token_frequencies=token_frequencies,
            source_summary=dict(source_summary or {}),
            notes=[
                'LeWM-inspired adaptation: next-embedding prediction plus Gaussian-style latent regularization proxy.',
                'This implementation is a lightweight repo adaptation, not the full published pixel JEPA training stack.',
            ],
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        return artifact

    def load(self, path: str | None) -> LeWorldModelArtifact | None:
        if not path or not Path(path).exists():
            return None
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        return LeWorldModelArtifact.from_dict(payload if isinstance(payload, dict) else {})

    def score_sequence(self, artifact: LeWorldModelArtifact | None, steps: Sequence[Sequence[str]]) -> LeWorldModelAlignment:
        if artifact is None or not artifact.goal_prototype:
            return LeWorldModelAlignment(
                alignment_score=0.0,
                goal_similarity=0.0,
                predicted_goal_similarity=0.0,
                transition_similarity=0.0,
                straightness_estimate=0.0,
                token_overlap=0.0,
                notes=['LeWM prior unavailable.'],
            )
        vectors = [self._encode_tokens(step, latent_dim=artifact.latent_dim) for step in steps if step]
        if not vectors:
            return LeWorldModelAlignment(
                alignment_score=0.0,
                goal_similarity=0.0,
                predicted_goal_similarity=0.0,
                transition_similarity=0.0,
                straightness_estimate=0.0,
                token_overlap=0.0,
                notes=['No latent steps were available for scoring.'],
            )
        goal_similarity = max(0.0, self._cosine(vectors[-1], artifact.goal_prototype))
        predicted = [a + b for a, b in zip(vectors[-1], artifact.transition_delta or [0.0] * artifact.latent_dim)]
        predicted_goal_similarity = max(0.0, self._cosine(predicted, artifact.goal_prototype))
        if len(vectors) > 1:
            current_delta = [b - a for a, b in zip(vectors[-2], vectors[-1])]
            transition_similarity = max(0.0, self._cosine(current_delta, artifact.transition_delta))
            path_length = sum(self._norm([b - a for a, b in zip(cur, nxt)]) for cur, nxt in zip(vectors, vectors[1:]))
            direct = self._norm([b - a for a, b in zip(vectors[0], vectors[-1])])
            straightness = direct / path_length if path_length > 1e-8 else 1.0
        else:
            transition_similarity = 0.5
            straightness = 1.0
        step_tokens = {token for step in steps for token in step}
        top_tokens = set(artifact.top_tokens)
        overlap = len(step_tokens & top_tokens) / float(len(step_tokens) or 1)
        alignment = (0.35 * goal_similarity) + (0.30 * predicted_goal_similarity) + (0.20 * transition_similarity) + (0.10 * straightness) + (0.05 * overlap)
        notes = [
            f'goal similarity={goal_similarity:.3f}',
            f'predicted goal similarity={predicted_goal_similarity:.3f}',
            f'transition similarity={transition_similarity:.3f}',
            f'straightness={straightness:.3f}',
        ]
        return LeWorldModelAlignment(
            alignment_score=round(min(0.999, alignment), 4),
            goal_similarity=round(goal_similarity, 4),
            predicted_goal_similarity=round(predicted_goal_similarity, 4),
            transition_similarity=round(transition_similarity, 4),
            straightness_estimate=round(straightness, 4),
            token_overlap=round(overlap, 4),
            notes=notes,
        )

    def make_record(self, sequence_id: str, domain: str, step_items: Sequence[str | Sequence[str]], metadata: dict[str, Any] | None = None) -> LeWorldModelSequenceRecord:
        steps: list[list[str]] = []
        for item in step_items:
            if isinstance(item, str):
                tokens = self.tokenize(item)
            else:
                tokens = [token for raw in item for token in self.tokenize(str(raw))]
            if tokens:
                steps.append(tokens)
        return LeWorldModelSequenceRecord(sequence_id=sequence_id, domain=domain, steps=steps, metadata=dict(metadata or {}))

    @staticmethod
    def tokenize(value: str) -> list[str]:
        return [item.lower() for item in _TOKEN_RE.findall(value or '') if item.strip()]

    def _encode_tokens(self, tokens: Sequence[str], *, latent_dim: int | None = None) -> list[float]:
        dim = latent_dim or self.latent_dim
        vector = [0.0] * dim
        for token in tokens:
            if not token:
                continue
            digest = hashlib.sha256(token.encode('utf-8')).hexdigest()
            index = int(digest[:12], 16) % dim
            vector[index] += 1.0
        norm = self._norm(vector)
        if norm > 1e-8:
            vector = [item / norm for item in vector]
        return vector

    def _sigreg_score(self, vectors: Sequence[Sequence[float]]) -> float:
        if not vectors:
            return 0.0
        mean = self._mean_vector(vectors, self.latent_dim)
        centered = [[value - center for value, center in zip(vector, mean)] for vector in vectors]
        directions = [self._random_direction(self.latent_dim, seed=index) for index in range(self.projection_count)]
        scores: list[float] = []
        for direction in directions:
            projected = [sum(value * axis for value, axis in zip(vector, direction)) for vector in centered]
            proj_mean = sum(projected) / float(len(projected))
            variance = sum((value - proj_mean) ** 2 for value in projected) / float(len(projected) or 1)
            std = math.sqrt(max(variance, 1e-8))
            normalized = [(value - proj_mean) / std for value in projected]
            skew = sum(value ** 3 for value in normalized) / float(len(normalized) or 1)
            kurt = sum(value ** 4 for value in normalized) / float(len(normalized) or 1)
            scores.append(1.0 / (1.0 + abs(skew) + abs(kurt - 3.0)))
        return sum(scores) / float(len(scores) or 1)

    @staticmethod
    def _random_direction(dim: int, *, seed: int) -> list[float]:
        rng = random.Random(seed)
        vector = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(item * item for item in vector)) or 1.0
        return [item / norm for item in vector]

    @staticmethod
    def _mean_vector(vectors: Sequence[Sequence[float]], dim: int) -> list[float]:
        if not vectors:
            return [0.0] * dim
        result = [0.0] * dim
        for vector in vectors:
            for index, value in enumerate(vector[:dim]):
                result[index] += value
        count = float(len(vectors))
        return [value / count for value in result]

    @staticmethod
    def _norm(vector: Sequence[float]) -> float:
        return math.sqrt(sum(value * value for value in vector))

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm <= 1e-8 or right_norm <= 1e-8:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

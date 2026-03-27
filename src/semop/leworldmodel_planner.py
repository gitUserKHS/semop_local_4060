from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .leworldmodel_adapter import LeWorldModelAdapter, LeWorldModelArtifact


@dataclass
class LeWorldModelPlannedAction:
    position: int
    token: str
    probability: float

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LeWorldModelPlan:
    enabled: bool
    horizon: int
    iterations: int
    sample_count: int
    elite_count: int
    plan_tokens: list[str] = field(default_factory=list)
    plan_score: float = 0.0
    predicted_goal_similarity: float = 0.0
    transition_similarity: float = 0.0
    token_overlap: float = 0.0
    sampled_action_count: int = 0
    actions: list[LeWorldModelPlannedAction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            'enabled': self.enabled,
            'horizon': self.horizon,
            'iterations': self.iterations,
            'sample_count': self.sample_count,
            'elite_count': self.elite_count,
            'plan_tokens': list(self.plan_tokens),
            'plan_score': self.plan_score,
            'predicted_goal_similarity': self.predicted_goal_similarity,
            'transition_similarity': self.transition_similarity,
            'token_overlap': self.token_overlap,
            'sampled_action_count': self.sampled_action_count,
            'actions': [item.model_dump() for item in self.actions],
            'notes': list(self.notes),
        }


class LeWorldModelTrajectoryPlanner:
    def __init__(self, *, adapter: LeWorldModelAdapter | None = None) -> None:
        self.adapter = adapter or LeWorldModelAdapter()

    def plan(
        self,
        artifact: LeWorldModelArtifact | None,
        *,
        domain: str,
        initial_steps: Sequence[Sequence[str] | str],
        candidate_actions: Sequence[str],
        goal_tokens: Sequence[str],
        horizon: int = 4,
        samples: int = 32,
        elites: int = 6,
        iterations: int = 5,
        seed: int = 0,
    ) -> LeWorldModelPlan:
        if artifact is None or not candidate_actions:
            return LeWorldModelPlan(
                enabled=False,
                horizon=max(1, int(horizon)),
                iterations=max(1, int(iterations)),
                sample_count=max(1, int(samples)),
                elite_count=max(1, int(elites)),
                notes=['LeWM trajectory planning skipped because the latent prior or candidate actions were unavailable.'],
            )
        action_pool = [str(item) for item in candidate_actions if str(item).strip()]
        if not action_pool:
            return LeWorldModelPlan(
                enabled=False,
                horizon=max(1, int(horizon)),
                iterations=max(1, int(iterations)),
                sample_count=max(1, int(samples)),
                elite_count=max(1, int(elites)),
                notes=['LeWM trajectory planning skipped because the action pool was empty after normalization.'],
            )
        horizon = max(1, int(horizon))
        samples = max(8, int(samples))
        elites = max(2, min(int(elites), samples))
        iterations = max(2, int(iterations))
        rng = random.Random(seed)
        probabilities: list[dict[str, float]] = [
            {token: 1.0 / float(len(action_pool)) for token in action_pool}
            for _ in range(horizon)
        ]
        best_tokens: list[str] = []
        best_alignment = None
        best_score = -1.0
        base_steps = self._normalize_steps(initial_steps)
        goal_step = [str(item) for item in goal_tokens if str(item).strip()]
        for iteration in range(iterations):
            scored_sequences: list[tuple[float, list[str], Any]] = []
            for _ in range(samples):
                sequence = [self._sample_token(probabilities[position], rng) for position in range(horizon)]
                rollout_steps = list(base_steps)
                for token in sequence:
                    rollout_steps.append(self.adapter.tokenize(token))
                if goal_step:
                    rollout_steps.append(goal_step)
                alignment = self.adapter.score_sequence(artifact, rollout_steps)
                diversity = len(set(sequence)) / float(len(sequence) or 1)
                score = (
                    0.45 * float(alignment.alignment_score)
                    + 0.30 * float(alignment.predicted_goal_similarity)
                    + 0.15 * float(alignment.transition_similarity)
                    + 0.05 * float(alignment.token_overlap)
                    + 0.05 * diversity
                )
                scored_sequences.append((score, sequence, alignment))
                if score > best_score:
                    best_score = score
                    best_tokens = list(sequence)
                    best_alignment = alignment
            scored_sequences.sort(key=lambda item: item[0], reverse=True)
            elite_sequences = scored_sequences[:elites]
            for position in range(horizon):
                counts = {token: 1.0 for token in action_pool}
                for _score, sequence, _alignment in elite_sequences:
                    counts[sequence[position]] += 1.0
                total = sum(counts.values()) or 1.0
                probabilities[position] = {token: value / total for token, value in counts.items()}
        alignment = best_alignment or self.adapter.score_sequence(artifact, base_steps)
        actions = [
            LeWorldModelPlannedAction(
                position=index,
                token=token,
                probability=round(float(probabilities[index].get(token, 0.0)), 4),
            )
            for index, token in enumerate(best_tokens)
        ]
        notes = [
            'LeWM-inspired CEM planner sampled latent action trajectories and kept elite sequences with the lowest latent cost.',
            f'domain={domain}',
        ]
        notes.extend(list(alignment.notes[:3]))
        return LeWorldModelPlan(
            enabled=True,
            horizon=horizon,
            iterations=iterations,
            sample_count=samples,
            elite_count=elites,
            plan_tokens=best_tokens,
            plan_score=round(max(0.0, best_score), 4),
            predicted_goal_similarity=round(float(alignment.predicted_goal_similarity), 4),
            transition_similarity=round(float(alignment.transition_similarity), 4),
            token_overlap=round(float(alignment.token_overlap), 4),
            sampled_action_count=samples * iterations,
            actions=actions,
            notes=notes,
        )

    def _normalize_steps(self, steps: Sequence[Sequence[str] | str]) -> list[list[str]]:
        normalized: list[list[str]] = []
        for step in steps:
            if isinstance(step, str):
                tokens = self.adapter.tokenize(step)
            else:
                tokens = [token for raw in step for token in self.adapter.tokenize(str(raw))]
            if tokens:
                normalized.append(tokens)
        return normalized

    @staticmethod
    def _sample_token(distribution: dict[str, float], rng: random.Random) -> str:
        threshold = rng.random()
        cumulative = 0.0
        last_token = next(iter(distribution.keys()))
        for token, probability in distribution.items():
            cumulative += max(0.0, float(probability))
            last_token = token
            if threshold <= cumulative:
                return token
        return last_token

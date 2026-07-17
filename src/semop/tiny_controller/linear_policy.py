from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Sequence

from semop.kernel.engine import ActionPolicy, PolicyDecision
from semop.kernel.model import Goal, GroundAction, WorldState
from semop.kernel.self_learning import PolicyCandidate
from semop.kernel.traces import DecisionTrainingCase

from .features import ACTION_STRUCTURAL_FEATURE_COUNT, CanonicalAction, canonicalize_problem


_DENSE_FEATURES = (
    "dense:effect_goal_exact",
    "dense:effect_goal_predicate",
    "dense:effect_goal_types",
    "dense:effect_goal_argument_overlap",
    "dense:novel_effect_ratio",
    "dense:binding_count",
    "dense:precondition_count",
    "dense:effect_count",
)
_SAFE_OPERATOR_PREFIXES = (
    "family:",
    "precondition:",
    "effect:",
    "structure:",
    "tag:",
)


@dataclass(frozen=True)
class StructuralLinearPolicy:
    """Tiny sparse policy learned only from replay-verified action choices."""

    weights: tuple[tuple[str, float], ...] = ()
    action_limit_score_margin: float = 0.5

    FORMAT_VERSION = 1
    KIND = "structural-linear-v1"

    def __post_init__(self) -> None:
        if self.action_limit_score_margin <= 0:
            raise ValueError("action-limit score margin must be positive")
        names = [name for name, _value in self.weights]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("structural policy weights must be unique and sorted")
        if any(not name.strip() for name in names):
            raise ValueError("structural policy feature names must not be empty")
        if any(not math.isfinite(value) for _name, value in self.weights):
            raise ValueError("structural policy weights must be finite")

    @property
    def parameter_count(self) -> int:
        return len(self.weights)

    def score_actions(
        self,
        state: WorldState,
        goals: Sequence[Goal],
        actions: Sequence[GroundAction],
    ) -> PolicyDecision:
        graph = canonicalize_problem(state, goals, actions)
        weights = dict(self.weights)
        scores = tuple(
            sum(
                weights.get(name, 0.0) * value
                for name, value in _action_features(action).items()
            )
            for action in graph.actions
        )
        ordered = sorted(scores, reverse=True)
        clear_preference = (
            len(ordered) <= 1
            or ordered[0] - ordered[1] >= self.action_limit_score_margin
        )
        return PolicyDecision(
            action_scores=scores,
            halt_probability=0.0,
            state_value=0.0,
            action_limit=1 if clear_preference else None,
        )

    def to_artifact(self) -> bytes:
        return json.dumps(
            {
                "format_version": self.FORMAT_VERSION,
                "kind": self.KIND,
                "action_limit_score_margin": self.action_limit_score_margin,
                "weights": self.weights,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_artifact(cls, artifact: bytes) -> "StructuralLinearPolicy":
        try:
            payload = json.loads(artifact.decode("utf-8"))
            if payload.get("format_version") != cls.FORMAT_VERSION:
                raise ValueError("unsupported structural policy format")
            if payload.get("kind") != cls.KIND:
                raise ValueError("structural policy kind mismatch")
            return cls(
                weights=tuple(
                    (str(name), float(value))
                    for name, value in payload["weights"]
                ),
                action_limit_score_margin=float(
                    payload["action_limit_score_margin"]
                ),
            )
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid structural policy artifact") from exc


@dataclass(frozen=True)
class StructuralPolicyLearner:
    """Deterministic pairwise-margin learner for the CPU fallback brain."""

    epochs: int = 8
    learning_rate: float = 0.25
    ranking_margin: float = 1.0
    action_limit_score_margin: float = 0.5
    max_features: int = 4_096
    weight_clip: float = 16.0
    min_absolute_weight: float = 1e-9

    name = StructuralLinearPolicy.KIND

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.max_features <= 0:
            raise ValueError("structural learner limits must be positive")
        numeric = (
            self.learning_rate,
            self.ranking_margin,
            self.action_limit_score_margin,
            self.weight_clip,
        )
        if any(value <= 0 or not math.isfinite(value) for value in numeric):
            raise ValueError("structural learner numeric settings must be finite and positive")
        if self.min_absolute_weight < 0:
            raise ValueError("minimum absolute weight must not be negative")

    def train(
        self,
        cases: Sequence[DecisionTrainingCase],
        incumbent: ActionPolicy | None = None,
    ) -> PolicyCandidate:
        if not cases:
            raise ValueError("structural policy training requires decision cases")
        if isinstance(incumbent, StructuralLinearPolicy):
            weights = dict(incumbent.weights)
        else:
            weights = {}

        encoded: list[tuple[dict[str, float], tuple[dict[str, float], ...]]] = []
        for case in cases:
            if not 0 <= case.target_action < len(case.actions):
                raise ValueError("decision case target is out of range")
            graph = canonicalize_problem(case.state, case.goals, case.actions)
            action_features = tuple(_action_features(action) for action in graph.actions)
            gold = action_features[case.target_action]
            negatives = tuple(
                features
                for index, features in enumerate(action_features)
                if index != case.target_action
            )
            encoded.append((gold, negatives))

        updates = 0
        for _epoch in range(self.epochs):
            epoch_updates = 0
            for gold, negatives in encoded:
                for negative in negatives:
                    if (
                        _score(weights, gold)
                        >= _score(weights, negative) + self.ranking_margin
                    ):
                        continue
                    difference = _subtract(gold, negative)
                    for name, value in difference.items():
                        updated = weights.get(name, 0.0) + self.learning_rate * value
                        weights[name] = max(-self.weight_clip, min(self.weight_clip, updated))
                    updates += 1
                    epoch_updates += 1
            weights = _prune_weights(
                weights,
                max_features=self.max_features,
                minimum=self.min_absolute_weight,
            )
            if epoch_updates == 0:
                break

        policy = StructuralLinearPolicy(
            weights=tuple(sorted(weights.items())),
            action_limit_score_margin=self.action_limit_score_margin,
        )
        artifact = policy.to_artifact()
        return PolicyCandidate(
            policy=policy,
            kind=self.name,
            artifact_suffix=".json",
            artifact=artifact,
            parameter_count=policy.parameter_count,
            training_updates=updates,
            diagnostics=(
                f"verified_decision_cases={len(cases)}",
                f"pairwise_updates={updates}",
                f"sparse_parameters={policy.parameter_count}",
            ),
        )

    def restore(self, artifact: bytes) -> StructuralLinearPolicy:
        return StructuralLinearPolicy.from_artifact(artifact)


def _action_features(action: CanonicalAction) -> dict[str, float]:
    if len(action.structural_values) != ACTION_STRUCTURAL_FEATURE_COUNT:
        raise ValueError("canonical action structural feature count changed")
    features = {
        name: float(value)
        for name, value in zip(
            _DENSE_FEATURES,
            action.structural_values,
            strict=True,
        )
    }
    for name in action.operator_features:
        if name.startswith(_SAFE_OPERATOR_PREFIXES):
            features[f"operator:{name}"] = 1.0
    return features


def _score(weights: dict[str, float], features: dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def _subtract(
    left: dict[str, float],
    right: dict[str, float],
) -> dict[str, float]:
    names = left.keys() | right.keys()
    return {
        name: left.get(name, 0.0) - right.get(name, 0.0)
        for name in names
        if left.get(name, 0.0) != right.get(name, 0.0)
    }


def _prune_weights(
    weights: dict[str, float],
    *,
    max_features: int,
    minimum: float,
) -> dict[str, float]:
    retained = [
        (name, value)
        for name, value in weights.items()
        if abs(value) > minimum
    ]
    retained.sort(key=lambda item: (-abs(item[1]), item[0]))
    return dict(retained[:max_features])

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math

from semop.kernel.grounding import (
    GroundingAuthority,
    GroundingCandidate,
    GroundingDisposition,
    GroundingRecord,
    GroundingTrace,
    stage_grounding_proposal,
)

from .grounding_features import (
    DEFAULT_SURFACE_BUCKETS,
    GROUNDING_FEATURE_VERSION,
    GroundingFeatureVector,
    encode_grounding_candidate,
    grounding_feature_support,
)


class GroundingPolicyOutcome(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class GroundingPrediction:
    candidate_id: str
    candidate_digest: str
    feature_digest: str
    outcome: GroundingPolicyOutcome
    accept_probability: float
    confidence: float
    feature_support: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", GroundingPolicyOutcome(self.outcome))
        if not 0.0 <= self.accept_probability <= 1.0:
            raise ValueError("grounding accept probability must be between zero and one")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("grounding prediction confidence must be between zero and one")
        if self.feature_support < 0:
            raise ValueError("grounding feature support cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_digest": self.candidate_digest,
            "feature_digest": self.feature_digest,
            "outcome": self.outcome.value,
            "accept_probability": self.accept_probability,
            "confidence": self.confidence,
            "feature_support": self.feature_support,
        }


@dataclass(frozen=True)
class GroundingPolicyReport:
    predictions: tuple[GroundingPrediction, ...]

    @property
    def accepted(self) -> tuple[GroundingPrediction, ...]:
        return tuple(
            item for item in self.predictions if item.outcome is GroundingPolicyOutcome.ACCEPT
        )

    @property
    def rejected(self) -> tuple[GroundingPrediction, ...]:
        return tuple(
            item for item in self.predictions if item.outcome is GroundingPolicyOutcome.REJECT
        )

    @property
    def abstained(self) -> tuple[GroundingPrediction, ...]:
        return tuple(
            item for item in self.predictions if item.outcome is GroundingPolicyOutcome.ABSTAIN
        )


@dataclass(frozen=True)
class GroundingReviewItem:
    candidate: GroundingCandidate = field(compare=False, repr=False)
    prediction: GroundingPrediction
    priority: int

    def __post_init__(self) -> None:
        if self.priority < 0:
            raise ValueError("grounding review priority cannot be negative")


@dataclass(frozen=True)
class SparseGroundingPolicy:
    """Dependency-free selective grounding head for ordinary CPUs.

    ``ACCEPT`` means "stage this as a proposal". This policy has no API that can
    materialize an observed fact; an independent verifier or human remains the
    only promotion path.
    """

    weights: tuple[tuple[str, float], ...] = ()
    feature_support: tuple[tuple[str, int], ...] = ()
    accept_threshold: float = 0.75
    min_feature_support: int = 2
    surface_buckets: int = DEFAULT_SURFACE_BUCKETS
    training_examples: int = 0
    positive_examples: int = 0
    negative_examples: int = 0

    FORMAT_VERSION = 1
    KIND = "sparse-grounding-selective-v1"

    def __post_init__(self) -> None:
        _validate_named_values(self.weights, "grounding policy weights")
        support_names = [name for name, _count in self.feature_support]
        if support_names != sorted(support_names) or len(support_names) != len(
            set(support_names)
        ):
            raise ValueError("grounding feature support must be unique and sorted")
        if any(count < 0 for _name, count in self.feature_support):
            raise ValueError("grounding feature support cannot be negative")
        if not 0.5 < self.accept_threshold < 1.0:
            raise ValueError("grounding accept threshold must be between 0.5 and 1")
        if self.min_feature_support <= 0 or self.surface_buckets <= 0:
            raise ValueError("grounding policy support and bucket limits must be positive")
        counts = (
            self.training_examples,
            self.positive_examples,
            self.negative_examples,
        )
        if any(value < 0 for value in counts):
            raise ValueError("grounding policy training counts cannot be negative")
        if self.positive_examples + self.negative_examples > self.training_examples:
            raise ValueError("grounding policy class counts exceed training examples")

    @property
    def parameter_count(self) -> int:
        return len(self.weights)

    @property
    def artifact_bytes(self) -> int:
        return len(self.to_artifact())

    def predict(self, candidate: GroundingCandidate) -> GroundingPrediction:
        vector = encode_grounding_candidate(
            candidate,
            surface_buckets=self.surface_buckets,
        )
        values = vector.as_dict()
        score = sum(dict(self.weights).get(name, 0.0) * value for name, value in values.items())
        probability = _sigmoid(score)
        support_map = dict(self.feature_support)
        support = grounding_feature_support(vector.values, support_map)
        if support < self.min_feature_support:
            outcome = GroundingPolicyOutcome.ABSTAIN
        elif probability >= self.accept_threshold:
            outcome = GroundingPolicyOutcome.ACCEPT
        elif probability <= 1.0 - self.accept_threshold:
            outcome = GroundingPolicyOutcome.REJECT
        else:
            outcome = GroundingPolicyOutcome.ABSTAIN
        return GroundingPrediction(
            candidate.candidate_id,
            candidate.candidate_digest,
            vector.digest,
            outcome,
            probability,
            max(probability, 1.0 - probability),
            support,
        )

    def stage_if_accepted(
        self,
        candidate: GroundingCandidate,
        *,
        policy_id: str = KIND,
    ) -> tuple[GroundingPrediction, GroundingRecord | None]:
        prediction = self.predict(candidate)
        if prediction.outcome is not GroundingPolicyOutcome.ACCEPT:
            return prediction, None
        record = stage_grounding_proposal(
            candidate,
            authority=GroundingAuthority.MODEL_PROPOSAL,
            verifier_id=policy_id,
            rationale=(
                "selective grounding policy recommended a proposal; independent "
                "verification is still required"
            ),
            confidence=prediction.accept_probability,
            evidence_refs=(f"feature:{prediction.feature_digest}",),
            source="sparse_grounding_policy",
        )
        return prediction, record

    def predict_trace(self, trace: GroundingTrace) -> GroundingPolicyReport:
        candidates = (
            record.candidate
            for record in trace.records
            if record.decision.disposition is GroundingDisposition.PROPOSED
        )
        return GroundingPolicyReport(tuple(self.predict(candidate) for candidate in candidates))

    def to_artifact(self) -> bytes:
        return json.dumps(
            {
                "format_version": self.FORMAT_VERSION,
                "feature_version": GROUNDING_FEATURE_VERSION,
                "kind": self.KIND,
                "weights": self.weights,
                "feature_support": self.feature_support,
                "accept_threshold": self.accept_threshold,
                "min_feature_support": self.min_feature_support,
                "surface_buckets": self.surface_buckets,
                "training_examples": self.training_examples,
                "positive_examples": self.positive_examples,
                "negative_examples": self.negative_examples,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_artifact(cls, artifact: bytes) -> "SparseGroundingPolicy":
        try:
            payload = json.loads(artifact.decode("utf-8"))
            if payload.get("format_version") != cls.FORMAT_VERSION:
                raise ValueError("unsupported sparse grounding policy format")
            if payload.get("feature_version") != GROUNDING_FEATURE_VERSION:
                raise ValueError("sparse grounding feature version mismatch")
            if payload.get("kind") != cls.KIND:
                raise ValueError("sparse grounding policy kind mismatch")
            return cls(
                weights=tuple((str(name), float(value)) for name, value in payload["weights"]),
                feature_support=tuple(
                    (str(name), int(value)) for name, value in payload["feature_support"]
                ),
                accept_threshold=float(payload["accept_threshold"]),
                min_feature_support=int(payload["min_feature_support"]),
                surface_buckets=int(payload["surface_buckets"]),
                training_examples=int(payload["training_examples"]),
                positive_examples=int(payload["positive_examples"]),
                negative_examples=int(payload["negative_examples"]),
            )
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid sparse grounding policy artifact") from exc


def _validate_named_values(values: tuple[tuple[str, float], ...], label: str) -> None:
    names = [name for name, _value in values]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError(f"{label} must be unique and sorted")
    if any(not name for name in names):
        raise ValueError(f"{label} names cannot be empty")
    if any(not math.isfinite(value) for _name, value in values):
        raise ValueError(f"{label} must be finite")


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        factor = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + factor)
    factor = math.exp(max(value, -60.0))
    return factor / (1.0 + factor)


def select_grounding_review_candidates(
    trace: GroundingTrace,
    policy: SparseGroundingPolicy,
    *,
    limit: int = 32,
) -> tuple[GroundingReviewItem, ...]:
    """Prioritize unresolved proposals without treating predictions as labels."""

    if limit <= 0:
        raise ValueError("grounding review selection limit must be positive")
    ranked: list[GroundingReviewItem] = []
    for record in trace.records:
        if record.decision.disposition is not GroundingDisposition.PROPOSED:
            continue
        prediction = policy.predict(record.candidate)
        priority = 0 if prediction.outcome is GroundingPolicyOutcome.ABSTAIN else 1
        ranked.append(GroundingReviewItem(record.candidate, prediction, priority))
    ranked.sort(
        key=lambda item: (
            item.priority,
            item.prediction.confidence,
            item.candidate.candidate_id,
        )
    )
    return tuple(ranked[:limit])


__all__ = [
    "GroundingPolicyOutcome",
    "GroundingPolicyReport",
    "GroundingReviewItem",
    "GroundingPrediction",
    "SparseGroundingPolicy",
    "select_grounding_review_candidates",
]

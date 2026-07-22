from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
from uuid import uuid4

from semop.kernel.grounding import (
    GroundingAuthority,
    GroundingBoundaryError,
    GroundingLabel,
    GroundingLearningExample,
    GroundingTrace,
)

from .grounding_features import encode_grounding_candidate, grounding_feature_support
from .grounding_policy import (
    GroundingPolicyOutcome,
    SparseGroundingPolicy,
)


_TRUSTED_LABEL_AUTHORITIES = {
    GroundingAuthority.DETERMINISTIC_ADAPTER,
    GroundingAuthority.EXTERNAL_VERIFIER,
    GroundingAuthority.HUMAN_REVIEW,
}


@dataclass(frozen=True)
class GroundingReplayItem:
    """Portable verifier-labeled sparse example used for online replay."""

    record_digest: str
    candidate_digest: str
    group_digest: str
    input_digest: str
    feature_digest: str
    domain: str
    label: GroundingLabel
    authority: GroundingAuthority
    weight: float
    features: tuple[tuple[str, float], ...] = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        for value, label in (
            (self.record_digest, "record"),
            (self.candidate_digest, "candidate"),
            (self.group_digest, "group"),
            (self.input_digest, "input"),
            (self.feature_digest, "feature"),
        ):
            _validate_digest(value, label)
        object.__setattr__(self, "label", GroundingLabel(self.label))
        object.__setattr__(self, "authority", GroundingAuthority(self.authority))
        if self.authority not in _TRUSTED_LABEL_AUTHORITIES:
            raise GroundingBoundaryError(
                "grounding replay requires an independently verified label"
            )
        if not self.domain.strip():
            raise ValueError("grounding replay domain cannot be empty")
        object.__setattr__(self, "domain", self.domain.strip().lower())
        if not 0.0 < self.weight <= 1.0:
            raise ValueError("grounding replay weight must be in (0, 1]")
        names = [name for name, _value in self.features]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("grounding replay features must be unique and sorted")
        if any(not math.isfinite(value) for _name, value in self.features):
            raise ValueError("grounding replay features must be finite")

    @classmethod
    def from_example(cls, example: GroundingLearningExample) -> "GroundingReplayItem":
        vector = encode_grounding_candidate(example.candidate)
        group_digest = example.candidate.input_digest
        return cls(
            example.record_digest,
            example.candidate.candidate_digest,
            group_digest,
            example.candidate.input_digest,
            vector.digest,
            example.candidate.domain,
            example.label,
            example.authority,
            example.weight,
            vector.values,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "record_digest": self.record_digest,
            "candidate_digest": self.candidate_digest,
            "group_digest": self.group_digest,
            "input_digest": self.input_digest,
            "feature_digest": self.feature_digest,
            "domain": self.domain,
            "label": self.label.value,
            "authority": self.authority.value,
            "weight": self.weight,
            "features": [list(item) for item in self.features],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GroundingReplayItem":
        try:
            raw_features = payload["features"]
            if not isinstance(raw_features, list):
                raise TypeError("features must be a list")
            return cls(
                str(payload["record_digest"]),
                str(payload["candidate_digest"]),
                str(payload["group_digest"]),
                str(payload["input_digest"]),
                str(payload["feature_digest"]),
                str(payload["domain"]),
                GroundingLabel(str(payload["label"])),
                GroundingAuthority(str(payload["authority"])),
                float(payload["weight"]),
                tuple((str(name), float(value)) for name, value in raw_features),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid grounding replay item") from exc


@dataclass(frozen=True)
class GroundingReplayBuffer:
    items: tuple[GroundingReplayItem, ...] = ()

    def __post_init__(self) -> None:
        records = [item.record_digest for item in self.items]
        if len(records) != len(set(records)):
            raise GroundingBoundaryError("grounding replay contains duplicate records")
        labels_by_candidate: dict[str, GroundingLabel] = {}
        for item in self.items:
            existing = labels_by_candidate.setdefault(item.candidate_digest, item.label)
            if existing is not item.label:
                raise GroundingBoundaryError(
                    "grounding replay contains conflicting labels for one candidate"
                )

    @classmethod
    def from_examples(
        cls,
        examples: Iterable[GroundingLearningExample],
    ) -> "GroundingReplayBuffer":
        return cls(tuple(GroundingReplayItem.from_example(item) for item in examples))

    def extend(
        self,
        examples: Iterable[GroundingLearningExample],
    ) -> "GroundingReplayBuffer":
        by_record = {item.record_digest: item for item in self.items}
        for example in examples:
            item = GroundingReplayItem.from_example(example)
            existing = by_record.get(item.record_digest)
            if existing is not None and existing != item:
                raise GroundingBoundaryError(
                    "grounding replay record digest collision"
                )
            by_record[item.record_digest] = item
        return GroundingReplayBuffer(tuple(sorted(by_record.values(), key=_replay_sort_key)))

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(sorted({item.domain for item in self.items}))

    @property
    def label_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(item.label.value for item in self.items).items()))

    def to_jsonl(self) -> bytes:
        lines = (
            json.dumps(item.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            for item in sorted(self.items, key=_replay_sort_key)
        )
        payload = "\n".join(lines)
        return (payload + ("\n" if payload else "")).encode("utf-8")

    @classmethod
    def from_jsonl(cls, payload: bytes) -> "GroundingReplayBuffer":
        try:
            items = tuple(
                GroundingReplayItem.from_dict(json.loads(line))
                for line in payload.decode("utf-8").splitlines()
                if line.strip()
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid grounding replay JSONL") from exc
        return cls(items)


@dataclass(frozen=True)
class DomainGroundingMetrics:
    domain: str
    examples: int
    decided: int
    correct: int
    false_accepts: int
    false_rejects: int
    abstained: int
    coverage: float
    selective_accuracy: float
    brier_score: float
    aurc: float


@dataclass(frozen=True)
class GroundingRiskCoveragePoint:
    coverage: float
    risk: float
    confidence_threshold: float


@dataclass(frozen=True)
class GroundingPolicyMetrics:
    examples: int
    decided: int
    correct: int
    false_accepts: int
    false_rejects: int
    abstained: int
    coverage: float
    selective_accuracy: float
    brier_score: float
    aurc: float
    by_domain: tuple[DomainGroundingMetrics, ...]

    def domain(self, name: str) -> DomainGroundingMetrics | None:
        normalized = name.strip().lower()
        return next((item for item in self.by_domain if item.domain == normalized), None)


@dataclass(frozen=True)
class GroundingPolicyCandidate:
    policy: SparseGroundingPolicy = field(compare=False, repr=False)
    artifact: bytes = field(compare=False, repr=False)
    training_updates: int
    training_examples: int
    diagnostics: tuple[str, ...] = ()

    @property
    def parameter_count(self) -> int:
        return self.policy.parameter_count

    @property
    def artifact_bytes(self) -> int:
        return len(self.artifact)

    @property
    def artifact_sha256(self) -> str:
        return sha256(self.artifact).hexdigest()


@dataclass(frozen=True)
class SparseGroundingLearner:
    epochs: int = 24
    learning_rate: float = 0.2
    l2: float = 1e-5
    max_parameters: int = 16_384
    weight_clip: float = 16.0
    minimum_weight: float = 1e-9
    accept_threshold: float = 0.75
    min_feature_support: int = 2

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.max_parameters <= 0 or self.min_feature_support <= 0:
            raise ValueError("grounding learner integer limits must be positive")
        if self.learning_rate <= 0 or self.l2 < 0 or self.weight_clip <= 0:
            raise ValueError("grounding learner numeric settings are invalid")
        if not 0.5 < self.accept_threshold < 1.0:
            raise ValueError("grounding learner threshold must be between 0.5 and 1")

    def train(
        self,
        buffer: GroundingReplayBuffer,
        incumbent: SparseGroundingPolicy | None = None,
    ) -> GroundingPolicyCandidate:
        if not buffer.items:
            raise ValueError("grounding policy training requires examples")
        counts = Counter(item.label for item in buffer.items)
        if not counts[GroundingLabel.ACCEPT] or not counts[GroundingLabel.REJECT]:
            raise ValueError("grounding policy training requires both labels")

        weights = dict(incumbent.weights) if incumbent is not None else {}
        support: dict[str, int] = {}
        for item in buffer.items:
            for name, value in item.features:
                if value:
                    support[name] = support.get(name, 0) + 1

        total = len(buffer.items)
        class_scale = {
            GroundingLabel.ACCEPT: total / (2.0 * counts[GroundingLabel.ACCEPT]),
            GroundingLabel.REJECT: total / (2.0 * counts[GroundingLabel.REJECT]),
        }
        updates = 0
        ordered = tuple(sorted(buffer.items, key=_replay_sort_key))
        for _epoch in range(self.epochs):
            epoch_delta = 0.0
            for item in ordered:
                values = dict(item.features)
                score = sum(weights.get(name, 0.0) * value for name, value in values.items())
                probability = _sigmoid(score)
                target = 1.0 if item.label is GroundingLabel.ACCEPT else 0.0
                error = (
                    class_scale[item.label]
                    * item.weight
                    * (target - probability)
                )
                for name, value in values.items():
                    current = weights.get(name, 0.0)
                    delta = self.learning_rate * (error * value - self.l2 * current)
                    if not delta:
                        continue
                    weights[name] = max(
                        -self.weight_clip,
                        min(self.weight_clip, current + delta),
                    )
                    epoch_delta += abs(delta)
                updates += 1
            weights = _prune_weights(
                weights,
                maximum=self.max_parameters,
                minimum=self.minimum_weight,
            )
            if epoch_delta < 1e-10:
                break

        policy = SparseGroundingPolicy(
            weights=tuple(sorted(weights.items())),
            feature_support=tuple(sorted(support.items())),
            accept_threshold=self.accept_threshold,
            min_feature_support=self.min_feature_support,
            training_examples=total,
            positive_examples=counts[GroundingLabel.ACCEPT],
            negative_examples=counts[GroundingLabel.REJECT],
        )
        artifact = policy.to_artifact()
        return GroundingPolicyCandidate(
            policy,
            artifact,
            updates,
            total,
            (
                f"verified_examples={total}",
                f"accept_examples={counts[GroundingLabel.ACCEPT]}",
                f"reject_examples={counts[GroundingLabel.REJECT]}",
                f"sparse_parameters={policy.parameter_count}",
            ),
        )


@dataclass(frozen=True)
class GroundingLearningBudget:
    max_training_examples: int = 5_000
    max_validation_examples: int = 2_000
    min_training_examples: int = 6
    min_validation_examples: int = 6
    min_examples_per_label: int = 2
    required_selective_accuracy: float = 0.95
    min_validation_coverage: float = 0.2
    min_domain_coverage: float = 0.0
    max_false_accepts: int = 0
    max_accuracy_drop: float = 0.0
    max_coverage_drop: float = 0.0
    max_parameters: int = 15_000_000
    max_artifact_bytes: int = 64 * 1024 * 1024
    required_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        integer_limits = (
            self.max_training_examples,
            self.max_validation_examples,
            self.min_training_examples,
            self.min_validation_examples,
            self.min_examples_per_label,
            self.max_parameters,
            self.max_artifact_bytes,
        )
        if any(value <= 0 for value in integer_limits):
            raise ValueError("grounding learning integer limits must be positive")
        if self.max_false_accepts < 0:
            raise ValueError("grounding false-accept limit cannot be negative")
        probabilities = (
            self.required_selective_accuracy,
            self.min_validation_coverage,
            self.min_domain_coverage,
            self.max_accuracy_drop,
            self.max_coverage_drop,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("grounding learning rates must be between zero and one")
        domains = tuple(
            dict.fromkeys(domain.strip().lower() for domain in self.required_domains if domain.strip())
        )
        if len(domains) != len(self.required_domains):
            raise ValueError("required grounding domains must be non-empty and unique")
        object.__setattr__(self, "required_domains", domains)


@dataclass(frozen=True)
class GroundingSplitAudit:
    training_examples: int
    validation_examples: int
    overlapping_records: int
    overlapping_candidates: int
    overlapping_input_groups: int

    @property
    def valid(self) -> bool:
        return not (
            self.overlapping_records
            or self.overlapping_candidates
            or self.overlapping_input_groups
        )


@dataclass(frozen=True)
class GroundingLearningResult:
    active_policy: SparseGroundingPolicy = field(compare=False, repr=False)
    candidate: GroundingPolicyCandidate | None = field(compare=False, repr=False)
    promoted: bool
    rejection_reasons: tuple[str, ...]
    split_audit: GroundingSplitAudit
    baseline_metrics: GroundingPolicyMetrics
    candidate_metrics: GroundingPolicyMetrics | None
    checkpoint: Path | None = field(default=None, compare=False)


@dataclass(frozen=True)
class GroundingOnlineState:
    generation: int
    active_policy: SparseGroundingPolicy = field(compare=False, repr=False)
    replay: GroundingReplayBuffer = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("grounding online generation cannot be negative")

    @classmethod
    def empty(cls) -> "GroundingOnlineState":
        return cls(0, SparseGroundingPolicy(), GroundingReplayBuffer())

    @classmethod
    def from_store(
        cls,
        store: "GroundingPolicyStore",
        *,
        generation: int,
    ) -> "GroundingOnlineState":
        return cls(generation, store.load_policy(), store.load_training())


@dataclass(frozen=True)
class GroundingOnlineLearningResult:
    state: GroundingOnlineState = field(compare=False, repr=False)
    learning: GroundingLearningResult
    received_examples: int
    new_examples: int
    replay_examples: int


class VerifiedGroundingOnlineLearningLoop:
    """Continual update loop with replay and rollback on validation failure."""

    def __init__(
        self,
        *,
        learner: SparseGroundingLearner | None = None,
        budget: GroundingLearningBudget | None = None,
        store: "GroundingPolicyStore | None" = None,
    ) -> None:
        self.loop = VerifiedGroundingLearningLoop(
            learner=learner,
            budget=budget,
            store=store,
        )

    def update(
        self,
        new_examples: Sequence[GroundingLearningExample] | GroundingReplayBuffer,
        validation: Sequence[GroundingLearningExample] | GroundingReplayBuffer,
        *,
        state: GroundingOnlineState | None = None,
    ) -> GroundingOnlineLearningResult:
        current = state or GroundingOnlineState.empty()
        incoming = _as_buffer(new_examples)
        existing_records = {item.record_digest for item in current.replay.items}
        new_count = sum(
            item.record_digest not in existing_records for item in incoming.items
        )
        if not new_count:
            raise ValueError(
                "online grounding update requires a new independently verified label"
            )
        merged = current.replay
        by_record = {item.record_digest: item for item in merged.items}
        for item in incoming.items:
            previous = by_record.get(item.record_digest)
            if previous is not None and previous != item:
                raise GroundingBoundaryError(
                    "online grounding replay record digest collision"
                )
            by_record[item.record_digest] = item
        merged = GroundingReplayBuffer(
            tuple(sorted(by_record.values(), key=_replay_sort_key))
        )
        learning = self.loop.run(
            merged,
            validation,
            incumbent=current.active_policy,
        )
        next_state = GroundingOnlineState(
            current.generation + int(learning.promoted),
            learning.active_policy,
            merged,
        )
        return GroundingOnlineLearningResult(
            next_state,
            learning,
            len(incoming.items),
            new_count,
            len(merged.items),
        )


class VerifiedGroundingLearningLoop:
    """Train and promote a grounder using only independent labels."""

    def __init__(
        self,
        *,
        learner: SparseGroundingLearner | None = None,
        budget: GroundingLearningBudget | None = None,
        store: "GroundingPolicyStore | None" = None,
    ) -> None:
        self.learner = learner or SparseGroundingLearner()
        self.budget = budget or GroundingLearningBudget()
        self.store = store

    def run(
        self,
        training: Sequence[GroundingLearningExample] | GroundingReplayBuffer,
        validation: Sequence[GroundingLearningExample] | GroundingReplayBuffer,
        *,
        incumbent: SparseGroundingPolicy | None = None,
    ) -> GroundingLearningResult:
        training_buffer = _as_buffer(training)
        validation_buffer = _as_buffer(validation)
        split_audit = audit_grounding_split(training_buffer, validation_buffer)
        self._validate_data(training_buffer, validation_buffer, split_audit)

        baseline = incumbent or SparseGroundingPolicy()
        baseline_metrics = evaluate_grounding_policy(baseline, validation_buffer)
        reasons: list[str] = []
        candidate: GroundingPolicyCandidate | None = None
        candidate_metrics: GroundingPolicyMetrics | None = None
        try:
            candidate = self.learner.train(training_buffer, incumbent)
            candidate_metrics = evaluate_grounding_policy(
                candidate.policy,
                validation_buffer,
            )
            reasons.extend(
                self._promotion_rejections(
                    baseline_metrics,
                    candidate_metrics,
                    candidate,
                )
            )
        except Exception as exc:
            reasons.append(
                f"grounding_training_or_evaluation_failed:{type(exc).__name__}:{exc}"
            )

        promoted = candidate is not None and not reasons
        active = candidate.policy if promoted and candidate is not None else baseline
        result = GroundingLearningResult(
            active,
            candidate,
            promoted,
            tuple(reasons),
            split_audit,
            baseline_metrics,
            candidate_metrics,
        )
        if self.store is not None:
            checkpoint = self.store.persist(
                active,
                training_buffer,
                validation_buffer,
                result,
            )
            result = GroundingLearningResult(
                result.active_policy,
                result.candidate,
                result.promoted,
                result.rejection_reasons,
                result.split_audit,
                result.baseline_metrics,
                result.candidate_metrics,
                checkpoint,
            )
        return result

    def run_from_traces(
        self,
        training: Sequence[GroundingTrace],
        validation: Sequence[GroundingTrace],
        *,
        incumbent: SparseGroundingPolicy | None = None,
    ) -> GroundingLearningResult:
        return self.run(
            collect_grounding_examples(training),
            collect_grounding_examples(validation),
            incumbent=incumbent,
        )

    def _validate_data(
        self,
        training: GroundingReplayBuffer,
        validation: GroundingReplayBuffer,
        audit: GroundingSplitAudit,
    ) -> None:
        if not training.items or not validation.items:
            raise ValueError("grounding learning requires train and validation examples")
        if len(training.items) > self.budget.max_training_examples:
            raise ValueError("grounding training example limit exceeded")
        if len(validation.items) > self.budget.max_validation_examples:
            raise ValueError("grounding validation example limit exceeded")
        if len(training.items) < self.budget.min_training_examples:
            raise ValueError("insufficient grounding training examples")
        if len(validation.items) < self.budget.min_validation_examples:
            raise ValueError("insufficient grounding validation examples")
        if not audit.valid:
            raise GroundingBoundaryError(
                "grounding train and validation splits share raw input lineage"
            )
        for name, buffer in (("training", training), ("validation", validation)):
            counts = Counter(item.label for item in buffer.items)
            for label in GroundingLabel:
                if counts[label] < self.budget.min_examples_per_label:
                    raise ValueError(
                        f"{name} grounding split lacks {label.value} examples"
                    )
        validation_domains = set(validation.domains)
        missing = set(self.budget.required_domains) - validation_domains
        if missing:
            raise ValueError(
                "grounding validation split lacks required domains: "
                + ",".join(sorted(missing))
            )

    def _promotion_rejections(
        self,
        baseline: GroundingPolicyMetrics,
        candidate: GroundingPolicyMetrics,
        artifact: GroundingPolicyCandidate,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if candidate.false_accepts > self.budget.max_false_accepts:
            reasons.append(
                f"false_accepts:{candidate.false_accepts}>{self.budget.max_false_accepts}"
            )
        if candidate.selective_accuracy < self.budget.required_selective_accuracy:
            reasons.append(
                "selective_accuracy_below_gate:"
                f"{candidate.selective_accuracy:.6f}<"
                f"{self.budget.required_selective_accuracy:.6f}"
            )
        if candidate.coverage < self.budget.min_validation_coverage:
            reasons.append(
                f"coverage_below_gate:{candidate.coverage:.6f}<"
                f"{self.budget.min_validation_coverage:.6f}"
            )
        if baseline.decided and (
            candidate.selective_accuracy + self.budget.max_accuracy_drop
            < baseline.selective_accuracy
        ):
            reasons.append("selective_accuracy_regressed")
        if (
            candidate.coverage + self.budget.max_coverage_drop
            < baseline.coverage
        ):
            reasons.append("coverage_regressed")
        for domain in self.budget.required_domains:
            metrics = candidate.domain(domain)
            if metrics is None or metrics.coverage < self.budget.min_domain_coverage:
                reasons.append(f"domain_coverage_below_gate:{domain}")
        if artifact.parameter_count > self.budget.max_parameters:
            reasons.append("grounding_parameter_limit_exceeded")
        if artifact.artifact_bytes > self.budget.max_artifact_bytes:
            reasons.append("grounding_artifact_limit_exceeded")
        return tuple(reasons)


class GroundingPolicyStore:
    """Atomic policy and replay checkpoint with digest verification."""

    FORMAT_VERSION = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def persist(
        self,
        policy: SparseGroundingPolicy,
        training: GroundingReplayBuffer,
        validation: GroundingReplayBuffer,
        result: GroundingLearningResult,
    ) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        policy_payload = policy.to_artifact()
        training_payload = training.to_jsonl()
        validation_payload = validation.to_jsonl()
        _atomic_write(self.root / "policy.json", policy_payload)
        _atomic_write(self.root / "training.jsonl", training_payload)
        _atomic_write(self.root / "validation.jsonl", validation_payload)
        manifest = {
            "format_version": self.FORMAT_VERSION,
            "policy_kind": SparseGroundingPolicy.KIND,
            "policy_sha256": sha256(policy_payload).hexdigest(),
            "policy_parameters": policy.parameter_count,
            "training_sha256": sha256(training_payload).hexdigest(),
            "validation_sha256": sha256(validation_payload).hexdigest(),
            "training_examples": len(training.items),
            "validation_examples": len(validation.items),
            "promoted": result.promoted,
            "rejection_reasons": list(result.rejection_reasons),
            "split_audit": asdict(result.split_audit),
            "baseline_metrics": _metrics_dict(result.baseline_metrics),
            "candidate_metrics": (
                None
                if result.candidate_metrics is None
                else _metrics_dict(result.candidate_metrics)
            ),
        }
        _atomic_write(
            self.manifest_path,
            json.dumps(
                manifest,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
        )
        return self.manifest_path.resolve()

    def load_policy(self) -> SparseGroundingPolicy:
        manifest = self._load_manifest()
        payload = self._verified_payload("policy.json", str(manifest["policy_sha256"]))
        return SparseGroundingPolicy.from_artifact(payload)

    def load_training(self) -> GroundingReplayBuffer:
        manifest = self._load_manifest()
        payload = self._verified_payload(
            "training.jsonl",
            str(manifest["training_sha256"]),
        )
        return GroundingReplayBuffer.from_jsonl(payload)

    def load_validation(self) -> GroundingReplayBuffer:
        manifest = self._load_manifest()
        payload = self._verified_payload(
            "validation.jsonl",
            str(manifest["validation_sha256"]),
        )
        return GroundingReplayBuffer.from_jsonl(payload)

    def _load_manifest(self) -> dict[str, object]:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid grounding policy manifest") from exc
        if payload.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("unsupported grounding policy checkpoint format")
        if payload.get("policy_kind") != SparseGroundingPolicy.KIND:
            raise ValueError("grounding policy checkpoint kind mismatch")
        return payload

    def _verified_payload(self, relative: str, expected_digest: str) -> bytes:
        payload = (self.root / relative).read_bytes()
        if sha256(payload).hexdigest() != expected_digest:
            raise ValueError(f"grounding checkpoint artifact digest mismatch: {relative}")
        return payload


def collect_grounding_examples(
    traces: Iterable[GroundingTrace],
) -> tuple[GroundingLearningExample, ...]:
    by_record: dict[str, GroundingLearningExample] = {}
    for trace in traces:
        for example in trace.learning_examples:
            existing = by_record.get(example.record_digest)
            if existing is not None and existing != example:
                raise GroundingBoundaryError("grounding learning record digest collision")
            by_record[example.record_digest] = example
    return tuple(sorted(by_record.values(), key=lambda item: item.record_digest))


def audit_grounding_split(
    training: GroundingReplayBuffer,
    validation: GroundingReplayBuffer,
) -> GroundingSplitAudit:
    def values(attribute: str, buffer: GroundingReplayBuffer) -> set[str]:
        return {str(getattr(item, attribute)) for item in buffer.items}

    return GroundingSplitAudit(
        len(training.items),
        len(validation.items),
        len(values("record_digest", training) & values("record_digest", validation)),
        len(values("candidate_digest", training) & values("candidate_digest", validation)),
        len(values("group_digest", training) & values("group_digest", validation)),
    )


def evaluate_grounding_policy(
    policy: SparseGroundingPolicy,
    examples: Sequence[GroundingLearningExample] | GroundingReplayBuffer,
) -> GroundingPolicyMetrics:
    buffer = _as_buffer(examples)
    grouped: dict[str, list[tuple[GroundingReplayItem, GroundingPolicyOutcome, float]]] = {}
    for item in buffer.items:
        probability, support = _predict_encoded(policy, item)
        if support < policy.min_feature_support:
            outcome = GroundingPolicyOutcome.ABSTAIN
        elif probability >= policy.accept_threshold:
            outcome = GroundingPolicyOutcome.ACCEPT
        elif probability <= 1.0 - policy.accept_threshold:
            outcome = GroundingPolicyOutcome.REJECT
        else:
            outcome = GroundingPolicyOutcome.ABSTAIN
        grouped.setdefault(item.domain, []).append((item, outcome, probability))

    per_domain = tuple(
        _summarize_predictions(domain, predictions)
        for domain, predictions in sorted(grouped.items())
    )
    all_predictions = tuple(
        prediction
        for domain in sorted(grouped)
        for prediction in grouped[domain]
    )
    return _combine_metrics(per_domain, all_predictions)


def grounding_risk_coverage_curve(
    policy: SparseGroundingPolicy,
    examples: Sequence[GroundingLearningExample] | GroundingReplayBuffer,
) -> tuple[GroundingRiskCoveragePoint, ...]:
    """Return empirical selective risk as confidence coverage expands."""

    buffer = _as_buffer(examples)
    predictions = []
    for item in buffer.items:
        probability, _support = _predict_encoded(policy, item)
        predictions.append((item, GroundingPolicyOutcome.ABSTAIN, probability))
    return _risk_coverage_points(tuple(predictions))


def _predict_encoded(
    policy: SparseGroundingPolicy,
    item: GroundingReplayItem,
) -> tuple[float, int]:
    weights = dict(policy.weights)
    support = dict(policy.feature_support)
    score = sum(weights.get(name, 0.0) * value for name, value in item.features)
    max_support = grounding_feature_support(item.features, support)
    return _sigmoid(score), max_support


def _summarize_predictions(
    domain: str,
    values: Sequence[tuple[GroundingReplayItem, GroundingPolicyOutcome, float]],
) -> DomainGroundingMetrics:
    decided = correct = false_accepts = false_rejects = abstained = 0
    squared_error = 0.0
    for item, outcome, probability in values:
        target = 1.0 if item.label is GroundingLabel.ACCEPT else 0.0
        squared_error += (probability - target) ** 2
        if outcome is GroundingPolicyOutcome.ABSTAIN:
            abstained += 1
            continue
        decided += 1
        predicted = (
            GroundingLabel.ACCEPT
            if outcome is GroundingPolicyOutcome.ACCEPT
            else GroundingLabel.REJECT
        )
        if predicted is item.label:
            correct += 1
        elif predicted is GroundingLabel.ACCEPT:
            false_accepts += 1
        else:
            false_rejects += 1
    examples = len(values)
    curve = _risk_coverage_points(values)
    return DomainGroundingMetrics(
        domain,
        examples,
        decided,
        correct,
        false_accepts,
        false_rejects,
        abstained,
        decided / examples if examples else 0.0,
        correct / decided if decided else 0.0,
        squared_error / examples if examples else 0.0,
        sum(point.risk for point in curve) / len(curve) if curve else 0.0,
    )


def _combine_metrics(
    domains: tuple[DomainGroundingMetrics, ...],
    predictions: Sequence[tuple[GroundingReplayItem, GroundingPolicyOutcome, float]],
) -> GroundingPolicyMetrics:
    examples = sum(item.examples for item in domains)
    decided = sum(item.decided for item in domains)
    correct = sum(item.correct for item in domains)
    false_accepts = sum(item.false_accepts for item in domains)
    false_rejects = sum(item.false_rejects for item in domains)
    abstained = sum(item.abstained for item in domains)
    brier = (
        sum(item.brier_score * item.examples for item in domains) / examples
        if examples
        else 0.0
    )
    curve = _risk_coverage_points(predictions)
    return GroundingPolicyMetrics(
        examples,
        decided,
        correct,
        false_accepts,
        false_rejects,
        abstained,
        decided / examples if examples else 0.0,
        correct / decided if decided else 0.0,
        brier,
        sum(point.risk for point in curve) / len(curve) if curve else 0.0,
        domains,
    )


def _risk_coverage_points(
    values: Sequence[tuple[GroundingReplayItem, GroundingPolicyOutcome, float]],
) -> tuple[GroundingRiskCoveragePoint, ...]:
    ranked = sorted(
        values,
        key=lambda item: (
            -max(item[2], 1.0 - item[2]),
            item[0].record_digest,
        ),
    )
    errors = 0
    points: list[GroundingRiskCoveragePoint] = []
    for index, (item, _outcome, probability) in enumerate(ranked, start=1):
        predicted = (
            GroundingLabel.ACCEPT if probability >= 0.5 else GroundingLabel.REJECT
        )
        errors += int(predicted is not item.label)
        points.append(
            GroundingRiskCoveragePoint(
                index / len(ranked),
                errors / index,
                max(probability, 1.0 - probability),
            )
        )
    return tuple(points)


def _as_buffer(
    values: Sequence[GroundingLearningExample] | GroundingReplayBuffer,
) -> GroundingReplayBuffer:
    if isinstance(values, GroundingReplayBuffer):
        return values
    return GroundingReplayBuffer.from_examples(values)


def _metrics_dict(metrics: GroundingPolicyMetrics) -> dict[str, object]:
    payload = asdict(metrics)
    payload["by_domain"] = [asdict(item) for item in metrics.by_domain]
    return payload


def _replay_sort_key(item: GroundingReplayItem) -> tuple[str, str, str]:
    return item.domain, item.group_digest, item.record_digest


def _prune_weights(
    weights: dict[str, float],
    *,
    maximum: int,
    minimum: float,
) -> dict[str, float]:
    retained = [
        (name, value)
        for name, value in weights.items()
        if abs(value) > minimum
    ]
    retained.sort(key=lambda item: (-abs(item[1]), item[0]))
    return dict(retained[:maximum])


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        factor = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + factor)
    factor = math.exp(max(value, -60.0))
    return factor / (1.0 + factor)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"grounding {label} digest must be SHA-256 hex")


__all__ = [
    "DomainGroundingMetrics",
    "GroundingLearningBudget",
    "GroundingLearningResult",
    "GroundingOnlineLearningResult",
    "GroundingOnlineState",
    "GroundingPolicyCandidate",
    "GroundingPolicyMetrics",
    "GroundingPolicyStore",
    "GroundingReplayBuffer",
    "GroundingReplayItem",
    "GroundingRiskCoveragePoint",
    "GroundingSplitAudit",
    "SparseGroundingLearner",
    "VerifiedGroundingLearningLoop",
    "VerifiedGroundingOnlineLearningLoop",
    "audit_grounding_split",
    "collect_grounding_examples",
    "evaluate_grounding_policy",
    "grounding_risk_coverage_curve",
]

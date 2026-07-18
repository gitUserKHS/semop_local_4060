from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any

from .runtime import DomainKind, TypedDomainRequest
from .experience import RawLearningExample
from .semantic_benchmark import (
    SemanticBenchmarkCase,
    SemanticReviewDecision,
    SemanticReviewRecord,
    create_semantic_review,
)
from .semantic_codec import (
    canonical_json,
    decode_semantic_request,
    encode_semantic_request,
    semantic_request_digest,
)
from .self_learning import (
    LearningSplit,
    SemanticLabelAuthority,
    SemanticLabelEvidence,
)


EXPERIENCE_OBSERVATION_SCHEMA_VERSION = 1
EXPERIENCE_REVIEW_SCHEMA_VERSION = 1
EXPERIENCE_STORE_SCHEMA_VERSION = 1


class ExperienceProposalAuthority(str, Enum):
    UNKNOWN = "unknown"
    USER_PROPOSAL = "user_proposal"
    FRONTIER_JUDGE = "frontier_judge"
    PROGRAMMATIC = "programmatic"


class ExperienceTrigger(str, Enum):
    GROUNDING_FAILURE = "grounding_failure"
    RUNTIME_EXCEPTION = "runtime_exception"
    UNSOLVED = "unsolved"
    REPLAY_FAILURE = "replay_failure"
    EXPECTATION_MISMATCH = "expectation_mismatch"
    POLICY_DISAGREEMENT = "policy_disagreement"
    MANUAL = "manual"
    SUCCESS_SAMPLE = "success_sample"


class ExperienceQueueStatus(str, Enum):
    PENDING = "pending"
    CONFLICTED = "conflicted"
    APPROVED = "approved"
    REJECTED = "rejected"


class ExperienceSplitRole(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    CANDIDATE_HELDOUT = "candidate_heldout"
    JOINT_HELDOUT = "joint_heldout"

    @property
    def learning_split(self) -> LearningSplit:
        if self is ExperienceSplitRole.TRAIN:
            return LearningSplit.TRAIN
        return LearningSplit.HELDOUT


@dataclass(frozen=True)
class ExperiencePartitionConfig:
    """Precommitted hash partition for four-stage rule learning."""

    seed: str = "semop-online-experience-v1"
    train_buckets: int = 50
    validation_buckets: int = 20
    candidate_heldout_buckets: int = 15
    joint_heldout_buckets: int = 15

    def __post_init__(self) -> None:
        if not self.seed.strip():
            raise ValueError("experience partition seed must not be empty")
        buckets = (
            self.train_buckets,
            self.validation_buckets,
            self.candidate_heldout_buckets,
            self.joint_heldout_buckets,
        )
        if any(type(value) is not int for value in buckets):
            raise TypeError("experience partition buckets must be integers")
        if any(value <= 0 for value in buckets):
            raise ValueError("experience partition buckets must be positive")
        if sum(buckets) != 100:
            raise ValueError("experience partition buckets must sum to 100")

    @property
    def fingerprint(self) -> str:
        return sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()

    def role_for(self, request_digest: str) -> ExperienceSplitRole:
        _require_sha256(request_digest, "experience request")
        material = f"{self.seed}:{request_digest.lower()}".encode("utf-8")
        bucket = int(sha256(material).hexdigest()[:16], 16) % 100
        train_end = self.train_buckets
        validation_end = train_end + self.validation_buckets
        candidate_end = validation_end + self.candidate_heldout_buckets
        if bucket < train_end:
            return ExperienceSplitRole.TRAIN
        if bucket < validation_end:
            return ExperienceSplitRole.VALIDATION
        if bucket < candidate_end:
            return ExperienceSplitRole.CANDIDATE_HELDOUT
        return ExperienceSplitRole.JOINT_HELDOUT


@dataclass(frozen=True)
class ExperienceObservation:
    event_id: str
    request_digest: str
    domain: DomainKind
    payload_json: str
    observed_success: bool | None
    observed_verified: bool
    proposed_expected_solved: bool | None
    proposal_authority: ExperienceProposalAuthority
    trigger: ExperienceTrigger
    rationale: str
    source: str
    grounded_fingerprint: str = ""
    proof_program: tuple[str, ...] = ()
    halt_reason: str = ""
    error_type: str = ""
    error_message: str = ""
    expansions: int = 0
    created_at: str = ""
    schema_version: int = EXPERIENCE_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        event_id = self.event_id.strip()
        if not event_id:
            raise ValueError("experience event id must not be empty")
        request_digest = self.request_digest.strip().lower()
        _require_sha256(request_digest, "experience request")
        domain = DomainKind(self.domain)
        if domain not in {
            DomainKind.LANGUAGE,
            DomainKind.MATH,
            DomainKind.VISION,
        }:
            raise ValueError(f"experience queue does not support {domain.value}")
        payload = _parse_json_object(self.payload_json, "experience payload")
        payload_json = canonical_json(payload)
        request = decode_semantic_request(domain, payload)
        if semantic_request_digest(request) != request_digest:
            raise ValueError("experience request digest does not match its payload")
        if (
            self.observed_success is not None
            and type(self.observed_success) is not bool
        ):
            raise TypeError("experience observed_success must be boolean or null")
        if type(self.observed_verified) is not bool:
            raise TypeError("experience observed_verified must be boolean")
        if self.observed_verified and self.observed_success is not True:
            raise ValueError("verified experience observations must be successful")
        proposed = self.proposed_expected_solved
        if proposed is not None and type(proposed) is not bool:
            raise TypeError("experience proposed outcome must be boolean or null")
        authority = ExperienceProposalAuthority(self.proposal_authority)
        if proposed is None and authority is not ExperienceProposalAuthority.UNKNOWN:
            raise ValueError("experience proposal authority requires a proposed label")
        if proposed is not None and authority is ExperienceProposalAuthority.UNKNOWN:
            raise ValueError("experience proposed labels require their authority")
        rationale = self.rationale.strip()
        if proposed is not None and not rationale:
            raise ValueError("experience proposed labels require a rationale")
        source = self.source.strip()
        if not source:
            raise ValueError("experience source must not be empty")
        grounded = self.grounded_fingerprint.strip().lower()
        if grounded:
            _require_sha256(grounded, "experience grounding")
        if self.observed_success is not None and not grounded:
            raise ValueError("executed experiences require a grounding fingerprint")
        if self.observed_success is None and not self.error_type.strip():
            raise ValueError("failed experience observations require an error type")
        if self.observed_success is not None and self.error_type.strip():
            raise ValueError("executed experience observations cannot contain an error")
        if type(self.expansions) is not int:
            raise TypeError("experience expansions must be an integer")
        if self.expansions < 0:
            raise ValueError("experience expansions must not be negative")
        if isinstance(self.proof_program, (str, bytes)) or not isinstance(
            self.proof_program,
            Sequence,
        ):
            raise TypeError("experience proof program must be a list of names")
        if any(not isinstance(name, str) for name in self.proof_program):
            raise TypeError("experience proof operator names must be strings")
        proof_program = tuple(name.strip() for name in self.proof_program)
        if any(not name for name in proof_program):
            raise ValueError("experience proof operator names must not be empty")
        created_at = self.created_at.strip() or _utc_now()
        _parse_timestamp(created_at, "experience observation")
        if self.schema_version != EXPERIENCE_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("unsupported experience observation schema")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "request_digest", request_digest)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "payload_json", payload_json)
        object.__setattr__(self, "proposal_authority", authority)
        object.__setattr__(self, "trigger", ExperienceTrigger(self.trigger))
        object.__setattr__(self, "rationale", rationale)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "grounded_fingerprint", grounded)
        object.__setattr__(self, "proof_program", proof_program)
        object.__setattr__(self, "halt_reason", self.halt_reason.strip())
        object.__setattr__(self, "error_type", self.error_type.strip())
        object.__setattr__(self, "error_message", self.error_message.strip())
        object.__setattr__(self, "created_at", created_at)

    @property
    def payload(self) -> dict[str, Any]:
        return _parse_json_object(self.payload_json, "experience payload")

    @property
    def request(self) -> TypedDomainRequest:
        return decode_semantic_request(self.domain, self.payload)

    @property
    def event_digest(self) -> str:
        return sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "request_digest": self.request_digest,
            "domain": self.domain.value,
            "payload": self.payload,
            "observed_success": self.observed_success,
            "observed_verified": self.observed_verified,
            "proposed_expected_solved": self.proposed_expected_solved,
            "proposal_authority": self.proposal_authority.value,
            "trigger": self.trigger.value,
            "rationale": self.rationale,
            "source": self.source,
            "grounded_fingerprint": self.grounded_fingerprint,
            "proof_program": list(self.proof_program),
            "halt_reason": self.halt_reason,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "expansions": self.expansions,
            "created_at": self.created_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperienceObservation":
        if not isinstance(value, Mapping):
            raise TypeError("experience observation must be an object")
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise TypeError("experience observation payload must be an object")
        proof_program = value.get("proof_program", ())
        if isinstance(proof_program, (str, bytes)) or not isinstance(
            proof_program,
            Sequence,
        ):
            raise TypeError("experience proof program must be a list of names")
        return cls(
            event_id=str(value.get("event_id", "")),
            request_digest=str(value.get("request_digest", "")),
            domain=DomainKind(str(value.get("domain", ""))),
            payload_json=canonical_json(payload),
            observed_success=value.get("observed_success"),
            observed_verified=value.get("observed_verified"),
            proposed_expected_solved=value.get("proposed_expected_solved"),
            proposal_authority=ExperienceProposalAuthority(
                str(value.get("proposal_authority", ""))
            ),
            trigger=ExperienceTrigger(str(value.get("trigger", ""))),
            rationale=str(value.get("rationale", "")),
            source=str(value.get("source", "")),
            grounded_fingerprint=str(value.get("grounded_fingerprint", "")),
            proof_program=tuple(proof_program),
            halt_reason=str(value.get("halt_reason", "")),
            error_type=str(value.get("error_type", "")),
            error_message=str(value.get("error_message", "")),
            expansions=value.get("expansions", 0),
            created_at=str(value.get("created_at", "")),
            schema_version=value.get(
                "schema_version",
                EXPERIENCE_OBSERVATION_SCHEMA_VERSION,
            ),
        )


@dataclass(frozen=True)
class ExperienceReviewRecord:
    request_digest: str
    partition_fingerprint: str
    split_role: ExperienceSplitRole
    case: SemanticBenchmarkCase
    review: SemanticReviewRecord
    schema_version: int = EXPERIENCE_REVIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        request_digest = self.request_digest.strip().lower()
        _require_sha256(request_digest, "experience review request")
        partition_fingerprint = self.partition_fingerprint.strip().lower()
        _require_sha256(partition_fingerprint, "experience review partition")
        role = ExperienceSplitRole(self.split_role)
        if self.case.split is not role.learning_split:
            raise ValueError("experience review case has the wrong learning split")
        if self.review.case_id != self.case.case_id:
            raise ValueError("experience review case id mismatch")
        if self.review.case_digest != self.case.digest:
            raise ValueError("experience review is stale for its case")
        if semantic_request_digest(self.case.to_request()) != request_digest:
            raise ValueError("experience review request digest mismatch")
        if self.schema_version != EXPERIENCE_REVIEW_SCHEMA_VERSION:
            raise ValueError("unsupported experience review schema")
        object.__setattr__(self, "request_digest", request_digest)
        object.__setattr__(
            self,
            "partition_fingerprint",
            partition_fingerprint,
        )
        object.__setattr__(self, "split_role", role)

    @property
    def approved(self) -> bool:
        return self.review.decision is SemanticReviewDecision.APPROVED

    @property
    def review_digest(self) -> str:
        return sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_digest": self.request_digest,
            "partition_fingerprint": self.partition_fingerprint,
            "split_role": self.split_role.value,
            "case": self.case.to_dict(),
            "review": self.review.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperienceReviewRecord":
        if not isinstance(value, Mapping):
            raise TypeError("experience review record must be an object")
        return cls(
            request_digest=str(value.get("request_digest", "")),
            partition_fingerprint=str(
                value.get("partition_fingerprint", "")
            ),
            split_role=ExperienceSplitRole(str(value.get("split_role", ""))),
            case=SemanticBenchmarkCase.from_mapping(value.get("case", {})),
            review=SemanticReviewRecord.from_mapping(value.get("review", {})),
            schema_version=int(
                value.get("schema_version", EXPERIENCE_REVIEW_SCHEMA_VERSION)
            ),
        )

    def raw_example(self) -> RawLearningExample:
        evidence = SemanticLabelEvidence(
            case_digest=self.review.case_digest,
            reviewer=self.review.reviewer,
            reviewed_at=self.review.reviewed_at,
            attestation=self.review.attestation,
        )
        return RawLearningExample(
            example_id=self.case.case_id,
            request=self.case.to_request(),
            expected_solved=self.case.expected_solved,
            split=self.case.split,
            source="reviewed",
            capability=self.case.phenomenon,
            structure_key=f"{self.case.domain.value}:{self.case.phenomenon}",
            difficulty=self.case.difficulty,
            label_authority=SemanticLabelAuthority.HUMAN_REVIEWED,
            label_evidence=evidence,
        )


@dataclass(frozen=True)
class ExperienceQueueItem:
    request_digest: str
    domain: str
    payload_json: str
    occurrences: int
    observed_failures: int
    unverified_results: int
    proposed_positive: int
    proposed_negative: int
    triggers: tuple[str, ...]
    status: ExperienceQueueStatus
    priority: int
    latest_review: ExperienceReviewRecord | None = None

    @property
    def conflicted(self) -> bool:
        return self.proposed_positive > 0 and self.proposed_negative > 0


@dataclass(frozen=True)
class ExperienceCaptureResult:
    observation: ExperienceObservation
    inserted: bool
    item: ExperienceQueueItem


@dataclass(frozen=True)
class ExperienceQueueStats:
    requests: int
    observations: int
    pending: int
    conflicted: int
    approved: int
    rejected: int
    by_domain: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ExperienceReviewCorpus:
    records: tuple[ExperienceReviewRecord, ...]
    partition_fingerprint: str

    def __post_init__(self) -> None:
        partition_fingerprint = self.partition_fingerprint.strip().lower()
        _require_sha256(
            partition_fingerprint,
            "experience corpus partition",
        )
        digests = tuple(record.request_digest for record in self.records)
        if len(digests) != len(set(digests)):
            raise ValueError("experience review corpus request digests must be unique")
        if any(not record.approved for record in self.records):
            raise ValueError("experience review corpus accepts approved reviews only")
        if any(
            record.partition_fingerprint != partition_fingerprint
            for record in self.records
        ):
            raise ValueError(
                "experience review corpus mixes partition contracts"
            )
        object.__setattr__(
            self,
            "partition_fingerprint",
            partition_fingerprint,
        )

    def for_role(
        self,
        role: ExperienceSplitRole | str,
    ) -> tuple[ExperienceReviewRecord, ...]:
        resolved = ExperienceSplitRole(role)
        return tuple(record for record in self.records if record.split_role is resolved)

    def raw_examples(
        self,
        role: ExperienceSplitRole | str | None = None,
    ) -> tuple[RawLearningExample, ...]:
        selected = self.records if role is None else self.for_role(role)
        return tuple(record.raw_example() for record in selected)

    @property
    def role_counts(self) -> tuple[tuple[str, int], ...]:
        counts = Counter(record.split_role.value for record in self.records)
        return tuple(
            (role.value, counts[role.value]) for role in ExperienceSplitRole
        )


class ExperienceStoreIntegrityError(ValueError):
    pass


class TypedExperienceStore:
    """Dependency-free append-audited queue for raw LMV execution experience."""

    def __init__(
        self,
        path: str | Path,
        *,
        partition: ExperiencePartitionConfig | None = None,
        max_requests: int = 10_000,
        max_payload_bytes: int = 4_000_000,
        max_record_bytes: int = 5_000_000,
        max_proof_steps: int = 256,
    ) -> None:
        self.path = Path(path)
        self.partition = partition or ExperiencePartitionConfig()
        self.max_requests = max_requests
        self.max_payload_bytes = max_payload_bytes
        self.max_record_bytes = max_record_bytes
        self.max_proof_steps = max_proof_steps
        if any(
            type(value) is not int
            for value in (
                max_requests,
                max_payload_bytes,
                max_record_bytes,
                max_proof_steps,
            )
        ):
            raise TypeError("typed experience store limits must be integers")
        if any(
            value <= 0
            for value in (
                max_requests,
                max_payload_bytes,
                max_record_bytes,
                max_proof_steps,
            )
        ):
            raise ValueError("typed experience store limits must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @classmethod
    def open_existing(
        cls,
        path: str | Path,
        *,
        max_requests: int = 10_000,
        max_payload_bytes: int = 4_000_000,
        max_record_bytes: int = 5_000_000,
        max_proof_steps: int = 256,
    ) -> "TypedExperienceStore":
        """Open a store using its persisted, digest-checked partition contract."""

        resolved_path = Path(path)
        if not resolved_path.is_file():
            raise FileNotFoundError(resolved_path)
        try:
            with closing(sqlite3.connect(resolved_path)) as connection:
                rows = connection.execute(
                    "SELECT key, value FROM experience_metadata"
                ).fetchall()
        except sqlite3.Error as exc:
            raise ExperienceStoreIntegrityError(
                "typed experience store metadata cannot be read"
            ) from exc
        metadata = {str(key): str(value) for key, value in rows}
        partition_value = metadata.get("partition_json")
        if partition_value is None:
            raise ExperienceStoreIntegrityError(
                "typed experience store partition metadata is missing"
            )
        if len(partition_value.encode("utf-8")) > 16_384:
            raise ExperienceStoreIntegrityError(
                "typed experience store partition metadata is too large"
            )
        try:
            partition_mapping = json.loads(partition_value)
            if not isinstance(partition_mapping, dict):
                raise TypeError("partition metadata must be an object")
            partition = ExperiencePartitionConfig(**partition_mapping)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExperienceStoreIntegrityError(
                "typed experience store partition metadata is invalid"
            ) from exc
        if metadata.get("partition_fingerprint") != partition.fingerprint:
            raise ExperienceStoreIntegrityError(
                "typed experience store partition fingerprint mismatch"
            )
        return cls(
            resolved_path,
            partition=partition,
            max_requests=max_requests,
            max_payload_bytes=max_payload_bytes,
            max_record_bytes=max_record_bytes,
            max_proof_steps=max_proof_steps,
        )

    def _init_schema(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experience_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experience_observations (
                    event_id TEXT PRIMARY KEY,
                    event_digest TEXT NOT NULL,
                    request_digest TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_experience_request
                    ON experience_observations(request_digest, created_at);
                CREATE TABLE IF NOT EXISTS experience_reviews (
                    review_digest TEXT PRIMARY KEY,
                    request_digest TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    review_json TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_experience_review_request
                    ON experience_reviews(request_digest, reviewed_at);
                """
            )
            metadata = {
                row["key"]: row["value"]
                for row in connection.execute(
                    "SELECT key, value FROM experience_metadata"
                )
            }
            expected = {
                "schema_version": str(EXPERIENCE_STORE_SCHEMA_VERSION),
                "partition_json": canonical_json(asdict(self.partition)),
                "partition_fingerprint": self.partition.fingerprint,
            }
            if metadata:
                for key, value in expected.items():
                    if metadata.get(key) != value:
                        raise ExperienceStoreIntegrityError(
                            f"typed experience store metadata mismatch: {key}"
                        )
            else:
                stored_rows = sum(
                    int(
                        connection.execute(
                            f"SELECT COUNT(*) FROM {table}"
                        ).fetchone()[0]
                    )
                    for table in (
                        "experience_observations",
                        "experience_reviews",
                    )
                )
                if stored_rows:
                    raise ExperienceStoreIntegrityError(
                        "typed experience store metadata is missing"
                    )
                connection.executemany(
                    "INSERT INTO experience_metadata(key, value) VALUES (?, ?)",
                    tuple(expected.items()),
                )
            connection.commit()

    def add_observation(
        self,
        observation: ExperienceObservation,
    ) -> ExperienceCaptureResult:
        if not isinstance(observation, ExperienceObservation):
            raise TypeError("typed experience store requires ExperienceObservation")
        if len(observation.payload_json.encode("utf-8")) > self.max_payload_bytes:
            raise ValueError("typed experience payload byte limit exceeded")
        if len(observation.proof_program) > self.max_proof_steps:
            raise ValueError("typed experience proof-step limit exceeded")
        event_json = canonical_json(observation.to_dict())
        if len(event_json.encode("utf-8")) > self.max_record_bytes:
            raise ValueError("typed experience record byte limit exceeded")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM experience_observations "
                "WHERE event_id = ?",
                (observation.event_id,),
            ).fetchone()
            if row is not None:
                existing = self._observation_from_row(row)
                if existing.event_digest != observation.event_digest:
                    raise ExperienceStoreIntegrityError(
                        "experience event id was reused with different content"
                    )
                return ExperienceCaptureResult(
                    observation=existing,
                    inserted=False,
                    item=self.get_item(existing.request_digest),
                )
            existing_request = connection.execute(
                "SELECT * FROM experience_observations "
                "WHERE request_digest = ? LIMIT 1",
                (observation.request_digest,),
            ).fetchone()
            if existing_request is None:
                requests = connection.execute(
                    "SELECT COUNT(DISTINCT request_digest) "
                    "FROM experience_observations"
                ).fetchone()[0]
                if int(requests) >= self.max_requests:
                    raise ValueError("typed experience request limit exceeded")
            else:
                existing = self._observation_from_row(existing_request)
                if (
                    existing.domain is not observation.domain
                    or existing.payload_json != observation.payload_json
                ):
                    raise ExperienceStoreIntegrityError(
                        "experience request digest collision detected"
                    )
            connection.execute(
                """
                INSERT INTO experience_observations(
                    event_id, event_digest, request_digest, domain,
                    event_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.event_id,
                    observation.event_digest,
                    observation.request_digest,
                    observation.domain.value,
                    event_json,
                    observation.created_at,
                ),
            )
            connection.commit()
        return ExperienceCaptureResult(
            observation=observation,
            inserted=True,
            item=self.get_item(observation.request_digest),
        )

    def review(
        self,
        request_digest: str,
        *,
        expected_solved: bool,
        phenomenon: str,
        rationale: str,
        reviewer: str,
        decision: SemanticReviewDecision | str,
        notes: str = "",
        reviewed_at: str | None = None,
        difficulty: int = 1,
        tags: Sequence[str] = (),
    ) -> ExperienceReviewRecord:
        if type(expected_solved) is not bool:
            raise TypeError("experience review expected_solved must be boolean")
        if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
            raise TypeError("experience review tags must be a list of strings")
        if any(not isinstance(tag, str) for tag in tags):
            raise TypeError("experience review tags must contain strings")
        normalized_reviewer = reviewer.strip()
        reviewer_parts = normalized_reviewer.split(":", 1)
        if (
            len(reviewer_parts) != 2
            or reviewer_parts[0].lower() != "human"
            or not reviewer_parts[1].strip()
        ):
            raise ValueError(
                "experience review requires an authenticated human: reviewer id"
            )
        normalized_digest = request_digest.strip().lower()
        item = self.get_item(normalized_digest)
        role = self.partition.role_for(normalized_digest)
        case = SemanticBenchmarkCase(
            case_id=f"experience-{normalized_digest[:24]}",
            domain=DomainKind(item.domain),
            payload_json=item.payload_json,
            expected_solved=expected_solved,
            phenomenon=phenomenon,
            rationale=rationale,
            split=role.learning_split,
            difficulty=difficulty,
            author="semop-experience-queue",
            tags=tuple(
                dict.fromkeys(
                    ("online_experience", role.value, *tuple(tags))
                )
            ),
        )
        semantic_review = create_semantic_review(
            case,
            reviewer=normalized_reviewer,
            decision=decision,
            notes=notes,
            reviewed_at=reviewed_at,
        )
        record = ExperienceReviewRecord(
            request_digest=normalized_digest,
            partition_fingerprint=self.partition.fingerprint,
            split_role=role,
            case=case,
            review=semantic_review,
        )
        review_json = canonical_json(record.to_dict())
        if len(review_json.encode("utf-8")) > self.max_record_bytes:
            raise ValueError("typed experience review byte limit exceeded")
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO experience_reviews(
                    review_digest, request_digest, decision,
                    review_json, reviewed_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.review_digest,
                    record.request_digest,
                    record.review.decision.value,
                    review_json,
                    record.review.reviewed_at,
                ),
            )
            connection.commit()
        return record

    def get_item(self, request_digest: str) -> ExperienceQueueItem:
        normalized = request_digest.strip().lower()
        _require_sha256(normalized, "experience request")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM experience_observations "
                "WHERE request_digest = ? ORDER BY created_at, event_id",
                (normalized,),
            ).fetchall()
        if not rows:
            raise KeyError(normalized)
        observations = tuple(self._observation_from_row(row) for row in rows)
        review = self.latest_review(normalized)
        return _summarize_item(observations, review)

    def latest_review(
        self,
        request_digest: str,
    ) -> ExperienceReviewRecord | None:
        normalized = request_digest.strip().lower()
        _require_sha256(normalized, "experience review request")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM experience_reviews "
                "WHERE request_digest = ? "
                "ORDER BY rowid DESC LIMIT 1",
                (normalized,),
            ).fetchone()
        return self._review_from_row(row) if row is not None else None

    def list_items(
        self,
        *,
        status: ExperienceQueueStatus | str | None = None,
        limit: int = 100,
    ) -> tuple[ExperienceQueueItem, ...]:
        if limit <= 0:
            raise ValueError("experience queue list limit must be positive")
        resolved_status = (
            ExperienceQueueStatus(status) if status is not None else None
        )
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT request_digest, MAX(rowid) AS latest
                FROM experience_observations
                GROUP BY request_digest
                ORDER BY latest DESC, request_digest
                """
            ).fetchall()
        items = tuple(self.get_item(row["request_digest"]) for row in rows)
        filtered = (
            items
            if resolved_status is None
            else tuple(item for item in items if item.status is resolved_status)
        )
        return tuple(
            sorted(
                filtered,
                key=lambda item: (-item.priority, item.request_digest),
            )[:limit]
        )

    def stats(self) -> ExperienceQueueStats:
        items = self.list_items(limit=self.max_requests)
        counts = Counter(item.status for item in items)
        domain_counts = Counter(item.domain for item in items)
        with closing(self._connect()) as connection:
            observations = int(
                connection.execute(
                    "SELECT COUNT(*) FROM experience_observations"
                ).fetchone()[0]
            )
        return ExperienceQueueStats(
            requests=len(items),
            observations=observations,
            pending=counts[ExperienceQueueStatus.PENDING],
            conflicted=counts[ExperienceQueueStatus.CONFLICTED],
            approved=counts[ExperienceQueueStatus.APPROVED],
            rejected=counts[ExperienceQueueStatus.REJECTED],
            by_domain=tuple(sorted(domain_counts.items())),
        )

    def export_reviewed(self) -> ExperienceReviewCorpus:
        records: list[ExperienceReviewRecord] = []
        for item in self.list_items(limit=self.max_requests):
            if item.latest_review is not None and item.latest_review.approved:
                records.append(item.latest_review)
        return ExperienceReviewCorpus(
            records=tuple(sorted(records, key=lambda item: item.request_digest)),
            partition_fingerprint=self.partition.fingerprint,
        )

    @staticmethod
    def _observation_from_row(row: sqlite3.Row) -> ExperienceObservation:
        try:
            value = json.loads(row["event_json"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ExperienceStoreIntegrityError(
                "stored experience observation is invalid JSON"
            ) from exc
        observation = ExperienceObservation.from_mapping(value)
        if observation.event_digest != row["event_digest"]:
            raise ExperienceStoreIntegrityError(
                "stored experience observation digest mismatch"
            )
        stored_fields = {
            "event_id": observation.event_id,
            "request_digest": observation.request_digest,
            "domain": observation.domain.value,
            "created_at": observation.created_at,
        }
        for field_name, expected in stored_fields.items():
            if field_name in row.keys() and row[field_name] != expected:
                raise ExperienceStoreIntegrityError(
                    f"stored experience observation {field_name} mismatch"
                )
        return observation

    @staticmethod
    def _review_from_row(row: sqlite3.Row) -> ExperienceReviewRecord:
        try:
            value = json.loads(row["review_json"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ExperienceStoreIntegrityError(
                "stored experience review is invalid JSON"
            ) from exc
        record = ExperienceReviewRecord.from_mapping(value)
        if record.review_digest != row["review_digest"]:
            raise ExperienceStoreIntegrityError(
                "stored experience review digest mismatch"
            )
        stored_fields = {
            "request_digest": record.request_digest,
            "decision": record.review.decision.value,
            "reviewed_at": record.review.reviewed_at,
        }
        for field_name, expected in stored_fields.items():
            if field_name in row.keys() and row[field_name] != expected:
                raise ExperienceStoreIntegrityError(
                    f"stored experience review {field_name} mismatch"
                )
        return record


def _summarize_item(
    observations: tuple[ExperienceObservation, ...],
    review: ExperienceReviewRecord | None,
) -> ExperienceQueueItem:
    first = observations[0]
    proposed_positive = sum(
        item.proposed_expected_solved is True for item in observations
    )
    proposed_negative = sum(
        item.proposed_expected_solved is False for item in observations
    )
    conflicted = proposed_positive > 0 and proposed_negative > 0
    if review is not None:
        status = (
            ExperienceQueueStatus.APPROVED
            if review.approved
            else ExperienceQueueStatus.REJECTED
        )
    elif conflicted:
        status = ExperienceQueueStatus.CONFLICTED
    else:
        status = ExperienceQueueStatus.PENDING
    observed_failures = sum(item.observed_success is not True for item in observations)
    unverified = sum(
        item.observed_success is not None and not item.observed_verified
        for item in observations
    )
    mismatches = sum(
        item.proposed_expected_solved is not None
        and item.observed_success is not None
        and item.proposed_expected_solved != item.observed_success
        for item in observations
    )
    priority = (
        5 * sum(item.observed_success is None for item in observations)
        + 4 * int(conflicted)
        + 3 * unverified
        + 3 * mismatches
        + 2 * observed_failures
        + min(len(observations), 10)
    )
    return ExperienceQueueItem(
        request_digest=first.request_digest,
        domain=first.domain.value,
        payload_json=first.payload_json,
        occurrences=len(observations),
        observed_failures=observed_failures,
        unverified_results=unverified,
        proposed_positive=proposed_positive,
        proposed_negative=proposed_negative,
        triggers=tuple(sorted({item.trigger.value for item in observations})),
        status=status,
        priority=priority,
        latest_review=review,
    )


def observation_from_request(
    request: TypedDomainRequest,
    *,
    event_id: str,
    observed_success: bool | None,
    observed_verified: bool,
    proposed_expected_solved: bool | None,
    proposal_authority: ExperienceProposalAuthority | str,
    trigger: ExperienceTrigger | str,
    rationale: str,
    source: str,
    grounded_fingerprint: str = "",
    proof_program: Sequence[str] = (),
    halt_reason: str = "",
    error_type: str = "",
    error_message: str = "",
    expansions: int = 0,
    created_at: str | None = None,
) -> ExperienceObservation:
    encoded = encode_semantic_request(request)
    return ExperienceObservation(
        event_id=event_id,
        request_digest=semantic_request_digest(request),
        domain=DomainKind(encoded["domain"]),
        payload_json=canonical_json(encoded["payload"]),
        observed_success=observed_success,
        observed_verified=observed_verified,
        proposed_expected_solved=proposed_expected_solved,
        proposal_authority=ExperienceProposalAuthority(proposal_authority),
        trigger=ExperienceTrigger(trigger),
        rationale=rationale,
        source=source,
        grounded_fingerprint=grounded_fingerprint,
        proof_program=tuple(proof_program),
        halt_reason=halt_reason,
        error_type=error_type,
        error_message=error_message,
        expansions=expansions,
        created_at=created_at or _utc_now(),
    )


def _parse_json_object(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise TypeError(f"{label} must be a JSON object")
    return parsed


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"{label} digest must be SHA-256 hex")


def _parse_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} timestamp must include a timezone")
    return parsed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

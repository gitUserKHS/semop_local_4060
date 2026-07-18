from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .engine import OperatorKernel
from .grounding import (
    GroundingAuthority,
    GroundingCandidate,
    GroundingLabel,
    GroundingLearningExample,
    grounding_payload_digest,
    make_grounding_candidate,
)
from .model import AssertionStatus
from .runtime import DomainKind, UnifiedTypedReasoner
from .semantic_benchmark import SemanticBenchmark, SemanticBenchmarkCase
from .semantic_codec import canonical_json, semantic_request_digest
from .semantic_grounding_cases import (
    PROGRAMMATIC_ORACLE_ID,
    generate_controlled_semantic_benchmark,
    generate_structural_semantic_benchmark,
)
from .semantic_grounding_features import (
    semantic_candidate_statement,
    semantic_sensor_features,
)
from .self_learning import LearningSplit


SEMANTIC_GROUNDING_SCHEMA_VERSION = 1
SEMANTIC_GROUNDING_REVIEW_SCHEMA_VERSION = 1
SEMANTIC_GROUNDING_REVIEW_ATTESTATION = (
    "human_reviewed_exact_semantic_grounding_candidate_and_label"
)


class SemanticGroundingReviewDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SemanticGroundingTarget:
    """One typed goal-support candidate derived from an exact raw LMV case.

    ``proposed_label`` is benchmark metadata, not a trusted training label. It
    becomes trainable only after an independent programmatic oracle or an exact
    candidate-level human review binds the label to this target's digest.
    """

    target_id: str
    case_id: str
    case_digest: str
    domain: DomainKind
    phenomenon: str
    split: LearningSplit
    proposed_label: GroundingLabel
    candidate: GroundingCandidate = field(compare=False, repr=False)
    rationale: str = ""
    schema_version: int = SEMANTIC_GROUNDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, GroundingCandidate):
            raise TypeError("semantic grounding target requires GroundingCandidate")
        if not self.target_id.strip():
            raise ValueError("semantic grounding target id must not be empty")
        if not self.case_id.strip():
            raise ValueError("semantic grounding case id must not be empty")
        _validate_digest(self.case_digest, "semantic grounding case")
        if not self.phenomenon.strip():
            raise ValueError("semantic grounding phenomenon must not be empty")
        if self.schema_version != SEMANTIC_GROUNDING_SCHEMA_VERSION:
            raise ValueError(
                "unsupported semantic grounding target schema version: "
                f"{self.schema_version}"
            )
        domain = DomainKind(self.domain)
        if self.candidate.domain != domain.value:
            raise ValueError("semantic grounding candidate domain does not match target")
        object.__setattr__(self, "target_id", self.target_id.strip())
        object.__setattr__(self, "case_id", self.case_id.strip())
        object.__setattr__(self, "case_digest", self.case_digest.lower())
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "phenomenon", self.phenomenon.strip())
        object.__setattr__(self, "split", LearningSplit(self.split))
        object.__setattr__(self, "proposed_label", GroundingLabel(self.proposed_label))
        object.__setattr__(self, "rationale", self.rationale.strip())

    @property
    def target_digest(self) -> str:
        return grounding_payload_digest(
            canonical_json(self.to_dict(include_digest=False))
        )

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "target_id": self.target_id,
            "case_id": self.case_id,
            "case_digest": self.case_digest,
            "domain": self.domain.value,
            "phenomenon": self.phenomenon,
            "split": self.split.value,
            "proposed_label": self.proposed_label.value,
            "candidate_id": self.candidate.candidate_id,
            "candidate_digest": self.candidate.candidate_digest,
            "candidate_statement": self.candidate.statement,
            "candidate_atom": str(self.candidate.atom),
            "sensor_features": [list(item) for item in self.candidate.sensor_features],
            "rationale": self.rationale,
        }
        if include_digest:
            payload["target_digest"] = self.target_digest
        return payload


@dataclass(frozen=True)
class SemanticGroundingReview:
    """Human decision bound to one exact candidate and one reviewed label."""

    target_id: str
    target_digest: str
    case_id: str
    case_digest: str
    candidate_digest: str
    reviewed_label: GroundingLabel
    reviewer: str
    reviewed_at: str
    decision: SemanticGroundingReviewDecision
    attestation: str = SEMANTIC_GROUNDING_REVIEW_ATTESTATION
    notes: str = ""
    schema_version: int = SEMANTIC_GROUNDING_REVIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.target_id.strip() or not self.case_id.strip():
            raise ValueError("semantic grounding review ids must not be empty")
        for value, label in (
            (self.target_digest, "semantic grounding target"),
            (self.case_digest, "semantic grounding case"),
            (self.candidate_digest, "semantic grounding candidate"),
        ):
            _validate_digest(value, label)
        reviewer = self.reviewer.strip()
        if not reviewer.startswith("human:") or len(reviewer) <= len("human:"):
            raise ValueError(
                "semantic grounding reviewer must use the 'human:' namespace"
            )
        if self.attestation != SEMANTIC_GROUNDING_REVIEW_ATTESTATION:
            raise ValueError(
                "semantic grounding review requires the exact-candidate attestation"
            )
        _parse_timestamp(self.reviewed_at)
        if self.schema_version != SEMANTIC_GROUNDING_REVIEW_SCHEMA_VERSION:
            raise ValueError(
                "unsupported semantic grounding review schema version: "
                f"{self.schema_version}"
            )
        object.__setattr__(self, "target_id", self.target_id.strip())
        object.__setattr__(self, "target_digest", self.target_digest.lower())
        object.__setattr__(self, "case_id", self.case_id.strip())
        object.__setattr__(self, "case_digest", self.case_digest.lower())
        object.__setattr__(self, "candidate_digest", self.candidate_digest.lower())
        object.__setattr__(self, "reviewed_label", GroundingLabel(self.reviewed_label))
        object.__setattr__(self, "reviewer", reviewer)
        object.__setattr__(
            self,
            "decision",
            SemanticGroundingReviewDecision(self.decision),
        )
        object.__setattr__(self, "notes", self.notes.strip())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SemanticGroundingReview":
        if not isinstance(value, Mapping):
            raise TypeError("semantic grounding review record must be an object")
        return cls(
            target_id=str(value.get("target_id", "")),
            target_digest=str(value.get("target_digest", "")),
            case_id=str(value.get("case_id", "")),
            case_digest=str(value.get("case_digest", "")),
            candidate_digest=str(value.get("candidate_digest", "")),
            reviewed_label=GroundingLabel(str(value.get("reviewed_label", ""))),
            reviewer=str(value.get("reviewer", "")),
            reviewed_at=str(value.get("reviewed_at", "")),
            decision=SemanticGroundingReviewDecision(str(value.get("decision", ""))),
            attestation=str(value.get("attestation", "")),
            notes=str(value.get("notes", "")),
            schema_version=int(
                value.get(
                    "schema_version",
                    SEMANTIC_GROUNDING_REVIEW_SCHEMA_VERSION,
                )
            ),
        )

    @property
    def review_digest(self) -> str:
        return grounding_payload_digest(canonical_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_id": self.target_id,
            "target_digest": self.target_digest,
            "case_id": self.case_id,
            "case_digest": self.case_digest,
            "candidate_digest": self.candidate_digest,
            "reviewed_label": self.reviewed_label.value,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "decision": self.decision.value,
            "attestation": self.attestation,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SemanticGroundingAudit:
    targets: int
    reviews: int
    approved_targets: tuple[str, ...]
    pending_targets: tuple[str, ...]
    rejected_targets: tuple[str, ...]
    corrected_label_targets: tuple[str, ...]
    stale_review_targets: tuple[str, ...]
    orphan_review_targets: tuple[str, ...]
    domain_target_counts: tuple[tuple[str, int], ...]
    domain_approved_counts: tuple[tuple[str, int], ...]

    @property
    def review_coverage(self) -> float:
        return len(self.approved_targets) / self.targets if self.targets else 0.0

    @property
    def clean(self) -> bool:
        return not self.stale_review_targets and not self.orphan_review_targets

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["review_coverage"] = self.review_coverage
        payload["clean"] = self.clean
        return payload


@dataclass(frozen=True)
class SemanticGroundingCorpus:
    """Compiled candidates plus exact candidate-level human review records."""

    targets: tuple[SemanticGroundingTarget, ...]
    reviews: tuple[SemanticGroundingReview, ...] = ()

    def __post_init__(self) -> None:
        if any(
            not isinstance(target, SemanticGroundingTarget)
            for target in self.targets
        ):
            raise TypeError("semantic grounding corpus targets have the wrong type")
        if any(
            not isinstance(review, SemanticGroundingReview)
            for review in self.reviews
        ):
            raise TypeError("semantic grounding corpus reviews have the wrong type")
        if not self.targets:
            raise ValueError("semantic grounding corpus requires targets")
        target_ids = tuple(target.target_id for target in self.targets)
        review_ids = tuple(review.target_id for review in self.reviews)
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("semantic grounding target ids must be unique")
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("semantic grounding review target ids must be unique")

    @classmethod
    def compile(
        cls,
        benchmark: SemanticBenchmark,
        reviews: Sequence[SemanticGroundingReview] = (),
        *,
        reasoner: UnifiedTypedReasoner | None = None,
    ) -> "SemanticGroundingCorpus":
        return cls(
            build_semantic_grounding_targets(benchmark, reasoner=reasoner),
            tuple(reviews),
        )

    @property
    def target_by_id(self) -> dict[str, SemanticGroundingTarget]:
        return {target.target_id: target for target in self.targets}

    @property
    def review_by_id(self) -> dict[str, SemanticGroundingReview]:
        return {review.target_id: review for review in self.reviews}

    def audit(self) -> SemanticGroundingAudit:
        targets = self.target_by_id
        reviews = self.review_by_id
        approved: list[str] = []
        rejected: list[str] = []
        corrected: list[str] = []
        stale: list[str] = []
        for target_id, target in sorted(targets.items()):
            review = reviews.get(target_id)
            if review is None:
                continue
            if not _review_matches_target(review, target):
                stale.append(target_id)
                continue
            if review.decision is SemanticGroundingReviewDecision.APPROVED:
                approved.append(target_id)
                if review.reviewed_label is not target.proposed_label:
                    corrected.append(target_id)
            else:
                rejected.append(target_id)
        approved_set = set(approved)
        return SemanticGroundingAudit(
            targets=len(targets),
            reviews=len(reviews),
            approved_targets=tuple(approved),
            pending_targets=tuple(sorted(set(targets) - approved_set)),
            rejected_targets=tuple(rejected),
            corrected_label_targets=tuple(corrected),
            stale_review_targets=tuple(stale),
            orphan_review_targets=tuple(sorted(set(reviews) - set(targets))),
            domain_target_counts=_count_domains(self.targets),
            domain_approved_counts=_count_domains(
                tuple(targets[target_id] for target_id in approved)
            ),
        )

    def learning_examples(self) -> tuple[GroundingLearningExample, ...]:
        targets = self.target_by_id
        examples: list[GroundingLearningExample] = []
        for target_id, review in sorted(self.review_by_id.items()):
            target = targets.get(target_id)
            if (
                target is None
                or not _review_matches_target(review, target)
                or review.decision is not SemanticGroundingReviewDecision.APPROVED
            ):
                continue
            examples.append(
                GroundingLearningExample(
                    candidate=target.candidate,
                    label=review.reviewed_label,
                    authority=GroundingAuthority.HUMAN_REVIEW,
                    record_digest=review.review_digest,
                )
            )
        return tuple(examples)


def build_semantic_grounding_targets(
    benchmark: SemanticBenchmark,
    *,
    reasoner: UnifiedTypedReasoner | None = None,
) -> tuple[SemanticGroundingTarget, ...]:
    active = reasoner or UnifiedTypedReasoner()
    targets = [
        build_semantic_grounding_target(case, reasoner=active)
        for case in benchmark.cases
    ]
    return tuple(sorted(targets, key=lambda target: target.target_id))


def build_semantic_grounding_target(
    case: SemanticBenchmarkCase,
    *,
    reasoner: UnifiedTypedReasoner | None = None,
) -> SemanticGroundingTarget:
    active = reasoner or UnifiedTypedReasoner()
    request = case.to_request()
    instance = active.ground(request)
    if len(instance.goals) != 1:
        raise ValueError(
            "semantic grounding v1 requires exactly one typed goal per case"
        )
    goal = instance.goals[0]
    request_digest = semantic_request_digest(request)
    statement = semantic_candidate_statement(case, instance)
    candidate = make_grounding_candidate(
        domain=case.domain.value,
        statement=statement,
        atom=goal.atom,
        producer_id="semantic_goal_support_candidate_v1",
        source="semantic_goal_support_candidate",
        input_digest=request_digest,
        evidence=(f"semantic-case:{case.digest}", f"input:{request_digest}"),
        assertion_status=AssertionStatus.INFERRED,
        confidence=0.5,
        sensor_features=semantic_sensor_features(case, instance),
    )
    return SemanticGroundingTarget(
        target_id=f"{case.case_id}:goal:0",
        case_id=case.case_id,
        case_digest=case.digest,
        domain=case.domain,
        phenomenon=case.phenomenon,
        split=case.split,
        proposed_label=(
            GroundingLabel.ACCEPT if case.expected_solved else GroundingLabel.REJECT
        ),
        candidate=candidate,
        rationale=case.rationale,
    )


def create_semantic_grounding_review(
    target: SemanticGroundingTarget,
    *,
    reviewed_label: GroundingLabel | str,
    reviewer: str,
    decision: SemanticGroundingReviewDecision | str = (
        SemanticGroundingReviewDecision.APPROVED
    ),
    notes: str = "",
    reviewed_at: str | None = None,
) -> SemanticGroundingReview:
    timestamp = reviewed_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    return SemanticGroundingReview(
        target_id=target.target_id,
        target_digest=target.target_digest,
        case_id=target.case_id,
        case_digest=target.case_digest,
        candidate_digest=target.candidate.candidate_digest,
        reviewed_label=GroundingLabel(reviewed_label),
        reviewer=reviewer,
        reviewed_at=timestamp,
        decision=SemanticGroundingReviewDecision(decision),
        notes=notes,
    )


def load_semantic_grounding_reviews(
    path: str | Path,
) -> tuple[SemanticGroundingReview, ...]:
    target = Path(path)
    if not target.exists():
        return ()
    reviews: list[SemanticGroundingReview] = []
    for line_number, line in enumerate(
        target.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{target}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        reviews.append(SemanticGroundingReview.from_mapping(value))
    return tuple(reviews)


def write_semantic_grounding_review(
    path: str | Path,
    review: SemanticGroundingReview,
) -> None:
    target = Path(path)
    by_target = {
        item.target_id: item for item in load_semantic_grounding_reviews(target)
    }
    by_target[review.target_id] = review
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        canonical_json(by_target[target_id].to_dict())
        for target_id in sorted(by_target)
    )
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(content + "\n", encoding="utf-8")
    temporary.replace(target)


def programmatic_semantic_learning_examples(
    benchmark: SemanticBenchmark,
    *,
    reasoner: UnifiedTypedReasoner | None = None,
) -> tuple[GroundingLearningExample, ...]:
    """Build labels only for cases carrying the controlled-oracle contract.

    The raw request still passes through the production LMV adapters. The
    programmatic case generator supplies the independent expected support label,
    while the kernel must reproduce every positive proof before a label is emitted.
    """

    active = reasoner or UnifiedTypedReasoner()
    examples: list[GroundingLearningExample] = []
    for case in benchmark.cases:
        if "programmatic_oracle" not in case.tags:
            raise ValueError(
                f"case {case.case_id} lacks the programmatic oracle contract"
            )
        target = build_semantic_grounding_target(case, reasoner=active)
        instance = active.ground(case.to_request())
        result = OperatorKernel(instance.registry).solve(
            instance.state,
            instance.goals,
        )
        if result.success != case.expected_solved:
            raise ValueError(
                f"programmatic oracle disagrees with adapter for {case.case_id}"
            )
        if result.success and not result.verified:
            raise ValueError(
                f"positive semantic grounding proof did not replay for {case.case_id}"
            )
        transcript = {
            "oracle_id": PROGRAMMATIC_ORACLE_ID,
            "target_digest": target.target_digest,
            "candidate_digest": target.candidate.candidate_digest,
            "expected_support": case.expected_solved,
            "kernel_success": result.success,
            "kernel_replay_verified": result.verified,
            "halt_reason": result.halt_reason,
        }
        examples.append(
            GroundingLearningExample(
                candidate=target.candidate,
                label=target.proposed_label,
                authority=GroundingAuthority.EXTERNAL_VERIFIER,
                record_digest=grounding_payload_digest(canonical_json(transcript)),
            )
        )
    return tuple(examples)


def _review_matches_target(
    review: SemanticGroundingReview,
    target: SemanticGroundingTarget,
) -> bool:
    return (
        review.target_digest == target.target_digest
        and review.case_id == target.case_id
        and review.case_digest == target.case_digest
        and review.candidate_digest == target.candidate.candidate_digest
    )


def _count_domains(
    targets: Sequence[SemanticGroundingTarget],
) -> tuple[tuple[str, int], ...]:
    counts = Counter(target.domain.value for target in targets)
    return tuple(sorted(counts.items()))


def _bounded_count(value: int) -> float:
    return min(max(value, 0), 64) / 64.0


def _relation_name(operator: str) -> str:
    return {
        "==": "equal",
        "!=": "not_equal",
        ">": "greater",
        ">=": "greater_equal",
        "<": "less",
        "<=": "less_equal",
    }[operator]


def _parse_timestamp(value: str) -> datetime:
    normalized = value.strip()
    if not normalized:
        raise ValueError("semantic grounding review timestamp must not be empty")
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "semantic grounding review timestamp must be ISO-8601"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError("semantic grounding review timestamp must include a timezone")
    return parsed


def _validate_digest(value: str, label: str) -> None:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} digest must be a SHA-256 hex digest")


__all__ = [
    "PROGRAMMATIC_ORACLE_ID",
    "SEMANTIC_GROUNDING_REVIEW_ATTESTATION",
    "SEMANTIC_GROUNDING_REVIEW_SCHEMA_VERSION",
    "SEMANTIC_GROUNDING_SCHEMA_VERSION",
    "SemanticGroundingAudit",
    "SemanticGroundingCorpus",
    "SemanticGroundingReview",
    "SemanticGroundingReviewDecision",
    "SemanticGroundingTarget",
    "build_semantic_grounding_target",
    "build_semantic_grounding_targets",
    "create_semantic_grounding_review",
    "generate_controlled_semantic_benchmark",
    "generate_structural_semantic_benchmark",
    "load_semantic_grounding_reviews",
    "programmatic_semantic_learning_examples",
    "write_semantic_grounding_review",
]

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from time import process_time
from typing import Any, Mapping, Sequence

from .engine import OperatorKernel
from .experience import (
    GroundedLearningBatch,
    RawExperienceGrounder,
    RawLearningExample,
)
from .model import SolveBudget
from .runtime import DomainKind, TypedDomainRequest
from .semantic_codec import canonical_json, decode_semantic_request
from .self_learning import (
    HUMAN_REVIEW_ATTESTATION,
    LearningMetrics,
    LearningSplit,
    SemanticLabelAuthority,
    SemanticLabelEvidence,
    TaskEvaluation,
    summarize_task_evaluations,
    task_evaluation_from_result,
)


SEMANTIC_CASE_SCHEMA_VERSION = 1
SEMANTIC_REVIEW_SCHEMA_VERSION = 1
_SUPPORTED_DOMAINS = frozenset(
    {DomainKind.LANGUAGE, DomainKind.MATH, DomainKind.VISION}
)
class SemanticReviewDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SemanticBenchmarkCase:
    """Immutable raw LMV case whose expected outcome can be independently reviewed."""

    case_id: str
    domain: DomainKind
    payload_json: str = field(repr=False)
    expected_solved: bool
    phenomenon: str
    rationale: str
    split: LearningSplit = LearningSplit.HELDOUT
    difficulty: int = 1
    author: str = "semop-curated"
    tags: tuple[str, ...] = ()
    schema_version: int = SEMANTIC_CASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        case_id = self.case_id.strip()
        if not case_id:
            raise ValueError("semantic case id must not be empty")
        domain = DomainKind(self.domain)
        if domain not in _SUPPORTED_DOMAINS:
            raise ValueError(f"semantic benchmark does not support domain: {domain.value}")
        if type(self.expected_solved) is not bool:
            raise TypeError("semantic case expected_solved must be boolean")
        if not self.phenomenon.strip():
            raise ValueError("semantic case phenomenon must not be empty")
        if not self.rationale.strip():
            raise ValueError("semantic case rationale must not be empty")
        if self.difficulty <= 0:
            raise ValueError("semantic case difficulty must be positive")
        if self.schema_version != SEMANTIC_CASE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported semantic case schema version: {self.schema_version}"
            )
        payload = _parse_payload_json(self.payload_json)
        canonical_payload = _canonical_json(payload)
        normalized_tags = tuple(
            dict.fromkeys(tag.strip() for tag in self.tags if tag.strip())
        )
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "payload_json", canonical_payload)
        object.__setattr__(self, "phenomenon", self.phenomenon.strip())
        object.__setattr__(self, "rationale", self.rationale.strip())
        object.__setattr__(self, "split", LearningSplit(self.split))
        object.__setattr__(self, "author", self.author.strip() or "semop-curated")
        object.__setattr__(self, "tags", normalized_tags)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SemanticBenchmarkCase":
        if not isinstance(value, Mapping):
            raise TypeError("semantic case record must be an object")
        if "payload" not in value:
            raise ValueError("semantic case payload is required")
        if not isinstance(value["payload"], Mapping):
            raise TypeError("semantic case payload must be an object")
        expected = value.get("expected_solved")
        if type(expected) is not bool:
            raise TypeError("semantic case expected_solved must be boolean")
        tags = value.get("tags", ())
        if isinstance(tags, str) or not isinstance(tags, Sequence):
            raise TypeError("semantic case tags must be a list of strings")
        return cls(
            case_id=str(value.get("case_id", "")),
            domain=DomainKind(str(value.get("domain", ""))),
            payload_json=_canonical_json(value["payload"]),
            expected_solved=expected,
            phenomenon=str(value.get("phenomenon", "")),
            rationale=str(value.get("rationale", "")),
            split=LearningSplit(str(value.get("split", LearningSplit.HELDOUT.value))),
            difficulty=int(value.get("difficulty", 1)),
            author=str(value.get("author", "semop-curated")),
            tags=tuple(str(tag) for tag in tags),
            schema_version=int(
                value.get("schema_version", SEMANTIC_CASE_SCHEMA_VERSION)
            ),
        )

    @property
    def payload(self) -> dict[str, Any]:
        return _parse_payload_json(self.payload_json)

    @property
    def digest(self) -> str:
        return sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "domain": self.domain.value,
            "payload": self.payload,
            "expected_solved": self.expected_solved,
            "phenomenon": self.phenomenon,
            "rationale": self.rationale,
            "split": self.split.value,
            "difficulty": self.difficulty,
            "author": self.author,
            "tags": list(self.tags),
        }

    def to_request(self) -> TypedDomainRequest:
        return decode_semantic_request(self.domain, self.payload)


@dataclass(frozen=True)
class SemanticReviewRecord:
    case_id: str
    case_digest: str
    reviewer: str
    reviewed_at: str
    decision: SemanticReviewDecision
    attestation: str = HUMAN_REVIEW_ATTESTATION
    notes: str = ""
    schema_version: int = SEMANTIC_REVIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        case_id = self.case_id.strip()
        digest = self.case_digest.strip().lower()
        reviewer = self.reviewer.strip()
        if not case_id:
            raise ValueError("semantic review case id must not be empty")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("semantic review case digest must be a SHA-256 hex digest")
        if not reviewer:
            raise ValueError("semantic review reviewer must not be empty")
        if self.attestation != HUMAN_REVIEW_ATTESTATION:
            raise ValueError("semantic review requires the human review attestation")
        _parse_review_timestamp(self.reviewed_at)
        if self.schema_version != SEMANTIC_REVIEW_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported semantic review schema version: {self.schema_version}"
            )
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "case_digest", digest)
        object.__setattr__(self, "reviewer", reviewer)
        object.__setattr__(self, "decision", SemanticReviewDecision(self.decision))
        object.__setattr__(self, "notes", self.notes.strip())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SemanticReviewRecord":
        if not isinstance(value, Mapping):
            raise TypeError("semantic review record must be an object")
        return cls(
            case_id=str(value.get("case_id", "")),
            case_digest=str(value.get("case_digest", "")),
            reviewer=str(value.get("reviewer", "")),
            reviewed_at=str(value.get("reviewed_at", "")),
            decision=SemanticReviewDecision(str(value.get("decision", ""))),
            attestation=str(value.get("attestation", "")),
            notes=str(value.get("notes", "")),
            schema_version=int(
                value.get("schema_version", SEMANTIC_REVIEW_SCHEMA_VERSION)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "case_digest": self.case_digest,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "decision": self.decision.value,
            "attestation": self.attestation,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SemanticReviewAudit:
    cases: int
    reviews: int
    approved_cases: tuple[str, ...]
    pending_cases: tuple[str, ...]
    rejected_cases: tuple[str, ...]
    stale_review_cases: tuple[str, ...]
    orphan_review_cases: tuple[str, ...]
    domain_case_counts: tuple[tuple[str, int], ...]
    domain_approved_counts: tuple[tuple[str, int], ...]

    @property
    def review_coverage(self) -> float:
        return len(self.approved_cases) / self.cases if self.cases else 0.0

    @property
    def all_reviewed(self) -> bool:
        return bool(self.cases) and not self.pending_cases

    @property
    def clean(self) -> bool:
        return not self.stale_review_cases and not self.orphan_review_cases

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["review_coverage"] = self.review_coverage
        payload["all_reviewed"] = self.all_reviewed
        payload["clean"] = self.clean
        return payload


@dataclass(frozen=True)
class SemanticCaseOutcome:
    case_id: str
    case_digest: str
    domain: str
    phenomenon: str
    expected_solved: bool
    success: bool
    verified: bool
    correct: bool
    label_authority: SemanticLabelAuthority
    expansions: int
    proof_steps: int
    halt_reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["label_authority"] = self.label_authority.value
        return payload


@dataclass(frozen=True)
class SemanticBenchmarkEvaluation:
    audit: SemanticReviewAudit
    metrics: LearningMetrics
    outcomes: tuple[SemanticCaseOutcome, ...]
    grounding: GroundedLearningBatch = field(compare=False, repr=False)

    @property
    def passed(self) -> bool:
        return (
            all(outcome.correct for outcome in self.outcomes)
            and self.metrics.primitive_replay_integrity == 1.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "audit": self.audit.to_dict(),
            "metrics": self.metrics.to_dict(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "grounding_failures": [
                asdict(failure) for failure in self.grounding.failures
            ],
        }


@dataclass(frozen=True)
class SemanticBenchmark:
    cases: tuple[SemanticBenchmarkCase, ...]
    reviews: tuple[SemanticReviewRecord, ...] = ()

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("semantic benchmark requires at least one case")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("semantic benchmark case ids must be unique")
        review_ids = tuple(review.case_id for review in self.reviews)
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("semantic benchmark review case ids must be unique")

    @property
    def case_by_id(self) -> dict[str, SemanticBenchmarkCase]:
        return {case.case_id: case for case in self.cases}

    @property
    def review_by_id(self) -> dict[str, SemanticReviewRecord]:
        return {review.case_id: review for review in self.reviews}

    def audit(self) -> SemanticReviewAudit:
        cases = self.case_by_id
        reviews = self.review_by_id
        approved: list[str] = []
        rejected: list[str] = []
        stale: list[str] = []
        for case_id, case in sorted(cases.items()):
            review = reviews.get(case_id)
            if review is None:
                continue
            if review.case_digest != case.digest:
                stale.append(case_id)
            elif review.decision is SemanticReviewDecision.APPROVED:
                approved.append(case_id)
            else:
                rejected.append(case_id)
        approved_set = set(approved)
        domain_counts = _count_domains(self.cases)
        approved_domain_counts = _count_domains(
            tuple(cases[case_id] for case_id in approved)
        )
        return SemanticReviewAudit(
            cases=len(self.cases),
            reviews=len(self.reviews),
            approved_cases=tuple(approved),
            pending_cases=tuple(sorted(set(cases) - approved_set)),
            rejected_cases=tuple(rejected),
            stale_review_cases=tuple(stale),
            orphan_review_cases=tuple(sorted(set(reviews) - set(cases))),
            domain_case_counts=domain_counts,
            domain_approved_counts=approved_domain_counts,
        )

    def require_domain_coverage(
        self,
        domains: Sequence[DomainKind | str] = (
            DomainKind.LANGUAGE,
            DomainKind.MATH,
            DomainKind.VISION,
        ),
    ) -> None:
        present = {case.domain for case in self.cases}
        missing = tuple(
            DomainKind(domain).value
            for domain in domains
            if DomainKind(domain) not in present
        )
        if missing:
            raise ValueError(
                "semantic benchmark is missing domains: " + ", ".join(missing)
            )

    def require_all_reviewed(self) -> None:
        audit = self.audit()
        if audit.pending_cases:
            raise ValueError(
                "semantic benchmark has unapproved cases: "
                + ", ".join(audit.pending_cases)
            )
        if not audit.clean:
            raise ValueError("semantic benchmark review audit is not clean")

    def label_authority(self, case: SemanticBenchmarkCase) -> SemanticLabelAuthority:
        review = self.review_by_id.get(case.case_id)
        if (
            review is not None
            and review.case_digest == case.digest
            and review.decision is SemanticReviewDecision.APPROVED
        ):
            return SemanticLabelAuthority.HUMAN_REVIEWED
        return SemanticLabelAuthority.CURATED_UNREVIEWED

    def label_evidence(self, case: SemanticBenchmarkCase) -> SemanticLabelEvidence:
        review = self.review_by_id.get(case.case_id)
        if (
            review is None
            or review.case_digest != case.digest
            or review.decision is not SemanticReviewDecision.APPROVED
        ):
            return SemanticLabelEvidence()
        return SemanticLabelEvidence(
            case_digest=review.case_digest,
            reviewer=review.reviewer,
            reviewed_at=review.reviewed_at,
            attestation=review.attestation,
        )

    def raw_examples(
        self,
        *,
        split: LearningSplit | str | None = None,
        reviewed_only: bool = False,
    ) -> tuple[RawLearningExample, ...]:
        selected_split = LearningSplit(split) if split is not None else None
        examples: list[RawLearningExample] = []
        for case in self.cases:
            if selected_split is not None and case.split is not selected_split:
                continue
            authority = self.label_authority(case)
            if reviewed_only and authority is not SemanticLabelAuthority.HUMAN_REVIEWED:
                continue
            examples.append(
                RawLearningExample(
                    example_id=case.case_id,
                    request=case.to_request(),
                    expected_solved=case.expected_solved,
                    split=case.split,
                    source=(
                        "reviewed"
                        if authority is SemanticLabelAuthority.HUMAN_REVIEWED
                        else "verifier"
                    ),
                    capability=case.phenomenon,
                    structure_key=f"{case.domain.value}:{case.phenomenon}",
                    difficulty=case.difficulty,
                    label_authority=authority,
                    label_evidence=self.label_evidence(case),
                )
            )
        return tuple(examples)


def load_semantic_benchmark(
    cases_path: str | Path,
    reviews_path: str | Path | None = None,
) -> SemanticBenchmark:
    cases = tuple(
        SemanticBenchmarkCase.from_mapping(record)
        for record in _load_jsonl(Path(cases_path), required=True)
    )
    reviews = tuple(
        SemanticReviewRecord.from_mapping(record)
        for record in (
            _load_jsonl(Path(reviews_path), required=False)
            if reviews_path is not None
            else ()
        )
    )
    return SemanticBenchmark(cases, reviews)


def create_semantic_review(
    case: SemanticBenchmarkCase,
    *,
    reviewer: str,
    decision: SemanticReviewDecision | str,
    notes: str = "",
    reviewed_at: str | None = None,
) -> SemanticReviewRecord:
    timestamp = reviewed_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    return SemanticReviewRecord(
        case_id=case.case_id,
        case_digest=case.digest,
        reviewer=reviewer,
        reviewed_at=timestamp,
        decision=SemanticReviewDecision(decision),
        notes=notes,
    )


def write_semantic_review(
    path: str | Path,
    review: SemanticReviewRecord,
) -> None:
    target = Path(path)
    existing = tuple(
        SemanticReviewRecord.from_mapping(record)
        for record in _load_jsonl(target, required=False)
    )
    by_case = {item.case_id: item for item in existing}
    by_case[review.case_id] = review
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        _canonical_json(by_case[case_id].to_dict()) for case_id in sorted(by_case)
    )
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(content + "\n", encoding="utf-8")
    temporary.replace(target)


def evaluate_semantic_benchmark(
    benchmark: SemanticBenchmark,
    *,
    budget: SolveBudget | None = None,
    reviewed_only: bool = False,
    require_all_reviewed: bool = False,
) -> SemanticBenchmarkEvaluation:
    benchmark.require_domain_coverage()
    if require_all_reviewed:
        benchmark.require_all_reviewed()
    examples = benchmark.raw_examples(reviewed_only=reviewed_only)
    if not examples:
        raise ValueError("semantic benchmark selection contains no cases")
    grounder = RawExperienceGrounder(hard_negatives_per_example=0)
    grounding = grounder.ground(examples, namespace="semantic-benchmark")
    tasks = grounding.require_complete()
    case_by_id = benchmark.case_by_id
    evaluations: list[TaskEvaluation] = []
    outcomes: list[SemanticCaseOutcome] = []
    for example, task in zip(examples, tasks, strict=True):
        started = process_time()
        result = OperatorKernel(task.instance.registry).solve(
            task.instance.state,
            task.instance.goals,
            budget=budget,
        )
        elapsed = process_time() - started
        evaluation = task_evaluation_from_result(task, result, elapsed)
        evaluations.append(evaluation)
        case = case_by_id[example.example_id]
        outcomes.append(
            SemanticCaseOutcome(
                case_id=case.case_id,
                case_digest=case.digest,
                domain=case.domain.value,
                phenomenon=case.phenomenon,
                expected_solved=case.expected_solved,
                success=result.success,
                verified=result.verified,
                correct=result.success == case.expected_solved,
                label_authority=task.label_authority,
                expansions=result.expansions,
                proof_steps=len(result.proof),
                halt_reason=result.halt_reason,
            )
        )
    return SemanticBenchmarkEvaluation(
        audit=benchmark.audit(),
        metrics=summarize_task_evaluations(tuple(evaluations)),
        outcomes=tuple(outcomes),
        grounding=grounding,
    )


def _load_jsonl(path: Path, *, required: bool) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return ()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: JSONL record must be an object")
        records.append(value)
    return tuple(records)


def _parse_payload_json(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"semantic case payload is invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise TypeError("semantic case payload must be a JSON object")
    return payload


def _parse_review_timestamp(value: str) -> datetime:
    normalized = value.strip()
    if not normalized:
        raise ValueError("semantic review timestamp must not be empty")
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("semantic review timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("semantic review timestamp must include a timezone")
    return parsed


def _canonical_json(value: Any) -> str:
    return canonical_json(value)


def _count_domains(
    cases: Sequence[SemanticBenchmarkCase],
) -> tuple[tuple[str, int], ...]:
    counts = {domain.value: 0 for domain in sorted(_SUPPORTED_DOMAINS, key=str)}
    for case in cases:
        counts[case.domain.value] += 1
    return tuple((domain, count) for domain, count in sorted(counts.items()))

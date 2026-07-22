from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
import json
import math

from .model import (
    AssertionStatus,
    Atom,
    EvidenceStatus,
    Fact,
    FactStatus,
    KernelError,
)


class GroundingBoundaryError(KernelError):
    """Raised when an input producer crosses the typed grounding boundary."""


_RESERVED_SENSOR_FEATURE_FRAGMENTS = (
    "accepted",
    "authority",
    "decision",
    "ground_truth",
    "label",
    "rejected",
    "verdict",
    "verified",
)


class GroundingDisposition(str, Enum):
    """Outcome of evaluating a candidate before symbolic reasoning starts."""

    OBSERVED = "observed"
    PROPOSED = "proposed"
    REJECTED = "rejected"
    CONTRADICTED = "contradicted"


class GroundingAuthority(str, Enum):
    """Authority that made a grounding decision.

    The authority is separate from confidence. A highly confident model remains a
    proposal source, while a deterministic adapter can produce adapter-verified
    observations.
    """

    EXPLICIT_INPUT = "explicit_input"
    DETERMINISTIC_ADAPTER = "deterministic_adapter"
    EXTERNAL_VERIFIER = "external_verifier"
    HUMAN_REVIEW = "human_review"
    MODEL_PROPOSAL = "model_proposal"
    HEURISTIC_PROPOSAL = "heuristic_proposal"
    IMPORTED_PROPOSAL = "imported_proposal"

    @property
    def proposal_only(self) -> bool:
        return self in {
            GroundingAuthority.MODEL_PROPOSAL,
            GroundingAuthority.HEURISTIC_PROPOSAL,
            GroundingAuthority.IMPORTED_PROPOSAL,
        }


class GroundingLabel(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass(frozen=True)
class GroundingCandidate:
    """A typed concept proposed by a parser, sensor adapter, model, or reviewer."""

    candidate_id: str
    domain: str
    statement: str
    atom: Atom
    evidence: tuple[str, ...] = ()
    source: str = "grounding_candidate"
    producer_id: str = "unspecified"
    input_digest: str = ""
    assertion_status: AssertionStatus = AssertionStatus.INFERRED
    confidence: float = 1.0
    sensor_features: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("grounding candidate id cannot be empty")
        if not self.domain.strip():
            raise ValueError("grounding candidate domain cannot be empty")
        if not self.statement.strip():
            raise ValueError("grounding candidate statement cannot be empty")
        if not self.source.strip():
            raise ValueError("grounding candidate source cannot be empty")
        if not self.producer_id.strip():
            raise ValueError("grounding candidate producer id cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("grounding candidate confidence must be between 0 and 1")
        features = tuple(
            sorted((str(name).strip(), float(value)) for name, value in self.sensor_features)
        )
        feature_names = [name for name, _value in features]
        if any(not name for name in feature_names):
            raise ValueError("grounding sensor feature names cannot be empty")
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("grounding sensor feature names must be unique")
        if any(
            fragment in name.lower()
            for name in feature_names
            for fragment in _RESERVED_SENSOR_FEATURE_FRAGMENTS
        ):
            raise GroundingBoundaryError(
                "grounding sensor features must not encode labels or verifier decisions"
            )
        if any(not math.isfinite(value) for _name, value in features):
            raise ValueError("grounding sensor features must be finite")
        object.__setattr__(self, "domain", self.domain.strip().lower())
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "sensor_features", features)
        object.__setattr__(
            self,
            "assertion_status",
            AssertionStatus(self.assertion_status),
        )
        digest = self.input_digest or grounding_payload_digest(self.statement)
        _validate_digest(digest, "grounding input")
        object.__setattr__(self, "input_digest", digest.lower())

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "domain": self.domain,
            "statement": self.statement,
            "atom": str(self.atom),
            "atom_key": self.atom.canonical_key(),
            "evidence": list(self.evidence),
            "source": self.source,
            "producer_id": self.producer_id,
            "input_digest": self.input_digest,
            "assertion_status": self.assertion_status.value,
            "confidence": self.confidence,
            "sensor_features": [list(item) for item in self.sensor_features],
        }

    @property
    def candidate_digest(self) -> str:
        semantic_payload = {
            "domain": self.domain,
            "statement": self.statement,
            "atom_key": self.atom.canonical_key(),
            "input_digest": self.input_digest,
            "assertion_status": self.assertion_status.value,
            "sensor_features": self.sensor_features,
        }
        return grounding_payload_digest(
            json.dumps(
                semantic_payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )


@dataclass(frozen=True)
class GroundingDecision:
    """Auditable decision about whether a typed candidate may enter a world."""

    disposition: GroundingDisposition
    authority: GroundingAuthority
    verifier_id: str
    rationale: str
    evidence_status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    confidence: float = 1.0
    evidence_refs: tuple[str, ...] = ()
    fact_source: str = ""
    supersedes_record_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            GroundingDisposition(self.disposition),
        )
        object.__setattr__(self, "authority", GroundingAuthority(self.authority))
        object.__setattr__(
            self,
            "evidence_status",
            EvidenceStatus(self.evidence_status),
        )
        if not self.verifier_id.strip():
            raise ValueError("grounding verifier id cannot be empty")
        if not self.rationale.strip():
            raise ValueError("grounding decision rationale cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("grounding decision confidence must be between 0 and 1")
        if self.supersedes_record_digest:
            _validate_digest(
                self.supersedes_record_digest,
                "superseded grounding record",
            )
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))

    def to_dict(self) -> dict[str, object]:
        return {
            "disposition": self.disposition.value,
            "authority": self.authority.value,
            "verifier_id": self.verifier_id,
            "rationale": self.rationale,
            "evidence_status": self.evidence_status.value,
            "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
            "fact_source": self.fact_source,
            "supersedes_record_digest": self.supersedes_record_digest,
        }


@dataclass(frozen=True)
class GroundingRecord:
    """A candidate, its decision, and the fact materialized by that decision."""

    candidate: GroundingCandidate
    decision: GroundingDecision
    fact: Fact | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        expected_status = {
            GroundingDisposition.OBSERVED: FactStatus.OBSERVED,
            GroundingDisposition.PROPOSED: FactStatus.PROPOSED,
            GroundingDisposition.CONTRADICTED: FactStatus.CONTRADICTED,
        }.get(self.decision.disposition)
        if expected_status is None:
            if self.fact is not None:
                raise GroundingBoundaryError(
                    "a rejected grounding candidate cannot materialize a fact"
                )
        elif self.fact is None:
            raise GroundingBoundaryError(
                f"{self.decision.disposition.value} grounding requires a fact"
            )
        elif self.fact.atom != self.candidate.atom or self.fact.status is not expected_status:
            raise GroundingBoundaryError(
                "grounding fact does not match its candidate and disposition"
            )
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def record_digest(self) -> str:
        return grounding_payload_digest(
            json.dumps(
                self.to_dict(include_digest=False),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    @property
    def hard_negative(self) -> bool:
        return (
            self.decision.disposition
            in {
                GroundingDisposition.REJECTED,
                GroundingDisposition.CONTRADICTED,
            }
            and self.decision.authority
            in {
                GroundingAuthority.DETERMINISTIC_ADAPTER,
                GroundingAuthority.EXTERNAL_VERIFIER,
                GroundingAuthority.HUMAN_REVIEW,
            }
        )

    def to_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate": self.candidate.to_dict(),
            "decision": self.decision.to_dict(),
            "fact": None
            if self.fact is None
            else {
                "atom": str(self.fact.atom),
                "status": self.fact.status.value,
                "source": self.fact.source,
                "confidence": self.fact.confidence,
                "assertion_status": self.fact.assertion_status.value,
                "evidence_status": self.fact.evidence_status.value,
            },
            "diagnostics": list(self.diagnostics),
        }
        if include_digest:
            payload["record_digest"] = self.record_digest
        return payload


@dataclass(frozen=True)
class GroundingAudit:
    record_count: int
    materialized_count: int
    proof_eligible_count: int
    hard_negative_count: int
    covered_state_fact_count: int
    missing_record_fact_count: int
    uncovered_state_fact_count: int

    @property
    def valid(self) -> bool:
        return self.missing_record_fact_count == 0

    @property
    def coverage_complete(self) -> bool:
        return self.valid and self.uncovered_state_fact_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "record_count": self.record_count,
            "materialized_count": self.materialized_count,
            "proof_eligible_count": self.proof_eligible_count,
            "hard_negative_count": self.hard_negative_count,
            "covered_state_fact_count": self.covered_state_fact_count,
            "missing_record_fact_count": self.missing_record_fact_count,
            "uncovered_state_fact_count": self.uncovered_state_fact_count,
            "valid": self.valid,
            "coverage_complete": self.coverage_complete,
        }


@dataclass(frozen=True)
class GroundingLearningExample:
    """Verifier- or review-labeled candidate safe for grounding-policy training."""

    candidate: GroundingCandidate
    label: GroundingLabel
    authority: GroundingAuthority
    record_digest: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", GroundingLabel(self.label))
        object.__setattr__(self, "authority", GroundingAuthority(self.authority))
        if self.authority not in {
            GroundingAuthority.DETERMINISTIC_ADAPTER,
            GroundingAuthority.EXTERNAL_VERIFIER,
            GroundingAuthority.HUMAN_REVIEW,
        }:
            raise GroundingBoundaryError(
                "grounding learning labels require an independent verifier or human review"
            )
        _validate_digest(self.record_digest, "grounding learning record")
        if not 0.0 < self.weight <= 1.0:
            raise ValueError("grounding learning weight must be in (0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "label": self.label.value,
            "authority": self.authority.value,
            "record_digest": self.record_digest,
            "candidate_digest": self.candidate.candidate_digest,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class GroundingTrace:
    """Immutable grounding ledger carried beside a :class:`DomainInstance`."""

    records: tuple[GroundingRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        ids = [record.candidate.candidate_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise GroundingBoundaryError(
                "grounding trace contains duplicate candidate ids"
            )

    @property
    def facts(self) -> tuple[Fact, ...]:
        return tuple(
            record.fact for record in self.records if record.fact is not None
        )

    @property
    def hard_negatives(self) -> tuple[GroundingRecord, ...]:
        return tuple(record for record in self.records if record.hard_negative)

    @property
    def learning_examples(self) -> tuple[GroundingLearningExample, ...]:
        """Expose only independently decided labels, never raw model proposals."""

        trusted = {
            GroundingAuthority.DETERMINISTIC_ADAPTER,
            GroundingAuthority.EXTERNAL_VERIFIER,
            GroundingAuthority.HUMAN_REVIEW,
        }
        examples: list[GroundingLearningExample] = []
        for record in self.records:
            if record.decision.authority not in trusted:
                continue
            if record.decision.disposition is GroundingDisposition.OBSERVED:
                label = GroundingLabel.ACCEPT
            elif record.decision.disposition in {
                GroundingDisposition.REJECTED,
                GroundingDisposition.CONTRADICTED,
            }:
                label = GroundingLabel.REJECT
            else:
                continue
            examples.append(
                GroundingLearningExample(
                    record.candidate,
                    label,
                    record.decision.authority,
                    record.record_digest,
                    max(1e-9, record.decision.confidence),
                )
            )
        return tuple(examples)

    @property
    def trace_digest(self) -> str:
        return grounding_payload_digest(
            json.dumps(
                [record.record_digest for record in self.records],
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )

    def with_record(
        self,
        record: GroundingRecord,
        *,
        replace_existing: bool = False,
    ) -> GroundingTrace:
        existing = {
            item.candidate.candidate_id: index
            for index, item in enumerate(self.records)
        }
        index = existing.get(record.candidate.candidate_id)
        if index is None:
            return GroundingTrace(self.records + (record,))
        if not replace_existing:
            raise GroundingBoundaryError(
                f"grounding candidate already recorded: {record.candidate.candidate_id}"
            )
        revised = list(self.records)
        revised[index] = record
        return GroundingTrace(tuple(revised))

    def restrict_to_facts(self, state_facts: Iterable[Fact]) -> GroundingTrace:
        """Drop materialized records whose facts were removed from a derived case."""

        retained_facts = set(state_facts)
        return GroundingTrace(
            tuple(
                record
                for record in self.records
                if record.fact is None or record.fact in retained_facts
            )
        )

    def audit(self, state_facts: Iterable[Fact]) -> GroundingAudit:
        state = set(state_facts)
        materialized = set(self.facts)
        return GroundingAudit(
            record_count=len(self.records),
            materialized_count=len(materialized),
            proof_eligible_count=sum(
                fact.proof_eligible for fact in materialized
            ),
            hard_negative_count=len(self.hard_negatives),
            covered_state_fact_count=len(state & materialized),
            missing_record_fact_count=len(materialized - state),
            uncovered_state_fact_count=len(state - materialized),
        )

    def to_dict(self, *, include_records: bool = True) -> dict[str, object]:
        dispositions = Counter(
            record.decision.disposition.value for record in self.records
        )
        authorities = Counter(
            record.decision.authority.value for record in self.records
        )
        payload: dict[str, object] = {
            "trace_digest": self.trace_digest,
            "record_count": len(self.records),
            "hard_negative_count": len(self.hard_negatives),
            "learning_example_count": len(self.learning_examples),
            "dispositions": dict(sorted(dispositions.items())),
            "authorities": dict(sorted(authorities.items())),
        }
        if include_records:
            payload["records"] = [record.to_dict() for record in self.records]
        return payload


def grounding_payload_digest(payload: str | bytes) -> str:
    encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
    return sha256(encoded).hexdigest()


def make_grounding_candidate(
    *,
    domain: str,
    statement: str,
    atom: Atom,
    producer_id: str,
    source: str,
    input_digest: str = "",
    evidence: Iterable[str] = (),
    assertion_status: AssertionStatus = AssertionStatus.INFERRED,
    confidence: float = 1.0,
    candidate_id: str = "",
    sensor_features: Iterable[tuple[str, float]] = (),
) -> GroundingCandidate:
    normalized_features = tuple(
        sorted((str(name).strip(), float(value)) for name, value in sensor_features)
    )
    digest = input_digest or grounding_payload_digest(statement)
    identity = {
        "domain": domain.strip().lower(),
        "producer_id": producer_id,
        "source": source,
        "input_digest": digest,
        "atom": atom.canonical_key(),
        "statement": statement,
        "sensor_features": normalized_features,
    }
    stable_id = candidate_id or (
        "grounding:"
        + grounding_payload_digest(
            json.dumps(
                identity,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    )
    return GroundingCandidate(
        stable_id,
        domain,
        statement,
        atom,
        tuple(evidence),
        source,
        producer_id,
        digest,
        assertion_status,
        confidence,
        normalized_features,
    )


def decide_grounding(
    candidate: GroundingCandidate,
    decision: GroundingDecision,
) -> GroundingRecord:
    """Materialize a fact only when the authority/disposition pair allows it."""

    _validate_decision_boundary(candidate, decision)
    status = {
        GroundingDisposition.OBSERVED: FactStatus.OBSERVED,
        GroundingDisposition.PROPOSED: FactStatus.PROPOSED,
        GroundingDisposition.CONTRADICTED: FactStatus.CONTRADICTED,
    }.get(decision.disposition)
    fact = None
    if status is not None:
        fact = Fact(
            candidate.atom,
            status,
            source=decision.fact_source or candidate.source,
            confidence=min(candidate.confidence, decision.confidence),
            assertion_status=candidate.assertion_status,
            evidence_status=decision.evidence_status,
        )
    return GroundingRecord(candidate, decision, fact)


def make_grounding_record(
    *,
    domain: str,
    statement: str,
    atom: Atom,
    producer_id: str,
    source: str,
    disposition: GroundingDisposition,
    authority: GroundingAuthority,
    assertion_status: AssertionStatus,
    evidence_status: EvidenceStatus,
    rationale: str,
    input_digest: str = "",
    evidence: Iterable[str] = (),
    evidence_refs: Iterable[str] = (),
    confidence: float = 1.0,
    verifier_id: str = "",
    sensor_features: Iterable[tuple[str, float]] = (),
) -> GroundingRecord:
    candidate = make_grounding_candidate(
        domain=domain,
        statement=statement,
        atom=atom,
        producer_id=producer_id,
        source=source,
        input_digest=input_digest,
        evidence=evidence,
        assertion_status=assertion_status,
        confidence=confidence,
        sensor_features=sensor_features,
    )
    decision = GroundingDecision(
        disposition,
        authority,
        verifier_id or producer_id,
        rationale,
        evidence_status,
        confidence,
        tuple(evidence_refs),
        source,
    )
    return decide_grounding(candidate, decision)


def stage_grounding_proposal(
    candidate: GroundingCandidate,
    *,
    authority: GroundingAuthority,
    verifier_id: str,
    rationale: str,
    confidence: float,
    evidence_refs: Iterable[str] = (),
    source: str = "",
) -> GroundingRecord:
    if not GroundingAuthority(authority).proposal_only:
        raise GroundingBoundaryError(
            "staged grounding proposals require a proposal-only authority"
        )
    return decide_grounding(
        candidate,
        GroundingDecision(
            GroundingDisposition.PROPOSED,
            authority,
            verifier_id,
            rationale,
            EvidenceStatus.UNVERIFIED,
            confidence,
            tuple(evidence_refs),
            source,
        ),
    )


def promote_grounding_proposal(
    record: GroundingRecord,
    verifier: Callable[[Atom], bool],
    *,
    authority: GroundingAuthority = GroundingAuthority.DETERMINISTIC_ADAPTER,
    verifier_id: str = "deterministic_grounding_verifier",
    rationale: str = "independent verifier evaluated the typed grounding candidate",
    evidence_refs: Iterable[str] = (),
    source: str = "",
) -> GroundingRecord:
    """Verify a proposal, returning either an observed fact or a hard negative."""

    if record.decision.disposition is not GroundingDisposition.PROPOSED:
        raise GroundingBoundaryError("only a proposed grounding can be promoted")
    authority = GroundingAuthority(authority)
    if authority not in {
        GroundingAuthority.DETERMINISTIC_ADAPTER,
        GroundingAuthority.EXTERNAL_VERIFIER,
    }:
        raise GroundingBoundaryError(
            "proposal promotion requires an independent non-human verifier"
        )
    try:
        accepted = bool(verifier(record.candidate.atom))
    except Exception as exc:
        raise GroundingBoundaryError(
            f"grounding verifier failed: {type(exc).__name__}: {exc}"
        ) from exc
    disposition = (
        GroundingDisposition.OBSERVED
        if accepted
        else GroundingDisposition.REJECTED
    )
    evidence_status = (
        _verified_evidence_for(authority)
        if accepted
        else EvidenceStatus.UNVERIFIED
    )
    return decide_grounding(
        record.candidate,
        GroundingDecision(
            disposition,
            authority,
            verifier_id,
            rationale,
            evidence_status,
            1.0,
            tuple(evidence_refs),
            source,
            record.record_digest,
        ),
    )


def review_grounding_proposal(
    record: GroundingRecord,
    *,
    approved: bool,
    reviewer_id: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
    source: str = "human_grounding_review",
) -> GroundingRecord:
    """Apply an explicit human review without treating model confidence as truth."""

    if record.decision.disposition is not GroundingDisposition.PROPOSED:
        raise GroundingBoundaryError("human review requires a proposed grounding")
    if not reviewer_id.startswith("human:") or len(reviewer_id) <= len("human:"):
        raise GroundingBoundaryError(
            "human grounding reviewer id must use the 'human:' namespace"
        )
    return decide_grounding(
        record.candidate,
        GroundingDecision(
            (
                GroundingDisposition.OBSERVED
                if approved
                else GroundingDisposition.REJECTED
            ),
            GroundingAuthority.HUMAN_REVIEW,
            reviewer_id,
            rationale,
            (
                EvidenceStatus.EXTERNAL_VERIFIED
                if approved
                else EvidenceStatus.UNVERIFIED
            ),
            1.0,
            tuple(evidence_refs),
            source,
            record.record_digest,
        ),
    )


def remap_grounding_record(
    record: GroundingRecord,
    atom_mapper: Callable[[Atom], Atom],
    *,
    candidate_id_prefix: str = "",
) -> GroundingRecord:
    """Map a grounding record into a composed registry without losing provenance."""

    mapped_atom = atom_mapper(record.candidate.atom)
    candidate = replace(
        record.candidate,
        candidate_id=(
            f"{candidate_id_prefix}:{record.candidate.candidate_id}"
            if candidate_id_prefix
            else record.candidate.candidate_id
        ),
        atom=mapped_atom,
    )
    fact = None
    if record.fact is not None:
        fact = Fact(
            mapped_atom,
            record.fact.status,
            record.fact.source,
            record.fact.confidence,
            assertion_status=record.fact.assertion_status,
            evidence_status=record.fact.evidence_status,
        )
    return GroundingRecord(candidate, record.decision, fact, record.diagnostics)


def _validate_decision_boundary(
    candidate: GroundingCandidate,
    decision: GroundingDecision,
) -> None:
    if decision.disposition is GroundingDisposition.OBSERVED:
        if decision.authority.proposal_only:
            raise GroundingBoundaryError(
                f"{decision.authority.value} cannot create observed facts"
            )
        if not candidate.atom.predicate.verified:
            raise GroundingBoundaryError(
                "an unverified predicate cannot become an observed fact"
            )
        expected = _verified_evidence_for(decision.authority)
        if expected is not None and decision.evidence_status is not expected:
            raise GroundingBoundaryError(
                f"{decision.authority.value} observations require "
                f"{expected.value} evidence"
            )
        if (
            decision.authority is GroundingAuthority.HUMAN_REVIEW
            and not decision.verifier_id.startswith("human:")
        ):
            raise GroundingBoundaryError(
                "human-reviewed observations require a 'human:' verifier id"
            )
    elif decision.disposition is GroundingDisposition.PROPOSED:
        if decision.evidence_status is not EvidenceStatus.UNVERIFIED:
            raise GroundingBoundaryError(
                "a proposed grounding must remain evidence-unverified"
            )
    elif decision.disposition is GroundingDisposition.REJECTED:
        if decision.evidence_status in {
            EvidenceStatus.ASSUMED,
            EvidenceStatus.DERIVED,
        }:
            raise GroundingBoundaryError(
                "a rejected grounding cannot carry proof evidence status"
            )


def _verified_evidence_for(
    authority: GroundingAuthority,
) -> EvidenceStatus | None:
    if authority is GroundingAuthority.DETERMINISTIC_ADAPTER:
        return EvidenceStatus.ADAPTER_VERIFIED
    if authority in {
        GroundingAuthority.EXTERNAL_VERIFIER,
        GroundingAuthority.HUMAN_REVIEW,
    }:
        return EvidenceStatus.EXTERNAL_VERIFIED
    return None


def _validate_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError(f"{label} digest must be a 64-character SHA-256 hex value")


__all__ = [
    "GroundingAudit",
    "GroundingAuthority",
    "GroundingBoundaryError",
    "GroundingCandidate",
    "GroundingDecision",
    "GroundingDisposition",
    "GroundingLabel",
    "GroundingLearningExample",
    "GroundingRecord",
    "GroundingTrace",
    "decide_grounding",
    "grounding_payload_digest",
    "make_grounding_candidate",
    "make_grounding_record",
    "promote_grounding_proposal",
    "remap_grounding_record",
    "review_grounding_proposal",
    "stage_grounding_proposal",
]

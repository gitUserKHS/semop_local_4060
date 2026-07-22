from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Mapping, Sequence

from .semantic_distillation import (
    DistillationDecision,
    SemanticDistillationCorpus,
    SemanticDistillationRecord,
)
from .semantic_student_evaluation import (
    SemanticRuntimeObservation,
    semantic_completion_signature,
)
from .verified_semantic_curriculum import GENERATOR_VERSION


@dataclass(frozen=True)
class SemanticDevelopmentOutcome:
    record_id: str
    template_id: str
    operation: str
    semantic_match: bool
    repeated_hard_negative: bool
    model_used: bool
    proof_eligible: bool
    replay_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticDevelopmentMetrics:
    model_id: str
    split: str
    positive_records: int
    semantic_matches: int
    false_acceptances: int
    model_coverage: float
    replay_integrity: float
    median_seconds: float
    peak_vram_bytes: int | None
    matches_by_operation: Mapping[str, tuple[int, int]]
    outcomes: tuple[SemanticDevelopmentOutcome, ...]

    @property
    def completion_rate(self) -> float:
        return self.semantic_matches / self.positive_records

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "split": self.split,
            "promotion_eligible": False,
            "positive_records": self.positive_records,
            "semantic_matches": self.semantic_matches,
            "completion_rate": self.completion_rate,
            "false_acceptances": self.false_acceptances,
            "model_coverage": self.model_coverage,
            "replay_integrity": self.replay_integrity,
            "median_seconds": self.median_seconds,
            "peak_vram_bytes": self.peak_vram_bytes,
            "matches_by_operation": {
                operation: {"matched": counts[0], "total": counts[1]}
                for operation, counts in self.matches_by_operation.items()
            },
            "outcomes": [item.to_dict() for item in self.outcomes],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticDevelopmentMetrics":
        if value.get("promotion_eligible") is not False:
            raise ValueError("development metrics were not marked non-promotional")
        raw_operations = value.get("matches_by_operation")
        raw_outcomes = value.get("outcomes")
        if not isinstance(raw_operations, Mapping) or not isinstance(raw_outcomes, list):
            raise ValueError("development metrics have an invalid structure")
        operations: dict[str, tuple[int, int]] = {}
        for operation, counts in raw_operations.items():
            if not isinstance(counts, Mapping):
                raise ValueError("development operation metrics must be objects")
            operations[str(operation)] = (
                int(counts["matched"]),
                int(counts["total"]),
            )
        return cls(
            model_id=str(value["model_id"]),
            split=str(value["split"]),
            positive_records=int(value["positive_records"]),
            semantic_matches=int(value["semantic_matches"]),
            false_acceptances=int(value["false_acceptances"]),
            model_coverage=float(value["model_coverage"]),
            replay_integrity=float(value["replay_integrity"]),
            median_seconds=float(value["median_seconds"]),
            peak_vram_bytes=(
                None
                if value.get("peak_vram_bytes") is None
                else int(value["peak_vram_bytes"])
            ),
            matches_by_operation=operations,
            outcomes=tuple(SemanticDevelopmentOutcome(**item) for item in raw_outcomes),
        )


def evaluate_generated_semantic_model(
    corpus: SemanticDistillationCorpus,
    observations: Sequence[SemanticRuntimeObservation],
    *,
    model_id: str,
) -> SemanticDevelopmentMetrics:
    """Measure generator-known semantics without creating promotion evidence."""

    corpus.validate_budget()
    splits = {record.data_split for record in corpus.records}
    if len(splits) != 1:
        raise ValueError("development corpus mixes semantic splits")
    split = next(iter(splits))
    if split not in {"validation", "sealed"}:
        raise ValueError("development evaluation accepts validation or sealed only")
    corpus.require_split(split)
    if any(not record.source.startswith(GENERATOR_VERSION + ":") for record in corpus.records):
        raise ValueError("development corpus lacks generator-known semantic authority")

    positives = tuple(
        record
        for record in corpus.records
        if record.decision is not DistillationDecision.HARD_NEGATIVE
    )
    if not positives:
        raise ValueError("development corpus requires positive records")
    positive_by_prompt = {record.prompt: record for record in positives}
    if len(positive_by_prompt) != len(positives):
        raise ValueError("development corpus has duplicate positive prompts")
    negative_signatures: dict[str, set[str]] = {prompt: set() for prompt in positive_by_prompt}
    for record in corpus.records:
        if record.decision is DistillationDecision.HARD_NEGATIVE:
            if record.prompt not in positive_by_prompt:
                raise ValueError("hard negative has no paired positive prompt")
            negative_signatures[record.prompt].add(
                semantic_completion_signature(record.completion)
            )
    if any(not values for values in negative_signatures.values()):
        raise ValueError("each development prompt requires a hard negative")

    observed: dict[str, SemanticRuntimeObservation] = {}
    for observation in observations:
        if observation.record_id in observed:
            raise ValueError("duplicate development model observation")
        observed[observation.record_id] = observation
    expected_ids = {record.record_id for record in positives}
    if set(observed) != expected_ids:
        raise ValueError("development observations do not match positive records")
    if any(item.model_used and item.model_id != model_id for item in observed.values()):
        raise ValueError("development observations do not match the declared model")

    outcomes: list[SemanticDevelopmentOutcome] = []
    by_operation: dict[str, list[int]] = {}
    semantic_matches = 0
    false_acceptances = 0
    proof_eligible = 0
    replayed = 0
    for record in positives:
        observation = observed[record.record_id]
        expected = semantic_completion_signature(record.completion)
        actual = semantic_completion_signature(observation.completion_json)
        match = bool(
            observation.model_used
            and observation.proof_eligible
            and observation.replay_verified
            and actual == expected
        )
        repeated_negative = bool(
            observation.model_used
            and observation.proof_eligible
            and observation.replay_verified
            and actual in negative_signatures[record.prompt]
        )
        semantic_matches += int(match)
        false_acceptances += int(repeated_negative)
        if observation.proof_eligible:
            proof_eligible += 1
            replayed += int(observation.replay_verified)
        template_id = record.source.split(":", 2)[1]
        operation = _operation(record)
        counts = by_operation.setdefault(operation, [0, 0])
        counts[0] += int(match)
        counts[1] += 1
        outcomes.append(
            SemanticDevelopmentOutcome(
                record_id=record.record_id,
                template_id=template_id,
                operation=operation,
                semantic_match=match,
                repeated_hard_negative=repeated_negative,
                model_used=observation.model_used,
                proof_eligible=observation.proof_eligible,
                replay_verified=observation.replay_verified,
            )
        )
    latencies = tuple(item.elapsed_seconds for item in observed.values())
    vram = tuple(
        item.peak_vram_bytes
        for item in observed.values()
        if item.peak_vram_bytes is not None
    )
    return SemanticDevelopmentMetrics(
        model_id=model_id,
        split=split,
        positive_records=len(positives),
        semantic_matches=semantic_matches,
        false_acceptances=false_acceptances,
        model_coverage=sum(item.model_used for item in observed.values()) / len(positives),
        replay_integrity=replayed / proof_eligible if proof_eligible else 0.0,
        median_seconds=float(median(latencies)),
        peak_vram_bytes=max(vram) if vram else None,
        matches_by_operation={
            operation: (counts[0], counts[1])
            for operation, counts in sorted(by_operation.items())
        },
        outcomes=tuple(outcomes),
    )


def development_candidate_passes(
    baseline: SemanticDevelopmentMetrics,
    candidate: SemanticDevelopmentMetrics,
    *,
    observed_probe_match: bool,
) -> bool:
    if baseline.split != candidate.split:
        raise ValueError("development comparison mixes splits")
    return bool(
        candidate.completion_rate > baseline.completion_rate
        and candidate.false_acceptances == 0
        and candidate.model_coverage == 1.0
        and candidate.replay_integrity == 1.0
        and observed_probe_match
    )


def _operation(record: SemanticDistillationRecord) -> str:
    completion = _completion_object(record.completion)
    expression = str(completion["payload"]["expression"])
    if "*" in expression:
        return "multiply"
    if "-" in expression:
        return "subtract"
    return "add"


def _completion_object(value: str) -> dict[str, Any]:
    import json

    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("semantic completion must be an object")
    return decoded

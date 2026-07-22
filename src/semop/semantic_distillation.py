from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .prompt_api import AnswerEnvelope, AnswerStatus, PromptRequest
from .semantic_models import QWEN_BALANCED, QWEN_ECONOMY


class DistillationDecision(str, Enum):
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    HARD_NEGATIVE = "hard_negative"


@dataclass(frozen=True)
class SemanticDistillationConfig:
    teacher_model_id: str = QWEN_BALANCED.model_id
    student_model_id: str = QWEN_ECONOMY.model_id
    max_sequence_length: int = 2_048
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    epochs: int = 2
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bf16: bool = True
    max_records_per_domain: int = 5_000
    hard_negatives_per_positive: int = 4

    def __post_init__(self) -> None:
        integer_values = (
            self.max_sequence_length,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.epochs,
            self.lora_rank,
            self.lora_alpha,
            self.max_records_per_domain,
            self.hard_negatives_per_positive,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("semantic distillation integer limits must be positive")
        if self.learning_rate <= 0 or not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("invalid semantic distillation optimizer settings")


@dataclass(frozen=True)
class SemanticDistillationRecord:
    record_id: str
    domain: str
    prompt: str
    completion: str
    decision: DistillationDecision | str
    proposal_fingerprint: str
    replay_verified: bool
    semantic_review_digest: str = ""
    data_split: str = "train"
    rejection_reason: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        decision = DistillationDecision(self.decision)
        if not self.prompt.strip() or not self.completion.strip():
            raise ValueError("distillation prompt and completion cannot be empty")
        try:
            completion = json.loads(self.completion)
        except json.JSONDecodeError as exc:
            raise ValueError("distillation completion must be valid JSON") from exc
        if not isinstance(completion, dict):
            raise ValueError("distillation completion must be a JSON object")
        if not _is_digest(self.proposal_fingerprint):
            raise ValueError("distillation proposal fingerprint must be SHA-256")
        if decision in {DistillationDecision.ACCEPTED, DistillationDecision.CORRECTED}:
            if not self.replay_verified:
                raise ValueError("positive distillation records require proof replay")
        if decision is DistillationDecision.HARD_NEGATIVE and not self.rejection_reason.strip():
            raise ValueError("hard negatives require a rejection reason")
        if self.semantic_review_digest and not _is_digest(self.semantic_review_digest):
            raise ValueError("semantic review digest must be SHA-256")
        data_split = self.data_split.strip().lower()
        if data_split not in {"train", "validation", "sealed"}:
            raise ValueError("semantic distillation split must be train, validation, or sealed")
        expected_id = _record_digest(
            self.domain,
            self.prompt,
            self.completion,
            decision.value,
            self.proposal_fingerprint,
            self.semantic_review_digest,
            data_split,
        )
        if self.record_id and self.record_id != expected_id:
            raise ValueError("distillation record id does not match its content")
        object.__setattr__(self, "record_id", expected_id)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "data_split", data_split)

    @classmethod
    def from_answer(
        cls,
        request: PromptRequest,
        answer: AnswerEnvelope,
        completion: Mapping[str, Any],
        *,
        decision: DistillationDecision | str = DistillationDecision.ACCEPTED,
        semantic_review_digest: str = "",
        data_split: str = "train",
        rejection_reason: str = "",
        source: str = "verified_runtime",
    ) -> "SemanticDistillationRecord":
        if not answer.proposals:
            raise ValueError("distillation source answer has no semantic proposal")
        proposal = answer.proposals[0]
        if answer.status is AnswerStatus.BEST_EFFORT and not semantic_review_digest:
            raise ValueError(
                "model-grounded traces require an independent semantic review digest"
            )
        replay_verified = bool(answer.provenance.get("logical_replay_verified"))
        serialized = json.dumps(
            dict(completion),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return cls(
            record_id="",
            domain=proposal.domain,
            prompt=request.text,
            completion=serialized,
            decision=decision,
            proposal_fingerprint=proposal.fingerprint,
            replay_verified=replay_verified,
            semantic_review_digest=semantic_review_digest,
            data_split=data_split,
            rejection_reason=rejection_reason,
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        return payload


@dataclass(frozen=True)
class SemanticDistillationCorpus:
    records: tuple[SemanticDistillationRecord, ...]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        ids = tuple(record.record_id for record in records)
        if len(ids) != len(set(ids)):
            raise ValueError("distillation corpus contains duplicate records")
        object.__setattr__(self, "records", records)

    @property
    def positives(self) -> tuple[SemanticDistillationRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.decision is not DistillationDecision.HARD_NEGATIVE
        )

    def validate_budget(
        self,
        config: SemanticDistillationConfig | None = None,
    ) -> None:
        active = config or SemanticDistillationConfig()
        domains = {record.domain for record in self.records}
        for domain in domains:
            records = tuple(record for record in self.records if record.domain == domain)
            positives = tuple(
                record
                for record in records
                if record.decision is not DistillationDecision.HARD_NEGATIVE
            )
            negatives = len(records) - len(positives)
            if len(records) > active.max_records_per_domain:
                raise ValueError(f"{domain} exceeds the 5,000-record domain cap")
            if negatives > len(positives) * active.hard_negatives_per_positive:
                raise ValueError(f"{domain} exceeds the hard-negative ratio")

    def require_split(self, expected: str) -> None:
        normalized = expected.strip().lower()
        if normalized not in {"train", "validation", "sealed"}:
            raise ValueError("unknown semantic distillation split")
        mismatched = tuple(
            record.record_id
            for record in self.records
            if record.data_split != normalized
        )
        if mismatched:
            raise ValueError(
                f"semantic corpus contains records outside the {normalized} split"
            )

    def save_jsonl(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(
                    json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
        temporary.replace(destination)
        return destination

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "SemanticDistillationCorpus":
        records: list[SemanticDistillationRecord] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    records.append(SemanticDistillationRecord(**value))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid distillation record at line {line_number}: {exc}"
                    ) from exc
        corpus = cls(tuple(records))
        corpus.validate_budget()
        return corpus


def build_training_messages(
    record: SemanticDistillationRecord,
) -> tuple[dict[str, str], ...]:
    system = (
        "Return one SemOp semantic proposal JSON object. The proposal is untrusted "
        "until typed validation and proof replay. Do not invent observed facts."
    )
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": record.prompt},
        {"role": "assistant", "content": record.completion},
    )


def _record_digest(*values: str) -> str:
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence
import unicodedata

from .kernel.semantic_codec import canonical_json, encode_semantic_request
from .prompt_api import AnswerEnvelope, PromptRequest, SemanticProposal
from .semantic_distillation import (
    DistillationDecision,
    SemanticDistillationCorpus,
    SemanticDistillationRecord,
)


SEMANTIC_REVIEW_ATTESTATION = (
    "I reviewed this exact prompt, typed proposal, and replay result."
)
SEMANTIC_MEDIA_MAX_BYTES = 16 * 1024 * 1024
SEMANTIC_REVIEWABLE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png"})


class SemanticTraceDecision(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class SemanticTraceSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    SEALED = "sealed"


@dataclass(frozen=True)
class SemanticTraceRecord:
    trace_id: str
    request_json: str
    completion_json: str
    domain: str
    proposal_fingerprint: str
    model_id: str
    answer_status: str
    replay_verified: bool
    created_at: str
    media_manifest_json: str = "[]"

    def __post_init__(self) -> None:
        request = _json_object(self.request_json, "semantic trace request")
        completion = _json_object(self.completion_json, "semantic trace completion")
        media_manifest = _json_array(
            self.media_manifest_json,
            "semantic trace media manifest",
        )
        if not str(request.get("text", "")).strip() and not request.get("images"):
            raise ValueError("semantic trace request is empty")
        if str(completion.get("domain", "")).strip().lower() != self.domain:
            raise ValueError("semantic trace completion domain mismatch")
        _require_digest(self.proposal_fingerprint, "semantic proposal")
        if type(self.replay_verified) is not bool:
            raise TypeError("semantic trace replay flag must be boolean")
        request_images = request.get("images", ())
        if not isinstance(request_images, list):
            raise ValueError("semantic trace request images must be a list")
        seen_indices: set[int] = set()
        for item in media_manifest:
            if not isinstance(item, Mapping):
                raise ValueError("semantic trace media manifest item must be an object")
            image_index = item.get("image_index")
            if type(image_index) is not int or not 0 <= image_index < len(request_images):
                raise ValueError("semantic trace media index is out of range")
            if image_index in seen_indices:
                raise ValueError("semantic trace media index is duplicated")
            seen_indices.add(image_index)
            _require_digest(str(item.get("sha256", "")), "semantic trace media")
            size = item.get("size")
            if type(size) is not int or not 0 < size <= SEMANTIC_MEDIA_MAX_BYTES:
                raise ValueError("semantic trace media size is invalid")
            if not str(item.get("mime_type", "")).strip():
                raise ValueError("semantic trace media MIME type is missing")
        expected = _trace_id(
            self.request_json,
            self.completion_json,
            self.proposal_fingerprint,
            self.model_id,
            self.answer_status,
            self.replay_verified,
            self.media_manifest_json,
        )
        if self.trace_id != expected:
            raise ValueError("semantic trace id does not match its content")
        _parse_timestamp(self.created_at)

    @property
    def prompt(self) -> str:
        return str(json.loads(self.request_json).get("text", ""))

    @property
    def has_images(self) -> bool:
        return bool(json.loads(self.request_json).get("images"))

    @property
    def split(self) -> SemanticTraceSplit:
        return semantic_request_split(
            self.request_json,
            self.media_manifest_json,
        )

    @property
    def media_manifest(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(_json_array(self.media_manifest_json, "semantic trace media manifest"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticTraceReview:
    review_digest: str
    trace_id: str
    decision: SemanticTraceDecision | str
    reviewer: str
    attestation: str
    note: str
    reviewed_at: str

    def __post_init__(self) -> None:
        decision = SemanticTraceDecision(self.decision)
        _require_digest(self.trace_id, "semantic trace")
        if not self.reviewer.startswith("human:") or not self.reviewer[6:].strip():
            raise ValueError("semantic review requires a human: reviewer id")
        if self.attestation != SEMANTIC_REVIEW_ATTESTATION:
            raise ValueError("semantic review attestation does not match")
        _parse_timestamp(self.reviewed_at)
        expected = _review_digest(
            self.trace_id,
            decision.value,
            self.reviewer,
            self.attestation,
            self.note,
            self.reviewed_at,
        )
        if self.review_digest != expected:
            raise ValueError("semantic review digest does not match its content")
        object.__setattr__(self, "decision", decision)

    @property
    def accepted(self) -> bool:
        return self.decision is SemanticTraceDecision.ACCEPTED

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        return payload


@dataclass(frozen=True)
class SemanticTraceMedia:
    trace_id: str
    image_index: int
    mime_type: str
    content_sha256: str
    content: bytes

    def __post_init__(self) -> None:
        _require_digest(self.trace_id, "semantic trace")
        if type(self.image_index) is not int or self.image_index < 0:
            raise ValueError("semantic trace media index cannot be negative")
        if not self.mime_type.strip():
            raise ValueError("semantic trace media MIME type is missing")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("semantic trace media content is empty")
        if len(self.content) > SEMANTIC_MEDIA_MAX_BYTES:
            raise ValueError("semantic trace media exceeds the local size limit")
        _require_digest(self.content_sha256, "semantic trace media")
        if sha256(self.content).hexdigest() != self.content_sha256:
            raise ValueError("semantic trace media digest does not match its content")

    @property
    def size(self) -> int:
        return len(self.content)

    def manifest(self) -> dict[str, Any]:
        return {
            "image_index": self.image_index,
            "mime_type": self.mime_type,
            "sha256": self.content_sha256,
            "size": self.size,
        }


@dataclass(frozen=True)
class SemanticTraceStats:
    total: int
    pending: int
    accepted: int
    rejected: int
    exportable: int
    train: int
    validation: int
    sealed: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class SemanticTraceStore:
    """Local digest-bound journal for model semantics and human review."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS semantic_trace_records (
                    trace_id TEXT PRIMARY KEY,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS semantic_trace_reviews (
                    review_digest TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    review_json TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_semantic_review_trace
                    ON semantic_trace_reviews(trace_id, reviewed_at);
                CREATE TABLE IF NOT EXISTS semantic_trace_media (
                    trace_id TEXT NOT NULL,
                    image_index INTEGER NOT NULL,
                    mime_type TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    content BLOB NOT NULL,
                    PRIMARY KEY(trace_id, image_index)
                );
                """
            )

    def capture(
        self,
        request: PromptRequest,
        answer: AnswerEnvelope,
    ) -> SemanticTraceRecord | None:
        if not answer.proposals:
            return None
        proposal = answer.proposals[0]
        if proposal.deterministic:
            return None
        prepared_media = _prepare_semantic_media(request.images)
        media_manifest_json = canonical_json(
            [item[0] for item in prepared_media]
        )
        request_json = canonical_json(request.to_dict())
        completion_json = canonical_json(
            canonical_completion_from_proposal(request, proposal)
        )
        replay_verified = bool(answer.provenance.get("logical_replay_verified"))
        trace_id = _trace_id(
            request_json,
            completion_json,
            proposal.fingerprint,
            proposal.model_id,
            answer.status.value,
            replay_verified,
            media_manifest_json,
        )
        record = SemanticTraceRecord(
            trace_id=trace_id,
            request_json=request_json,
            completion_json=completion_json,
            domain=proposal.domain,
            proposal_fingerprint=proposal.fingerprint,
            model_id=proposal.model_id,
            answer_status=answer.status.value,
            replay_verified=replay_verified,
            created_at=_timestamp(),
            media_manifest_json=media_manifest_json,
        )
        serialized = canonical_json(record.to_dict())
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT record_json FROM semantic_trace_records WHERE trace_id = ?",
                (record.trace_id,),
            ).fetchone()
            if existing is not None:
                stored = SemanticTraceRecord(**json.loads(existing["record_json"]))
                return stored
            connection.execute(
                "INSERT INTO semantic_trace_records(trace_id, record_json, created_at) "
                "VALUES (?, ?, ?)",
                (record.trace_id, serialized, record.created_at),
            )
            for manifest, content in prepared_media:
                connection.execute(
                    "INSERT INTO semantic_trace_media("
                    "trace_id, image_index, mime_type, content_sha256, content"
                    ") VALUES (?, ?, ?, ?, ?)",
                    (
                        record.trace_id,
                        manifest["image_index"],
                        manifest["mime_type"],
                        manifest["sha256"],
                        content,
                    ),
                )
        return record

    def get(self, trace_id: str) -> SemanticTraceRecord:
        _require_digest(trace_id, "semantic trace")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT record_json FROM semantic_trace_records WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
        if row is None:
            raise KeyError(trace_id)
        return SemanticTraceRecord(**json.loads(row["record_json"]))

    def media(self, trace_id: str) -> tuple[SemanticTraceMedia, ...]:
        trace = self.get(trace_id)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT image_index, mime_type, content_sha256, content "
                "FROM semantic_trace_media WHERE trace_id = ? ORDER BY image_index",
                (trace.trace_id,),
            ).fetchall()
        media = tuple(
            SemanticTraceMedia(
                trace_id=trace.trace_id,
                image_index=int(row["image_index"]),
                mime_type=str(row["mime_type"]),
                content_sha256=str(row["content_sha256"]),
                content=bytes(row["content"]),
            )
            for row in rows
        )
        if tuple(item.manifest() for item in media) != trace.media_manifest:
            raise ValueError("semantic trace media does not match its bound manifest")
        return media

    def media_item(self, trace_id: str, image_index: int) -> SemanticTraceMedia:
        if type(image_index) is not int or image_index < 0:
            raise ValueError("semantic trace media index cannot be negative")
        for item in self.media(trace_id):
            if item.image_index == image_index:
                return item
        raise KeyError((trace_id, image_index))

    def media_is_reviewable(self, trace: SemanticTraceRecord | str) -> bool:
        record = self.get(trace) if isinstance(trace, str) else trace
        if not record.has_images:
            return False
        request = _json_object(record.request_json, "semantic trace request")
        images = request.get("images", ())
        media = self.media(record.trace_id)
        return (
            isinstance(images, list)
            and len(media) == len(images)
            and all(
                item.mime_type in SEMANTIC_REVIEWABLE_MEDIA_TYPES
                for item in media
            )
        )

    def review(
        self,
        trace_id: str,
        *,
        accepted: bool,
        reviewer: str,
        attest_human_review: bool,
        note: str = "",
        reviewed_at: str | None = None,
    ) -> SemanticTraceReview:
        if type(accepted) is not bool:
            raise TypeError("semantic review decision must be boolean")
        if attest_human_review is not True:
            raise ValueError("semantic review requires explicit human attestation")
        trace = self.get(trace_id)
        if trace.has_images and not self.media_is_reviewable(trace):
            raise ValueError(
                "image semantic review requires exact locally persisted preview evidence"
            )
        if accepted and not trace.replay_verified and not trace.has_images:
            raise ValueError("accepted semantics require a replay-verified typed result")
        timestamp = reviewed_at or _timestamp()
        decision = (
            SemanticTraceDecision.ACCEPTED
            if accepted
            else SemanticTraceDecision.REJECTED
        )
        digest = _review_digest(
            trace.trace_id,
            decision.value,
            reviewer,
            SEMANTIC_REVIEW_ATTESTATION,
            str(note).strip(),
            timestamp,
        )
        review = SemanticTraceReview(
            review_digest=digest,
            trace_id=trace.trace_id,
            decision=decision,
            reviewer=reviewer,
            attestation=SEMANTIC_REVIEW_ATTESTATION,
            note=str(note).strip(),
            reviewed_at=timestamp,
        )
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO semantic_trace_reviews("
                "review_digest, trace_id, review_json, reviewed_at) VALUES (?, ?, ?, ?)",
                (
                    review.review_digest,
                    review.trace_id,
                    canonical_json(review.to_dict()),
                    review.reviewed_at,
                ),
            )
        return review

    def latest_reviews(self) -> dict[str, SemanticTraceReview]:
        latest: dict[str, SemanticTraceReview] = {}
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT review_json FROM semantic_trace_reviews "
                "ORDER BY reviewed_at, rowid"
            ).fetchall()
        for row in rows:
            review = SemanticTraceReview(**json.loads(row["review_json"]))
            latest[review.trace_id] = review
        return latest

    def pending(
        self,
        *,
        limit: int = 20,
        include_images: bool = True,
    ) -> tuple[SemanticTraceRecord, ...]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("semantic pending limit must be between 1 and 100")
        if type(include_images) is not bool:
            raise TypeError("semantic pending image flag must be boolean")
        records: list[SemanticTraceRecord] = []
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT record_json FROM semantic_trace_records AS records "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM semantic_trace_reviews AS reviews "
                "WHERE reviews.trace_id = records.trace_id"
                ") ORDER BY records.created_at DESC, records.rowid DESC"
            ).fetchall()
        for row in rows:
            record = SemanticTraceRecord(**json.loads(row["record_json"]))
            if record.has_images and not include_images:
                continue
            records.append(record)
            if len(records) >= limit:
                break
        return tuple(records)

    def stats(self) -> SemanticTraceStats:
        with self._connection() as connection:
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM semantic_trace_records"
                ).fetchone()[0]
            )
        reviews = self.latest_reviews()
        accepted = sum(review.accepted for review in reviews.values())
        rejected = len(reviews) - accepted
        exportable = 0
        split_counts = {split: 0 for split in SemanticTraceSplit}
        for trace_id, review in reviews.items():
            trace = self.get(trace_id)
            if not trace.has_images and (not review.accepted or trace.replay_verified):
                exportable += 1
                split_counts[trace.split] += 1
        return SemanticTraceStats(
            total=total,
            pending=total - len(reviews),
            accepted=accepted,
            rejected=rejected,
            exportable=exportable,
            train=split_counts[SemanticTraceSplit.TRAIN],
            validation=split_counts[SemanticTraceSplit.VALIDATION],
            sealed=split_counts[SemanticTraceSplit.SEALED],
        )

    def export_distillation(
        self,
        path: str | Path,
        *,
        splits: Sequence[SemanticTraceSplit | str] = (SemanticTraceSplit.TRAIN,),
    ) -> SemanticDistillationCorpus:
        corpus = self.distillation_corpus(splits=splits)
        corpus.save_jsonl(path)
        return corpus

    def distillation_corpus(
        self,
        *,
        splits: Sequence[SemanticTraceSplit | str] = (SemanticTraceSplit.TRAIN,),
    ) -> SemanticDistillationCorpus:
        """Build an in-memory reviewed text corpus without writing an artifact."""

        selected_splits = frozenset(SemanticTraceSplit(split) for split in splits)
        if not selected_splits:
            raise ValueError("semantic export requires at least one split")
        records: list[SemanticDistillationRecord] = []
        for trace_id, review in sorted(self.latest_reviews().items()):
            trace = self.get(trace_id)
            # Text-only LoRA must not silently train without the reviewed pixels.
            if trace.has_images:
                continue
            if trace.split not in selected_splits:
                continue
            decision = (
                DistillationDecision.ACCEPTED
                if review.accepted
                else DistillationDecision.HARD_NEGATIVE
            )
            records.append(
                SemanticDistillationRecord(
                    record_id="",
                    domain=trace.domain,
                    prompt=trace.prompt,
                    completion=trace.completion_json,
                    decision=decision,
                    proposal_fingerprint=trace.proposal_fingerprint,
                    replay_verified=trace.replay_verified,
                    semantic_review_digest=review.review_digest,
                    data_split=trace.split.value,
                    rejection_reason=(
                        "" if review.accepted else review.note or "human rejected semantics"
                    ),
                    source="local_semantic_trace_review",
                )
            )
        corpus = SemanticDistillationCorpus(tuple(records))
        corpus.validate_budget()
        return corpus


def canonical_completion_from_proposal(
    prompt: PromptRequest,
    proposal: SemanticProposal,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if proposal.request is not None:
        if proposal.domain == "composed":
            payload = {"controlled_text": str(proposal.request.payload.text)}
        else:
            encoded = encode_semantic_request(proposal.request)
            raw_payload = dict(encoded["payload"])
            if proposal.domain == "language":
                payload = {"controlled_text": str(raw_payload["text"])}
            elif proposal.domain == "vision":
                payload = {"question": str(raw_payload.get("query") or prompt.text)}
            else:
                payload = raw_payload
    completion: dict[str, Any] = {
        "domain": proposal.domain,
        "confidence": proposal.confidence,
        "operator_program": [item.value for item in proposal.operator_program],
        "payload": payload,
    }
    if proposal.candidate_answer:
        completion["answer"] = proposal.candidate_answer
    return completion


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or export SemOp's local reviewed semantic traces."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/experience/beginner-experience.db"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    export = commands.add_parser("export")
    export.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/distillation/semantic-reviewed.jsonl"),
    )
    export.add_argument(
        "--split",
        action="append",
        choices=[item.value for item in SemanticTraceSplit],
        default=None,
        help="split to export; repeat for multiple splits (default: train)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        store = SemanticTraceStore(args.db)
        if args.command == "status":
            payload: Mapping[str, Any] = store.stats().to_dict()
        else:
            corpus = store.export_distillation(
                args.output,
                splits=args.split or (SemanticTraceSplit.TRAIN,),
            )
            payload = {
                "output": str(args.output.resolve()),
                "records": len(corpus.records),
                "positives": len(corpus.positives),
                "stats": store.stats().to_dict(),
            }
    except (KeyError, OSError, sqlite3.Error, TypeError, ValueError) as exc:
        print(f"SemOp semantic experience error: {exc}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _trace_id(
    request_json: str,
    completion_json: str,
    proposal_fingerprint: str,
    model_id: str,
    answer_status: str,
    replay_verified: bool,
    media_manifest_json: str = "[]",
) -> str:
    values = [
        request_json,
        completion_json,
        proposal_fingerprint,
        model_id,
        answer_status,
        str(replay_verified),
    ]
    # Empty manifests retain the legacy text-trace identity.
    if media_manifest_json != "[]":
        values.append(media_manifest_json)
    return _digest(*values)


def semantic_trace_split(trace_id: str) -> SemanticTraceSplit:
    _require_digest(trace_id, "semantic trace")
    bucket = int(trace_id[:2], 16)
    if bucket < 205:
        return SemanticTraceSplit.TRAIN
    if bucket < 230:
        return SemanticTraceSplit.VALIDATION
    return SemanticTraceSplit.SEALED


def semantic_request_split(
    request_json: str,
    media_manifest_json: str = "[]",
) -> SemanticTraceSplit:
    """Assign one raw task identity to a stable role independent of model output."""

    request = _json_object(request_json, "semantic trace request")
    manifest = _json_array(media_manifest_json, "semantic trace media manifest")
    text = " ".join(
        unicodedata.normalize("NFKC", str(request.get("text", "")))
        .casefold()
        .split()
    )
    workspace = " ".join(
        unicodedata.normalize("NFKC", str(request.get("workspace") or ""))
        .casefold()
        .split()
    )
    media = (
        [str(item.get("sha256", "")) for item in manifest if isinstance(item, Mapping)]
        if manifest
        else request.get("images", [])
    )
    identity = _digest(
        canonical_json(
            {
                "text": text,
                "workspace": workspace,
                "media": media,
            }
        )
    )
    return semantic_trace_split(identity)


def _review_digest(*values: str) -> str:
    return _digest(*values)


def _digest(*values: str) -> str:
    return sha256(canonical_json(values).encode("utf-8")).hexdigest()


def _json_object(value: str, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return decoded


def _json_array(value: str, label: str) -> list[Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(decoded, list):
        raise ValueError(f"{label} must be a JSON array")
    return decoded


def _prepare_semantic_media(
    images: Sequence[Any],
) -> tuple[tuple[dict[str, Any], bytes], ...]:
    prepared: list[tuple[dict[str, Any], bytes]] = []
    for image_index, image in enumerate(images):
        if isinstance(image, bytes):
            content = image
        else:
            path = Path(image)
            try:
                if not 0 < path.stat().st_size <= SEMANTIC_MEDIA_MAX_BYTES:
                    continue
                content = path.read_bytes()
            except OSError:
                continue
        if not 0 < len(content) <= SEMANTIC_MEDIA_MAX_BYTES:
            continue
        digest = sha256(content).hexdigest()
        manifest = {
            "image_index": image_index,
            "mime_type": _semantic_media_type(content),
            "sha256": digest,
            "size": len(content),
        }
        prepared.append((manifest, content))
    return tuple(prepared)


def _semantic_media_type(content: bytes) -> str:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(content) >= 3 and content[:2] in {
        b"P1",
        b"P2",
        b"P3",
        b"P4",
        b"P5",
        b"P6",
    } and content[2:3] in b" \t\r\n":
        return "image/x-portable-anymap"
    return "application/octet-stream"


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} digest must be SHA-256")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value: str) -> None:
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("semantic timestamp is invalid") from exc


if __name__ == "__main__":
    raise SystemExit(main())

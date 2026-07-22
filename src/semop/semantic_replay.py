from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import re
from typing import Any, Protocol
import unicodedata

from .semantic_experience import (
    SemanticTraceRecord,
    SemanticTraceReview,
    SemanticTraceStats,
    SemanticTraceStore,
)


class SemanticReplaySource(Protocol):
    def pending(
        self,
        *,
        limit: int = 20,
        include_images: bool = True,
    ) -> tuple[SemanticTraceRecord, ...]: ...

    def latest_reviews(self) -> dict[str, SemanticTraceReview]: ...

    def stats(self) -> SemanticTraceStats: ...

    def get(self, trace_id: str) -> SemanticTraceRecord: ...

    def media_is_reviewable(self, trace: SemanticTraceRecord | str) -> bool: ...


@dataclass(frozen=True)
class SemanticReplayCandidate:
    trace: SemanticTraceRecord
    priority: int
    novelty: float
    reasons: tuple[str, ...]
    can_accept: bool
    can_reject: bool
    blocked_reason: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.priority <= 100:
            raise ValueError("semantic replay priority must be between 0 and 100")
        if not 0.0 <= self.novelty <= 1.0:
            raise ValueError("semantic replay novelty must be between 0 and 1")
        if not self.reasons:
            raise ValueError("semantic replay candidate requires at least one reason")
        if self.blocked_reason and (self.can_accept or self.can_reject):
            raise ValueError("blocked replay candidates cannot be reviewable")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trace"] = self.trace.to_dict()
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class SemanticReplayBatch:
    candidates: tuple[SemanticReplayCandidate, ...]
    pending_total: int
    reviewed_total: int
    pool_size: int

    def __post_init__(self) -> None:
        if min(self.pending_total, self.reviewed_total, self.pool_size) < 0:
            raise ValueError("semantic replay counts cannot be negative")
        if self.pool_size > self.pending_total:
            raise ValueError("semantic replay pool cannot exceed pending records")
        trace_ids = tuple(candidate.trace.trace_id for candidate in self.candidates)
        if len(trace_ids) != len(set(trace_ids)):
            raise ValueError("semantic replay batch contains duplicate traces")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "semop.semantic-replay.v1",
            "pending_total": self.pending_total,
            "reviewed_total": self.reviewed_total,
            "pool_size": self.pool_size,
            "selected": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class SemanticReplayPlanner:
    """Select a small, explainable review batch without changing learned state."""

    def __init__(self, source: SemanticReplaySource | SemanticTraceStore) -> None:
        self.source = source

    def select(
        self,
        *,
        limit: int = 8,
        pool_limit: int = 100,
    ) -> SemanticReplayBatch:
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("semantic replay limit must be between 1 and 20")
        if type(pool_limit) is not int or not limit <= pool_limit <= 100:
            raise ValueError("semantic replay pool limit must be between limit and 100")

        pending_total = self.source.stats().pending
        pool = list(self.source.pending(limit=pool_limit))
        reviews = self.source.latest_reviews()
        reviewed = [self.source.get(trace_id) for trace_id in sorted(reviews)]
        accepted = [
            record
            for record in reviewed
            if reviews[record.trace_id].accepted
        ]
        accepted_domains = _counts(record.domain for record in accepted)
        accepted_splits = _counts(record.split.value for record in accepted)
        reviewability = {
            record.trace_id: self._reviewability(record) for record in pool
        }
        reference_features = [
            features
            for record in reviewed
            if (features := _prompt_features(record.prompt))
        ]

        selected: list[SemanticReplayCandidate] = []
        while pool and len(selected) < limit:
            evaluated = [
                self._candidate(
                    record,
                    reference_features=reference_features,
                    accepted_domains=accepted_domains,
                    accepted_splits=accepted_splits,
                    reviewability=reviewability[record.trace_id],
                )
                for record in pool
            ]
            evaluated.sort(
                key=lambda candidate: (
                    -candidate.priority,
                    not candidate.can_accept,
                    candidate.trace.created_at,
                    candidate.trace.trace_id,
                )
            )
            choice = evaluated[0]
            selected.append(choice)
            pool = [record for record in pool if record.trace_id != choice.trace.trace_id]
            features = _prompt_features(choice.trace.prompt)
            if features:
                reference_features.append(features)

        return SemanticReplayBatch(
            candidates=tuple(selected),
            pending_total=pending_total,
            reviewed_total=len(reviews),
            pool_size=min(pending_total, pool_limit),
        )

    def _candidate(
        self,
        record: SemanticTraceRecord,
        *,
        reference_features: list[frozenset[str]],
        accepted_domains: dict[str, int],
        accepted_splits: dict[str, int],
        reviewability: tuple[bool, bool, str],
    ) -> SemanticReplayCandidate:
        can_accept, can_reject, blocked_reason = reviewability
        if not can_accept and not can_reject:
            return SemanticReplayCandidate(
                trace=record,
                priority=0,
                novelty=0.0,
                reasons=("검토할 원본 이미지 증거가 없어 우선순위에서 제외",),
                can_accept=False,
                can_reject=False,
                blocked_reason=blocked_reason,
            )

        score = 0
        reasons: list[str] = []
        if can_accept:
            score += 40
            reasons.append(
                "proof replay 또는 보존된 이미지 근거가 있어 positive 검토 가능"
            )
        else:
            score += 25
            reasons.append("검증되지 않은 모델 해석이라 오류 교정 후보")

        if record.answer_status in {"best_effort", "unsupported"}:
            score += 15
            reasons.append("현재 답변이 미검증 또는 미지원 상태")

        features = _prompt_features(record.prompt)
        similarity = max(
            (_jaccard(features, reference) for reference in reference_features),
            default=0.0,
        )
        novelty = 1.0 - similarity
        if novelty >= 0.75:
            score += 20
            reasons.append("기존 검토와 다른 표현 또는 구조")
        elif novelty >= 0.35:
            score += 10
            reasons.append("기존 검토와 일부만 겹치는 표현")

        if accepted_domains.get(record.domain, 0) == 0:
            score += 10
            reasons.append("아직 승인 positive가 없는 도메인")
        if accepted_splits.get(record.split.value, 0) == 0:
            score += 10
            reasons.append(f"{record.split.value} 학습 역할의 positive가 부족")

        return SemanticReplayCandidate(
            trace=record,
            priority=min(score, 100),
            novelty=round(novelty, 6),
            reasons=tuple(reasons),
            can_accept=can_accept,
            can_reject=can_reject,
        )

    def _reviewability(
        self,
        record: SemanticTraceRecord,
    ) -> tuple[bool, bool, str]:
        if not record.has_images:
            return record.replay_verified, True, ""
        if self.source.media_is_reviewable(record):
            return True, True, ""
        return (
            False,
            False,
            "정확한 PNG/JPEG 픽셀 증거가 로컬 journal에 필요해.",
        )


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _prompt_features(text: str) -> frozenset[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = re.findall(r"[\w]+", normalized, flags=re.UNICODE)
    compact = "".join(character for character in normalized if not character.isspace())
    character_pairs = (
        {f"c:{compact[index:index + 2]}" for index in range(len(compact) - 1)}
        if len(compact) >= 2
        else ({f"c:{compact}"} if compact else set())
    )
    return frozenset({f"w:{word}" for word in words} | character_pairs)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0

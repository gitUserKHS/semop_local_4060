from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Literal
import unicodedata

from .prompt_api import PromptImage, PromptRequest
from .semantic_experience import (
    SEMANTIC_MEDIA_MAX_BYTES,
    SemanticTraceDecision,
    SemanticTraceStore,
)
from .semantic_student_evaluation import semantic_completion_signature


SEMANTIC_PROTOTYPE_MIN_SUPPORT = 3
SEMANTIC_PROTOTYPE_MIN_SIMILARITY = 0.35


@dataclass(frozen=True)
class SemanticPrototype:
    prototype_id: str
    domain: str
    semantic_signature: str
    support: int
    representative_trace_id: str
    trace_ids: tuple[str, ...]
    review_digests: tuple[str, ...]
    prompts: tuple[str, ...]
    completion_json: str
    model_id: str
    latest_reviewed_at: str

    def __post_init__(self) -> None:
        if len(self.prototype_id) != 64:
            raise ValueError("semantic prototype id must be SHA-256")
        if self.support < SEMANTIC_PROTOTYPE_MIN_SUPPORT:
            raise ValueError("semantic prototype support is too small")
        if not self.domain or not self.semantic_signature:
            raise ValueError("semantic prototype identity cannot be empty")
        if not self.completion_json or not self.latest_reviewed_at:
            raise ValueError("semantic prototype evidence cannot be empty")
        if not (
            len(self.trace_ids)
            == len(self.review_digests)
            == len(self.prompts)
            == self.support
        ):
            raise ValueError("semantic prototype evidence counts do not match")
        if self.representative_trace_id not in self.trace_ids:
            raise ValueError("semantic prototype representative is not evidence")
        if len({_normalize_text(prompt) for prompt in self.prompts}) != self.support:
            raise ValueError("semantic prototype prompts must be distinct")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticMemoryMatch:
    trace_id: str
    decision: SemanticTraceDecision | str
    modality: Literal["text", "vision"]
    score: float
    prompt_similarity: float
    prompt: str
    completion_json: str
    model_id: str
    review_note: str = ""
    source: Literal["exact", "prototype"] = "exact"
    support: int = 1
    prototype_id: str = ""
    examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", SemanticTraceDecision(self.decision))
        if self.modality not in {"text", "vision"}:
            raise ValueError("semantic memory modality must be text or vision")
        if self.source not in {"exact", "prototype"}:
            raise ValueError("semantic memory source must be exact or prototype")
        if self.support < 1:
            raise ValueError("semantic memory support must be positive")
        if self.source == "prototype" and (
            self.decision is not SemanticTraceDecision.ACCEPTED
            or self.support < SEMANTIC_PROTOTYPE_MIN_SUPPORT
            or len(self.prototype_id) != 64
        ):
            raise ValueError("semantic prototype match is invalid")
        for value in (self.score, self.prompt_similarity):
            if not 0.0 <= value <= 1.0:
                raise ValueError("semantic memory scores must be between 0 and 1")

    def to_hint(self) -> str:
        if self.source == "prototype":
            authority = "HUMAN_ACCEPTED_SEMANTIC_PROTOTYPE"
        else:
            authority = (
                "HUMAN_ACCEPTED_SEMANTIC_MEMORY"
                if self.decision is SemanticTraceDecision.ACCEPTED
                else "HUMAN_REJECTED_SEMANTIC_MEMORY"
            )
        proposal = self.completion_json[:2_000]
        note = f"\nreview_note={self.review_note[:300]}" if self.review_note else ""
        prototype = ""
        if self.source == "prototype":
            examples = " | ".join(item[:240] for item in self.examples[:3])
            prototype = (
                f" source=prototype support={self.support} "
                f"prototype_id={self.prototype_id}\nprior_examples={examples}"
            )
        return (
            f"{authority} modality={self.modality} similarity={self.score:.4f} "
            f"prompt={self.prompt_similarity:.4f}{prototype}\n"
            f"prior_prompt={self.prompt[:800]}\n"
            f"prior_proposal={proposal}{note}\n"
            "Use this only as an analogy. The current request still requires a fresh "
            "PROPOSED interpretation."
        )


class ReviewedSemanticMemory:
    """Reuse exact human-reviewed semantics without granting fact authority."""

    def __init__(self, store: SemanticTraceStore) -> None:
        self.store = store

    def retrieve(
        self,
        request: PromptRequest,
        *,
        limit: int = 3,
    ) -> tuple[str, ...]:
        return tuple(match.to_hint() for match in self.search(request, limit=limit))

    def search(
        self,
        request: PromptRequest,
        *,
        limit: int = 3,
    ) -> tuple[SemanticMemoryMatch, ...]:
        if type(limit) is not int or not 1 <= limit <= 8:
            raise ValueError("semantic memory limit must be between 1 and 8")
        if not request.images:
            return self._search_text(request, limit=limit)
        if len(request.images) == 1:
            return self._search_image(request, limit=limit)
        return ()

    def _search_text(
        self,
        request: PromptRequest,
        *,
        limit: int,
    ) -> tuple[SemanticMemoryMatch, ...]:
        query = _normalize_text(request.text)
        if not query:
            return ()
        matches: list[tuple[str, SemanticMemoryMatch]] = []
        try:
            for trace_id, review in self.store.latest_reviews().items():
                trace = self.store.get(trace_id)
                if trace.has_images or _normalize_text(trace.prompt) != query:
                    continue
                matches.append(
                    (
                        review.reviewed_at,
                        SemanticMemoryMatch(
                            trace_id=trace.trace_id,
                            decision=review.decision,
                            modality="text",
                            score=1.0,
                            prompt_similarity=1.0,
                            prompt=trace.prompt,
                            completion_json=trace.completion_json,
                            model_id=trace.model_id,
                            review_note=review.note,
                        ),
                    )
                )
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return ()
        matches.sort(key=lambda item: (item[0], item[1].trace_id), reverse=True)
        if matches:
            return tuple(item[1] for item in matches[:1])
        return self._search_prototypes(request.text, limit=limit)

    def prototypes(
        self,
        *,
        min_support: int = SEMANTIC_PROTOTYPE_MIN_SUPPORT,
    ) -> tuple[SemanticPrototype, ...]:
        if type(min_support) is not int or not 3 <= min_support <= 20:
            raise ValueError("semantic prototype support must be between 3 and 20")
        grouped: dict[str, dict[str, tuple[str, Any, Any]]] = {}
        try:
            for trace_id, review in self.store.latest_reviews().items():
                if not review.accepted:
                    continue
                trace = self.store.get(trace_id)
                if trace.has_images or not trace.replay_verified:
                    continue
                signature = semantic_completion_signature(trace.completion_json)
                normalized_prompt = _normalize_text(trace.prompt)
                if not signature or not normalized_prompt:
                    continue
                entries = grouped.setdefault(signature, {})
                current = entries.get(normalized_prompt)
                candidate = (review.reviewed_at, trace, review)
                if current is None or candidate[0] > current[0]:
                    entries[normalized_prompt] = candidate
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return ()

        prototypes: list[SemanticPrototype] = []
        for signature, by_prompt in sorted(grouped.items()):
            if len(by_prompt) < min_support:
                continue
            evidence = sorted(
                by_prompt.values(),
                key=lambda item: (item[0], item[1].trace_id),
                reverse=True,
            )
            representative = evidence[0][1]
            trace_ids = tuple(sorted(item[1].trace_id for item in evidence))
            review_by_trace = {item[1].trace_id: item[2] for item in evidence}
            trace_by_id = {item[1].trace_id: item[1] for item in evidence}
            review_digests = tuple(
                review_by_trace[trace_id].review_digest for trace_id in trace_ids
            )
            prompts = tuple(trace_by_id[trace_id].prompt for trace_id in trace_ids)
            prototype_id = _prototype_id(signature, trace_ids, review_digests)
            prototypes.append(
                SemanticPrototype(
                    prototype_id=prototype_id,
                    domain=representative.domain,
                    semantic_signature=signature,
                    support=len(trace_ids),
                    representative_trace_id=representative.trace_id,
                    trace_ids=trace_ids,
                    review_digests=review_digests,
                    prompts=prompts,
                    completion_json=representative.completion_json,
                    model_id=representative.model_id,
                    latest_reviewed_at=evidence[0][0],
                )
            )
        prototypes.sort(
            key=lambda item: (
                -item.support,
                item.domain,
                item.prototype_id,
            )
        )
        return tuple(prototypes)

    def _search_prototypes(
        self,
        query: str,
        *,
        limit: int,
    ) -> tuple[SemanticMemoryMatch, ...]:
        matches: list[SemanticMemoryMatch] = []
        for prototype in self.prototypes():
            if _explicit_math_operator_conflicts(query, prototype):
                continue
            scored_prompts = sorted(
                (
                    (_text_similarity(query, prompt), prompt)
                    for prompt in prototype.prompts
                ),
                key=lambda item: (-item[0], _normalize_text(item[1])),
            )
            similarity = scored_prompts[0][0]
            if similarity < SEMANTIC_PROTOTYPE_MIN_SIMILARITY:
                continue
            matches.append(
                SemanticMemoryMatch(
                    trace_id=prototype.representative_trace_id,
                    decision=SemanticTraceDecision.ACCEPTED,
                    modality="text",
                    score=similarity,
                    prompt_similarity=similarity,
                    prompt=scored_prompts[0][1],
                    completion_json=prototype.completion_json,
                    model_id=prototype.model_id,
                    source="prototype",
                    support=prototype.support,
                    prototype_id=prototype.prototype_id,
                    examples=tuple(prompt for _, prompt in scored_prompts[:3]),
                )
            )
        matches.sort(
            key=lambda item: (
                -item.score,
                -item.support,
                item.prototype_id,
            )
        )
        return tuple(matches[:limit])

    def _search_image(
        self,
        request: PromptRequest,
        *,
        limit: int,
    ) -> tuple[SemanticMemoryMatch, ...]:
        query_content = _read_prompt_image(request.images[0])
        if query_content is None:
            return ()
        query_sha = sha256(query_content).hexdigest()
        latest_by_prompt: dict[str, tuple[str, SemanticMemoryMatch]] = {}
        try:
            for trace_id, review in self.store.latest_reviews().items():
                trace = self.store.get(trace_id)
                if not trace.has_images or not self.store.media_is_reviewable(trace):
                    continue
                media = self.store.media(trace_id)
                if len(media) != 1 or query_sha != media[0].content_sha256:
                    continue
                prompt_similarity = _token_similarity(request.text, trace.prompt)
                match = SemanticMemoryMatch(
                    trace_id=trace.trace_id,
                    decision=review.decision,
                    modality="vision",
                    score=round(0.9 + 0.1 * prompt_similarity, 6),
                    prompt_similarity=prompt_similarity,
                    prompt=trace.prompt,
                    completion_json=trace.completion_json,
                    model_id=trace.model_id,
                    review_note=review.note,
                )
                key = _normalize_text(trace.prompt)
                existing = latest_by_prompt.get(key)
                candidate = (review.reviewed_at, match)
                if existing is None or candidate[0] > existing[0] or (
                    candidate[0] == existing[0]
                    and candidate[1].trace_id > existing[1].trace_id
                ):
                    latest_by_prompt[key] = candidate
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return ()
        matches = list(latest_by_prompt.values())
        matches.sort(key=lambda item: (item[0], item[1].trace_id), reverse=True)
        matches.sort(key=lambda item: item[1].score, reverse=True)
        return tuple(item[1] for item in matches[:limit])


def _read_prompt_image(image: PromptImage) -> bytes | None:
    if isinstance(image, bytes):
        content = image
    else:
        path = Path(image)
        try:
            if not 0 < path.stat().st_size <= SEMANTIC_MEDIA_MAX_BYTES:
                return None
            content = path.read_bytes()
        except OSError:
            return None
    if not 0 < len(content) <= SEMANTIC_MEDIA_MAX_BYTES:
        return None
    return content


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _token_similarity(left: str, right: str) -> float:
    left_tokens = set(_normalize_text(left).split())
    right_tokens = set(_normalize_text(right).split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return round(
        len(left_tokens.intersection(right_tokens))
        / len(left_tokens.union(right_tokens)),
        6,
    )


def _text_similarity(left: str, right: str) -> float:
    word_score = _token_similarity(left, right)
    left_pairs = _character_pairs(left)
    right_pairs = _character_pairs(right)
    if not left_pairs or not right_pairs:
        character_score = 0.0
    else:
        character_score = len(left_pairs & right_pairs) / len(left_pairs | right_pairs)
    return round(max(word_score, character_score), 6)


def _character_pairs(value: str) -> set[str]:
    compact = re.sub(r"\s+", "", _normalize_text(value))
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _prototype_id(
    semantic_signature: str,
    trace_ids: tuple[str, ...],
    review_digests: tuple[str, ...],
) -> str:
    payload = json.dumps(
        {
            "semantic_signature": semantic_signature,
            "trace_ids": trace_ids,
            "review_digests": review_digests,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _explicit_math_operator_conflicts(
    query: str,
    prototype: SemanticPrototype,
) -> bool:
    if prototype.domain != "math":
        return False
    query_operator = _query_math_operator(query)
    prototype_operator = _prototype_math_operator(prototype.semantic_signature)
    return bool(
        query_operator
        and prototype_operator
        and query_operator != prototype_operator
    )


def _query_math_operator(text: str) -> str:
    normalized = _normalize_text(text)
    compact = normalized.replace(" ", "")
    operators: set[str] = set(
        re.findall(r"(?<=\d)[+\-*/](?=\d)", compact)
    )
    cues = {
        "+": ("더하", "더하기", "합", "추가", "plus", "add"),
        "-": ("빼", "제외", "먹", "남", "minus", "subtract"),
        "*": ("곱", "몇 배", "배로", "times", "multiply"),
        "/": ("나누", "몫", "divide", "divided"),
    }
    for operator, words in cues.items():
        if any(word in normalized for word in words):
            operators.add(operator)
    return next(iter(operators)) if len(operators) == 1 else ""


def _prototype_math_operator(semantic_signature: str) -> str:
    try:
        value = json.loads(semantic_signature)
        expression = str(value["payload"]["expression"]).replace(" ", "")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return ""
    operators = set(re.findall(r"(?<=\d)[+\-*/](?=\d)", expression))
    return next(iter(operators)) if len(operators) == 1 else ""

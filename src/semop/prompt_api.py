from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .kernel import TypedDomainRequest, UnifiedTypedResult


class ResourceTier(str, Enum):
    AUTO = "auto"
    SYMBOLIC = "symbolic"
    ECONOMY = "economy"
    BALANCED = "balanced"


class AnswerStatus(str, Enum):
    VERIFIED = "verified"
    CONDITIONAL = "conditional"
    BEST_EFFORT = "best_effort"
    UNSUPPORTED = "unsupported"


class CognitiveOperator(str, Enum):
    DECOMPOSE = "DECOMPOSE"
    RETRIEVE = "RETRIEVE"
    SELECT = "SELECT"
    FILTER = "FILTER"
    ALIGN = "ALIGN"
    COMPARE = "COMPARE"
    COMPOSE = "COMPOSE"
    INFER = "INFER"
    VERIFY = "VERIFY"
    REPAIR = "REPAIR"
    EXPLAIN = "EXPLAIN"
    ASK = "ASK"


class ConversationRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ConversationMessage:
    role: ConversationRole | str
    content: str
    status: str = ""

    def __post_init__(self) -> None:
        role = ConversationRole(self.role)
        content = str(self.content).strip()
        status = str(self.status).strip()
        if not content:
            raise ValueError("conversation message content cannot be empty")
        if len(content) > 16_000:
            raise ValueError("conversation message cannot exceed 16,000 characters")
        if len(status) > 64:
            raise ValueError("conversation message status is too long")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "status", status)

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role.value,
            "content": self.content,
            "status": self.status,
        }


@dataclass(frozen=True)
class RecalledMemory:
    memory_id: str
    content: str
    score: float
    saved_at: str = ""
    retrieval: str = "lexical"

    def __post_init__(self) -> None:
        memory_id = str(self.memory_id).strip()
        content = str(self.content).strip()
        saved_at = str(self.saved_at).strip()
        retrieval = str(self.retrieval).strip()
        if not memory_id or len(memory_id) > 64:
            raise ValueError("recalled memory id must be 1..64 characters")
        if not content or len(content) > 1_000:
            raise ValueError("recalled memory content must be 1..1,000 characters")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("recalled memory score must be between 0 and 1")
        if len(saved_at) > 64:
            raise ValueError("recalled memory timestamp is too long")
        if retrieval not in {"lexical", "semantic_e5"}:
            raise ValueError("recalled memory retrieval method is invalid")
        object.__setattr__(self, "memory_id", memory_id)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "saved_at", saved_at)
        object.__setattr__(self, "retrieval", retrieval)

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(frozen=True)
class RecalledEpisode:
    episode_id: str
    user_content: str
    assistant_content: str
    answer_status: str
    score: float
    occurred_at: str = ""
    scope: str = "cross_session"

    def __post_init__(self) -> None:
        episode_id = str(self.episode_id).strip()
        user_content = str(self.user_content).strip()
        assistant_content = str(self.assistant_content).strip()
        answer_status = str(self.answer_status).strip()
        occurred_at = str(self.occurred_at).strip()
        scope = str(self.scope).strip()
        if len(episode_id) != 64:
            raise ValueError("recalled episode id must be SHA-256")
        if not user_content or len(user_content) > 4_000:
            raise ValueError("recalled episode user content must be 1..4,000 characters")
        if not assistant_content or len(assistant_content) > 4_000:
            raise ValueError(
                "recalled episode assistant content must be 1..4,000 characters"
            )
        if not answer_status or len(answer_status) > 64:
            raise ValueError("recalled episode answer status must be 1..64 characters")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("recalled episode score must be between 0 and 1")
        if len(occurred_at) > 64:
            raise ValueError("recalled episode timestamp is too long")
        if scope not in {"cross_session", "same_session_archive"}:
            raise ValueError("recalled episode scope is invalid")
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "user_content", user_content)
        object.__setattr__(self, "assistant_content", assistant_content)
        object.__setattr__(self, "answer_status", answer_status)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "occurred_at", occurred_at)
        object.__setattr__(self, "scope", scope)

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    revision: int
    title: str
    objective: str
    status: str
    progress: str = ""
    decisions: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        task_id = str(self.task_id).strip()
        title = str(self.title).strip()
        objective = str(self.objective).strip()
        status = str(self.status).strip().lower()
        progress = str(self.progress).strip()
        decisions = tuple(str(item).strip() for item in self.decisions if str(item).strip())
        next_actions = tuple(
            str(item).strip() for item in self.next_actions if str(item).strip()
        )
        if not task_id or len(task_id) > 64:
            raise ValueError("task context id must be 1..64 characters")
        if self.revision < 1:
            raise ValueError("task context revision must be positive")
        if not title or len(title) > 120:
            raise ValueError("task context title must be 1..120 characters")
        if not objective or len(objective) > 2_000:
            raise ValueError("task context objective must be 1..2,000 characters")
        if status not in {"active", "completed"}:
            raise ValueError("task context status must be active or completed")
        if len(progress) > 2_000:
            raise ValueError("task context progress cannot exceed 2,000 characters")
        if len(decisions) > 20 or len(next_actions) > 20:
            raise ValueError("task context lists can contain at most 20 items")
        if any(len(item) > 500 for item in (*decisions, *next_actions)):
            raise ValueError("task context list items cannot exceed 500 characters")
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "next_actions", next_actions)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int
    text: str = ""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("source span must satisfy 0 <= start <= end")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageRegion:
    image_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.image_index < 0:
            raise ValueError("image index cannot be negative")
        coordinates = (self.x1, self.y1, self.x2, self.y2)
        if any(not 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError("image-region coordinates must be normalized to 0..1")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("image region must satisfy x1 <= x2 and y1 <= y2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PromptImage = str | Path | bytes


@dataclass(frozen=True)
class PromptRequest:
    text: str = ""
    images: tuple[PromptImage, ...] = ()
    workspace: str | Path | None = None
    resource_tier: ResourceTier | str = ResourceTier.AUTO
    judge_opt_in: bool = False
    conversation: tuple[ConversationMessage, ...] = ()
    recalled_memories: tuple[RecalledMemory, ...] = ()
    task_contexts: tuple[TaskContext, ...] = ()
    recalled_episodes: tuple[RecalledEpisode, ...] = ()

    def __post_init__(self) -> None:
        text = str(self.text).strip()
        images = tuple(self.images)
        if not text and not images:
            raise ValueError("prompt requires text or at least one image")
        if len(text) > 16_000:
            raise ValueError("prompt text cannot exceed 16,000 characters")
        if len(images) > 4:
            raise ValueError("a prompt can contain at most four images")
        for image in images:
            if not isinstance(image, (str, Path, bytes)):
                raise TypeError("prompt images must be paths or bytes")
            if isinstance(image, bytes) and len(image) > 16 * 1024 * 1024:
                raise ValueError("an in-memory image cannot exceed 16 MiB")
        if type(self.judge_opt_in) is not bool:
            raise TypeError("judge_opt_in must be a bool")
        conversation = tuple(self.conversation)
        if len(conversation) > 12:
            raise ValueError("prompt conversation cannot exceed 12 messages")
        if any(not isinstance(item, ConversationMessage) for item in conversation):
            raise TypeError("prompt conversation requires ConversationMessage items")
        if sum(len(item.content) for item in conversation) > 12_000:
            raise ValueError("prompt conversation exceeds the 12,000-character cap")
        recalled_memories = tuple(self.recalled_memories)
        if len(recalled_memories) > 4:
            raise ValueError("prompt can recall at most four long-term memories")
        if any(not isinstance(item, RecalledMemory) for item in recalled_memories):
            raise TypeError("prompt recalled memories require RecalledMemory items")
        if sum(len(item.content) for item in recalled_memories) > 4_000:
            raise ValueError("prompt recalled memories exceed the 4,000-character cap")
        recalled_episodes = tuple(self.recalled_episodes)
        if len(recalled_episodes) > 2:
            raise ValueError("prompt can recall at most two past episodes")
        if any(not isinstance(item, RecalledEpisode) for item in recalled_episodes):
            raise TypeError("prompt recalled episodes require RecalledEpisode items")
        if (
            sum(
                len(item.user_content) + len(item.assistant_content)
                for item in recalled_episodes
            )
            > 8_000
        ):
            raise ValueError("prompt recalled episodes exceed the 8,000-character cap")
        task_contexts = tuple(self.task_contexts)
        if len(task_contexts) > 2:
            raise ValueError("prompt can include at most two task contexts")
        if any(not isinstance(item, TaskContext) for item in task_contexts):
            raise TypeError("prompt task contexts require TaskContext items")
        workspace = None if self.workspace is None else Path(self.workspace)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "conversation", conversation)
        object.__setattr__(self, "recalled_memories", recalled_memories)
        object.__setattr__(self, "recalled_episodes", recalled_episodes)
        object.__setattr__(self, "task_contexts", task_contexts)
        object.__setattr__(
            self,
            "resource_tier",
            ResourceTier(self.resource_tier),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "images": [_image_summary(image) for image in self.images],
            "workspace": str(self.workspace) if self.workspace is not None else None,
            "resource_tier": self.resource_tier.value,
            "judge_opt_in": self.judge_opt_in,
            "conversation": [item.to_dict() for item in self.conversation],
            "recalled_memories": [
                item.to_dict() for item in self.recalled_memories
            ],
            "recalled_episodes": [
                item.to_dict() for item in self.recalled_episodes
            ],
            "task_contexts": [item.to_dict() for item in self.task_contexts],
        }


@dataclass(frozen=True)
class SemanticProposal:
    domain: str
    operator_program: tuple[CognitiveOperator | str, ...]
    confidence: float
    producer_id: str
    request: TypedDomainRequest | None = None
    candidate_answer: str = ""
    source_spans: tuple[SourceSpan, ...] = ()
    image_regions: tuple[ImageRegion, ...] = ()
    diagnostics: tuple[str, ...] = ()
    model_id: str = ""
    resource_tier: ResourceTier | str = ResourceTier.SYMBOLIC
    deterministic: bool = False
    fingerprint: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        domain = self.domain.strip().lower()
        if domain not in {"coding", "language", "math", "vision", "composed", "unsupported"}:
            raise ValueError(f"unsupported proposal domain: {domain!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("proposal confidence must be between 0 and 1")
        if not self.producer_id.strip():
            raise ValueError("proposal producer_id cannot be empty")
        operators = tuple(CognitiveOperator(item) for item in self.operator_program)
        if not operators:
            raise ValueError("proposal requires at least one cognitive operator")
        spans = tuple(self.source_spans)
        regions = tuple(self.image_regions)
        diagnostics = tuple(str(item) for item in self.diagnostics)
        tier = ResourceTier(self.resource_tier)
        if self.request is not None and str(self.request.domain) not in {
            domain,
            f"DomainKind.{domain.upper()}",
        }:
            request_domain = getattr(self.request.domain, "value", self.request.domain)
            if str(request_domain) != domain:
                raise ValueError("proposal domain and typed request domain disagree")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "operator_program", operators)
        object.__setattr__(self, "source_spans", spans)
        object.__setattr__(self, "image_regions", regions)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "resource_tier", tier)
        if not self.fingerprint:
            payload = {
                "domain": domain,
                "operators": [item.value for item in operators],
                "confidence": round(float(self.confidence), 8),
                "producer_id": self.producer_id,
                "request": _stable_value(self.request),
                "candidate_answer": self.candidate_answer,
                "spans": [item.to_dict() for item in spans],
                "regions": [item.to_dict() for item in regions],
                "model_id": self.model_id,
                "tier": tier.value,
                "deterministic": self.deterministic,
            }
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            object.__setattr__(self, "fingerprint", sha256(encoded).hexdigest())

    @property
    def disposition(self) -> str:
        return "proposed"

    def to_dict(self) -> dict[str, Any]:
        request_domain = None
        if self.request is not None:
            request_domain = getattr(self.request.domain, "value", self.request.domain)
        return {
            "domain": self.domain,
            "disposition": self.disposition,
            "operator_program": [item.value for item in self.operator_program],
            "confidence": self.confidence,
            "producer_id": self.producer_id,
            "model_id": self.model_id,
            "resource_tier": self.resource_tier.value,
            "deterministic": self.deterministic,
            "typed_request": request_domain,
            "candidate_answer": self.candidate_answer,
            "source_spans": [item.to_dict() for item in self.source_spans],
            "image_regions": [item.to_dict() for item in self.image_regions],
            "diagnostics": list(self.diagnostics),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class ResourceMetrics:
    elapsed_seconds: float = 0.0
    model_loaded: bool = False
    model_id: str = ""
    controller_used: bool = False
    expansions: int = 0
    peak_vram_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.elapsed_seconds < 0 or self.expansions < 0:
            raise ValueError("resource metrics cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerEnvelope:
    answer: str
    status: AnswerStatus | str
    proposals: tuple[SemanticProposal, ...] = ()
    results: tuple[UnifiedTypedResult, ...] = ()
    proof: str = ""
    assumption_dependencies: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metrics: ResourceMetrics = ResourceMetrics()

    def __post_init__(self) -> None:
        answer = self.answer.strip()
        if not answer:
            raise ValueError("answer envelope requires non-empty answer text")
        object.__setattr__(self, "answer", answer)
        object.__setattr__(self, "status", AnswerStatus(self.status))
        object.__setattr__(self, "proposals", tuple(self.proposals))
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(
            self,
            "assumption_dependencies",
            tuple(str(item) for item in self.assumption_dependencies),
        )
        object.__setattr__(self, "unresolved", tuple(str(item) for item in self.unresolved))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def success(self) -> bool:
        return self.status in {AnswerStatus.VERIFIED, AnswerStatus.CONDITIONAL}

    @property
    def verified(self) -> bool:
        return self.status is AnswerStatus.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status.value,
            "success": self.success,
            "verified": self.verified,
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "results": [result.to_dict() for result in self.results],
            "proof": self.proof,
            "assumption_dependencies": list(self.assumption_dependencies),
            "unresolved": list(self.unresolved),
            "provenance": dict(self.provenance),
            "metrics": self.metrics.to_dict(),
        }


def _image_summary(image: PromptImage) -> dict[str, Any]:
    if isinstance(image, bytes):
        return {
            "kind": "bytes",
            "size": len(image),
            "sha256": sha256(image).hexdigest(),
        }
    path = Path(image)
    return {"kind": "path", "path": str(path)}


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": sha256(value).hexdigest(), "size": len(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _stable_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable_value(item) for item in value]
    return repr(value)

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
from uuid import uuid4

from .prompt_api import TaskContext


_TASK_PATTERN = re.compile(r"[0-9a-f]{32}")
_SESSION_PATTERN = re.compile(r"[A-Za-z0-9_-]{16,64}")


class TaskStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"


class TaskQueryKind(str, Enum):
    NEXT_ACTION = "next_action"
    DECISIONS = "decisions"
    SUMMARY = "summary"


@dataclass(frozen=True)
class TaskCheckpoint:
    task_id: str
    revision: int
    title: str
    objective: str
    status: TaskStatus | str
    progress: str
    decisions: tuple[str, ...]
    next_actions: tuple[str, ...]
    source_session_id: str
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        status = TaskStatus(self.status)
        context = TaskContext(
            task_id=self.task_id,
            revision=self.revision,
            title=self.title,
            objective=self.objective,
            status=status.value,
            progress=self.progress,
            decisions=tuple(self.decisions),
            next_actions=tuple(self.next_actions),
        )
        source_session_id = str(self.source_session_id).strip()
        if _TASK_PATTERN.fullmatch(context.task_id) is None:
            raise ValueError("task checkpoint id is invalid")
        if source_session_id and _SESSION_PATTERN.fullmatch(source_session_id) is None:
            raise ValueError("task checkpoint session id is invalid")
        if not self.created_at or not self.updated_at:
            raise ValueError("task checkpoint timestamps cannot be empty")
        object.__setattr__(self, "task_id", context.task_id)
        object.__setattr__(self, "title", context.title)
        object.__setattr__(self, "objective", context.objective)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "progress", context.progress)
        object.__setattr__(self, "decisions", context.decisions)
        object.__setattr__(self, "next_actions", context.next_actions)
        object.__setattr__(self, "source_session_id", source_session_id)

    def to_context(self) -> TaskContext:
        return TaskContext(
            task_id=self.task_id,
            revision=self.revision,
            title=self.title,
            objective=self.objective,
            status=self.status.value,
            progress=self.progress,
            decisions=self.decisions,
            next_actions=self.next_actions,
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class TaskCheckpointMatch:
    checkpoint: TaskCheckpoint
    score: float
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("task checkpoint match score must be between 0 and 1")
        if self.reason not in {"only_active_task", "lexical_match"}:
            raise ValueError("task checkpoint match reason is invalid")
        if self.checkpoint.status is not TaskStatus.ACTIVE:
            raise ValueError("task checkpoint match must reference an active task")

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.checkpoint.to_dict(),
            "score": self.score,
            "reason": self.reason,
        }


class TaskCheckpointStore:
    """Append-only user-confirmed checkpoints for multi-session work."""

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
                CREATE TABLE IF NOT EXISTS task_checkpoints (
                    task_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress TEXT NOT NULL,
                    decisions_json TEXT NOT NULL,
                    next_actions_json TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, revision)
                );
                CREATE INDEX IF NOT EXISTS idx_task_checkpoint_updated
                    ON task_checkpoints(updated_at DESC);
                """
            )

    def create_task(
        self,
        *,
        title: str,
        objective: str,
        source_session_id: str = "",
    ) -> TaskCheckpoint:
        timestamp = _timestamp()
        checkpoint = TaskCheckpoint(
            task_id=uuid4().hex,
            revision=1,
            title=title,
            objective=objective,
            status=TaskStatus.ACTIVE,
            progress="",
            decisions=(),
            next_actions=(),
            source_session_id=source_session_id,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._insert(checkpoint)
        return checkpoint

    def checkpoint(
        self,
        task_id: str,
        *,
        progress: str | None = None,
        decisions: Sequence[str] | None = None,
        next_actions: Sequence[str] | None = None,
        title: str | None = None,
        objective: str | None = None,
        status: TaskStatus | str | None = None,
    ) -> TaskCheckpoint:
        current = self.get(task_id)
        if current.status is TaskStatus.COMPLETED:
            raise ValueError("completed task checkpoints are immutable")
        updated = TaskCheckpoint(
            task_id=current.task_id,
            revision=current.revision + 1,
            title=current.title if title is None else title,
            objective=current.objective if objective is None else objective,
            status=current.status if status is None else status,
            progress=current.progress if progress is None else progress,
            decisions=(current.decisions if decisions is None else tuple(decisions)),
            next_actions=(
                current.next_actions if next_actions is None else tuple(next_actions)
            ),
            source_session_id=current.source_session_id,
            created_at=current.created_at,
            updated_at=_timestamp(),
        )
        self._insert(updated)
        return updated

    def complete_task(
        self,
        task_id: str,
        *,
        progress: str | None = None,
        decisions: Sequence[str] | None = None,
    ) -> TaskCheckpoint:
        return self.checkpoint(
            task_id,
            progress=progress,
            decisions=decisions,
            next_actions=(),
            status=TaskStatus.COMPLETED,
        )

    def get(self, task_id: str, *, revision: int | None = None) -> TaskCheckpoint:
        normalized = _validate_task_id(task_id)
        if revision is not None and revision < 1:
            raise ValueError("task checkpoint revision must be positive")
        query = (
            "SELECT * FROM task_checkpoints WHERE task_id = ? "
            "ORDER BY revision DESC LIMIT 1"
            if revision is None
            else "SELECT * FROM task_checkpoints WHERE task_id = ? AND revision = ?"
        )
        parameters: tuple[object, ...] = (
            (normalized,) if revision is None else (normalized, revision)
        )
        with self._connection() as connection:
            row = connection.execute(query, parameters).fetchone()
        if row is None:
            raise KeyError(normalized)
        return _checkpoint_from_row(row)

    def history(self, task_id: str) -> tuple[TaskCheckpoint, ...]:
        normalized = _validate_task_id(task_id)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM task_checkpoints WHERE task_id = ? "
                "ORDER BY revision ASC",
                (normalized,),
            ).fetchall()
        if not rows:
            raise KeyError(normalized)
        return tuple(_checkpoint_from_row(row) for row in rows)

    def list_tasks(
        self,
        *,
        include_completed: bool = True,
        limit: int = 100,
    ) -> tuple[TaskCheckpoint, ...]:
        if not 1 <= limit <= 200:
            raise ValueError("task checkpoint limit must be between 1 and 200")
        status_filter = "" if include_completed else "WHERE current.status = 'active'"
        query = f"""
            SELECT current.*
            FROM task_checkpoints AS current
            JOIN (
                SELECT task_id, MAX(revision) AS revision
                FROM task_checkpoints
                GROUP BY task_id
            ) AS latest
              ON current.task_id = latest.task_id
             AND current.revision = latest.revision
            {status_filter}
            ORDER BY current.updated_at DESC, current.task_id DESC
            LIMIT ?
        """
        with self._connection() as connection:
            rows = connection.execute(query, (limit,)).fetchall()
        return tuple(_checkpoint_from_row(row) for row in rows)

    def retrieve_active_task(self, query: str) -> TaskCheckpointMatch | None:
        active = self.list_tasks(include_completed=False, limit=200)
        if not active:
            return None
        scored = sorted(
            (
                (_task_similarity(query, checkpoint), checkpoint)
                for checkpoint in active
            ),
            key=lambda item: (
                -item[0],
                item[1].title.casefold(),
                item[1].task_id,
            ),
        )
        if len(scored) == 1:
            score, checkpoint = scored[0]
            return TaskCheckpointMatch(checkpoint, score, "only_active_task")
        best_score, best = scored[0]
        second_score = scored[1][0]
        if best_score < 0.18 or best_score == second_score:
            return None
        return TaskCheckpointMatch(best, best_score, "lexical_match")

    def _insert(self, checkpoint: TaskCheckpoint) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO task_checkpoints("
                "task_id, revision, title, objective, status, progress, "
                "decisions_json, next_actions_json, source_session_id, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    checkpoint.task_id,
                    checkpoint.revision,
                    checkpoint.title,
                    checkpoint.objective,
                    checkpoint.status.value,
                    checkpoint.progress,
                    json.dumps(checkpoint.decisions, ensure_ascii=False),
                    json.dumps(checkpoint.next_actions, ensure_ascii=False),
                    checkpoint.source_session_id,
                    checkpoint.created_at,
                    checkpoint.updated_at,
                ),
            )


def _checkpoint_from_row(row: sqlite3.Row) -> TaskCheckpoint:
    decisions = json.loads(str(row["decisions_json"]))
    next_actions = json.loads(str(row["next_actions_json"]))
    if not isinstance(decisions, list) or not isinstance(next_actions, list):
        raise ValueError("task checkpoint list payload is invalid")
    return TaskCheckpoint(
        task_id=str(row["task_id"]),
        revision=int(row["revision"]),
        title=str(row["title"]),
        objective=str(row["objective"]),
        status=str(row["status"]),
        progress=str(row["progress"]),
        decisions=tuple(str(item) for item in decisions),
        next_actions=tuple(str(item) for item in next_actions),
        source_session_id=str(row["source_session_id"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _validate_task_id(task_id: str) -> str:
    normalized = str(task_id).strip()
    if _TASK_PATTERN.fullmatch(normalized) is None:
        raise ValueError("task checkpoint id is invalid")
    return normalized


def is_task_resume_request(text: str) -> bool:
    normalized = _normalize_text(text)
    names_a_task = "작업" in normalized or "task" in normalized
    if not names_a_task:
        return False
    if "장기 작업" in normalized or "선택한" in normalized:
        return any(
            marker in normalized for marker in ("다음", "이어", "계속", "재개")
        )
    return any(marker in normalized for marker in ("이어", "계속", "재개"))


def task_query_kind(
    text: str,
    *,
    allow_implicit: bool = False,
) -> TaskQueryKind | None:
    normalized = _normalize_text(text)
    names_a_task = any(
        marker in normalized
        for marker in ("작업", "task", "checkpoint", "체크포인트")
    )
    if not names_a_task and not allow_implicit:
        return None
    decision_query = any(marker in normalized for marker in ("결정", "합의"))
    next_query = any(
        marker in normalized
        for marker in (
            "다음 행동",
            "다음 단계",
            "다음으로",
            "뭘 해야",
            "무엇을 해야",
            "해야 할 행동",
        )
    )
    summary_query = any(
        marker in normalized
        for marker in (
            "어디까지",
            "진행 상황",
            "진행상황",
            "작업 상태",
            "현재 상태",
            "요약",
        )
    )
    if summary_query or (decision_query and next_query):
        return TaskQueryKind.SUMMARY
    if decision_query:
        return TaskQueryKind.DECISIONS
    if next_query or is_task_resume_request(text):
        return TaskQueryKind.NEXT_ACTION
    return None


def _task_similarity(query: str, checkpoint: TaskCheckpoint) -> float:
    left = _normalize_text(query)
    right = _normalize_text(f"{checkpoint.title} {checkpoint.objective}")
    if not left or not right:
        return 0.0
    left_words = set(left.split())
    right_words = set(right.split())
    word_coverage = len(left_words & right_words) / len(left_words)
    left_grams = _character_bigrams(left)
    right_grams = _character_bigrams(right)
    gram_coverage = (
        len(left_grams & right_grams) / len(left_grams)
        if left_grams
        else 0.0
    )
    return round(0.25 * word_coverage + 0.75 * gram_coverage, 6)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(normalized.split())


def _character_bigrams(value: str) -> set[str]:
    compact = "".join(character for character in value if character.isalnum())
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

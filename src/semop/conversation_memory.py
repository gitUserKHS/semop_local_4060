from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from typing import Protocol, Sequence
import unicodedata
from uuid import uuid4

from .prompt_api import (
    ConversationMessage,
    ConversationRole,
    RecalledEpisode,
    RecalledMemory,
)


_SESSION_PATTERN = re.compile(r"[A-Za-z0-9_-]{16,64}")
_MEMORY_PATTERN = re.compile(r"[0-9a-f]{32}")
_CORRECTION_PATTERN = re.compile(r"[0-9a-f]{32}")
_EPISODE_SELECT = (
    "SELECT user_turn_id, assistant_turn_id, session_id, user_content, "
    "assistant_content, answer_status, occurred_at FROM conversation_episodes "
)


class MemoryFacet(str, Enum):
    RESPONSE_LENGTH = "response_length"
    RESPONSE_ORDER = "response_order"
    GPU_HARDWARE = "gpu_hardware"
    PROJECT_GOAL = "project_goal"
    TEST_SCHEDULE = "test_schedule"


class MemorySemanticRanker(Protocol):
    def rank_texts(
        self,
        query: str,
        passages: Sequence[str],
        *,
        limit: int = 8,
    ) -> tuple[tuple[int, float], ...]: ...


@dataclass(frozen=True)
class ConversationTurn:
    turn_id: int
    session_id: str
    role: ConversationRole | str
    content: str
    status: str
    created_at: str

    def __post_init__(self) -> None:
        role = ConversationRole(self.role)
        _validate_session_id(self.session_id)
        if self.turn_id < 1:
            raise ValueError("conversation turn id must be positive")
        message = ConversationMessage(role, self.content, self.status)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", message.content)
        object.__setattr__(self, "status", message.status)

    def to_message(self) -> ConversationMessage:
        return ConversationMessage(self.role, self.content, self.status)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["role"] = self.role.value
        return payload


@dataclass(frozen=True)
class ExplicitMemory:
    memory_id: str
    content: str
    source_session_id: str
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        if _MEMORY_PATTERN.fullmatch(self.memory_id) is None:
            raise ValueError("explicit memory id is invalid")
        content = str(self.content).strip()
        if not content or len(content) > 1_000:
            raise ValueError("explicit memory content must be 1..1,000 characters")
        if self.source_session_id:
            _validate_session_id(self.source_session_id)
        if not self.created_at or not self.updated_at:
            raise ValueError("explicit memory timestamps cannot be empty")
        object.__setattr__(self, "content", content)

    def to_recalled(
        self,
        score: float,
        *,
        retrieval: str = "lexical",
    ) -> RecalledMemory:
        return RecalledMemory(
            memory_id=self.memory_id,
            content=self.content,
            score=score,
            saved_at=self.updated_at,
            retrieval=retrieval,
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class UserCorrection:
    correction_id: str
    prompt: str
    prior_answer: str
    corrected_answer: str
    source_session_id: str
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        if _CORRECTION_PATTERN.fullmatch(self.correction_id) is None:
            raise ValueError("user correction id is invalid")
        prompt = str(self.prompt).strip()
        prior_answer = str(self.prior_answer).strip()
        corrected_answer = str(self.corrected_answer).strip()
        if not prompt or len(prompt) > 16_000:
            raise ValueError("correction prompt must be 1..16,000 characters")
        if not prior_answer or len(prior_answer) > 16_000:
            raise ValueError("correction prior answer must be 1..16,000 characters")
        if not corrected_answer or len(corrected_answer) > 8_000:
            raise ValueError("corrected answer must be 1..8,000 characters")
        _validate_session_id(self.source_session_id)
        if not self.created_at or not self.updated_at:
            raise ValueError("user correction timestamps cannot be empty")
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "prior_answer", prior_answer)
        object.__setattr__(self, "corrected_answer", corrected_answer)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CorrectionPrototype:
    prototype_id: str
    corrected_answer: str
    support: int
    correction_ids: tuple[str, ...]
    prompts: tuple[str, ...]
    latest_updated_at: str

    def __post_init__(self) -> None:
        answer = str(self.corrected_answer).strip()
        if len(self.prototype_id) != 64:
            raise ValueError("correction prototype id must be SHA-256")
        if not answer or self.support < 3:
            raise ValueError("correction prototype requires at least three examples")
        if not (
            len(self.correction_ids) == len(self.prompts) == self.support
        ):
            raise ValueError("correction prototype evidence counts do not match")
        if len({_normalize_memory_text(item) for item in self.prompts}) != self.support:
            raise ValueError("correction prototype prompts must be distinct")
        expected = _correction_prototype_id(answer, self.correction_ids)
        if self.prototype_id != expected:
            raise ValueError("correction prototype id does not match its evidence")
        if not self.latest_updated_at:
            raise ValueError("correction prototype timestamp cannot be empty")
        object.__setattr__(self, "corrected_answer", answer)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CorrectionPrototypeMatch:
    prototype: CorrectionPrototype
    score: float
    matched_prompt: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("correction prototype score must be between 0 and 1")
        if self.matched_prompt not in self.prototype.prompts:
            raise ValueError("matched prompt is not prototype evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            **self.prototype.to_dict(),
            "score": self.score,
            "matched_prompt": self.matched_prompt,
        }


class ConversationStore:
    """Persistent episodic conversation memory with bounded working-memory recall."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.create_function(
            "semop_episode_terms",
            1,
            _episode_search_terms,
            deterministic=True,
        )
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
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversation_turn_session
                    ON conversation_turns(session_id, turn_id);
                CREATE TABLE IF NOT EXISTS conversation_episodes (
                    user_turn_id INTEGER PRIMARY KEY,
                    assistant_turn_id INTEGER NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    user_content TEXT NOT NULL,
                    assistant_content TEXT NOT NULL,
                    search_terms TEXT NOT NULL,
                    answer_status TEXT NOT NULL,
                    occurred_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversation_episode_session
                    ON conversation_episodes(session_id, user_turn_id DESC);
                CREATE VIRTUAL TABLE IF NOT EXISTS conversation_episode_fts
                    USING fts5(
                        search_terms,
                        content='conversation_episodes',
                        content_rowid='user_turn_id',
                        tokenize='unicode61'
                    );
                CREATE TRIGGER IF NOT EXISTS conversation_episode_fts_insert
                    AFTER INSERT ON conversation_episodes BEGIN
                        INSERT INTO conversation_episode_fts(rowid, search_terms)
                        VALUES (new.user_turn_id, new.search_terms);
                    END;
                CREATE TRIGGER IF NOT EXISTS conversation_episode_fts_delete
                    AFTER DELETE ON conversation_episodes BEGIN
                        INSERT INTO conversation_episode_fts(
                            conversation_episode_fts, rowid, search_terms
                        ) VALUES ('delete', old.user_turn_id, old.search_terms);
                    END;
                CREATE TRIGGER IF NOT EXISTS conversation_episode_fts_update
                    AFTER UPDATE ON conversation_episodes BEGIN
                        INSERT INTO conversation_episode_fts(
                            conversation_episode_fts, rowid, search_terms
                        ) VALUES ('delete', old.user_turn_id, old.search_terms);
                        INSERT INTO conversation_episode_fts(rowid, search_terms)
                        VALUES (new.user_turn_id, new.search_terms);
                    END;
                CREATE TABLE IF NOT EXISTS explicit_memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    normalized_content TEXT NOT NULL UNIQUE,
                    source_session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_explicit_memory_updated
                    ON explicit_memories(updated_at DESC);
                CREATE TABLE IF NOT EXISTS user_corrections (
                    correction_id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    normalized_prompt TEXT NOT NULL UNIQUE,
                    prior_answer TEXT NOT NULL,
                    corrected_answer TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_user_correction_updated
                    ON user_corrections(updated_at DESC);
                INSERT OR IGNORE INTO conversation_episodes(
                    user_turn_id,
                    assistant_turn_id,
                    session_id,
                    user_content,
                    assistant_content,
                    search_terms,
                    answer_status,
                    occurred_at
                )
                SELECT
                    user_turn.turn_id,
                    assistant_turn.turn_id,
                    user_turn.session_id,
                    user_turn.content,
                    assistant_turn.content,
                    semop_episode_terms(user_turn.content),
                    assistant_turn.status,
                    assistant_turn.created_at
                FROM conversation_turns AS user_turn
                JOIN conversation_turns AS assistant_turn
                    ON assistant_turn.turn_id = user_turn.turn_id + 1
                    AND assistant_turn.session_id = user_turn.session_id
                WHERE user_turn.role = 'user'
                    AND assistant_turn.role = 'assistant';
                """
            )

    def create_session(self) -> str:
        session_id = uuid4().hex
        self.ensure_session(session_id)
        return session_id

    def ensure_session(self, session_id: str = "") -> str:
        normalized = session_id.strip() or uuid4().hex
        _validate_session_id(normalized)
        timestamp = _timestamp()
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO conversation_sessions("
                "session_id, created_at, updated_at) VALUES (?, ?, ?)",
                (normalized, timestamp, timestamp),
            )
        return normalized

    def exists(self, session_id: str) -> bool:
        _validate_session_id(session_id)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM conversation_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row is not None

    def append_exchange(
        self,
        session_id: str,
        *,
        user_content: str,
        assistant_content: str,
        answer_status: str,
    ) -> tuple[ConversationTurn, ConversationTurn]:
        normalized = self.ensure_session(session_id)
        user = ConversationMessage(ConversationRole.USER, user_content)
        assistant = ConversationMessage(
            ConversationRole.ASSISTANT,
            assistant_content,
            answer_status,
        )
        timestamp = _timestamp()
        with self._connection() as connection:
            user_cursor = connection.execute(
                "INSERT INTO conversation_turns("
                "session_id, role, content, status, created_at"
                ") VALUES (?, ?, ?, ?, ?)",
                (normalized, user.role.value, user.content, user.status, timestamp),
            )
            assistant_cursor = connection.execute(
                "INSERT INTO conversation_turns("
                "session_id, role, content, status, created_at"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    normalized,
                    assistant.role.value,
                    assistant.content,
                    assistant.status,
                    timestamp,
                ),
            )
            connection.execute(
                "INSERT INTO conversation_episodes("
                "user_turn_id, assistant_turn_id, session_id, user_content, "
                "assistant_content, search_terms, answer_status, occurred_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(user_cursor.lastrowid),
                    int(assistant_cursor.lastrowid),
                    normalized,
                    user.content,
                    assistant.content,
                    _episode_search_terms(user.content),
                    assistant.status,
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE conversation_sessions SET updated_at = ? "
                "WHERE session_id = ?",
                (timestamp, normalized),
            )
        return (
            ConversationTurn(
                int(user_cursor.lastrowid),
                normalized,
                user.role,
                user.content,
                user.status,
                timestamp,
            ),
            ConversationTurn(
                int(assistant_cursor.lastrowid),
                normalized,
                assistant.role,
                assistant.content,
                assistant.status,
                timestamp,
            ),
        )

    def history(
        self,
        session_id: str,
        *,
        limit: int = 100,
    ) -> tuple[ConversationTurn, ...]:
        _validate_session_id(session_id)
        if not 1 <= limit <= 200:
            raise ValueError("conversation history limit must be between 1 and 200")
        if not self.exists(session_id):
            raise KeyError(session_id)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT turn_id, session_id, role, content, status, created_at "
                "FROM conversation_turns WHERE session_id = ? "
                "ORDER BY turn_id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return tuple(_turn_from_row(row) for row in reversed(rows))

    def recent_messages(
        self,
        session_id: str,
        *,
        max_messages: int = 8,
        max_chars: int = 6_000,
    ) -> tuple[ConversationMessage, ...]:
        return tuple(
            message
            for _turn, message in self._recent_context(
                session_id,
                max_messages=max_messages,
                max_chars=max_chars,
            )
        )

    def _recent_context(
        self,
        session_id: str,
        *,
        max_messages: int,
        max_chars: int,
    ) -> tuple[tuple[ConversationTurn, ConversationMessage], ...]:
        if not 1 <= max_messages <= 12:
            raise ValueError("working memory message limit must be between 1 and 12")
        if not 256 <= max_chars <= 12_000:
            raise ValueError("working memory character limit must be 256..12,000")
        turns = self.history(session_id, limit=min(64, max_messages * 4))
        selected: list[tuple[ConversationTurn, ConversationMessage]] = []
        used = 0
        for turn in reversed(turns):
            remaining = max_chars - used
            if remaining <= 0:
                break
            message = turn.to_message()
            if len(message.content) > remaining:
                if selected:
                    break
                prefix = "[앞부분 생략]\n"
                content = prefix + message.content[-(remaining - len(prefix)) :]
                message = ConversationMessage(message.role, content, message.status)
            selected.append((turn, message))
            used += len(message.content)
            if len(selected) >= max_messages:
                break
        return tuple(reversed(selected))

    def latest_exchange(
        self,
        session_id: str,
    ) -> tuple[ConversationTurn, ConversationTurn]:
        turns = self.history(session_id, limit=2)
        if (
            len(turns) != 2
            or turns[0].role is not ConversationRole.USER
            or turns[1].role is not ConversationRole.ASSISTANT
        ):
            raise KeyError(session_id)
        return turns[0], turns[1]

    def search_episodes(
        self,
        query: str,
        *,
        limit: int = 2,
        minimum_score: float = 0.30,
        exclude_session_id: str = "",
    ) -> tuple[RecalledEpisode, ...]:
        """Recall related exchanges from older sessions as untrusted context."""

        if not 1 <= limit <= 2:
            raise ValueError("recalled episode limit must be between 1 and 2")
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("recalled episode minimum score must be between 0 and 1")
        excluded = str(exclude_session_id).strip()
        if excluded:
            _validate_session_id(excluded)
        if not _normalize_memory_text(query):
            return ()

        with self._connection() as connection:
            indexed = self._search_indexed_episodes(
                connection,
                query,
                exclude_session_id=excluded,
            )
            if excluded:
                recent = connection.execute(
                    _EPISODE_SELECT
                    + "WHERE session_id != ? ORDER BY user_turn_id DESC LIMIT ?",
                    (excluded, 400),
                ).fetchall()
            else:
                recent = connection.execute(
                    _EPISODE_SELECT
                    + "ORDER BY user_turn_id DESC LIMIT ?",
                    (400,),
                ).fetchall()
            rows = _deduplicate_episode_rows((*indexed, *recent))

        return _rank_episode_rows(
            query,
            rows,
            limit=limit,
            minimum_score=minimum_score,
            scope="cross_session",
        )

    def search_archived_session_episodes(
        self,
        query: str,
        session_id: str,
        *,
        limit: int = 2,
        minimum_score: float = 0.30,
        working_memory_messages: int = 8,
        working_memory_chars: int = 6_000,
    ) -> tuple[RecalledEpisode, ...]:
        """Recall same-session exchanges that have left the working-memory window."""

        _validate_session_id(session_id)
        if not 1 <= limit <= 2:
            raise ValueError("recalled episode limit must be between 1 and 2")
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("recalled episode minimum score must be between 0 and 1")
        if not _normalize_memory_text(query):
            return ()
        recent = self._recent_context(
            session_id,
            max_messages=working_memory_messages,
            max_chars=working_memory_chars,
        )
        if not recent:
            return ()
        oldest_working_turn = min(turn.turn_id for turn, _message in recent)
        with self._connection() as connection:
            indexed = self._search_indexed_episodes(
                connection,
                query,
                session_id=session_id,
                before_assistant_turn=oldest_working_turn,
            )
            recent = connection.execute(
                _EPISODE_SELECT
                + "WHERE session_id = ? AND assistant_turn_id < ? "
                "ORDER BY user_turn_id DESC LIMIT ?",
                (session_id, oldest_working_turn, 400),
            ).fetchall()
            rows = _deduplicate_episode_rows((*indexed, *recent))
        return _rank_episode_rows(
            query,
            rows,
            limit=limit,
            minimum_score=minimum_score,
            scope="same_session_archive",
        )

    def _search_indexed_episodes(
        self,
        connection: sqlite3.Connection,
        query: str,
        *,
        session_id: str = "",
        exclude_session_id: str = "",
        before_assistant_turn: int | None = None,
        limit: int = 400,
    ) -> list[sqlite3.Row]:
        expression = _episode_fts_query(query)
        if not expression:
            return []
        clauses = ["conversation_episode_fts MATCH ?"]
        parameters: list[object] = [expression]
        if session_id:
            clauses.append("episode.session_id = ?")
            parameters.append(session_id)
        if exclude_session_id:
            clauses.append("episode.session_id != ?")
            parameters.append(exclude_session_id)
        if before_assistant_turn is not None:
            clauses.append("episode.assistant_turn_id < ?")
            parameters.append(before_assistant_turn)
        parameters.append(limit)
        return connection.execute(
            "SELECT episode.user_turn_id, episode.assistant_turn_id, "
            "episode.session_id, episode.user_content, "
            "episode.assistant_content, episode.answer_status, "
            "episode.occurred_at "
            "FROM conversation_episode_fts "
            "JOIN conversation_episodes AS episode "
            "ON episode.user_turn_id = conversation_episode_fts.rowid "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY bm25(conversation_episode_fts), "
            "episode.user_turn_id DESC LIMIT ?",
            tuple(parameters),
        ).fetchall()

    def recall_relevant_episodes(
        self,
        query: str,
        *,
        current_session_id: str,
        limit: int = 2,
        minimum_score: float = 0.30,
        working_memory_messages: int = 8,
        working_memory_chars: int = 6_000,
    ) -> tuple[RecalledEpisode, ...]:
        """Combine archived current-session and cross-session episodic recall."""

        if not 1 <= limit <= 2:
            raise ValueError("recalled episode limit must be between 1 and 2")
        archived = self.search_archived_session_episodes(
            query,
            current_session_id,
            limit=limit,
            minimum_score=minimum_score,
            working_memory_messages=working_memory_messages,
            working_memory_chars=working_memory_chars,
        )
        cross_session = self.search_episodes(
            query,
            limit=limit,
            minimum_score=minimum_score,
            exclude_session_id=current_session_id,
        )
        status_priority = {"verified": 2, "conditional": 1}
        combined = list((*archived, *cross_session))
        combined.sort(
            key=lambda item: (
                item.score,
                status_priority.get(item.answer_status, 0),
            ),
            reverse=True,
        )
        seen: set[str] = set()
        selected: list[RecalledEpisode] = []
        for episode in combined:
            if episode.episode_id in seen:
                continue
            seen.add(episode.episode_id)
            selected.append(episode)
            if len(selected) >= limit:
                break
        return tuple(selected)

    def remember(
        self,
        content: str,
        *,
        source_session_id: str = "",
    ) -> ExplicitMemory:
        normalized_content = _normalize_memory_text(content)
        if not normalized_content or len(str(content).strip()) > 1_000:
            raise ValueError("explicit memory content must be 1..1,000 characters")
        source = source_session_id.strip()
        if source and not self.exists(source):
            raise KeyError(source)
        memory_id = uuid4().hex
        timestamp = _timestamp()
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO explicit_memories("
                "memory_id, content, normalized_content, source_session_id, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    memory_id,
                    str(content).strip(),
                    normalized_content,
                    source,
                    timestamp,
                    timestamp,
                ),
            )
            row = connection.execute(
                "SELECT memory_id, content, source_session_id, created_at, updated_at "
                "FROM explicit_memories WHERE normalized_content = ?",
                (normalized_content,),
            ).fetchone()
        assert row is not None
        return _memory_from_row(row)

    def list_memories(self, *, limit: int = 100) -> tuple[ExplicitMemory, ...]:
        if not 1 <= limit <= 200:
            raise ValueError("explicit memory limit must be between 1 and 200")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT memory_id, content, source_session_id, created_at, updated_at "
                "FROM explicit_memories ORDER BY updated_at DESC, memory_id DESC "
                "LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(_memory_from_row(row) for row in rows)

    def search_memories(
        self,
        query: str,
        *,
        limit: int = 3,
        minimum_score: float = 0.18,
        semantic_ranker: MemorySemanticRanker | None = None,
        semantic_minimum_score: float = 0.75,
    ) -> tuple[RecalledMemory, ...]:
        if not 1 <= limit <= 4:
            raise ValueError("recalled memory limit must be between 1 and 4")
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("recalled memory minimum score must be between 0 and 1")
        if not 0.0 <= semantic_minimum_score <= 1.0:
            raise ValueError("semantic memory minimum score must be between 0 and 1")
        if not _normalize_memory_text(query):
            return ()
        memories = self.list_memories(limit=200)
        matches = [
            (memory, _memory_similarity(query, memory.content))
            for memory in memories
        ]
        matches = [item for item in matches if item[1] >= minimum_score]
        matches.sort(
            key=lambda item: (item[1], item[0].updated_at, item[0].memory_id),
            reverse=True,
        )
        if matches or semantic_ranker is None:
            return tuple(
                memory.to_recalled(score)
                for memory, score in matches[:limit]
            )

        query_facets = classify_memory_facets(query)
        if not query_facets:
            return ()
        semantic_candidates = tuple(
            memory
            for memory in memories
            if query_facets.intersection(classify_memory_facets(memory.content))
        )
        if not semantic_candidates:
            return ()
        ranked = semantic_ranker.rank_texts(
            query,
            tuple(memory.content for memory in semantic_candidates),
            limit=min(limit, len(semantic_candidates)),
        )
        recalled: list[RecalledMemory] = []
        for index, score in ranked:
            if not 0 <= index < len(semantic_candidates):
                raise ValueError("semantic memory ranker returned an invalid index")
            if not semantic_minimum_score <= score <= 1.0:
                continue
            recalled.append(
                semantic_candidates[index].to_recalled(
                    round(score, 6),
                    retrieval="semantic_e5",
                )
            )
        return tuple(recalled)

    def forget(self, memory_id: str) -> bool:
        normalized = str(memory_id).strip()
        if _MEMORY_PATTERN.fullmatch(normalized) is None:
            raise ValueError("explicit memory id is invalid")
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM explicit_memories WHERE memory_id = ?",
                (normalized,),
            )
        return cursor.rowcount == 1

    def remember_correction(
        self,
        *,
        prompt: str,
        prior_answer: str,
        corrected_answer: str,
        source_session_id: str,
    ) -> UserCorrection:
        normalized_prompt = _normalize_memory_text(prompt)
        source = source_session_id.strip()
        if not normalized_prompt:
            raise ValueError("correction prompt cannot be empty")
        if not self.exists(source):
            raise KeyError(source)
        candidate = UserCorrection(
            correction_id=uuid4().hex,
            prompt=prompt,
            prior_answer=prior_answer,
            corrected_answer=corrected_answer,
            source_session_id=source,
            created_at=_timestamp(),
            updated_at=_timestamp(),
        )
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT correction_id, created_at FROM user_corrections "
                "WHERE normalized_prompt = ?",
                (normalized_prompt,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO user_corrections("
                    "correction_id, prompt, normalized_prompt, prior_answer, "
                    "corrected_answer, source_session_id, created_at, updated_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        candidate.correction_id,
                        candidate.prompt,
                        normalized_prompt,
                        candidate.prior_answer,
                        candidate.corrected_answer,
                        candidate.source_session_id,
                        candidate.created_at,
                        candidate.updated_at,
                    ),
                )
            else:
                connection.execute(
                    "UPDATE user_corrections SET prompt = ?, prior_answer = ?, "
                    "corrected_answer = ?, source_session_id = ?, updated_at = ? "
                    "WHERE normalized_prompt = ?",
                    (
                        candidate.prompt,
                        candidate.prior_answer,
                        candidate.corrected_answer,
                        candidate.source_session_id,
                        candidate.updated_at,
                        normalized_prompt,
                    ),
                )
            row = connection.execute(
                "SELECT correction_id, prompt, prior_answer, corrected_answer, "
                "source_session_id, created_at, updated_at FROM user_corrections "
                "WHERE normalized_prompt = ?",
                (normalized_prompt,),
            ).fetchone()
        assert row is not None
        return _correction_from_row(row)

    def find_correction(self, prompt: str) -> UserCorrection | None:
        normalized_prompt = _normalize_memory_text(prompt)
        if not normalized_prompt:
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT correction_id, prompt, prior_answer, corrected_answer, "
                "source_session_id, created_at, updated_at FROM user_corrections "
                "WHERE normalized_prompt = ?",
                (normalized_prompt,),
            ).fetchone()
        return _correction_from_row(row) if row is not None else None

    def list_corrections(self, *, limit: int = 100) -> tuple[UserCorrection, ...]:
        if not 1 <= limit <= 200:
            raise ValueError("user correction limit must be between 1 and 200")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT correction_id, prompt, prior_answer, corrected_answer, "
                "source_session_id, created_at, updated_at FROM user_corrections "
                "ORDER BY updated_at DESC, correction_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(_correction_from_row(row) for row in rows)

    def correction_prototypes(
        self,
        *,
        min_support: int = 3,
    ) -> tuple[CorrectionPrototype, ...]:
        if type(min_support) is not int or min_support < 3:
            raise ValueError("correction prototype support must be at least 3")
        grouped: dict[str, list[UserCorrection]] = {}
        for correction in self.list_corrections(limit=200):
            key = _normalize_memory_text(correction.corrected_answer)
            grouped.setdefault(key, []).append(correction)

        prototypes: list[CorrectionPrototype] = []
        for evidence in grouped.values():
            if len(evidence) < min_support:
                continue
            newest = max(
                evidence,
                key=lambda item: (item.updated_at, item.correction_id),
            )
            ordered = sorted(
                evidence,
                key=lambda item: (
                    _normalize_memory_text(item.prompt),
                    item.correction_id,
                ),
            )
            correction_ids = tuple(item.correction_id for item in ordered)
            prompts = tuple(item.prompt for item in ordered)
            prototypes.append(
                CorrectionPrototype(
                    prototype_id=_correction_prototype_id(
                        newest.corrected_answer,
                        correction_ids,
                    ),
                    corrected_answer=newest.corrected_answer,
                    support=len(ordered),
                    correction_ids=correction_ids,
                    prompts=prompts,
                    latest_updated_at=newest.updated_at,
                )
            )
        prototypes.sort(
            key=lambda item: (
                -item.support,
                item.prototype_id,
            )
        )
        return tuple(prototypes)

    def search_correction_prototypes(
        self,
        prompt: str,
        *,
        limit: int = 1,
        minimum_score: float = 0.35,
    ) -> tuple[CorrectionPrototypeMatch, ...]:
        if not 1 <= limit <= 4:
            raise ValueError("correction prototype limit must be between 1 and 4")
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("correction prototype score must be between 0 and 1")
        if not _normalize_memory_text(prompt):
            return ()
        matches: list[CorrectionPrototypeMatch] = []
        for prototype in self.correction_prototypes():
            scored = sorted(
                (
                    (_memory_similarity(prompt, example), example)
                    for example in prototype.prompts
                ),
                key=lambda item: (-item[0], _normalize_memory_text(item[1])),
            )
            score, matched_prompt = scored[0]
            if score < minimum_score:
                continue
            matches.append(
                CorrectionPrototypeMatch(
                    prototype=prototype,
                    score=score,
                    matched_prompt=matched_prompt,
                )
            )
        matches.sort(
            key=lambda item: (
                -item.score,
                -item.prototype.support,
                item.prototype.prototype_id,
            )
        )
        return tuple(matches[:limit])


def _turn_from_row(row: sqlite3.Row) -> ConversationTurn:
    return ConversationTurn(
        turn_id=int(row["turn_id"]),
        session_id=str(row["session_id"]),
        role=str(row["role"]),
        content=str(row["content"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
    )


def _memory_from_row(row: sqlite3.Row) -> ExplicitMemory:
    return ExplicitMemory(
        memory_id=str(row["memory_id"]),
        content=str(row["content"]),
        source_session_id=str(row["source_session_id"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _correction_from_row(row: sqlite3.Row) -> UserCorrection:
    return UserCorrection(
        correction_id=str(row["correction_id"]),
        prompt=str(row["prompt"]),
        prior_answer=str(row["prior_answer"]),
        corrected_answer=str(row["corrected_answer"]),
        source_session_id=str(row["source_session_id"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _normalize_memory_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(normalized.split())


def classify_memory_facets(value: str) -> frozenset[MemoryFacet]:
    text = _normalize_memory_text(value)
    facets: set[MemoryFacet] = set()
    response_cues = ("답변", "설명", "response", "answer", "explanation")
    if _contains_any(text, response_cues) and _contains_any(
        text,
        ("짧", "간결", "길이", "길게", "concise", "brief", "length"),
    ):
        facets.add(MemoryFacet.RESPONSE_LENGTH)
    if _contains_any(text, ("결론", "순서", "먼저", "핵심부터", "order", "first")):
        facets.add(MemoryFacet.RESPONSE_ORDER)
    if _contains_any(
        text,
        ("그래픽카드", "gpu", "vram", "rtx", "graphics card"),
    ):
        facets.add(MemoryFacet.GPU_HARDWARE)
    if _contains_any(text, ("목표", "방향", "핵심", "목적", "goal", "purpose")) and _contains_any(
        text,
        (
            "프로젝트",
            "우리",
            "만들",
            "지능",
            "모델",
            "연산자",
            "project",
            "intelligence",
            "operator",
        ),
    ):
        facets.add(MemoryFacet.PROJECT_GOAL)
    if _contains_any(
        text,
        ("매주", "일요일", "언제", "일정", "요일", "weekly", "sunday", "schedule"),
    ) and _contains_any(
        text,
        ("테스트", "검사", "회귀", "실행", "test", "regression", "run"),
    ):
        facets.add(MemoryFacet.TEST_SCHEDULE)
    return frozenset(facets)


def _contains_any(text: str, candidates: Sequence[str]) -> bool:
    return any(candidate in text for candidate in candidates)


def _episode_search_terms(value: object) -> str:
    normalized = _normalize_memory_text(str(value))[:4_000]
    if not normalized:
        return ""
    words = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    grams = (
        f"g{gram.encode('utf-8').hex()}"
        for gram in sorted(_character_bigrams(normalized))
    )
    return " ".join(dict.fromkeys((*words, *grams)))


def _episode_fts_query(value: str) -> str:
    terms = _episode_search_terms(value).split()
    return " OR ".join(f'"{term}"' for term in terms[:96])


def _deduplicate_episode_rows(
    rows: tuple[sqlite3.Row, ...],
) -> list[sqlite3.Row]:
    selected: dict[int, sqlite3.Row] = {}
    for row in rows:
        selected.setdefault(int(row["user_turn_id"]), row)
    return list(selected.values())


def _episode_id(session_id: str, user_turn_id: int, assistant_turn_id: int) -> str:
    payload = f"{session_id}:{user_turn_id}:{assistant_turn_id}".encode("utf-8")
    return sha256(payload).hexdigest()


def _rank_episode_rows(
    query: str,
    rows: list[sqlite3.Row],
    *,
    limit: int,
    minimum_score: float,
    scope: str,
) -> tuple[RecalledEpisode, ...]:
    matches: list[tuple[float, int, int, RecalledEpisode]] = []
    status_priority = {"verified": 2, "conditional": 1}
    for row in rows:
        score = _memory_similarity(query, str(row["user_content"]))
        if score < minimum_score:
            continue
        answer_status = str(row["answer_status"]).strip() or "unknown"
        user_turn_id = int(row["user_turn_id"])
        assistant_turn_id = int(row["assistant_turn_id"])
        episode = RecalledEpisode(
            episode_id=_episode_id(
                str(row["session_id"]),
                user_turn_id,
                assistant_turn_id,
            ),
            user_content=_bounded_episode_content(str(row["user_content"])),
            assistant_content=_bounded_episode_content(str(row["assistant_content"])),
            answer_status=answer_status,
            score=score,
            occurred_at=str(row["occurred_at"]),
            scope=scope,
        )
        matches.append(
            (
                score,
                status_priority.get(answer_status, 0),
                user_turn_id,
                episode,
            )
        )
    matches.sort(key=lambda item: item[:3], reverse=True)
    return tuple(item[3] for item in matches[:limit])


def _bounded_episode_content(value: str, *, limit: int = 4_000) -> str:
    content = str(value).strip()
    if len(content) <= limit:
        return content
    marker = "[앞부분 생략]\n"
    return marker + content[-(limit - len(marker)) :]


def _correction_prototype_id(
    corrected_answer: str,
    correction_ids: tuple[str, ...],
) -> str:
    payload = {
        "corrected_answer": _normalize_memory_text(corrected_answer),
        "correction_ids": tuple(sorted(correction_ids)),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _memory_similarity(query: str, memory: str) -> float:
    left = _normalize_memory_text(query)
    right = _normalize_memory_text(memory)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_words = set(left.split())
    right_words = set(right.split())
    word_coverage = (
        len(left_words.intersection(right_words)) / len(left_words)
        if left_words
        else 0.0
    )
    left_grams = _character_bigrams(left)
    right_grams = _character_bigrams(right)
    gram_coverage = (
        len(left_grams.intersection(right_grams)) / len(left_grams)
        if left_grams
        else 0.0
    )
    return round(0.25 * word_coverage + 0.75 * gram_coverage, 6)


def _character_bigrams(value: str) -> set[str]:
    compact = "".join(character for character in value if character.isalnum())
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _validate_session_id(session_id: str) -> None:
    if _SESSION_PATTERN.fullmatch(session_id) is None:
        raise ValueError("conversation session id is invalid")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

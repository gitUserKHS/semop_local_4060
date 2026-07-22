from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
import sqlite3
from time import perf_counter
import tracemalloc
from typing import Protocol

from .conversation_memory import ConversationStore, MemorySemanticRanker
from .prompt_api import AnswerEnvelope, PromptRequest, ResourceTier
from .task_memory import TaskCheckpointStore


@dataclass(frozen=True)
class MemoryContinuityConfig:
    exchanges: int = 50
    working_messages: int = 8
    task_updates: int = 10
    distractor_memories: int = 24

    def __post_init__(self) -> None:
        if not 1 <= self.exchanges <= 100:
            raise ValueError("memory evaluation exchanges must be between 1 and 100")
        if not 1 <= self.working_messages <= 12:
            raise ValueError("working-memory messages must be between 1 and 12")
        if not 1 <= self.task_updates <= 50:
            raise ValueError("task updates must be between 1 and 50")
        if not 0 <= self.distractor_memories <= 100:
            raise ValueError("distractor memories must be between 0 and 100")


@dataclass(frozen=True)
class MemoryRecallResult:
    case_id: str
    query: str
    expected_memory_id: str
    recalled_memory_id: str
    recalled_content: str
    score: float
    correct: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryContinuityReport:
    exchanges: int
    stored_turns: int
    restored_turns: int
    expected_working_messages: tuple[str, ...]
    restored_working_messages: tuple[str, ...]
    recall_results: tuple[MemoryRecallResult, ...]
    cross_session_episode_recalled: bool
    cross_session_episode_score: float
    same_session_episode_excluded: bool
    same_session_archive_tested: bool
    same_session_archived_episode_recalled: bool
    same_session_archived_episode_score: float
    recent_working_episode_excluded: bool
    unrelated_queries: int
    unrelated_false_recalls: int
    expected_task_revision: int
    restored_task_revision: int
    task_history_revisions: tuple[int, ...]
    completion_immutable: bool
    database_bytes: int
    peak_python_bytes: int
    elapsed_seconds: float
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    @property
    def recall_at_1(self) -> float:
        if not self.recall_results:
            return 0.0
        return sum(item.correct for item in self.recall_results) / len(
            self.recall_results
        )

    @property
    def unrelated_false_recall_rate(self) -> float:
        if self.unrelated_queries == 0:
            return 0.0
        return self.unrelated_false_recalls / self.unrelated_queries

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "passed": self.passed,
            "recall_at_1": self.recall_at_1,
            "unrelated_false_recall_rate": self.unrelated_false_recall_rate,
        }


@dataclass(frozen=True)
class LongHistoryEpisodicReport:
    exchanges: int
    indexed_episodes: int
    target_answer: str
    cross_session_answer: str
    same_session_answer: str
    cross_session_recalled: bool
    same_session_recalled: bool
    backfill_seconds: float
    cross_session_seconds: float
    same_session_seconds: float
    database_bytes: int
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "passed": self.passed}


@dataclass(frozen=True)
class SemanticParaphraseMemoryReport:
    positive_cases: int
    lexical_correct: int
    hybrid_correct: int
    unrelated_queries: int
    unrelated_false_recalls: int
    hybrid_scores: tuple[float, ...]
    hybrid_retrievals: tuple[str, ...]
    elapsed_seconds: float
    warm_replay_seconds: float
    warm_correct: int

    @property
    def lexical_recall_at_1(self) -> float:
        return self.lexical_correct / self.positive_cases

    @property
    def hybrid_recall_at_1(self) -> float:
        return self.hybrid_correct / self.positive_cases

    @property
    def unrelated_false_recall_rate(self) -> float:
        return self.unrelated_false_recalls / self.unrelated_queries

    @property
    def measured_gain(self) -> bool:
        return self.hybrid_correct > self.lexical_correct

    @property
    def passed(self) -> bool:
        return (
            self.hybrid_correct == self.positive_cases
            and self.warm_correct == self.positive_cases
            and self.unrelated_false_recalls == 0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "lexical_recall_at_1": self.lexical_recall_at_1,
            "hybrid_recall_at_1": self.hybrid_recall_at_1,
            "unrelated_false_recall_rate": self.unrelated_false_recall_rate,
            "measured_gain": self.measured_gain,
            "passed": self.passed,
        }


class PromptAssistant(Protocol):
    def solve(self, request: PromptRequest) -> AnswerEnvelope: ...


@dataclass(frozen=True)
class SemanticMemoryABReport:
    query: str
    expected_marker: str
    recalled_memory_ids: tuple[str, ...]
    recall_score: float
    without_memory_answer: str
    with_memory_answer: str
    without_memory_status: str
    with_memory_status: str
    without_generated_tokens: int
    with_generated_tokens: int
    without_hit_token_limit: bool
    with_hit_token_limit: bool
    without_repair_attempts: int
    with_repair_attempts: int
    without_memory_seconds: float
    with_memory_seconds: float
    without_memory_contains_marker: bool
    with_memory_contains_marker: bool
    memory_context_reported: bool

    @property
    def passed(self) -> bool:
        return bool(
            self.recalled_memory_ids
            and self.with_memory_contains_marker
            and self.memory_context_reported
        )

    @property
    def measured_gain(self) -> bool:
        return self.with_memory_contains_marker and not self.without_memory_contains_marker

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "passed": self.passed,
            "measured_gain": self.measured_gain,
        }


@dataclass(frozen=True)
class EpisodicMemoryABReport:
    query: str
    expected_marker: str
    recalled_episode_ids: tuple[str, ...]
    recalled_episode_scopes: tuple[str, ...]
    recall_score: float
    without_episode_answer: str
    with_episode_answer: str
    without_episode_status: str
    with_episode_status: str
    without_generated_tokens: int
    with_generated_tokens: int
    without_hit_token_limit: bool
    with_hit_token_limit: bool
    without_repair_attempts: int
    with_repair_attempts: int
    without_episode_seconds: float
    with_episode_seconds: float
    without_episode_contains_marker: bool
    with_episode_contains_marker: bool
    episode_context_reported: bool

    @property
    def passed(self) -> bool:
        return bool(
            self.recalled_episode_ids
            and self.with_episode_contains_marker
            and self.episode_context_reported
        )

    @property
    def measured_gain(self) -> bool:
        return (
            self.with_episode_contains_marker
            and not self.without_episode_contains_marker
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "passed": self.passed,
            "measured_gain": self.measured_gain,
        }


@dataclass(frozen=True)
class TaskContextABReport:
    query: str
    task_id: str
    task_revision: int
    expected_marker: str
    stale_marker: str
    without_context_answer: str
    with_context_answer: str
    without_context_status: str
    with_context_status: str
    without_context_seconds: float
    with_context_seconds: float
    without_context_contains_marker: bool
    with_context_contains_marker: bool
    with_context_contains_stale_marker: bool
    task_context_reported: bool
    summary_query: str
    summary_answer: str
    summary_seconds: float
    summary_contains_revision: bool
    summary_contains_progress: bool
    summary_contains_decision: bool
    summary_contains_next_action: bool
    summary_context_reported: bool

    @property
    def passed(self) -> bool:
        return bool(
            self.with_context_contains_marker
            and not self.with_context_contains_stale_marker
            and self.task_context_reported
            and self.summary_contains_revision
            and self.summary_contains_progress
            and self.summary_contains_decision
            and self.summary_contains_next_action
            and self.summary_context_reported
        )

    @property
    def measured_gain(self) -> bool:
        return self.with_context_contains_marker and not self.without_context_contains_marker

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "passed": self.passed,
            "measured_gain": self.measured_gain,
        }


@dataclass(frozen=True)
class CorrectionConsolidationReport:
    query: str
    expected_answer: str
    examples_before: int
    examples_after: int
    matched_before_consolidation: bool
    matched_after_consolidation: bool
    prototype_id: str
    prototype_support: int
    matched_prompt: str
    score: float
    recalled_answer: str
    restart_persistent: bool
    elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return bool(
            not self.matched_before_consolidation
            and self.matched_after_consolidation
            and self.prototype_support == self.examples_after
            and self.recalled_answer == self.expected_answer
            and self.restart_persistent
        )

    @property
    def measured_gain(self) -> bool:
        return self.matched_after_consolidation and not self.matched_before_consolidation

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "passed": self.passed,
            "measured_gain": self.measured_gain,
        }


@dataclass(frozen=True)
class TaskAutoResumeReport:
    ambiguous_query: str
    named_query: str
    target_task_id: str
    target_revision: int
    expected_action: str
    stale_action: str
    ambiguous_query_rejected: bool
    named_query_selected: bool
    selected_task_id: str
    selected_revision: int
    selected_action: str
    selected_contains_stale_action: bool
    score: float
    reason: str
    restart_persistent: bool
    elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return bool(
            self.ambiguous_query_rejected
            and self.named_query_selected
            and self.selected_task_id == self.target_task_id
            and self.selected_revision == self.target_revision
            and self.selected_action == self.expected_action
            and not self.selected_contains_stale_action
            and self.restart_persistent
        )

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "passed": self.passed}


def evaluate_memory_continuity(
    database: str | Path,
    *,
    config: MemoryContinuityConfig | None = None,
) -> MemoryContinuityReport:
    """Exercise persistent memory and task resumption without loading a model."""

    settings = config or MemoryContinuityConfig()
    path = Path(database)
    if path.exists():
        raise ValueError("memory evaluation database must not already exist")
    path.parent.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    tracemalloc.start()
    try:
        conversation = ConversationStore(path)
        session_id = conversation.create_session()
        expected_turns: list[str] = []
        for index in range(settings.exchanges):
            user = f"장기 대화 단계 {index:03d} 사용자 요청"
            assistant = f"장기 대화 단계 {index:03d} SemOp 응답"
            conversation.append_exchange(
                session_id,
                user_content=user,
                assistant_content=assistant,
                answer_status="best_effort",
            )
            expected_turns.extend((user, assistant))

        memory_specs = (
            (
                "project",
                "내 프로젝트 코드명은 별빛-27이야",
                "프로젝트 코드명이 뭐였지?",
            ),
            (
                "hardware",
                "내 그래픽카드는 RTX 4060 8GB야",
                "내 그래픽카드 사양을 기억해?",
            ),
            (
                "style",
                "답변은 한국어로 짧고 명확하게 해줘",
                "답변은 어떤 언어로 짧게 해 달랬지?",
            ),
        )
        expected_memories = {
            case_id: conversation.remember(content, source_session_id=session_id)
            for case_id, content, _query in memory_specs
        }
        for index in range(settings.distractor_memories):
            conversation.remember(
                f"보관용 참고 항목 {index:03d}: 분류-{index:03d}",
                source_session_id=session_id,
            )

        task_store = TaskCheckpointStore(path)
        task = task_store.create_task(
            title="SemOp 장기기억 실측",
            objective="여러 세션과 revision에 걸쳐 작업 상태를 정확히 복원한다",
            source_session_id=session_id,
        )
        for index in range(1, settings.task_updates + 1):
            task = task_store.checkpoint(
                task.task_id,
                progress=f"장기 작업 단계 {index:03d} 완료",
                decisions=(f"결정 {index:03d}을 유지",),
                next_actions=(f"단계 {index + 1:03d} 진행",),
            )

        reopened_conversation = ConversationStore(path)
        restored_history = reopened_conversation.history(session_id, limit=200)
        restored_turns = tuple(item.content for item in restored_history)
        restored_working = tuple(
            item.content
            for item in reopened_conversation.recent_messages(
                session_id,
                max_messages=settings.working_messages,
            )
        )
        expected_working = tuple(expected_turns[-settings.working_messages :])
        recall_session_id = reopened_conversation.create_session()
        episode_query = "장기 대화 단계 000 사용자 요청"
        recalled_episodes = reopened_conversation.search_episodes(
            episode_query,
            limit=1,
            exclude_session_id=recall_session_id,
        )
        cross_session_episode_recalled = bool(
            recalled_episodes
            and recalled_episodes[0].user_content == episode_query
        )
        cross_session_episode_score = (
            recalled_episodes[0].score if recalled_episodes else 0.0
        )
        same_session_episode_excluded = not reopened_conversation.search_episodes(
            episode_query,
            limit=1,
            exclude_session_id=session_id,
        )
        same_session_archive_tested = (
            len(expected_turns) > settings.working_messages
        )
        archived_episodes = reopened_conversation.search_archived_session_episodes(
            episode_query,
            session_id,
            limit=1,
            working_memory_messages=settings.working_messages,
        )
        same_session_archived_episode_recalled = bool(
            archived_episodes
            and archived_episodes[0].user_content == episode_query
            and archived_episodes[0].scope == "same_session_archive"
        )
        same_session_archived_episode_score = (
            archived_episodes[0].score if archived_episodes else 0.0
        )
        latest_episode_query = (
            f"장기 대화 단계 {settings.exchanges - 1:03d} 사용자 요청"
        )
        recent_query_matches = (
            reopened_conversation.search_archived_session_episodes(
                latest_episode_query,
                session_id,
                limit=2,
                working_memory_messages=settings.working_messages,
            )
        )
        recent_working_episode_excluded = all(
            item.user_content != latest_episode_query
            for item in recent_query_matches
        )

        recall_results: list[MemoryRecallResult] = []
        for case_id, _content, query in memory_specs:
            matches = reopened_conversation.search_memories(query, limit=1)
            recalled = matches[0] if matches else None
            expected = expected_memories[case_id]
            recall_results.append(
                MemoryRecallResult(
                    case_id=case_id,
                    query=query,
                    expected_memory_id=expected.memory_id,
                    recalled_memory_id=(recalled.memory_id if recalled else ""),
                    recalled_content=(recalled.content if recalled else ""),
                    score=(recalled.score if recalled else 0.0),
                    correct=bool(recalled and recalled.memory_id == expected.memory_id),
                )
            )

        unrelated = (
            "오늘 비가 오는지 알려줘",
            "파스타 조리법을 찾아줘",
            "야구 경기 결과가 궁금해",
        )
        unrelated_false_recalls = sum(
            bool(reopened_conversation.search_memories(query, limit=1))
            for query in unrelated
        )

        reopened_tasks = TaskCheckpointStore(path)
        restored_task = reopened_tasks.get(task.task_id)
        completed = reopened_tasks.complete_task(
            task.task_id,
            progress="장기 작업 지속성 평가 완료",
            decisions=("복원된 최종 revision을 검증",),
        )
        completion_immutable = False
        try:
            reopened_tasks.checkpoint(
                task.task_id,
                progress="완료 후 변경 시도",
            )
        except ValueError:
            completion_immutable = True
        history_revisions = tuple(
            item.revision for item in reopened_tasks.history(task.task_id)
        )

        violations: list[str] = []
        if restored_turns != tuple(expected_turns):
            violations.append("episodic history changed after reopening the SQLite store")
        if restored_working != expected_working:
            violations.append("bounded working memory did not retain the latest messages")
        if any(not item.correct for item in recall_results):
            violations.append("explicit long-term memory recall@1 was below 100%")
        if not cross_session_episode_recalled:
            violations.append("related episode was not recalled across sessions")
        if not same_session_episode_excluded:
            violations.append("episode recall duplicated the current session history")
        if (
            same_session_archive_tested
            and not same_session_archived_episode_recalled
        ):
            violations.append(
                "an old same-session episode was not recalled outside working memory"
            )
        if not recent_working_episode_excluded:
            violations.append("episodic archive duplicated a recent working-memory turn")
        if unrelated_false_recalls:
            violations.append("unrelated queries recalled an explicit memory")
        if restored_task.revision != task.revision:
            violations.append("latest task revision changed after reopening the store")
        expected_completion_revision = settings.task_updates + 2
        if completed.revision != expected_completion_revision:
            violations.append("completed task revision is not append-only")
        if history_revisions != tuple(range(1, expected_completion_revision + 1)):
            violations.append("task revision history is incomplete")
        if not completion_immutable:
            violations.append("completed task accepted a later mutation")

        _current_python_bytes, peak_python_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return MemoryContinuityReport(
        exchanges=settings.exchanges,
        stored_turns=len(expected_turns),
        restored_turns=len(restored_turns),
        expected_working_messages=expected_working,
        restored_working_messages=restored_working,
        recall_results=tuple(recall_results),
        cross_session_episode_recalled=cross_session_episode_recalled,
        cross_session_episode_score=cross_session_episode_score,
        same_session_episode_excluded=same_session_episode_excluded,
        same_session_archive_tested=same_session_archive_tested,
        same_session_archived_episode_recalled=(
            same_session_archived_episode_recalled
        ),
        same_session_archived_episode_score=same_session_archived_episode_score,
        recent_working_episode_excluded=recent_working_episode_excluded,
        unrelated_queries=len(unrelated),
        unrelated_false_recalls=unrelated_false_recalls,
        expected_task_revision=settings.task_updates + 1,
        restored_task_revision=restored_task.revision,
        task_history_revisions=history_revisions,
        completion_immutable=completion_immutable,
        database_bytes=path.stat().st_size,
        peak_python_bytes=peak_python_bytes,
        elapsed_seconds=perf_counter() - started,
        violations=tuple(violations),
    )


def evaluate_long_history_episodic_recall(
    database: str | Path,
    *,
    exchanges: int = 5_000,
) -> LongHistoryEpisodicReport:
    """Measure whole-history episodic recall beyond the former 400-row window."""

    if not 401 <= exchanges <= 100_000:
        raise ValueError("long-history exchanges must be between 401 and 100,000")
    path = Path(database)
    if path.exists():
        raise ValueError("long-history evaluation database must not already exist")
    path.parent.mkdir(parents=True, exist_ok=True)
    store = ConversationStore(path)
    source_session = store.create_session()
    current_session = store.create_session()
    target_prompt = "전체 기록 경계 실험의 오래된 코드명은 해마-9001이야"
    target_answer = "전체 기록에서 해마-9001을 다시 찾았어."
    timestamp = "2026-07-22T00:00:00Z"

    def rows():
        yield (source_session, "user", target_prompt, "", timestamp)
        yield (
            source_session,
            "assistant",
            target_answer,
            "best_effort",
            timestamp,
        )
        for index in range(1, exchanges):
            yield (
                source_session,
                "user",
                f"무관한 장기 기록 채우기 요청 {index:06d}",
                "",
                timestamp,
            )
            yield (
                source_session,
                "assistant",
                f"무관한 장기 기록 채우기 응답 {index:06d}",
                "best_effort",
                timestamp,
            )

    with closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.executemany(
                "INSERT INTO conversation_turns("
                "session_id, role, content, status, created_at"
                ") VALUES (?, ?, ?, ?, ?)",
                rows(),
            )

    started = perf_counter()
    reopened = ConversationStore(path)
    backfill_seconds = perf_counter() - started
    with closing(sqlite3.connect(path)) as connection:
        indexed_episodes = int(
            connection.execute(
                "SELECT COUNT(*) FROM conversation_episodes"
            ).fetchone()[0]
        )

    started = perf_counter()
    cross = reopened.search_episodes(
        target_prompt,
        limit=1,
        exclude_session_id=current_session,
    )
    cross_session_seconds = perf_counter() - started
    started = perf_counter()
    same = reopened.search_archived_session_episodes(
        target_prompt,
        source_session,
        limit=1,
    )
    same_session_seconds = perf_counter() - started
    cross_answer = cross[0].assistant_content if cross else ""
    same_answer = same[0].assistant_content if same else ""
    violations: list[str] = []
    if indexed_episodes != exchanges:
        violations.append("episode index did not cover the complete stored history")
    if cross_answer != target_answer:
        violations.append("cross-session recall missed the old indexed episode")
    if same_answer != target_answer:
        violations.append("same-session archive recall missed the old indexed episode")
    return LongHistoryEpisodicReport(
        exchanges=exchanges,
        indexed_episodes=indexed_episodes,
        target_answer=target_answer,
        cross_session_answer=cross_answer,
        same_session_answer=same_answer,
        cross_session_recalled=cross_answer == target_answer,
        same_session_recalled=same_answer == target_answer,
        backfill_seconds=backfill_seconds,
        cross_session_seconds=cross_session_seconds,
        same_session_seconds=same_session_seconds,
        database_bytes=path.stat().st_size,
        violations=tuple(violations),
    )


def evaluate_semantic_paraphrase_memory(
    store: ConversationStore,
    ranker: MemorySemanticRanker,
) -> SemanticParaphraseMemoryReport:
    """Compare lexical and typed-facet E5 recall on held-out Korean wording."""

    cases = (
        ("나는 답변을 짧게 받는 걸 좋아해", "설명 길이는 어떻게 해주면 좋다고 했지?"),
        ("내 그래픽카드는 RTX 4060 8GB야", "내 컴퓨터의 GPU 사양 기억나?"),
        ("중요한 결론부터 먼저 말해줘", "답변 순서를 어떻게 해달라고 했지?"),
        (
            "프로젝트의 목표는 작은 모델이 연산자를 조합하게 만드는 거야",
            "우리가 만들려는 지능의 핵심 방향이 뭐였지?",
        ),
        ("매주 일요일에는 전체 테스트를 실행해", "회귀 검사는 언제 돌리기로 했지?"),
    )
    unrelated = (
        "오늘 비가 올까?",
        "파스타를 어떻게 삶아?",
        "야구 경기 결과를 알려줘",
    )
    expected = [store.remember(content) for content, _query in cases]
    started = perf_counter()
    lexical_correct = 0
    hybrid_correct = 0
    scores: list[float] = []
    retrievals: list[str] = []
    for (_content, query), memory in zip(cases, expected, strict=True):
        lexical = store.search_memories(query, limit=1)
        hybrid = store.search_memories(
            query,
            limit=1,
            semantic_ranker=ranker,
        )
        lexical_correct += bool(lexical and lexical[0].memory_id == memory.memory_id)
        hybrid_correct += bool(hybrid and hybrid[0].memory_id == memory.memory_id)
        scores.append(hybrid[0].score if hybrid else 0.0)
        retrievals.append(hybrid[0].retrieval if hybrid else "")
    unrelated_false_recalls = sum(
        bool(store.search_memories(query, limit=1, semantic_ranker=ranker))
        for query in unrelated
    )
    cold_elapsed_seconds = perf_counter() - started
    warm_started = perf_counter()
    warm_correct = sum(
        bool(
            (matches := store.search_memories(
                query,
                limit=1,
                semantic_ranker=ranker,
            ))
            and matches[0].memory_id == memory.memory_id
        )
        for (_content, query), memory in zip(cases, expected, strict=True)
    )
    warm_replay_seconds = perf_counter() - warm_started
    return SemanticParaphraseMemoryReport(
        positive_cases=len(cases),
        lexical_correct=lexical_correct,
        hybrid_correct=hybrid_correct,
        unrelated_queries=len(unrelated),
        unrelated_false_recalls=unrelated_false_recalls,
        hybrid_scores=tuple(scores),
        hybrid_retrievals=tuple(retrievals),
        elapsed_seconds=cold_elapsed_seconds,
        warm_replay_seconds=warm_replay_seconds,
        warm_correct=warm_correct,
    )


def evaluate_semantic_memory_effect(
    assistant: PromptAssistant,
    store: ConversationStore,
    *,
    resource_tier: ResourceTier | str = ResourceTier.BALANCED,
    expected_marker: str = "별빛-27",
) -> SemanticMemoryABReport:
    """Measure whether one explicit memory changes a local model's answer."""

    marker = str(expected_marker).strip()
    if not marker:
        raise ValueError("semantic memory marker cannot be empty")
    query = "내가 저장해 둔 프로젝트 코드명이 뭐였지? 코드명만 답해줘."
    store.remember(f"내 프로젝트 코드명은 {marker}이야")

    started = perf_counter()
    without_memory = assistant.solve(
        PromptRequest(text=query, resource_tier=resource_tier)
    )
    without_seconds = perf_counter() - started

    recalled = store.search_memories(query, limit=3)
    started = perf_counter()
    with_memory = assistant.solve(
        PromptRequest(
            text=query,
            resource_tier=resource_tier,
            recalled_memories=recalled,
        )
    )
    with_seconds = perf_counter() - started
    normalized_marker = marker.casefold()
    without_generation = without_memory.provenance.get("semantic_generation", {})
    with_generation = with_memory.provenance.get("semantic_generation", {})

    return SemanticMemoryABReport(
        query=query,
        expected_marker=marker,
        recalled_memory_ids=tuple(item.memory_id for item in recalled),
        recall_score=(recalled[0].score if recalled else 0.0),
        without_memory_answer=without_memory.answer,
        with_memory_answer=with_memory.answer,
        without_memory_status=without_memory.status.value,
        with_memory_status=with_memory.status.value,
        without_generated_tokens=int(without_generation.get("generated_tokens", 0)),
        with_generated_tokens=int(with_generation.get("generated_tokens", 0)),
        without_hit_token_limit=bool(without_generation.get("hit_token_limit")),
        with_hit_token_limit=bool(with_generation.get("hit_token_limit")),
        without_repair_attempts=int(
            without_memory.provenance.get("repair_attempts", 0)
        ),
        with_repair_attempts=int(
            with_memory.provenance.get("repair_attempts", 0)
        ),
        without_memory_seconds=without_seconds,
        with_memory_seconds=with_seconds,
        without_memory_contains_marker=(
            normalized_marker in without_memory.answer.casefold()
        ),
        with_memory_contains_marker=normalized_marker in with_memory.answer.casefold(),
        memory_context_reported=bool(
            with_memory.provenance.get("memory_context_used_by_model")
            or with_memory.provenance.get("memory_context_used_by_retriever")
        ),
    )


def evaluate_episodic_memory_effect(
    assistant: PromptAssistant,
    store: ConversationStore,
    *,
    resource_tier: ResourceTier | str = ResourceTier.BALANCED,
    expected_marker: str = "해마-7319",
) -> EpisodicMemoryABReport:
    """Measure whether a recalled past exchange changes a local model answer."""

    marker = str(expected_marker).strip()
    if not marker:
        raise ValueError("episodic-memory marker cannot be empty")
    source_session = store.create_session()
    store.append_exchange(
        source_session,
        user_content=(
            f"이번 실험에서 정한 해마 회상 코드명은 {marker}이야. "
            "코드명만 기억해."
        ),
        assistant_content=f"알겠어. 해마 회상 코드명은 {marker}로 기억할게.",
        answer_status="best_effort",
    )
    target_session = store.create_session()
    query = "이번 실험에서 정한 해마 회상 코드명이 뭐였지? 코드명만 답해줘."

    started = perf_counter()
    without_episode = assistant.solve(
        PromptRequest(text=query, resource_tier=resource_tier)
    )
    without_seconds = perf_counter() - started

    recalled = store.search_episodes(
        query,
        limit=2,
        exclude_session_id=target_session,
    )
    started = perf_counter()
    with_episode = assistant.solve(
        PromptRequest(
            text=query,
            resource_tier=resource_tier,
            recalled_episodes=recalled,
        )
    )
    with_seconds = perf_counter() - started
    normalized_marker = marker.casefold()
    without_generation = without_episode.provenance.get("semantic_generation", {})
    with_generation = with_episode.provenance.get("semantic_generation", {})

    return EpisodicMemoryABReport(
        query=query,
        expected_marker=marker,
        recalled_episode_ids=tuple(item.episode_id for item in recalled),
        recalled_episode_scopes=tuple(item.scope for item in recalled),
        recall_score=(recalled[0].score if recalled else 0.0),
        without_episode_answer=without_episode.answer,
        with_episode_answer=with_episode.answer,
        without_episode_status=without_episode.status.value,
        with_episode_status=with_episode.status.value,
        without_generated_tokens=int(without_generation.get("generated_tokens", 0)),
        with_generated_tokens=int(with_generation.get("generated_tokens", 0)),
        without_hit_token_limit=bool(without_generation.get("hit_token_limit")),
        with_hit_token_limit=bool(with_generation.get("hit_token_limit")),
        without_repair_attempts=int(
            without_episode.provenance.get("repair_attempts", 0)
        ),
        with_repair_attempts=int(
            with_episode.provenance.get("repair_attempts", 0)
        ),
        without_episode_seconds=without_seconds,
        with_episode_seconds=with_seconds,
        without_episode_contains_marker=(
            normalized_marker in without_episode.answer.casefold()
        ),
        with_episode_contains_marker=(
            normalized_marker in with_episode.answer.casefold()
        ),
        episode_context_reported=bool(
            with_episode.provenance.get("episode_context_used_by_model")
            or with_episode.provenance.get("episode_context_used_by_retriever")
        ),
    )


def evaluate_long_session_memory_effect(
    assistant: PromptAssistant,
    store: ConversationStore,
    *,
    resource_tier: ResourceTier | str = ResourceTier.BALANCED,
    expected_marker: str = "등대-4821",
) -> EpisodicMemoryABReport:
    """Measure model use of an episode that fell out of same-session context."""

    marker = str(expected_marker).strip()
    if not marker:
        raise ValueError("long-session memory marker cannot be empty")
    session_id = store.create_session()
    store.append_exchange(
        session_id,
        user_content=(
            f"긴 대화 기억 실험의 코드명은 {marker}이야. 코드명만 기억해."
        ),
        assistant_content=f"알겠어. 긴 대화 기억 실험 코드명은 {marker}야.",
        answer_status="best_effort",
    )
    for index in range(4):
        store.append_exchange(
            session_id,
            user_content=f"별도 대화 주제 {index}: 연산자 상태를 점검해줘.",
            assistant_content=f"별도 대화 주제 {index} 점검을 마쳤어.",
            answer_status="best_effort",
        )
    query = "긴 대화 기억 실험의 코드명이 뭐였지? 코드명만 답해줘."

    started = perf_counter()
    without_episode = assistant.solve(
        PromptRequest(text=query, resource_tier=resource_tier)
    )
    without_seconds = perf_counter() - started

    recalled = store.search_archived_session_episodes(
        query,
        session_id,
        limit=2,
    )
    started = perf_counter()
    with_episode = assistant.solve(
        PromptRequest(
            text=query,
            resource_tier=resource_tier,
            recalled_episodes=recalled,
        )
    )
    with_seconds = perf_counter() - started
    normalized_marker = marker.casefold()
    without_generation = without_episode.provenance.get("semantic_generation", {})
    with_generation = with_episode.provenance.get("semantic_generation", {})

    return EpisodicMemoryABReport(
        query=query,
        expected_marker=marker,
        recalled_episode_ids=tuple(item.episode_id for item in recalled),
        recalled_episode_scopes=tuple(item.scope for item in recalled),
        recall_score=(recalled[0].score if recalled else 0.0),
        without_episode_answer=without_episode.answer,
        with_episode_answer=with_episode.answer,
        without_episode_status=without_episode.status.value,
        with_episode_status=with_episode.status.value,
        without_generated_tokens=int(without_generation.get("generated_tokens", 0)),
        with_generated_tokens=int(with_generation.get("generated_tokens", 0)),
        without_hit_token_limit=bool(without_generation.get("hit_token_limit")),
        with_hit_token_limit=bool(with_generation.get("hit_token_limit")),
        without_repair_attempts=int(
            without_episode.provenance.get("repair_attempts", 0)
        ),
        with_repair_attempts=int(
            with_episode.provenance.get("repair_attempts", 0)
        ),
        without_episode_seconds=without_seconds,
        with_episode_seconds=with_seconds,
        without_episode_contains_marker=(
            normalized_marker in without_episode.answer.casefold()
        ),
        with_episode_contains_marker=(
            normalized_marker in with_episode.answer.casefold()
        ),
        episode_context_reported=bool(
            with_episode.provenance.get("episode_context_used_by_model")
            or with_episode.provenance.get("episode_context_used_by_retriever")
        ),
    )


def evaluate_correction_consolidation(
    store: ConversationStore,
    *,
    expected_answer: str = "별빛-27이야.",
) -> CorrectionConsolidationReport:
    """Measure three-example correction consolidation on a held-out phrase."""

    answer = str(expected_answer).strip()
    if not answer:
        raise ValueError("correction consolidation answer cannot be empty")
    prompts = (
        "SemOp 프로젝트 코드명은 뭐야?",
        "SemOp 프로젝트 코드명을 알려줘",
        "SemOp 프로젝트 코드명이 무엇인지 말해줘",
    )
    query = "SemOp 프로젝트 코드명 좀 알려줘"
    started = perf_counter()
    session_id = store.create_session()
    for prompt in prompts[:2]:
        store.remember_correction(
            prompt=prompt,
            prior_answer="아직 모르겠어.",
            corrected_answer=answer,
            source_session_id=session_id,
        )
    before = store.search_correction_prototypes(query)

    store.remember_correction(
        prompt=prompts[2],
        prior_answer="확인할 수 없어.",
        corrected_answer=answer,
        source_session_id=session_id,
    )
    reopened = ConversationStore(store.path)
    after = reopened.search_correction_prototypes(query)
    match = after[0] if after else None
    prototype = match.prototype if match is not None else None
    return CorrectionConsolidationReport(
        query=query,
        expected_answer=answer,
        examples_before=2,
        examples_after=3,
        matched_before_consolidation=bool(before),
        matched_after_consolidation=bool(after),
        prototype_id=(prototype.prototype_id if prototype is not None else ""),
        prototype_support=(prototype.support if prototype is not None else 0),
        matched_prompt=(match.matched_prompt if match is not None else ""),
        score=(match.score if match is not None else 0.0),
        recalled_answer=(prototype.corrected_answer if prototype is not None else ""),
        restart_persistent=bool(
            match is not None
            and prototype is not None
            and prototype.corrected_answer == answer
        ),
        elapsed_seconds=perf_counter() - started,
    )


def evaluate_task_auto_resume(store: TaskCheckpointStore) -> TaskAutoResumeReport:
    """Measure named active-task retrieval against an ambiguous continuation."""

    started = perf_counter()
    stale_action = "이전 행동-13"
    expected_action = "최신 행동-42"
    target = store.create_task(
        title="SemOp 지속 학습",
        objective="교정 기억과 student 학습을 연결한다",
    )
    store.checkpoint(
        target.task_id,
        progress="이전 단계를 완료했다",
        next_actions=(stale_action,),
    )
    target = store.checkpoint(
        target.task_id,
        progress="교정 prototype 평가를 통과했다",
        next_actions=(expected_action,),
    )
    store.create_task(
        title="여행 사진 정리",
        objective="휴대폰 사진을 날짜별로 정리한다",
    )
    ambiguous_query = "장기 작업 계속하자"
    named_query = "SemOp 지속 학습 작업 계속하자"
    ambiguous = store.retrieve_active_task(ambiguous_query)

    reopened = TaskCheckpointStore(store.path)
    selected = reopened.retrieve_active_task(named_query)
    checkpoint = selected.checkpoint if selected is not None else None
    action = (
        checkpoint.next_actions[0]
        if checkpoint is not None and checkpoint.next_actions
        else ""
    )
    return TaskAutoResumeReport(
        ambiguous_query=ambiguous_query,
        named_query=named_query,
        target_task_id=target.task_id,
        target_revision=target.revision,
        expected_action=expected_action,
        stale_action=stale_action,
        ambiguous_query_rejected=ambiguous is None,
        named_query_selected=selected is not None,
        selected_task_id=(checkpoint.task_id if checkpoint is not None else ""),
        selected_revision=(checkpoint.revision if checkpoint is not None else 0),
        selected_action=action,
        selected_contains_stale_action=stale_action in action,
        score=(selected.score if selected is not None else 0.0),
        reason=(selected.reason if selected is not None else ""),
        restart_persistent=bool(
            checkpoint is not None
            and checkpoint.task_id == target.task_id
            and checkpoint.revision == target.revision
        ),
        elapsed_seconds=perf_counter() - started,
    )


def evaluate_task_context_effect(
    assistant: PromptAssistant,
    store: TaskCheckpointStore,
    *,
    resource_tier: ResourceTier | str = ResourceTier.BALANCED,
    expected_marker: str = "검증-단계-42",
) -> TaskContextABReport:
    """Measure whether the latest task revision changes a local model's answer."""

    marker = str(expected_marker).strip()
    if not marker:
        raise ValueError("task-context marker cannot be empty")
    stale_marker = "이전-단계-13"
    task = store.create_task(
        title="SemOp 장기 작업 A/B",
        objective="최신 checkpoint의 다음 행동을 이어서 수행한다",
    )
    task = store.checkpoint(
        task.task_id,
        progress="초기 준비를 마쳤다",
        decisions=("작업을 revision별로 보존한다",),
        next_actions=(stale_marker,),
    )
    task = store.checkpoint(
        task.task_id,
        progress="이전 행동을 완료했다",
        decisions=("항상 최신 revision에서 재개한다",),
        next_actions=(marker,),
    )
    query = "내가 선택한 장기 작업에서 지금 다음으로 해야 할 행동이 뭐야? 행동 문구만 답해줘."

    started = perf_counter()
    without_context = assistant.solve(
        PromptRequest(text=query, resource_tier=resource_tier)
    )
    without_seconds = perf_counter() - started

    started = perf_counter()
    with_context = assistant.solve(
        PromptRequest(
            text=query,
            resource_tier=resource_tier,
            task_contexts=(task.to_context(),),
        )
    )
    with_seconds = perf_counter() - started
    summary_query = "선택한 장기 작업 어디까지 했지?"
    started = perf_counter()
    summary = assistant.solve(
        PromptRequest(
            text=summary_query,
            resource_tier=resource_tier,
            task_contexts=(task.to_context(),),
        )
    )
    summary_seconds = perf_counter() - started
    normalized_marker = marker.casefold()
    normalized_stale = stale_marker.casefold()

    return TaskContextABReport(
        query=query,
        task_id=task.task_id,
        task_revision=task.revision,
        expected_marker=marker,
        stale_marker=stale_marker,
        without_context_answer=without_context.answer,
        with_context_answer=with_context.answer,
        without_context_status=without_context.status.value,
        with_context_status=with_context.status.value,
        without_context_seconds=without_seconds,
        with_context_seconds=with_seconds,
        without_context_contains_marker=(
            normalized_marker in without_context.answer.casefold()
        ),
        with_context_contains_marker=(
            normalized_marker in with_context.answer.casefold()
        ),
        with_context_contains_stale_marker=(
            normalized_stale in with_context.answer.casefold()
        ),
        task_context_reported=bool(
            with_context.provenance.get("task_context_used_by_model")
            or with_context.provenance.get("task_context_used_by_retriever")
        ),
        summary_query=summary_query,
        summary_answer=summary.answer,
        summary_seconds=summary_seconds,
        summary_contains_revision=f"revision {task.revision}" in summary.answer,
        summary_contains_progress=task.progress in summary.answer,
        summary_contains_decision=task.decisions[-1] in summary.answer,
        summary_contains_next_action=marker in summary.answer,
        summary_context_reported=bool(
            summary.provenance.get("task_context_used_by_model")
            or summary.provenance.get("task_context_used_by_retriever")
        ),
    )

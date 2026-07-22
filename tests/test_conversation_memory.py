from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import threading
from urllib.request import Request, urlopen

from semop.assistant import SemOpAssistant
from semop.beginner_web import create_server
from semop.conversation_memory import ConversationStore, MemoryFacet, classify_memory_facets
from semop.prompt_api import ConversationRole
from semop.prompt_compiler import PromptCompiler


class _FacetSemanticRanker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def retrieve(self, query: str, *, limit: int = 8) -> tuple[str, ...]:
        return ()

    def rank_texts(
        self,
        query: str,
        passages: tuple[str, ...],
        *,
        limit: int = 8,
    ) -> tuple[tuple[int, float], ...]:
        self.calls.append((query, passages))
        return ((0, 0.86),) if passages else ()


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def test_conversation_store_persists_episodic_history_and_bounds_recall(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conversation.db"
    first = ConversationStore(path)
    session_id = first.create_session()
    first.append_exchange(
        session_id,
        user_content="내 프로젝트는 SemOp이야",
        assistant_content="프로젝트 이름을 기억했어",
        answer_status="best_effort",
    )
    first.append_exchange(
        session_id,
        user_content="핵심은 typed operator야",
        assistant_content="typed operator를 중심으로 이어갈게",
        answer_status="best_effort",
    )

    reopened = ConversationStore(path)
    history = reopened.history(session_id)
    recent = reopened.recent_messages(session_id, max_messages=2)

    assert [turn.role for turn in history] == [
        ConversationRole.USER,
        ConversationRole.ASSISTANT,
        ConversationRole.USER,
        ConversationRole.ASSISTANT,
    ]
    assert [message.content for message in recent] == [
        "핵심은 typed operator야",
        "typed operator를 중심으로 이어갈게",
    ]


def test_working_memory_truncates_one_oversized_latest_message(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "conversation.db")
    session_id = store.create_session()
    store.append_exchange(
        session_id,
        user_content="짧은 질문",
        assistant_content="가" * 1_000,
        answer_status="best_effort",
    )

    recent = store.recent_messages(session_id, max_messages=1, max_chars=256)

    assert len(recent) == 1
    assert len(recent[0].content) == 256
    assert recent[0].content.startswith("[앞부분 생략]")


def test_related_episode_is_recalled_across_sessions_but_not_from_current_session(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conversation.db"
    store = ConversationStore(path)
    source_session = store.create_session()
    current_session = store.create_session()
    store.append_exchange(
        source_session,
        user_content="해마형 장기기억 구조를 어떻게 설계할까?",
        assistant_content="빠른 일화 저장과 느린 의미 통합을 분리하자.",
        answer_status="best_effort",
    )
    store.append_exchange(
        current_session,
        user_content="해마형 장기기억 구조를 어떻게 설계할까?",
        assistant_content="현재 세션의 최근 답변",
        answer_status="best_effort",
    )

    recalled = ConversationStore(path).search_episodes(
        "해마형 장기기억 구조를 어떻게 설계할까?",
        exclude_session_id=current_session,
    )

    assert len(recalled) == 1
    assert len(recalled[0].episode_id) == 64
    assert recalled[0].user_content == "해마형 장기기억 구조를 어떻게 설계할까?"
    assert recalled[0].assistant_content == (
        "빠른 일화 저장과 느린 의미 통합을 분리하자."
    )
    assert recalled[0].answer_status == "best_effort"
    assert recalled[0].score == 1.0
    assert recalled[0].scope == "cross_session"


def test_old_same_session_episode_is_recalled_after_leaving_working_memory(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "conversation.db")
    session_id = store.create_session()
    store.append_exchange(
        session_id,
        user_content="긴 대화의 실험 코드명은 등대-4821이야",
        assistant_content="실험 코드명 등대-4821을 확인했어.",
        answer_status="best_effort",
    )
    for index in range(4):
        store.append_exchange(
            session_id,
            user_content=f"작업기억 채우기 요청 {index}",
            assistant_content=f"작업기억 채우기 응답 {index}",
            answer_status="best_effort",
        )

    recalled = store.search_archived_session_episodes(
        "긴 대화의 실험 코드명이 뭐였지?",
        session_id,
    )
    recent_only = store.search_archived_session_episodes(
        "작업기억 채우기 요청 3",
        session_id,
    )

    assert len(recalled) == 1
    assert recalled[0].assistant_content == "실험 코드명 등대-4821을 확인했어."
    assert recalled[0].scope == "same_session_archive"
    assert recent_only == ()


def test_relevant_episode_recall_combines_session_archive_and_past_sessions(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "conversation.db")
    old_session = store.create_session()
    current_session = store.create_session()
    store.append_exchange(
        old_session,
        user_content="연산자 기억 실험을 설명해줘",
        assistant_content="다른 세션의 답변",
        answer_status="conditional",
    )
    store.append_exchange(
        current_session,
        user_content="연산자 기억 실험을 설명해줘",
        assistant_content="현재 세션의 오래된 답변",
        answer_status="verified",
    )
    for index in range(4):
        store.append_exchange(
            current_session,
            user_content=f"별도 주제 {index}",
            assistant_content=f"별도 응답 {index}",
            answer_status="best_effort",
        )

    recalled = store.recall_relevant_episodes(
        "연산자 기억 실험을 설명해줘",
        current_session_id=current_session,
    )

    assert [item.scope for item in recalled] == [
        "same_session_archive",
        "cross_session",
    ]
    assert [item.assistant_content for item in recalled] == [
        "현재 세션의 오래된 답변",
        "다른 세션의 답변",
    ]


def test_full_history_index_recalls_episode_older_than_recent_400_after_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conversation.db"
    initial = ConversationStore(path)
    source_session = initial.create_session()
    current_session = initial.create_session()
    exchanges = [
        (
            source_session,
            "아주 오래된 장기기억 실험 코드명은 해마-9001이야",
            "장기기억 실험 코드명 해마-9001을 확인했어.",
        )
    ]
    exchanges.extend(
        (
            source_session,
            f"무관한 최신 채우기 요청 {index}",
            f"무관한 최신 채우기 응답 {index}",
        )
        for index in range(420)
    )
    timestamp = "2026-07-22T00:00:00Z"
    rows: list[tuple[str, str, str, str, str]] = []
    for session_id, user_content, assistant_content in exchanges:
        rows.extend(
            (
                (session_id, "user", user_content, "", timestamp),
                (
                    session_id,
                    "assistant",
                    assistant_content,
                    "best_effort",
                    timestamp,
                ),
            )
        )
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.executemany(
                "INSERT INTO conversation_turns("
                "session_id, role, content, status, created_at"
                ") VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    reopened = ConversationStore(path)
    cross_session = reopened.search_episodes(
        "오래된 장기기억 실험 코드명이 뭐였지?",
        exclude_session_id=current_session,
    )
    same_session = reopened.search_archived_session_episodes(
        "오래된 장기기억 실험 코드명이 뭐였지?",
        source_session,
    )

    assert cross_session[0].assistant_content == (
        "장기기억 실험 코드명 해마-9001을 확인했어."
    )
    assert cross_session[0].scope == "cross_session"
    assert same_session[0].assistant_content == cross_session[0].assistant_content
    assert same_session[0].scope == "same_session_archive"


def test_explicit_long_term_memory_is_searchable_deduplicated_and_deletable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conversation.db"
    store = ConversationStore(path)
    session_id = store.create_session()
    first = store.remember(
        "내 프로젝트 이름은 SemOp이고 핵심은 typed operator야",
        source_session_id=session_id,
    )
    duplicate = store.remember(
        "  내 프로젝트 이름은 SemOp이고 핵심은 typed operator야  ",
        source_session_id=session_id,
    )

    matches = store.search_memories("프로젝트 이름이 뭐였지?")

    assert duplicate.memory_id == first.memory_id
    assert len(store.list_memories()) == 1
    assert matches[0].memory_id == first.memory_id
    assert matches[0].score >= 0.18
    assert store.search_memories("오늘 날씨를 알려줘") == ()
    assert ConversationStore(path).list_memories()[0] == first
    assert store.forget(first.memory_id) is True
    assert store.list_memories() == ()
    assert store.forget(first.memory_id) is False


def test_typed_facets_enable_semantic_paraphrase_recall_only_after_lexical_miss(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "semantic-memory.db")
    memories = (
        ("나는 답변을 짧게 받는 걸 좋아해", "설명 길이는 어떻게 해주면 좋다고 했지?"),
        ("내 그래픽카드는 RTX 4060 8GB야", "내 컴퓨터의 GPU 사양 기억나?"),
        ("중요한 결론부터 먼저 말해줘", "답변 순서를 어떻게 해달라고 했지?"),
        (
            "프로젝트의 목표는 작은 모델이 연산자를 조합하게 만드는 거야",
            "우리가 만들려는 지능의 핵심 방향이 뭐였지?",
        ),
        ("매주 일요일에는 전체 테스트를 실행해", "회귀 검사는 언제 돌리기로 했지?"),
    )
    saved = [store.remember(content) for content, _query in memories]
    ranker = _FacetSemanticRanker()

    for (content, query), expected in zip(memories, saved, strict=True):
        assert store.search_memories(query) == ()
        recalled = store.search_memories(
            query,
            limit=1,
            semantic_ranker=ranker,
        )
        assert recalled[0].memory_id == expected.memory_id
        assert recalled[0].content == content
        assert recalled[0].retrieval == "semantic_e5"

    calls_after_paraphrases = len(ranker.calls)
    exact = store.search_memories(
        memories[0][0],
        semantic_ranker=ranker,
    )
    unrelated = store.search_memories(
        "오늘 비가 올까?",
        semantic_ranker=ranker,
    )

    assert exact[0].retrieval == "lexical"
    assert unrelated == ()
    assert len(ranker.calls) == calls_after_paraphrases
    assert classify_memory_facets(memories[1][1]) == {
        MemoryFacet.GPU_HARDWARE
    }


def test_conversation_store_releases_sqlite_file_handles(tmp_path: Path) -> None:
    path = tmp_path / "conversation.db"
    store = ConversationStore(path)
    memory = store.remember("Windows에서도 DB 연결을 즉시 닫아")
    assert store.search_memories("DB 연결")[0].memory_id == memory.memory_id

    path.unlink()

    assert path.exists() is False


def test_user_correction_is_bound_to_prompt_replaced_and_persistent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conversation.db"
    store = ConversationStore(path)
    session_id = store.create_session()
    store.append_exchange(
        session_id,
        user_content="SemOp의 프로젝트 코드명은 뭐야?",
        assistant_content="아직 모르겠어.",
        answer_status="best_effort",
    )
    user_turn, assistant_turn = store.latest_exchange(session_id)

    first = store.remember_correction(
        prompt=user_turn.content,
        prior_answer=assistant_turn.content,
        corrected_answer="별빛-27이야.",
        source_session_id=session_id,
    )
    replaced = store.remember_correction(
        prompt="  SemOp의 프로젝트 코드명은 뭐야?  ",
        prior_answer="별빛-27이야.",
        corrected_answer="별빛-42야.",
        source_session_id=session_id,
    )

    assert replaced.correction_id == first.correction_id
    assert replaced.created_at == first.created_at
    assert replaced.corrected_answer == "별빛-42야."
    assert len(store.list_corrections()) == 1
    assert store.find_correction("SemOp의 프로젝트 코드명은 뭐야?") == replaced
    assert store.find_correction("다른 질문") is None
    assert ConversationStore(path).find_correction(user_turn.content) == replaced


def test_three_distinct_corrections_form_a_restart_safe_prototype(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conversation.db"
    store = ConversationStore(path)
    session_id = store.create_session()
    prompts = (
        "SemOp 프로젝트 코드명은 뭐야?",
        "SemOp 프로젝트 코드명을 알려줘",
        "SemOp 프로젝트 코드명이 무엇인지 말해줘",
    )
    for prompt in prompts[:2]:
        store.remember_correction(
            prompt=prompt,
            prior_answer="아직 모르겠어.",
            corrected_answer="별빛-27이야.",
            source_session_id=session_id,
        )

    held_out = "SemOp 프로젝트 코드명 좀 알려줘"
    assert store.correction_prototypes() == ()
    assert store.search_correction_prototypes(held_out) == ()

    store.remember_correction(
        prompt=prompts[2],
        prior_answer="확인할 수 없어.",
        corrected_answer="별빛-27이야.",
        source_session_id=session_id,
    )
    reopened = ConversationStore(path)
    prototypes = reopened.correction_prototypes()
    matches = reopened.search_correction_prototypes(held_out)

    assert len(prototypes) == 1
    assert prototypes[0].support == 3
    assert prototypes[0].corrected_answer == "별빛-27이야."
    assert set(prototypes[0].prompts) == set(prompts)
    assert len(matches) == 1
    assert matches[0].prototype == prototypes[0]
    assert matches[0].matched_prompt in prompts
    assert matches[0].score >= 0.35


def test_chat_memory_command_saves_content_and_recalls_it_on_next_turn(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "conversation.db")
    server = create_server(0, conversation_store=store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        saved = _post_json(
            f"{base}/api/chat",
            {
                "text": "기억해: 내 프로젝트 코드명은 별빛-27이야",
                "images": [],
                "resource_tier": "symbolic",
            },
        )
        session_id = str(saved["session_id"])
        recalled = _post_json(
            f"{base}/api/chat",
            {
                "text": "프로젝트 코드명이 뭐였지?",
                "images": [],
                "resource_tier": "symbolic",
                "session_id": session_id,
            },
        )
        with urlopen(f"{base}/api/memory", timeout=10) as response:
            listed = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    memory = saved["saved_memory"]
    assert saved["answer"]["status"] == "best_effort"
    assert saved["answer"]["provenance"]["answer_source"] == (
        "explicit_memory_write"
    )
    assert saved["answer"]["provenance"]["memory_id"] == memory["memory_id"]
    assert memory["content"] == "내 프로젝트 코드명은 별빛-27이야"
    assert len(saved["turns"]) == 2
    assert recalled["session_id"] == session_id
    assert recalled["saved_memory"] is None
    assert recalled["recalled_memories"][0]["memory_id"] == memory["memory_id"]
    assert len(recalled["turns"]) == 4
    assert listed["memories"] == [memory]


def test_chat_uses_lazy_semantic_memory_fallback_with_retrieval_provenance(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "conversation.db")
    memory = store.remember("내 그래픽카드는 RTX 4060 8GB야")
    ranker = _FacetSemanticRanker()
    assistant = SemOpAssistant(
        compiler=PromptCompiler(retriever=ranker)
    )
    server = create_server(0, conversation_store=store)
    server.prompt_assistant = lambda _tier: assistant  # type: ignore[method-assign]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        recalled = _post_json(
            f"{base}/api/chat",
            {
                "text": "내 컴퓨터의 GPU 사양 기억나?",
                "images": [],
                "resource_tier": "balanced",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert recalled["answer"]["answer"] == memory.content
    assert recalled["answer"]["status"] == "best_effort"
    assert recalled["recalled_memories"][0]["retrieval"] == "semantic_e5"
    assert recalled["answer"]["provenance"]["recalled_memory_retrievals"] == [
        "semantic_e5"
    ]
    assert recalled["answer"]["provenance"][
        "memory_context_used_by_retriever"
    ] is True
    assert len(ranker.calls) == 1


def test_chat_correction_command_changes_the_next_exact_answer(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "conversation.db")
    server = create_server(0, conversation_store=store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        before = _post_json(
            f"{base}/api/chat",
            {
                "text": "수학: 6 * 7",
                "images": [],
                "resource_tier": "symbolic",
            },
        )
        session_id = str(before["session_id"])
        learned = _post_json(
            f"{base}/api/chat",
            {
                "text": "정정해: 이 실험에서는 정답을 별빛-42로 기억해",
                "images": [],
                "resource_tier": "symbolic",
                "session_id": session_id,
            },
        )
        after = _post_json(
            f"{base}/api/chat",
            {
                "text": "수학: 6 * 7",
                "images": [],
                "resource_tier": "symbolic",
                "session_id": session_id,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    correction = learned["saved_correction"]
    assert before["answer"]["status"] == "verified"
    assert "42" in before["answer"]["answer"]
    assert learned["answer"]["status"] == "best_effort"
    assert learned["answer"]["provenance"]["answer_source"] == (
        "user_correction_write"
    )
    assert correction["prompt"] == "수학: 6 * 7"
    assert correction["prior_answer"] == before["answer"]["answer"]
    assert after["answer"]["answer"] == "이 실험에서는 정답을 별빛-42로 기억해"
    assert after["answer"]["status"] == "best_effort"
    assert after["answer"]["provenance"]["answer_source"] == (
        "user_correction_memory"
    )
    assert after["answer"]["provenance"]["semantic_verified"] is False
    assert after["applied_correction"] == correction
    assert len(after["turns"]) == 6


def test_chat_uses_consolidated_corrections_for_a_new_phrase(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "conversation.db")
    source_session_id = store.create_session()
    for prompt in (
        "SemOp 프로젝트 코드명은 뭐야?",
        "SemOp 프로젝트 코드명을 알려줘",
        "SemOp 프로젝트 코드명이 무엇인지 말해줘",
    ):
        store.remember_correction(
            prompt=prompt,
            prior_answer="아직 모르겠어.",
            corrected_answer="별빛-27이야.",
            source_session_id=source_session_id,
        )

    server = create_server(0, conversation_store=store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        response = _post_json(
            f"{base}/api/chat",
            {
                "text": "SemOp 프로젝트 코드명 좀 알려줘",
                "images": [],
                "resource_tier": "symbolic",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    applied = response["applied_correction_prototype"]
    assert response["answer"]["answer"] == "별빛-27이야."
    assert response["answer"]["status"] == "best_effort"
    assert response["answer"]["provenance"]["answer_source"] == (
        "user_correction_prototype"
    )
    assert response["answer"]["provenance"]["semantic_verified"] is False
    assert response["answer"]["provenance"]["prototype_support"] == 3
    assert applied["support"] == 3
    assert applied["score"] >= 0.35
    assert len(response["turns"]) == 2


def test_http_chat_session_is_created_reused_and_restored(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversation.db")
    server = create_server(0, conversation_store=store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        first = _post_json(
            f"{base}/api/chat",
            {
                "text": "수학: 6 * 7",
                "images": [],
                "resource_tier": "symbolic",
            },
        )
        session_id = str(first["session_id"])
        saved = _post_json(
            f"{base}/api/memory",
            {
                "content": "수학은 검증형으로 답해",
                "session_id": session_id,
            },
        )
        with urlopen(f"{base}/api/memory", timeout=10) as response:
            listed = json.loads(response.read().decode("utf-8"))
        second = _post_json(
            f"{base}/api/chat",
            {
                "text": "수학: 8 + 1",
                "images": [],
                "resource_tier": "symbolic",
                "session_id": session_id,
            },
        )
        with urlopen(f"{base}/api/conversation/{session_id}", timeout=10) as response:
            restored = json.loads(response.read().decode("utf-8"))
        deleted = _post_json(
            f"{base}/api/memory/delete",
            {"memory_id": saved["memory"]["memory_id"]},
        )
        fresh = _post_json(f"{base}/api/conversation/new", {"create": True})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert first["answer"]["status"] == "verified"
    assert len(first["turns"]) == 2
    assert second["session_id"] == session_id
    assert len(second["turns"]) == 4
    assert second["recalled_memories"][0]["content"] == "수학은 검증형으로 답해"
    assert second["answer"]["provenance"]["recalled_memories_provided"] == 1
    assert second["answer"]["provenance"]["memory_context_used_by_model"] is False
    assert listed["memories"] == [saved["memory"]]
    assert deleted["deleted"] is True
    assert deleted["memories"] == []
    assert restored["turns"] == second["turns"]
    assert fresh["session_id"] != session_id
    assert fresh["turns"] == []


def test_http_chat_recalls_a_past_episode_in_a_new_session(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversation.db")
    server = create_server(0, conversation_store=store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        first = _post_json(
            f"{base}/api/chat",
            {
                "text": "수학: 6 * 7",
                "images": [],
                "resource_tier": "symbolic",
            },
        )
        second = _post_json(
            f"{base}/api/chat",
            {
                "text": "수학: 6 * 7",
                "images": [],
                "resource_tier": "symbolic",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert second["session_id"] != first["session_id"]
    assert len(second["recalled_episodes"]) == 1
    episode = second["recalled_episodes"][0]
    assert episode["user_content"] == "수학: 6 * 7"
    assert episode["assistant_content"] == first["answer"]["answer"]
    assert episode["answer_status"] == "verified"
    assert second["answer"]["status"] == "verified"
    assert second["answer"]["provenance"]["recalled_episodes_provided"] == 1
    assert second["answer"]["provenance"]["episode_context_used_by_model"] is False


def test_http_chat_recalls_an_old_episode_inside_a_long_session(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversation.db")
    server = create_server(0, conversation_store=store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        first = _post_json(
            f"{base}/api/chat",
            {
                "text": "수학: 9 * 9",
                "images": [],
                "resource_tier": "symbolic",
            },
        )
        session_id = first["session_id"]
        for value in range(1, 5):
            _post_json(
                f"{base}/api/chat",
                {
                    "text": f"수학: {value} + {value}",
                    "images": [],
                    "resource_tier": "symbolic",
                    "session_id": session_id,
                },
            )
        resumed = _post_json(
            f"{base}/api/chat",
            {
                "text": "수학: 9 * 9",
                "images": [],
                "resource_tier": "symbolic",
                "session_id": session_id,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert resumed["session_id"] == session_id
    assert len(resumed["recalled_episodes"]) == 1
    assert resumed["recalled_episodes"][0]["scope"] == "same_session_archive"
    assert resumed["recalled_episodes"][0]["assistant_content"] == first["answer"][
        "answer"
    ]
    assert resumed["answer"]["provenance"]["recalled_episode_scopes"] == [
        "same_session_archive"
    ]
    assert resumed["answer"]["status"] == "verified"

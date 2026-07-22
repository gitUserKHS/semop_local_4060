from __future__ import annotations

import json
from pathlib import Path
import threading
from urllib.request import Request, urlopen

import pytest

from semop.beginner_web import create_server
from semop.conversation_memory import ConversationStore
from semop.task_memory import TaskCheckpointStore, TaskStatus


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def test_task_checkpoint_revisions_persist_and_completion_is_immutable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.db"
    store = TaskCheckpointStore(path)
    created = store.create_task(
        title="SemOp 로컬 챗봇",
        objective="언어·수학·비전 작업을 여러 세션에 걸쳐 이어간다",
    )
    checkpoint = store.checkpoint(
        created.task_id,
        progress="장기기억을 구현했다",
        decisions=("typed proof와 모델 문맥을 분리한다",),
        next_actions=("장기 작업 checkpoint를 연결한다",),
    )

    reopened = TaskCheckpointStore(path)
    restored = reopened.get(created.task_id)

    assert created.revision == 1
    assert checkpoint.revision == 2
    assert restored == checkpoint
    assert [item.revision for item in reopened.history(created.task_id)] == [1, 2]
    assert reopened.list_tasks(include_completed=False) == (checkpoint,)

    completed = reopened.complete_task(
        created.task_id,
        progress="checkpoint 연결과 검증을 마쳤다",
        decisions=(
            "typed proof와 모델 문맥을 분리한다",
            "완료된 작업은 immutable로 둔다",
        ),
    )

    assert completed.revision == 3
    assert completed.status is TaskStatus.COMPLETED
    assert completed.next_actions == ()
    assert reopened.list_tasks(include_completed=False) == ()
    assert reopened.list_tasks() == (completed,)
    with pytest.raises(ValueError, match="immutable"):
        reopened.checkpoint(created.task_id, progress="완료 후 변조")


def test_task_store_releases_windows_sqlite_handle(tmp_path: Path) -> None:
    path = tmp_path / "tasks.db"
    store = TaskCheckpointStore(path)
    task = store.create_task(title="핸들 검사", objective="SQLite 파일을 닫는다")
    assert store.get(task.task_id) == task

    path.unlink()

    assert path.exists() is False


def test_active_task_retrieval_uses_single_task_or_unique_title_match(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.db"
    store = TaskCheckpointStore(path)
    semop = store.create_task(
        title="SemOp 언어 모델 학습",
        objective="저자원 로컬 챗봇을 개선한다",
    )
    semop = store.checkpoint(
        semop.task_id,
        progress="교정 기억을 연결했다",
        next_actions=("student 학습 후보를 만든다",),
    )

    only = store.retrieve_active_task("장기 작업 계속하자")
    assert only is not None
    assert only.checkpoint == semop
    assert only.reason == "only_active_task"

    store.create_task(
        title="여행 사진 정리",
        objective="사진을 날짜별 폴더로 정리한다",
    )
    matched = TaskCheckpointStore(path).retrieve_active_task(
        "SemOp 언어 모델 학습 작업 계속하자"
    )

    assert matched is not None
    assert matched.checkpoint == semop
    assert matched.reason == "lexical_match"
    assert matched.score >= 0.18
    assert store.retrieve_active_task("장기 작업 계속하자") is None


def test_task_http_flow_selects_context_without_widening_symbolic_proof(
    tmp_path: Path,
) -> None:
    path = tmp_path / "beginner.db"
    conversation_store = ConversationStore(path)
    task_store = TaskCheckpointStore(path)
    server = create_server(
        0,
        conversation_store=conversation_store,
        task_store=task_store,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        created = _post_json(
            f"{base}/api/tasks",
            {
                "title": "수학 검증 작업",
                "objective": "계산은 symbolic executor로 검증한다",
            },
        )
        task_id = str(created["task"]["task_id"])
        checkpoint = _post_json(
            f"{base}/api/tasks/checkpoint",
            {
                "task_id": task_id,
                "progress": "정확한 사칙연산 경로를 확인함",
                "decisions": ["모델 답보다 executor 결과를 우선"],
                "next_actions": ["42 계산을 재검증"],
            },
        )
        chat = _post_json(
            f"{base}/api/chat",
            {
                "text": "수학: 6 * 7",
                "images": [],
                "resource_tier": "symbolic",
                "task_id": task_id,
            },
        )
        with urlopen(f"{base}/api/tasks", timeout=10) as response:
            listed = json.loads(response.read().decode("utf-8"))
        completed = _post_json(
            f"{base}/api/tasks/complete",
            {
                "task_id": task_id,
                "progress": "42를 검증함",
                "decisions": ["executor 결과 42를 채택"],
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert checkpoint["task"]["revision"] == 2
    assert chat["answer"]["status"] == "verified"
    assert chat["answer"]["provenance"]["task_contexts_provided"] == 1
    assert chat["answer"]["provenance"]["task_context_used_by_model"] is False
    assert chat["task_contexts"][0]["next_actions"] == ["42 계산을 재검증"]
    assert listed["tasks"][0]["revision"] == 2
    assert completed["task"]["revision"] == 3
    assert completed["task"]["status"] == "completed"
    assert completed["task"]["decisions"] == ["executor 결과 42를 채택"]


def test_chat_auto_resumes_named_task_at_latest_revision(tmp_path: Path) -> None:
    path = tmp_path / "beginner.db"
    conversation_store = ConversationStore(path)
    task_store = TaskCheckpointStore(path)
    target = task_store.create_task(
        title="SemOp 언어 모델 학습",
        objective="저자원 student를 개선한다",
    )
    task_store.checkpoint(
        target.task_id,
        progress="이전 단계를 마쳤다",
        next_actions=("오래된 행동-13",),
    )
    latest = task_store.checkpoint(
        target.task_id,
        progress="교정 prototype 평가를 통과했다",
        next_actions=("최신 행동-42",),
    )
    task_store.create_task(
        title="여행 사진 정리",
        objective="휴대폰 사진을 정리한다",
    )
    server = create_server(
        0,
        conversation_store=conversation_store,
        task_store=TaskCheckpointStore(path),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        chat = _post_json(
            f"{base}/api/chat",
            {
                "text": "SemOp 언어 모델 학습 작업 계속하자",
                "images": [],
                "resource_tier": "symbolic",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert chat["answer"]["answer"] == "최신 행동-42"
    assert chat["answer"]["status"] == "best_effort"
    assert chat["answer"]["provenance"]["answer_source"] == "task_context"
    assert chat["answer"]["provenance"]["task_context_used_by_retriever"] is True
    assert chat["task_selection"]["mode"] == "automatic"
    assert chat["task_selection"]["task_id"] == target.task_id
    assert chat["task_selection"]["revision"] == latest.revision
    assert chat["task_selection"]["reason"] == "lexical_match"
    assert chat["task_contexts"][0]["next_actions"] == ["최신 행동-42"]


def test_chat_auto_reports_named_task_summary_from_latest_revision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "task-summary.db"
    task_store = TaskCheckpointStore(path)
    target = task_store.create_task(
        title="SemOp 지속 학습",
        objective="검증된 경험을 student에 통합한다",
    )
    task_store.checkpoint(
        target.task_id,
        progress="오래된 준비 단계",
        decisions=("이전 결정은 폐기",),
        next_actions=("오래된 행동-13",),
    )
    latest = task_store.checkpoint(
        target.task_id,
        progress="교정 prototype 평가 통과",
        decisions=("proof 경계를 유지",),
        next_actions=("최신 student 후보 학습",),
    )
    task_store.create_task(
        title="여행 사진 정리",
        objective="사진을 날짜별로 정리한다",
    )
    server = create_server(
        0,
        conversation_store=ConversationStore(path),
        task_store=TaskCheckpointStore(path),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        chat = _post_json(
            f"{base}/api/chat",
            {
                "text": "SemOp 지속 학습 작업 어디까지 했지?",
                "images": [],
                "resource_tier": "symbolic",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert chat["answer"]["status"] == "best_effort"
    assert "revision 3" in chat["answer"]["answer"]
    assert "교정 prototype 평가 통과" in chat["answer"]["answer"]
    assert "proof 경계를 유지" in chat["answer"]["answer"]
    assert "최신 student 후보 학습" in chat["answer"]["answer"]
    assert "오래된 행동-13" not in chat["answer"]["answer"]
    assert chat["answer"]["provenance"]["answer_source"] == "task_context"
    assert chat["answer"]["provenance"]["task_context_used_by_retriever"] is True
    assert chat["task_selection"]["mode"] == "automatic"
    assert chat["task_selection"]["task_id"] == target.task_id
    assert chat["task_selection"]["revision"] == latest.revision


def test_chat_checkpoint_command_persists_and_resumes_after_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "beginner.db"
    conversation_store = ConversationStore(path)
    task_store = TaskCheckpointStore(path)
    task = task_store.create_task(
        title="SemOp 지속 학습",
        objective="사용자 확인 checkpoint로 긴 작업을 이어간다",
    )
    task = task_store.checkpoint(
        task.task_id,
        progress="교정 prototype을 구현했다",
        decisions=("typed verifier를 유지한다",),
        next_actions=("이전 행동-13",),
    )
    server = create_server(
        0,
        conversation_store=conversation_store,
        task_store=task_store,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        saved = _post_json(
            f"{base}/api/chat",
            {
                "text": (
                    "작업 기록: 교정 기억 평가 완료 | "
                    "다음: student 후보 봉인 평가 | "
                    "결정: 사용자 교정은 best_effort 유지; 자동 활성화 금지"
                ),
                "images": [],
                "resource_tier": "symbolic",
                "task_id": task.task_id,
            },
        )
        session_id = str(saved["session_id"])
        progress_only = _post_json(
            f"{base}/api/chat",
            {
                "text": "작업 기록: 봉인 평가 준비 완료",
                "images": [],
                "resource_tier": "symbolic",
                "session_id": session_id,
                "task_id": task.task_id,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    reopened = TaskCheckpointStore(path).get(task.task_id)
    assert saved["answer"]["provenance"]["answer_source"] == (
        "task_checkpoint_write"
    )
    assert saved["saved_task_checkpoint"]["revision"] == 3
    assert saved["saved_task_checkpoint"]["next_actions"] == [
        "student 후보 봉인 평가"
    ]
    assert saved["saved_task_checkpoint"]["decisions"] == [
        "typed verifier를 유지한다",
        "사용자 교정은 best_effort 유지",
        "자동 활성화 금지",
    ]
    assert progress_only["saved_task_checkpoint"]["revision"] == 4
    assert reopened.progress == "봉인 평가 준비 완료"
    assert reopened.next_actions == ("student 후보 봉인 평가",)
    assert reopened.decisions[-1] == "자동 활성화 금지"

    restarted_server = create_server(
        0,
        conversation_store=ConversationStore(path),
        task_store=TaskCheckpointStore(path),
    )
    restarted_thread = threading.Thread(
        target=restarted_server.serve_forever,
        daemon=True,
    )
    restarted_thread.start()
    restarted_base = f"http://127.0.0.1:{restarted_server.server_address[1]}"
    try:
        resumed = _post_json(
            f"{restarted_base}/api/chat",
            {
                "text": "SemOp 지속 학습 작업 계속하자",
                "images": [],
                "resource_tier": "symbolic",
                "session_id": session_id,
            },
        )
    finally:
        restarted_server.shutdown()
        restarted_server.server_close()
        restarted_thread.join(timeout=5)

    assert resumed["answer"]["answer"] == "student 후보 봉인 평가"
    assert resumed["task_selection"]["mode"] == "automatic"
    assert resumed["task_selection"]["revision"] == 4
    assert len(resumed["turns"]) == 6


def test_chat_only_task_lifecycle_creates_records_and_completes_immutably(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chat-only-tasks.db"
    server = create_server(
        0,
        conversation_store=ConversationStore(path),
        task_store=TaskCheckpointStore(path),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        created = _post_json(
            f"{base}/api/chat",
            {
                "text": (
                    "작업 만들기: SemOp 채팅 생명주기 | "
                    "완료 조건: 채팅만으로 생성 기록 완료가 유지됨"
                ),
                "images": [],
                "resource_tier": "symbolic",
            },
        )
        session_id = str(created["session_id"])
        task_id = str(created["created_task"]["task_id"])
        recorded = _post_json(
            f"{base}/api/chat",
            {
                "text": (
                    "작업 기록: 생성과 기록 연결 완료 | "
                    "다음: 완료 명령 검증 | 결정: 명시적 명령만 저장"
                ),
                "images": [],
                "resource_tier": "symbolic",
                "session_id": session_id,
            },
        )
        completed = _post_json(
            f"{base}/api/chat",
            {
                "text": (
                    "작업 완료: 전체 생명주기 검증 통과 | "
                    "결정: 완료 revision은 immutable"
                ),
                "images": [],
                "resource_tier": "symbolic",
                "session_id": session_id,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    reopened_store = TaskCheckpointStore(path)
    restored = reopened_store.get(task_id)
    assert created["answer"]["provenance"]["answer_source"] == "task_create_write"
    assert created["created_task"]["revision"] == 1
    assert created["task_selection"]["mode"] == "created"
    assert recorded["answer"]["provenance"]["answer_source"] == (
        "task_checkpoint_write"
    )
    assert recorded["saved_task_checkpoint"]["revision"] == 2
    assert recorded["task_selection"]["reason"] == "only_active_task"
    assert completed["answer"]["provenance"]["answer_source"] == (
        "task_complete_write"
    )
    assert completed["completed_task"]["revision"] == 3
    assert completed["completed_task"]["status"] == "completed"
    assert completed["completed_task"]["next_actions"] == []
    assert restored.status is TaskStatus.COMPLETED
    assert restored.progress == "전체 생명주기 검증 통과"
    assert restored.decisions == (
        "명시적 명령만 저장",
        "완료 revision은 immutable",
    )
    assert reopened_store.list_tasks(include_completed=False) == ()
    assert [item.revision for item in reopened_store.history(task_id)] == [1, 2, 3]
    with pytest.raises(ValueError, match="immutable"):
        reopened_store.checkpoint(task_id, progress="완료 후 변경")
    assert len(completed["turns"]) == 6

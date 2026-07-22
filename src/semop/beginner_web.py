from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Sequence

from .assistant import SemOpAssistant
from .beginner import (
    BeginnerInputError,
    BeginnerReasoner,
    vision_presets_for_ui,
)
from .beginner_learning import (
    load_evaluated_controller,
    load_beginner_controller,
    load_beginner_rule_library,
)
from .chat_cli import create_local_assistant, semantic_activation_identity
from .conversation_memory import ConversationStore
from .kernel import TypedExperienceStore
from .prompt_api import AnswerEnvelope, AnswerStatus, PromptRequest, ResourceTier
from .semantic_experience import (
    SEMANTIC_REVIEWABLE_MEDIA_TYPES,
    SemanticTraceStore,
)
from .semantic_learning_cycle import inspect_semantic_learning_readiness
from .semantic_memory import ReviewedSemanticMemory
from .semantic_replay import SemanticReplayPlanner
from .task_memory import TaskCheckpointStore, task_query_kind


MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_BROWSER_IMAGE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class _TaskCheckpointCommand:
    progress: str | None
    decisions: tuple[str, ...] | None
    next_actions: tuple[str, ...] | None


@dataclass(frozen=True)
class _TaskCreateCommand:
    title: str
    objective: str


@dataclass(frozen=True)
class _TaskCompleteCommand:
    progress: str | None
    decisions: tuple[str, ...] | None


class BeginnerHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        service: BeginnerReasoner | None = None,
        semantic_trace_store: SemanticTraceStore | None = None,
        conversation_store: ConversationStore | None = None,
        task_store: TaskCheckpointStore | None = None,
    ) -> None:
        super().__init__(server_address, BeginnerRequestHandler)
        self.service = service or BeginnerReasoner()
        self.semantic_trace_store = semantic_trace_store
        self.conversation_store = conversation_store
        self.task_store = task_store
        self._prompt_assistants: dict[
            str, tuple[tuple[object, ...], SemOpAssistant]
        ] = {}

    def prompt_assistant(self, tier: ResourceTier | str) -> SemOpAssistant:
        resolved = ResourceTier(tier)
        key = resolved.value
        identity = (
            id(self.service.reasoner),
            id(self.service.policy),
            id(self.service.experience_collector),
            id(self.semantic_trace_store),
            (
                semantic_activation_identity()
                if resolved is ResourceTier.ECONOMY
                else ""
            ),
        )
        cached = self._prompt_assistants.get(key)
        if cached is not None and cached[0] == identity:
            return cached[1]
        configured = create_local_assistant(
            resolved,
            semantic_trace_store=self.semantic_trace_store,
        )
        assistant = SemOpAssistant(
            compiler=configured.compiler,
            reasoner=self.service.reasoner,
            policy=self.service.policy,
            experience_collector=self.service.experience_collector,
            semantic_trace_store=self.semantic_trace_store,
            config=configured.config,
        )
        self._prompt_assistants[key] = (identity, assistant)
        return assistant


class BeginnerRequestHandler(BaseHTTPRequestHandler):
    server: BeginnerHttpServer

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/tasks":
            store = self.server.task_store
            if store is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "장기 작업 checkpoint가 꺼져 있어."},
                )
                return
            try:
                tasks = store.list_tasks()
            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                self.log_error(
                    "task checkpoint list failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "장기 작업 목록을 읽지 못했어."},
                )
                return
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, "tasks": [task.to_dict() for task in tasks]},
            )
            return
        if path == "/api/memory":
            store = self.server.conversation_store
            if store is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "장기기억이 꺼져 있어."},
                )
                return
            try:
                memories = store.list_memories()
            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                self.log_error(
                    "explicit memory list failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "장기기억을 읽지 못했어."},
                )
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "memories": [memory.to_dict() for memory in memories],
                },
            )
            return
        conversation_match = re.fullmatch(
            r"/api/conversation/([A-Za-z0-9_-]{16,64})",
            path,
        )
        if conversation_match is not None:
            store = self.server.conversation_store
            if store is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "대화 기억이 꺼져 있어."},
                )
                return
            session_id = conversation_match.group(1)
            try:
                turns = store.history(session_id)
            except KeyError:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "저장된 대화를 찾지 못했어."},
                )
                return
            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                self.log_error(
                    "conversation history failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "대화 기록을 읽지 못했어."},
                )
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "session_id": session_id,
                    "turns": [turn.to_dict() for turn in turns],
                },
            )
            return
        media_match = re.fullmatch(
            r"/api/semantic/media/([0-9a-f]{64})/([0-3])",
            path,
        )
        if media_match is not None:
            store = self.server.semantic_trace_store
            if store is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "이미지 검토 기록이 꺼져 있어."},
                )
                return
            try:
                media = store.media_item(
                    media_match.group(1),
                    int(media_match.group(2)),
                )
            except KeyError:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "검토 이미지를 찾지 못했어."},
                )
                return
            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                self.log_error(
                    "semantic media verification failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"ok": False, "error": "검토 이미지의 digest가 맞지 않아."},
                )
                return
            if media.mime_type not in SEMANTIC_REVIEWABLE_MEDIA_TYPES:
                self._send_json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    {"ok": False, "error": "브라우저에서 확인할 수 없는 이미지 형식이야."},
                )
                return
            self._send_bytes(media.mime_type, media.content)
            return
        if path == "/":
            self._send_html(
                render_home_page(
                    experience_enabled=self.server.service.experience_enabled,
                    semantic_review_enabled=(
                        self.server.semantic_trace_store is not None
                    ),
                    conversation_enabled=(
                        self.server.conversation_store is not None
                    ),
                    task_enabled=(self.server.task_store is not None),
                    controller_summary=self.server.service.controller_summary,
                )
            )
            return
        if path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "service": "semop-beginner",
                    "experience_collection": (
                        self.server.service.experience_enabled
                    ),
                    "active_learned_rules": len(
                        self.server.service.active_rule_library.records
                    ),
                    "controller": self.server.service.controller_summary,
                    "conversation_memory": self.server.conversation_store is not None,
                    "explicit_long_term_memory": (
                        self.server.conversation_store is not None
                    ),
                    "long_task_checkpoints": self.server.task_store is not None,
                },
            )
            return
        if path == "/api/experience":
            try:
                snapshot = self.server.service.experience_snapshot()
            except Exception as exc:
                self.log_error(
                    "experience snapshot failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "로컬 학습 후보 큐를 읽지 못했어."},
                )
                return
            self._send_json(HTTPStatus.OK, {"ok": True, **snapshot})
            return
        if path == "/api/semantic":
            try:
                snapshot = self._semantic_snapshot()
            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                self.log_error(
                    "semantic review snapshot failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "로컬 의미 검토함을 읽지 못했어."},
                )
                return
            self._send_json(HTTPStatus.OK, {"ok": True, **snapshot})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "페이지를 찾지 못했어."})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in {
            "/api/chat",
            "/api/solve",
            "/api/experience/review",
            "/api/experience/review-example",
            "/api/experience/learn",
            "/api/semantic/review",
            "/api/conversation/new",
            "/api/memory",
            "/api/memory/delete",
            "/api/tasks",
            "/api/tasks/checkpoint",
            "/api/tasks/complete",
        }:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "요청 주소가 달라."})
            return

        try:
            payload = self._read_json()
            if path == "/api/chat":
                response = self._chat_payload(payload)
            elif path == "/api/solve":
                response = self._solve_payload(payload)
            elif path == "/api/experience/review-example":
                response = self._review_example_payload(payload)
            elif path == "/api/experience/learn":
                response = self._learn_payload(payload)
            elif path == "/api/semantic/review":
                response = self._semantic_review_payload(payload)
            elif path == "/api/conversation/new":
                response = self._conversation_new_payload()
            elif path == "/api/memory":
                response = self._memory_add_payload(payload)
            elif path == "/api/memory/delete":
                response = self._memory_delete_payload(payload)
            elif path == "/api/tasks":
                response = self._task_create_payload(payload)
            elif path == "/api/tasks/checkpoint":
                response = self._task_checkpoint_payload(payload)
            elif path == "/api/tasks/complete":
                response = self._task_complete_payload(payload)
            else:
                response = self._review_payload(payload)
        except BeginnerInputError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "요청 내용을 읽지 못했어. 화면을 새로고침해 줘."},
            )
            return
        except Exception as exc:
            self.log_error("solve failed: %s: %s", type(exc).__name__, exc)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": "풀이 중 오류가 났어. 터미널의 오류 기록을 확인해 줘."},
            )
            return

        self._send_json(HTTPStatus.OK, {"ok": True, **response})

    def _chat_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            tier = ResourceTier(str(payload.get("resource_tier", "auto")))
            images = _decode_browser_images(payload.get("images", ()))
            conversation_store = self.server.conversation_store
            text = str(payload.get("text", ""))
            session_id = (
                conversation_store.ensure_session(
                    str(payload.get("session_id", ""))
                )
                if conversation_store is not None
                else ""
            )
            memory_content = _explicit_memory_command(text)
            if memory_content is not None:
                if conversation_store is None:
                    raise BeginnerInputError(
                        "장기기억이 꺼져 있어. 기억 기능을 켠 뒤 다시 시도해 줘."
                    )
                memory = conversation_store.remember(
                    memory_content,
                    source_session_id=session_id,
                )
                answer = AnswerEnvelope(
                    answer=f"기억했어: {memory.content}",
                    status=AnswerStatus.BEST_EFFORT,
                    provenance={
                        "answer_source": "explicit_memory_write",
                        "semantic_authority": "explicit_user_memory_command",
                        "semantic_verified": False,
                        "logical_replay_verified": False,
                        "memory_id": memory.memory_id,
                    },
                )
                conversation_store.append_exchange(
                    session_id,
                    user_content=text,
                    assistant_content=answer.answer,
                    answer_status=answer.status.value,
                )
                return {
                    "answer": answer.to_dict(),
                    "session_id": session_id,
                    "turns": [
                        turn.to_dict()
                        for turn in conversation_store.history(session_id)
                    ],
                    "recalled_memories": [],
                    "recalled_episodes": [],
                    "task_contexts": [],
                    "saved_memory": memory.to_dict(),
                    "saved_correction": None,
                    "applied_correction": None,
                    "applied_correction_prototype": None,
                    "saved_task_checkpoint": None,
                    "task_selection": None,
                }
            correction_content = _explicit_correction_command(text)
            if correction_content is not None:
                if conversation_store is None:
                    raise BeginnerInputError(
                        "대화 기억이 꺼져 있어. 기억 기능을 켠 뒤 다시 시도해 줘."
                    )
                try:
                    previous_user, previous_assistant = (
                        conversation_store.latest_exchange(session_id)
                    )
                except KeyError as exc:
                    raise BeginnerInputError(
                        "먼저 질문을 한 뒤 `정정해: 올바른 답`이라고 적어 줘."
                    ) from exc
                correction = conversation_store.remember_correction(
                    prompt=previous_user.content,
                    prior_answer=previous_assistant.content,
                    corrected_answer=correction_content,
                    source_session_id=session_id,
                )
                answer = AnswerEnvelope(
                    answer=(
                        "정정을 기억했어. 같은 질문에는 이렇게 답할게:\n"
                        f"{correction.corrected_answer}"
                    ),
                    status=AnswerStatus.BEST_EFFORT,
                    provenance={
                        "answer_source": "user_correction_write",
                        "semantic_authority": "explicit_user_correction",
                        "semantic_verified": False,
                        "logical_replay_verified": False,
                        "correction_id": correction.correction_id,
                        "corrected_prompt": correction.prompt,
                    },
                )
                conversation_store.append_exchange(
                    session_id,
                    user_content=text,
                    assistant_content=answer.answer,
                    answer_status=answer.status.value,
                )
                return {
                    "answer": answer.to_dict(),
                    "session_id": session_id,
                    "turns": [
                        turn.to_dict()
                        for turn in conversation_store.history(session_id)
                    ],
                    "recalled_memories": [],
                    "recalled_episodes": [],
                    "task_contexts": [],
                    "saved_memory": None,
                    "saved_correction": correction.to_dict(),
                    "applied_correction": None,
                    "applied_correction_prototype": None,
                    "saved_task_checkpoint": None,
                    "task_selection": None,
                }
            create_command = _task_create_command(text)
            if create_command is not None:
                task_store = self.server.task_store
                if task_store is None:
                    raise BeginnerInputError("장기 작업 checkpoint가 꺼져 있어.")
                task = task_store.create_task(
                    title=create_command.title,
                    objective=create_command.objective,
                    source_session_id=session_id,
                )
                answer = AnswerEnvelope(
                    answer=(
                        f"{task.title} 장기 작업을 만들었어."
                        f"\n완료 조건: {task.objective}"
                    ),
                    status=AnswerStatus.BEST_EFFORT,
                    provenance={
                        "answer_source": "task_create_write",
                        "semantic_authority": "explicit_user_task_command",
                        "semantic_verified": False,
                        "logical_replay_verified": False,
                        "task_id": task.task_id,
                        "task_revision": task.revision,
                    },
                )
                if conversation_store is not None:
                    conversation_store.append_exchange(
                        session_id,
                        user_content=text,
                        assistant_content=answer.answer,
                        answer_status=answer.status.value,
                    )
                    turns = conversation_store.history(session_id)
                else:
                    turns = ()
                return {
                    "answer": answer.to_dict(),
                    "session_id": session_id,
                    "turns": [turn.to_dict() for turn in turns],
                    "recalled_memories": [],
                    "recalled_episodes": [],
                    "task_contexts": [task.to_context().to_dict()],
                    "saved_memory": None,
                    "saved_correction": None,
                    "applied_correction": None,
                    "applied_correction_prototype": None,
                    "created_task": task.to_dict(),
                    "saved_task_checkpoint": None,
                    "completed_task": None,
                    "task_selection": {
                        "mode": "created",
                        "task_id": task.task_id,
                        "revision": task.revision,
                        "score": 1.0,
                        "reason": "explicit_create_command",
                    },
                }
            checkpoint_command = _task_checkpoint_command(text)
            if checkpoint_command is not None:
                task_store = self.server.task_store
                if task_store is None:
                    raise BeginnerInputError("장기 작업 checkpoint가 꺼져 있어.")
                selected_task_id = str(payload.get("task_id", "")).strip()
                try:
                    if selected_task_id:
                        current_task = task_store.get(selected_task_id)
                        selection_mode = "explicit"
                        selection_score = 1.0
                        selection_reason = "browser_selection"
                    else:
                        task_match = task_store.retrieve_active_task(text)
                        if task_match is None:
                            raise BeginnerInputError(
                                "저장할 작업을 먼저 선택하거나 작업 이름을 함께 적어 줘."
                            )
                        current_task = task_match.checkpoint
                        selection_mode = "automatic"
                        selection_score = task_match.score
                        selection_reason = task_match.reason
                    decisions = (
                        None
                        if checkpoint_command.decisions is None
                        else tuple(
                            dict.fromkeys(
                                (
                                    *current_task.decisions,
                                    *checkpoint_command.decisions,
                                )
                            )
                        )
                    )
                    task = task_store.checkpoint(
                        current_task.task_id,
                        progress=checkpoint_command.progress,
                        decisions=decisions,
                        next_actions=checkpoint_command.next_actions,
                    )
                except KeyError as exc:
                    raise BeginnerInputError(
                        "checkpoint를 저장할 장기 작업을 찾지 못했어."
                    ) from exc
                next_action = task.next_actions[0] if task.next_actions else ""
                answer_text = (
                    f"{task.title} revision {task.revision} checkpoint를 저장했어."
                )
                if next_action:
                    answer_text += f"\n다음 행동: {next_action}"
                answer = AnswerEnvelope(
                    answer=answer_text,
                    status=AnswerStatus.BEST_EFFORT,
                    provenance={
                        "answer_source": "task_checkpoint_write",
                        "semantic_authority": "explicit_user_task_checkpoint",
                        "semantic_verified": False,
                        "logical_replay_verified": False,
                        "task_id": task.task_id,
                        "task_revision": task.revision,
                    },
                )
                if conversation_store is not None:
                    conversation_store.append_exchange(
                        session_id,
                        user_content=text,
                        assistant_content=answer.answer,
                        answer_status=answer.status.value,
                    )
                    turns = conversation_store.history(session_id)
                else:
                    turns = ()
                task_selection = {
                    "mode": selection_mode,
                    "task_id": task.task_id,
                    "revision": task.revision,
                    "score": selection_score,
                    "reason": selection_reason,
                }
                return {
                    "answer": answer.to_dict(),
                    "session_id": session_id,
                    "turns": [turn.to_dict() for turn in turns],
                    "recalled_memories": [],
                    "recalled_episodes": [],
                    "task_contexts": [task.to_context().to_dict()],
                    "saved_memory": None,
                    "saved_correction": None,
                    "applied_correction": None,
                    "applied_correction_prototype": None,
                    "created_task": None,
                    "saved_task_checkpoint": task.to_dict(),
                    "completed_task": None,
                    "task_selection": task_selection,
                }
            complete_command = _task_complete_command(text)
            if complete_command is not None:
                task_store = self.server.task_store
                if task_store is None:
                    raise BeginnerInputError("장기 작업 checkpoint가 꺼져 있어.")
                selected_task_id = str(payload.get("task_id", "")).strip()
                try:
                    if selected_task_id:
                        current_task = task_store.get(selected_task_id)
                        selection_mode = "explicit"
                        selection_score = 1.0
                        selection_reason = "browser_selection"
                    else:
                        task_match = task_store.retrieve_active_task(text)
                        if task_match is None:
                            raise BeginnerInputError(
                                "완료할 작업을 먼저 선택하거나 작업 이름을 함께 적어 줘."
                            )
                        current_task = task_match.checkpoint
                        selection_mode = "automatic"
                        selection_score = task_match.score
                        selection_reason = task_match.reason
                    decisions = (
                        None
                        if complete_command.decisions is None
                        else tuple(
                            dict.fromkeys(
                                (*current_task.decisions, *complete_command.decisions)
                            )
                        )
                    )
                    task = task_store.complete_task(
                        current_task.task_id,
                        progress=complete_command.progress,
                        decisions=decisions,
                    )
                except KeyError as exc:
                    raise BeginnerInputError(
                        "완료할 장기 작업을 찾지 못했어."
                    ) from exc
                answer = AnswerEnvelope(
                    answer=(
                        f"{task.title} 작업을 revision {task.revision}에서 "
                        "완료로 고정했어."
                    ),
                    status=AnswerStatus.BEST_EFFORT,
                    provenance={
                        "answer_source": "task_complete_write",
                        "semantic_authority": "explicit_user_task_command",
                        "semantic_verified": False,
                        "logical_replay_verified": False,
                        "task_id": task.task_id,
                        "task_revision": task.revision,
                    },
                )
                if conversation_store is not None:
                    conversation_store.append_exchange(
                        session_id,
                        user_content=text,
                        assistant_content=answer.answer,
                        answer_status=answer.status.value,
                    )
                    turns = conversation_store.history(session_id)
                else:
                    turns = ()
                return {
                    "answer": answer.to_dict(),
                    "session_id": session_id,
                    "turns": [turn.to_dict() for turn in turns],
                    "recalled_memories": [],
                    "recalled_episodes": [],
                    "task_contexts": [task.to_context().to_dict()],
                    "saved_memory": None,
                    "saved_correction": None,
                    "applied_correction": None,
                    "applied_correction_prototype": None,
                    "created_task": None,
                    "saved_task_checkpoint": None,
                    "completed_task": task.to_dict(),
                    "task_selection": {
                        "mode": selection_mode,
                        "task_id": task.task_id,
                        "revision": task.revision,
                        "score": selection_score,
                        "reason": selection_reason,
                    },
                }
            correction = (
                conversation_store.find_correction(text)
                if conversation_store is not None and text.strip() and not images
                else None
            )
            if correction is not None:
                answer = AnswerEnvelope(
                    answer=correction.corrected_answer,
                    status=AnswerStatus.BEST_EFFORT,
                    provenance={
                        "answer_source": "user_correction_memory",
                        "semantic_authority": "explicit_user_correction",
                        "semantic_verified": False,
                        "logical_replay_verified": False,
                        "correction_id": correction.correction_id,
                        "corrected_prompt": correction.prompt,
                    },
                )
                conversation_store.append_exchange(
                    session_id,
                    user_content=text,
                    assistant_content=answer.answer,
                    answer_status=answer.status.value,
                )
                return {
                    "answer": answer.to_dict(),
                    "session_id": session_id,
                    "turns": [
                        turn.to_dict()
                        for turn in conversation_store.history(session_id)
                    ],
                    "recalled_memories": [],
                    "recalled_episodes": [],
                    "task_contexts": [],
                    "saved_memory": None,
                    "saved_correction": None,
                    "applied_correction": correction.to_dict(),
                    "applied_correction_prototype": None,
                    "saved_task_checkpoint": None,
                    "task_selection": None,
                }
            prototype_matches = (
                conversation_store.search_correction_prototypes(text, limit=1)
                if conversation_store is not None and text.strip() and not images
                else ()
            )
            if prototype_matches:
                prototype_match = prototype_matches[0]
                prototype = prototype_match.prototype
                answer = AnswerEnvelope(
                    answer=prototype.corrected_answer,
                    status=AnswerStatus.BEST_EFFORT,
                    provenance={
                        "answer_source": "user_correction_prototype",
                        "semantic_authority": "consolidated_user_corrections",
                        "semantic_verified": False,
                        "logical_replay_verified": False,
                        "prototype_id": prototype.prototype_id,
                        "prototype_support": prototype.support,
                        "prototype_score": prototype_match.score,
                        "correction_ids": list(prototype.correction_ids),
                        "matched_prompt": prototype_match.matched_prompt,
                    },
                )
                conversation_store.append_exchange(
                    session_id,
                    user_content=text,
                    assistant_content=answer.answer,
                    answer_status=answer.status.value,
                )
                return {
                    "answer": answer.to_dict(),
                    "session_id": session_id,
                    "turns": [
                        turn.to_dict()
                        for turn in conversation_store.history(session_id)
                    ],
                    "recalled_memories": [],
                    "recalled_episodes": [],
                    "task_contexts": [],
                    "saved_memory": None,
                    "saved_correction": None,
                    "applied_correction": None,
                    "applied_correction_prototype": prototype_match.to_dict(),
                    "saved_task_checkpoint": None,
                    "task_selection": None,
                }
            prompt_assistant: SemOpAssistant | None = None
            recalled_memories = (
                conversation_store.search_memories(text, limit=3)
                if conversation_store is not None and text.strip()
                else ()
            )
            if (
                not recalled_memories
                and conversation_store is not None
                and text.strip()
                and not images
                and tier is not ResourceTier.SYMBOLIC
            ):
                prompt_assistant = self.server.prompt_assistant(tier)
                semantic_ranker = getattr(
                    prompt_assistant.compiler.retriever,
                    "rank_texts",
                    None,
                )
                if callable(semantic_ranker):
                    try:
                        recalled_memories = conversation_store.search_memories(
                            text,
                            limit=3,
                            semantic_ranker=prompt_assistant.compiler.retriever,
                        )
                    except RuntimeError:
                        recalled_memories = ()
            recalled_episodes = (
                conversation_store.recall_relevant_episodes(
                    text,
                    current_session_id=session_id,
                    limit=2,
                )
                if conversation_store is not None and text.strip() and not images
                else ()
            )
            task_store = self.server.task_store
            task_id = str(payload.get("task_id", "")).strip()
            task_selection: dict[str, object] | None = None
            try:
                if task_store is not None and task_id:
                    selected_task = task_store.get(task_id)
                    task_contexts = (selected_task.to_context(),)
                    task_selection = {
                        "mode": "explicit",
                        "task_id": selected_task.task_id,
                        "revision": selected_task.revision,
                        "score": 1.0,
                        "reason": "browser_selection",
                    }
                elif task_store is not None and task_query_kind(text) is not None:
                    match = task_store.retrieve_active_task(text)
                    task_contexts = (
                        (match.checkpoint.to_context(),) if match is not None else ()
                    )
                    if match is not None:
                        task_selection = {
                            "mode": "automatic",
                            "task_id": match.checkpoint.task_id,
                            "revision": match.checkpoint.revision,
                            "score": match.score,
                            "reason": match.reason,
                        }
                else:
                    task_contexts = ()
            except KeyError as exc:
                raise BeginnerInputError("선택한 장기 작업을 찾지 못했어.") from exc
            request = PromptRequest(
                text=text,
                images=images,
                resource_tier=tier,
                judge_opt_in=payload.get("judge_opt_in") is True,
                conversation=(
                    conversation_store.recent_messages(session_id)
                    if conversation_store is not None
                    else ()
                ),
                recalled_memories=recalled_memories,
                recalled_episodes=recalled_episodes,
                task_contexts=task_contexts,
            )
            answer = (prompt_assistant or self.server.prompt_assistant(tier)).solve(
                request
            )
            if conversation_store is not None:
                display_text = request.text or f"[이미지 {len(images)}장 첨부]"
                if images and request.text:
                    display_text += f"\n[이미지 {len(images)}장 첨부]"
                conversation_store.append_exchange(
                    session_id,
                    user_content=display_text,
                    assistant_content=answer.answer,
                    answer_status=answer.status.value,
                )
                turns = conversation_store.history(session_id)
            else:
                turns = ()
        except (OSError, sqlite3.Error, TypeError, ValueError, RuntimeError) as exc:
            raise BeginnerInputError(f"통합 입력을 처리하지 못했어: {exc}") from exc
        return {
            "answer": answer.to_dict(),
            "session_id": session_id,
            "turns": [turn.to_dict() for turn in turns],
            "recalled_memories": [
                memory.to_dict() for memory in request.recalled_memories
            ],
            "recalled_episodes": [
                episode.to_dict() for episode in request.recalled_episodes
            ],
            "task_contexts": [task.to_dict() for task in request.task_contexts],
            "saved_memory": None,
            "saved_correction": None,
            "applied_correction": None,
            "applied_correction_prototype": None,
            "saved_task_checkpoint": None,
            "task_selection": task_selection,
        }

    def _conversation_new_payload(self) -> dict[str, Any]:
        store = self.server.conversation_store
        if store is None:
            raise BeginnerInputError("대화 기억이 꺼져 있어.")
        return {"session_id": store.create_session(), "turns": []}

    def _memory_add_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        store = self.server.conversation_store
        if store is None:
            raise BeginnerInputError("장기기억이 꺼져 있어.")
        try:
            memory = store.remember(
                str(payload.get("content", "")),
                source_session_id=str(payload.get("session_id", "")),
            )
            memories = store.list_memories()
        except KeyError as exc:
            raise BeginnerInputError("연결할 대화 세션을 찾지 못했어.") from exc
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise BeginnerInputError(f"장기기억을 저장하지 못했어: {exc}") from exc
        return {
            "memory": memory.to_dict(),
            "memories": [item.to_dict() for item in memories],
        }

    def _memory_delete_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        store = self.server.conversation_store
        if store is None:
            raise BeginnerInputError("장기기억이 꺼져 있어.")
        try:
            deleted = store.forget(str(payload.get("memory_id", "")))
            memories = store.list_memories()
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise BeginnerInputError(f"장기기억을 지우지 못했어: {exc}") from exc
        if not deleted:
            raise BeginnerInputError("지울 장기기억을 찾지 못했어.")
        return {
            "deleted": True,
            "memories": [item.to_dict() for item in memories],
        }

    def _task_create_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        store = self.server.task_store
        if store is None:
            raise BeginnerInputError("장기 작업 checkpoint가 꺼져 있어.")
        try:
            task = store.create_task(
                title=str(payload.get("title", "")),
                objective=str(payload.get("objective", "")),
                source_session_id=str(payload.get("session_id", "")),
            )
            tasks = store.list_tasks()
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise BeginnerInputError(f"장기 작업을 만들지 못했어: {exc}") from exc
        return {
            "task": task.to_dict(),
            "tasks": [item.to_dict() for item in tasks],
        }

    def _task_checkpoint_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        store = self.server.task_store
        if store is None:
            raise BeginnerInputError("장기 작업 checkpoint가 꺼져 있어.")
        try:
            task = store.checkpoint(
                str(payload.get("task_id", "")),
                progress=str(payload.get("progress", "")),
                decisions=_payload_string_list(payload.get("decisions", [])),
                next_actions=_payload_string_list(payload.get("next_actions", [])),
            )
            tasks = store.list_tasks()
        except KeyError as exc:
            raise BeginnerInputError("갱신할 장기 작업을 찾지 못했어.") from exc
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise BeginnerInputError(f"작업 checkpoint를 저장하지 못했어: {exc}") from exc
        return {
            "task": task.to_dict(),
            "tasks": [item.to_dict() for item in tasks],
        }

    def _task_complete_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        store = self.server.task_store
        if store is None:
            raise BeginnerInputError("장기 작업 checkpoint가 꺼져 있어.")
        raw_progress = payload.get("progress")
        raw_decisions = payload.get("decisions")
        try:
            task = store.complete_task(
                str(payload.get("task_id", "")),
                progress=(None if raw_progress is None else str(raw_progress)),
                decisions=(
                    None
                    if raw_decisions is None
                    else _payload_string_list(raw_decisions)
                ),
            )
            tasks = store.list_tasks()
        except KeyError as exc:
            raise BeginnerInputError("완료할 장기 작업을 찾지 못했어.") from exc
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise BeginnerInputError(f"장기 작업을 완료하지 못했어: {exc}") from exc
        return {
            "task": task.to_dict(),
            "tasks": [item.to_dict() for item in tasks],
        }

    def _solve_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        domain = payload.get("domain", "")
        values = payload.get("values", {})
        if not isinstance(values, Mapping):
            raise BeginnerInputError("입력 형식이 올바르지 않아. 화면을 새로고침해 줘.")
        result = self.server.service.solve(str(domain), values)
        return {"result": result.to_dict()}

    def _review_example_payload(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        values = payload.get("values", {})
        if not isinstance(values, Mapping):
            raise BeginnerInputError("검토할 입력 형식이 올바르지 않아.")
        return self.server.service.review_example(
            str(payload.get("domain", "")),
            values,
            expected_solved=payload.get("expected_solved"),
            attest_human_review=payload.get("attest_human_review"),
            note=payload.get("note", ""),
        )

    def _review_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.server.service.review_experience(
            payload.get("request_digest", ""),
            expected_solved=payload.get("expected_solved"),
            include_for_learning=payload.get("include_for_learning", True),
            attest_human_review=payload.get("attest_human_review"),
            note=payload.get("note", ""),
        )

    def _learn_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if payload.get("confirm_learning") is not True:
            raise BeginnerInputError(
                "승인된 사례로 검증 학습을 실행한다는 확인이 필요해."
            )
        return {"learning": self.server.service.learn_from_reviewed_experience()}

    def _semantic_review_payload(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        store = self.server.semantic_trace_store
        if store is None:
            raise BeginnerInputError("이 실행에서는 의미 해석 학습 기록이 꺼져 있어.")
        accepted = payload.get("accepted")
        if type(accepted) is not bool:
            raise BeginnerInputError("해석이 맞는지 선택해 줘.")
        try:
            review = store.review(
                str(payload.get("trace_id", "")),
                accepted=accepted,
                reviewer="human:local-user",
                attest_human_review=payload.get("attest_human_review") is True,
                note=str(payload.get("note", "")),
            )
            output = store.path.parent.parent / "distillation" / "semantic-reviewed.jsonl"
            corpus = store.export_distillation(output)
        except (KeyError, OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise BeginnerInputError(f"의미 해석 검토를 저장하지 못했어: {exc}") from exc
        snapshot = self._semantic_snapshot()
        return {
            "review": review.to_dict(),
            "distillation_records": len(corpus.records),
            "distillation_path": str(output.resolve()),
            "semantic_stats": snapshot["stats"],
            **snapshot,
        }

    def _semantic_snapshot(self, *, limit: int = 8) -> dict[str, Any]:
        store = self.server.semantic_trace_store
        if store is None:
            return {
                "enabled": False,
                "stats": {},
                "student_learning": {},
                "items": [],
            }
        replay = SemanticReplayPlanner(store).select(limit=min(limit, 20))
        items: list[dict[str, Any]] = []
        for candidate in replay.candidates:
            record = candidate.trace
            has_images = record.has_images
            media = store.media(record.trace_id) if has_images else ()
            media_reviewable = (
                store.media_is_reviewable(record) if has_images else False
            )
            items.append(
                {
                    "trace_id": record.trace_id,
                    "prompt": record.prompt,
                    "domain": record.domain,
                    "completion": json.loads(record.completion_json),
                    "model_id": record.model_id,
                    "answer_status": record.answer_status,
                    "replay_verified": record.replay_verified,
                    "created_at": record.created_at,
                    "split": record.split.value,
                    "replay_priority": candidate.priority,
                    "replay_novelty": candidate.novelty,
                    "replay_reasons": list(candidate.reasons),
                    "has_images": has_images,
                    "media": [
                        {
                            "image_index": item.image_index,
                            "mime_type": item.mime_type,
                            "sha256": item.content_sha256,
                            "size": item.size,
                            "url": (
                                f"/api/semantic/media/{record.trace_id}/"
                                f"{item.image_index}"
                            ),
                        }
                        for item in media
                        if item.mime_type in SEMANTIC_REVIEWABLE_MEDIA_TYPES
                    ],
                    "semantic_only": (
                        has_images and media_reviewable and not record.replay_verified
                    ),
                    "can_accept": candidate.can_accept,
                    "can_reject": candidate.can_reject,
                    "blocked_reason": (
                        candidate.blocked_reason
                        or (
                            "정확한 PNG/JPEG 픽셀 증거가 journal에 없어 persistent 검토를 막았어."
                            if has_images and not media_reviewable
                            else ""
                        )
                    ),
                }
            )
        student_learning = inspect_semantic_learning_readiness(store)
        prototypes = ReviewedSemanticMemory(store).prototypes()
        return {
            "enabled": True,
            "stats": student_learning.journal.to_dict(),
            "student_learning": student_learning.to_dict(),
            "replay": {
                "pending_total": replay.pending_total,
                "reviewed_total": replay.reviewed_total,
                "pool_size": replay.pool_size,
                "selected": len(replay.candidates),
            },
            "consolidation": {
                "prototype_count": len(prototypes),
                "prototypes": [
                    {
                        "prototype_id": prototype.prototype_id,
                        "domain": prototype.domain,
                        "support": prototype.support,
                        "examples": list(prototype.prompts[:3]),
                    }
                    for prototype in prototypes
                ],
            },
            "items": items,
        }

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise BeginnerInputError("요청 크기를 확인할 수 없어.") from exc
        if length <= 0:
            raise BeginnerInputError("입력 내용이 비어 있어.")
        if length > MAX_REQUEST_BYTES:
            raise BeginnerInputError("입력이 너무 길어. 64KB보다 작게 줄여 줘.")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise BeginnerInputError("입력 형식이 올바르지 않아.")
        return payload

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._common_headers("text/html; charset=utf-8", len(encoded))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=None,
            default=str,
        ).encode("utf-8")
        self.send_response(status)
        self._common_headers("application/json; charset=utf-8", len(encoded))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_bytes(self, content_type: str, content: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self._common_headers(content_type, len(content))
        self.send_header("Content-Disposition", "inline")
        self.end_headers()
        self.wfile.write(content)

    def _common_headers(self, content_type: str, content_length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'unsafe-inline'; "
            "script-src 'unsafe-inline'; connect-src 'self'",
        )


def render_home_page(
    *,
    experience_enabled: bool = False,
    semantic_review_enabled: bool = False,
    conversation_enabled: bool = False,
    task_enabled: bool = False,
    controller_summary: Mapping[str, Any] | None = None,
) -> str:
    presets = json.dumps(vision_presets_for_ui(), ensure_ascii=False).replace("</", "<\\/")
    experience_notice = (
        "미증명·파싱 실패 같은 개선 후보는 이 PC의 로컬 검토 큐에 저장돼. "
        "외부 전송이나 자동 학습은 하지 않아."
        if experience_enabled
        else "이 실행에서는 입력을 학습 후보로 저장하지 않아."
    )
    controller_notice = (
        "검증된 작은 탐색 컨트롤러가 연산자 순서를 안내하고 있어. "
        "결론은 여전히 typed 실행과 proof replay가 검증해."
        if controller_summary and controller_summary.get("active") is True
        else "저장된 탐색 컨트롤러가 없어 결정론적 탐색을 사용하고 있어."
    )
    return _PAGE.replace("__VISION_PRESETS__", presets).replace(
        "__EXPERIENCE_NOTICE__",
        experience_notice,
    ).replace(
        "__CONTROLLER_NOTICE__",
        controller_notice,
    ).replace(
        "__EXPERIENCE_ENABLED__",
        "true" if experience_enabled else "false",
    ).replace(
        "__SEMANTIC_REVIEW_ENABLED__",
        "true" if semantic_review_enabled else "false",
    ).replace(
        "__CONVERSATION_ENABLED__",
        "true" if conversation_enabled else "false",
    ).replace(
        "__TASK_ENABLED__",
        "true" if task_enabled else "false",
    )


def _explicit_memory_command(text: str) -> str | None:
    match = re.match(
        r"^\s*기억해(?:줘)?\s*:\s*(?P<content>.*)$",
        str(text),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    content = match.group("content").strip()
    if not content:
        raise BeginnerInputError("기억할 내용을 `기억해: 내용`처럼 적어 줘.")
    return content


def _explicit_correction_command(text: str) -> str | None:
    match = re.match(
        r"^\s*정정해(?:줘)?\s*:\s*(?P<content>.*)$",
        str(text),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    content = match.group("content").strip()
    if not content:
        raise BeginnerInputError("올바른 답을 `정정해: 내용`처럼 적어 줘.")
    return content


def _task_create_command(text: str) -> _TaskCreateCommand | None:
    match = re.match(
        r"^\s*작업\s*(?:만들기|생성)\s*:\s*(?P<body>.*)$",
        str(text),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    title = ""
    objective = ""
    for segment in (item.strip() for item in match.group("body").split("|")):
        if not segment:
            continue
        field = re.match(
            r"^(?P<name>이름|제목|목표|완료\s*조건)\s*:\s*(?P<value>.*)$",
            segment,
            flags=re.DOTALL,
        )
        if field is None:
            if title:
                raise BeginnerInputError(
                    "작업 이름 뒤에는 `완료 조건:`을 붙여 줘."
                )
            title = segment
            continue
        name = field.group("name").replace(" ", "")
        value = field.group("value").strip()
        if name in {"이름", "제목"}:
            title = value
        else:
            objective = value
    if not title or not objective:
        raise BeginnerInputError(
            "`작업 만들기: 이름 | 완료 조건: 원하는 결과`처럼 적어 줘."
        )
    return _TaskCreateCommand(title, objective)


def _task_complete_command(text: str) -> _TaskCompleteCommand | None:
    match = re.match(
        r"^\s*작업\s*(?:완료|끝내기)\s*:\s*(?P<body>.*)$",
        str(text),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    progress: str | None = None
    decisions: tuple[str, ...] | None = None
    for segment in (item.strip() for item in match.group("body").split("|")):
        if not segment:
            continue
        field = re.match(
            r"^(?P<name>진행|결정)\s*:\s*(?P<value>.*)$",
            segment,
            flags=re.DOTALL,
        )
        if field is None:
            if progress is not None:
                raise BeginnerInputError(
                    "완료 내용에는 `진행:` 또는 `결정:`을 붙여 줘."
                )
            progress = segment
            continue
        name = field.group("name")
        value = field.group("value").strip()
        if not value:
            raise BeginnerInputError(f"`{name}:` 뒤에 내용을 적어 줘.")
        if name == "진행":
            progress = value
        else:
            decisions = tuple(
                item.strip() for item in value.split(";") if item.strip()
            )
    return _TaskCompleteCommand(progress, decisions)


def _task_checkpoint_command(text: str) -> _TaskCheckpointCommand | None:
    match = re.match(
        r"^\s*(?:작업\s*(?:기록|체크포인트|checkpoint)|체크포인트|checkpoint)"
        r"\s*:\s*(?P<body>.*)$",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    body = match.group("body").strip()
    if not body:
        raise BeginnerInputError(
            "저장할 내용을 `작업 기록: 진행 내용 | 다음: 할 일`처럼 적어 줘."
        )

    progress: str | None = None
    decisions: tuple[str, ...] | None = None
    next_actions: tuple[str, ...] | None = None
    for segment in (item.strip() for item in body.split("|")):
        if not segment:
            continue
        field = re.match(
            r"^(?P<name>진행|결정|다음)\s*:\s*(?P<value>.*)$",
            segment,
            flags=re.DOTALL,
        )
        if field is None:
            if progress is not None:
                raise BeginnerInputError(
                    "구분한 내용에는 `진행:`, `결정:`, `다음:`을 붙여 줘."
                )
            progress = segment
            continue
        name = field.group("name")
        value = field.group("value").strip()
        if not value:
            raise BeginnerInputError(f"`{name}:` 뒤에 내용을 적어 줘.")
        if name == "진행":
            progress = value
        elif name == "결정":
            decisions = tuple(
                item.strip() for item in value.split(";") if item.strip()
            )
        else:
            next_actions = tuple(
                item.strip() for item in value.split(";") if item.strip()
            )
    if progress is None and decisions is None and next_actions is None:
        raise BeginnerInputError("저장할 작업 내용을 적어 줘.")
    return _TaskCheckpointCommand(progress, decisions, next_actions)


def _decode_browser_images(value: Any) -> tuple[bytes, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, list) or len(value) > 4:
        raise BeginnerInputError("이미지는 최대 네 장까지 첨부할 수 있어.")
    decoded: list[bytes] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise BeginnerInputError("이미지 첨부 형식이 올바르지 않아.")
        data_url = str(item.get("data_url", ""))
        match = re.fullmatch(
            r"data:(image/(?:png|jpeg|x-portable-pixmap|x-portable-graymap|x-portable-bitmap));base64,(.+)",
            data_url,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            raise BeginnerInputError("PNG, JPEG 또는 Netpbm 이미지만 첨부할 수 있어.")
        try:
            raw = base64.b64decode(match.group(2), validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise BeginnerInputError("이미지 데이터를 읽지 못했어.") from exc
        if not raw or len(raw) > MAX_BROWSER_IMAGE_BYTES:
            raise BeginnerInputError("이미지 한 장은 4MB 이하여야 해.")
        decoded.append(raw)
    return tuple(decoded)


def _payload_string_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError("task checkpoint lists must be arrays")
    return tuple(str(item).strip() for item in value if str(item).strip())


def create_server(
    preferred_port: int = 8765,
    *,
    service: BeginnerReasoner | None = None,
    semantic_trace_store: SemanticTraceStore | None = None,
    conversation_store: ConversationStore | None = None,
    task_store: TaskCheckpointStore | None = None,
    attempts: int = 10,
) -> BeginnerHttpServer:
    if not 0 <= preferred_port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    ports = (
        (0,)
        if preferred_port == 0
        else range(preferred_port, min(65536, preferred_port + attempts))
    )
    last_error: OSError | None = None
    for port in ports:
        try:
            return BeginnerHttpServer(
                ("127.0.0.1", port),
                service,
                semantic_trace_store,
                conversation_store,
                task_store,
            )
        except OSError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SemOp 초보자용 로컬 브라우저 화면을 실행합니다."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="시작 포트입니다. 사용 중이면 다음 포트를 자동으로 찾습니다. (기본: 8765)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="브라우저를 자동으로 열지 않습니다.",
    )
    parser.add_argument(
        "--experience-db",
        type=Path,
        default=Path("artifacts/experience/beginner-experience.db"),
        help=(
            "미증명·파싱 실패를 저장할 로컬 검토 큐입니다. "
            "(기본: artifacts/experience/beginner-experience.db)"
        ),
    )
    parser.add_argument(
        "--no-experience",
        action="store_true",
        help="로컬 학습 후보 수집을 끕니다.",
    )
    parser.add_argument(
        "--rules-artifact",
        type=Path,
        default=Path("artifacts/rules/beginner-active-rules.json"),
        help=(
            "검증을 통과한 학습 규칙 파일입니다. "
            "(기본: artifacts/rules/beginner-active-rules.json)"
        ),
    )
    parser.add_argument(
        "--no-learned-rules",
        action="store_true",
        help="저장된 학습 규칙의 로드와 새 규칙 승격을 끕니다.",
    )
    parser.add_argument(
        "--controller-checkpoint-root",
        type=Path,
        default=Path("artifacts/controller/beginner-controller"),
        help=(
            "검증을 통과한 sparse controller 체크포인트 폴더입니다. "
            "(기본: artifacts/controller/beginner-controller)"
        ),
    )
    parser.add_argument(
        "--no-controller",
        action="store_true",
        help="저장된 controller 로드를 끕니다.",
    )
    parser.add_argument(
        "--controller-evaluation-report",
        type=Path,
        help=(
            "모든 gate를 통과한 NumPy controller 평가 보고서를 명시적으로 "
            "선택합니다. 지정하지 않으면 기존 sparse checkpoint를 사용합니다."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        experience_store = (
            None
            if args.no_experience
            else TypedExperienceStore(args.experience_db)
        )
        semantic_trace_store = (
            None
            if args.no_experience
            else SemanticTraceStore(args.experience_db)
        )
        conversation_store = (
            None
            if args.no_experience
            else ConversationStore(args.experience_db)
        )
        task_store = (
            None
            if args.no_experience
            else TaskCheckpointStore(args.experience_db)
        )
        loaded_rules = (
            None
            if args.no_learned_rules
            else load_beginner_rule_library(args.rules_artifact)
        )
        loaded_controller = (
            None
            if args.no_controller
            else (
                load_evaluated_controller(args.controller_evaluation_report)
                if args.controller_evaluation_report is not None
                else load_beginner_controller(args.controller_checkpoint_root)
            )
        )
        service = BeginnerReasoner(
            experience_store=experience_store,
            active_rule_library=(
                loaded_rules.library if loaded_rules is not None else None
            ),
            rules_artifact_path=(
                None if args.no_learned_rules else args.rules_artifact
            ),
            policy=(
                loaded_controller.policy
                if loaded_controller is not None
                else None
            ),
        )
        server = create_server(
            args.port,
            service=service,
            semantic_trace_store=semantic_trace_store,
            conversation_store=conversation_store,
            task_store=task_store,
        )
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"SemOp을 시작하지 못했어: {exc}")
        return 1

    port = int(server.server_address[1])
    url = f"http://127.0.0.1:{port}/"
    print("SemOp 쉬운 시작이 준비됐어.")
    print(f"브라우저 주소: {url}")
    if experience_store is not None:
        print(f"로컬 학습 후보 큐: {experience_store.path.resolve()}")
        print(f"로컬 대화 기억: {conversation_store.path.resolve()}")
        print(f"장기 작업 checkpoint: {task_store.path.resolve()}")
        print("저장된 후보는 사람 검토 전에는 학습에 반영되지 않아.")
    else:
        print("로컬 학습 후보 수집: 꺼짐")
    if loaded_rules is None:
        print("검증된 학습 규칙: 꺼짐")
    else:
        print(f"활성 학습 규칙: {len(loaded_rules.library.records)}개")
    if loaded_controller is None:
        print("작은 controller: 꺼짐")
    elif loaded_controller.active:
        print(
            "작은 controller: "
            f"{loaded_controller.kind}, {loaded_controller.parameter_count} parameters, "
            f"{loaded_controller.feature_profile} profile"
        )
    else:
        print("작은 controller: 체크포인트 없음, 결정론적 탐색 사용")
    print("끝낼 때는 이 창에서 Ctrl+C를 눌러 줘.")
    if not args.no_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nSemOp을 종료했어.")
    finally:
        server.server_close()
    return 0


_PAGE = r'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SemOp 쉬운 시작</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #22233a;
      --muted: #67677d;
      --paper: #fffdf9;
      --panel: #ffffff;
      --line: #e9e3dc;
      --accent: #8758c7;
      --accent-dark: #6841a1;
      --accent-soft: #f1eafd;
      --ok: #177754;
      --ok-soft: #e8f7f0;
      --warn: #9a5b12;
      --warn-soft: #fff4df;
      --fail: #a63b49;
      --fail-soft: #fff0f1;
      font-family: "Pretendard", "Noto Sans KR", "Malgun Gothic", system-ui, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 0%, #f8ebff 0, transparent 31rem),
        radial-gradient(circle at 90% 10%, #fff0df 0, transparent 27rem),
        var(--paper);
    }
    button, input, textarea, select { font: inherit; }
    .shell { width: min(980px, calc(100% - 32px)); margin: 0 auto; padding: 42px 0 64px; }
    .hero { text-align: center; margin-bottom: 24px; }
    .eyebrow {
      display: inline-block; padding: 7px 11px; border-radius: 999px;
      color: var(--accent-dark); background: var(--accent-soft); font-size: 13px; font-weight: 800;
    }
    h1 { margin: 14px 0 8px; font-size: clamp(32px, 6vw, 50px); letter-spacing: -0.04em; }
    .hero p { margin: 0 auto; max-width: 650px; color: var(--muted); line-height: 1.7; }
    .scope {
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
      margin: 24px 0; padding: 12px; background: rgba(255,255,255,.72);
      border: 1px solid var(--line); border-radius: 20px;
    }
    .scope div { padding: 11px; text-align: center; font-size: 14px; color: var(--muted); }
    .scope strong { display: block; margin-bottom: 3px; color: var(--ink); }
    .workspace {
      overflow: hidden; background: rgba(255,255,255,.93); border: 1px solid var(--line);
      border-radius: 26px; box-shadow: 0 22px 70px rgba(57, 39, 78, .10);
    }
    .chat-workspace {
      padding: clamp(20px, 5vw, 36px); background: rgba(255,255,255,.96);
      border: 1px solid var(--line); border-radius: 26px;
      box-shadow: 0 22px 70px rgba(57, 39, 78, .10);
    }
    .chat-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; }
    .chat-workspace h2 { margin: 0 0 7px; font-size: 25px; }
    .new-chat {
      flex: 0 0 auto; padding: 9px 12px; border: 1px solid #cab4e7; border-radius: 11px;
      color: var(--accent-dark); background: white; cursor: pointer; font-weight: 800;
    }
    .chat-transcript {
      display: grid; gap: 10px; max-height: 430px; overflow-y: auto;
      margin: 18px 0; padding: 16px; border: 1px solid var(--line);
      border-radius: 16px; background: #fbfaf8;
    }
    .chat-turn { max-width: 86%; padding: 11px 14px; border-radius: 14px; white-space: pre-wrap; line-height: 1.62; }
    .chat-turn.user { justify-self: end; background: var(--accent-soft); }
    .chat-turn.assistant { justify-self: start; background: white; border: 1px solid var(--line); }
    .chat-turn-role { display: block; margin-bottom: 4px; color: var(--muted); font-size: 11px; font-weight: 900; }
    .memory-manager {
      margin: 16px 0 20px; padding: 14px 16px; border: 1px solid #d9c8f1;
      border-radius: 14px; background: #faf7ff;
    }
    .memory-manager > summary { color: var(--accent-dark); }
    .memory-manager p { color: var(--muted); line-height: 1.55; font-size: 13px; }
    .memory-add { display: grid; grid-template-columns: 1fr auto; gap: 9px; }
    .memory-add button, .memory-delete {
      padding: 10px 13px; border: 1px solid #cab4e7; border-radius: 10px;
      color: var(--accent-dark); background: white; cursor: pointer; font-weight: 800;
    }
    .memory-list { display: grid; gap: 8px; margin-top: 12px; }
    .memory-item {
      display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
      padding: 10px 11px; border: 1px solid var(--line); border-radius: 11px; background: white;
    }
    .memory-item span { white-space: pre-wrap; line-height: 1.5; }
    .memory-delete { flex: 0 0 auto; padding: 6px 9px; color: var(--fail); }
    .task-manager {
      margin: 16px 0 20px; padding: 14px 16px; border: 1px solid #c9dbef;
      border-radius: 14px; background: #f6faff;
    }
    .task-manager > summary { color: #315d8c; }
    .task-manager p { color: var(--muted); line-height: 1.55; font-size: 13px; }
    .task-create { display: grid; gap: 9px; }
    .task-create textarea { min-height: 68px; }
    .task-create button, .task-action {
      padding: 10px 13px; border: 1px solid #a9c5e2; border-radius: 10px;
      color: #315d8c; background: white; cursor: pointer; font-weight: 800;
    }
    .task-list { display: grid; gap: 8px; margin-top: 12px; }
    .task-item { padding: 11px; border: 1px solid var(--line); border-radius: 11px; background: white; }
    .task-item.active { border-color: #7da9d6; box-shadow: 0 0 0 2px #e3effb; }
    .task-item-head { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
    .task-item-meta { color: var(--muted); font-size: 12px; }
    .task-editor { margin-top: 14px; padding-top: 14px; border-top: 1px solid #ccdaea; }
    .task-editor textarea { min-height: 66px; }
    .task-actions { display: flex; flex-wrap: wrap; gap: 8px; }
    .task-action.complete { color: var(--ok); }
    .task-action.detach { color: var(--muted); }
    .chat-controls { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .chat-workspace textarea { min-height: 130px; }
    .chat-result { margin-top: 20px; padding: 20px; border-radius: 16px; background: #faf8ff; border: 1px solid #dfd1f1; }
    .chat-answer { white-space: pre-wrap; line-height: 1.72; }
    .answer-badge { display: inline-block; margin-bottom: 10px; padding: 5px 9px; border-radius: 999px; font-size: 12px; font-weight: 900; }
    .answer-badge.verified { color: var(--ok); background: var(--ok-soft); }
    .answer-badge.conditional, .answer-badge.best_effort { color: var(--warn); background: var(--warn-soft); }
    .answer-badge.unsupported { color: var(--fail); background: var(--fail-soft); }
    .advanced { margin-top: 18px; }
    .advanced > summary { padding: 14px 18px; border: 1px solid var(--line); border-radius: 14px; background: rgba(255,255,255,.8); }
    .advanced[open] > summary { margin-bottom: 12px; }
    .tabs { display: grid; grid-template-columns: repeat(4, 1fr); padding: 10px; gap: 8px; border-bottom: 1px solid var(--line); }
    .tab {
      padding: 13px 8px; border: 0; border-radius: 14px; color: var(--muted);
      background: transparent; cursor: pointer; font-weight: 800;
    }
    .tab[aria-selected="true"] { color: var(--accent-dark); background: var(--accent-soft); }
    .pane { display: none; padding: clamp(20px, 5vw, 38px); }
    .pane.active { display: block; }
    .pane h2 { margin: 0 0 7px; font-size: 24px; letter-spacing: -0.025em; }
    .lead { margin: 0 0 22px; color: var(--muted); line-height: 1.65; }
    .field { margin-bottom: 18px; }
    label { display: block; margin-bottom: 7px; font-weight: 800; }
    .hint { display: block; margin-top: 6px; color: var(--muted); font-size: 13px; line-height: 1.5; }
    input, textarea, select {
      width: 100%; padding: 13px 14px; border: 1px solid #d9d2ca; border-radius: 12px;
      color: var(--ink); background: white; outline: none;
    }
    input:focus, textarea:focus, select:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    textarea { min-height: 78px; resize: vertical; }
    .examples { display: flex; flex-wrap: wrap; gap: 8px; margin: 3px 0 22px; }
    .example {
      padding: 8px 11px; border: 1px solid var(--line); border-radius: 999px;
      color: var(--accent-dark); background: white; cursor: pointer; font-size: 13px; font-weight: 700;
    }
    .example:hover { background: var(--accent-soft); }
    .solve {
      width: 100%; padding: 14px 18px; border: 0; border-radius: 13px;
      color: white; background: var(--accent); cursor: pointer; font-weight: 900;
      box-shadow: 0 9px 24px rgba(104, 65, 161, .22);
    }
    .solve:hover { background: var(--accent-dark); }
    .solve:disabled { opacity: .58; cursor: wait; }
    .vision-layout { display: grid; grid-template-columns: minmax(220px, .9fr) minmax(250px, 1.1fr); gap: 22px; align-items: center; }
    .preview-wrap { padding: 20px; border: 1px solid var(--line); border-radius: 18px; background: #faf8f4; }
    .pixel-grid { display: grid; gap: 3px; width: min(100%, 330px); margin: auto; }
    .pixel { aspect-ratio: 1; border-radius: 3px; border: 1px solid rgba(0,0,0,.035); }
    .pixel.dot { background: #fff; }
    .pixel.R { background: #f44336; }
    .pixel.G { background: #19b765; }
    .pixel.B { background: #2962ff; }
    .question { margin: 13px 0 0; text-align: center; font-weight: 800; line-height: 1.5; }
    .result { margin-top: 22px; padding: clamp(20px, 4vw, 30px); border-radius: 22px; border: 1px solid var(--line); background: var(--panel); }
    .result.ok { border-color: #b9e3d1; background: var(--ok-soft); }
    .result.warn { border-color: #f1d29e; background: var(--warn-soft); }
    .result.fail { border-color: #efc2c8; background: var(--fail-soft); }
    .result h2 { margin: 0 0 8px; font-size: 22px; }
    .result > p { margin: 0; line-height: 1.65; }
    .status-line { display: flex; align-items: center; gap: 9px; margin-bottom: 15px; color: var(--muted); font-size: 13px; font-weight: 800; }
    .dot-status { width: 10px; height: 10px; border-radius: 50%; background: currentColor; }
    .ok .status-line { color: var(--ok); }
    .warn .status-line { color: var(--warn); }
    .fail .status-line { color: var(--fail); }
    .interpreted { margin: 20px 0 0; padding: 16px 18px; border-radius: 14px; background: rgba(255,255,255,.72); }
    .interpreted strong { display: block; margin-bottom: 8px; }
    .interpreted ul { margin: 0; padding-left: 20px; line-height: 1.7; }
    .trust { margin-top: 16px; padding: 13px 15px; border-left: 4px solid currentColor; border-radius: 8px; background: rgba(255,255,255,.62); line-height: 1.65; font-size: 14px; }
    details { margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(0,0,0,.10); }
    summary { cursor: pointer; font-weight: 800; }
    pre { overflow: auto; white-space: pre-wrap; padding: 14px; border-radius: 12px; background: #242334; color: #f7f4ff; line-height: 1.62; font-size: 13px; }
    .foot { margin-top: 18px; text-align: center; color: var(--muted); font-size: 13px; line-height: 1.6; }
    .review-actions, .experience {
      margin-top: 18px; padding: 18px; border: 1px solid #d9c8f1;
      border-radius: 16px; background: #faf7ff;
    }
    .review-actions h3, .experience h2 { margin: 0 0 7px; }
    .review-actions p, .experience p { color: var(--muted); line-height: 1.6; }
    .attest { display: flex; gap: 9px; align-items: flex-start; margin: 13px 0; font-weight: 700; }
    .attest input { width: auto; margin-top: 4px; }
    .review-buttons { display: flex; flex-wrap: wrap; gap: 9px; }
    .review-buttons button, .experience-refresh {
      padding: 10px 13px; border: 1px solid #cab4e7; border-radius: 11px;
      color: var(--accent-dark); background: white; cursor: pointer; font-weight: 800;
    }
    .experience-head { display: flex; justify-content: space-between; gap: 16px; align-items: center; }
    .experience-stats { margin: 12px 0; color: var(--muted); font-size: 14px; }
    .queue-list { display: grid; gap: 12px; }
    .queue-item { padding: 15px; border: 1px solid var(--line); border-radius: 13px; background: white; }
    .queue-item strong { display: block; margin-bottom: 6px; }
    .queue-meta { margin-bottom: 9px; color: var(--muted); font-size: 12px; }
    .queue-item pre { max-height: 180px; margin: 10px 0; }
    .semantic-media { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }
    .semantic-media img {
      max-width: min(100%, 320px); max-height: 240px; object-fit: contain;
      border: 1px solid var(--line); border-radius: 10px; background: #f4f2ef;
    }
    .review-status { min-height: 1.5em; margin: 9px 0 0; color: var(--accent-dark); font-weight: 800; }
    @media (max-width: 680px) {
      .shell { width: min(100% - 20px, 980px); padding-top: 24px; }
      .scope { grid-template-columns: 1fr; }
      .chat-controls { grid-template-columns: 1fr; }
      .chat-heading { display: block; }
      .new-chat { margin: 0 0 14px; }
      .chat-turn { max-width: 94%; }
      .memory-add { grid-template-columns: 1fr; }
      .scope div { text-align: left; }
      .vision-layout { grid-template-columns: 1fr; }
      .tab { font-size: 14px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="hero">
      <span class="eyebrow">내 PC에서 돌아가는 검증형 추론</span>
      <h1>SemOp 쉬운 시작</h1>
      <p>전문 명령어 없이 예제를 고르거나 빈칸을 채워 봐. SemOp이 작은 typed 연산자를 조합해 결론을 찾고, 찾은 과정은 처음부터 다시 실행해 확인해.</p>
    </header>

    <section class="scope" aria-label="현재 지원 범위">
      <div><strong>코딩</strong>C++ 생성·컴파일·실행 검증</div>
      <div><strong>언어</strong>목표와 필요 조건 확인</div>
      <div><strong>수학</strong>정확한 계산식과 일차방정식</div>
      <div><strong>비전</strong>작은 색상 격자 측정</div>
    </section>

    <section class="chat-workspace">
      <div class="chat-heading">
        <div>
          <h2>그냥 하고 싶은 말을 적어 줘</h2>
          <p class="lead">SemOp이 최근 대화 문맥을 참고하고, 가능한 문제는 typed 연산자로 검증해. 일반 대화와 모델 해석은 미검증이라고 분명히 표시할게.</p>
        </div>
        <button class="new-chat" id="new-chat" type="button" hidden>새 대화</button>
      </div>
      <section class="chat-transcript" id="chat-transcript" hidden aria-label="저장된 대화"></section>
      <details class="memory-manager" id="memory-manager" hidden>
        <summary>장기기억 관리 <span id="memory-count"></span></summary>
        <p>다음 대화에서도 참고할 짧은 정보만 직접 저장해 줘. 저장된 내용은 미검증 개인화 문맥이며 proof 사실이나 자동 학습 자료가 아니야.</p>
        <form class="memory-add" id="memory-form">
          <input id="memory-content" maxlength="1000" placeholder="예: 내 프로젝트 이름은 SemOp이고 Python을 사용해">
          <button type="submit">기억하기</button>
        </form>
        <div class="memory-list" id="memory-list"></div>
        <p class="review-status" id="memory-status" aria-live="polite"></p>
      </details>
      <details class="task-manager" id="task-manager" hidden>
        <summary>장기 작업 checkpoint <span id="task-count"></span></summary>
        <p>여러 날 이어갈 목표를 revision으로 남겨. 채팅에서도 `작업 만들기: 이름 | 완료 조건: 결과`, `작업 기록: 진행 | 다음: 행동`, `작업 완료: 최종 결과`를 사용할 수 있어.</p>
        <form class="task-create" id="task-create-form">
          <input id="task-title" maxlength="120" placeholder="작업 이름: SemOp 로컬 챗봇 완성">
          <textarea id="task-objective" maxlength="2000" placeholder="완료 조건과 원하는 결과를 적어 줘"></textarea>
          <button type="submit">새 장기 작업 만들기</button>
        </form>
        <div class="task-list" id="task-list"></div>
        <section class="task-editor" id="task-editor" hidden>
          <strong id="task-editor-title"></strong>
          <div class="field">
            <label for="task-progress">현재 진행 상황</label>
            <textarea id="task-progress" maxlength="2000"></textarea>
          </div>
          <div class="field">
            <label for="task-decisions">확정한 결정, 한 줄에 하나</label>
            <textarea id="task-decisions"></textarea>
          </div>
          <div class="field">
            <label for="task-next-actions">다음 행동, 한 줄에 하나</label>
            <textarea id="task-next-actions"></textarea>
          </div>
          <div class="task-actions">
            <button class="task-action" id="task-save" type="button">checkpoint 저장</button>
            <button class="task-action complete" id="task-complete" type="button">작업 완료</button>
            <button class="task-action detach" id="task-detach" type="button">채팅 문맥에서 해제</button>
          </div>
        </section>
        <p class="review-status" id="task-status" aria-live="polite"></p>
      </details>
      <form id="chat-form">
        <div class="examples">
          <button class="example chat-example" type="button" data-prompt="수학: (2 + 3) * 4">계산 예제</button>
          <button class="example chat-example" type="button" data-prompt="Goal: 배포&#10;Requires: 테스트 통과, 관리자 승인&#10;Satisfied: 테스트 통과&#10;Satisfied: 관리자 승인">조건 예제</button>
          <button class="example chat-example" type="button" data-prompt="코딩: Given a weighted graph with N nodes and M nonnegative edges, print the shortest distance from node 1 to every node. Input and output are standard.">코딩 예제</button>
          <button class="example chat-example" type="button" data-prompt="작업 기록: 현재 단계 완료 | 다음: 다음 단계 검증 | 결정: 검증 결과만 채택">작업 기록 예제</button>
          <button class="example chat-example" type="button" data-prompt="작업 만들기: SemOp 장기 작업 | 완료 조건: 원하는 결과와 검증을 모두 통과">작업 만들기 예제</button>
        </div>
        <div class="field">
          <label for="chat-text">질문 또는 문제</label>
          <textarea id="chat-text" maxlength="16000" placeholder="예: 어떤 수에 2를 더하면 5야. 기억해: 내 프로젝트 이름은 SemOp이야. 정정해: 올바른 답"></textarea>
        </div>
        <div class="chat-controls">
          <div class="field">
            <label for="chat-images">이미지 첨부</label>
            <input id="chat-images" type="file" accept="image/png,image/jpeg,.pbm,.pgm,.ppm,.pnm" multiple>
            <span class="hint">작은 색상 이미지는 픽셀로 검증하고, 일반 사진은 로컬 모델이 있을 때만 미검증 후보로 해석해.</span>
          </div>
          <div class="field">
            <label for="chat-tier">자원 등급</label>
            <select id="chat-tier">
              <option value="auto">자동: symbolic 우선, 필요할 때 2B</option>
              <option value="symbolic">symbolic: 모델 없이 가장 가볍게</option>
              <option value="economy">economy: 0.8B 학생 모델</option>
              <option value="balanced">balanced: 2B 의미 모델</option>
            </select>
          </div>
        </div>
        <button class="solve" id="chat-submit" type="submit">연산자로 풀기</button>
      </form>
      <section class="chat-result" id="chat-result" hidden aria-live="polite">
        <span class="answer-badge" id="chat-status"></span>
        <div class="chat-answer" id="chat-answer"></div>
        <details><summary>검증 과정 보기</summary><pre id="chat-proof"></pre></details>
        <details><summary>해석·자원 정보 보기</summary><pre id="chat-technical"></pre></details>
        <section class="review-actions" id="chat-semantic-review" hidden>
          <h3>로컬 의미 학습에 알려 주기</h3>
          <p>모델이 질문을 typed 식이나 조건으로 옮긴 내용이 맞는지 확인해 줘. 맞다고 해도 proof replay를 통과한 기록만 학생 모델 학습 후보가 돼.</p>
          <label class="attest"><input id="chat-review-attest" type="checkbox">위 질문, 모델 해석, 검증 결과를 직접 확인했어.</label>
          <div class="review-buttons">
            <button type="button" data-semantic-review="true">해석이 맞아</button>
            <button type="button" data-semantic-review="false">해석이 틀려</button>
          </div>
          <p class="review-status" id="chat-review-status" aria-live="polite"></p>
        </section>
      </section>
    </section>

    <section class="experience" id="semantic-review-panel" hidden>
      <div class="experience-head">
        <div>
          <h2>모델 해석 검토함</h2>
          <p>지나간 질문도 다시 확인해 학생 모델 학습 자료로 표시할 수 있어. 확인 전에는 어떤 모델에도 반영되지 않아.</p>
        </div>
        <button class="experience-refresh" id="semantic-review-refresh" type="button">새로고침</button>
      </div>
      <div class="experience-stats" id="semantic-review-stats"></div>
      <p class="review-status" id="semantic-learning-readiness" aria-live="polite"></p>
      <label class="attest"><input id="semantic-queue-attest" type="checkbox">선택할 질문과 typed 해석을 직접 확인했어.</label>
      <div class="queue-list" id="semantic-review-list"></div>
      <p class="review-status" id="semantic-queue-status" aria-live="polite"></p>
    </section>

    <details class="advanced">
      <summary>고급 직접 입력: 도메인을 직접 고르기</summary>
    <section class="workspace">
      <nav class="tabs" aria-label="문제 종류">
        <button class="tab" type="button" data-domain="coding" aria-selected="true">1. 코딩</button>
        <button class="tab" type="button" data-domain="language" aria-selected="false">2. 언어 조건</button>
        <button class="tab" type="button" data-domain="math" aria-selected="false">3. 수학식</button>
        <button class="tab" type="button" data-domain="vision" aria-selected="false">4. 색상 비전</button>
      </nav>

      <form class="pane active" id="pane-coding" data-domain="coding">
        <h2>알고리즘 문제를 C++로 풀고 검증하기</h2>
        <p class="lead">문제 한 문장을 넣으면 후보 알고리즘을 고르고, 코드를 컴파일한 뒤 등록된 표본·무작위 테스트를 실행해.</p>
        <div class="examples">
          <button class="example" type="button" data-coding-example="dijkstra">예제: 최단 거리</button>
          <button class="example" type="button" data-coding-example="prefix">예제: 구간 합</button>
        </div>
        <div class="field">
          <label for="coding-statement">풀고 싶은 문제</label>
          <textarea id="coding-statement" maxlength="4000" required>Given a weighted graph with N nodes and M nonnegative edges, print the shortest distance from node 1 to every node.</textarea>
          <span class="hint">현재는 검증기가 등록된 대표 알고리즘 문제에서 가장 신뢰할 수 있어.</span>
        </div>
        <button class="solve" type="submit">C++ 풀이 만들고 검증하기</button>
      </form>

      <form class="pane" id="pane-language" data-domain="language">
        <h2>조건이 모두 갖춰졌는지 확인하기</h2>
        <p class="lead">목표 하나와 그 목표에 필요한 조건을 적어 줘. 쉼표로 여러 개를 나눌 수 있어.</p>
        <div class="examples">
          <button class="example" type="button" data-language-example="ready">예제: 배포 준비 완료</button>
          <button class="example" type="button" data-language-example="blocked">예제: 승인에서 막힘</button>
          <button class="example" type="button" data-language-example="missing">예제: 확인이 덜 됨</button>
        </div>
        <div class="field">
          <label for="lang-goal">무엇을 하려는 거야?</label>
          <input id="lang-goal" value="배포" maxlength="80" required>
        </div>
        <div class="field">
          <label for="lang-required">꼭 필요한 조건</label>
          <textarea id="lang-required" required>테스트 통과, 관리자 승인</textarea>
          <span class="hint">예: 테스트 통과, 관리자 승인</span>
        </div>
        <div class="field">
          <label for="lang-satisfied">이미 충족한 조건</label>
          <textarea id="lang-satisfied">테스트 통과, 관리자 승인</textarea>
        </div>
        <div class="field">
          <label for="lang-blocked">막혔거나 실패한 조건</label>
          <textarea id="lang-blocked" placeholder="없으면 비워 둬"></textarea>
        </div>
        <button class="solve" type="submit">조건 검증하기</button>
      </form>

      <form class="pane" id="pane-math" data-domain="math">
        <h2>수학식이 맞는지 계산하고 검증하기</h2>
        <p class="lead">지원 범위 안에서는 답만 내지 않고, 리터럴과 연산을 한 단계씩 실행한 뒤 다시 확인해.</p>
        <div class="examples">
          <button class="example" type="button" data-math-example="(2 + 3) * 4 == 20">예제: 사칙연산</button>
          <button class="example" type="button" data-math-example="3*x + 2 = 11">예제: 일차방정식</button>
          <button class="example" type="button" data-math-example="7 &lt; 10">예제: 크기 비교</button>
        </div>
        <div class="field">
          <label for="math-expression">검증할 식</label>
          <input id="math-expression" value="(2 + 3) * 4 == 20" maxlength="240" required>
          <span class="hint">예: (2 + 3) * 4 == 20 또는 3*x + 2 = 11</span>
        </div>
        <button class="solve" type="submit">수학식 검증하기</button>
      </form>

      <form class="pane" id="pane-vision" data-domain="vision">
        <h2>작은 색상 장면을 연산자로 살펴보기</h2>
        <p class="lead">아직 일반 사진은 아니야. 먼저 정확히 검증할 수 있는 색상 픽셀 장면 네 개로 작동 방식을 체험해 봐.</p>
        <div class="vision-layout">
          <div class="field">
            <label for="vision-preset">장면과 질문</label>
            <select id="vision-preset"></select>
            <span class="hint" id="vision-description"></span>
          </div>
          <div class="preview-wrap">
            <div class="pixel-grid" id="pixel-grid" aria-label="색상 장면 미리보기"></div>
            <p class="question" id="vision-question"></p>
          </div>
        </div>
        <button class="solve" type="submit">색상 장면 검증하기</button>
      </form>
    </section>
    </details>

    <section class="result" id="result" hidden aria-live="polite">
      <div class="status-line"><span class="dot-status"></span><span id="result-status"></span></div>
      <h2 id="result-title"></h2>
      <p id="result-summary"></p>
      <div class="interpreted">
        <strong>SemOp이 이렇게 이해했어</strong>
        <ul id="result-input"></ul>
      </div>
      <div class="trust" id="result-trust"></div>
      <details id="result-artifact-wrap" hidden open>
        <summary>생성된 코드 보기</summary>
        <pre id="result-artifact"></pre>
      </details>
      <div class="review-actions" id="result-review" hidden>
        <h3>이 입력을 학습 후보로 검토하기</h3>
        <p>현재 결과를 그대로 믿는 버튼이 아니야. 정확한 입력을 보고 앞으로 이 문제가 풀려야 하는지 직접 표시해 줘.</p>
        <label class="attest"><input id="current-review-attest" type="checkbox">나는 위 입력과 기대 결과를 직접 확인했어.</label>
        <div class="review-buttons">
          <button type="button" data-review-current="true">이 문제는 풀려야 해</button>
          <button type="button" data-review-current="false">이 문제는 풀리지 않아야 해</button>
        </div>
        <p class="review-status" id="current-review-status" aria-live="polite"></p>
      </div>
      <details>
        <summary>검증 과정 보기</summary>
        <pre id="result-proof"></pre>
      </details>
      <details>
        <summary>연구용 기술 정보 보기</summary>
        <pre id="result-technical"></pre>
      </details>
    </section>

    <section class="experience" id="experience-panel" hidden>
      <div class="experience-head">
        <div>
          <h2>로컬 학습 후보 검토</h2>
          <p>미증명·파싱 실패처럼 자동으로 모인 입력을 확인해. 사람 승인 전에는 학습에 쓰이지 않아.</p>
        </div>
        <div class="review-buttons">
          <button class="experience-refresh" id="experience-refresh" type="button">새로고침</button>
          <button class="experience-refresh" id="experience-learn" type="button">검증 학습 시도</button>
        </div>
      </div>
      <div class="experience-stats" id="experience-stats"></div>
      <label class="attest"><input id="queue-review-attest" type="checkbox">아래에서 선택할 입력과 기대 결과를 직접 확인했어.</label>
      <div class="queue-list" id="experience-list"></div>
      <p class="review-status" id="queue-review-status" aria-live="polite"></p>
      <p class="review-status" id="experience-learning-status" aria-live="polite"></p>
    </section>

    <p class="foot">모든 처리는 이 PC의 로컬 서버에서 이뤄져. 일반 대화는 로컬 모델의 best-effort 답변이고, 검증 가능한 문제만 typed proof 상태로 표시해.<br>__EXPERIENCE_NOTICE__<br>__CONTROLLER_NOTICE__</p>
  </main>

  <script>
    const visionPresets = __VISION_PRESETS__;
    const experienceEnabled = __EXPERIENCE_ENABLED__;
    const semanticReviewEnabled = __SEMANTIC_REVIEW_ENABLED__;
    const conversationEnabled = __CONVERSATION_ENABLED__;
    const taskEnabled = __TASK_ENABLED__;
    const panes = [...document.querySelectorAll('.pane')];
    const tabs = [...document.querySelectorAll('.tab')];
    const resultBox = document.getElementById('result');
    const visionSelect = document.getElementById('vision-preset');
    const resultReview = document.getElementById('result-review');
    const experiencePanel = document.getElementById('experience-panel');
    const semanticReviewPanel = document.getElementById('semantic-review-panel');
    const chatForm = document.getElementById('chat-form');
    const chatResult = document.getElementById('chat-result');
    const chatTranscript = document.getElementById('chat-transcript');
    const newChatButton = document.getElementById('new-chat');
    const memoryManager = document.getElementById('memory-manager');
    const memoryForm = document.getElementById('memory-form');
    const memoryList = document.getElementById('memory-list');
    const taskManager = document.getElementById('task-manager');
    const taskList = document.getElementById('task-list');
    const taskEditor = document.getElementById('task-editor');
    const conversationStorageKey = 'semop.conversation.session.v1';
    const taskStorageKey = 'semop.task.active.v1';
    let conversationSessionId = conversationEnabled
      ? (window.localStorage.getItem(conversationStorageKey) || '')
      : '';
    let activeTaskId = taskEnabled
      ? (window.localStorage.getItem(taskStorageKey) || '')
      : '';
    let knownTasks = [];
    let lastSubmission = null;
    let currentSemanticTraceId = '';

    function rememberConversationSession(sessionId) {
      conversationSessionId = sessionId || '';
      if (conversationSessionId) {
        window.localStorage.setItem(conversationStorageKey, conversationSessionId);
      } else {
        window.localStorage.removeItem(conversationStorageKey);
      }
    }

    function renderConversation(turns) {
      chatTranscript.replaceChildren();
      if (!conversationEnabled || !Array.isArray(turns) || !turns.length) {
        chatTranscript.hidden = true;
        return;
      }
      turns.forEach(turn => {
        const item = document.createElement('article');
        item.className = `chat-turn ${turn.role === 'user' ? 'user' : 'assistant'}`;
        const role = document.createElement('span');
        role.className = 'chat-turn-role';
        role.textContent = turn.role === 'user'
          ? '나'
          : `SemOp${turn.status ? ` · ${turn.status}` : ''}`;
        const content = document.createElement('span');
        content.textContent = turn.content;
        item.append(role, content);
        chatTranscript.appendChild(item);
      });
      chatTranscript.hidden = false;
      chatTranscript.scrollTop = chatTranscript.scrollHeight;
    }

    function renderMemories(memories) {
      memoryList.replaceChildren();
      const items = Array.isArray(memories) ? memories : [];
      document.getElementById('memory-count').textContent = `(${items.length})`;
      if (!items.length) {
        const empty = document.createElement('p');
        empty.textContent = '직접 저장한 장기기억이 아직 없어.';
        memoryList.appendChild(empty);
        return;
      }
      items.forEach(memory => {
        const row = document.createElement('div');
        row.className = 'memory-item';
        const content = document.createElement('span');
        content.textContent = memory.content;
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'memory-delete';
        remove.textContent = '지우기';
        remove.addEventListener('click', async () => {
          const status = document.getElementById('memory-status');
          status.textContent = '장기기억을 지우는 중...';
          try {
            const data = await postJson('/api/memory/delete', {memory_id: memory.memory_id});
            renderMemories(data.memories);
            status.textContent = '장기기억을 완전히 지웠어.';
          } catch (error) {
            status.textContent = error.message || '장기기억을 지우지 못했어.';
          }
        });
        row.append(content, remove);
        memoryList.appendChild(row);
      });
    }

    async function refreshMemories() {
      if (!conversationEnabled) return;
      const status = document.getElementById('memory-status');
      try {
        const response = await fetch('/api/memory');
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || '장기기억을 읽지 못했어.');
        renderMemories(data.memories);
        status.textContent = '';
      } catch (error) {
        status.textContent = error.message || '장기기억을 읽지 못했어.';
      }
    }

    memoryForm.addEventListener('submit', async event => {
      event.preventDefault();
      const input = document.getElementById('memory-content');
      const content = input.value.trim();
      const status = document.getElementById('memory-status');
      if (!content) {
        status.textContent = '기억할 내용을 한 줄 적어 줘.';
        return;
      }
      status.textContent = '장기기억에 저장하는 중...';
      try {
        const data = await postJson('/api/memory', {
          content,
          session_id: conversationSessionId
        });
        renderMemories(data.memories);
        input.value = '';
        status.textContent = '이 PC의 장기기억에 저장했어.';
      } catch (error) {
        status.textContent = error.message || '장기기억을 저장하지 못했어.';
      }
    });

    function rememberActiveTask(taskId) {
      activeTaskId = taskId || '';
      if (activeTaskId) {
        window.localStorage.setItem(taskStorageKey, activeTaskId);
      } else {
        window.localStorage.removeItem(taskStorageKey);
      }
    }

    function renderTaskEditor() {
      const task = knownTasks.find(item =>
        item.task_id === activeTaskId && item.status === 'active'
      );
      if (!task) {
        if (activeTaskId) rememberActiveTask('');
        taskEditor.hidden = true;
        return;
      }
      document.getElementById('task-editor-title').textContent =
        `${task.title} · revision ${task.revision}`;
      document.getElementById('task-progress').value = task.progress || '';
      document.getElementById('task-decisions').value = (task.decisions || []).join('\n');
      document.getElementById('task-next-actions').value = (task.next_actions || []).join('\n');
      taskEditor.hidden = false;
    }

    function renderTasks(tasks) {
      knownTasks = Array.isArray(tasks) ? tasks : [];
      taskList.replaceChildren();
      document.getElementById('task-count').textContent = `(${knownTasks.length})`;
      if (!knownTasks.length) {
        const empty = document.createElement('p');
        empty.textContent = '저장된 장기 작업이 아직 없어.';
        taskList.appendChild(empty);
        renderTaskEditor();
        return;
      }
      knownTasks.forEach(task => {
        const card = document.createElement('article');
        card.className = `task-item${task.task_id === activeTaskId ? ' active' : ''}`;
        const head = document.createElement('div');
        head.className = 'task-item-head';
        const title = document.createElement('strong');
        title.textContent = task.title;
        const action = document.createElement('button');
        action.type = 'button';
        action.className = 'task-action';
        action.textContent = task.status === 'active'
          ? (task.task_id === activeTaskId ? '선택됨' : '이어가기')
          : '완료됨';
        action.disabled = task.status !== 'active' || task.task_id === activeTaskId;
        action.addEventListener('click', () => {
          rememberActiveTask(task.task_id);
          renderTasks(knownTasks);
          document.getElementById('task-status').textContent =
            '이 작업을 현재 채팅 문맥으로 선택했어.';
        });
        head.append(title, action);
        const meta = document.createElement('div');
        meta.className = 'task-item-meta';
        meta.textContent =
          `${task.status === 'active' ? '진행 중' : '완료'} · revision ${task.revision}`;
        const objective = document.createElement('p');
        objective.textContent = task.objective;
        card.append(head, meta, objective);
        taskList.appendChild(card);
      });
      renderTaskEditor();
    }

    function taskLines(elementId) {
      return document.getElementById(elementId).value
        .split(/\r?\n/)
        .map(item => item.trim())
        .filter(Boolean);
    }

    async function refreshTasks() {
      if (!taskEnabled) return;
      const status = document.getElementById('task-status');
      try {
        const response = await fetch('/api/tasks');
        const data = await response.json();
        if (!response.ok || !data.ok) {
          throw new Error(data.error || '장기 작업을 읽지 못했어.');
        }
        renderTasks(data.tasks);
        status.textContent = '';
      } catch (error) {
        status.textContent = error.message || '장기 작업을 읽지 못했어.';
      }
    }

    document.getElementById('task-create-form').addEventListener('submit', async event => {
      event.preventDefault();
      const title = document.getElementById('task-title').value.trim();
      const objective = document.getElementById('task-objective').value.trim();
      const status = document.getElementById('task-status');
      if (!title || !objective) {
        status.textContent = '작업 이름과 완료 조건을 모두 적어 줘.';
        return;
      }
      status.textContent = '장기 작업을 만드는 중...';
      try {
        const data = await postJson('/api/tasks', {
          title,
          objective,
          session_id: conversationSessionId
        });
        rememberActiveTask(data.task.task_id);
        renderTasks(data.tasks);
        document.getElementById('task-title').value = '';
        document.getElementById('task-objective').value = '';
        status.textContent = '작업을 만들고 현재 채팅에 연결했어.';
      } catch (error) {
        status.textContent = error.message || '장기 작업을 만들지 못했어.';
      }
    });

    document.getElementById('task-save').addEventListener('click', async () => {
      const status = document.getElementById('task-status');
      if (!activeTaskId) return;
      status.textContent = '새 checkpoint를 저장하는 중...';
      try {
        const data = await postJson('/api/tasks/checkpoint', {
          task_id: activeTaskId,
          progress: document.getElementById('task-progress').value.trim(),
          decisions: taskLines('task-decisions'),
          next_actions: taskLines('task-next-actions')
        });
        renderTasks(data.tasks);
        status.textContent = `revision ${data.task.revision}을 저장했어.`;
      } catch (error) {
        status.textContent = error.message || 'checkpoint를 저장하지 못했어.';
      }
    });

    document.getElementById('task-complete').addEventListener('click', async () => {
      const status = document.getElementById('task-status');
      if (!activeTaskId || !window.confirm('이 작업을 완료 상태로 고정할까?')) return;
      status.textContent = '마지막 checkpoint를 저장하는 중...';
      try {
        const data = await postJson('/api/tasks/complete', {
          task_id: activeTaskId,
          progress: document.getElementById('task-progress').value.trim(),
          decisions: taskLines('task-decisions')
        });
        rememberActiveTask('');
        renderTasks(data.tasks);
        status.textContent = `작업을 revision ${data.task.revision}에서 완료했어.`;
      } catch (error) {
        status.textContent = error.message || '작업을 완료하지 못했어.';
      }
    });

    document.getElementById('task-detach').addEventListener('click', () => {
      rememberActiveTask('');
      renderTasks(knownTasks);
      document.getElementById('task-status').textContent =
        '작업 기록은 보존하고 채팅 문맥에서만 해제했어.';
    });

    async function restoreConversation() {
      newChatButton.hidden = !conversationEnabled;
      if (!conversationEnabled || !conversationSessionId) return;
      try {
        const response = await fetch(`/api/conversation/${encodeURIComponent(conversationSessionId)}`);
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || '대화를 불러오지 못했어.');
        renderConversation(data.turns);
      } catch (_error) {
        rememberConversationSession('');
        renderConversation([]);
      }
    }

    newChatButton.addEventListener('click', async () => {
      try {
        const data = await postJson('/api/conversation/new', {create: true});
        rememberConversationSession(data.session_id);
        renderConversation(data.turns);
        chatResult.hidden = true;
        currentSemanticTraceId = '';
        document.getElementById('chat-text').value = '';
        document.getElementById('chat-images').value = '';
        document.getElementById('chat-text').focus();
      } catch (error) {
        showError(error.message || '새 대화를 시작하지 못했어.');
      }
    });

    document.querySelectorAll('.chat-example').forEach(button => {
      button.addEventListener('click', () => {
        document.getElementById('chat-text').value = button.dataset.prompt;
        document.getElementById('chat-text').focus();
      });
    });

    function fileAsDataUrl(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.addEventListener('load', () => resolve({name: file.name, data_url: reader.result}));
        reader.addEventListener('error', () => reject(new Error(`${file.name} 파일을 읽지 못했어.`)));
        reader.readAsDataURL(file);
      });
    }

    function showChatAnswer(answer, recalledMemories = [], recalledEpisodes = [], taskContexts = [], taskSelection = null) {
      const labels = {
        verified: '검증됨',
        conditional: '입력 사실을 전제로 검증됨',
        best_effort: '모델 해석은 미검증',
        unsupported: '추가 정보 필요'
      };
      const sourceLabels = {
        explicit_memory_write: '장기기억 저장됨',
        user_correction_write: '교정 기억 저장됨',
        user_correction_memory: '사용자 교정 기억',
        user_correction_prototype: '반복 교정에서 일반화',
        task_context: '최신 작업에서 재개',
        task_checkpoint_write: '작업 checkpoint 저장됨',
        task_create_write: '장기 작업 생성됨',
        task_complete_write: '장기 작업 완료됨'
      };
      const badge = document.getElementById('chat-status');
      badge.className = `answer-badge ${answer.status}`;
      badge.textContent = sourceLabels[answer.provenance.answer_source]
        || labels[answer.status]
        || answer.status;
      document.getElementById('chat-answer').textContent = answer.answer;
      document.getElementById('chat-proof').textContent = answer.proof || '실행된 proof가 없어.';
      document.getElementById('chat-technical').textContent = JSON.stringify({
        proposals: answer.proposals,
        provenance: answer.provenance,
        metrics: answer.metrics,
        assumptions: answer.assumption_dependencies,
        unresolved: answer.unresolved,
        recalled_memories: recalledMemories,
        recalled_episodes: recalledEpisodes,
        task_contexts: taskContexts,
        task_selection: taskSelection
      }, null, 2);
      currentSemanticTraceId = answer.provenance.semantic_trace_id || '';
      document.getElementById('chat-semantic-review').hidden = !currentSemanticTraceId;
      document.getElementById('chat-review-attest').checked = false;
      document.getElementById('chat-review-status').textContent = '';
      chatResult.hidden = false;
      chatResult.scrollIntoView({behavior: 'smooth', block: 'nearest'});
    }

    chatForm.addEventListener('submit', async event => {
      event.preventDefault();
      const text = document.getElementById('chat-text').value.trim();
      const files = [...document.getElementById('chat-images').files];
      if (!text && !files.length) {
        showError('질문을 적거나 이미지를 한 장 첨부해 줘.');
        return;
      }
      if (files.length > 4) {
        showError('이미지는 최대 네 장까지 첨부할 수 있어.');
        return;
      }
      const button = document.getElementById('chat-submit');
      const oldText = button.textContent;
      button.disabled = true;
      button.textContent = '해석하고 검증하는 중...';
      try {
        const images = await Promise.all(files.map(fileAsDataUrl));
        const data = await postJson('/api/chat', {
          text,
          images,
          resource_tier: document.getElementById('chat-tier').value,
          judge_opt_in: false,
          session_id: conversationSessionId,
          task_id: activeTaskId
        });
        if (conversationEnabled && data.session_id) {
          rememberConversationSession(data.session_id);
          renderConversation(data.turns);
        }
        showChatAnswer(
          data.answer,
          data.recalled_memories || [],
          data.recalled_episodes || [],
          data.task_contexts || [],
          data.task_selection || null
        );
        if (data.created_task) {
          rememberActiveTask(data.created_task.task_id);
          await refreshTasks();
        } else if (data.completed_task) {
          if (activeTaskId === data.completed_task.task_id) rememberActiveTask('');
          await refreshTasks();
        } else if (data.saved_task_checkpoint) {
          rememberActiveTask(data.saved_task_checkpoint.task_id);
          await refreshTasks();
        } else if (data.task_selection && data.task_selection.mode === 'automatic') {
          rememberActiveTask(data.task_selection.task_id);
          await refreshTasks();
        }
        if (data.saved_memory) {
          await refreshMemories();
        }
        document.getElementById('chat-text').value = '';
        document.getElementById('chat-images').value = '';
      } catch (error) {
        chatResult.hidden = false;
        document.getElementById('chat-status').className = 'answer-badge unsupported';
        document.getElementById('chat-status').textContent = '실행 오류';
        document.getElementById('chat-answer').textContent = error.message || '로컬 서버와 연결하지 못했어.';
        document.getElementById('chat-proof').textContent = '실행된 proof가 없어.';
        document.getElementById('chat-technical').textContent = '';
        currentSemanticTraceId = '';
        document.getElementById('chat-semantic-review').hidden = true;
      } finally {
        button.disabled = false;
        button.textContent = oldText;
      }
    });

    document.querySelectorAll('[data-semantic-review]').forEach(button => {
      button.addEventListener('click', async () => {
        const status = document.getElementById('chat-review-status');
        if (!currentSemanticTraceId) {
          status.textContent = '먼저 로컬 모델로 답을 만들어 줘.';
          return;
        }
        if (!document.getElementById('chat-review-attest').checked) {
          status.textContent = '직접 확인했다는 체크가 필요해.';
          return;
        }
        const accepted = button.dataset.semanticReview === 'true';
        status.textContent = '로컬 검토 기록을 저장하는 중...';
        try {
          const data = await postJson('/api/semantic/review', {
            trace_id: currentSemanticTraceId,
            accepted,
            attest_human_review: true,
            note: accepted ? '' : '사용자가 의미 해석이 틀렸다고 표시함'
          });
          status.textContent = accepted
            ? `검증된 의미 trace를 저장했어. 현재 증류 후보 ${data.distillation_records}개야.`
            : '잘못된 해석을 hard negative 후보로 저장했어.';
          document.getElementById('chat-review-attest').checked = false;
          currentSemanticTraceId = '';
          document.getElementById('chat-semantic-review').hidden = true;
          renderSemanticInbox(data);
        } catch (error) {
          status.textContent = error.message || '의미 검토를 저장하지 못했어.';
        }
      });
    });

    function renderSemanticInbox(snapshot) {
      if (!semanticReviewEnabled || !snapshot || snapshot.enabled === false) return;
      semanticReviewPanel.hidden = false;
      const stats = snapshot.stats || {};
      const replay = snapshot.replay || {};
      const consolidation = snapshot.consolidation || {};
      document.getElementById('semantic-review-stats').textContent =
        `전체 ${stats.total || 0}개 · 검토 대기 ${stats.pending || 0}개 · ` +
        `승인 ${stats.accepted || 0}개 · 거절 ${stats.rejected || 0}개 · ` +
        `이번 replay ${replay.selected || 0}개 · ` +
        `통합 prototype ${consolidation.prototype_count || 0}개`;
      const learning = snapshot.student_learning || {};
      const roles = learning.roles || {};
      const train = roles.train || {};
      const sealed = roles.sealed || {};
      const blockers = learning.blockers || [];
      let readinessText = `학생 후보는 아직 학습하지 않아: train ${train.positives || 0}개, sealed ${sealed.positives || 0}개.`;
      if (learning.ready) {
        readinessText = `학생 후보 학습 준비 완료: train ${train.positives || 0}개, sealed ${sealed.positives || 0}개. 실행은 semop-student-cycle run --confirm에서만 시작해.`;
      } else if (blockers.some(item => item.includes('overlap'))) {
        readinessText += ' 학습 역할 사이에 같은 질문이나 semantic target이 겹쳐 봉인 평가를 차단했어.';
      } else {
        readinessText += ' train과 sealed에 사람이 승인한 검증 사례가 각각 필요해.';
      }
      if ((consolidation.prototype_count || 0) > 0) {
        readinessText += ` 서로 다른 승인 표현 3개 이상에서 만든 의미 prototype ${consolidation.prototype_count}개를 새 질문의 미검증 analogy로 사용해.`;
      }
      document.getElementById('semantic-learning-readiness').textContent = readinessText;
      const list = document.getElementById('semantic-review-list');
      list.replaceChildren();
      const items = snapshot.items || [];
      if (!items.length) {
        const empty = document.createElement('p');
        empty.textContent = '지금 검토할 모델 해석이 없어.';
        list.appendChild(empty);
        return;
      }
      items.forEach(item => {
        const card = document.createElement('article');
        card.className = 'queue-item';
        const title = document.createElement('strong');
        title.textContent = item.prompt || '이미지만 첨부한 질문';
        const meta = document.createElement('div');
        meta.className = 'queue-meta';
        meta.textContent = `${item.domain} · ${item.split} · 우선순위 ${item.replay_priority || 0} · proof replay ${item.replay_verified ? '통과' : '미통과'} · ${item.model_id}`;
        const replayReason = document.createElement('p');
        replayReason.textContent = `선정 이유: ${(item.replay_reasons || []).join(' · ')}`;
        const raw = document.createElement('pre');
        raw.textContent = JSON.stringify(item.completion, null, 2);
        card.append(title, meta, replayReason, raw);
        if ((item.media || []).length) {
          const mediaWrap = document.createElement('div');
          mediaWrap.className = 'semantic-media';
          item.media.forEach(media => {
            const image = document.createElement('img');
            image.src = media.url;
            image.alt = `검토 이미지 ${media.image_index + 1}`;
            image.loading = 'lazy';
            mediaWrap.appendChild(image);
          });
          card.appendChild(mediaWrap);
        }
        if (item.semantic_only) {
          const semanticOnly = document.createElement('p');
          semanticOnly.textContent = '이 선택은 사람 기준 비전 의미 label만 만들며 proof 성공이나 텍스트 LoRA positive로 계산되지 않아.';
          card.appendChild(semanticOnly);
        }
        if (item.blocked_reason) {
          const blocked = document.createElement('p');
          blocked.textContent = item.blocked_reason;
          card.appendChild(blocked);
        }
        if (item.can_accept || item.can_reject) {
          const buttons = document.createElement('div');
          buttons.className = 'review-buttons';
          const accept = document.createElement('button');
          accept.type = 'button';
          accept.textContent = 'typed 해석이 맞아';
          accept.disabled = !item.can_accept;
          accept.title = item.can_accept ? '' : '승인에는 typed proof replay 통과가 필요해.';
          accept.addEventListener('click', () => reviewPendingSemantic(item.trace_id, true));
          const reject = document.createElement('button');
          reject.type = 'button';
          reject.textContent = 'typed 해석이 틀려';
          reject.disabled = !item.can_reject;
          reject.addEventListener('click', () => reviewPendingSemantic(item.trace_id, false));
          buttons.append(accept, reject);
          card.appendChild(buttons);
        }
        list.appendChild(card);
      });
    }

    async function refreshSemanticInbox() {
      if (!semanticReviewEnabled) return;
      const status = document.getElementById('semantic-queue-status');
      try {
        const response = await fetch('/api/semantic');
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || '의미 검토함을 읽지 못했어.');
        renderSemanticInbox(data);
        status.textContent = '';
      } catch (error) {
        status.textContent = error.message || '의미 검토함을 읽지 못했어.';
      }
    }

    async function reviewPendingSemantic(traceId, accepted) {
      const status = document.getElementById('semantic-queue-status');
      if (!document.getElementById('semantic-queue-attest').checked) {
        status.textContent = '먼저 질문과 typed 해석을 직접 확인했다는 칸을 체크해 줘.';
        return;
      }
      status.textContent = '검토 내용을 로컬에 기록하는 중...';
      try {
        const data = await postJson('/api/semantic/review', {
          trace_id: traceId,
          accepted,
          attest_human_review: true,
          note: accepted ? '' : '사용자가 persistent 검토함에서 의미 해석을 거절함'
        });
        document.getElementById('semantic-queue-attest').checked = false;
        renderSemanticInbox(data);
        status.textContent = accepted
          ? `승인 trace를 저장했어. train 증류 후보는 현재 ${data.distillation_records}개야.`
          : '틀린 해석을 hard negative 후보로 저장했어.';
      } catch (error) {
        status.textContent = error.message || '의미 검토를 저장하지 못했어.';
      }
    }

    document.getElementById('semantic-review-refresh').addEventListener('click', refreshSemanticInbox);

    function selectDomain(domain) {
      tabs.forEach(tab => tab.setAttribute('aria-selected', String(tab.dataset.domain === domain)));
      panes.forEach(pane => pane.classList.toggle('active', pane.dataset.domain === domain));
      resultBox.hidden = true;
      resultReview.hidden = true;
      lastSubmission = null;
    }

    tabs.forEach(tab => tab.addEventListener('click', () => selectDomain(tab.dataset.domain)));

    const languageExamples = {
      ready: ['배포', '테스트 통과, 관리자 승인', '테스트 통과, 관리자 승인', ''],
      blocked: ['배포', '테스트 통과, 관리자 승인', '테스트 통과', '관리자 승인'],
      missing: ['배포', '테스트 통과, 관리자 승인', '테스트 통과', '']
    };
    const codingExamples = {
      dijkstra: 'Given a weighted graph with N nodes and M nonnegative edges, print the shortest distance from node 1 to every node.',
      prefix: 'Given an array and many range sum queries, print the sum from left to right for every query.'
    };
    document.querySelectorAll('[data-coding-example]').forEach(button => {
      button.addEventListener('click', () => {
        document.getElementById('coding-statement').value = codingExamples[button.dataset.codingExample];
      });
    });
    document.querySelectorAll('[data-language-example]').forEach(button => {
      button.addEventListener('click', () => {
        const values = languageExamples[button.dataset.languageExample];
        ['lang-goal', 'lang-required', 'lang-satisfied', 'lang-blocked'].forEach((id, index) => {
          document.getElementById(id).value = values[index];
        });
      });
    });
    document.querySelectorAll('[data-math-example]').forEach(button => {
      button.addEventListener('click', () => {
        document.getElementById('math-expression').value = button.dataset.mathExample;
      });
    });

    visionPresets.forEach(preset => {
      const option = document.createElement('option');
      option.value = preset.key;
      option.textContent = `${preset.title}: ${preset.question}`;
      visionSelect.appendChild(option);
    });

    function renderVision() {
      const preset = visionPresets.find(item => item.key === visionSelect.value) || visionPresets[0];
      const grid = document.getElementById('pixel-grid');
      grid.replaceChildren();
      grid.style.gridTemplateColumns = `repeat(${preset.pattern[0].length}, 1fr)`;
      preset.pattern.forEach(row => [...row].forEach(color => {
        const pixel = document.createElement('span');
        pixel.className = `pixel ${color === '.' ? 'dot' : color}`;
        grid.appendChild(pixel);
      }));
      document.getElementById('vision-question').textContent = preset.question;
      document.getElementById('vision-description').textContent = preset.description;
    }
    visionSelect.addEventListener('change', renderVision);
    renderVision();

    function payloadFor(domain) {
      if (domain === 'coding') {
        return {statement: document.getElementById('coding-statement').value};
      }
      if (domain === 'language') {
        return {
          goal: document.getElementById('lang-goal').value,
          required: document.getElementById('lang-required').value,
          satisfied: document.getElementById('lang-satisfied').value,
          blocked: document.getElementById('lang-blocked').value
        };
      }
      if (domain === 'math') {
        return {expression: document.getElementById('math-expression').value};
      }
      return {preset: visionSelect.value};
    }

    function showError(message) {
      lastSubmission = null;
      resultReview.hidden = true;
      resultBox.hidden = false;
      resultBox.className = 'result fail';
      document.getElementById('result-status').textContent = '입력 확인 필요';
      document.getElementById('result-title').textContent = '입력을 다시 확인해 줘';
      document.getElementById('result-summary').textContent = message;
      document.getElementById('result-input').replaceChildren();
      document.getElementById('result-trust').textContent = '입력을 고치면 같은 자리에서 다시 검증할 수 있어.';
      document.getElementById('result-proof').textContent = '아직 실행된 증명이 없어.';
      document.getElementById('result-technical').textContent = '';
      document.getElementById('result-artifact').textContent = '';
      document.getElementById('result-artifact-wrap').hidden = true;
      resultBox.scrollIntoView({behavior: 'smooth', block: 'start'});
    }

    function showResult(result) {
      const isVerified = result.success && result.verified;
      const kind = result.conclusion === 'not_ready' ? 'warn' : (isVerified ? 'ok' : 'fail');
      resultBox.hidden = false;
      resultBox.className = `result ${kind}`;
      document.getElementById('result-status').textContent = result.conclusion === 'not_ready'
        ? '검증됨: 진행 조건 미충족'
        : (isVerified ? '검증된 결론' : '미증명');
      document.getElementById('result-title').textContent = result.title;
      document.getElementById('result-summary').textContent = result.summary;
      const list = document.getElementById('result-input');
      list.replaceChildren();
      result.interpreted_input.forEach(line => {
        const item = document.createElement('li');
        item.textContent = line;
        list.appendChild(item);
      });
      document.getElementById('result-trust').textContent = result.trust_notice;
      document.getElementById('result-proof').textContent = result.proof;
      document.getElementById('result-technical').textContent = JSON.stringify(result.technical, null, 2);
      const artifactWrap = document.getElementById('result-artifact-wrap');
      const artifact = result.artifact || '';
      artifactWrap.hidden = !artifact;
      document.getElementById('result-artifact').textContent = artifact;
      resultReview.hidden = !experienceEnabled || !lastSubmission;
      document.getElementById('current-review-attest').checked = false;
      document.getElementById('current-review-status').textContent = '';
      resultBox.scrollIntoView({behavior: 'smooth', block: 'start'});
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.error || '요청을 처리하지 못했어.');
      return data;
    }

    async function reviewCurrent(expectedSolved) {
      const status = document.getElementById('current-review-status');
      if (!lastSubmission) return;
      if (!document.getElementById('current-review-attest').checked) {
        status.textContent = '먼저 입력과 기대 결과를 직접 확인했다는 칸을 체크해 줘.';
        return;
      }
      status.textContent = '검토 내용을 로컬 큐에 기록하는 중...';
      try {
        const data = await postJson('/api/experience/review-example', {
          ...lastSubmission,
          expected_solved: expectedSolved,
          attest_human_review: true
        });
        status.textContent = '사람 검토가 기록됐어. 아직 규칙이나 모델에는 반영되지 않았어.';
        renderExperience(data.queue);
      } catch (error) {
        status.textContent = error.message || '검토 내용을 기록하지 못했어.';
      }
    }

    document.querySelectorAll('[data-review-current]').forEach(button => {
      button.addEventListener('click', () => reviewCurrent(button.dataset.reviewCurrent === 'true'));
    });

    function renderExperience(snapshot) {
      if (!experienceEnabled || !snapshot) return;
      experiencePanel.hidden = false;
      const stats = snapshot.stats || {};
      const learning = snapshot.learning || {};
      const learnButton = document.getElementById('experience-learn');
      learnButton.disabled = !learning.artifact_configured;
      learnButton.title = learning.artifact_configured
        ? `현재 활성 규칙 ${learning.active_rules || 0}개`
        : '이 실행에서는 학습 규칙 승격이 꺼져 있어.';
      document.getElementById('experience-stats').textContent =
        `전체 ${stats.requests || 0}개 · 검토 대기 ${stats.pending || 0}개 · ` +
        `충돌 ${stats.conflicted || 0}개 · 승인 ${stats.approved || 0}개`;
      const list = document.getElementById('experience-list');
      list.replaceChildren();
      const reviewable = (snapshot.items || []).filter(item =>
        item.status === 'pending' || item.status === 'conflicted'
      );
      if (!reviewable.length) {
        const empty = document.createElement('p');
        empty.textContent = '지금 검토할 후보가 없어.';
        list.appendChild(empty);
        return;
      }
      reviewable.forEach(item => {
        const card = document.createElement('article');
        card.className = 'queue-item';
        const title = document.createElement('strong');
        title.textContent = item.summary;
        const meta = document.createElement('div');
        meta.className = 'queue-meta';
        meta.textContent = `${item.domain} · ${item.status} · ${item.triggers.join(', ')}`;
        const raw = document.createElement('pre');
        raw.textContent = JSON.stringify(item.payload, null, 2);
        const buttons = document.createElement('div');
        buttons.className = 'review-buttons';
        [['풀려야 해', true], ['풀리지 않아야 해', false]].forEach(([label, expected]) => {
          const button = document.createElement('button');
          button.type = 'button';
          button.textContent = label;
          button.addEventListener('click', () => reviewQueued(item.request_digest, expected));
          buttons.appendChild(button);
        });
        card.append(title, meta, raw, buttons);
        list.appendChild(card);
      });
    }

    async function refreshExperience() {
      if (!experienceEnabled) return;
      const status = document.getElementById('queue-review-status');
      try {
        const response = await fetch('/api/experience');
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || '큐를 읽지 못했어.');
        renderExperience(data);
        status.textContent = '';
      } catch (error) {
        status.textContent = error.message || '큐를 읽지 못했어.';
      }
    }

    async function reviewQueued(requestDigest, expectedSolved) {
      const status = document.getElementById('queue-review-status');
      if (!document.getElementById('queue-review-attest').checked) {
        status.textContent = '먼저 입력과 기대 결과를 직접 확인했다는 칸을 체크해 줘.';
        return;
      }
      status.textContent = '검토 내용을 로컬 큐에 기록하는 중...';
      try {
        const data = await postJson('/api/experience/review', {
          request_digest: requestDigest,
          expected_solved: expectedSolved,
          include_for_learning: true,
          attest_human_review: true
        });
        document.getElementById('queue-review-attest').checked = false;
        renderExperience(data.queue);
        status.textContent = '사람 검토가 기록됐어. 실제 학습 승격은 별도 검증 뒤에만 가능해.';
      } catch (error) {
        status.textContent = error.message || '검토 내용을 기록하지 못했어.';
      }
    }

    document.getElementById('experience-refresh').addEventListener('click', refreshExperience);
    document.getElementById('experience-learn').addEventListener('click', async () => {
      const status = document.getElementById('experience-learning-status');
      const confirmed = window.confirm(
        '사람이 승인한 사례만 사용해 규칙 후보를 만들고, 별도 검증을 통과할 때만 활성화할까?'
      );
      if (!confirmed) return;
      status.textContent = '승인 사례를 네 분할로 검증하는 중...';
      try {
        const data = await postJson('/api/experience/learn', {confirm_learning: true});
        const learning = data.learning;
        if (learning.promoted) {
          status.textContent = `검증을 통과한 규칙을 활성화했어. 현재 ${learning.active_rules}개야.`;
        } else {
          status.textContent = '새 규칙이 최종 검증을 통과하지 못해 기존 규칙을 그대로 유지했어.';
        }
        renderExperience(learning.queue);
      } catch (error) {
        status.textContent = error.message || '검증 학습을 실행하지 못했어.';
      }
    });
    if (experienceEnabled) {
      experiencePanel.hidden = false;
      refreshExperience();
    }
    if (semanticReviewEnabled) {
      semanticReviewPanel.hidden = false;
      refreshSemanticInbox();
    }
    if (conversationEnabled) {
      memoryManager.hidden = false;
      refreshMemories();
    }
    if (taskEnabled) {
      taskManager.hidden = false;
      refreshTasks();
    }
    restoreConversation();

    panes.forEach(form => form.addEventListener('submit', async event => {
      event.preventDefault();
      const domain = form.dataset.domain;
      const button = form.querySelector('.solve');
      const oldText = button.textContent;
      button.disabled = true;
      button.textContent = '연산자를 조합하는 중...';
      try {
        const values = payloadFor(domain);
        const data = await postJson('/api/solve', {domain, values});
        lastSubmission = {domain, values};
        showResult(data.result);
      } catch (error) {
        showError(error.message || '로컬 서버와 연결하지 못했어.');
      } finally {
        button.disabled = false;
        button.textContent = oldText;
      }
    }));
  </script>
</body>
</html>
'''


if __name__ == "__main__":
    raise SystemExit(main())

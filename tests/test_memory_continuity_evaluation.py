from __future__ import annotations

import json
from pathlib import Path

import pytest

from semop.assistant import SemOpAssistant
from semop.conversation_memory import ConversationStore
from semop.memory_evaluation import (
    MemoryContinuityConfig,
    evaluate_correction_consolidation,
    evaluate_episodic_memory_effect,
    evaluate_long_session_memory_effect,
    evaluate_long_history_episodic_recall,
    evaluate_memory_continuity,
    evaluate_semantic_memory_effect,
    evaluate_semantic_paraphrase_memory,
    evaluate_task_auto_resume,
    evaluate_task_context_effect,
)
from semop.prompt_compiler import PromptCompiler
from semop.task_memory import TaskCheckpointStore
from tools.eval.evaluate_memory_continuity import main as memory_eval_main


class _MemoryAwareBackend:
    model_id = "test/memory-aware"

    def __init__(self) -> None:
        self.calls = 0

    @property
    def loaded(self) -> bool:
        return True

    def generate(self, request, *, operator_hints=(), repair_hint=""):
        self.calls += 1
        if request.task_contexts:
            answer = request.task_contexts[0].next_actions[0]
        elif request.recalled_memories:
            answer = request.recalled_memories[0].content
        elif request.recalled_episodes:
            answer = request.recalled_episodes[0].assistant_content
        else:
            answer = "저장된 문맥을 찾지 못했어."
        return {
            "domain": "unsupported",
            "confidence": 0.8,
            "operator_program": ["EXPLAIN"],
            "payload": {},
            "answer": answer,
        }


class _FacetRanker:
    def rank_texts(self, query, passages, *, limit=8):
        return ((0, 0.86),) if passages else ()


def test_memory_continuity_gate_survives_restart_and_long_task(
    tmp_path: Path,
) -> None:
    report = evaluate_memory_continuity(
        tmp_path / "continuity.db",
        config=MemoryContinuityConfig(
            exchanges=12,
            working_messages=6,
            task_updates=4,
            distractor_memories=8,
        ),
    )

    assert report.passed is True
    assert report.stored_turns == 24
    assert report.restored_turns == 24
    assert len(report.restored_working_messages) == 6
    assert report.restored_working_messages == report.expected_working_messages
    assert report.recall_at_1 == 1.0
    assert report.cross_session_episode_recalled is True
    assert report.cross_session_episode_score == 1.0
    assert report.same_session_episode_excluded is True
    assert report.same_session_archive_tested is True
    assert report.same_session_archived_episode_recalled is True
    assert report.same_session_archived_episode_score == 1.0
    assert report.recent_working_episode_excluded is True
    assert report.unrelated_false_recall_rate == 0.0
    assert report.expected_task_revision == 5
    assert report.restored_task_revision == 5
    assert report.task_history_revisions == (1, 2, 3, 4, 5, 6)
    assert report.completion_immutable is True
    assert report.database_bytes > 0
    assert report.peak_python_bytes > 0
    assert report.to_dict()["passed"] is True


def test_memory_continuity_gate_never_overwrites_an_existing_database(
    tmp_path: Path,
) -> None:
    path = tmp_path / "existing.db"
    path.write_bytes(b"user data")

    with pytest.raises(ValueError, match="must not already exist"):
        evaluate_memory_continuity(path)

    assert path.read_bytes() == b"user data"


def test_long_history_gate_recalls_beyond_400_exchange_boundary(
    tmp_path: Path,
) -> None:
    report = evaluate_long_history_episodic_recall(
        tmp_path / "long-history.db",
        exchanges=421,
    )

    assert report.passed is True
    assert report.indexed_episodes == 421
    assert report.cross_session_recalled is True
    assert report.same_session_recalled is True
    assert report.cross_session_answer == report.target_answer
    assert report.same_session_answer == report.target_answer
    assert report.database_bytes > 0


def test_semantic_paraphrase_memory_gate_measures_model_free_facet_gain(
    tmp_path: Path,
) -> None:
    report = evaluate_semantic_paraphrase_memory(
        ConversationStore(tmp_path / "semantic-paraphrase.db"),
        _FacetRanker(),
    )

    assert report.passed is True
    assert report.measured_gain is True
    assert report.lexical_recall_at_1 == 0.0
    assert report.hybrid_recall_at_1 == 1.0
    assert report.warm_correct == 5
    assert report.unrelated_false_recall_rate == 0.0
    assert report.hybrid_retrievals == ("semantic_e5",) * 5


def test_semantic_memory_ab_reports_retriever_gain(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "semantic-memory.db")
    assistant = SemOpAssistant(
        compiler=PromptCompiler(backend=_MemoryAwareBackend())
    )

    report = evaluate_semantic_memory_effect(assistant, store)

    assert report.passed is True
    assert report.measured_gain is True
    assert report.without_memory_contains_marker is False
    assert report.with_memory_contains_marker is True
    assert report.memory_context_reported is True
    assert report.recalled_memory_ids
    assert report.recall_score >= 0.18
    assert report.with_memory_answer == "별빛-27"
    assert report.with_generated_tokens == 0
    assert report.without_generated_tokens == 0
    assert report.without_memory_status == "unsupported"
    assert report.with_memory_status == "best_effort"


def test_episodic_memory_ab_reports_cross_session_retriever_gain(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "episodic-memory.db")
    assistant = SemOpAssistant(
        compiler=PromptCompiler(backend=_MemoryAwareBackend())
    )

    report = evaluate_episodic_memory_effect(assistant, store)

    assert report.passed is True
    assert report.measured_gain is True
    assert report.without_episode_contains_marker is False
    assert report.with_episode_contains_marker is True
    assert report.episode_context_reported is True
    assert report.recalled_episode_ids
    assert report.recalled_episode_scopes == ("cross_session",)
    assert report.recall_score >= 0.30
    assert report.with_episode_answer == "해마-7319"
    assert report.with_generated_tokens == 0
    assert report.without_generated_tokens == 0
    assert report.without_episode_status == "unsupported"
    assert report.with_episode_status == "best_effort"


def test_long_session_memory_ab_reports_archived_episode_gain(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "long-session-memory.db")
    assistant = SemOpAssistant(
        compiler=PromptCompiler(backend=_MemoryAwareBackend())
    )

    report = evaluate_long_session_memory_effect(assistant, store)

    assert report.passed is True
    assert report.measured_gain is True
    assert report.without_episode_contains_marker is False
    assert report.with_episode_contains_marker is True
    assert report.episode_context_reported is True
    assert report.recalled_episode_ids
    assert report.recalled_episode_scopes == ("same_session_archive",)
    assert report.recall_score >= 0.30
    assert report.with_episode_answer == "등대-4821"
    assert report.without_generated_tokens == 0
    assert report.with_generated_tokens == 0
    assert report.without_episode_status == "unsupported"


def test_correction_consolidation_ab_generalizes_only_after_three_examples(
    tmp_path: Path,
) -> None:
    report = evaluate_correction_consolidation(
        ConversationStore(tmp_path / "correction-consolidation.db")
    )

    assert report.passed is True
    assert report.measured_gain is True
    assert report.examples_before == 2
    assert report.examples_after == 3
    assert report.matched_before_consolidation is False
    assert report.matched_after_consolidation is True
    assert report.prototype_support == 3
    assert report.recalled_answer == "별빛-27이야."
    assert report.restart_persistent is True
    assert report.score >= 0.35


def test_task_context_ab_uses_latest_revision_and_ignores_stale_action(
    tmp_path: Path,
) -> None:
    backend = _MemoryAwareBackend()
    assistant = SemOpAssistant(
        compiler=PromptCompiler(backend=backend)
    )

    report = evaluate_task_context_effect(
        assistant,
        TaskCheckpointStore(tmp_path / "task-context.db"),
    )

    assert report.passed is True
    assert report.measured_gain is True
    assert report.task_revision == 3
    assert report.without_context_contains_marker is False
    assert report.with_context_contains_marker is True
    assert report.with_context_contains_stale_marker is False
    assert report.task_context_reported is True
    assert report.with_context_answer == "검증-단계-42"
    assert report.summary_contains_revision is True
    assert report.summary_contains_progress is True
    assert report.summary_contains_decision is True
    assert report.summary_contains_next_action is True
    assert report.summary_context_reported is True
    assert "revision 3" in report.summary_answer
    assert report.without_context_status == "unsupported"
    assert report.with_context_status == "best_effort"
    assert backend.calls == 0


def test_task_auto_resume_rejects_ambiguity_and_selects_named_latest_task(
    tmp_path: Path,
) -> None:
    report = evaluate_task_auto_resume(
        TaskCheckpointStore(tmp_path / "task-auto-resume.db")
    )

    assert report.passed is True
    assert report.ambiguous_query_rejected is True
    assert report.named_query_selected is True
    assert report.selected_task_id == report.target_task_id
    assert report.selected_revision == report.target_revision
    assert report.selected_action == "최신 행동-42"
    assert report.selected_contains_stale_action is False
    assert report.reason == "lexical_match"
    assert report.score >= 0.18
    assert report.restart_persistent is True


def test_memory_continuity_cli_writes_machine_readable_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"

    exit_code = memory_eval_main(
        [
            "--output",
            str(output),
            "--exchanges",
            "5",
            "--working-messages",
            "4",
            "--task-updates",
            "3",
            "--distractor-memories",
            "4",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["evaluation"] == "semop_memory_continuity_v1"
    assert payload["passed"] is True
    assert payload["continuity"]["recall_at_1"] == 1.0
    assert payload["correction_consolidation"]["passed"] is True
    assert payload["correction_consolidation"]["measured_gain"] is True
    assert payload["task_auto_resume"]["passed"] is True
    assert payload["task_auto_resume"]["ambiguous_query_rejected"] is True
    assert payload["long_history"] is None
    assert payload["semantic_paraphrase_ab"] is None
    assert payload["semantic_ab"] is None
    assert payload["episodic_ab"] is None
    assert payload["long_session_ab"] is None
    assert payload["task_ab"] is None


def test_memory_continuity_cli_cleans_up_long_history_database(
    tmp_path: Path,
) -> None:
    output = tmp_path / "long-history-report.json"

    exit_code = memory_eval_main(
        [
            "--output",
            str(output),
            "--exchanges",
            "2",
            "--task-updates",
            "1",
            "--distractor-memories",
            "0",
            "--long-history-exchanges",
            "401",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["long_history"]["passed"] is True
    assert payload["long_history"]["indexed_episodes"] == 401

from __future__ import annotations

from pathlib import Path

import pytest

from semop import (
    AnswerEnvelope,
    AnswerStatus,
    CognitiveOperator,
    PromptRequest,
    SemanticProposal,
    SemanticReplayPlanner,
)
from semop.kernel.semantic_codec import canonical_json
from semop.semantic_experience import (
    SemanticTraceRecord,
    SemanticTraceSplit,
    SemanticTraceStore,
    semantic_request_split,
)


def _capture(
    store: SemanticTraceStore,
    prompt: str,
    *,
    replay_verified: bool,
    images: tuple[bytes, ...] = (),
) -> SemanticTraceRecord:
    request = PromptRequest(prompt, images=images, resource_tier="balanced")
    proposal = SemanticProposal(
        domain="math" if not images else "vision",
        operator_program=(CognitiveOperator.INFER, CognitiveOperator.VERIFY),
        confidence=0.76,
        producer_id="test-semantic-replay",
        candidate_answer="후보 답",
        model_id="test/model",
        resource_tier="balanced",
    )
    answer = AnswerEnvelope(
        answer="후보 답",
        status=AnswerStatus.BEST_EFFORT,
        proposals=(proposal,),
        provenance={"logical_replay_verified": replay_verified},
    )
    record = store.capture(request, answer)
    assert record is not None
    return record


def _prompt_in_split(base: str, split: SemanticTraceSplit) -> str:
    for index in range(2_000):
        prompt = f"{base} 사례 {index}"
        request_json = canonical_json(PromptRequest(prompt).to_dict())
        if semantic_request_split(request_json) is split:
            return prompt
    raise AssertionError("could not find deterministic semantic split prompt")


def test_replay_prioritizes_actionable_positive_and_never_mutates_journal(
    tmp_path: Path,
) -> None:
    store = SemanticTraceStore(tmp_path / "experience.db")
    positive = _capture(store, "사과 열두 개에서 다섯 개 빼기", replay_verified=True)
    correction = _capture(store, "모호한 계산 문장", replay_verified=False)
    blocked_image = _capture(
        store,
        "원본이 없는 이미지",
        replay_verified=False,
        images=(b"not-an-image",),
    )
    before = store.stats()

    batch = SemanticReplayPlanner(store).select(limit=3)

    assert [item.trace.trace_id for item in batch.candidates] == [
        positive.trace_id,
        correction.trace_id,
        blocked_image.trace_id,
    ]
    assert batch.candidates[0].can_accept is True
    assert batch.candidates[1].can_accept is False
    assert batch.candidates[1].can_reject is True
    assert batch.candidates[2].priority == 0
    assert batch.candidates[2].can_reject is False
    assert "픽셀 증거" in batch.candidates[2].blocked_reason
    assert batch.pending_total == 3
    assert store.stats() == before


def test_replay_prefers_novel_structure_over_near_duplicate(tmp_path: Path) -> None:
    store = SemanticTraceStore(tmp_path / "experience.db")
    reviewed = _capture(
        store,
        "사과 12개에서 5개를 빼는 계산",
        replay_verified=True,
    )
    store.review(
        reviewed.trace_id,
        accepted=True,
        reviewer="human:test",
        attest_human_review=True,
    )
    near_prompt = _prompt_in_split(
        "사과 12개에서 5개를 빼는 계산",
        reviewed.split,
    )
    novel_prompt = _prompt_in_split(
        "기차 속력과 이동 시간을 비교하는 문제",
        reviewed.split,
    )
    near = _capture(store, near_prompt, replay_verified=True)
    novel = _capture(store, novel_prompt, replay_verified=True)

    batch = SemanticReplayPlanner(store).select(limit=2)

    assert batch.candidates[0].trace.trace_id == novel.trace_id
    by_id = {item.trace.trace_id: item for item in batch.candidates}
    assert by_id[novel.trace_id].novelty > by_id[near.trace_id].novelty
    assert any("다른 표현" in reason for reason in by_id[novel.trace_id].reasons)
    assert store.latest_reviews()[reviewed.trace_id].accepted is True


def test_replay_budget_validation(tmp_path: Path) -> None:
    planner = SemanticReplayPlanner(SemanticTraceStore(tmp_path / "experience.db"))

    with pytest.raises(ValueError, match="between 1 and 20"):
        planner.select(limit=0)
    with pytest.raises(ValueError, match="between limit and 100"):
        planner.select(limit=10, pool_limit=5)


def test_semantic_replay_releases_windows_sqlite_handle(tmp_path: Path) -> None:
    path = tmp_path / "experience.db"
    planner = SemanticReplayPlanner(SemanticTraceStore(path))

    assert planner.select().candidates == ()
    path.unlink()

    assert path.exists() is False

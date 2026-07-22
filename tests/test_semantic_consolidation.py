from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence
from urllib.request import urlopen

from semop import PromptRequest, ResourceTier
from semop.assistant import SemOpAssistant
from semop.beginner_web import create_server
from semop.prompt_compiler import PromptCompiler
from semop.semantic_experience import SemanticTraceStore
from semop.semantic_memory import ReviewedSemanticMemory


_SUBTRACTION = {
    "domain": "math",
    "confidence": 0.91,
    "operator_program": ["DECOMPOSE", "INFER", "VERIFY", "EXPLAIN"],
    "payload": {"expression": "12 - 5 = 7"},
    "answer": "7",
}


class _StaticBackend:
    model_id = "test/consolidation"
    loaded = True

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = dict(payload)
        self.hints: tuple[str, ...] = ()

    def generate(
        self,
        request: PromptRequest,
        *,
        operator_hints: Sequence[str] = (),
        repair_hint: str = "",
    ) -> Mapping[str, Any]:
        self.hints = tuple(operator_hints)
        return self.payload


def _review_trace(
    store: SemanticTraceStore,
    prompt: str,
    *,
    payload: Mapping[str, Any] = _SUBTRACTION,
    minute: int,
) -> str:
    answer = SemOpAssistant(
        compiler=PromptCompiler(backend=_StaticBackend(payload)),
        semantic_trace_store=store,
    ).solve(PromptRequest(prompt, resource_tier=ResourceTier.BALANCED))
    trace_id = str(answer.provenance["semantic_trace_id"])
    store.review(
        trace_id,
        accepted=True,
        reviewer="human:test",
        attest_human_review=True,
        reviewed_at=f"2026-07-22T00:{minute:02d}:00+00:00",
    )
    return trace_id


def _three_reviewed_paraphrases(store: SemanticTraceStore) -> tuple[str, ...]:
    prompts = (
        "사과 12개에서 5개를 빼 줘",
        "사과 열두 개 중 다섯 개를 먹으면 몇 개 남아",
        "12개의 사과에서 5개를 제외한 개수",
    )
    for minute, prompt in enumerate(prompts):
        _review_trace(store, prompt, minute=minute)
    return prompts


def test_three_reviewed_paraphrases_form_an_untrusted_semantic_prototype(
    tmp_path: Path,
) -> None:
    store = SemanticTraceStore(tmp_path / "experience.db")
    _three_reviewed_paraphrases(store)
    memory = ReviewedSemanticMemory(store)
    query = PromptRequest(
        "사과 12개 중 5개를 빼면 몇 개야?",
        resource_tier=ResourceTier.BALANCED,
    )

    prototypes = memory.prototypes()
    matches = memory.search(query)
    hints = memory.retrieve(query)

    assert len(prototypes) == 1
    assert prototypes[0].support == 3
    assert len(matches) == 1
    assert matches[0].source == "prototype"
    assert matches[0].support == 3
    assert "HUMAN_ACCEPTED_SEMANTIC_PROTOTYPE" in hints[0]
    assert "support=3" in hints[0]


def test_semantic_prototype_guides_a_fresh_proposal_without_new_authority(
    tmp_path: Path,
) -> None:
    store = SemanticTraceStore(tmp_path / "experience.db")
    _three_reviewed_paraphrases(store)
    backend = _StaticBackend(_SUBTRACTION)
    request = PromptRequest(
        "사과 12개 중 5개를 빼면 몇 개야?",
        resource_tier=ResourceTier.BALANCED,
    )

    compilation = PromptCompiler(
        backend=backend,
        memory_retriever=ReviewedSemanticMemory(store),
    ).compile(request)

    assert any("SEMANTIC_PROTOTYPE" in hint for hint in backend.hints)
    assert compilation.used_model is True
    assert compilation.proposals[0].deterministic is False
    assert compilation.proposals[0].disposition == "proposed"


def test_semantic_prototype_requires_three_distinct_matching_targets(
    tmp_path: Path,
) -> None:
    store = SemanticTraceStore(tmp_path / "experience.db")
    _review_trace(store, "사과 12개에서 5개 빼기", minute=0)
    _review_trace(store, "열두 사과 중 다섯 사과 제외", minute=1)
    addition = {
        **_SUBTRACTION,
        "payload": {"expression": "12 + 5 = 17"},
        "answer": "17",
    }
    _review_trace(
        store,
        "사과 12개에 5개 더하기",
        payload=addition,
        minute=2,
    )

    memory = ReviewedSemanticMemory(store)

    assert memory.prototypes() == ()
    assert memory.retrieve(PromptRequest("사과 12개 중 5개 빼면?")) == ()


def test_explicit_math_operator_conflict_blocks_near_miss_prototype(
    tmp_path: Path,
) -> None:
    store = SemanticTraceStore(tmp_path / "experience.db")
    _three_reviewed_paraphrases(store)
    memory = ReviewedSemanticMemory(store)

    assert memory.retrieve(
        PromptRequest("사과 12개에서 5개를 더하면 몇 개야?")
    ) == ()
    assert memory.retrieve(
        PromptRequest("사과 12개 중 5개를 빼면 몇 개야?")
    )


def test_latest_human_rejection_removes_prototype_support(tmp_path: Path) -> None:
    store = SemanticTraceStore(tmp_path / "experience.db")
    _three_reviewed_paraphrases(store)
    memory = ReviewedSemanticMemory(store)
    prototype = memory.prototypes()[0]

    store.review(
        prototype.trace_ids[0],
        accepted=False,
        reviewer="human:test",
        attest_human_review=True,
        note="이 표현의 이전 typed 해석을 교정함",
        reviewed_at="2026-07-22T00:10:00+00:00",
    )

    assert memory.prototypes() == ()
    assert memory.retrieve(PromptRequest("사과 12개 중 5개를 빼면?")) == ()


def test_beginner_api_exposes_consolidated_prototype_count(tmp_path: Path) -> None:
    store = SemanticTraceStore(tmp_path / "experience.db")
    _three_reviewed_paraphrases(store)
    server = create_server(0, semantic_trace_store=store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(
            f"http://127.0.0.1:{server.server_address[1]}/api/semantic",
            timeout=10,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert payload["consolidation"]["prototype_count"] == 1
    assert payload["consolidation"]["prototypes"][0]["support"] == 3
    assert payload["items"] == []

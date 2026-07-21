from __future__ import annotations

from semop.kernel import (
    CodingInputAdapter,
    CodingProblem,
    DomainKind,
    OperatorKernel,
    TypedDomainRequest,
    UnifiedTypedReasoner,
    decode_semantic_request,
    encode_semantic_request,
    evaluate_operator_core_gate,
    semantic_request_digest,
)


DIJKSTRA_PROBLEM = (
    "Given a weighted graph with N nodes and M nonnegative edges, print the "
    "shortest distance from node 1 to every node."
)


def test_coding_candidate_is_compiled_tested_and_replay_verified() -> None:
    result = UnifiedTypedReasoner().run(
        TypedDomainRequest(DomainKind.CODING, DIJKSTRA_PROBLEM)
    )

    assert result.success
    assert result.verified
    assert result.instance is not None
    assert result.typed_result is not None
    assert result.instance.metadata["category"] == "dijkstra_shortest_path"
    assert result.instance.metadata["compile_ok"] is True
    assert result.instance.metadata["validation"]["overall_ok"] is True
    assert "priority_queue" in result.instance.metadata["code"]
    assert result.instance.metadata["online_judge_acceptance_claimed"] is False

    replay = OperatorKernel(result.instance.registry).replay(
        result.instance.state,
        result.instance.goals,
        result.typed_result.proof,
    )
    assert replay.verified
    assert not result.typed_result.conditional
    assert result.typed_result.dependencies.evidence_complete


def test_coding_without_registered_execution_validator_fails_closed() -> None:
    statement = (
        "Given line segments in the plane, determine whether any two segments "
        "intersect and print their Euclidean distance."
    )

    result = UnifiedTypedReasoner().run(
        TypedDomainRequest(DomainKind.CODING, statement)
    )

    assert not result.success
    assert not result.verified
    assert result.instance is not None
    assert result.instance.metadata["category"] == "computational_geometry_analysis"
    assert result.instance.metadata["compile_ok"] is True
    assert result.instance.metadata["validation"]["checked"] is False
    assert not any(
        fact.atom.predicate.name == "TESTS_PASSED"
        for fact in result.instance.state.facts
    )


def test_coding_semantic_codec_roundtrip_and_digest_are_stable() -> None:
    request = TypedDomainRequest(
        DomainKind.CODING,
        CodingProblem(DIJKSTRA_PROBLEM, "c++"),
    )

    encoded = encode_semantic_request(request)
    decoded = decode_semantic_request(
        DomainKind.CODING,
        encoded["payload"],
    )

    assert encoded == {
        "domain": "coding",
        "payload": {"statement": DIJKSTRA_PROBLEM, "language": "cpp"},
    }
    assert decoded.payload == CodingProblem(DIJKSTRA_PROBLEM)
    assert semantic_request_digest(request) == semantic_request_digest(decoded)


def test_default_coding_knowledge_is_found_outside_repository_cwd(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    adapter = CodingInputAdapter()

    assert any(
        item.id == "dijkstra_shortest_path"
        for item in adapter.reasoner.knowledge.algorithms
    )


def test_operator_core_gate_covers_all_four_user_facing_domains() -> None:
    report = evaluate_operator_core_gate()

    assert report.passed
    assert tuple(item.domain for item in report.results) == (
        DomainKind.CODING,
        DomainKind.LANGUAGE,
        DomainKind.MATH,
        DomainKind.VISION,
    )
    assert all(item.proof_replay_verified for item in report.results)
    assert all(item.negative_fail_closed for item in report.results)
    assert "not online-judge acceptance" in report.to_dict()["claim_scope"]

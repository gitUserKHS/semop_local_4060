from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from semop.beginner_learning import (
    load_beginner_rule_library,
    run_beginner_reviewed_learning,
)
from semop.kernel import (
    AssertionStatus,
    DomainInstance,
    DomainKind,
    EvidenceStatus,
    ExperiencePartitionConfig,
    ExperienceSplitRole,
    Fact,
    FactStatus,
    Goal,
    KernelRegistry,
    LanguageTextProblem,
    TypedDomainRequest,
    TypedExperienceCollector,
    TypedExperienceStore,
    UnifiedTypedReasoner,
    WorldState,
    semantic_request_digest,
)


def test_reviewed_examples_promote_persist_and_reload_a_rule() -> None:
    partition = ExperiencePartitionConfig(seed="beginner-end-to-end-learning")
    reasoner = UnifiedTypedReasoner({DomainKind.LANGUAGE: _ChainLanguageAdapter()})
    specifications = [
        (ExperienceSplitRole.TRAIN, True, f"train-{index}")
        for index in range(3)
    ]
    specifications.extend(
        (role, expected, f"{role.value}-{'positive' if expected else 'negative'}")
        for role in (
            ExperienceSplitRole.VALIDATION,
            ExperienceSplitRole.CANDIDATE_HELDOUT,
            ExperienceSplitRole.JOINT_HELDOUT,
        )
        for expected in (True, False)
    )

    with TemporaryDirectory() as directory:
        root = Path(directory)
        store = TypedExperienceStore(root / "experience.db", partition=partition)
        collector = TypedExperienceCollector(store, reasoner=reasoner)
        for role, expected, token in specifications:
            request = _request_for_role(
                partition,
                role,
                expected_solved=expected,
                seed=token,
            )
            run = collector.run(
                request,
                proposed_expected_solved=expected,
                proposal_authority="programmatic",
                rationale="The reviewed curriculum expects this exact chain outcome.",
                event_id=f"beginner-{token}",
            )
            store.review(
                run.request_digest,
                expected_solved=expected,
                phenomenon="start_implies_middle",
                rationale="A person verified the exact premise and target symbols.",
                reviewer="human:beginner-learning-test",
                decision="approved",
                reviewed_at="2026-07-20T12:00:00Z",
            )

        artifact = root / "active-rules.json"
        learning = run_beginner_reviewed_learning(
            store,
            reasoner,
            artifact,
            required_domains=(DomainKind.LANGUAGE,),
        )
        loaded = load_beginner_rule_library(artifact)

        assert learning.promoted
        assert learning.checkpoint is not None
        assert learning.checkpoint.rule_count == 1
        assert artifact.is_file()
        assert artifact.with_suffix(".checkpoint.json").is_file()
        assert loaded.checkpoint is not None
        assert loaded.library.artifact_sha256 == learning.checkpoint.library_sha256

        unseen = _chain_request(source="unseen", target="unseen")
        baseline = reasoner.run(unseen)
        transferred = UnifiedTypedReasoner(
            {DomainKind.LANGUAGE: _ChainLanguageAdapter()},
            augmenters=(loaded.library,),
        ).run(unseen)

        assert not baseline.success
        assert transferred.success and transferred.verified

        artifact.write_bytes(artifact.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="hash mismatch"):
            load_beginner_rule_library(artifact)


def test_empty_beginner_rule_store_loads_an_empty_library() -> None:
    with TemporaryDirectory() as directory:
        loaded = load_beginner_rule_library(Path(directory) / "missing.json")

    assert loaded.checkpoint is None
    assert loaded.library.records == ()


def _request_for_role(
    partition: ExperiencePartitionConfig,
    role: ExperienceSplitRole,
    *,
    expected_solved: bool,
    seed: str,
) -> TypedDomainRequest:
    for index in range(10_000):
        source = f"{seed}-source-{index}"
        target = source if expected_solved else f"{seed}-other-{index}"
        request = _chain_request(source=source, target=target)
        if partition.role_for(semantic_request_digest(request)) is role:
            return request
    raise AssertionError(f"could not generate a request for {role.value}")


def _chain_request(*, source: str, target: str) -> TypedDomainRequest:
    return TypedDomainRequest(
        DomainKind.LANGUAGE,
        f"premise=START;goal=MIDDLE;source={source};target={target}",
    )


class _ChainLanguageAdapter:
    def adapt(self, payload: str | LanguageTextProblem) -> DomainInstance:
        text = payload.text if isinstance(payload, LanguageTextProblem) else payload
        fields = dict(field.split("=", 1) for field in text.split(";"))
        registry = KernelRegistry()
        entity = registry.types.register("Entity")
        for predicate in ("START", "MIDDLE"):
            registry.register_predicate(predicate, (entity,))
        source = registry.symbol(fields["source"], entity)
        target = registry.symbol(fields["target"], entity)
        fact = Fact(
            registry.atom(fields["premise"], source),
            FactStatus.OBSERVED,
            source="beginner-chain-adapter",
            assertion_status=AssertionStatus.EXPLICIT,
            evidence_status=EvidenceStatus.ADAPTER_VERIFIED,
        )
        return DomainInstance(
            registry=registry,
            state=WorldState((fact,)),
            goals=(Goal(registry.atom(fields["goal"], target)),),
            domain="language",
            metadata={"adapter": "beginner-chain-test"},
        )

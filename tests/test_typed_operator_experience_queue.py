from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (  # noqa: E402
    ExperienceObservation,
    ExperiencePartitionConfig,
    ExperienceProposalAuthority,
    ExperienceQueueStatus,
    ExperienceReviewCorpus,
    ExperienceStoreIntegrityError,
    ExperienceTrigger,
    LanguageTextProblem,
    RasterImage,
    RasterVisionConfig,
    RasterVisionProblem,
    SemanticLabelAuthority,
    TypedDomainRequest,
    TypedExperienceStore,
    VisionCountGoal,
    decode_semantic_request,
    encode_semantic_request,
    observation_from_request,
    semantic_request_digest,
)


class SemanticRequestCodecTests(unittest.TestCase):
    def test_language_math_and_vision_round_trip_canonically(self) -> None:
        requests = (
            TypedDomainRequest(
                "language",
                LanguageTextProblem(
                    "Goal: deploy; Requires: review; Satisfied: review",
                    "Controlled SOP excerpt",
                    False,
                ),
            ),
            TypedDomainRequest("math", "3 * (4 + 1) == 15"),
            TypedDomainRequest(
                "vision",
                RasterVisionProblem(
                    image=RasterImage.from_rows(
                        (
                            ((255, 255, 255), (255, 255, 255)),
                            ((255, 0, 0), (255, 255, 255)),
                        ),
                        background=(255, 255, 255),
                        source="codec-test",
                    ),
                    goals=(VisionCountGoal("red", 1, "one red object"),),
                    query="How many red objects?",
                    config=RasterVisionConfig(minimum_component_area=1),
                    count_selectors=("red",),
                ),
            ),
        )

        for request in requests:
            encoded = encode_semantic_request(request)
            restored = decode_semantic_request(
                encoded["domain"],
                encoded["payload"],
            )
            self.assertEqual(
                encode_semantic_request(restored),
                encoded,
            )
            self.assertEqual(
                semantic_request_digest(restored),
                semantic_request_digest(request),
            )

    def test_codec_rejects_legacy_and_composed_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy"):
            encode_semantic_request(
                TypedDomainRequest("math", "2 + 2", "legacy")
            )
        with self.assertRaisesRegex(ValueError, "does not support composed"):
            encode_semantic_request(
                TypedDomainRequest("composed", object(), "shadow")
            )

        with self.assertRaisesRegex(TypeError, "language semantic payload"):
            encode_semantic_request(TypedDomainRequest("language", object()))
        with self.assertRaisesRegex(TypeError, "query must be a string"):
            decode_semantic_request(
                "vision",
                {
                    "rows": [["white", "white"], ["white", "red"]],
                    "goals": [
                        {"kind": "count", "selector": "red", "expected": 1}
                    ],
                    "query": 123,
                },
            )


class TypedExperienceStoreTests(unittest.TestCase):
    def test_append_idempotency_occurrences_and_queue_stats(self) -> None:
        with TemporaryDirectory() as directory:
            store = TypedExperienceStore(Path(directory) / "experience.db")
            first = _observation("event-1", _language_request("deploy"), True)
            inserted = store.add_observation(first)
            repeated = store.add_observation(first)
            second = replace(first, event_id="event-2")
            store.add_observation(second)

            item = store.get_item(first.request_digest)
            stats = store.stats()

        self.assertTrue(inserted.inserted)
        self.assertFalse(repeated.inserted)
        self.assertEqual(item.occurrences, 2)
        self.assertEqual(item.status, ExperienceQueueStatus.PENDING)
        self.assertEqual(item.proposed_positive, 2)
        self.assertEqual(stats.requests, 1)
        self.assertEqual(stats.observations, 2)
        self.assertEqual(stats.pending, 1)

    def test_reused_event_id_with_different_content_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            store = TypedExperienceStore(Path(directory) / "experience.db")
            first = _observation("event-same", _language_request("deploy"), True)
            store.add_observation(first)

            with self.assertRaisesRegex(
                ExperienceStoreIntegrityError,
                "reused with different content",
            ):
                store.add_observation(
                    replace(first, proposed_expected_solved=False)
                )

    def test_conflicting_proposals_require_separate_human_review(self) -> None:
        request = _language_request("publish")
        with TemporaryDirectory() as directory:
            store = TypedExperienceStore(Path(directory) / "experience.db")
            positive = _observation("event-positive", request, True)
            negative = _observation(
                "event-negative",
                request,
                False,
                authority=ExperienceProposalAuthority.FRONTIER_JUDGE,
            )
            store.add_observation(positive)
            store.add_observation(negative)

            conflicted = store.get_item(positive.request_digest)
            review = store.review(
                positive.request_digest,
                expected_solved=True,
                phenomenon="missing_requirement_rule",
                rationale="The reviewed requirements imply the goal is reachable.",
                reviewer="human:test-reviewer",
                decision="approved",
                reviewed_at="2026-07-18T12:00:00Z",
            )
            approved = store.get_item(positive.request_digest)
            corpus = store.export_reviewed()
            example = corpus.raw_examples()[0]

        self.assertEqual(conflicted.status, ExperienceQueueStatus.CONFLICTED)
        self.assertTrue(conflicted.conflicted)
        self.assertTrue(review.approved)
        self.assertEqual(approved.status, ExperienceQueueStatus.APPROVED)
        self.assertEqual(len(corpus.records), 1)
        self.assertEqual(
            example.label_authority,
            SemanticLabelAuthority.HUMAN_REVIEWED,
        )
        self.assertTrue(example.label_evidence.complete)
        self.assertEqual(
            review.split_role,
            ExperiencePartitionConfig().role_for(positive.request_digest),
        )

    def test_latest_rejected_review_removes_case_from_export(self) -> None:
        request = TypedDomainRequest("math", "2 + 2 == 5")
        with TemporaryDirectory() as directory:
            store = TypedExperienceStore(Path(directory) / "experience.db")
            observation = _observation("math-event", request, False)
            store.add_observation(observation)
            store.review(
                observation.request_digest,
                expected_solved=False,
                phenomenon="false_equality",
                rationale="Two plus two is not five.",
                reviewer="human:first",
                decision="approved",
                reviewed_at="2026-07-18T12:00:00Z",
            )
            store.review(
                observation.request_digest,
                expected_solved=False,
                phenomenon="false_equality",
                rationale="The case needs another independent check.",
                reviewer="human:second",
                decision="rejected",
                reviewed_at="2026-07-18T13:00:00Z",
            )

            item = store.get_item(observation.request_digest)
            corpus = store.export_reviewed()

        self.assertEqual(item.status, ExperienceQueueStatus.REJECTED)
        self.assertEqual(item.latest_review.review.reviewer, "human:second")
        self.assertEqual(corpus.records, ())

    def test_partition_contract_cannot_change_after_store_creation(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experience.db"
            TypedExperienceStore(path)

            with self.assertRaisesRegex(
                ExperienceStoreIntegrityError,
                "partition_json",
            ):
                TypedExperienceStore(
                    path,
                    partition=ExperiencePartitionConfig(seed="different-seed"),
                )

    def test_review_corpus_cannot_mix_partition_contracts(self) -> None:
        with TemporaryDirectory() as directory:
            first_store = TypedExperienceStore(
                Path(directory) / "first.db",
                partition=ExperiencePartitionConfig(seed="first-partition"),
            )
            second_store = TypedExperienceStore(
                Path(directory) / "second.db",
                partition=ExperiencePartitionConfig(seed="second-partition"),
            )
            records = []
            for store, token in (
                (first_store, "first"),
                (second_store, "second"),
            ):
                observation = _observation(
                    f"event-{token}",
                    _language_request(token),
                    True,
                )
                store.add_observation(observation)
                records.append(
                    store.review(
                        observation.request_digest,
                        expected_solved=True,
                        phenomenon="partition_binding",
                        rationale="The exact case was reviewed for this partition.",
                        reviewer="human:partition-test",
                        decision="approved",
                        reviewed_at="2026-07-18T12:00:00Z",
                    )
                )

            with self.assertRaisesRegex(ValueError, "mixes partition"):
                ExperienceReviewCorpus(
                    tuple(records),
                    first_store.partition.fingerprint,
                )

    def test_tampered_observation_and_review_rows_are_detected(self) -> None:
        request = _language_request("audit")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experience.db"
            store = TypedExperienceStore(path)
            observation = _observation("tamper-event", request, True)
            store.add_observation(observation)

            with closing(sqlite3.connect(path)) as connection:
                payload = json.loads(
                    connection.execute(
                        "SELECT event_json FROM experience_observations"
                    ).fetchone()[0]
                )
                payload["rationale"] = "tampered"
                connection.execute(
                    "UPDATE experience_observations SET event_json = ?",
                    (json.dumps(payload),),
                )
                connection.commit()
            with self.assertRaisesRegex(
                ExperienceStoreIntegrityError,
                "digest mismatch",
            ):
                store.get_item(observation.request_digest)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "experience.db"
            store = TypedExperienceStore(path)
            observation = _observation("review-tamper-event", request, True)
            store.add_observation(observation)
            store.review(
                observation.request_digest,
                expected_solved=True,
                phenomenon="review_tamper",
                rationale="This exact case was reviewed.",
                reviewer="human:test",
                decision="approved",
                reviewed_at="2026-07-18T12:00:00Z",
            )
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    "UPDATE experience_reviews SET decision = 'rejected'"
                )
                connection.commit()
            with self.assertRaisesRegex(
                ExperienceStoreIntegrityError,
                "decision mismatch",
            ):
                store.latest_review(observation.request_digest)

    def test_resource_and_proposal_authority_limits_fail_closed(self) -> None:
        request = _language_request("limits")
        with self.assertRaisesRegex(ValueError, "authority"):
            observation_from_request(
                request,
                event_id="missing-authority",
                observed_success=False,
                observed_verified=False,
                proposed_expected_solved=True,
                proposal_authority="unknown",
                trigger="unsolved",
                rationale="Expected to solve.",
                source="test",
                grounded_fingerprint=_digest("grounded"),
            )

        with TemporaryDirectory() as directory:
            store = TypedExperienceStore(
                Path(directory) / "experience.db",
                max_requests=1,
                max_proof_steps=1,
            )
            store.add_observation(_observation("first", request, True))
            with self.assertRaisesRegex(ValueError, "request limit"):
                store.add_observation(
                    _observation(
                        "second",
                        _language_request("another"),
                        True,
                    )
                )
            with self.assertRaisesRegex(ValueError, "proof-step limit"):
                store.add_observation(
                    replace(
                        _observation("proof", request, True),
                        proof_program=("one", "two"),
                    )
                )

        with TemporaryDirectory() as directory:
            store = TypedExperienceStore(
                Path(directory) / "experience.db",
                max_record_bytes=100,
            )
            with self.assertRaisesRegex(ValueError, "record byte limit"):
                store.add_observation(_observation("large-record", request, True))

        with TemporaryDirectory() as directory:
            store = TypedExperienceStore(Path(directory) / "experience.db")
            observation = _observation("review-authority", request, True)
            store.add_observation(observation)
            with self.assertRaisesRegex(ValueError, "human: reviewer"):
                store.review(
                    observation.request_digest,
                    expected_solved=True,
                    phenomenon="authority_boundary",
                    rationale="A model cannot attest its own semantic label.",
                    reviewer="frontier:judge",
                    decision="approved",
                )
            with self.assertRaisesRegex(ValueError, "human: reviewer"):
                store.review(
                    observation.request_digest,
                    expected_solved=True,
                    phenomenon="authority_boundary",
                    rationale="An empty reviewer identity is not sufficient.",
                    reviewer="human:",
                    decision="approved",
                )

        with self.assertRaisesRegex(TypeError, "proof program"):
            replace(
                _observation("bad-proof", request, True),
                proof_program="not-a-list",
            )
        mapping = _observation("bad-mapping-proof", request, True).to_dict()
        mapping["proof_program"] = "not-a-list"
        with self.assertRaisesRegex(TypeError, "proof program"):
            ExperienceObservation.from_mapping(mapping)


def _language_request(token: str) -> TypedDomainRequest:
    return TypedDomainRequest(
        "language",
        LanguageTextProblem(
            f"Goal: {token}; Requires: review-{token}",
            "Experience queue test",
            False,
        ),
    )


def _observation(
    event_id: str,
    request: TypedDomainRequest,
    proposed: bool,
    *,
    authority: ExperienceProposalAuthority = (
        ExperienceProposalAuthority.USER_PROPOSAL
    ),
) -> ExperienceObservation:
    return observation_from_request(
        request,
        event_id=event_id,
        observed_success=False,
        observed_verified=False,
        proposed_expected_solved=proposed,
        proposal_authority=authority,
        trigger=ExperienceTrigger.EXPECTATION_MISMATCH,
        rationale="A reviewer or judge proposed the expected solver outcome.",
        source="typed-experience-test",
        grounded_fingerprint=_digest(f"grounded:{event_id}"),
        halt_reason="fixed_point",
        expansions=3,
        created_at="2026-07-18T11:00:00Z",
    )


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()

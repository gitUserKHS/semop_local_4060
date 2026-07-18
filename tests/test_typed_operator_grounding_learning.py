from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest


from semop.kernel import (
    AssertionStatus,
    FactStatus,
    GroundingAuthority,
    GroundingBoundaryError,
    GroundingLabel,
    GroundingLearningExample,
    GroundingTrace,
    KernelRegistry,
    grounding_payload_digest,
    make_grounding_candidate,
    promote_grounding_proposal,
    stage_grounding_proposal,
)
from semop.tiny_controller import (
    GroundingLearningBudget,
    GroundingPolicyOutcome,
    GroundingPolicyStore,
    GroundingReplayBuffer,
    SparseGroundingLearner,
    SparseGroundingPolicy,
    VerifiedGroundingLearningLoop,
    VerifiedGroundingOnlineLearningLoop,
    audit_grounding_split,
    encode_grounding_candidate,
    evaluate_grounding_policy,
    grounding_risk_coverage_curve,
    select_grounding_review_candidates,
)


def _candidate(
    domain: str,
    input_key: str,
    *,
    signal: float,
    symbol_name: str = "item",
    sensor_name: str = "measurement.consistency",
):
    registry = KernelRegistry()
    entity = registry.types.register("Entity")
    registry.register_predicate("SUPPORTED", (entity,))
    symbol = registry.symbol(symbol_name, entity)
    return make_grounding_candidate(
        domain=domain,
        statement=f"{symbol_name} has typed support",
        atom=registry.atom("SUPPORTED", symbol),
        producer_id=f"{domain}_sensor_v1",
        source=f"{domain}_candidate",
        input_digest=grounding_payload_digest(input_key),
        assertion_status=AssertionStatus.INFERRED,
        confidence=0.8,
        sensor_features=((sensor_name, signal), ("sensor.quality", 1.0)),
    )


def _labeled_example(
    domain: str,
    input_key: str,
    *,
    accepted: bool,
    signal: float | None = None,
):
    candidate = _candidate(
        domain,
        input_key,
        signal=(1.0 if accepted else -1.0) if signal is None else signal,
        symbol_name=f"entity_{input_key}",
    )
    proposal = stage_grounding_proposal(
        candidate,
        authority=GroundingAuthority.MODEL_PROPOSAL,
        verifier_id="test-grounder",
        rationale="test sensor proposed a typed concept",
        confidence=0.8,
    )
    decided = promote_grounding_proposal(
        proposal,
        lambda _atom: accepted,
        authority=GroundingAuthority.EXTERNAL_VERIFIER,
        verifier_id="exact-test-grounding-verifier",
        rationale="independent fixture checked the candidate",
    )
    trace = GroundingTrace((decided,))
    return trace.learning_examples[0], candidate, trace


def _lmv_examples(prefix: str):
    examples = []
    for domain in ("language", "math", "vision"):
        for index in range(2):
            examples.append(
                _labeled_example(
                    domain,
                    f"{prefix}:{domain}:accept:{index}",
                    accepted=True,
                )[0]
            )
            examples.append(
                _labeled_example(
                    domain,
                    f"{prefix}:{domain}:reject:{index}",
                    accepted=False,
                )[0]
            )
    return tuple(examples)


class GroundingFeatureTests(unittest.TestCase):
    def test_entity_names_are_anonymized_but_sensor_structure_is_retained(self) -> None:
        first = _candidate(
            "language",
            "first-input",
            signal=0.75,
            symbol_name="alpha",
        )
        second = _candidate(
            "language",
            "second-input",
            signal=0.75,
            symbol_name="beta",
        )

        first_features = encode_grounding_candidate(first)
        second_features = encode_grounding_candidate(second)

        self.assertEqual(first_features, second_features)
        self.assertIn(
            ("sensor:shared:measurement.consistency", 0.75),
            first_features.values,
        )
        self.assertFalse(any("alpha" in name for name, _value in first_features.values))

    def test_semantic_candidate_digest_ignores_record_and_producer_names(self) -> None:
        candidate = _candidate("language", "same-semantics", signal=1.0)
        renamed = replace(
            candidate,
            candidate_id="renamed-record-id",
            producer_id="different-producer",
            source="different-source",
        )

        self.assertEqual(candidate.candidate_digest, renamed.candidate_digest)

    def test_untrusted_authority_cannot_construct_a_training_label(self) -> None:
        candidate = _candidate("language", "untrusted", signal=1.0)
        with self.assertRaises(GroundingBoundaryError):
            GroundingLearningExample(
                candidate,
                GroundingLabel.ACCEPT,
                GroundingAuthority.MODEL_PROPOSAL,
                "a" * 64,
            )

    def test_unknown_sensor_contract_forces_abstention(self) -> None:
        policy = SparseGroundingPolicy(
            weights=(("sensor:shared:measurement.consistency", 4.0),),
            feature_support=(("sensor:shared:measurement.consistency", 10),),
            min_feature_support=2,
            training_examples=10,
            positive_examples=5,
            negative_examples=5,
        )
        unknown = _candidate(
            "vision",
            "unknown-sensor",
            signal=1.0,
            sensor_name="novel.depth.residual",
        )

        prediction = policy.predict(unknown)

        self.assertIs(prediction.outcome, GroundingPolicyOutcome.ABSTAIN)
        self.assertEqual(prediction.feature_support, 0)

    def test_sensor_contract_rejects_embedded_verifier_labels(self) -> None:
        with self.assertRaises(GroundingBoundaryError):
            _candidate(
                "vision",
                "leaking-sensor",
                signal=1.0,
                sensor_name="detector.verified_label",
            )


class VerifiedGroundingLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training = _lmv_examples("train")
        self.validation = _lmv_examples("validation")
        self.budget = GroundingLearningBudget(
            min_training_examples=12,
            min_validation_examples=12,
            min_examples_per_label=6,
            required_selective_accuracy=1.0,
            min_validation_coverage=1.0,
            min_domain_coverage=1.0,
            required_domains=("language", "math", "vision"),
        )

    def test_one_sparse_policy_learns_all_three_domains_and_remains_a_proposer(self) -> None:
        result = VerifiedGroundingLearningLoop(budget=self.budget).run(
            self.training,
            self.validation,
        )

        self.assertTrue(result.promoted, result.rejection_reasons)
        self.assertEqual(result.candidate_metrics.selective_accuracy, 1.0)
        self.assertEqual(result.candidate_metrics.coverage, 1.0)
        self.assertEqual(result.candidate_metrics.false_accepts, 0)
        self.assertEqual(
            {item.domain for item in result.candidate_metrics.by_domain},
            {"language", "math", "vision"},
        )
        self.assertLess(result.active_policy.parameter_count, 1_000)

        accepted_example = next(
            item for item in self.validation if item.label is GroundingLabel.ACCEPT
        )
        prediction, record = result.active_policy.stage_if_accepted(
            accepted_example.candidate
        )
        self.assertIs(prediction.outcome, GroundingPolicyOutcome.ACCEPT)
        self.assertIsNotNone(record)
        self.assertIs(record.fact.status, FactStatus.PROPOSED)
        self.assertFalse(record.fact.proof_eligible)
        self.assertIs(record.decision.authority, GroundingAuthority.MODEL_PROPOSAL)

        curve = grounding_risk_coverage_curve(
            result.active_policy,
            self.validation,
        )
        self.assertEqual(curve[-1].coverage, 1.0)
        self.assertEqual(result.candidate_metrics.aurc, 0.0)

    def test_abstentions_are_prioritized_for_independent_review(self) -> None:
        policy = SparseGroundingPolicy(
            weights=(("sensor:shared:measurement.consistency", 4.0),),
            feature_support=(("sensor:shared:measurement.consistency", 10),),
            training_examples=10,
            positive_examples=5,
            negative_examples=5,
        )
        known = _candidate("language", "known", signal=1.0)
        unknown = _candidate(
            "language",
            "unknown",
            signal=1.0,
            sensor_name="novel.sensor",
        )
        records = tuple(
            stage_grounding_proposal(
                candidate,
                authority=GroundingAuthority.MODEL_PROPOSAL,
                verifier_id="test-grounder",
                rationale="candidate awaits review",
                confidence=0.8,
            )
            for candidate in (known, unknown)
        )

        selected = select_grounding_review_candidates(
            GroundingTrace(records),
            policy,
        )

        self.assertEqual(selected[0].candidate.candidate_id, unknown.candidate_id)
        self.assertIs(
            selected[0].prediction.outcome,
            GroundingPolicyOutcome.ABSTAIN,
        )

    def test_training_is_deterministic_and_artifact_round_trips(self) -> None:
        learner = SparseGroundingLearner()
        buffer = GroundingReplayBuffer.from_examples(self.training)

        first = learner.train(buffer)
        second = learner.train(buffer)
        restored = SparseGroundingPolicy.from_artifact(first.artifact)

        self.assertEqual(first.artifact, second.artifact)
        self.assertEqual(restored, first.policy)
        self.assertEqual(
            evaluate_grounding_policy(restored, self.validation),
            evaluate_grounding_policy(first.policy, self.validation),
        )

    def test_raw_input_lineage_cannot_cross_the_split(self) -> None:
        training_example, _, _ = _labeled_example(
            "language",
            "shared-raw-input",
            accepted=True,
        )
        validation_example, _, _ = _labeled_example(
            "math",
            "shared-raw-input",
            accepted=False,
            signal=-1.0,
        )
        training = GroundingReplayBuffer.from_examples((training_example,))
        validation = GroundingReplayBuffer.from_examples((validation_example,))

        audit = audit_grounding_split(training, validation)

        self.assertFalse(audit.valid)
        self.assertEqual(audit.overlapping_input_groups, 1)
        with self.assertRaises(GroundingBoundaryError):
            VerifiedGroundingLearningLoop(
                budget=GroundingLearningBudget(
                    min_training_examples=1,
                    min_validation_examples=1,
                    min_examples_per_label=1,
                )
            ).run(training, validation)

    def test_false_accept_on_untouched_validation_blocks_promotion(self) -> None:
        misleading_training = tuple(
            _labeled_example(
                domain,
                f"misleading-train:{domain}:{index}",
                accepted=True,
                signal=-1.0,
            )[0]
            if index % 2 == 0
            else _labeled_example(
                domain,
                f"misleading-train:{domain}:{index}",
                accepted=False,
                signal=1.0,
            )[0]
            for domain in ("language", "math", "vision")
            for index in range(4)
        )

        result = VerifiedGroundingLearningLoop(budget=self.budget).run(
            misleading_training,
            self.validation,
        )

        self.assertFalse(result.promoted)
        self.assertTrue(
            any(reason.startswith("false_accepts:") for reason in result.rejection_reasons),
            result.rejection_reasons,
        )
        self.assertEqual(result.active_policy, SparseGroundingPolicy())

    def test_checkpoint_persists_replay_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = GroundingPolicyStore(Path(directory) / "grounder")
            result = VerifiedGroundingLearningLoop(
                budget=self.budget,
                store=store,
            ).run(self.training, self.validation)

            self.assertTrue(result.promoted)
            self.assertEqual(store.load_policy(), result.active_policy)
            self.assertEqual(len(store.load_training().items), len(self.training))
            self.assertEqual(len(store.load_validation().items), len(self.validation))

            (store.root / "policy.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                store.load_policy()

    def test_online_updates_replay_old_labels_and_advance_only_on_promotion(self) -> None:
        first_batch = tuple(
            item
            for index, item in enumerate(self.training)
            if index % 4 in (0, 1)
        )
        second_batch = tuple(
            item
            for index, item in enumerate(self.training)
            if index % 4 in (2, 3)
        )
        online = VerifiedGroundingOnlineLearningLoop(
            budget=GroundingLearningBudget(
                min_training_examples=6,
                min_validation_examples=12,
                min_examples_per_label=3,
                required_selective_accuracy=1.0,
                min_validation_coverage=1.0,
                min_domain_coverage=1.0,
                required_domains=("language", "math", "vision"),
            )
        )

        first = online.update(first_batch, self.validation)
        second = online.update(
            second_batch,
            self.validation,
            state=first.state,
        )

        self.assertTrue(first.learning.promoted)
        self.assertTrue(second.learning.promoted)
        self.assertEqual(first.state.generation, 1)
        self.assertEqual(second.state.generation, 2)
        self.assertEqual(second.replay_examples, 12)
        self.assertEqual(second.state.active_policy.training_examples, 12)
        self.assertEqual(
            evaluate_grounding_policy(
                second.state.active_policy,
                self.validation,
            ).selective_accuracy,
            1.0,
        )
        with self.assertRaisesRegex(ValueError, "requires a new"):
            online.update(
                second_batch,
                self.validation,
                state=second.state,
            )


if __name__ == "__main__":
    unittest.main()

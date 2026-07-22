from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    DomainKind,
    LearningSplit,
    MigrationMode,
    RasterImage,
    RasterVisionProblem,
    RawExperienceGrounder,
    RawGroundingError,
    RawLearningExample,
    RawSelfLearningLoop,
    SelfLearningBudget,
    SelfLearningLoop,
    TypedDomainRequest,
    UnifiedTypedReasoner,
    VisionPropertyGoal,
)


WHITE = (255, 255, 255)
RED = (255, 0, 0)


def _request(
    domain: str,
    payload,
    *,
    mode: str = "shadow",
) -> TypedDomainRequest:
    return TypedDomainRequest(domain, payload, mode)


def _raw(
    example_id: str,
    domain: str,
    payload,
    *,
    split: LearningSplit = LearningSplit.TRAIN,
    expected_solved: bool = True,
    mode: str = "shadow",
) -> RawLearningExample:
    return RawLearningExample(
        example_id,
        _request(domain, payload, mode=mode),
        split=split,
        expected_solved=expected_solved,
    )


def _square_problem() -> RasterVisionProblem:
    image = RasterImage.from_rows(
        (
            [WHITE] * 5,
            [WHITE, RED, RED, WHITE, WHITE],
            [WHITE, RED, RED, WHITE, WHITE],
            [WHITE] * 5,
        )
    )
    return RasterVisionProblem(
        image,
        (VisionPropertyGoal("SQUARE", "red"),),
    )


class _FailingAdapter:
    def adapt(self, _payload):
        raise RuntimeError("intentional adapter failure")


class RawGroundingTests(unittest.TestCase):
    def test_public_ground_boundary_dispatches_language_math_and_pixels(self) -> None:
        reasoner = UnifiedTypedReasoner()
        requests = (
            _request(
                "language",
                "Goal: deploy; Requires: tests; Satisfied: tests",
            ),
            _request("math", "2*x + 3 = 11"),
            _request("vision", _square_problem()),
        )

        grounded = tuple(reasoner.ground(request) for request in requests)
        executed = tuple(reasoner.run(request) for request in requests)

        self.assertEqual(
            [instance.domain for instance in grounded],
            ["language", "math", "vision"],
        )
        self.assertTrue(
            all(result.success and result.verified for result in executed)
        )
        self.assertEqual(
            [instance.goals for instance in grounded],
            [result.instance.goals for result in executed],
        )

    def test_grounder_adds_only_bounded_wrong_goal_actions(self) -> None:
        batch = RawExperienceGrounder(hard_negatives_per_example=4).ground(
            (
                _raw(
                    "language-train",
                    "language",
                    "Goal: deploy; Requires: tests; Satisfied: tests",
                ),
            )
        )
        task = batch.require_complete()[0]
        distractors = tuple(
            operator
            for operator in task.instance.registry.operators.values()
            if "hard_negative" in operator.tags
        )

        self.assertEqual(len(distractors), 4)
        self.assertEqual(task.instance.metadata["experience_hard_negatives"], 4)
        self.assertTrue(
            task.instance.metadata["experience_input_fingerprint"]
        )
        self.assertTrue(
            task.instance.metadata["experience_semantic_fingerprint"]
        )

    def test_untrusted_or_ungroundable_inputs_are_recorded_not_silently_dropped(self) -> None:
        reasoner = UnifiedTypedReasoner(
            {DomainKind.MATH: _FailingAdapter()}
        )
        grounder = RawExperienceGrounder(reasoner)
        batch = grounder.ground(
            (
                _raw(
                    "legacy",
                    "language",
                    "Goal: deploy; Satisfied: tests",
                    mode=MigrationMode.LEGACY.value,
                ),
                _raw("no-goal", "language", "An ambiguous sentence."),
                _raw("adapter-error", "math", "2 + 2"),
            )
        )

        self.assertFalse(batch.complete)
        self.assertEqual(batch.tasks, ())
        self.assertEqual(
            {failure.example_id for failure in batch.failures},
            {"legacy", "no-goal", "adapter-error"},
        )
        with self.assertRaises(RawGroundingError) as caught:
            batch.require_complete()
        self.assertEqual(len(caught.exception.failures), 3)

    def test_raw_grounder_rejects_prebuilt_instances_without_mutating_them(self) -> None:
        instance = UnifiedTypedReasoner().ground(
            _request("math", "2 + 2")
        )
        operator_names = tuple(sorted(instance.registry.operators))
        batch = RawExperienceGrounder().ground(
            (
                RawLearningExample(
                    "prebuilt",
                    TypedDomainRequest("math", instance),
                ),
            )
        )

        self.assertFalse(batch.complete)
        self.assertIn("prebuilt DomainInstance", batch.failures[0].message)
        self.assertEqual(tuple(sorted(instance.registry.operators)), operator_names)

    def test_contradicted_language_claims_do_not_become_training_evidence(self) -> None:
        batch = RawExperienceGrounder(hard_negatives_per_example=0).ground(
            (
                _raw(
                    "contradiction",
                    "language",
                    "Goal: deploy; Requires: tests; Satisfied: tests; Blocked: tests",
                ),
            )
        )
        instance = batch.require_complete()[0].instance
        status_facts = tuple(
            fact
            for fact in instance.state.facts
            if fact.atom.predicate.name in {"SATISFIED", "BLOCKED"}
        )

        self.assertTrue(status_facts)
        self.assertTrue(all(not fact.proof_eligible for fact in status_facts))


class RawSplitAuditTests(unittest.TestCase):
    def test_exact_input_and_grounded_semantic_overlap_are_rejected(self) -> None:
        text = "Goal: deploy; Requires: tests; Satisfied: tests"
        split = RawExperienceGrounder().ground_split(
            (_raw("train", "language", text),),
            (
                _raw(
                    "heldout",
                    "language",
                    text,
                    split=LearningSplit.HELDOUT,
                ),
            ),
        )

        self.assertFalse(split.leakage_free)
        self.assertEqual(len(split.input_overlap), 1)
        self.assertEqual(len(split.semantic_overlap), 1)
        with self.assertRaisesRegex(ValueError, "input overlap"):
            split.require_ready()

    def test_format_variation_still_detects_semantic_overlap(self) -> None:
        split = RawExperienceGrounder().ground_split(
            (
                _raw(
                    "train",
                    "language",
                    "Goal: deploy; Requires: tests; Satisfied: tests",
                ),
            ),
            (
                _raw(
                    "heldout",
                    "language",
                    "Goal: deploy\nRequires: tests\nSatisfied: tests",
                    split=LearningSplit.HELDOUT,
                ),
            ),
        )

        self.assertEqual(split.input_overlap, ())
        self.assertEqual(len(split.semantic_overlap), 1)
        with self.assertRaisesRegex(ValueError, "semantic overlap"):
            split.require_ready()

    def test_split_labels_and_ids_are_validated_before_learning(self) -> None:
        heldout = _raw(
            "heldout",
            "math",
            "2 + 2",
            split=LearningSplit.HELDOUT,
        )
        grounder = RawExperienceGrounder()

        with self.assertRaisesRegex(ValueError, "wrong split"):
            grounder.ground_split((heldout,), (heldout,))
        with self.assertRaisesRegex(ValueError, "unique across splits"):
            grounder.ground_split(
                (_raw("same", "math", "2 + 2"),),
                (
                    _raw(
                        "same",
                        "math",
                        "3 + 4",
                        split=LearningSplit.HELDOUT,
                    ),
                ),
            )


class RawSelfLearningTests(unittest.TestCase):
    def test_verified_raw_trace_reaches_the_corpus_with_audit_metadata(self) -> None:
        training = (
            _raw(
                "train-deploy",
                "language",
                "Goal: deploy; Requires: tests; Satisfied: tests",
            ),
        )
        heldout = (
            _raw(
                "heldout-publish",
                "language",
                "Goal: publish; Requires: review; Satisfied: review",
                split=LearningSplit.HELDOUT,
            ),
        )
        result = RawSelfLearningLoop(
            SelfLearningLoop(
                budget=SelfLearningBudget(min_expansion_reduction=0.0)
            )
        ).run(training, heldout, namespace="raw-test")

        self.assertTrue(result.grounding.leakage_free)
        self.assertEqual(len(result.learning.corpus.records), 1)
        record = result.learning.corpus.records[0]
        metadata = dict(record.metadata)
        self.assertIn("experience_input_fingerprint", metadata)
        self.assertIn("experience_semantic_fingerprint", metadata)
        self.assertEqual(metadata["experience_source_domain"], "language")
        self.assertEqual(record.source, "verifier")
        self.assertTrue(record.actions)
        self.assertTrue(record.hard_negatives)
        self.assertEqual(len(result.learning.iterations), 1)


if __name__ == "__main__":
    unittest.main()

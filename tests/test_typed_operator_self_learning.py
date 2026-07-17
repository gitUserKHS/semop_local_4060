from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    Goal,
    LearningSplit,
    LearningTask,
    PolicyCandidate,
    PolicyDecision,
    SelfLearningBudget,
    SelfLearningLoop,
    SelfLearningStore,
    SolveBudget,
    WorldState,
    generate_symbolic_curriculum,
    generate_symbolic_negative_controls,
    learning_tasks_from_synthetic,
    OperatorKernel,
)
from semop.tiny_controller import (
    StructuralLinearPolicy,
    StructuralPolicyLearner,
)


def _curriculum_tasks(
    count: int,
    *,
    seed: int,
    split: LearningSplit,
    namespace: str,
) -> tuple[LearningTask, ...]:
    return learning_tasks_from_synthetic(
        generate_symbolic_curriculum(
            count,
            seed=seed,
            curriculum="language-math-vision",
        ),
        split=split,
        namespace=namespace,
    )


def _negative_controls(
    heldout: tuple[LearningTask, ...],
) -> tuple[LearningTask, ...]:
    by_domain = {task.domain: task for task in heldout}

    language = by_domain["language"].instance
    language_without_premises = replace(
        language,
        state=WorldState(
            tuple(
                fact
                for fact in language.state.facts
                if fact.atom.predicate.name != "SATISFIED"
            )
        ),
    )

    math = by_domain["math"].instance
    math_goal = math.goals[0]
    wrong_number = math.registry.symbol("heldout_wrong_answer", "Number")
    wrong_math = replace(
        math,
        goals=(
            Goal(
                math.registry.atom(
                    math_goal.atom.predicate,
                    math_goal.atom.arguments[0],
                    wrong_number,
                ),
                label="intentionally false arithmetic target",
            ),
        ),
    )

    vision = by_domain["vision"].instance
    vision_goal = vision.goals[0]
    wrong_vision = replace(
        vision,
        goals=(
            Goal(
                vision.registry.atom(
                    vision_goal.atom.predicate,
                    vision_goal.atom.arguments[1],
                    vision_goal.atom.arguments[0],
                ),
                label="intentionally reversed spatial target",
            ),
        ),
    )

    return (
        LearningTask(
            "negative:language",
            language_without_premises,
            expected_solved=False,
            split=LearningSplit.HELDOUT,
            source="synthetic",
            domain="language",
        ),
        LearningTask(
            "negative:math",
            wrong_math,
            expected_solved=False,
            split=LearningSplit.HELDOUT,
            source="synthetic",
            domain="math",
        ),
        LearningTask(
            "negative:vision",
            wrong_vision,
            expected_solved=False,
            split=LearningSplit.HELDOUT,
            source="synthetic",
            domain="vision",
        ),
    )


class _HardNegativeFirstPolicy:
    def score_actions(self, _state, _goals, actions):
        return PolicyDecision(
            tuple(
                100.0 if "hard_negative" in action.operator.tags else 0.0
                for action in actions
            ),
            action_limit=1,
        )


class _BadLearner:
    name = "test-hard-negative-first"

    def train(self, _cases, _incumbent=None):
        return PolicyCandidate(
            policy=_HardNegativeFirstPolicy(),
            kind=self.name,
            artifact_suffix=".json",
            artifact=b"{}",
            parameter_count=1,
            training_updates=1,
        )

    def restore(self, _artifact):
        return _HardNegativeFirstPolicy()


class TypedSelfLearningTests(unittest.TestCase):
    def test_generated_negative_controls_remain_unsolved(self) -> None:
        positives = generate_symbolic_curriculum(
            1,
            seed=2,
            curriculum="language-math-vision",
        )
        controls = generate_symbolic_negative_controls(positives)

        self.assertEqual(
            [item.domain for item in controls],
            ["language", "math", "vision"],
        )
        for control in controls:
            result = OperatorKernel(control.instance.registry).solve(
                control.instance.state,
                control.instance.goals,
            )
            self.assertFalse(result.success, control.problem_id)
            self.assertFalse(result.verified, control.problem_id)

    def test_three_domain_policy_is_promoted_only_after_heldout_gate(self) -> None:
        training = _curriculum_tasks(
            3,
            seed=11,
            split=LearningSplit.TRAIN,
            namespace="train",
        )
        positives = _curriculum_tasks(
            3,
            seed=29,
            split=LearningSplit.HELDOUT,
            namespace="heldout",
        )
        heldout = positives + _negative_controls(positives)
        result = SelfLearningLoop(
            budget=SelfLearningBudget(
                solve_budget=SolveBudget(
                    max_expansions=2_000,
                    timeout_seconds=5.0,
                ),
                min_expansion_reduction=0.10,
            )
        ).run(training, heldout)

        iteration = result.iterations[0]
        self.assertTrue(iteration.accepted, iteration.rejection_reasons)
        self.assertTrue(result.promoted)
        self.assertIsInstance(result.final_policy, StructuralLinearPolicy)
        self.assertEqual(iteration.verified_training_traces, 9)
        self.assertGreater(iteration.decision_cases, 9)
        self.assertGreater(iteration.training_updates, 0)
        self.assertLess(iteration.parameter_count, 100)
        self.assertGreaterEqual(iteration.expansion_reduction, 0.30)
        self.assertEqual(iteration.candidate_metrics.proof_soundness, 1.0)
        self.assertEqual(iteration.candidate_metrics.false_positives, 0)
        self.assertEqual(iteration.candidate_metrics.verified_solve_rate, 1.0)
        self.assertEqual(iteration.candidate_metrics.expected_outcome_accuracy, 1.0)
        self.assertLess(
            iteration.candidate_metrics.positive_expansions,
            iteration.baseline_metrics.positive_expansions,
        )
        for domain in ("language", "math", "vision"):
            metrics = iteration.candidate_metrics.for_domain(domain)
            self.assertEqual(metrics.verified_solve_rate, 1.0)
            self.assertEqual(metrics.proof_soundness, 1.0)
            self.assertEqual(metrics.false_positives, 0)

    def test_bad_policy_is_rejected_and_cannot_create_false_facts(self) -> None:
        training = _curriculum_tasks(
            1,
            seed=3,
            split=LearningSplit.TRAIN,
            namespace="train",
        )
        positives = _curriculum_tasks(
            1,
            seed=5,
            split=LearningSplit.HELDOUT,
            namespace="heldout",
        )
        heldout = positives + _negative_controls(positives)
        result = SelfLearningLoop(
            learner=_BadLearner(),
            budget=SelfLearningBudget(
                solve_budget=SolveBudget(
                    max_expansions=2_000,
                    timeout_seconds=5.0,
                ),
                min_expansion_reduction=0.90,
            ),
        ).run(training, heldout)

        iteration = result.iterations[0]
        self.assertFalse(iteration.accepted)
        self.assertIsNone(result.final_policy)
        self.assertEqual(result.accepted_generations, 0)
        self.assertTrue(
            any(
                reason.startswith("expansion_reduction_below_gate")
                for reason in iteration.rejection_reasons
            )
        )
        self.assertEqual(iteration.candidate_metrics.false_positives, 0)
        self.assertEqual(iteration.candidate_metrics.proof_soundness, 1.0)
        negative_results = (
            item for item in iteration.candidate_tasks if not item.expected_solved
        )
        self.assertTrue(all(not item.success for item in negative_results))

    def test_training_label_conflict_blocks_learning(self) -> None:
        positive = _curriculum_tasks(
            1,
            seed=7,
            split=LearningSplit.TRAIN,
            namespace="train",
        )[0]
        mislabeled = replace(
            positive,
            task_id="train:mislabeled",
            expected_solved=False,
        )
        heldout = _curriculum_tasks(
            1,
            seed=9,
            split=LearningSplit.HELDOUT,
            namespace="heldout",
        )

        result = SelfLearningLoop().run((mislabeled,), heldout)

        iteration = result.iterations[0]
        self.assertFalse(iteration.accepted)
        self.assertEqual(
            iteration.training_label_conflicts,
            ("train:mislabeled",),
        )
        self.assertIn(
            "training_labels_conflict_with_typed_verifier",
            iteration.rejection_reasons,
        )
        self.assertEqual(len(result.corpus.records), 0)

    def test_checkpoint_roundtrip_hash_and_duplicate_generation_guard(self) -> None:
        training = _curriculum_tasks(
            2,
            seed=13,
            split=LearningSplit.TRAIN,
            namespace="train",
        )
        heldout = _curriculum_tasks(
            2,
            seed=17,
            split=LearningSplit.HELDOUT,
            namespace="heldout",
        )
        with tempfile.TemporaryDirectory() as directory:
            store = SelfLearningStore(Path(directory) / "learning-state")
            result = SelfLearningLoop(
                learner=StructuralPolicyLearner(),
                budget=SelfLearningBudget(
                    iterations=2,
                    solve_budget=SolveBudget(
                        max_expansions=2_000,
                        timeout_seconds=5.0,
                    ),
                    min_expansion_reduction=0.10,
                ),
                store=store,
            ).run(training, heldout)

            self.assertTrue(result.iterations[0].accepted)
            self.assertFalse(result.iterations[1].accepted)
            self.assertIn(
                "candidate_is_identical_to_incumbent",
                result.iterations[1].rejection_reasons,
            )
            self.assertEqual(result.accepted_generations, 1)
            checkpoint = store.load_checkpoint()
            self.assertEqual(checkpoint.generation, 1)
            self.assertEqual(checkpoint.iteration, 2)
            self.assertEqual(checkpoint.trace_count, len(training))
            self.assertEqual(
                len(store.load_corpus(checkpoint).records),
                len(training),
            )
            restored = store.restore_candidate(
                StructuralPolicyLearner(), checkpoint
            )
            self.assertIsInstance(restored.policy, StructuralLinearPolicy)
            self.assertEqual(restored.policy, result.final_policy)
            self.assertFalse(tuple(store.root.rglob("*.tmp")))

            policy_path = store.root / checkpoint.policy_artifact
            policy_path.write_bytes(policy_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                store.load_checkpoint()


if __name__ == "__main__":
    unittest.main()

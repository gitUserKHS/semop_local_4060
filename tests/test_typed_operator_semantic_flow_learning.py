from __future__ import annotations

import unittest

from semop.kernel import (
    ActiveCurriculumConfig,
    ActiveCurriculumScheduler,
    ActiveSelfLearningLoop,
    LearningSplit,
    OperatorKernel,
    SelfLearningBudget,
    SolveBudget,
    audit_structural_split,
    generate_semantic_flow_transfer_split,
    learning_tasks_from_synthetic,
)


def _learning_split(examples_per_structure: int = 1, *, seed: int = 17):
    split = generate_semantic_flow_transfer_split(
        examples_per_structure,
        seed=seed,
    )
    training = learning_tasks_from_synthetic(
        split.training,
        split=LearningSplit.TRAIN,
    )
    heldout_positive = learning_tasks_from_synthetic(
        split.heldout,
        split=LearningSplit.HELDOUT,
    )
    heldout_negative = learning_tasks_from_synthetic(
        split.negative_controls,
        split=LearningSplit.HELDOUT,
        expected_solved=False,
    )
    return split, training, heldout_positive, heldout_negative


class SemanticFlowTransferTests(unittest.TestCase):
    def test_split_holds_out_operator_programs_phrasing_and_larger_images(self) -> None:
        split, training, heldout_positive, heldout_negative = _learning_split(2)
        audit = audit_structural_split(
            training,
            heldout_positive + heldout_negative,
        )
        training_counts = [
            sum(count for _color, count in problem.instance.metadata["object_counts"])
            for problem in split.training
        ]
        heldout_counts = [
            sum(count for _color, count in problem.instance.metadata["object_counts"])
            for problem in split.heldout
        ]
        heldout_texts = tuple(
            str(problem.instance.metadata["text"]) for problem in split.heldout
        )

        self.assertTrue(audit.valid)
        self.assertEqual(audit.overlapping_structures, ())
        self.assertEqual(audit.overlapping_programs, ())
        self.assertLess(max(training_counts), max(heldout_counts))
        self.assertTrue(any("number of" in text for text in heldout_texts))
        self.assertTrue(any("물체의 개수" in text for text in heldout_texts))
        self.assertEqual(
            tuple(profile.proof_depth for profile in audit.training_profiles),
            (3, 3, 5, 5),
        )
        self.assertEqual(
            tuple(
                profile.proof_depth
                for profile in audit.heldout_profiles
                if profile.expected_solved
            ),
            (4, 4, 7, 7),
        )

    def test_every_positive_trace_crosses_all_three_domains(self) -> None:
        split, _training, _heldout_positive, _heldout_negative = _learning_split()
        for problem in split.training + split.heldout:
            result = OperatorKernel(problem.instance.registry).solve(
                problem.instance.state,
                problem.instance.goals,
            )
            tags = {
                tag
                for step in result.proof
                for tag in step.action.operator.tags
            }
            self.assertTrue(result.success and result.verified, problem.problem_id)
            self.assertTrue(
                {"vision", "math", "language"}.issubset(tags),
                problem.problem_id,
            )
        for problem in split.negative_controls:
            result = OperatorKernel(problem.instance.registry).solve(
                problem.instance.state,
                problem.instance.goals,
            )
            self.assertFalse(result.success, problem.problem_id)

    def test_verified_traces_learn_a_tiny_policy_for_heldout_flows(self) -> None:
        _split, training, heldout_positive, heldout_negative = _learning_split()
        result = ActiveSelfLearningLoop(
            scheduler=ActiveCurriculumScheduler(
                ActiveCurriculumConfig(max_tasks=len(training))
            ),
            self_learning_budget=SelfLearningBudget(
                solve_budget=SolveBudget(
                    max_expansions=5_000,
                    timeout_seconds=10.0,
                ),
                min_expansion_reduction=0.10,
            ),
            generations=1,
        ).run(training, heldout_positive + heldout_negative)
        iteration = result.rounds[0].learning_iteration
        candidate = iteration.candidate_metrics

        self.assertTrue(result.split_audit.valid)
        self.assertTrue(iteration.accepted, iteration.rejection_reasons)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.proof_soundness, 1.0)
        self.assertEqual(candidate.false_positives, 0)
        self.assertEqual(candidate.verified_solve_rate, 1.0)
        self.assertGreaterEqual(iteration.expansion_reduction, 0.30)
        self.assertLess(iteration.parameter_count, 100)
        self.assertLess(iteration.artifact_bytes, 2_000)


if __name__ == "__main__":
    unittest.main()

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
    ActiveCurriculumConfig,
    ActiveCurriculumScheduler,
    ActiveSelfLearningLoop,
    LearningSplit,
    OperatorKernel,
    SelfLearningBudget,
    SelfLearningStore,
    SolveBudget,
    audit_structural_split,
    generate_lmv_structural_transfer_split,
    learning_tasks_from_synthetic,
    profile_learning_task,
)
from semop.tiny_controller import (
    TinyControllerConfig,
    TinyControllerPolicyLearner,
)


def _tasks(examples_per_structure: int = 2):
    split = generate_lmv_structural_transfer_split(
        examples_per_structure,
        seed=17,
    )
    training = learning_tasks_from_synthetic(
        split.training,
        split=LearningSplit.TRAIN,
        namespace="train",
    )
    heldout = learning_tasks_from_synthetic(
        split.heldout,
        split=LearningSplit.HELDOUT,
        namespace="heldout",
    ) + learning_tasks_from_synthetic(
        split.negative_controls,
        split=LearningSplit.HELDOUT,
        namespace="negative",
        expected_solved=False,
    )
    return split, training, heldout


class TypedCurriculumTests(unittest.TestCase):
    def test_structural_split_uses_real_adapter_capabilities(self) -> None:
        split, training, heldout = _tasks(1)

        self.assertEqual(len(split.training), 6)
        self.assertEqual(len(split.heldout), 3)
        self.assertEqual(len(split.negative_controls), 3)
        self.assertFalse(
            set(split.training_structures) & set(split.heldout_structures)
        )
        self.assertEqual(
            {item.capability for item in split.training},
            {
                "premise_readiness",
                "inheritance_chain",
                "nested_arithmetic",
                "linear_equation",
                "spatial_transitivity",
                "pixel_shape",
            },
        )
        self.assertEqual(
            {item.capability for item in split.heldout},
            {
                "conjunctive_rule_chain",
                "exact_comparison",
                "pixel_quantification",
            },
        )
        self.assertTrue(all(task.structure_key for task in training + heldout))
        self.assertTrue(all(task.capability for task in training + heldout))
        self.assertIn(
            "natural_language_text",
            {item.instance.metadata.get("source") for item in split.training},
        )
        self.assertIn(
            "raster",
            {item.instance.metadata.get("input_kind") for item in split.training},
        )

    def test_profiles_and_audit_prove_non_overlapping_replayable_split(self) -> None:
        _split, training, heldout = _tasks(1)
        audit = audit_structural_split(training, heldout)

        self.assertTrue(audit.valid)
        self.assertEqual(audit.overlapping_structures, ())
        self.assertEqual(audit.overlapping_programs, ())
        self.assertEqual(
            audit.training_domains,
            ("language", "math", "vision"),
        )
        self.assertEqual(audit.training_domains, audit.heldout_domains)
        self.assertTrue(audit.training_outcomes_valid)
        self.assertTrue(audit.heldout_outcomes_valid)
        positive_profiles = [
            item for item in audit.heldout_profiles if item.expected_solved
        ]
        self.assertTrue(all(item.replay_verified for item in positive_profiles))
        self.assertEqual(
            {
                family
                for item in positive_profiles
                for family in item.operator_families
            },
            {"compare", "compose", "quantify"},
        )
        negative_profiles = [
            item for item in audit.heldout_profiles if not item.expected_solved
        ]
        self.assertTrue(
            all(
                item.outcome_matches_label and not item.replay_verified
                for item in negative_profiles
            )
        )

    def test_scheduler_is_balanced_diverse_and_deterministic(self) -> None:
        _split, training, _heldout = _tasks(2)
        scheduler = ActiveCurriculumScheduler(
            ActiveCurriculumConfig(max_tasks=6)
        )

        first = scheduler.select(training)
        second = scheduler.select(training)

        self.assertEqual(
            [item.task_id for item in first.selected_tasks],
            [item.task_id for item in second.selected_tasks],
        )
        self.assertEqual(
            dict(first.domain_counts),
            {"language": 2, "math": 2, "vision": 2},
        )
        self.assertEqual(
            len({item.structure_key for item in first.decisions}),
            6,
        )
        self.assertTrue(all(item.novelty == 1.0 for item in first.decisions))
        self.assertEqual(first.rejected_unverified, ())
        self.assertEqual(first.rejected_no_supervision, ())
        profile = profile_learning_task(first.selected_tasks[0])
        self.assertTrue(profile.replay_verified)
        self.assertGreater(profile.decision_cases, 0)

    def test_active_learning_transfers_to_unseen_compositions_and_checkpoints(self) -> None:
        _split, training, heldout = _tasks(2)
        budget = SelfLearningBudget(
            solve_budget=SolveBudget(
                max_expansions=2_000,
                timeout_seconds=5.0,
            ),
            min_expansion_reduction=0.10,
        )
        with tempfile.TemporaryDirectory() as directory:
            store = SelfLearningStore(Path(directory) / "active-learning")
            result = ActiveSelfLearningLoop(
                scheduler=ActiveCurriculumScheduler(
                    ActiveCurriculumConfig(max_tasks=6)
                ),
                self_learning_budget=budget,
                generations=1,
                store=store,
            ).run(training, heldout)

            self.assertTrue(result.split_audit.valid)
            self.assertEqual(result.promoted_generations, 1)
            self.assertEqual(len(result.selected_task_ids), 6)
            self.assertEqual(len(result.corpus.records), 6)
            iteration = result.rounds[0].learning_iteration
            self.assertTrue(iteration.accepted, iteration.rejection_reasons)
            self.assertGreaterEqual(iteration.expansion_reduction, 0.30)
            self.assertEqual(iteration.candidate_metrics.verified_solve_rate, 1.0)
            self.assertEqual(iteration.candidate_metrics.proof_soundness, 1.0)
            self.assertEqual(iteration.candidate_metrics.false_positives, 0)
            self.assertLess(iteration.parameter_count, 100)
            self.assertIsNotNone(result.checkpoint)
            self.assertEqual(store.load_checkpoint().generation, 1)

    def test_overlap_is_reported_and_active_loop_refuses_leakage(self) -> None:
        _split, training, heldout = _tasks(1)
        leaked = replace(
            heldout[0],
            structure_key=training[0].structure_key,
        )
        contaminated = (leaked,) + heldout[1:]
        audit = audit_structural_split(training, contaminated)

        self.assertFalse(audit.valid)
        self.assertEqual(
            audit.overlapping_structures,
            (training[0].structure_key,),
        )
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            ActiveSelfLearningLoop().run(training, contaminated)

    def test_second_generation_retains_traces_but_rolls_back_no_gain_policy(self) -> None:
        _split, training, heldout = _tasks(2)
        result = ActiveSelfLearningLoop(
            scheduler=ActiveCurriculumScheduler(
                ActiveCurriculumConfig(max_tasks=6)
            ),
            self_learning_budget=SelfLearningBudget(
                min_expansion_reduction=0.01,
            ),
            generations=2,
        ).run(training, heldout)

        self.assertEqual(len(result.rounds), 2)
        self.assertTrue(result.rounds[0].learning_iteration.accepted)
        second = result.rounds[1].learning_iteration
        self.assertFalse(second.accepted)
        self.assertTrue(
            any(
                reason.startswith("expansion_reduction_below_gate")
                for reason in second.rejection_reasons
            )
        )
        self.assertEqual(result.promoted_generations, 1)
        self.assertEqual(len(result.selected_task_ids), 12)
        self.assertEqual(len(set(result.selected_task_ids)), 12)
        self.assertEqual(len(result.corpus.records), 12)
        self.assertIs(result.final_policy, result.active_candidate.policy)
        self.assertEqual(second.candidate_metrics.proof_soundness, 1.0)
        self.assertEqual(second.candidate_metrics.false_positives, 0)

    def test_recurrent_candidate_uses_same_verifier_gate_and_rolls_back(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch training profile is not installed")
        _split, training, heldout = _tasks(1)
        learner = TinyControllerPolicyLearner(
            config=TinyControllerConfig(
                d_model=8,
                token_buckets=32,
                relation_buckets=8,
                operator_buckets=8,
                message_blocks=1,
                recursion_steps=1,
                action_limit_score_margin=100.0,
            ),
            epochs=1,
            learning_rate=1e-3,
            seed=23,
        )
        result = ActiveSelfLearningLoop(
            learner=learner,
            scheduler=ActiveCurriculumScheduler(
                ActiveCurriculumConfig(max_tasks=6)
            ),
            self_learning_budget=SelfLearningBudget(
                solve_budget=SolveBudget(
                    max_expansions=2_000,
                    timeout_seconds=5.0,
                    beam_width=5,
                ),
                min_expansion_reduction=0.95,
            ),
            generations=1,
        ).run(training, heldout)

        iteration = result.rounds[0].learning_iteration
        self.assertEqual(iteration.candidate_kind, "tiny-controller-v6")
        self.assertEqual(iteration.verified_training_traces, 6)
        self.assertGreater(iteration.training_updates, iteration.decision_cases)
        self.assertEqual(iteration.candidate_metrics.proof_soundness, 1.0)
        self.assertEqual(iteration.candidate_metrics.false_positives, 0)
        self.assertFalse(iteration.accepted)
        self.assertEqual(result.promoted_generations, 0)
        self.assertIsNone(result.final_policy)
        self.assertIsNone(result.active_candidate)
        self.assertTrue(
            any(
                reason.startswith("expansion_reduction_below_gate")
                for reason in iteration.rejection_reasons
            )
        )

    def test_all_structural_negative_controls_are_unprovable(self) -> None:
        split = generate_lmv_structural_transfer_split(2, seed=29)

        for control in split.negative_controls:
            result = OperatorKernel(control.instance.registry).solve(
                control.instance.state,
                control.instance.goals,
            )
            self.assertFalse(result.success, control.problem_id)
            self.assertFalse(result.verified, control.problem_id)


if __name__ == "__main__":
    unittest.main()

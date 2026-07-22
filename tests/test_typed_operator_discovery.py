from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    LearningSplit,
    OperatorKernel,
    SelfDiscoveringLearningLoop,
    SelfLearningBudget,
    SolveBudget,
    TaskDiscoveryBudget,
    VerifiedTaskDiscovery,
    generate_lmv_structural_transfer_split,
    learning_tasks_from_synthetic,
    profile_learning_task,
)


def _tasks(seed: int = 41, examples_per_structure: int = 1):
    split = generate_lmv_structural_transfer_split(
        examples_per_structure,
        seed=seed,
    )
    training = learning_tasks_from_synthetic(
        split.training,
        split=LearningSplit.TRAIN,
        namespace="discovery-train",
    )
    heldout = learning_tasks_from_synthetic(
        split.heldout,
        split=LearningSplit.HELDOUT,
        namespace="discovery-heldout",
    ) + learning_tasks_from_synthetic(
        split.negative_controls,
        split=LearningSplit.HELDOUT,
        namespace="discovery-negative",
        expected_solved=False,
    )
    return training, heldout


class _FailingPolicy:
    def score_actions(self, _state, _goals, _actions):
        raise RuntimeError("intentional discovery-policy failure")


class TypedTaskDiscoveryTests(unittest.TestCase):
    def test_budget_rejects_unbounded_or_invalid_discovery(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            TaskDiscoveryBudget(max_components=1)
        with self.assertRaisesRegex(ValueError, "proof depth"):
            TaskDiscoveryBudget(
                max_positive_proof_depth=7,
                max_candidate_proof_depth=6,
            )
        with self.assertRaisesRegex(ValueError, "negative limits"):
            TaskDiscoveryBudget(max_counterfactual_attempts_per_positive=0)

    def test_discovers_novel_replayable_multidomain_programs(self) -> None:
        training, _heldout = _tasks()
        discovery = VerifiedTaskDiscovery().discover(training)

        self.assertGreaterEqual(len(discovery.positive_tasks), 6)
        self.assertLessEqual(len(discovery.positive_tasks), 12)
        self.assertEqual(
            len(discovery.positive_tasks),
            len({task.structure_key for task in discovery.positive_tasks}),
        )
        self.assertTrue(
            all(task.domain == "multidomain" for task in discovery.positive_tasks)
        )
        self.assertTrue(
            any(
                len(record.parent_task_ids) == 3
                for record in discovery.accepted_records
                if record.expected_solved
            )
        )

        source_programs = {
            profile_learning_task(task).program_signature for task in training
        }
        discovered_programs = set()
        for task in discovery.positive_tasks:
            result = OperatorKernel(task.instance.registry).solve(
                task.instance.state,
                task.instance.goals,
            )
            profile = profile_learning_task(task)
            self.assertTrue(result.success and result.verified, task.task_id)
            self.assertLessEqual(len(result.proof), 6, task.task_id)
            self.assertNotIn(profile.program_signature, source_programs)
            discovered_programs.add(profile.program_signature)
        self.assertEqual(
            len(discovered_programs),
            len(discovery.positive_tasks),
        )

    def test_counterfactuals_are_verified_unsolved_not_label_flips(self) -> None:
        training, _heldout = _tasks()
        discovery = VerifiedTaskDiscovery().discover(training)

        self.assertEqual(
            len(discovery.negative_tasks),
            len(discovery.positive_tasks),
        )
        for task in discovery.negative_tasks:
            result = OperatorKernel(task.instance.registry).solve(
                task.instance.state,
                task.instance.goals,
            )
            self.assertFalse(result.success, task.task_id)
            self.assertFalse(result.verified, task.task_id)
        negative_records = tuple(
            record
            for record in discovery.accepted_records
            if not record.expected_solved
        )
        self.assertEqual(len(negative_records), len(discovery.negative_tasks))
        self.assertTrue(all(record.removed_fact for record in negative_records))

    def test_discovery_is_deterministic_and_deduplicates_renamed_samples(self) -> None:
        training, _heldout = _tasks(examples_per_structure=2)
        first = VerifiedTaskDiscovery().discover(training)
        second = VerifiedTaskDiscovery().discover(training)

        self.assertEqual(
            [task.task_id for task in first.positive_tasks],
            [task.task_id for task in second.positive_tasks],
        )
        self.assertEqual(
            first.discovered_structures,
            second.discovered_structures,
        )
        self.assertTrue(
            any(
                "duplicate_seed_structure" in record.rejection_reasons
                for record in first.rejected_records
            )
        )

    def test_policy_fallback_is_recorded_as_discovery_failure_signal(self) -> None:
        training, _heldout = _tasks()
        result = VerifiedTaskDiscovery(
            TaskDiscoveryBudget(max_positive_tasks=4, max_negative_tasks=0)
        ).discover(training, policy=_FailingPolicy())

        positives = tuple(
            record
            for record in result.accepted_records
            if record.expected_solved
        )
        self.assertTrue(positives)
        self.assertTrue(all(record.policy_success is False for record in positives))
        self.assertTrue(all(record.priority >= 1.0 for record in positives))

    def test_tight_seed_budget_still_covers_all_three_domains(self) -> None:
        training, _heldout = _tasks()
        result = VerifiedTaskDiscovery(
            TaskDiscoveryBudget(
                max_seed_tasks=3,
                max_positive_tasks=4,
                max_negative_tasks=0,
            )
        ).discover(training)

        source_domains = {task.task_id: task.domain for task in training}
        used_domains = {
            source_domains[parent_id]
            for record in result.accepted_records
            if record.expected_solved
            for parent_id in record.parent_task_ids
        }
        self.assertEqual(used_domains, {"language", "math", "vision"})

    def test_self_discovering_loop_transfers_to_long_heldout_compositions(self) -> None:
        training, heldout = _tasks()
        result = SelfDiscoveringLearningLoop(
            self_learning_budget=SelfLearningBudget(
                solve_budget=SolveBudget(
                    max_expansions=5_000,
                    timeout_seconds=10.0,
                ),
                min_expansion_reduction=0.10,
            )
        ).run(training, heldout)

        self.assertEqual(len(result.training_discovery.positive_tasks), 12)
        self.assertEqual(len(result.heldout_discovery.positive_tasks), 4)
        self.assertTrue(result.learning.split_audit.valid)
        self.assertEqual(result.learning.split_audit.overlapping_structures, ())
        self.assertEqual(result.learning.split_audit.overlapping_programs, ())
        self.assertIn("multidomain", result.learning.split_audit.training_domains)
        self.assertEqual(
            result.learning.split_audit.training_domains,
            result.learning.split_audit.heldout_domains,
        )
        self.assertGreaterEqual(
            max(
                profile.proof_depth
                for profile in result.learning.split_audit.heldout_profiles
                if profile.domain == "multidomain" and profile.expected_solved
            ),
            7,
        )

        iteration = result.learning.rounds[0].learning_iteration
        self.assertTrue(iteration.accepted, iteration.rejection_reasons)
        self.assertGreaterEqual(iteration.expansion_reduction, 0.30)
        self.assertEqual(iteration.candidate_metrics.verified_solve_rate, 1.0)
        self.assertEqual(iteration.candidate_metrics.proof_soundness, 1.0)
        self.assertEqual(iteration.candidate_metrics.false_positives, 0)
        self.assertGreater(
            iteration.baseline_metrics.positive_expansions,
            iteration.candidate_metrics.positive_expansions,
        )
        self.assertTrue(
            all(len(record.actions) <= 6 for record in result.learning.corpus.records)
        )
        expanded_heldout_ids = {
            task.task_id for task in result.expanded_heldout_tasks
        }
        self.assertTrue(
            all(
                task.task_id in expanded_heldout_ids
                for task in result.training_discovery.negative_tasks
            )
        )
        corpus_ids = {record.trace_id for record in result.learning.corpus.records}
        self.assertTrue(
            all(
                task.task_id not in corpus_ids
                for task in result.training_discovery.negative_tasks
            )
        )
        multidomain_record = next(
            record
            for record in result.learning.corpus.records
            if record.domain == "multidomain"
        )
        trace_metadata = dict(multidomain_record.metadata)
        self.assertEqual(
            trace_metadata["discovery_mutation"],
            "domain_composition",
        )
        self.assertIn("discovery_parent_ids", trace_metadata)
        self.assertIn("structure_key", trace_metadata)

    def test_second_generation_keeps_new_traces_but_rejects_no_gain_policy(self) -> None:
        training, heldout = _tasks()
        result = SelfDiscoveringLearningLoop(
            generations=2,
            self_learning_budget=SelfLearningBudget(
                solve_budget=SolveBudget(
                    max_expansions=5_000,
                    timeout_seconds=10.0,
                ),
                min_expansion_reduction=0.01,
            ),
        ).run(training, heldout)

        self.assertEqual(len(result.learning.rounds), 2)
        self.assertTrue(result.learning.rounds[0].learning_iteration.accepted)
        second = result.learning.rounds[1].learning_iteration
        self.assertFalse(second.accepted)
        self.assertTrue(
            any(
                reason.startswith("expansion_reduction_below_gate")
                for reason in second.rejection_reasons
            )
        )
        self.assertEqual(result.learning.promoted_generations, 1)
        self.assertEqual(len(result.learning.selected_task_ids), 16)
        self.assertEqual(len(set(result.learning.selected_task_ids)), 16)
        self.assertEqual(len(result.learning.corpus.records), 16)
        self.assertIs(
            result.learning.final_policy,
            result.learning.active_candidate.policy,
        )
        self.assertEqual(second.candidate_metrics.proof_soundness, 1.0)
        self.assertEqual(second.candidate_metrics.false_positives, 0)


if __name__ == "__main__":
    unittest.main()

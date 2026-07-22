from __future__ import annotations

from base64 import b64encode
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    HierarchicalOperatorBrain,
    HierarchicalLearningBudget,
    HierarchicalSelfLearningLoop,
    OperatorKernel,
    TypedDomainRequest,
    UnifiedTypedReasoner,
    generate_hierarchical_brain_transfer_split,
    hierarchical_learning_tasks_from_curriculum,
)
from semop.tiny_controller import StructuralPolicyLearner


class _FailingPolicyProvider:
    def policy_for(self, _registry):
        raise RuntimeError("intentional provider failure")


class TypedOperatorHierarchicalBrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.curriculum = generate_hierarchical_brain_transfer_split(seed=23)
        cls.tasks = hierarchical_learning_tasks_from_curriculum(
            cls.curriculum,
            controller_domain="language",
            namespace="hierarchical-test",
        )
        cls.controller_training = cls.tasks.controller_training
        cls.controller_heldout = cls.tasks.controller_heldout
        cls.macro_training = cls.tasks.macro_training
        cls.macro_validation = cls.tasks.macro_validation
        cls.macro_heldout = cls.tasks.macro_heldout
        cls.joint_heldout = cls.tasks.joint_heldout
        cls.result = HierarchicalSelfLearningLoop().run_tasks(cls.tasks)

    def test_curriculum_separates_component_names_and_vision_inputs(self) -> None:
        self.assertEqual(len(self.curriculum.controller_training), 9)
        self.assertEqual(len(self.curriculum.controller_heldout), 3)
        self.assertEqual(len(self.curriculum.macro_training), 9)
        self.assertEqual(len(self.curriculum.macro_validation), 3)
        self.assertEqual(len(self.curriculum.macro_heldout), 3)
        self.assertEqual(len(self.curriculum.joint_heldout), 3)

        positive_groups = (
            self.curriculum.controller_training,
            self.curriculum.controller_heldout,
            self.curriculum.macro_training,
            self.curriculum.macro_validation,
            self.curriculum.macro_heldout,
            self.curriculum.joint_heldout,
        )
        identifiers = [
            problem.problem_id
            for group in positive_groups
            for problem in group
        ]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for group in positive_groups:
            self.assertEqual(
                {problem.domain for problem in group},
                {"language", "math", "vision"},
            )

        controller_operators = {
            problem.instance.metadata["completion_operator"]
            for problem in (
                self.curriculum.controller_training
                + self.curriculum.controller_heldout
            )
        }
        heldout_operators = {
            problem.instance.metadata["completion_operator"]
            for problem in self.curriculum.joint_heldout
        }
        self.assertTrue(controller_operators.isdisjoint(heldout_operators))
        self.assertTrue(
            all(
                problem.instance.registry.operators[
                    problem.instance.metadata["completion_operator"]
                ].family
                == "verify"
                for problem in self.curriculum.controller_training
                + self.curriculum.controller_heldout
                + self.curriculum.joint_heldout
            )
        )

        digest_sets = tuple(
            {
                problem.instance.metadata["image_digest"]
                for problem in group
                if problem.domain == "vision"
            }
            for group in positive_groups
        )
        for left_index, left in enumerate(digest_sets):
            for right in digest_sets[left_index + 1 :]:
                self.assertTrue(left.isdisjoint(right))

        for seed in range(19, 26):
            curriculum = generate_hierarchical_brain_transfer_split(seed=seed)
            groups = (
                curriculum.controller_training,
                curriculum.controller_heldout,
                curriculum.macro_training,
                curriculum.macro_validation,
                curriculum.macro_heldout,
                curriculum.joint_heldout,
            )
            seeded_digests = tuple(
                {
                    problem.instance.metadata["image_digest"]
                    for problem in group
                    if problem.domain == "vision"
                }
                for group in groups
            )
            for left_index, left in enumerate(seeded_digests):
                for right in seeded_digests[left_index + 1 :]:
                    self.assertTrue(left.isdisjoint(right), seed)

    def test_task_split_builder_rejects_ambiguous_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "controller_domain"):
            hierarchical_learning_tasks_from_curriculum(
                self.curriculum,
                controller_domain="audio",
            )
        with self.assertRaisesRegex(ValueError, "namespace"):
            hierarchical_learning_tasks_from_curriculum(
                self.curriculum,
                namespace="  ",
            )

    def test_joint_brain_is_promoted_only_after_complementary_ablation(self) -> None:
        result = self.result

        self.assertEqual(
            {task.domain for task in self.controller_training},
            {"language"},
        )
        self.assertEqual(
            {task.domain for task in self.controller_heldout},
            {"language"},
        )
        self.assertTrue(result.promoted, result.rejection_reasons)
        self.assertIsNotNone(result.active_brain)
        self.assertIsNotNone(result.controller_learning.active_candidate)
        self.assertTrue(result.macro_learning.promoted)
        self.assertEqual(result.active_domains, ("language", "math", "vision"))
        self.assertEqual(result.deterministic.metrics.positive_expansions, 22)
        self.assertEqual(result.controller_only.metrics.positive_expansions, 14)
        self.assertEqual(result.macro_only.metrics.positive_expansions, 22)
        self.assertEqual(result.joint.metrics.positive_expansions, 10)
        self.assertGreaterEqual(result.joint_expansion_reduction, 0.50)
        self.assertLess(
            result.joint.metrics.positive_expansions,
            result.controller_only.metrics.positive_expansions,
        )
        self.assertLess(
            result.joint.metrics.positive_expansions,
            result.macro_only.metrics.positive_expansions,
        )
        self.assertEqual(result.joint.metrics.proof_soundness, 1.0)
        self.assertEqual(result.joint.metrics.false_positives, 0)
        self.assertLess(result.active_brain.parameter_count, 100)
        self.assertLess(result.artifact_bytes, 8_192)
        weights = dict(result.active_brain.base_policy.weights)
        self.assertGreater(weights["operator:family:verify"], 0.0)
        self.assertLess(weights["operator:family:search"], 0.0)
        self.assertLess(
            result.controller_only.metrics.for_domain("math").positive_expansions,
            result.deterministic.metrics.for_domain("math").positive_expansions,
        )
        self.assertLess(
            result.controller_only.metrics.for_domain("vision").positive_expansions,
            result.deterministic.metrics.for_domain("vision").positive_expansions,
        )

    def test_backward_relevance_uses_macro_below_the_final_goal(self) -> None:
        brain = self.result.active_brain
        self.assertIsNotNone(brain)
        for task in self.joint_heldout:
            if not task.expected_solved:
                continue
            policy = brain.policy_for(task.instance.registry)
            self.assertEqual(policy.active_program_count, 1)
            final_signature = (
                f"{task.instance.goals[0].atom.predicate.name}("
                + ",".join(
                    argument.type.name
                    for argument in task.instance.goals[0].atom.arguments
                )
                + ")"
            )
            self.assertTrue(
                all(
                    final_signature not in program.effect_signature
                    for program in policy.activation.programs
                )
            )
            solved = brain.solve(task.instance)
            self.assertTrue(solved.success)
            self.assertTrue(solved.verified)
            self.assertTrue(
                any(
                    step.action.operator.family == "verify"
                    for step in solved.proof
                )
            )
            self.assertTrue(
                all(
                    step.action.operator.name in task.instance.registry.operators
                    and not step.action.operator.name.startswith("macro_")
                    for step in solved.proof
                )
            )

    def test_brain_artifact_round_trip_preserves_proofs_and_detects_tampering(self) -> None:
        brain = self.result.active_brain
        self.assertIsNotNone(brain)
        learner = StructuralPolicyLearner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "hierarchical-brain.json"
            brain.save(path)
            self.assertFalse(tuple(root.glob("*.tmp")))
            restored = HierarchicalOperatorBrain.load(path, learner)
            self.assertEqual(restored.to_artifact(), brain.to_artifact())

            for task in self.joint_heldout:
                if not task.expected_solved:
                    continue
                original = brain.solve(task.instance)
                loaded = restored.solve(task.instance)
                self.assertEqual(
                    tuple(step.action.operator.name for step in loaded.proof),
                    tuple(step.action.operator.name for step in original.proof),
                )

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["base_policy"]["artifact_base64"] = b64encode(
                b"tampered"
            ).decode("ascii")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                HierarchicalOperatorBrain.from_artifact(
                    json.dumps(payload).encode("utf-8"),
                    learner,
                )

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["macro_library"]["records"][0]["support"] += 1
            with self.assertRaisesRegex(ValueError, "macro artifact hash mismatch"):
                HierarchicalOperatorBrain.from_artifact(
                    json.dumps(payload).encode("utf-8"),
                    learner,
                )

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["base_policy"]["parameter_count"] += 1
            with self.assertRaisesRegex(ValueError, "parameter count mismatch"):
                HierarchicalOperatorBrain.from_artifact(
                    json.dumps(payload).encode("utf-8"),
                    learner,
                )

    def test_unified_reasoner_accepts_registry_policy_provider(self) -> None:
        brain = self.result.active_brain
        self.assertIsNotNone(brain)
        reasoner = UnifiedTypedReasoner()
        for task in self.joint_heldout:
            if not task.expected_solved:
                continue
            result = reasoner.run(
                TypedDomainRequest(
                    domain=task.domain,
                    payload=task.instance,
                    mode="typed",
                ),
                policy=brain,
            )
            self.assertTrue(result.success)
            self.assertTrue(result.verified)
            self.assertTrue(result.typed_result.policy_used)

    def test_rejected_joint_candidate_does_not_replace_known_good_brain(self) -> None:
        rejected = HierarchicalSelfLearningLoop(
            budget=HierarchicalLearningBudget(
                min_joint_expansion_reduction=1.0,
            )
        ).run_tasks(self.tasks)
        self.assertFalse(rejected.promoted)
        self.assertIsNotNone(rejected.candidate_brain)
        self.assertIsNone(rejected.active_brain)
        self.assertIn(
            "joint_expansion_reduction_below_gate",
            rejected.rejection_reasons,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "active-brain.json"
            path.write_text("known-good", encoding="utf-8")
            self.assertIsNone(
                HierarchicalSelfLearningLoop.persist_promoted(rejected, path)
            )
            self.assertEqual(path.read_text(encoding="utf-8"), "known-good")

    def test_policy_provider_failure_falls_back_to_deterministic_search(self) -> None:
        task = next(task for task in self.joint_heldout if task.expected_solved)
        result = OperatorKernel(task.instance.registry).solve(
            task.instance.state,
            task.instance.goals,
            policy=_FailingPolicyProvider(),
        )

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertTrue(result.policy_used)
        self.assertTrue(result.fallback_used)
        self.assertTrue(
            any("policy provider failed" in item for item in result.diagnostics)
        )


if __name__ == "__main__":
    unittest.main()

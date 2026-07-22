from __future__ import annotations

from dataclasses import asdict, replace
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
    LearningSplit,
    MacroLearningBudget,
    MdlMacroLibrary,
    OperatorKernel,
    VerifiedMacroLearningLoop,
    generate_macro_reuse_transfer_split,
    learning_tasks_from_synthetic,
)


class TypedOperatorMacroActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.curriculum = generate_macro_reuse_transfer_split(seed=23)
        cls.training = learning_tasks_from_synthetic(
            cls.curriculum.training,
            split=LearningSplit.TRAIN,
            namespace="macro-train",
        )
        cls.validation = learning_tasks_from_synthetic(
            cls.curriculum.validation,
            split=LearningSplit.HELDOUT,
            namespace="macro-validation",
        )
        cls.heldout = (
            learning_tasks_from_synthetic(
                cls.curriculum.heldout,
                split=LearningSplit.HELDOUT,
                namespace="macro-heldout",
            )
            + learning_tasks_from_synthetic(
                cls.curriculum.negative_controls,
                split=LearningSplit.HELDOUT,
                namespace="macro-negative",
                expected_solved=False,
            )
        )
        cls.result = VerifiedMacroLearningLoop(
            MacroLearningBudget(min_domains_improved=3)
        ).run(cls.training, cls.validation, cls.heldout)

    def test_curriculum_has_disjoint_three_domain_partitions(self) -> None:
        self.assertEqual(len(self.curriculum.training), 9)
        self.assertEqual(len(self.curriculum.validation), 3)
        self.assertEqual(len(self.curriculum.heldout), 3)
        self.assertEqual(len(self.curriculum.negative_controls), 3)

        groups = (
            self.curriculum.training,
            self.curriculum.validation,
            self.curriculum.heldout,
        )
        identifiers = [problem.problem_id for group in groups for problem in group]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for group in groups:
            self.assertEqual(
                {problem.domain for problem in group},
                {"language", "math", "vision"},
            )

        vision_digests = [
            {
                problem.instance.metadata["image_digest"]
                for problem in group
                if problem.domain == "vision"
            }
            for group in groups
        ]
        self.assertTrue(vision_digests[0].isdisjoint(vision_digests[1]))
        self.assertTrue(vision_digests[0].isdisjoint(vision_digests[2]))
        self.assertTrue(vision_digests[1].isdisjoint(vision_digests[2]))

    def test_verified_macros_improve_all_domains_without_false_proofs(self) -> None:
        result = self.result

        self.assertTrue(result.promoted, result.rejection_reasons)
        self.assertEqual(len(result.active_library.records), 3)
        self.assertEqual(result.verified_training_traces, 9)
        self.assertEqual(result.verified_validation_traces, 3)
        self.assertGreaterEqual(result.expansion_reduction, 0.50)
        self.assertEqual(result.improved_domains, ("language", "math", "vision"))
        self.assertEqual(result.guided_metrics.proof_soundness, 1.0)
        self.assertEqual(result.guided_metrics.false_positives, 0)
        self.assertEqual(result.guided_metrics.verified_solve_rate, 1.0)

    def test_guided_successes_are_primitive_and_replayable(self) -> None:
        for task in self.heldout:
            result = self.result.guided_results[task.task_id]
            if task.expected_solved:
                self.assertTrue(result.success)
                self.assertTrue(result.verified)
                self.assertTrue(result.proof)
                self.assertTrue(
                    all(
                        step.action.operator.name in task.instance.registry.operators
                        and not step.action.operator.name.startswith("macro_")
                        for step in result.proof
                    )
                )
                replay = OperatorKernel(task.instance.registry).replay(
                    task.instance.state,
                    task.instance.goals,
                    result.proof,
                )
                self.assertTrue(replay.verified)
            else:
                self.assertFalse(result.success)

    def test_operator_schema_change_disables_matching_macro(self) -> None:
        task = next(task for task in self.heldout if task.domain == "language")
        registry = task.instance.registry
        initial = self.result.active_library.activate(registry)
        self.assertEqual(len(initial.programs), 1)
        target = initial.programs[0]
        operator_name = target.primitive_operator_names[0]
        original = registry.operators[operator_name]
        registry.operators[operator_name] = replace(
            original,
            tags=original.tags + ("schema-change",),
        )
        try:
            activation = self.result.active_library.activate(registry)
        finally:
            registry.operators[operator_name] = original

        self.assertNotIn(target.name, {program.name for program in activation.programs})
        issue = next(
            item for item in activation.rejected if item.macro_name == target.name
        )
        self.assertEqual(
            issue.reason,
            f"primitive_operator_schema_mismatch:{operator_name}",
        )

    def test_legacy_library_loads_but_cannot_activate(self) -> None:
        records = []
        for record in self.result.active_library.records:
            raw = asdict(record)
            raw.pop("operator_schema_fingerprints")
            records.append(raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-macros.json"
            path.write_text(
                json.dumps({"format_version": 1, "records": records}),
                encoding="utf-8",
            )
            library = MdlMacroLibrary.load(path)

        task = next(task for task in self.heldout if task.expected_solved)
        activation = library.activate(task.instance.registry)
        self.assertFalse(activation.programs)
        self.assertEqual(len(activation.rejected), 3)
        self.assertTrue(
            all(
                item.reason == "legacy_record_missing_operator_schema_fingerprints"
                for item in activation.rejected
            )
        )

    def test_promoted_artifact_round_trips_and_rejected_result_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active_path = root / "active-macros.json"
            written = VerifiedMacroLearningLoop.persist_promoted(
                self.result,
                active_path,
            )
            self.assertEqual(written, active_path)
            payload = json.loads(active_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["format_version"], 2)
            self.assertFalse(tuple(root.glob("*.tmp")))

            restored = MdlMacroLibrary.load(active_path)
            self.assertEqual(restored.records, self.result.active_library.records)
            for task in self.heldout:
                if task.expected_solved:
                    self.assertEqual(
                        len(restored.activate(task.instance.registry).programs),
                        1,
                    )

            active_path.write_text("known-good", encoding="utf-8")
            rejected = replace(
                self.result,
                promoted=False,
                rejection_reasons=("forced-test-rejection",),
                active_library=MdlMacroLibrary(),
            )
            self.assertIsNone(
                VerifiedMacroLearningLoop.persist_promoted(rejected, active_path)
            )
            self.assertEqual(active_path.read_text(encoding="utf-8"), "known-good")


if __name__ == "__main__":
    unittest.main()

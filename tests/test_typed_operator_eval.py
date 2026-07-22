from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EVAL = ROOT / "tools" / "eval"
for path in (SRC, EVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_low_resource_transfer import FAST_CORE_TEST_FILES, evaluate
from evaluate_active_macro_learning import evaluate_active_macro_learning
from evaluate_hierarchical_self_learning import (
    _controller_learner as _hierarchical_controller_learner,
    evaluate_hierarchical_self_learning,
)
from evaluate_raw_grounded_self_learning import (
    evaluate_raw_grounded_self_learning,
)
from evaluate_semantic_flow_self_learning import (
    evaluate_semantic_flow_self_learning,
)
from evaluate_typed_task_discovery import evaluate_self_discovery
from evaluate_typed_self_learning import evaluate_self_learning
from semop.tiny_controller import NumpyTinyController, TinyControllerConfig


class LowResourceEvalTests(unittest.TestCase):
    def test_fast_core_evaluator_matches_dependency_free_ci_files(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "typed-core.yml").read_text(
            encoding="utf-8"
        )
        workflow_files = tuple(
            line.strip()
            for line in workflow.splitlines()
            if line.strip().startswith("tests/")
        )

        self.assertEqual(FAST_CORE_TEST_FILES, workflow_files)

    def test_compact_controller_profile_selects_1_46m_challenger(self) -> None:
        learner = _hierarchical_controller_learner(
            "recurrent-compact",
            epochs=1,
            learning_rate=1e-3,
            device="cpu",
            seed=0,
        )

        self.assertEqual(
            learner.config.estimated_parameter_count(),
            1_458_698,
        )

    def test_raw_grounded_evaluator_rejects_unknown_controller_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "controller_profile"):
            evaluate_raw_grounded_self_learning(controller_profile="unknown")

    def test_sparse_raw_language_learning_transfers_to_math_and_pixels(self) -> None:
        report = evaluate_raw_grounded_self_learning(
            controller_profile="sparse",
            seed=31,
        )

        self.assertEqual(report["data"]["training_domains"], ("language",) * 3)
        self.assertEqual(report["data"]["transfer_domains"], ("math", "vision"))
        self.assertEqual(report["learning"]["verified_training_traces"], 3)
        self.assertEqual(report["learning"]["decision_cases"], 9)
        self.assertEqual(
            report["transfer"]["baseline"]["positive_expansions"],
            12,
        )
        self.assertEqual(
            report["transfer"]["candidate"]["positive_expansions"],
            4,
        )
        self.assertTrue(report["artifact"]["round_trip_verified"])
        self.assertEqual(report["artifact"]["runtime"], "StructuralLinearPolicy")
        self.assertTrue(report["gates"]["all_passed"])

    def test_recurrent_raw_language_learning_transfers_to_math_and_pixels(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch training profile is not installed")

        report = evaluate_raw_grounded_self_learning(
            controller_profile="recurrent-diagnostic",
            controller_epochs=5,
            controller_learning_rate=1e-3,
            seed=31,
        )

        self.assertEqual(report["controller"]["kind"], "tiny-controller-v6")
        self.assertEqual(report["controller"]["parameters"], 29_834)
        self.assertEqual(report["controller"]["training_updates"], 60)
        self.assertEqual(
            report["transfer"]["per_domain_expansion_reduction"],
            {"math": 2 / 3, "vision": 2 / 3},
        )
        self.assertTrue(report["gates"]["final_inputs_untouched"])
        self.assertTrue(report["gates"]["primitive_full_replay_verified"])
        self.assertTrue(report["gates"]["portable_cpu_runtime"])
        self.assertEqual(report["artifact"]["runtime"], "NumpyTinyController")
        self.assertTrue(report["gates"]["all_passed"])

    def test_hierarchical_evaluator_rejects_unknown_controller_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "controller_profile"):
            evaluate_hierarchical_self_learning(controller_profile="unknown")

    def test_recurrent_hierarchical_controller_transfers_from_language(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch training profile is not installed")

        report = evaluate_hierarchical_self_learning(
            examples_per_domain=3,
            validation_per_domain=1,
            heldout_per_domain=1,
            seed=23,
            controller_domain="language",
            controller_profile="recurrent-diagnostic",
            controller_epochs=5,
            controller_learning_rate=1e-3,
            min_joint_expansion_reduction=0.50,
        )

        self.assertEqual(report["controller"]["profile"], "recurrent-diagnostic")
        self.assertEqual(report["learning"]["controller_kind"], "tiny-controller-v6")
        self.assertEqual(report["learning"]["controller_parameters"], 29_834)
        self.assertEqual(report["learning"]["controller_training_updates"], 30)
        self.assertEqual(
            report["ab"]["positive_expansions"],
            {
                "deterministic": 22,
                "controller_only": 14,
                "macro_only": 22,
                "joint": 10,
            },
        )
        self.assertTrue(report["artifact"]["round_trip_verified"])
        self.assertTrue(report["gates"]["shared_verify_family_transfers"])
        self.assertTrue(report["gates"]["all_passed"])

    def test_hierarchical_self_learning_report_passes_all_gates(self) -> None:
        report = evaluate_hierarchical_self_learning(
            examples_per_domain=3,
            validation_per_domain=1,
            heldout_per_domain=1,
            seed=23,
            controller_domain="language",
            min_joint_expansion_reduction=0.50,
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["split"]["controller_training_pool"], 9)
        self.assertEqual(report["split"]["controller_training_selected"], 3)
        self.assertEqual(
            report["split"]["zero_shot_controller_domains"],
            ("math", "vision"),
        )
        self.assertEqual(report["split"]["macro_training"], 9)
        self.assertEqual(report["split"]["joint_heldout_positive"], 3)
        self.assertTrue(report["learning"]["promoted"])
        self.assertEqual(report["learning"]["controller_parameters"], 12)
        self.assertEqual(report["learning"]["active_macros"], 3)
        self.assertEqual(
            report["ab"]["positive_expansions"],
            {
                "deterministic": 22,
                "controller_only": 14,
                "macro_only": 22,
                "joint": 10,
            },
        )
        self.assertGreaterEqual(
            report["ab"]["joint_expansion_reduction"],
            0.50,
        )
        self.assertEqual(report["ablations"]["joint"]["proof_soundness"], 1.0)
        self.assertEqual(report["ablations"]["joint"]["false_positives"], 0)
        self.assertTrue(report["gates"]["all_passed"])

    def test_active_macro_learning_report_passes_all_gates(self) -> None:
        report = evaluate_active_macro_learning(
            examples_per_domain=3,
            validation_per_domain=1,
            heldout_per_domain=1,
            seed=23,
            min_expansion_reduction=0.50,
            max_expansions=5_000,
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["split"]["training_tasks"], 9)
        self.assertEqual(report["split"]["validation_tasks"], 3)
        self.assertEqual(report["split"]["heldout_positive_tasks"], 3)
        self.assertTrue(report["learning"]["promoted"])
        self.assertEqual(report["learning"]["retained_macros"], 3)
        self.assertEqual(
            set(report["learning"]["improved_domains"]),
            {"language", "math", "vision"},
        )
        self.assertEqual(report["after"]["proof_soundness"], 1.0)
        self.assertEqual(report["after"]["false_positives"], 0)
        self.assertGreaterEqual(report["ab"]["expansion_reduction"], 0.50)
        self.assertEqual(report["artifact"]["format_version"], 2)
        self.assertTrue(report["gates"]["all_passed"])

    def test_semantic_flow_self_learning_report_passes_all_gates(self) -> None:
        report = evaluate_semantic_flow_self_learning(
            examples_per_structure=1,
            seed=17,
            min_expansion_reduction=0.10,
            max_expansions=5_000,
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["data"]["training_tasks"], 2)
        self.assertEqual(report["data"]["heldout_positive_tasks"], 2)
        self.assertEqual(report["split"]["overlapping_structures"], ())
        self.assertEqual(report["split"]["overlapping_programs"], ())
        self.assertEqual(report["split"]["training_proof_depths"], (3, 5))
        self.assertEqual(
            report["split"]["heldout_positive_proof_depths"],
            (4, 7),
        )
        self.assertTrue(report["policy"]["promoted"])
        self.assertLess(report["policy"]["parameters"], 100)
        self.assertEqual(report["after"]["proof_soundness"], 1.0)
        self.assertEqual(report["after"]["false_positives"], 0)
        self.assertGreaterEqual(report["ab"]["expansion_reduction"], 0.30)
        self.assertTrue(report["gates"]["all_passed"])

    def test_verifier_backed_task_discovery_report_passes_all_gates(self) -> None:
        report = evaluate_self_discovery(
            examples_per_structure=1,
            seed=41,
            min_expansion_reduction=0.10,
            max_expansions=5_000,
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["discovery"]["training_positive"], 12)
        self.assertEqual(report["discovery"]["training_negative"], 12)
        self.assertEqual(report["discovery"]["heldout_positive"], 4)
        self.assertEqual(report["discovery"]["max_training_proof_depth"], 6)
        self.assertGreater(
            report["discovery"]["max_heldout_proof_depth"],
            6,
        )
        self.assertEqual(report["expanded_split"]["overlapping_structures"], ())
        self.assertEqual(report["expanded_split"]["overlapping_programs"], ())
        self.assertEqual(report["after"]["proof_soundness"], 1.0)
        self.assertEqual(report["after"]["false_positives"], 0)
        self.assertGreater(
            report["ab"]["positive_expansions_before"],
            report["ab"]["positive_expansions_after"],
        )
        self.assertTrue(report["gates"]["all_passed"])

    def test_verifier_gated_self_learning_report_is_promotable(self) -> None:
        report = evaluate_self_learning(
            examples_per_domain=1,
            seed=31,
            min_expansion_reduction=0.10,
            max_expansions=2_000,
        )

        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["data"]["training_pool_tasks"], 6)
        self.assertEqual(report["data"]["selected_training_tasks"], 6)
        self.assertEqual(report["data"]["heldout_negative_controls"], 3)
        self.assertEqual(report["split"]["overlapping_structures"], ())
        self.assertEqual(report["split"]["overlapping_programs"], ())
        self.assertEqual(
            set(report["split"]["heldout_structures"]),
            {
                "language:conjunctive_rule_chain",
                "math:exact_comparison",
                "vision:pixel_quantification",
            },
        )
        self.assertEqual(
            report["curriculum"]["domain_counts"],
            {"language": 2, "math": 2, "vision": 2},
        )
        self.assertTrue(report["policy"]["promoted"])
        self.assertLess(report["policy"]["parameters"], 100)
        self.assertEqual(report["after"]["proof_soundness"], 1.0)
        self.assertEqual(report["after"]["false_positives"], 0)
        self.assertGreater(
            report["ab"]["positive_expansions_before"],
            report["ab"]["positive_expansions_after"],
        )
        self.assertFalse(report["macros"]["active"])
        self.assertTrue(report["gates"]["all_passed"])

    def test_report_is_honest_and_machine_readable(self) -> None:
        report = evaluate(max_expansions=2_000)

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["guided"]["proof_soundness"], 1.0)
        self.assertEqual(report["guided"]["false_positives"], 0)
        self.assertIn("median_inference_rounds", report["guided"])
        self.assertIn("0", report["data"]["shot_curves"])
        self.assertIsNone(
            report["gates"]["twenty_shot_reaches_90_percent_of_hundred_shot"]
        )
        self.assertEqual(report["gates"]["overall_status"], "not_evaluated")

    def test_language_math_vision_suite_has_sound_negative_controls(self) -> None:
        report = evaluate(
            max_expansions=5_000,
            suite="language-math-vision",
        )

        self.assertEqual(report["suite"], "language-math-vision")
        self.assertEqual(set(report["guided"]["by_domain"]), {
            "language",
            "math",
            "vision",
        })
        self.assertEqual(report["guided"]["proof_soundness"], 1.0)
        self.assertEqual(report["guided"]["false_positives"], 0)
        self.assertEqual(report["guided"]["verified_solve_rate"], 1.0)
        reductions = report["ab_comparison"]["expansion_reduction_by_case"]
        self.assertGreater(
            reductions["language_typed_argument_distractors"], 0.30
        )
        self.assertGreater(
            reductions["vision_spatial_distractor_chains"], 0.30
        )
        self.assertGreater(
            reductions["language_goal_effect_distractors"], 0.30
        )
        self.assertGreater(
            reductions["math_goal_effect_distractors"], 0.30
        )
        self.assertTrue(
            report["gates"]["expansion_reduction_30_percent_in_two_domains"]
        )
        guided_cases = {
            item["case_id"]: item for item in report["guided"]["cases"]
        }
        self.assertEqual(len(guided_cases), 36)
        self.assertTrue(guided_cases["language_korean_text_ready"]["success"])
        self.assertEqual(
            guided_cases["language_korean_text_ready"]["proof_steps"], 3
        )
        self.assertTrue(guided_cases["language_english_text_blocked"]["success"])
        self.assertEqual(
            guided_cases["language_english_text_blocked"]["proof_steps"], 2
        )
        self.assertFalse(guided_cases["language_english_text_missing"]["success"])
        self.assertFalse(
            guided_cases["language_english_text_missing"]["false_positive"]
        )
        self.assertTrue(guided_cases["language_logic_two_hop"]["success"])
        self.assertEqual(guided_cases["language_logic_two_hop"]["proof_steps"], 2)
        self.assertFalse(guided_cases["language_logic_contradiction"]["success"])
        self.assertTrue(guided_cases["language_logic_conjunction"]["success"])
        self.assertFalse(
            guided_cases["language_logic_missing_conjunct"]["success"]
        )
        self.assertGreater(
            reductions["language_logic_binding_distractors"],
            0.60,
        )
        self.assertTrue(guided_cases["math_linear_equation"]["success"])
        self.assertEqual(guided_cases["math_linear_equation"]["proof_steps"], 2)
        self.assertFalse(guided_cases["math_wrong_equation_target"]["success"])
        self.assertTrue(guided_cases["math_numeric_comparison"]["success"])
        self.assertFalse(guided_cases["math_false_comparison"]["success"])
        self.assertTrue(guided_cases["vision_raster_square"]["success"])
        self.assertTrue(guided_cases["vision_raster_count"]["success"])
        self.assertTrue(guided_cases["vision_raster_area"]["success"])
        self.assertFalse(guided_cases["vision_raster_wrong_count"]["success"])
        self.assertTrue(
            guided_cases["vision_goal_independent_count"]["success"]
        )
        self.assertTrue(
            all(
                item["evaluation_mode"] == "non_learned_policy_baseline"
                for item in report["leave_one_domain_out"].values()
            )
        )

    def test_composed_v3_suite_tracks_cross_domain_positive_and_negative_cases(self) -> None:
        report = evaluate(max_expansions=5_000, suite="composed-v3")

        self.assertEqual(set(report["guided"]["by_domain"]), {"composed"})
        self.assertEqual(report["guided"]["verified_solve_rate"], 1.0)
        self.assertEqual(report["guided"]["proof_soundness"], 1.0)
        self.assertEqual(report["guided"]["false_positives"], 0)
        cases = {item["case_id"]: item for item in report["guided"]["cases"]}
        self.assertEqual(len(cases), 4)
        self.assertEqual(cases["composed_scene_english"]["proof_steps"], 3)
        self.assertEqual(cases["composed_scene_english"]["expansions"], 3)
        self.assertEqual(cases["composed_scene_korean_color"]["proof_steps"], 3)
        self.assertEqual(cases["composed_scene_korean_color"]["expansions"], 3)
        self.assertFalse(cases["composed_scene_false_threshold"]["success"])
        self.assertFalse(cases["composed_scene_explicit_negation"]["success"])
        self.assertLess(
            report["guided"]["expansions"],
            report["unguided"]["expansions"],
        )

    def test_composed_v4_suite_tracks_multi_condition_programs(self) -> None:
        report = evaluate(max_expansions=5_000, suite="composed-v4")

        self.assertEqual(report["guided"]["verified_solve_rate"], 1.0)
        self.assertEqual(report["guided"]["proof_soundness"], 1.0)
        self.assertEqual(report["guided"]["false_positives"], 0)
        self.assertEqual(report["unguided"]["expansions"], 47)
        self.assertEqual(report["guided"]["expansions"], 34)
        cases = {item["case_id"]: item for item in report["guided"]["cases"]}
        self.assertEqual(len(cases), 8)
        self.assertEqual(
            cases["composed_scene_two_color_conjunction"]["proof_steps"],
            5,
        )
        self.assertEqual(
            cases["composed_scene_two_color_conjunction"]["expansions"],
            5,
        )
        self.assertEqual(
            cases["composed_scene_korean_conjunction"]["proof_steps"],
            5,
        )
        self.assertEqual(
            cases["composed_scene_reused_measurement_bounds"]["proof_steps"],
            4,
        )
        self.assertFalse(cases["composed_scene_false_conjunct"]["success"])
        self.assertFalse(cases["composed_scene_false_conjunct"]["false_positive"])

    def test_shot_label_requires_matching_training_metadata(self) -> None:
        config = TinyControllerConfig(
            d_model=8,
            token_buckets=32,
            relation_buckets=8,
            operator_buckets=8,
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "controller.npz"
            NumpyTinyController.random(config, seed=1).save(model_path)
            model_path.with_suffix(".summary.json").write_text(
                json.dumps({"reviewed_examples_per_domain": 0}),
                encoding="utf-8",
            )
            report = evaluate(
                shot_models={20: str(model_path)}, max_expansions=2_000
            )

        curve = report["data"]["shot_curves"]["20"]
        self.assertFalse(curve["available"])
        self.assertIn("declared=0", curve["reason"])
        self.assertTrue(
            all(
                not item["available"]
                for item in report["leave_one_domain_out"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()

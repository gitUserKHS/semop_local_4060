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

from evaluate_low_resource_transfer import evaluate
from evaluate_typed_self_learning import evaluate_self_learning
from semop.tiny_controller import NumpyTinyController, TinyControllerConfig


class LowResourceEvalTests(unittest.TestCase):
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

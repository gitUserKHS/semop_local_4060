from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    GoalDirectedPolicy,
    OperatorKernel,
    RasterImage,
    SceneThresholdAdapter,
    SceneThresholdError,
    SceneThresholdParser,
    SceneThresholdProblem,
    TypedDomainRequest,
    UnifiedTypedReasoner,
)


WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def _three_object_image() -> RasterImage:
    return RasterImage.from_rows(
        (
            [WHITE] * 11,
            [WHITE, RED, RED, WHITE, GREEN, GREEN, WHITE, BLUE, BLUE, WHITE, WHITE],
            [WHITE, RED, RED, WHITE, GREEN, GREEN, WHITE, BLUE, BLUE, WHITE, WHITE],
            [WHITE] * 11,
        )
    )


def _solve(text: str, *, policy=None):
    instance = SceneThresholdAdapter().adapt(
        SceneThresholdProblem(_three_object_image(), text)
    )
    result = OperatorKernel(instance.registry).solve(
        instance.state,
        instance.goals,
        policy=policy,
    )
    return instance, result


class SceneThresholdParserTests(unittest.TestCase):
    def test_english_and_korean_rules_normalize_to_the_same_operator_contract(self) -> None:
        english = SceneThresholdParser().parse(
            "If the number of all objects is greater than 2, the scene is crowded. "
            "Prove: the scene is crowded."
        )
        korean = SceneThresholdParser().parse(
            "모든 물체의 개수가 2보다 크면 장면은 혼잡하다. "
            "증명: 장면은 혼잡하다."
        )

        self.assertEqual(
            (english.selector, english.comparator, str(english.threshold)),
            ("all", ">", "2"),
        )
        self.assertEqual(
            (korean.selector, korean.comparator, str(korean.threshold)),
            ("all", ">", "2"),
        )
        self.assertEqual(english.rule_property, "crowded")
        self.assertEqual(korean.rule_property, "혼잡")

    def test_english_and_korean_conjunctions_share_one_condition_ir(self) -> None:
        english = SceneThresholdParser().parse(
            "If the count of red objects is at least 1 and the count of blue "
            "objects is at least 1, the scene is colorful. "
            "Prove: the scene is colorful."
        )
        korean = SceneThresholdParser().parse(
            "빨간 물체의 개수가 1보다 크거나 같고 파란 물체의 개수가 "
            "1보다 크거나 같으면 장면은 다채롭다. 증명: 장면은 다채롭다."
        )

        english_conditions = tuple(
            (item.selector, item.comparator, str(item.threshold))
            for item in english.conditions
        )
        korean_conditions = tuple(
            (item.selector, item.comparator, str(item.threshold))
            for item in korean.conditions
        )

        self.assertEqual(
            english_conditions,
            (("red", ">=", "1"), ("blue", ">=", "1")),
        )
        self.assertEqual(korean_conditions, english_conditions)

    def test_missing_or_duplicate_contract_parts_are_rejected(self) -> None:
        with self.assertRaisesRegex(SceneThresholdError, "requires one count"):
            SceneThresholdParser().parse("Prove: the scene is crowded.")
        with self.assertRaisesRegex(SceneThresholdError, "more than one proof"):
            SceneThresholdParser().parse(
                "If the count of all objects is at least 1, the scene is occupied. "
                "Prove: the scene is occupied. Prove: the scene is crowded."
            )
        with self.assertRaisesRegex(
            SceneThresholdError,
            "only scene classification",
        ):
            SceneThresholdParser().parse(
                "If the count of all objects is at least 1, the scene is occupied. "
                "Prove: the object is occupied."
            )
        with self.assertRaisesRegex(SceneThresholdError, "duplicate"):
            SceneThresholdParser().parse(
                "If the count of red objects is at least 1 and the count of red "
                "objects is at least 1, the scene is duplicated. "
                "Prove: the scene is duplicated."
            )
        conditions = " and ".join(
            f"the count of color_{index} objects is at least 1"
            for index in range(9)
        )
        with self.assertRaisesRegex(SceneThresholdError, "at most eight"):
            SceneThresholdParser().parse(
                f"If {conditions}, the scene is overloaded. "
                "Prove: the scene is overloaded."
            )


class SceneThresholdReasoningTests(unittest.TestCase):
    def test_pixels_count_compare_and_language_classification_share_one_proof(self) -> None:
        instance, result = _solve(
            "If the number of all objects is greater than 2, the scene is crowded. "
            "Prove: the scene is crowded."
        )

        self.assertTrue(result.success and result.verified)
        self.assertEqual(
            [step.action.operator.name for step in result.proof],
            [
                "count_raster_objects_000",
                "compare_scene_object_count",
                "classify_scene_from_verified_count",
            ],
        )
        tags = {
            tag
            for step in result.proof
            for tag in step.action.operator.tags
        }
        self.assertTrue({"vision", "math", "language"}.issubset(tags))
        self.assertEqual(
            instance.metadata["operator_domains"],
            ("vision", "math", "language"),
        )

    def test_korean_query_uses_the_same_three_stage_program(self) -> None:
        _instance, result = _solve(
            "모든 물체의 개수가 2보다 크면 장면은 혼잡하다. "
            "증명: 장면은 혼잡하다."
        )

        self.assertTrue(result.success and result.verified)
        self.assertEqual(len(result.proof), 3)

    def test_two_visual_counts_require_one_verified_conjunction_program(self) -> None:
        instance, result = _solve(
            "If the count of red objects is at least 1 and the count of blue "
            "objects is at least 1, the scene is colorful. "
            "Prove: the scene is colorful.",
            policy=GoalDirectedPolicy(),
        )

        operators = [step.action.operator.name for step in result.proof]
        self.assertTrue(result.success and result.verified)
        self.assertEqual(result.expansions, 5)
        self.assertEqual(len(result.proof), 5)
        self.assertEqual(
            sum(name.startswith("count_raster_objects") for name in operators),
            2,
        )
        self.assertEqual(
            sum(name.startswith("compare_scene_object_count") for name in operators),
            2,
        )
        self.assertEqual(operators[-1], "classify_scene_from_verified_counts")
        self.assertEqual(instance.metadata["condition_count"], 2)
        self.assertEqual(
            tuple(
                item["selector"]
                for item in instance.metadata["parsed_rule"]["conditions"]
            ),
            ("red", "blue"),
        )

    def test_korean_conjunction_executes_the_same_five_operator_families(self) -> None:
        _instance, result = _solve(
            "빨간 물체의 개수가 1보다 크거나 같고 그리고 파란 물체의 "
            "개수가 1보다 크거나 같으면 장면은 다채롭다. "
            "증명: 장면은 다채롭다.",
            policy=GoalDirectedPolicy(),
        )

        self.assertTrue(result.success and result.verified)
        self.assertEqual(result.expansions, 5)
        self.assertEqual(len(result.proof), 5)

    def test_one_false_conjunct_blocks_classification(self) -> None:
        instance, result = _solve(
            "If the count of red objects is at least 1 and the count of blue "
            "objects is at least 2, the scene is colorful. "
            "Prove: the scene is colorful.",
            policy=GoalDirectedPolicy(),
        )
        property_term = instance.registry.symbol("colorful", "Concept")
        first_condition = instance.registry.symbol("condition_001", "Condition")
        second_condition = instance.registry.symbol("condition_002", "Condition")

        self.assertFalse(result.success)
        self.assertTrue(
            result.final_state.contains(
                instance.registry.atom(
                    "COUNT_CONDITION_MET",
                    first_condition,
                    property_term,
                )
            )
        )
        self.assertFalse(
            result.final_state.contains(
                instance.registry.atom(
                    "COUNT_CONDITION_MET",
                    second_condition,
                    property_term,
                )
            )
        )

    def test_two_bounds_can_reuse_one_visual_measurement(self) -> None:
        _instance, result = _solve(
            "If the count of red objects is at least 1 and the count of red "
            "objects is at most 1, the scene is bounded. "
            "Prove: the scene is bounded.",
            policy=GoalDirectedPolicy(),
        )

        self.assertTrue(result.success and result.verified)
        self.assertEqual(len(result.proof), 4)
        self.assertEqual(
            sum(
                step.action.operator.name.startswith("count_raster_objects")
                for step in result.proof
            ),
            1,
        )

    def test_replay_rechecks_each_conjunct_guard(self) -> None:
        instance, result = _solve(
            "If the count of red objects is at least 1 and the count of blue "
            "objects is at least 1, the scene is colorful. "
            "Prove: the scene is colorful."
        )
        kernel = OperatorKernel(instance.registry)
        instance.registry.guards["verify_scene_count_threshold_002"] = (
            lambda _binding, _state: False
        )

        replay = kernel.replay(instance.state, instance.goals, result.proof)

        self.assertTrue(result.success and result.verified)
        self.assertFalse(replay.verified)
        self.assertTrue(
            any("guard rejected" in diagnostic for diagnostic in replay.diagnostics)
        )

    def test_false_threshold_and_mismatched_goal_remain_unproved(self) -> None:
        _instance, false_threshold = _solve(
            "If the number of all objects is greater than 3, the scene is crowded. "
            "Prove: the scene is crowded."
        )
        _instance, wrong_property = _solve(
            "If the number of all objects is greater than 2, the scene is crowded. "
            "Prove: the scene is sparse."
        )

        self.assertFalse(false_threshold.success)
        self.assertFalse(wrong_property.success)
        self.assertEqual(false_threshold.proof, ())
        self.assertEqual(wrong_property.proof, ())

    def test_explicit_language_negation_blocks_cross_domain_classification(self) -> None:
        _instance, result = _solve(
            "Scene is not crowded. "
            "If the number of all objects is greater than 2, the scene is crowded. "
            "Prove: the scene is crowded."
        )

        self.assertFalse(result.success)

    def test_replay_rechecks_cross_domain_numeric_guard(self) -> None:
        instance, result = _solve(
            "If the count of red objects is at least 1, the scene is occupied. "
            "Prove: the scene is occupied."
        )
        kernel = OperatorKernel(instance.registry)
        instance.registry.guards["verify_scene_count_threshold"] = (
            lambda _binding, _state: False
        )

        replay = kernel.replay(instance.state, instance.goals, result.proof)

        self.assertTrue(result.success and result.verified)
        self.assertFalse(replay.verified)
        self.assertIn("guard rejected", replay.diagnostics[0])

    def test_unified_runtime_and_goal_policy_accept_composed_domain(self) -> None:
        problem = SceneThresholdProblem(
            _three_object_image(),
            "If the count of blue objects is equal to 1, the scene is balanced. "
            "Prove: the scene is balanced.",
        )
        result = UnifiedTypedReasoner().run(
            TypedDomainRequest("composed", problem, "typed"),
            policy=GoalDirectedPolicy(),
        )

        self.assertTrue(result.success and result.verified)
        self.assertEqual(result.domain.value, "composed")
        self.assertFalse(result.projection_applied)


if __name__ == "__main__":
    unittest.main()

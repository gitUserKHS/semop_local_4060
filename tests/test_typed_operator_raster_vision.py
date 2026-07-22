from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    FactStatus,
    Goal,
    GroundingAuthority,
    GroundingDisposition,
    OperatorKernel,
    RasterImage,
    RasterVisionAdapter,
    RasterVisionConfig,
    RasterVisionProblem,
    TypedDomainRequest,
    UnifiedTypedReasoner,
    VisionRelationGoal,
    VisionAreaGoal,
    VisionCountGoal,
    VisionPropertyGoal,
)


WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class RasterImageTests(unittest.TestCase):
    def test_image_is_normalized_and_background_is_inferred(self) -> None:
        image = RasterImage.from_rows(
            (
                (255, 255, 255),
                (255, RED, 255),
                (255, 255, 255),
            )
        )

        self.assertEqual(image.width, 3)
        self.assertEqual(image.height, 3)
        self.assertEqual(image.background, WHITE)
        self.assertEqual(image.rows[1][1], RED)
        self.assertEqual(len(image.digest()), 64)

    def test_ascii_ppm_loads_without_runtime_dependencies(self) -> None:
        content = (
            "P3\n"
            "# tiny red component\n"
            "4 3\n"
            "255\n"
            "255 255 255  255 255 255  255 255 255  255 255 255\n"
            "255 255 255  255 0 0      255 0 0      255 255 255\n"
            "255 255 255  255 255 255  255 255 255  255 255 255\n"
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scene.ppm"
            path.write_text(content, encoding="ascii")
            image = RasterImage.from_pnm(path)
            dispatched = RasterImage.from_file(path)

        analysis = RasterVisionAdapter().analyze(image)
        self.assertEqual(dispatched, image)
        self.assertEqual([item.id for item in analysis.objects], ["red_1"])
        self.assertEqual(analysis.objects[0].area, 2)
        self.assertEqual(image.source, str(path))


class RasterVisionReasoningTests(unittest.TestCase):
    def test_pixels_compile_to_a_transitive_operator_proof(self) -> None:
        problem = RasterVisionProblem(
            _horizontal_three_object_image(),
            (VisionRelationGoal("LEFT_OF", "red", "blue"),),
        )
        instance = RasterVisionAdapter().adapt(problem)
        result = OperatorKernel(instance.registry).solve(
            instance.state, instance.goals
        )

        self.assertEqual(instance.metadata["input_kind"], "raster")
        self.assertEqual(
            [item[1] for item in instance.metadata["selector_resolution"]],
            ["red_1", "blue_1"],
        )
        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(len(result.proof), 1)
        self.assertEqual(
            result.proof[0].action.operator.name,
            "left_of_transitivity",
        )

    def test_centroid_only_relation_remains_proposed(self) -> None:
        image = RasterImage.from_rows(
            (
                [WHITE] * 6,
                [WHITE, RED, RED, RED, WHITE, WHITE],
                [WHITE] * 6,
                [WHITE, WHITE, BLUE, BLUE, BLUE, WHITE],
                [WHITE] * 6,
            )
        )
        problem = RasterVisionProblem(
            image,
            (VisionRelationGoal("LEFT_OF", "red", "blue"),),
        )
        instance = RasterVisionAdapter().adapt(problem)
        result = OperatorKernel(instance.registry).solve(
            instance.state, instance.goals
        )

        left_of = [
            fact
            for fact in instance.state.facts
            if fact.atom.predicate.name == "LEFT_OF"
        ]
        self.assertEqual(len(left_of), 1)
        self.assertEqual(left_of[0].status, FactStatus.PROPOSED)
        proposal = next(
            record
            for record in instance.grounding_trace.records
            if record.candidate.atom == left_of[0].atom
        )
        self.assertEqual(
            proposal.decision.disposition,
            GroundingDisposition.PROPOSED,
        )
        self.assertEqual(
            proposal.decision.authority,
            GroundingAuthority.HEURISTIC_PROPOSAL,
        )
        self.assertFalse(result.success)

    def test_touching_is_verified_and_symmetry_aware(self) -> None:
        image = RasterImage.from_rows(
            (
                [WHITE] * 6,
                [WHITE, RED, RED, BLUE, BLUE, WHITE],
                [WHITE, RED, RED, BLUE, BLUE, WHITE],
                [WHITE] * 6,
            )
        )
        problem = RasterVisionProblem(
            image,
            (VisionRelationGoal("TOUCHING", "blue", "red"),),
        )
        instance = RasterVisionAdapter().adapt(problem)
        result = OperatorKernel(instance.registry).solve(
            instance.state, instance.goals
        )

        self.assertTrue(result.success)
        self.assertEqual(len(result.proof), 0)
        touching = [
            fact
            for fact in instance.state.facts
            if fact.atom.predicate.name == "TOUCHING"
        ]
        self.assertEqual(touching[0].status, FactStatus.OBSERVED)

    def test_color_selector_requires_unique_object(self) -> None:
        image = RasterImage.from_rows(
            (
                [WHITE] * 8,
                [WHITE, RED, RED, WHITE, WHITE, RED, RED, WHITE],
                [WHITE] * 8,
            )
        )
        problem = RasterVisionProblem(
            image,
            (VisionRelationGoal("LEFT_OF", "red", "red_2"),),
        )

        with self.assertRaisesRegex(ValueError, "ambiguous.*red_1, red_2"):
            RasterVisionAdapter().adapt(problem)

    def test_pixel_budget_prevents_accidental_large_allocations(self) -> None:
        image = RasterImage.from_rows(([WHITE] * 4, [WHITE] * 4))
        adapter = RasterVisionAdapter(RasterVisionConfig(max_pixels=7))

        with self.assertRaisesRegex(ValueError, "8 pixels; limit is 7"):
            adapter.analyze(image)

    def test_unified_runtime_dispatches_raster_and_keeps_it_immutable(self) -> None:
        problem = RasterVisionProblem(
            _horizontal_three_object_image(),
            (VisionRelationGoal("LEFT_OF", "red", "blue"),),
        )
        runtime = UnifiedTypedReasoner()

        shadow = runtime.run(TypedDomainRequest("vision", problem, "shadow"))
        typed = runtime.run(TypedDomainRequest("vision", problem, "typed"))

        self.assertTrue(shadow.success)
        self.assertEqual(shadow.instance.metadata["input_kind"], "raster")
        self.assertTrue(typed.success)
        self.assertFalse(typed.projection_applied)
        self.assertIn("immutable source payload", typed.diagnostics[0])

    def test_square_is_composed_from_exact_pixel_properties(self) -> None:
        image = _measured_object_image()
        problem = RasterVisionProblem(
            image,
            (VisionPropertyGoal("SQUARE", "red"),),
        )
        instance = RasterVisionAdapter().adapt(problem)
        result = OperatorKernel(instance.registry).solve(
            instance.state, instance.goals
        )

        self.assertTrue(result.success and result.verified)
        self.assertEqual(
            [step.action.operator.name for step in result.proof],
            ["filled_bbox_is_rectangle", "equal_extent_rectangle_is_square"],
        )

    def test_hollow_or_incomplete_bbox_is_not_a_square(self) -> None:
        image = RasterImage.from_rows(
            (
                [WHITE] * 5,
                [WHITE, RED, RED, WHITE, WHITE],
                [WHITE, RED, WHITE, WHITE, WHITE],
                [WHITE] * 5,
            )
        )
        instance = RasterVisionAdapter().adapt(
            RasterVisionProblem(
                image,
                (VisionPropertyGoal("SQUARE", "red"),),
            )
        )
        result = OperatorKernel(instance.registry).solve(
            instance.state, instance.goals
        )

        self.assertFalse(result.success)
        self.assertFalse(
            any(
                fact.atom.predicate.name == "FILLS_BOUNDING_BOX"
                for fact in instance.state.facts
            )
        )

    def test_closed_component_count_supports_all_color_and_wrong_target(self) -> None:
        image = _measured_object_image()
        valid = RasterVisionAdapter().adapt(
            RasterVisionProblem(
                image,
                (VisionCountGoal("all", 2), VisionCountGoal("blue", 1)),
            )
        )
        valid_result = OperatorKernel(valid.registry).solve(
            valid.state, valid.goals
        )
        wrong = RasterVisionAdapter().adapt(
            RasterVisionProblem(image, (VisionCountGoal("all", 3),))
        )
        wrong_result = OperatorKernel(wrong.registry).solve(
            wrong.state, wrong.goals
        )

        self.assertTrue(valid_result.success and valid_result.verified)
        self.assertEqual(len(valid_result.proof), 2)
        self.assertFalse(wrong_result.success)
        self.assertEqual(
            valid.metadata["raster_reasoning"]["color_counts"],
            (("blue", 1), ("red", 1)),
        )

    def test_exact_pixel_area_comparison_rejects_reverse_order(self) -> None:
        image = _measured_object_image()
        larger = RasterVisionAdapter().adapt(
            RasterVisionProblem(image, (VisionAreaGoal("blue", "red"),))
        )
        larger_result = OperatorKernel(larger.registry).solve(
            larger.state, larger.goals
        )
        reverse = RasterVisionAdapter().adapt(
            RasterVisionProblem(image, (VisionAreaGoal("red", "blue"),))
        )
        reverse_result = OperatorKernel(reverse.registry).solve(
            reverse.state, reverse.goals
        )

        self.assertTrue(larger_result.success and larger_result.verified)
        self.assertEqual(len(larger_result.proof), 1)
        self.assertFalse(reverse_result.success)

    def test_observation_mode_prepares_count_and_area_without_goal_leakage(self) -> None:
        image = RasterImage.from_rows(
            (
                (WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE),
                (WHITE, RED, RED, WHITE, BLUE, BLUE, WHITE),
                (WHITE, RED, RED, WHITE, BLUE, WHITE, WHITE),
                (WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE),
            )
        )
        observed = RasterVisionAdapter().adapt(RasterVisionProblem(image))
        registry = observed.registry
        goals = (
            Goal(
                registry.atom(
                    "OBJECT_COUNT",
                    registry.symbol("all", "Color"),
                    registry.symbol("2", "Number"),
                )
            ),
            Goal(
                registry.atom(
                    "LARGER_AREA",
                    registry.symbol("red_1", "VisualEntity"),
                    registry.symbol("blue_1", "VisualEntity"),
                )
            ),
        )
        instance = replace(observed, goals=goals)

        result = OperatorKernel(registry).solve(instance.state, instance.goals)

        self.assertEqual(observed.goals, ())
        self.assertTrue(result.success and result.verified)
        self.assertEqual(
            observed.metadata["raster_reasoning"]["prepared_count_selectors"],
            ("all", "blue", "red"),
        )
        self.assertIn(
            ("red_1", "blue_1"),
            observed.metadata["raster_reasoning"]["prepared_area_comparisons"],
        )

    def test_component_budget_stops_quadratic_relation_growth(self) -> None:
        image = RasterImage.from_rows(
            ((WHITE, RED, WHITE, BLUE, WHITE, GREEN, WHITE),)
        )
        config = RasterVisionConfig(
            minimum_component_area=1,
            max_components=2,
        )

        with self.assertRaisesRegex(ValueError, "3 components; limit is 2"):
            RasterVisionAdapter(config).adapt(RasterVisionProblem(image))


def _measured_object_image() -> RasterImage:
    return RasterImage.from_rows(
        (
            [WHITE] * 10,
            [WHITE, RED, RED, WHITE, BLUE, BLUE, BLUE, WHITE, WHITE, WHITE],
            [WHITE, RED, RED, WHITE, BLUE, BLUE, BLUE, WHITE, WHITE, WHITE],
            [WHITE] * 10,
        )
    )


def _horizontal_three_object_image() -> RasterImage:
    return RasterImage.from_rows(
        (
            [WHITE] * 11,
            [WHITE, RED, RED, WHITE, GREEN, GREEN, WHITE, BLUE, BLUE, WHITE, WHITE],
            [WHITE, RED, RED, WHITE, GREEN, GREEN, WHITE, BLUE, BLUE, WHITE, WHITE],
            [WHITE] * 11,
        )
    )


if __name__ == "__main__":
    unittest.main()

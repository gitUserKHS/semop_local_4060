from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    AssertionStatus,
    EvidenceStatus,
    Fact,
    FactStatus,
    LanguageTextAdapter,
    LanguageTextProblem,
    OperatorKernel,
    RasterImage,
    RasterVisionAdapter,
    RasterVisionProblem,
    SemanticLabelAuthority,
    SemanticLabelEvidence,
    TaskEvaluation,
    VisionPropertyGoal,
    VisionProblem,
    VisionRelationGoal,
    VisionWorldAdapter,
    WorldState,
    generate_semantic_near_miss_controls,
    generate_symbolic_curriculum,
    parse_linear_equation,
    render_proof_ko,
    summarize_task_evaluations,
)
from semop.kernel.domains import parse_geometry_dsl


class TypedTrustBoundaryTests(unittest.TestCase):
    def test_explicit_language_is_logical_observation_not_external_evidence(self) -> None:
        instance = LanguageTextAdapter().adapt(
            LanguageTextProblem(
                "Goal: deploy\n"
                "Requires: tests, approval\n"
                "Satisfied: tests\n"
                "Satisfied: approval",
                use_legacy_heuristics=False,
            )
        )

        result = OperatorKernel(instance.registry).solve(
            instance.state,
            instance.goals,
        )

        self.assertTrue(result.success and result.verified)
        self.assertFalse(result.conditional)
        self.assertTrue(result.unverified_dependencies)
        self.assertFalse(result.dependencies.evidence_complete)
        self.assertTrue(
            all(
                fact.assertion_status is AssertionStatus.EXPLICIT
                and fact.evidence_status is EvidenceStatus.UNVERIFIED
                for fact in result.observed_dependencies
            )
        )
        self.assertIn("의미 증거 미검증", render_proof_ko(result))

    def test_geometry_proof_reports_its_assumption_dependency(self) -> None:
        instance = parse_geometry_dsl(
            "point A, B, M\n"
            "assume midpoint(M, A, B)\n"
            "prove equal_length(segment(A, M), segment(M, B))\n"
        )

        result = OperatorKernel(instance.registry).solve(
            instance.state,
            instance.goals,
        )

        self.assertTrue(result.success and result.verified)
        self.assertTrue(result.conditional)
        self.assertEqual(len(result.assumption_dependencies), 1)
        dependency = result.assumption_dependencies[0]
        self.assertIs(dependency.status, FactStatus.ASSUMED)
        self.assertIs(dependency.evidence_status, EvidenceStatus.ASSUMED)
        self.assertIn("가정 의존", render_proof_ko(result))
        self.assertTrue(result.to_dict()["trust"]["conditional"])

    def test_exact_math_and_pixel_measurements_are_adapter_verified(self) -> None:
        equation = parse_linear_equation("3*x + 2 = 17")
        equation_result = OperatorKernel(equation.registry).solve(
            equation.state,
            equation.goals,
        )
        image = RasterImage.from_rows(
            (
                ((255, 255, 255),) * 4,
                ((255, 255, 255), (255, 0, 0), (255, 0, 0), (255, 255, 255)),
                ((255, 255, 255), (255, 0, 0), (255, 0, 0), (255, 255, 255)),
                ((255, 255, 255),) * 4,
            )
        )
        vision = RasterVisionAdapter().adapt(
            RasterVisionProblem(
                image,
                goals=(VisionPropertyGoal("SQUARE", "red"),),
            )
        )
        vision_result = OperatorKernel(vision.registry).solve(
            vision.state,
            vision.goals,
        )

        for result in (equation_result, vision_result):
            self.assertTrue(result.success and result.verified)
            self.assertFalse(result.unverified_dependencies)
            self.assertTrue(result.dependencies.evidence_complete)
            self.assertTrue(
                all(
                    fact.evidence_status is EvidenceStatus.ADAPTER_VERIFIED
                    for fact in result.observed_dependencies
                )
            )

    def test_state_digest_is_sensitive_to_provenance(self) -> None:
        instance = parse_linear_equation("x + 1 = 2")
        original = instance.state.facts[0]
        altered = Fact(
            original.atom,
            original.status,
            original.source,
            original.confidence,
            assertion_status=AssertionStatus.IMPORTED,
            evidence_status=EvidenceStatus.UNVERIFIED,
        )

        self.assertNotEqual(
            WorldState((original,)).digest(),
            WorldState((altered,)).digest(),
        )

    def test_structured_vision_verified_marker_is_not_external_evidence(self) -> None:
        world = SimpleNamespace(
            query="Is A left of B?",
            entities=[
                SimpleNamespace(id="A", attributes={"verified": True}),
                SimpleNamespace(id="B", attributes={"verified": True}),
            ],
            relations=[
                SimpleNamespace(
                    source="A",
                    relation="LEFT_OF",
                    target="B",
                    confidence=1.0,
                    attributes={"geometry_verified": True},
                )
            ],
        )
        instance = VisionWorldAdapter().adapt(
            VisionProblem(world, (VisionRelationGoal("LEFT_OF", "A", "B"),))
        )

        result = OperatorKernel(instance.registry).solve(
            instance.state,
            instance.goals,
        )

        self.assertTrue(result.success and result.verified)
        self.assertTrue(result.unverified_dependencies)
        self.assertFalse(result.dependencies.evidence_complete)

    def test_metrics_separate_replay_integrity_from_human_semantics(self) -> None:
        human_evidence = SemanticLabelEvidence(
            case_digest="a" * 64,
            reviewer="human:test-reviewer",
            reviewed_at="2026-07-18T12:00:00Z",
            attestation="human_reviewed_input_and_expected_outcome",
        )
        common = {
            "domain": "language",
            "verified": False,
            "false_positive": False,
            "expansions": 0,
            "proof_steps": 0,
            "inference_rounds": 0,
            "cpu_seconds": 0.0,
            "halt_reason": "fixed-test",
            "policy_used": False,
            "fallback_used": False,
        }
        evaluations = (
            TaskEvaluation(
                task_id="human-correct",
                expected_solved=False,
                success=False,
                label_authority=SemanticLabelAuthority.HUMAN_REVIEWED,
                label_evidence=human_evidence,
                **common,
            ),
            TaskEvaluation(
                task_id="human-wrong",
                expected_solved=True,
                success=False,
                label_authority=SemanticLabelAuthority.HUMAN_REVIEWED,
                label_evidence=human_evidence,
                **common,
            ),
            TaskEvaluation(
                task_id="programmatic-success",
                expected_solved=True,
                success=True,
                verified=True,
                label_authority=SemanticLabelAuthority.PROGRAMMATIC,
                **{key: value for key, value in common.items() if key != "verified"},
            ),
        )

        metrics = summarize_task_evaluations(evaluations)
        payload = metrics.to_dict()

        self.assertEqual(metrics.primitive_replay_integrity, 1.0)
        self.assertEqual(metrics.labeled_outcome_accuracy, 2 / 3)
        self.assertEqual(metrics.programmatic_outcome_accuracy, 1.0)
        self.assertEqual(metrics.programmatic_tasks, 1)
        self.assertEqual(metrics.semantic_correctness, 0.5)
        self.assertEqual(metrics.semantic_gold_tasks, 2)
        self.assertEqual(
            payload["replay_verified_goal_completion"],
            payload["verified_solve_rate"],
        )
        self.assertEqual(
            payload["primitive_replay_integrity"],
            payload["proof_soundness"],
        )

    def test_near_misses_keep_original_goals_and_remove_real_dependencies(self) -> None:
        positives = generate_symbolic_curriculum(
            1,
            seed=2,
            curriculum="language-math-vision",
        )
        controls = generate_semantic_near_miss_controls(positives)

        self.assertEqual({item.domain for item in controls}, {"language", "math", "vision"})
        for positive, control in zip(positives, controls, strict=True):
            self.assertEqual(
                tuple(goal.atom for goal in positive.instance.goals),
                tuple(goal.atom for goal in control.instance.goals),
            )
            self.assertEqual(
                control.instance.metadata["negative_control_kind"],
                "missing_proof_premise",
            )
            self.assertIn("removed_dependency", control.instance.metadata)
            result = OperatorKernel(control.instance.registry).solve(
                control.instance.state,
                control.instance.goals,
            )
            self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()

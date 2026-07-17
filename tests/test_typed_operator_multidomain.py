from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    ArithmeticDslError,
    DomainInstance,
    GoalDirectedPolicy,
    OperatorKernel,
    TypedDomainRequest,
    UnifiedTypedReasoner,
    VisionProblem,
    VisionRelationGoal,
    VisionWorldAdapter,
    parse_arithmetic_expression,
)
from semop.structures import StructuredMeaningGraph
from semop.vlso.types import SharedWorldModel, VLSOEntity, VLSORelation


class TypedArithmeticTests(unittest.TestCase):
    def test_exact_arithmetic_is_compiled_and_replayed(self) -> None:
        problem = parse_arithmetic_expression("(2 + 3) * 4 - 1/2")
        kernel = OperatorKernel(problem.registry)
        result = kernel.solve(problem.state, problem.goals)

        self.assertEqual(problem.domain, "math")
        self.assertEqual(problem.metadata["answer"], "39/2")
        self.assertFalse(
            any(fact.atom.predicate.name == "VALUE" for fact in problem.state.facts)
        )
        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(len(result.proof), problem.metadata["node_count"])
        self.assertTrue(
            all("math" in step.action.operator.tags for step in result.proof)
        )

    def test_decimal_unary_and_precedence_are_exact(self) -> None:
        problem = parse_arithmetic_expression("-.5 + 0.2 * 5")
        result = OperatorKernel(problem.registry).solve(problem.state, problem.goals)

        self.assertEqual(problem.metadata["answer"], "1/2")
        self.assertTrue(result.success)

    def test_division_by_zero_and_syntax_have_locations(self) -> None:
        with self.assertRaises(ArithmeticDslError) as division:
            parse_arithmetic_expression("8 / (3 - 3)")
        self.assertEqual(division.exception.column, 3)

        with self.assertRaises(ArithmeticDslError) as syntax:
            parse_arithmetic_expression("2 + nope")
        self.assertGreater(syntax.exception.column, 1)

    def test_replay_rechecks_arithmetic_guard(self) -> None:
        problem = parse_arithmetic_expression("2 + 3")
        kernel = OperatorKernel(problem.registry)
        result = kernel.solve(problem.state, problem.goals)
        final_guard = result.proof[-1].action.operator.guards[0]
        problem.registry.guards[final_guard] = lambda _binding, _state: False

        replay = kernel.replay(problem.state, problem.goals, result.proof)
        self.assertFalse(replay.verified)
        self.assertIn("guard rejected", replay.diagnostics[0])


class TypedVisionTests(unittest.TestCase):
    def test_only_independently_verified_relations_can_prove(self) -> None:
        proposed = _vision_problem(verified=False)
        proposed_instance = VisionWorldAdapter().adapt(proposed)
        proposed_result = OperatorKernel(proposed_instance.registry).solve(
            proposed_instance.state, proposed_instance.goals
        )

        verified = _vision_problem(verified=True)
        verified_instance = VisionWorldAdapter().adapt(verified)
        verified_result = OperatorKernel(verified_instance.registry).solve(
            verified_instance.state, verified_instance.goals
        )

        self.assertFalse(proposed_result.success)
        self.assertEqual(proposed_instance.metadata["proposed_relation_count"], 2)
        self.assertTrue(verified_result.success)
        self.assertTrue(verified_result.verified)
        self.assertEqual(
            verified_result.proof[-1].action.operator.name,
            "left_of_transitivity",
        )

    def test_inverse_relations_are_canonicalized(self) -> None:
        world = _world(
            [
                VLSORelation(
                    "B",
                    "RIGHT_OF",
                    "A",
                    "vision",
                    0.9,
                    {"geometry_verified": True},
                )
            ]
        )
        problem = VisionProblem(
            world, (VisionRelationGoal("LEFT_OF", "A", "B"),)
        )
        instance = VisionWorldAdapter().adapt(problem)
        result = OperatorKernel(instance.registry).solve(
            instance.state, instance.goals
        )

        self.assertTrue(result.success)
        self.assertEqual(len(result.proof), 0)

    def test_unknown_relation_is_preserved_but_unverified(self) -> None:
        world = _world(
            [
                VLSORelation(
                    "A",
                    "near-ish",
                    "B",
                    "vision",
                    1.0,
                    {"verified": True},
                )
            ]
        )
        problem = VisionProblem(
            world, (VisionRelationGoal("near-ish", "A", "B"),)
        )
        instance = VisionWorldAdapter().adapt(problem)
        result = OperatorKernel(instance.registry).solve(
            instance.state, instance.goals
        )

        self.assertEqual(instance.metadata["unverified_relations"], ("NEAR_ISH",))
        self.assertFalse(result.success)


class UnifiedTypedRuntimeTests(unittest.TestCase):
    def test_one_policy_runs_language_math_and_vision(self) -> None:
        graph = StructuredMeaningGraph(
            query="배포해도 될까?",
            intent="decision",
            hidden_goals=["deploy"],
            required_premises=["tests_pass"],
            satisfied_premises=["tests_pass"],
        )
        requests = (
            TypedDomainRequest("language", graph, "shadow"),
            TypedDomainRequest("math", "6 * (7 - 2)", "shadow"),
            TypedDomainRequest("vision", _vision_problem(verified=True), "shadow"),
        )
        results = UnifiedTypedReasoner().run_many(
            requests, policy=GoalDirectedPolicy()
        )

        self.assertEqual([item.domain.value for item in results], [
            "language",
            "math",
            "vision",
        ])
        self.assertTrue(all(item.success and item.verified for item in results))
        self.assertTrue(all(item.typed_result.policy_used for item in results))
        self.assertTrue(all("검증 완료" in item.proof_ko for item in results))

    def test_modes_control_execution_and_projection(self) -> None:
        runtime = UnifiedTypedReasoner()
        legacy = runtime.run(TypedDomainRequest("math", "2 + 2", "legacy"))
        self.assertIsNone(legacy.typed_result)

        shadow_problem = _vision_problem(verified=True)
        shadow = runtime.run(
            TypedDomainRequest("vision", shadow_problem, "shadow")
        )
        self.assertTrue(shadow.success)
        self.assertFalse(shadow.projection_applied)
        self.assertEqual(shadow_problem.world.inferred_steps, [])

        typed_problem = _vision_problem(verified=True)
        typed = runtime.run(TypedDomainRequest("vision", typed_problem, "typed"))
        self.assertTrue(typed.success)
        self.assertTrue(typed.projection_applied)
        self.assertTrue(typed_problem.world.inferred_steps)
        self.assertTrue(typed_problem.world.metadata["typed_kernel"]["verified"])

    def test_prebuilt_domain_instance_uses_same_runtime(self) -> None:
        problem = parse_arithmetic_expression("9 / 3")
        self.assertIsInstance(problem, DomainInstance)

        result = UnifiedTypedReasoner().run(
            TypedDomainRequest("math", problem, "typed")
        )
        self.assertTrue(result.success)
        self.assertFalse(result.projection_applied)


def _vision_problem(*, verified: bool) -> VisionProblem:
    attributes = {"geometry_verified": True} if verified else {}
    world = _world(
        [
            VLSORelation("A", "LEFT_OF", "B", "vision", 0.99, attributes),
            VLSORelation("B", "LEFT_OF", "C", "vision", 0.99, attributes),
        ]
    )
    return VisionProblem(world, (VisionRelationGoal("LEFT_OF", "A", "C"),))


def _world(relations: list[VLSORelation]) -> SharedWorldModel:
    return SharedWorldModel(
        query="A는 C의 왼쪽인가?",
        entities=[
            VLSOEntity(
                name,
                name,
                "vision",
                "object",
                {"verified": True},
            )
            for name in ("A", "B", "C")
        ],
        relations=relations,
    )


if __name__ == "__main__":
    unittest.main()

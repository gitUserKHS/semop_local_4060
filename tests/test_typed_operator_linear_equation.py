from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    Goal,
    LinearEquationAdapter,
    LinearEquationError,
    LinearEquationProblem,
    MathInputAdapter,
    OperatorKernel,
    TypedDomainRequest,
    UnifiedTypedReasoner,
)


class LinearEquationReasoningTests(unittest.TestCase):
    def test_problem_requires_non_empty_equation_text(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a string"):
            LinearEquationProblem(123)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            LinearEquationProblem("   ")

    def test_exact_equation_is_normalized_solved_and_replayed(self) -> None:
        instance = LinearEquationAdapter().adapt("2*x + 3 = 11")
        kernel = OperatorKernel(instance.registry)
        result = kernel.solve(instance.state, instance.goals)

        self.assertEqual(instance.metadata["answer"], "4")
        self.assertEqual(instance.metadata["numeric_kind"], "exact_rational")
        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(
            [step.action.operator.name for step in result.proof],
            ["normalize_linear_equation", "divide_linear_coefficient"],
        )

    def test_fraction_decimal_implicit_product_and_both_sides_are_exact(self) -> None:
        cases = {
            "x/2 + 1 = 5/2": "3",
            "0.1*x = 0.3": "3",
            "2x + 3 = 11": "4",
            "3*(x - 2) = x + 4": "5",
        }
        for equation, expected in cases.items():
            with self.subTest(equation=equation):
                instance = LinearEquationAdapter().adapt(equation)
                result = OperatorKernel(instance.registry).solve(
                    instance.state, instance.goals
                )
                self.assertEqual(instance.metadata["answer"], expected)
                self.assertTrue(result.success and result.verified)

    def test_wrong_solution_target_is_not_proved(self) -> None:
        instance = LinearEquationAdapter().adapt("2*x + 3 = 11")
        variable = instance.goals[0].atom.arguments[0]
        wrong = instance.registry.symbol("5", "Number")
        wrong_instance = replace(
            instance,
            goals=(Goal(instance.registry.atom("SOLUTION", variable, wrong)),),
        )
        result = OperatorKernel(instance.registry).solve(
            wrong_instance.state,
            wrong_instance.goals,
        )

        self.assertFalse(result.success)

    def test_replay_rechecks_solution_guard(self) -> None:
        instance = LinearEquationAdapter().adapt("2*x + 3 = 11")
        kernel = OperatorKernel(instance.registry)
        result = kernel.solve(instance.state, instance.goals)
        instance.registry.guards["verify_linear_solution"] = lambda _binding, _state: False

        replay = kernel.replay(instance.state, instance.goals, result.proof)
        self.assertFalse(replay.verified)
        self.assertIn("guard rejected", replay.diagnostics[0])

    def test_invalid_equation_classes_are_rejected_with_locations(self) -> None:
        cases = {
            "x*x = 4": "non-linear",
            "x + y = 2": "multiple variables",
            "x = x": "infinitely many",
            "x = x + 1": "no solution",
            "x / 0 = 1": "division by zero",
            "2 + 3 = 5": "requires one variable",
        }
        for equation, message in cases.items():
            with self.subTest(equation=equation):
                with self.assertRaisesRegex(LinearEquationError, message) as error:
                    LinearEquationAdapter().adapt(equation)
                self.assertGreaterEqual(error.exception.column, 1)

    def test_math_runtime_dispatch_preserves_arithmetic_and_adds_equations(self) -> None:
        adapter = MathInputAdapter()
        arithmetic = adapter.adapt("2 + 3")
        equation = adapter.adapt("2*x + 3 = 11")

        self.assertEqual(arithmetic.metadata["answer"], "5")
        self.assertEqual(equation.metadata["input_kind"], "linear_equation")
        runtime = UnifiedTypedReasoner().run(
            TypedDomainRequest("math", "2*x + 3 = 11", "typed")
        )
        self.assertTrue(runtime.success and runtime.verified)
        self.assertFalse(runtime.projection_applied)


if __name__ == "__main__":
    unittest.main()

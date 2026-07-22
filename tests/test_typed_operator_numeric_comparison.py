from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    MathInputAdapter,
    NumericComparisonAdapter,
    NumericComparisonError,
    NumericComparisonProblem,
    OperatorKernel,
    TypedDomainRequest,
    UnifiedTypedReasoner,
)


class NumericComparisonTests(unittest.TestCase):
    def test_exact_nested_comparisons_are_composed_and_replayed(self) -> None:
        cases = (
            "1/3 < 0.34",
            "(2 + 3) * 4 >= 20",
            "0.1 + 0.2 == 0.3",
            "5 != 4",
            "-2 <= -(1 + 1)",
        )
        for expression in cases:
            with self.subTest(expression=expression):
                instance = NumericComparisonAdapter().adapt(expression)
                result = OperatorKernel(instance.registry).solve(
                    instance.state,
                    instance.goals,
                )

                self.assertTrue(result.success and result.verified)
                self.assertTrue(instance.metadata["truth_value"])
                self.assertEqual(
                    result.proof[-1].action.operator.name,
                    "verify_exact_numeric_comparison",
                )
                self.assertTrue(
                    any(
                        str(term).startswith("left::")
                        for term in result.proof[-1].premises[0].arguments
                    )
                )

    def test_false_comparison_remains_unproved(self) -> None:
        instance = NumericComparisonAdapter().adapt("2 * 3 > 7")
        result = OperatorKernel(instance.registry).solve(
            instance.state,
            instance.goals,
        )

        self.assertFalse(instance.metadata["truth_value"])
        self.assertFalse(result.success)
        self.assertEqual(result.proof, ())

    def test_replay_rechecks_numeric_comparison_guard(self) -> None:
        instance = NumericComparisonAdapter().adapt("2 + 3 == 5")
        kernel = OperatorKernel(instance.registry)
        result = kernel.solve(instance.state, instance.goals)
        instance.registry.guards["verify_exact_numeric_comparison"] = (
            lambda _binding, _state: False
        )

        replay = kernel.replay(instance.state, instance.goals, result.proof)

        self.assertFalse(replay.verified)
        self.assertIn("guard rejected", replay.diagnostics[0])

    def test_parser_rejects_ambiguous_or_broken_comparisons_with_locations(self) -> None:
        cases = (
            ("1 < 2 < 3", "exactly one"),
            ("(1 + 2 > 2", "expected"),
            ("> 2", "left expression is empty"),
            ("2 <=", "right expression is empty"),
            ("1 + nope > 2", "invalid left expression"),
        )
        for expression, message in cases:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(NumericComparisonError, message) as error:
                    NumericComparisonAdapter().adapt(expression)
                self.assertGreaterEqual(error.exception.column, 1)

    def test_problem_and_math_runtime_dispatch_are_typed(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a string"):
            NumericComparisonProblem(123)  # type: ignore[arg-type]

        adapter = MathInputAdapter()
        comparison = adapter.adapt("3 >= 2")
        equation = adapter.adapt("2*x = 4")
        arithmetic = adapter.adapt("2 + 3")
        runtime = UnifiedTypedReasoner().run(
            TypedDomainRequest("math", "3 >= 2", "typed")
        )

        self.assertEqual(comparison.metadata["input_kind"], "numeric_comparison")
        self.assertEqual(equation.metadata["input_kind"], "linear_equation")
        self.assertEqual(arithmetic.metadata["answer"], "5")
        self.assertTrue(runtime.success and runtime.verified)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from semop import SemOpAssistant
from semop.kernel import (
    Goal,
    MathInputAdapter,
    OperatorKernel,
    QuadraticEquationAdapter,
    QuadraticEquationError,
    QuadraticEquationProblem,
    TypedDomainRequest,
    UnifiedTypedReasoner,
)
from semop.prompt_api import AnswerStatus
from semop.prompt_compiler import PromptCompiler


class _StaticMathBackend:
    model_id = "test/quadratic-proposer"
    loaded = True

    def __init__(self, expression: str) -> None:
        self.expression = expression
        self.calls = 0

    def generate(self, request, *, operator_hints=(), repair_hint=""):
        self.calls += 1
        return {
            "domain": "math",
            "confidence": 0.9,
            "operator_program": ["DECOMPOSE", "INFER", "VERIFY", "EXPLAIN"],
            "payload": {"expression": self.expression},
            "answer": "검증되지 않은 후보 답",
        }


def _solve(equation: str):
    instance = QuadraticEquationAdapter().adapt(equation)
    result = OperatorKernel(instance.registry).solve(instance.state, instance.goals)
    assert result.success and result.verified
    return instance, result


def test_problem_requires_non_empty_equation_text() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        QuadraticEquationProblem(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be empty"):
        QuadraticEquationProblem("   ")


def test_rational_roots_are_exact_and_replay_verified() -> None:
    instance, result = _solve("x^2 - 5*x + 6 = 0")

    assert instance.metadata["answers"] == ["2", "3"]
    assert instance.metadata["discriminant"] == "1"
    assert instance.metadata["numeric_kind"] == "exact_rational"
    assert [step.action.operator.name for step in result.proof] == [
        "compute_quadratic_discriminant",
        "solve_quadratic_real_roots",
    ]


@pytest.mark.parametrize(
    ("equation", "answers"),
    (
        ("(x - 2)^2 = 0", ["2"]),
        ("x*x - 9 = 0", ["-3", "3"]),
        ("x^2 - 3/2*x + 1/2 = 0", ["1/2", "1"]),
        ("0.5*x^2 - 1 = 0", ["-sqrt(2)", "sqrt(2)"]),
        ("x^2 + x - 1 = 0", ["(-1 - sqrt(5))/2", "(-1 + sqrt(5))/2"]),
    ),
)
def test_parentheses_implicit_products_and_radicals(equation, answers) -> None:
    instance, _result = _solve(equation)
    assert instance.metadata["answers"] == answers


def test_negative_discriminant_proves_empty_real_solution_set() -> None:
    instance, result = _solve("x^2 + 1 = 0")

    assert instance.metadata["answers"] == []
    assert instance.metadata["discriminant"] == "-4"
    assert instance.metadata["numeric_kind"] == "empty_real_solution_set"
    assert result.goals[0].proven is True


def test_integer_root_grid_round_trips_through_coefficients() -> None:
    adapter = QuadraticEquationAdapter()
    for scale in (1, 2, 3):
        for left_root in range(-4, 5):
            for right_root in range(-4, 5):
                linear = -scale * (left_root + right_root)
                constant = scale * left_root * right_root
                equation = (
                    f"{scale}*x^2 + {linear}*x + {constant} = 0"
                )
                instance = adapter.adapt(equation)
                expected = [
                    str(value.numerator)
                    if value.denominator == 1
                    else f"{value.numerator}/{value.denominator}"
                    for value in sorted(
                        {Fraction(left_root), Fraction(right_root)}
                    )
                ]
                assert instance.metadata["answers"] == expected


def test_wrong_solution_set_target_is_not_proved() -> None:
    instance, _result = _solve("x^2 - 5*x + 6 = 0")
    variable = instance.goals[0].atom.arguments[0]
    wrong = instance.registry.symbol("{1;6}", "RealSolutionSet")
    changed = replace(
        instance,
        goals=(Goal(instance.registry.atom("REAL_SOLUTION_SET", variable, wrong)),),
    )

    result = OperatorKernel(instance.registry).solve(changed.state, changed.goals)

    assert result.success is False


def test_replay_rechecks_quadratic_solution_guard() -> None:
    instance, result = _solve("x^2 - 2 = 0")
    kernel = OperatorKernel(instance.registry)
    instance.registry.guards["verify_quadratic_real_roots"] = (
        lambda _bindings, _state: False
    )

    replay = kernel.replay(instance.state, instance.goals, result.proof)

    assert replay.verified is False
    assert "guard rejected" in replay.diagnostics[0]


@pytest.mark.parametrize(
    ("equation", "message"),
    (
        ("x^3 = 1", "only exponent 2"),
        ("x^2 + y = 0", "multiple variables"),
        ("x^2 / (x + 1) = 0", "division by a variable"),
        ("x^2 = x^2", "not quadratic"),
        ("x^2 + = 0", "expected a number"),
    ),
)
def test_unsupported_equation_classes_are_rejected_with_locations(
    equation,
    message,
) -> None:
    with pytest.raises(QuadraticEquationError, match=message) as error:
        QuadraticEquationAdapter().adapt(equation)
    assert error.value.column >= 1


def test_math_dispatch_keeps_linear_and_routes_degree_two() -> None:
    adapter = MathInputAdapter()
    linear = adapter.adapt("2*x + 3 = 11")
    quadratic = adapter.adapt("x^2 - 5*x + 6 = 0")
    runtime = UnifiedTypedReasoner().run(
        TypedDomainRequest("math", "x^2 - 5*x + 6 = 0", "typed")
    )

    assert linear.metadata["input_kind"] == "linear_equation"
    assert quadratic.metadata["input_kind"] == "quadratic_equation"
    assert runtime.success and runtime.verified


def test_prompt_first_assistant_renders_verified_quadratic_answer_in_korean() -> None:
    answer = SemOpAssistant().solve("수학: x^2 - 5*x + 6 = 0")

    assert answer.status is AnswerStatus.VERIFIED
    assert "x = 2" in answer.answer
    assert "x = 3" in answer.answer
    assert answer.provenance["logical_replay_verified"] is True


def test_prompt_accepts_common_unicode_quadratic_notation() -> None:
    answer = SemOpAssistant().solve("수학: x² − 5x + 6 = 0")

    assert answer.status is AnswerStatus.VERIFIED
    assert "x = 2" in answer.answer and "x = 3" in answer.answer


def test_model_proposed_variable_equation_reaches_replay_without_constant_rewrite() -> None:
    backend = _StaticMathBackend("x² - 5x + 6 = 0")
    answer = SemOpAssistant(
        compiler=PromptCompiler(backend=backend)
    ).solve("이 이차방정식을 풀어줘")

    assert backend.calls == 1
    assert answer.status is AnswerStatus.BEST_EFFORT
    assert "x = 2" in answer.answer and "x = 3" in answer.answer
    assert "후보 답" not in answer.answer
    assert answer.provenance["logical_replay_verified"] is True
    assert answer.provenance["answer_source"] == "model_grounded_typed_executor"

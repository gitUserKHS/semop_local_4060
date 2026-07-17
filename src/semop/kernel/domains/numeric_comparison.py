from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..composition import CompositionComponent, compose_domain_instances
from ..model import Fact, FactStatus, Goal, OperatorFamily, Rule, SolveResult, WorldState
from .arithmetic import ArithmeticDslError, ArithmeticExpressionAdapter
from .base import DomainInstance


class NumericComparisonError(ValueError):
    def __init__(self, message: str, column: int) -> None:
        self.message = message
        self.column = column
        super().__init__(f"<numeric-comparison>:1:{column}: {message}")


@dataclass(frozen=True)
class NumericComparisonProblem:
    expression: str

    def __post_init__(self) -> None:
        if not isinstance(self.expression, str):
            raise TypeError("numeric comparison must be a string")
        if not self.expression.strip():
            raise ValueError("numeric comparison cannot be empty")


@dataclass(frozen=True)
class NumericComparisonParse:
    left: str
    operator: str
    right: str
    operator_column: int
    left_column: int
    right_column: int


class NumericComparisonAdapter:
    """Compose two exact arithmetic programs with a replayed order comparison."""

    _OPERATOR_NAMES = {
        "<": "less_than",
        "<=": "less_than_or_equal",
        ">": "greater_than",
        ">=": "greater_than_or_equal",
        "==": "equal_to",
        "!=": "not_equal_to",
    }

    def __init__(
        self,
        *,
        arithmetic: ArithmeticExpressionAdapter | None = None,
        max_characters: int = 4096,
    ) -> None:
        if max_characters <= 0:
            raise ValueError("numeric comparison character limit must be positive")
        self.arithmetic = arithmetic or ArithmeticExpressionAdapter()
        self.max_characters = max_characters

    @staticmethod
    def looks_like(value: str) -> bool:
        return isinstance(value, str) and any(
            token in value for token in ("<", ">", "==", "!=")
        )

    def parse(self, value: str | NumericComparisonProblem) -> NumericComparisonParse:
        problem = (
            value
            if isinstance(value, NumericComparisonProblem)
            else NumericComparisonProblem(value)
        )
        if len(problem.expression) > self.max_characters:
            raise NumericComparisonError(
                f"comparison exceeds the {self.max_characters}-character limit",
                self.max_characters + 1,
            )
        return _split_numeric_comparison(problem.expression)

    def adapt(self, value: str | NumericComparisonProblem) -> DomainInstance:
        problem = (
            value
            if isinstance(value, NumericComparisonProblem)
            else NumericComparisonProblem(value)
        )
        parsed = self.parse(problem)
        try:
            left = self.arithmetic.adapt(parsed.left)
        except ArithmeticDslError as exc:
            raise NumericComparisonError(
                f"invalid left expression: {exc.message}",
                parsed.left_column + exc.column - 1,
            ) from exc
        try:
            right = self.arithmetic.adapt(parsed.right)
        except ArithmeticDslError as exc:
            raise NumericComparisonError(
                f"invalid right expression: {exc.message}",
                parsed.right_column + exc.column - 1,
            ) from exc

        composition = compose_domain_instances(
            (
                CompositionComponent(
                    left,
                    alias="left",
                    include_goals=False,
                    namespace_symbol_types=frozenset({"Expression"}),
                ),
                CompositionComponent(
                    right,
                    alias="right",
                    include_goals=False,
                    namespace_symbol_types=frozenset({"Expression"}),
                ),
            ),
            domain="math",
        )
        registry = composition.instance.registry
        left_import = composition.import_for("left")
        right_import = composition.import_for("right")
        left_root = left_import.map_term(left.goals[0].atom.arguments[0])
        right_root = right_import.map_term(right.goals[0].atom.arguments[0])

        entity = registry.types.resolve("Entity")
        comparison_type = registry.types.ensure("Comparison", entity)
        registry.register_predicate(
            "COMPARISON_REQUEST",
            ("Expression", "Expression", comparison_type),
        )
        registry.register_predicate(
            "COMPARISON_TRUE",
            ("Expression", "Expression", comparison_type),
        )
        comparison_symbol = registry.symbol(
            self._OPERATOR_NAMES[parsed.operator],
            comparison_type,
        )
        request = registry.atom(
            "COMPARISON_REQUEST",
            left_root,
            right_root,
            comparison_symbol,
        )
        result_atom = registry.atom(
            "COMPARISON_TRUE",
            left_root,
            right_root,
            comparison_symbol,
        )
        left_value = registry.variable("left_value", "Number")
        right_value = registry.variable("right_value", "Number")
        registry.register_guard(
            "verify_exact_numeric_comparison",
            _numeric_comparison_guard(parsed.operator),
        )
        registry.register_operator(
            Rule(
                name="verify_exact_numeric_comparison",
                parameters=(left_value, right_value),
                preconditions=(
                    registry.atom("VALUE", left_root, left_value),
                    registry.atom("VALUE", right_root, right_value),
                    request,
                ),
                effects=(result_atom,),
                guards=("verify_exact_numeric_comparison",),
                description_ko=(
                    "정확한 유리수 {left_value}와 {right_value}를 비교해 "
                    f"관계 {parsed.operator}가 참인지 검산한다."
                ),
            ),
            family=OperatorFamily.COMPARE.value,
            tags=("math", "comparison", self._OPERATOR_NAMES[parsed.operator]),
        )

        left_answer = Fraction(str(left.metadata["answer"]))
        right_answer = Fraction(str(right.metadata["answer"]))
        truth_value = _compare(parsed.operator, left_answer, right_answer)
        return DomainInstance(
            registry=registry,
            state=WorldState(
                composition.instance.state.facts
                + (Fact(request, FactStatus.OBSERVED, "math_comparison_parser"),)
            ),
            goals=(Goal(result_atom, label=problem.expression.strip()),),
            domain="math",
            metadata={
                **composition.instance.metadata,
                "source": "exact_numeric_comparison",
                "input_kind": "numeric_comparison",
                "expression": problem.expression.strip(),
                "left_expression": parsed.left,
                "right_expression": parsed.right,
                "comparison_operator": parsed.operator,
                "left_value": str(left.metadata["answer"]),
                "right_value": str(right.metadata["answer"]),
                "truth_value": truth_value,
                "numeric_kind": "exact_rational",
                "operator_depth": max(
                    int(left.metadata["operator_depth"]),
                    int(right.metadata["operator_depth"]),
                )
                + 1,
                "node_count": int(left.metadata["node_count"])
                + int(right.metadata["node_count"])
                + 1,
                "reviewed_examples": 0,
            },
        )

    @staticmethod
    def project(
        _value: str | NumericComparisonProblem,
        _result: SolveResult,
    ) -> bool:
        return False


def parse_numeric_comparison(text: str) -> DomainInstance:
    return NumericComparisonAdapter().adapt(text)


def _split_numeric_comparison(text: str) -> NumericComparisonParse:
    depth = 0
    matches: list[tuple[int, str]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth -= 1
            if depth < 0:
                raise NumericComparisonError("unexpected ')'", index + 1)
            index += 1
            continue
        if depth == 0:
            pair = text[index : index + 2]
            if pair in {"<=", ">=", "==", "!="}:
                matches.append((index, pair))
                index += 2
                continue
            if char in {"<", ">"}:
                matches.append((index, char))
        index += 1
    if depth:
        raise NumericComparisonError("expected ')'", len(text) + 1)
    if not matches:
        raise NumericComparisonError(
            "expected one of <, <=, >, >=, ==, !=",
            1,
        )
    if len(matches) > 1:
        raise NumericComparisonError(
            "comparison must contain exactly one top-level operator",
            matches[1][0] + 1,
        )
    operator_index, operator = matches[0]
    raw_left = text[:operator_index]
    raw_right = text[operator_index + len(operator) :]
    left = raw_left.strip()
    right = raw_right.strip()
    if not left:
        raise NumericComparisonError("left expression is empty", operator_index + 1)
    if not right:
        raise NumericComparisonError(
            "right expression is empty",
            operator_index + len(operator) + 1,
        )
    left_offset = len(raw_left) - len(raw_left.lstrip())
    right_offset = len(raw_right) - len(raw_right.lstrip())
    return NumericComparisonParse(
        left=left,
        operator=operator,
        right=right,
        operator_column=operator_index + 1,
        left_column=left_offset + 1,
        right_column=operator_index + len(operator) + right_offset + 1,
    )


def _numeric_comparison_guard(operator: str):
    def verify(binding, _state: WorldState) -> bool:
        try:
            left = Fraction(str(binding["left_value"]))
            right = Fraction(str(binding["right_value"]))
        except (KeyError, ValueError, ZeroDivisionError):
            return False
        return _compare(operator, left, right)

    return verify


def _compare(operator: str, left: Fraction, right: Fraction) -> bool:
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    raise ValueError(f"unsupported comparison operator: {operator}")

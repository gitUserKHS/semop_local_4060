from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..grounding import (
    GroundingAuthority,
    GroundingDisposition,
    GroundingTrace,
    grounding_payload_digest,
    make_grounding_record,
)
from ..model import (
    AssertionStatus,
    EvidenceStatus,
    Goal,
    OperatorFamily,
    Rule,
    SolveResult,
    WorldState,
)
from ..registry import KernelRegistry
from .arithmetic import ArithmeticExpressionAdapter
from .base import DomainInstance
from .math_common import format_fraction
from .numeric_comparison import NumericComparisonAdapter, NumericComparisonProblem


class LinearEquationError(ValueError):
    def __init__(self, message: str, column: int) -> None:
        self.message = message
        self.column = column
        super().__init__(f"<linear-equation>:1:{column}: {message}")


@dataclass(frozen=True)
class LinearEquationProblem:
    equation: str

    def __post_init__(self) -> None:
        if not isinstance(self.equation, str):
            raise TypeError("linear equation must be a string")
        if not self.equation.strip():
            raise ValueError("linear equation cannot be empty")


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    column: int


@dataclass(frozen=True)
class _LinearForm:
    coefficient: Fraction = Fraction(0)
    constant: Fraction = Fraction(0)
    variable: str | None = None

    def add(self, other: _LinearForm, column: int) -> _LinearForm:
        variable = _merge_variable(self.variable, other.variable, column)
        return _LinearForm(
            self.coefficient + other.coefficient,
            self.constant + other.constant,
            variable,
        )

    def subtract(self, other: _LinearForm, column: int) -> _LinearForm:
        variable = _merge_variable(self.variable, other.variable, column)
        return _LinearForm(
            self.coefficient - other.coefficient,
            self.constant - other.constant,
            variable,
        )

    def multiply(self, other: _LinearForm, column: int) -> _LinearForm:
        variable = _merge_variable(self.variable, other.variable, column)
        if self.coefficient and other.coefficient:
            raise LinearEquationError("non-linear multiplication is not supported", column)
        if self.coefficient:
            return _LinearForm(
                self.coefficient * other.constant,
                self.constant * other.constant,
                variable,
            )
        if other.coefficient:
            return _LinearForm(
                other.coefficient * self.constant,
                other.constant * self.constant,
                variable,
            )
        return _LinearForm(Fraction(0), self.constant * other.constant, variable)

    def divide(self, other: _LinearForm, column: int) -> _LinearForm:
        _merge_variable(self.variable, other.variable, column)
        if other.coefficient:
            raise LinearEquationError("division by a variable expression is not supported", column)
        if other.constant == 0:
            raise LinearEquationError("division by zero", column)
        return _LinearForm(
            self.coefficient / other.constant,
            self.constant / other.constant,
            self.variable,
        )

    def negate(self) -> _LinearForm:
        return _LinearForm(-self.coefficient, -self.constant, self.variable)


class _LinearEquationParser:
    def __init__(self, text: str, *, max_nodes: int) -> None:
        self.tokens = _tokenize(text)
        self.index = 0
        self.max_nodes = max_nodes
        self.nodes = 0

    def parse(self) -> tuple[_LinearForm, _LinearForm, str, int]:
        left = self._expression()
        if self._peek().kind != "EQUAL":
            raise LinearEquationError("expected exactly one '='", self._peek().column)
        self._advance()
        right = self._expression()
        if self._peek().kind != "EOF":
            token = self._peek()
            message = (
                "expected exactly one '='"
                if token.kind == "EQUAL"
                else f"unexpected token {token.value!r}"
            )
            raise LinearEquationError(message, token.column)
        variable = _merge_variable(left.variable, right.variable, 1)
        if variable is None:
            raise LinearEquationError("equation requires one variable", 1)
        return left, right, variable, self.nodes

    def _expression(self) -> _LinearForm:
        left = self._term()
        while self._peek().kind in {"PLUS", "MINUS"}:
            operator = self._advance()
            right = self._term()
            self._count_node(operator.column)
            left = (
                left.add(right, operator.column)
                if operator.kind == "PLUS"
                else left.subtract(right, operator.column)
            )
        return left

    def _term(self) -> _LinearForm:
        left = self._factor()
        while self._peek().kind in {"STAR", "SLASH"}:
            operator = self._advance()
            right = self._factor()
            self._count_node(operator.column)
            left = (
                left.multiply(right, operator.column)
                if operator.kind == "STAR"
                else left.divide(right, operator.column)
            )
        return left

    def _factor(self) -> _LinearForm:
        token = self._peek()
        if token.kind in {"PLUS", "MINUS"}:
            operator = self._advance()
            value = self._factor()
            if operator.kind == "PLUS":
                return value
            self._count_node(operator.column)
            return value.negate()
        if token.kind == "NUMBER":
            self._advance()
            self._count_node(token.column)
            number = Fraction(token.value)
            if self._peek().kind == "IDENTIFIER":
                variable = self._advance()
                self._count_node(variable.column)
                return _LinearForm(number, Fraction(0), variable.value.lower())
            return _LinearForm(Fraction(0), number)
        if token.kind == "IDENTIFIER":
            self._advance()
            self._count_node(token.column)
            return _LinearForm(Fraction(1), Fraction(0), token.value.lower())
        if token.kind == "LPAREN":
            opening = self._advance()
            value = self._expression()
            if self._peek().kind != "RPAREN":
                raise LinearEquationError("expected ')'", opening.column)
            self._advance()
            return value
        if token.kind == "EOF":
            raise LinearEquationError("expected a number, variable, or '('", token.column)
        raise LinearEquationError(
            f"expected a number, variable, or '(', got {token.value!r}",
            token.column,
        )

    def _count_node(self, column: int) -> None:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise LinearEquationError(
                f"equation exceeds the {self.max_nodes}-node limit",
                column,
            )

    def _peek(self) -> _Token:
        return self.tokens[self.index]

    def _advance(self) -> _Token:
        token = self._peek()
        self.index += 1
        return token


class LinearEquationAdapter:
    """Compile one-variable linear equations into exact replayable operators."""

    def __init__(self, *, max_nodes: int = 256, max_characters: int = 4096) -> None:
        if max_nodes <= 0 or max_characters <= 0:
            raise ValueError("linear equation adapter limits must be positive")
        self.max_nodes = max_nodes
        self.max_characters = max_characters

    def adapt(self, value: str | LinearEquationProblem) -> DomainInstance:
        if not isinstance(value, (str, LinearEquationProblem)):
            raise TypeError(
                "linear equation payload must be a string or LinearEquationProblem"
            )
        problem = (
            value
            if isinstance(value, LinearEquationProblem)
            else LinearEquationProblem(value)
        )
        text = problem.equation.strip()
        if len(text) > self.max_characters:
            raise LinearEquationError(
                f"equation exceeds the {self.max_characters}-character limit",
                self.max_characters + 1,
            )
        left, right, variable_name, node_count = _LinearEquationParser(
            text,
            max_nodes=self.max_nodes,
        ).parse()
        coefficient = left.coefficient - right.coefficient
        constant = right.constant - left.constant
        if coefficient == 0:
            message = (
                "equation has infinitely many solutions"
                if constant == 0
                else "equation has no solution"
            )
            raise LinearEquationError(message, text.index("=") + 1)
        solution = constant / coefficient

        registry = _create_linear_registry()
        equation = registry.symbol("equation_0", "Equation")
        variable = registry.symbol(variable_name, "Unknown")
        numbers = {
            value: registry.symbol(format_fraction(value), "Number")
            for value in {
                left.coefficient,
                left.constant,
                right.coefficient,
                right.constant,
                coefficient,
                constant,
                solution,
            }
        }
        equation_fact = registry.atom("EQUATION", equation, variable)
        left_fact = registry.atom(
            "LEFT_LINEAR_FORM",
            equation,
            numbers[left.coefficient],
            numbers[left.constant],
        )
        right_fact = registry.atom(
            "RIGHT_LINEAR_FORM",
            equation,
            numbers[right.coefficient],
            numbers[right.constant],
        )
        normalized = registry.atom(
            "NORMALIZED_LINEAR",
            equation,
            numbers[coefficient],
            numbers[constant],
        )
        solved = registry.atom("SOLUTION", variable, numbers[solution])

        registry.register_guard(
            "verify_linear_normalization",
            _normalization_guard(left, right, coefficient, constant),
        )
        registry.register_operator(
            Rule(
                name="normalize_linear_equation",
                parameters=(),
                preconditions=(equation_fact, left_fact, right_fact),
                effects=(normalized,),
                guards=("verify_linear_normalization",),
                description_ko=(
                    f"양변의 {variable_name} 항과 상수항을 모아 "
                    f"{format_fraction(coefficient)}*{variable_name} = "
                    f"{format_fraction(constant)} 형태로 정리한다."
                ),
            ),
            family=OperatorFamily.TRANSFORM.value,
            tags=("math", "algebra", "normalize"),
        )
        registry.register_guard(
            "verify_linear_solution",
            _solution_guard(coefficient, constant, solution),
        )
        registry.register_operator(
            Rule(
                name="divide_linear_coefficient",
                parameters=(),
                preconditions=(equation_fact, normalized),
                effects=(solved,),
                guards=("verify_linear_solution",),
                description_ko=(
                    f"0이 아닌 계수 {format_fraction(coefficient)}로 나누어 "
                    f"{variable_name} = {format_fraction(solution)}을 얻고 검산한다."
                ),
            ),
            family=OperatorFamily.QUANTIFY.value,
            tags=("math", "algebra", "solve"),
        )
        input_digest = grounding_payload_digest(text)
        grounding_trace = GroundingTrace(
            tuple(
                make_grounding_record(
                    domain="math",
                    statement=str(atom),
                    atom=atom,
                    producer_id="exact_linear_equation_parser",
                    source="linear_equation_parser",
                    disposition=GroundingDisposition.OBSERVED,
                    authority=GroundingAuthority.DETERMINISTIC_ADAPTER,
                    assertion_status=AssertionStatus.EXPLICIT,
                    evidence_status=EvidenceStatus.ADAPTER_VERIFIED,
                    rationale="exact linear parser verified the equation structure",
                    input_digest=input_digest,
                    evidence=(f"input:{input_digest}",),
                )
                for atom in (equation_fact, left_fact, right_fact)
            )
        )
        return DomainInstance(
            registry=registry,
            state=WorldState(grounding_trace.facts),
            goals=(Goal(solved, label=f"{variable_name} = {format_fraction(solution)}"),),
            domain="math",
            metadata={
                "input_kind": "linear_equation",
                "equation": text,
                "variable": variable_name,
                "answer": format_fraction(solution),
                "left_form": (
                    format_fraction(left.coefficient),
                    format_fraction(left.constant),
                ),
                "right_form": (
                    format_fraction(right.coefficient),
                    format_fraction(right.constant),
                ),
                "normalized_form": (
                    format_fraction(coefficient),
                    format_fraction(constant),
                ),
                "numeric_kind": "exact_rational",
                "operator_depth": 2,
                "node_count": node_count,
                "reviewed_examples": 0,
                "grounding": grounding_trace.to_dict(include_records=False),
            },
            grounding_trace=grounding_trace,
        )

    @staticmethod
    def project(_value: str | LinearEquationProblem, _result: SolveResult) -> bool:
        return False


class MathInputAdapter:
    """Dispatch exact arithmetic, comparisons, and one-variable equations."""

    def __init__(
        self,
        *,
        arithmetic: ArithmeticExpressionAdapter | None = None,
        comparison: NumericComparisonAdapter | None = None,
        linear: LinearEquationAdapter | None = None,
    ) -> None:
        self.arithmetic = arithmetic or ArithmeticExpressionAdapter()
        self.comparison = comparison or NumericComparisonAdapter(
            arithmetic=self.arithmetic
        )
        self.linear = linear or LinearEquationAdapter()

    def adapt(
        self,
        value: str | LinearEquationProblem | NumericComparisonProblem,
    ) -> DomainInstance:
        if isinstance(value, NumericComparisonProblem):
            return self.comparison.adapt(value)
        if isinstance(value, LinearEquationProblem):
            return self.linear.adapt(value)
        if not isinstance(value, str):
            raise TypeError(
                "math payload must be a string, NumericComparisonProblem, "
                "or LinearEquationProblem"
            )
        if self.comparison.looks_like(value):
            return self.comparison.adapt(value)
        if "=" in value:
            return self.linear.adapt(value)
        return self.arithmetic.adapt(value)

    @staticmethod
    def project(
        _value: str | LinearEquationProblem | NumericComparisonProblem,
        _result: SolveResult,
    ) -> bool:
        return False


def parse_linear_equation(text: str) -> DomainInstance:
    return LinearEquationAdapter().adapt(text)


def _create_linear_registry() -> KernelRegistry:
    registry = KernelRegistry()
    entity = registry.types.register("Entity")
    equation = registry.types.register("Equation", entity)
    unknown = registry.types.register("Unknown", entity)
    number = registry.types.register("Number", entity)
    registry.register_predicate("EQUATION", (equation, unknown))
    registry.register_predicate("LEFT_LINEAR_FORM", (equation, number, number))
    registry.register_predicate("RIGHT_LINEAR_FORM", (equation, number, number))
    registry.register_predicate("NORMALIZED_LINEAR", (equation, number, number))
    registry.register_predicate("SOLUTION", (unknown, number))
    return registry


def _merge_variable(
    left: str | None,
    right: str | None,
    column: int,
) -> str | None:
    if left is not None and right is not None and left != right:
        raise LinearEquationError("multiple variables are not supported", column)
    return left or right


def _tokenize(text: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    index = 0
    symbols = {
        "+": "PLUS",
        "-": "MINUS",
        "*": "STAR",
        "/": "SLASH",
        "(": "LPAREN",
        ")": "RPAREN",
        "=": "EQUAL",
    }
    while index < len(text):
        char = text[index]
        column = index + 1
        if char.isspace():
            index += 1
            continue
        if char in symbols:
            tokens.append(_Token(symbols[char], char, column))
            index += 1
            continue
        if char.isdigit() or char == ".":
            start = index
            dots = 0
            digits = 0
            while index < len(text) and (text[index].isdigit() or text[index] == "."):
                dots += text[index] == "."
                digits += text[index].isdigit()
                index += 1
            literal = text[start:index]
            if dots > 1 or digits == 0:
                raise LinearEquationError(f"invalid number {literal!r}", column)
            tokens.append(_Token("NUMBER", literal, column))
            continue
        if char.isalpha() or char == "_":
            start = index
            while index < len(text) and (text[index].isalnum() or text[index] == "_"):
                index += 1
            tokens.append(_Token("IDENTIFIER", text[start:index], column))
            continue
        raise LinearEquationError(f"unexpected character {char!r}", column)
    tokens.append(_Token("EOF", "", len(text) + 1))
    return tuple(tokens)


def _normalization_guard(
    left: _LinearForm,
    right: _LinearForm,
    coefficient: Fraction,
    constant: Fraction,
):
    def verify(_binding, _state) -> bool:
        return (
            left.coefficient - right.coefficient == coefficient
            and right.constant - left.constant == constant
        )

    return verify


def _solution_guard(
    coefficient: Fraction,
    constant: Fraction,
    solution: Fraction,
):
    def verify(_binding, _state) -> bool:
        return coefficient != 0 and constant / coefficient == solution

    return verify

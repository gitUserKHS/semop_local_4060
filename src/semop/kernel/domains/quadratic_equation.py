from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt, lcm

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
from .base import DomainInstance
from .math_common import format_fraction


class QuadraticEquationError(ValueError):
    def __init__(self, message: str, column: int) -> None:
        self.message = message
        self.column = column
        super().__init__(f"<quadratic-equation>:1:{column}: {message}")


@dataclass(frozen=True)
class QuadraticEquationProblem:
    equation: str

    def __post_init__(self) -> None:
        if not isinstance(self.equation, str):
            raise TypeError("quadratic equation must be a string")
        if not self.equation.strip():
            raise ValueError("quadratic equation cannot be empty")


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    column: int


@dataclass(frozen=True)
class _Polynomial:
    constant: Fraction = Fraction(0)
    linear: Fraction = Fraction(0)
    quadratic: Fraction = Fraction(0)
    variable: str | None = None

    @property
    def degree(self) -> int:
        if self.quadratic:
            return 2
        if self.linear:
            return 1
        return 0

    def add(self, other: _Polynomial, column: int) -> _Polynomial:
        return _Polynomial(
            self.constant + other.constant,
            self.linear + other.linear,
            self.quadratic + other.quadratic,
            _merge_variable(self.variable, other.variable, column),
        )

    def subtract(self, other: _Polynomial, column: int) -> _Polynomial:
        return _Polynomial(
            self.constant - other.constant,
            self.linear - other.linear,
            self.quadratic - other.quadratic,
            _merge_variable(self.variable, other.variable, column),
        )

    def multiply(self, other: _Polynomial, column: int) -> _Polynomial:
        variable = _merge_variable(self.variable, other.variable, column)
        left = (self.constant, self.linear, self.quadratic)
        right = (other.constant, other.linear, other.quadratic)
        coefficients = [Fraction(0) for _ in range(5)]
        for left_degree, left_value in enumerate(left):
            for right_degree, right_value in enumerate(right):
                coefficients[left_degree + right_degree] += left_value * right_value
        if any(coefficients[degree] for degree in (3, 4)):
            raise QuadraticEquationError(
                "polynomial degree above two is not supported",
                column,
            )
        return _Polynomial(
            coefficients[0],
            coefficients[1],
            coefficients[2],
            variable,
        )

    def divide(self, other: _Polynomial, column: int) -> _Polynomial:
        _merge_variable(self.variable, other.variable, column)
        if other.linear or other.quadratic:
            raise QuadraticEquationError(
                "division by a variable expression is not supported",
                column,
            )
        if other.constant == 0:
            raise QuadraticEquationError("division by zero", column)
        return _Polynomial(
            self.constant / other.constant,
            self.linear / other.constant,
            self.quadratic / other.constant,
            self.variable,
        )

    def negate(self) -> _Polynomial:
        return _Polynomial(
            -self.constant,
            -self.linear,
            -self.quadratic,
            self.variable,
        )


class _QuadraticEquationParser:
    def __init__(self, text: str, *, max_nodes: int) -> None:
        self.tokens = _tokenize(text)
        self.index = 0
        self.max_nodes = max_nodes
        self.nodes = 0

    def parse(self) -> tuple[_Polynomial, str, int]:
        left = self._expression()
        if self._peek().kind != "EQUAL":
            raise QuadraticEquationError("expected exactly one '='", self._peek().column)
        equal = self._advance()
        right = self._expression()
        if self._peek().kind != "EOF":
            token = self._peek()
            message = (
                "expected exactly one '='"
                if token.kind == "EQUAL"
                else f"unexpected token {token.value!r}"
            )
            raise QuadraticEquationError(message, token.column)
        normalized = left.subtract(right, equal.column)
        if normalized.variable is None:
            raise QuadraticEquationError("equation requires one variable", 1)
        if normalized.quadratic == 0:
            raise QuadraticEquationError("equation is not quadratic", equal.column)
        return normalized, normalized.variable, self.nodes

    def _expression(self) -> _Polynomial:
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

    def _term(self) -> _Polynomial:
        left = self._factor()
        while True:
            token = self._peek()
            if token.kind in {"STAR", "SLASH"}:
                operator = self._advance()
                right = self._factor()
                self._count_node(operator.column)
                left = (
                    left.multiply(right, operator.column)
                    if operator.kind == "STAR"
                    else left.divide(right, operator.column)
                )
                continue
            if token.kind in {"NUMBER", "IDENTIFIER", "LPAREN"}:
                right = self._factor()
                self._count_node(token.column)
                left = left.multiply(right, token.column)
                continue
            return left

    def _factor(self) -> _Polynomial:
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
            value = _Polynomial(constant=Fraction(token.value))
        elif token.kind == "IDENTIFIER":
            self._advance()
            self._count_node(token.column)
            value = _Polynomial(linear=Fraction(1), variable=token.value.lower())
        elif token.kind == "LPAREN":
            opening = self._advance()
            value = self._expression()
            if self._peek().kind != "RPAREN":
                raise QuadraticEquationError("expected ')'", opening.column)
            self._advance()
        elif token.kind == "EOF":
            raise QuadraticEquationError(
                "expected a number, variable, or '('",
                token.column,
            )
        else:
            raise QuadraticEquationError(
                f"expected a number, variable, or '(', got {token.value!r}",
                token.column,
            )
        if self._peek().kind == "CARET":
            caret = self._advance()
            exponent = self._peek()
            if exponent.kind != "NUMBER" or Fraction(exponent.value) != 2:
                raise QuadraticEquationError("only exponent 2 is supported", caret.column)
            self._advance()
            self._count_node(caret.column)
            value = value.multiply(value, caret.column)
        return value

    def _count_node(self, column: int) -> None:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise QuadraticEquationError(
                f"equation exceeds the {self.max_nodes}-node limit",
                column,
            )

    def _peek(self) -> _Token:
        return self.tokens[self.index]

    def _advance(self) -> _Token:
        token = self._peek()
        self.index += 1
        return token


@dataclass(frozen=True)
class _RealSolution:
    discriminant: Fraction
    roots: tuple[str, ...]
    numeric_kind: str


class QuadraticEquationAdapter:
    """Compile one-variable degree-two equations into exact replayable operators."""

    def __init__(self, *, max_nodes: int = 256, max_characters: int = 4096) -> None:
        if max_nodes <= 0 or max_characters <= 0:
            raise ValueError("quadratic equation adapter limits must be positive")
        self.max_nodes = max_nodes
        self.max_characters = max_characters

    def looks_like(self, value: str) -> bool:
        if not isinstance(value, str) or "=" not in value:
            return False
        if len(value) > self.max_characters:
            return False
        try:
            _QuadraticEquationParser(value.strip(), max_nodes=self.max_nodes).parse()
        except (QuadraticEquationError, ValueError, ZeroDivisionError):
            return False
        return True

    def adapt(self, value: str | QuadraticEquationProblem) -> DomainInstance:
        if not isinstance(value, (str, QuadraticEquationProblem)):
            raise TypeError(
                "quadratic equation payload must be a string or QuadraticEquationProblem"
            )
        problem = (
            value
            if isinstance(value, QuadraticEquationProblem)
            else QuadraticEquationProblem(value)
        )
        text = problem.equation.strip()
        if len(text) > self.max_characters:
            raise QuadraticEquationError(
                f"equation exceeds the {self.max_characters}-character limit",
                self.max_characters + 1,
            )
        polynomial, variable_name, node_count = _QuadraticEquationParser(
            text,
            max_nodes=self.max_nodes,
        ).parse()
        a = polynomial.quadratic
        b = polynomial.linear
        c = polynomial.constant
        solution = _solve_real_roots(a, b, c)

        registry = _create_quadratic_registry()
        equation = registry.symbol("equation_0", "Equation")
        variable = registry.symbol(variable_name, "Unknown")
        numbers = {
            item: registry.symbol(format_fraction(item), "Number")
            for item in {a, b, c, solution.discriminant}
        }
        solution_name = (
            "empty" if not solution.roots else "{" + ";".join(solution.roots) + "}"
        )
        solution_set = registry.symbol(solution_name, "RealSolutionSet")
        equation_fact = registry.atom("QUADRATIC_EQUATION", equation, variable)
        coefficients_fact = registry.atom(
            "QUADRATIC_COEFFICIENTS",
            equation,
            numbers[a],
            numbers[b],
            numbers[c],
        )
        discriminant_fact = registry.atom(
            "DISCRIMINANT",
            equation,
            numbers[solution.discriminant],
        )
        roots_fact = registry.atom("REAL_SOLUTION_SET", variable, solution_set)

        registry.register_guard(
            "verify_quadratic_discriminant",
            _discriminant_guard(a, b, c, solution.discriminant),
        )
        registry.register_operator(
            Rule(
                name="compute_quadratic_discriminant",
                parameters=(),
                preconditions=(equation_fact, coefficients_fact),
                effects=(discriminant_fact,),
                guards=("verify_quadratic_discriminant",),
                description_ko=(
                    f"계수 a={format_fraction(a)}, b={format_fraction(b)}, "
                    f"c={format_fraction(c)}에서 판별식 b^2-4ac = "
                    f"{format_fraction(solution.discriminant)}을 계산한다."
                ),
            ),
            family=OperatorFamily.TRANSFORM.value,
            tags=("math", "algebra", "quadratic", "discriminant"),
        )
        registry.register_guard(
            "verify_quadratic_real_roots",
            _solution_guard(a, b, c, solution),
        )
        roots_description = (
            f"근의 공식으로 실수해 {', '.join(solution.roots)}을 얻고 계수로 재검산한다."
            if solution.roots
            else "판별식이 음수이므로 실수해가 없음을 확인한다."
        )
        registry.register_operator(
            Rule(
                name="solve_quadratic_real_roots",
                parameters=(),
                preconditions=(equation_fact, coefficients_fact, discriminant_fact),
                effects=(roots_fact,),
                guards=("verify_quadratic_real_roots",),
                description_ko=roots_description,
            ),
            family=OperatorFamily.QUANTIFY.value,
            tags=("math", "algebra", "quadratic", "solve"),
        )

        input_digest = grounding_payload_digest(text)
        grounding_trace = GroundingTrace(
            tuple(
                make_grounding_record(
                    domain="math",
                    statement=str(atom),
                    atom=atom,
                    producer_id="exact_quadratic_equation_parser",
                    source="quadratic_equation_parser",
                    disposition=GroundingDisposition.OBSERVED,
                    authority=GroundingAuthority.DETERMINISTIC_ADAPTER,
                    assertion_status=AssertionStatus.EXPLICIT,
                    evidence_status=EvidenceStatus.ADAPTER_VERIFIED,
                    rationale="exact degree-two parser verified the equation coefficients",
                    input_digest=input_digest,
                    evidence=(f"input:{input_digest}",),
                    sensor_features=(
                        ("parser.exact", 1.0),
                        ("equation.quadratic_form", 1.0),
                        ("expression.atom_arity", len(atom.arguments) / 5.0),
                    ),
                )
                for atom in (equation_fact, coefficients_fact)
            )
        )
        return DomainInstance(
            registry=registry,
            state=WorldState(grounding_trace.facts),
            goals=(Goal(roots_fact, label=f"{variable_name}의 실수해 집합"),),
            domain="math",
            metadata={
                "input_kind": "quadratic_equation",
                "equation": text,
                "variable": variable_name,
                "answer": ", ".join(solution.roots),
                "answers": list(solution.roots),
                "coefficients": tuple(format_fraction(item) for item in (a, b, c)),
                "discriminant": format_fraction(solution.discriminant),
                "numeric_kind": solution.numeric_kind,
                "operator_depth": 2,
                "node_count": node_count,
                "reviewed_examples": 0,
                "grounding": grounding_trace.to_dict(include_records=False),
            },
            grounding_trace=grounding_trace,
        )

    @staticmethod
    def project(_value: str | QuadraticEquationProblem, _result: SolveResult) -> bool:
        return False


def parse_quadratic_equation(text: str) -> DomainInstance:
    return QuadraticEquationAdapter().adapt(text)


def _create_quadratic_registry() -> KernelRegistry:
    registry = KernelRegistry()
    entity = registry.types.register("Entity")
    equation = registry.types.register("Equation", entity)
    unknown = registry.types.register("Unknown", entity)
    number = registry.types.register("Number", entity)
    solution_set = registry.types.register("RealSolutionSet", entity)
    registry.register_predicate("QUADRATIC_EQUATION", (equation, unknown))
    registry.register_predicate(
        "QUADRATIC_COEFFICIENTS",
        (equation, number, number, number),
    )
    registry.register_predicate("DISCRIMINANT", (equation, number))
    registry.register_predicate("REAL_SOLUTION_SET", (unknown, solution_set))
    return registry


def _discriminant_guard(
    a: Fraction,
    b: Fraction,
    c: Fraction,
    expected: Fraction,
):
    def guard(_bindings, _state) -> bool:
        return a != 0 and b * b - 4 * a * c == expected

    return guard


def _solution_guard(
    a: Fraction,
    b: Fraction,
    c: Fraction,
    expected: _RealSolution,
):
    def guard(_bindings, _state) -> bool:
        return _solve_real_roots(a, b, c) == expected

    return guard


def _solve_real_roots(a: Fraction, b: Fraction, c: Fraction) -> _RealSolution:
    if a == 0:
        raise ValueError("quadratic coefficient cannot be zero")
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return _RealSolution(discriminant, (), "empty_real_solution_set")
    square_root = _perfect_fraction_sqrt(discriminant)
    if square_root is not None:
        roots = tuple(
            sorted(
                {
                    (-b - square_root) / (2 * a),
                    (-b + square_root) / (2 * a),
                }
            )
        )
        return _RealSolution(
            discriminant,
            tuple(format_fraction(root) for root in roots),
            "exact_rational",
        )
    return _RealSolution(
        discriminant,
        _symbolic_radical_roots(a, b, c),
        "exact_real_algebraic",
    )


def _perfect_fraction_sqrt(value: Fraction) -> Fraction | None:
    if value < 0:
        return None
    numerator = isqrt(value.numerator)
    denominator = isqrt(value.denominator)
    if numerator * numerator != value.numerator:
        return None
    if denominator * denominator != value.denominator:
        return None
    return Fraction(numerator, denominator)


def _symbolic_radical_roots(
    a: Fraction,
    b: Fraction,
    c: Fraction,
) -> tuple[str, str]:
    denominator_lcm = lcm(a.denominator, b.denominator, c.denominator)
    integers = [
        int(value * denominator_lcm)
        for value in (a, b, c)
    ]
    common = gcd(gcd(abs(integers[0]), abs(integers[1])), abs(integers[2]))
    if common:
        integers = [value // common for value in integers]
    if integers[0] < 0:
        integers = [-value for value in integers]
    integer_a, integer_b, integer_c = integers
    discriminant = integer_b * integer_b - 4 * integer_a * integer_c
    outside, inside = _squarefree_parts(discriminant)
    base = -integer_b
    denominator = 2 * integer_a
    reducible = gcd(gcd(abs(base), outside), denominator)
    if reducible > 1:
        base //= reducible
        outside //= reducible
        denominator //= reducible
    radical = "sqrt(" + str(inside) + ")"
    if outside != 1:
        radical = f"{outside}*{radical}"
    return (
        _format_radical_root(base, radical, denominator, negative=True),
        _format_radical_root(base, radical, denominator, negative=False),
    )


def _squarefree_parts(value: int) -> tuple[int, int]:
    if value <= 0:
        raise ValueError("radical discriminant must be positive")
    outside = 1
    inside = value
    factor = 2
    while factor * factor <= inside:
        square = factor * factor
        while inside % square == 0:
            outside *= factor
            inside //= square
        factor += 1
    return outside, inside


def _format_radical_root(
    base: int,
    radical: str,
    denominator: int,
    *,
    negative: bool,
) -> str:
    if base == 0:
        numerator = f"-{radical}" if negative else radical
    else:
        operator = "-" if negative else "+"
        numerator = f"{base} {operator} {radical}"
    if denominator == 1:
        return numerator
    return f"({numerator})/{denominator}"


def _merge_variable(
    left: str | None,
    right: str | None,
    column: int,
) -> str | None:
    if left is not None and right is not None and left != right:
        raise QuadraticEquationError("multiple variables are not supported", column)
    return left or right


def _tokenize(text: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    index = 0
    symbols = {
        "+": "PLUS",
        "-": "MINUS",
        "*": "STAR",
        "/": "SLASH",
        "^": "CARET",
        "(": "LPAREN",
        ")": "RPAREN",
        "=": "EQUAL",
    }
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        column = index + 1
        if char in symbols:
            tokens.append(_Token(symbols[char], char, column))
            index += 1
            continue
        if char.isdigit() or (char == "." and index + 1 < len(text) and text[index + 1].isdigit()):
            start = index
            dots = 0
            while index < len(text) and (text[index].isdigit() or text[index] == "."):
                dots += int(text[index] == ".")
                if dots > 1:
                    raise QuadraticEquationError("invalid number", index + 1)
                index += 1
            tokens.append(_Token("NUMBER", text[start:index], column))
            continue
        if char.isalpha():
            start = index
            while index < len(text) and text[index].isalpha():
                index += 1
            tokens.append(_Token("IDENTIFIER", text[start:index], column))
            continue
        raise QuadraticEquationError(f"unsupported character {char!r}", column)
    tokens.append(_Token("EOF", "", len(text) + 1))
    return tuple(tokens)


__all__ = [
    "QuadraticEquationAdapter",
    "QuadraticEquationError",
    "QuadraticEquationProblem",
    "parse_quadratic_equation",
]

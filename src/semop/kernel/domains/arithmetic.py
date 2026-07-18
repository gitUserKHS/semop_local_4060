from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..model import (
    AssertionStatus,
    EvidenceStatus,
    Fact,
    FactStatus,
    Goal,
    OperatorFamily,
    Rule,
    WorldState,
)
from ..registry import KernelRegistry
from .base import DomainInstance
from .math_common import format_fraction as _format_fraction


class ArithmeticDslError(ValueError):
    def __init__(self, message: str, column: int) -> None:
        self.message = message
        self.column = column
        super().__init__(f"<expression>:1:{column}: {message}")


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    column: int


@dataclass(frozen=True)
class _Expression:
    operation: str
    column: int
    literal: str = ""
    children: tuple["_Expression", ...] = ()


class _ArithmeticParser:
    def __init__(self, text: str, *, max_nodes: int) -> None:
        self.tokens = _tokenize(text)
        self.index = 0
        self.max_nodes = max_nodes
        self.nodes = 0

    def parse(self) -> _Expression:
        expression = self._expression()
        token = self._peek()
        if token.kind != "EOF":
            raise ArithmeticDslError(
                f"unexpected token {token.value!r}", token.column
            )
        return expression

    def _expression(self) -> _Expression:
        left = self._term()
        while self._peek().kind in {"PLUS", "MINUS"}:
            operator = self._advance()
            right = self._term()
            left = self._node(
                "add" if operator.kind == "PLUS" else "sub",
                operator.column,
                children=(left, right),
            )
        return left

    def _term(self) -> _Expression:
        left = self._factor()
        while self._peek().kind in {"STAR", "SLASH"}:
            operator = self._advance()
            right = self._factor()
            left = self._node(
                "mul" if operator.kind == "STAR" else "div",
                operator.column,
                children=(left, right),
            )
        return left

    def _factor(self) -> _Expression:
        token = self._peek()
        if token.kind in {"PLUS", "MINUS"}:
            operator = self._advance()
            child = self._factor()
            if operator.kind == "PLUS":
                return child
            return self._node("neg", operator.column, children=(child,))
        if token.kind == "NUMBER":
            self._advance()
            return self._node("literal", token.column, literal=token.value)
        if token.kind == "LPAREN":
            opening = self._advance()
            expression = self._expression()
            if self._peek().kind != "RPAREN":
                raise ArithmeticDslError("expected ')'", opening.column)
            self._advance()
            return expression
        if token.kind == "EOF":
            raise ArithmeticDslError("expected a number or '('", token.column)
        raise ArithmeticDslError(
            f"expected a number or '(', got {token.value!r}", token.column
        )

    def _node(
        self,
        operation: str,
        column: int,
        *,
        literal: str = "",
        children: tuple[_Expression, ...] = (),
    ) -> _Expression:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise ArithmeticDslError(
                f"expression exceeds the {self.max_nodes}-node limit", column
            )
        return _Expression(operation, column, literal, children)

    def _peek(self) -> _Token:
        return self.tokens[self.index]

    def _advance(self) -> _Token:
        token = self._peek()
        self.index += 1
        return token


class ArithmeticExpressionAdapter:
    """Compile exact rational arithmetic into replayable typed operators."""

    def __init__(self, *, max_nodes: int = 256, max_characters: int = 4096) -> None:
        if max_nodes <= 0 or max_characters <= 0:
            raise ValueError("arithmetic adapter limits must be positive")
        self.max_nodes = max_nodes
        self.max_characters = max_characters

    def adapt(self, value: str) -> DomainInstance:
        if not isinstance(value, str):
            raise TypeError("math payload must be an arithmetic expression string")
        if len(value) > self.max_characters:
            raise ArithmeticDslError(
                f"expression exceeds the {self.max_characters}-character limit",
                self.max_characters + 1,
            )
        expression = _ArithmeticParser(value, max_nodes=self.max_nodes).parse()
        registry = _create_arithmetic_registry()
        facts: list[Fact] = []
        number_symbols = {}
        node_count = 0
        max_depth = 0

        def number_symbol(number: Fraction):
            name = _format_fraction(number)
            if name not in number_symbols:
                number_symbols[name] = registry.symbol(name, "Number")
            return number_symbols[name]

        def compile_node(node: _Expression) -> tuple[object, Fraction, int]:
            nonlocal node_count, max_depth
            compiled_children = tuple(compile_node(child) for child in node.children)
            node_id = node_count
            node_count += 1
            expression_symbol = registry.symbol(f"expr_{node_id}", "Expression")

            if node.operation == "literal":
                numeric_value = Fraction(node.literal)
                numeric_symbol = number_symbol(numeric_value)
                structure = registry.atom("LITERAL", expression_symbol, numeric_symbol)
                facts.append(
                    Fact(
                        structure,
                        FactStatus.OBSERVED,
                        "math_parser",
                        assertion_status=AssertionStatus.EXPLICIT,
                        evidence_status=EvidenceStatus.ADAPTER_VERIFIED,
                    )
                )
                guard_name = f"verify_literal_{node_id:03d}"
                registry.register_guard(
                    guard_name,
                    _literal_guard(node.literal, numeric_symbol.name),
                )
                registry.register_operator(
                    Rule(
                        name=f"evaluate_literal_{node_id:03d}",
                        parameters=(),
                        preconditions=(structure,),
                        effects=(
                            registry.atom("VALUE", expression_symbol, numeric_symbol),
                        ),
                        guards=(guard_name,),
                        description_ko=(
                            f"리터럴 {node.literal}: 정확한 유리수 값은 "
                            f"{numeric_symbol.name}이다."
                        ),
                    ),
                    family=OperatorFamily.QUANTIFY.value,
                    tags=("math", "arithmetic", "literal"),
                )
                depth = 1
            elif node.operation == "neg":
                child_symbol, child_value, child_depth = compiled_children[0]
                numeric_value = -child_value
                numeric_symbol = number_symbol(numeric_value)
                structure = registry.atom("NEG_NODE", expression_symbol, child_symbol)
                facts.append(
                    Fact(
                        structure,
                        FactStatus.OBSERVED,
                        "math_parser",
                        assertion_status=AssertionStatus.EXPLICIT,
                        evidence_status=EvidenceStatus.ADAPTER_VERIFIED,
                    )
                )
                _register_ground_evaluation(
                    registry,
                    node_id=node_id,
                    operation=node.operation,
                    structure=structure,
                    expression_symbol=expression_symbol,
                    child_symbols=(child_symbol,),
                    child_values=(child_value,),
                    result_symbol=numeric_symbol,
                    result_value=numeric_value,
                )
                depth = child_depth + 1
            else:
                left_symbol, left_value, left_depth = compiled_children[0]
                right_symbol, right_value, right_depth = compiled_children[1]
                numeric_value = _evaluate_binary(
                    node.operation, left_value, right_value, node.column
                )
                numeric_symbol = number_symbol(numeric_value)
                structure = registry.atom(
                    f"{node.operation.upper()}_NODE",
                    expression_symbol,
                    left_symbol,
                    right_symbol,
                )
                facts.append(
                    Fact(
                        structure,
                        FactStatus.OBSERVED,
                        "math_parser",
                        assertion_status=AssertionStatus.EXPLICIT,
                        evidence_status=EvidenceStatus.ADAPTER_VERIFIED,
                    )
                )
                _register_ground_evaluation(
                    registry,
                    node_id=node_id,
                    operation=node.operation,
                    structure=structure,
                    expression_symbol=expression_symbol,
                    child_symbols=(left_symbol, right_symbol),
                    child_values=(left_value, right_value),
                    result_symbol=numeric_symbol,
                    result_value=numeric_value,
                )
                depth = max(left_depth, right_depth) + 1
            max_depth = max(max_depth, depth)
            return expression_symbol, numeric_value, depth

        root_symbol, answer, _ = compile_node(expression)
        answer_symbol = number_symbol(answer)
        return DomainInstance(
            registry=registry,
            state=WorldState(tuple(facts)),
            goals=(
                Goal(
                    registry.atom("VALUE", root_symbol, answer_symbol),
                    label=f"{value.strip()} = {_format_fraction(answer)}",
                ),
            ),
            domain="math",
            metadata={
                "expression": value.strip(),
                "answer": _format_fraction(answer),
                "numeric_kind": "exact_rational",
                "operator_depth": max_depth,
                "node_count": node_count,
                "reviewed_examples": 0,
            },
        )


def parse_arithmetic_expression(text: str) -> DomainInstance:
    return ArithmeticExpressionAdapter().adapt(text)


def _create_arithmetic_registry() -> KernelRegistry:
    registry = KernelRegistry()
    entity = registry.types.register("Entity")
    expression = registry.types.register("Expression", entity)
    number = registry.types.register("Number", entity)
    registry.register_predicate("LITERAL", (expression, number))
    registry.register_predicate("NEG_NODE", (expression, expression))
    for operation in ("ADD", "SUB", "MUL", "DIV"):
        registry.register_predicate(
            f"{operation}_NODE", (expression, expression, expression)
        )
    registry.register_predicate("VALUE", (expression, number))
    return registry


def _register_ground_evaluation(
    registry: KernelRegistry,
    *,
    node_id: int,
    operation: str,
    structure,
    expression_symbol,
    child_symbols: tuple[object, ...],
    child_values: tuple[Fraction, ...],
    result_symbol,
    result_value: Fraction,
) -> None:
    guard_name = f"verify_{operation}_{node_id:03d}"
    registry.register_guard(
        guard_name,
        _operation_guard(operation, child_values, result_symbol.name),
    )
    preconditions = [structure]
    preconditions.extend(
        registry.atom("VALUE", child_symbol, registry.symbol(_format_fraction(value), "Number"))
        for child_symbol, value in zip(child_symbols, child_values, strict=True)
    )
    if operation == "neg":
        equation = f"-({_format_fraction(child_values[0])})"
    else:
        symbols = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
        equation = (
            f"{_format_fraction(child_values[0])} {symbols[operation]} "
            f"{_format_fraction(child_values[1])}"
        )
    registry.register_operator(
        Rule(
            name=f"evaluate_{operation}_{node_id:03d}",
            parameters=(),
            preconditions=tuple(preconditions),
            effects=(registry.atom("VALUE", expression_symbol, result_symbol),),
            guards=(guard_name,),
            description_ko=(
                f"{equation}의 결과가 {_format_fraction(result_value)}임을 "
                "정확히 계산하고 검산한다."
            ),
        ),
        family=OperatorFamily.QUANTIFY.value,
        tags=("math", "arithmetic", operation),
    )


def _tokenize(text: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    index = 0
    while index < len(text):
        char = text[index]
        column = index + 1
        if char.isspace():
            index += 1
            continue
        token_kinds = {
            "+": "PLUS",
            "-": "MINUS",
            "*": "STAR",
            "/": "SLASH",
            "(": "LPAREN",
            ")": "RPAREN",
        }
        if char in token_kinds:
            tokens.append(_Token(token_kinds[char], char, column))
            index += 1
            continue
        if char.isdigit() or char == ".":
            start = index
            dots = 0
            digits = 0
            while index < len(text) and (
                text[index].isdigit() or text[index] == "."
            ):
                if text[index] == ".":
                    dots += 1
                else:
                    digits += 1
                index += 1
            literal = text[start:index]
            if dots > 1 or digits == 0:
                raise ArithmeticDslError(f"invalid number {literal!r}", column)
            tokens.append(_Token("NUMBER", literal, column))
            continue
        raise ArithmeticDslError(f"unexpected character {char!r}", column)
    tokens.append(_Token("EOF", "", len(text) + 1))
    return tuple(tokens)


def _evaluate_binary(
    operation: str, left: Fraction, right: Fraction, column: int
) -> Fraction:
    if operation == "add":
        return left + right
    if operation == "sub":
        return left - right
    if operation == "mul":
        return left * right
    if right == 0:
        raise ArithmeticDslError("division by zero", column)
    return left / right


def _literal_guard(literal: str, result_name: str):
    def verify(_binding, _state) -> bool:
        return Fraction(literal) == Fraction(result_name)

    return verify

def _operation_guard(
    operation: str,
    child_values: tuple[Fraction, ...],
    result_name: str,
):
    def verify(_binding, _state) -> bool:
        if operation == "neg":
            expected = -child_values[0]
        else:
            expected = _evaluate_binary(operation, *child_values, column=1)
        return expected == Fraction(result_name)

    return verify

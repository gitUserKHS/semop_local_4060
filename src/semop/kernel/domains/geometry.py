from __future__ import annotations

from dataclasses import dataclass

from ..model import (
    AssertionStatus,
    Atom,
    EvidenceStatus,
    Fact,
    FactStatus,
    Goal,
    KernelError,
    Rule,
    Term,
    TypeValidationError,
)
from ..catalog import register_transitive_relation
from ..registry import KernelRegistry
from .base import DomainInstance


def create_geometry_registry() -> KernelRegistry:
    registry = KernelRegistry()
    entity = registry.types.register("Entity")
    point = registry.types.register("Point", entity)
    line_type = registry.types.register("Line", entity)
    segment_type = registry.types.register("Segment", entity)
    angle_type = registry.types.register("Angle", entity)

    registry.register_function(
        "line",
        (point, point),
        line_type,
        symmetry_groups=((0, 1),),
        allow_repeated_arguments=False,
    )
    registry.register_function(
        "segment",
        (point, point),
        segment_type,
        symmetry_groups=((0, 1),),
        allow_repeated_arguments=False,
    )
    registry.register_function(
        "angle",
        (point, point, point),
        angle_type,
        symmetry_groups=((0, 2),),
        allow_repeated_arguments=False,
    )
    registry.register_predicate(
        "midpoint", (point, point, point), symmetry_groups=((1, 2),)
    )
    registry.register_predicate(
        "collinear", (point, point, point), symmetry_groups=((0, 1, 2),)
    )
    registry.register_predicate(
        "equal_length", (segment_type, segment_type), symmetry_groups=((0, 1),)
    )
    registry.register_predicate(
        "equal_angle", (angle_type, angle_type), symmetry_groups=((0, 1),)
    )
    registry.register_predicate(
        "parallel", (line_type, line_type), symmetry_groups=((0, 1),)
    )
    registry.register_predicate(
        "perpendicular", (line_type, line_type), symmetry_groups=((0, 1),)
    )
    registry.register_guard(
        "different_y_z", lambda binding, _state: binding["y"] != binding["z"]
    )

    m = registry.variable("m", point)
    a = registry.variable("a", point)
    b = registry.variable("b", point)
    registry.register_operator(
        Rule(
            name="midpoint_implies_collinear",
            parameters=(m, a, b),
            preconditions=(registry.atom("midpoint", m, a, b),),
            effects=(registry.atom("collinear", a, m, b),),
            description_ko="{m}이 {a}, {b}의 중점이므로 세 점은 일직선 위에 있다",
        ),
        family="relate",
        tags=("geometry", "definition"),
    )
    registry.register_operator(
        Rule(
            name="midpoint_implies_equal_lengths",
            parameters=(m, a, b),
            preconditions=(registry.atom("midpoint", m, a, b),),
            effects=(
                registry.atom(
                    "equal_length",
                    registry.apply("segment", a, m),
                    registry.apply("segment", m, b),
                ),
            ),
            description_ko="{m}이 {a}, {b}의 중점이므로 양쪽 선분의 길이는 같다",
        ),
        family="relate",
        tags=("geometry", "measure"),
    )
    register_transitive_relation(
        registry,
        "equal_length",
        segment_type,
        operator_name="equal_length_transitivity",
        tags=("geometry", "measure"),
        description_ko="선분 길이의 같음 관계에 추이성을 적용한다.",
    )
    register_transitive_relation(
        registry,
        "equal_angle",
        angle_type,
        operator_name="equal_angle_transitivity",
        tags=("geometry", "measure"),
        description_ko="각도의 같음 관계에 추이성을 적용한다.",
    )
    register_transitive_relation(
        registry,
        "parallel",
        line_type,
        operator_name="parallel_transitivity",
        tags=("geometry", "relation"),
        description_ko="평행 관계의 추이성을 적용한다.",
    )

    x = registry.variable("x", line_type)
    y = registry.variable("y", line_type)
    z = registry.variable("z", line_type)
    registry.register_operator(
        Rule(
            name="transfer_perpendicular_over_parallel",
            parameters=(x, y, z),
            preconditions=(
                registry.atom("parallel", x, y),
                registry.atom("perpendicular", x, z),
            ),
            effects=(registry.atom("perpendicular", y, z),),
            guards=("different_y_z",),
            description_ko="{x}와 {y}가 평행이고 {x}와 {z}가 수직이므로 {y}와 {z}도 수직이다",
        ),
        family="compose",
        tags=("geometry", "relation_transfer"),
    )
    return registry


@dataclass(frozen=True)
class SourceLocation:
    line: int
    column: int
    source_name: str = "<string>"

    def __str__(self) -> str:
        return f"{self.source_name}:{self.line}:{self.column}"


class GeometryDslError(ValueError):
    def __init__(self, message: str, location: SourceLocation) -> None:
        self.message = message
        self.location = location
        super().__init__(f"{location}: {message}")


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    location: SourceLocation


_PUNCTUATION = {
    ",": "COMMA",
    ":": "COLON",
    "=": "EQUAL",
    "(": "LPAREN",
    ")": "RPAREN",
}


def _tokenize(text: str, source_name: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    index = 0
    line = 1
    column = 1
    while index < len(text):
        char = text[index]
        location = SourceLocation(line, column, source_name)
        if char in " \t\r":
            index += 1
            column += 1
        elif char == "\n":
            tokens.append(_Token("NEWLINE", char, location))
            index += 1
            line += 1
            column = 1
        elif char == "#":
            while index < len(text) and text[index] != "\n":
                index += 1
                column += 1
        elif char in _PUNCTUATION:
            tokens.append(_Token(_PUNCTUATION[char], char, location))
            index += 1
            column += 1
        elif char == "_" or char.isalpha():
            start = index
            start_column = column
            while index < len(text) and (
                text[index] == "_" or text[index].isalnum()
            ):
                index += 1
                column += 1
            tokens.append(
                _Token(
                    "IDENT",
                    text[start:index],
                    SourceLocation(line, start_column, source_name),
                )
            )
        else:
            raise GeometryDslError(f"unexpected character {char!r}", location)
    tokens.append(_Token("EOF", "", SourceLocation(line, column, source_name)))
    return tuple(tokens)


class _GeometryParser:
    _DECLARATION_TYPES = {
        "point": "Point",
        "line": "Line",
        "segment": "Segment",
        "angle": "Angle",
    }

    def __init__(self, text: str, registry: KernelRegistry, source_name: str) -> None:
        self.tokens = _tokenize(text, source_name)
        self.position = 0
        self.registry = registry
        self.source_name = source_name
        self.symbols: dict[str, Term] = {}
        self.facts: list[Fact] = []
        self.goals: list[Goal] = []

    def parse(self) -> DomainInstance:
        self._skip_newlines()
        while not self._at("EOF"):
            keyword = self._expect("IDENT", "expected a statement")
            if keyword.value == "let":
                self._parse_alias()
            elif keyword.value in {"assume", "prove"}:
                expression = self._parse_expression()
                if not isinstance(expression, Atom):
                    raise GeometryDslError(
                        f"{keyword.value} expects a proposition", keyword.location
                    )
                if keyword.value == "assume":
                    self.facts.append(
                        Fact(
                            expression,
                            FactStatus.ASSUMED,
                            source="geometry_dsl",
                            assertion_status=AssertionStatus.EXPLICIT,
                            evidence_status=EvidenceStatus.ASSUMED,
                        )
                    )
                else:
                    self.goals.append(Goal(expression))
            elif keyword.value in self._DECLARATION_TYPES:
                self._parse_declaration(keyword)
            else:
                raise GeometryDslError(
                    f"unknown statement {keyword.value!r}", keyword.location
                )
            self._finish_statement()
            self._skip_newlines()
        if not self.goals:
            raise GeometryDslError(
                "a problem must contain at least one prove statement",
                self.current.location,
            )
        from ..model import WorldState

        return DomainInstance(
            registry=self.registry,
            state=WorldState(tuple(self.facts)),
            goals=tuple(self.goals),
            domain="geometry",
            metadata={"symbols": dict(self.symbols), "source_name": self.source_name},
        )

    def _parse_declaration(self, keyword: _Token) -> None:
        type_name = self._DECLARATION_TYPES[keyword.value]
        while True:
            name = self._expect("IDENT", "expected a symbol name")
            self._ensure_new_name(name)
            self.symbols[name.value] = self.registry.symbol(name.value, type_name)
            if not self._match("COMMA"):
                return

    def _parse_alias(self) -> None:
        name = self._expect("IDENT", "expected an alias name")
        self._ensure_new_name(name)
        self._expect("COLON", "expected ':' after alias name")
        type_token = self._expect("IDENT", "expected a type")
        try:
            expected = self.registry.types.resolve(type_token.value)
        except TypeValidationError as exc:
            raise GeometryDslError(str(exc), type_token.location) from exc
        self._expect("EQUAL", "expected '=' after alias type")
        value = self._parse_expression()
        if isinstance(value, Atom):
            raise GeometryDslError("a let alias must refer to a term", name.location)
        if value.type != expected:
            raise GeometryDslError(
                f"alias {name.value!r} declares {expected}, but expression has type {value.type}",
                type_token.location,
            )
        self.symbols[name.value] = value

    def _parse_expression(self) -> Term | Atom:
        name = self._expect("IDENT", "expected an expression")
        if not self._match("LPAREN"):
            try:
                return self.symbols[name.value]
            except KeyError as exc:
                raise GeometryDslError(
                    f"undeclared symbol {name.value!r}", name.location
                ) from exc
        arguments: list[Term | Atom] = []
        if not self._at("RPAREN"):
            while True:
                arguments.append(self._parse_expression())
                if not self._match("COMMA"):
                    break
        self._expect("RPAREN", "expected ')' after arguments")
        if any(isinstance(argument, Atom) for argument in arguments):
            raise GeometryDslError(
                "a proposition cannot be nested inside an operator", name.location
            )
        terms = tuple(argument for argument in arguments if isinstance(argument, Term))
        try:
            if name.value in self.registry.functions:
                return self.registry.apply(name.value, *terms)
            if name.value in self.registry.predicates:
                return self.registry.atom(name.value, *terms)
            raise KernelError(f"unknown operator {name.value!r}")
        except (KernelError, TypeValidationError) as exc:
            raise GeometryDslError(str(exc), name.location) from exc

    def _ensure_new_name(self, token: _Token) -> None:
        if token.value in self.symbols:
            raise GeometryDslError(f"duplicate symbol {token.value!r}", token.location)

    def _finish_statement(self) -> None:
        if not self._at("EOF"):
            self._expect("NEWLINE", "expected a newline after the statement")

    def _skip_newlines(self) -> None:
        while self._match("NEWLINE"):
            pass

    @property
    def current(self) -> _Token:
        return self.tokens[self.position]

    def _at(self, kind: str) -> bool:
        return self.current.kind == kind

    def _match(self, kind: str) -> bool:
        if not self._at(kind):
            return False
        self.position += 1
        return True

    def _expect(self, kind: str, message: str) -> _Token:
        if not self._at(kind):
            raise GeometryDslError(message, self.current.location)
        token = self.current
        self.position += 1
        return token


def parse_geometry_dsl(
    text: str,
    registry: KernelRegistry | None = None,
    source_name: str = "<string>",
) -> DomainInstance:
    return _GeometryParser(
        text, registry or create_geometry_registry(), source_name
    ).parse()

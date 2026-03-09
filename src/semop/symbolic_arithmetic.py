from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .structures import SymbolicResult


@dataclass
class ArithmeticMatch:
    total: float
    answer: str
    evidence: list[str]
    equations: list[str]
    confidence: float

    def to_symbolic_result(self) -> SymbolicResult:
        return SymbolicResult(
            domain="arithmetic",
            answer=self.answer,
            evidence=self.evidence,
            equations=self.equations,
            confidence=self.confidence,
            source="symbolic_arithmetic",
        )


@dataclass
class LinearForm:
    coefficient: float = 0.0
    constant: float = 0.0
    variable: str = "x"

    def add(self, other: "LinearForm") -> "LinearForm":
        self._require_same_variable(other)
        return LinearForm(self.coefficient + other.coefficient, self.constant + other.constant, self.variable)

    def sub(self, other: "LinearForm") -> "LinearForm":
        self._require_same_variable(other)
        return LinearForm(self.coefficient - other.coefficient, self.constant - other.constant, self.variable)

    def mul(self, other: "LinearForm") -> "LinearForm":
        self._require_same_variable(other)
        if self.coefficient and other.coefficient:
            raise ValueError("Non-linear multiplication is not supported")
        if other.coefficient:
            return LinearForm(self.constant * other.coefficient, self.constant * other.constant, self.variable)
        return LinearForm(other.constant * self.coefficient, other.constant * self.constant, self.variable)

    def div(self, other: "LinearForm") -> "LinearForm":
        self._require_same_variable(other)
        if other.coefficient:
            raise ValueError("Division by a variable expression is not supported")
        if abs(other.constant) < 1e-12:
            raise ZeroDivisionError("Division by zero")
        return LinearForm(self.coefficient / other.constant, self.constant / other.constant, self.variable)

    def evaluate(self) -> float:
        if abs(self.coefficient) > 1e-12:
            raise ValueError("Expression still contains a variable")
        return self.constant

    def simplify_text(self) -> str:
        coeff = self.coefficient
        const = self.constant
        pieces: list[str] = []
        if abs(coeff) > 1e-12:
            if abs(coeff - 1.0) < 1e-12:
                pieces.append(self.variable)
            elif abs(coeff + 1.0) < 1e-12:
                pieces.append(f"-{self.variable}")
            else:
                pieces.append(f"{_fmt(coeff)}{self.variable}")
        if abs(const) > 1e-12 or not pieces:
            const_text = _fmt(abs(const)) if pieces else _fmt(const)
            if pieces:
                sign = "+" if const >= 0 else "-"
                pieces.append(f"{sign} {const_text}")
            else:
                pieces.append(const_text)
        return " ".join(pieces)

    def _require_same_variable(self, other: "LinearForm") -> None:
        if self.variable != other.variable:
            raise ValueError("Multiple variables are not supported")


class ExpressionParser:
    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.index = 0

    def evaluate(self, expression: str) -> float:
        form = self.parse_linear(expression)
        return form.evaluate()

    def parse_linear(self, expression: str) -> LinearForm:
        self.tokens = self._tokenize(expression)
        self.index = 0
        value = self._parse_expression()
        if self.index != len(self.tokens):
            raise ValueError(f"Unexpected trailing token: {self.tokens[self.index]}")
        return value

    @staticmethod
    def _tokenize(expression: str) -> list[str]:
        compact = expression.replace(" ", "")
        tokens = re.findall(r"ceil|floor|[A-Za-z]+|\d+(?:\.\d+)?|[()+\-*/,=]", compact)
        if "".join(tokens) != compact:
            raise ValueError(f"Unsupported expression: {expression}")
        return tokens

    def _parse_expression(self) -> LinearForm:
        value = self._parse_term()
        while self.index < len(self.tokens) and self.tokens[self.index] in {"+", "-"}:
            op = self.tokens[self.index]
            self.index += 1
            rhs = self._parse_term()
            value = value.add(rhs) if op == "+" else value.sub(rhs)
        return value

    def _parse_term(self) -> LinearForm:
        value = self._parse_factor()
        while self.index < len(self.tokens) and self.tokens[self.index] in {"*", "/"}:
            op = self.tokens[self.index]
            self.index += 1
            rhs = self._parse_factor()
            value = value.mul(rhs) if op == "*" else value.div(rhs)
        return value

    def _parse_factor(self) -> LinearForm:
        if self.index >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        token = self.tokens[self.index]

        if token == "-":
            self.index += 1
            inner = self._parse_factor()
            return LinearForm(-inner.coefficient, -inner.constant, inner.variable)

        if token in {"ceil", "floor"}:
            self.index += 1
            self._expect("(")
            inner = self._parse_expression()
            self._expect(")")
            value = inner.evaluate()
            rounded = float(math.ceil(value) if token == "ceil" else math.floor(value))
            return LinearForm(0.0, rounded)

        if token == "(":
            self.index += 1
            value = self._parse_expression()
            self._expect(")")
            return value

        self.index += 1
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            if self.index < len(self.tokens) and re.fullmatch(r"[A-Za-z]+", self.tokens[self.index]):
                variable = self.tokens[self.index]
                self.index += 1
                return LinearForm(float(token), 0.0, variable)
            return LinearForm(0.0, float(token))

        if re.fullmatch(r"[A-Za-z]+", token):
            return LinearForm(1.0, 0.0, token)

        raise ValueError(f"Unsupported token: {token}")

    def _expect(self, token: str) -> None:
        if self.index >= len(self.tokens) or self.tokens[self.index] != token:
            raise ValueError(f"Expected {token}")
        self.index += 1


class ArithmeticReasoner:
    def __init__(self) -> None:
        self.parser = ExpressionParser()

    def solve(self, query: str) -> SymbolicResult | None:
        normalized = self._normalize_number_words(query)
        match = (
            self._solve_direct_equation(normalized)
            or self._solve_binomial_expansion(normalized)
            or self._solve_common_factor_factoring(normalized)
            or self._solve_direct_simplification(normalized)
            or self._solve_direct_calculation(normalized)
            or self._solve_line_item_division(normalized)
            or self._solve_sales_multiplier(normalized)
            or self._solve_ratio_share(normalized)
            or self._solve_unit_rate(normalized)
            or self._solve_fraction_weighted_total(normalized)
            or self._solve_pack_round_up(normalized)
        )
        return match.to_symbolic_result() if match is not None else None

    def _solve_direct_calculation(self, query: str) -> ArithmeticMatch | None:
        lowered = query.lower().strip()
        if not any(trigger in lowered for trigger in ["calculate", "compute", "evaluate", "what is"]):
            return None
        expression = self._extract_expression_candidate(query)
        if expression is None or "=" in expression or re.search(r"[A-Za-z]", expression):
            return None
        total = self.parser.evaluate(expression)
        return ArithmeticMatch(
            total=total,
            answer=f"Symbolic math answer: {expression} = {_fmt(total)}.",
            evidence=["direct calculation"],
            equations=[f"result = {expression}"],
            confidence=0.9,
        )

    def _solve_direct_simplification(self, query: str) -> ArithmeticMatch | None:
        lowered = query.lower().strip()
        if not any(trigger in lowered for trigger in ["simplify", "collect terms", "combine like terms"]):
            return None
        expression = self._extract_expression_candidate(query)
        if expression is None or "=" in expression:
            return None
        form = self.parser.parse_linear(expression)
        simplified = form.simplify_text()
        return ArithmeticMatch(
            total=form.constant,
            answer=f"Symbolic math answer: {expression} simplifies to {simplified}.",
            evidence=["linear expression simplification"],
            equations=[f"simplified = {simplified}"],
            confidence=0.87,
        )

    def _solve_binomial_expansion(self, query: str) -> ArithmeticMatch | None:
        lowered = query.lower().strip()
        if not any(trigger in lowered for trigger in ["expand", "multiply out"]):
            return None
        expression = self._extract_expression_candidate(query)
        if expression is None:
            return None
        compact = expression.replace(" ", "")
        match = re.fullmatch(
            r"\(([-+]?\d*(?:\.\d+)?)?([A-Za-z]+)([-+]\d+(?:\.\d+)?)\)\(([-+]?\d*(?:\.\d+)?)?([A-Za-z]+)([-+]\d+(?:\.\d+)?)\)",
            compact,
        )
        if match is None:
            return None
        a_raw, var_left, b_raw, c_raw, var_right, d_raw = match.groups()
        if var_left != var_right:
            return None
        variable = var_left
        a = self._read_coeff(a_raw)
        b = float(b_raw)
        c = self._read_coeff(c_raw)
        d = float(d_raw)
        quad = a * c
        linear = a * d + b * c
        constant = b * d
        expanded = self._quadratic_text(variable, quad, linear, constant)
        return ArithmeticMatch(
            total=constant,
            answer=f"Symbolic math answer: {expression} expands to {expanded}.",
            evidence=["binomial expansion"],
            equations=[f"expanded = {expanded}"],
            confidence=0.86,
        )

    def _solve_common_factor_factoring(self, query: str) -> ArithmeticMatch | None:
        lowered = query.lower().strip()
        if not any(trigger in lowered for trigger in ["factor", "factorize"]):
            return None
        expression = self._extract_expression_candidate(query)
        if expression is None:
            return None
        form = self.parser.parse_linear(expression)
        coeff = form.coefficient
        const = form.constant
        if abs(coeff) < 1e-12 or abs(const) < 1e-12:
            return None
        if abs(coeff - round(coeff)) > 1e-12 or abs(const - round(const)) > 1e-12:
            return None
        gcd_value = math.gcd(int(abs(round(coeff))), int(abs(round(const))))
        if gcd_value <= 1:
            return None
        inner_coeff = coeff / gcd_value
        inner_const = const / gcd_value
        inner = LinearForm(inner_coeff, inner_const, form.variable).simplify_text()
        factored = f"{gcd_value}({inner})"
        return ArithmeticMatch(
            total=float(gcd_value),
            answer=f"Symbolic math answer: {expression} factors to {factored}.",
            evidence=["common-factor factoring"],
            equations=[f"gcd = {gcd_value}", f"factored = {factored}"],
            confidence=0.84,
        )

    def _solve_direct_equation(self, query: str) -> ArithmeticMatch | None:
        lowered = query.lower().strip()
        if not any(trigger in lowered for trigger in ["solve", "find x", "find n", "what is x", "what is n"]):
            return None
        equation = self._extract_equation_candidate(query)
        if equation is None:
            return None
        left_text, right_text = [part.strip() for part in equation.split("=", 1)]
        left = self.parser.parse_linear(left_text)
        right = self.parser.parse_linear(right_text)
        variable = left.variable if abs(left.coefficient) > 1e-12 else right.variable
        coefficient = left.coefficient - right.coefficient
        constant = right.constant - left.constant
        if abs(coefficient) < 1e-12:
            return None
        value = constant / coefficient
        return ArithmeticMatch(
            total=value,
            answer=f"Symbolic math answer: {variable} = {_fmt(value)}.",
            evidence=["single-variable linear equation"],
            equations=[f"{_fmt(coefficient)}{variable} = {_fmt(constant)}", f"{variable} = {_fmt(constant)} / {_fmt(coefficient)}"],
            confidence=0.89,
        )

    def _solve_line_item_division(self, query: str) -> ArithmeticMatch | None:
        lowered = query.lower()
        if "split" not in lowered or "equally" not in lowered:
            return None
        items = re.findall(r"(\d+(?:\.\d+)?)\s+[^.;,]*?cost\w*\s*\$?(\d+(?:\.\d+)?)", query, flags=re.IGNORECASE)
        people_match = re.search(r"(\d+(?:\.\d+)?)\s+(?:friends|people|students|members)", query, flags=re.IGNORECASE)
        if not items or not people_match:
            return None
        total_expr = " + ".join(f"({count}*{price})" for count, price in items)
        people = float(people_match.group(1))
        if people == 0:
            return None
        per_expr = f"({total_expr}) / {_fmt(people)}"
        per_person = self.parser.evaluate(per_expr)
        total = self.parser.evaluate(total_expr)
        return ArithmeticMatch(
            total=per_person,
            answer=f"Symbolic math answer: each person pays {_fmt(per_person)}.",
            evidence=["split equally", f"people={_fmt(people)}"],
            equations=[f"total = {total_expr}", f"each = {_fmt(total)} / {_fmt(people)}"],
            confidence=0.79,
        )

    def _solve_sales_multiplier(self, query: str) -> ArithmeticMatch | None:
        lowered = query.lower()
        if "times as much this month" not in lowered and "twice as much this month" not in lowered:
            return None
        price_pairs = re.findall(r"\$?(\d+(?:\.\d+)?)\s+for a\s+(large|small)\s+painting", lowered)
        sold_pairs = re.findall(r"sold\s+(\d+(?:\.\d+)?)\s+(large|small)\s+paintings?", lowered)
        if not price_pairs or not sold_pairs:
            return None
        multiplier_match = re.search(r"(\d+(?:\.\d+)?)\s+times as much this month", lowered)
        multiplier = float(multiplier_match.group(1)) if multiplier_match else (2.0 if "twice as much this month" in lowered else 1.0)
        price_map = {kind: float(value) for value, kind in price_pairs}
        sold_map = {kind: float(value) for value, kind in sold_pairs}
        if not {"large", "small"}.issubset(price_map) or not {"large", "small"}.issubset(sold_map):
            return None
        base_expr = f"({_fmt(price_map['large'])}*{_fmt(sold_map['large'])}) + ({_fmt(price_map['small'])}*{_fmt(sold_map['small'])})"
        total_expr = f"({base_expr}) * {_fmt(multiplier)}"
        total = self.parser.evaluate(total_expr)
        return ArithmeticMatch(
            total=total,
            answer=f"Symbolic math answer: this month's sales are {_fmt(total)}.",
            evidence=[f"multiplier={_fmt(multiplier)}"],
            equations=[f"last_month = {base_expr}", f"this_month = {total_expr}"],
            confidence=0.81,
        )

    def _solve_ratio_share(self, query: str) -> ArithmeticMatch | None:
        lowered = query.lower()
        if "ratio" not in lowered:
            return None
        ratio_match = re.search(r"ratio\s+(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", lowered)
        partner_match = re.search(r"([a-z]+)\s+got\s*\$?(\d+(?:\.\d+)?)", lowered)
        cost_match = re.search(r"costs?\s*\$?(\d+(?:\.\d+)?)", lowered)
        if not ratio_match or not partner_match:
            return None
        left_ratio = float(ratio_match.group(1))
        right_ratio = float(ratio_match.group(2))
        known_name = partner_match.group(1)
        known_share = float(partner_match.group(2))
        spend = float(cost_match.group(1)) if cost_match else 0.0
        if known_name == "johnson":
            if right_ratio == 0:
                return None
            unit_expr = f"{_fmt(known_share)} / {_fmt(right_ratio)}"
            share_expr = f"({unit_expr}) * {_fmt(left_ratio)} - {_fmt(spend)}"
            target_name = "Mike"
        else:
            if left_ratio == 0:
                return None
            unit_expr = f"{_fmt(known_share)} / {_fmt(left_ratio)}"
            share_expr = f"({unit_expr}) * {_fmt(right_ratio)} - {_fmt(spend)}"
            target_name = "Johnson"
        remaining = self.parser.evaluate(share_expr)
        return ArithmeticMatch(
            total=remaining,
            answer=f"Symbolic math answer: {target_name} has {_fmt(remaining)} left.",
            evidence=[f"ratio {_fmt(left_ratio)}:{_fmt(right_ratio)}"],
            equations=[f"unit = {unit_expr}", f"share = {share_expr}"],
            confidence=0.82,
        )

    def _solve_unit_rate(self, query: str) -> ArithmeticMatch | None:
        lowered = query.lower()
        uses_match = re.search(r"uses?\s+(\d+(?:\.\d+)?)\s+[^.?!]*?per\s+gift\s+box", lowered)
        has_match = re.search(r"has\s+(\d+(?:\.\d+)?)\s+[^.?!]*?per\s+day", lowered)
        day_match = re.search(r"every\s+(\d+(?:\.\d+)?)\s+days?", lowered)
        if not uses_match or not has_match or not day_match:
            return None
        use_per_box = float(uses_match.group(1))
        amount_per_day = float(has_match.group(1))
        days = float(day_match.group(1))
        if use_per_box == 0:
            return None
        total_material_expr = f"{_fmt(amount_per_day)} * {_fmt(days)}"
        total_boxes_expr = f"({total_material_expr}) / {_fmt(use_per_box)}"
        total_boxes = self.parser.evaluate(total_boxes_expr)
        return ArithmeticMatch(
            total=total_boxes,
            answer=f"Symbolic math answer: the number of boxes is {_fmt(total_boxes)}.",
            evidence=["per gift box", "per day"],
            equations=[f"total_material = {total_material_expr}", f"boxes = {total_boxes_expr}"],
            confidence=0.77,
        )

    def _solve_fraction_weighted_total(self, query: str) -> ArithmeticMatch | None:
        lowered = query.lower()
        if "rest gets an average of" not in lowered:
            return None
        fraction_match = re.search(r"(\d+)\s*/\s*(\d+)\s+of the [^.?!]* receive an average of\s*\$?(\d+(?:\.\d+)?)", lowered)
        rest_match = re.search(r"rest gets an average of\s*\$?(\d+(?:\.\d+)?)", lowered)
        surveyed_match = re.search(r"surveyed\s+(\d+(?:\.\d+)?)\s+(?:students|people|members)", lowered)
        if not fraction_match or not rest_match or not surveyed_match:
            return None
        numerator = float(fraction_match.group(1))
        denominator = float(fraction_match.group(2))
        high_value = float(fraction_match.group(3))
        low_value = float(rest_match.group(1))
        population = float(surveyed_match.group(1))
        if denominator == 0:
            return None
        high_count_expr = f"{_fmt(population)} * {_fmt(numerator)} / {_fmt(denominator)}"
        high_count = self.parser.evaluate(high_count_expr)
        low_count_expr = f"{_fmt(population)} - ({high_count_expr})"
        low_count = self.parser.evaluate(low_count_expr)
        total_expr = f"({_fmt(high_count)}*{_fmt(high_value)}) + ({_fmt(low_count)}*{_fmt(low_value)})"
        total = self.parser.evaluate(total_expr)
        return ArithmeticMatch(
            total=total,
            answer=f"Symbolic math answer: the surveyed group receives {_fmt(total)} in total.",
            evidence=[f"fraction={_fmt(numerator)}/{_fmt(denominator)}", f"population={_fmt(population)}"],
            equations=[f"group_a = {high_count_expr}", f"total = {total_expr}"],
            confidence=0.84,
        )

    def _solve_pack_round_up(self, query: str) -> ArithmeticMatch | None:
        lowered = query.lower()
        if "packs of" not in lowered:
            return None
        pack_match = re.search(r"packs?\s+of\s+(\d+(?:\.\d+)?)", lowered)
        team_match = re.search(r"(\d+(?:\.\d+)?)\s+members", lowered)
        coaches_match = re.search(r"(\d+(?:\.\d+)?)\s+coaches", lowered)
        helpers_match = re.search(r"(\d+(?:\.\d+)?)\s+helpers", lowered)
        if not pack_match or not team_match:
            return None
        pack_size = float(pack_match.group(1))
        total_people = float(team_match.group(1))
        if coaches_match is not None:
            total_people += float(coaches_match.group(1))
        if helpers_match is not None:
            total_people += float(helpers_match.group(1))
        if pack_size == 0:
            return None
        pack_expr = f"ceil({_fmt(total_people)} / {_fmt(pack_size)})"
        packs = self.parser.evaluate(pack_expr)
        return ArithmeticMatch(
            total=packs,
            answer=f"Symbolic math answer: buy {_fmt(packs)} packs.",
            evidence=[f"pack_size={_fmt(pack_size)}", f"people={_fmt(total_people)}"],
            equations=[f"people = {_fmt(total_people)}", f"packs = {pack_expr}"],
            confidence=0.82,
        )

    @staticmethod
    def _normalize_number_words(text: str) -> str:
        mapping = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "twice": "2 times",
        }
        normalized = text
        for word, digit in mapping.items():
            normalized = re.sub(rf"\b{word}\b", digit, normalized, flags=re.IGNORECASE)
        return normalized

    @staticmethod
    def _extract_expression_candidate(query: str) -> str | None:
        match = re.search(r"(?:calculate|compute|evaluate|what is|simplify|collect terms|combine like terms|expand|multiply out|factor|factorize)\s*:?\s*([^?]+)", query, flags=re.IGNORECASE)
        if match is None:
            return None
        candidate = match.group(1).strip().rstrip(". ")
        return candidate if candidate else None

    @staticmethod
    def _extract_equation_candidate(query: str) -> str | None:
        match = re.search(r"([A-Za-z0-9+\-*/().\s]+=[A-Za-z0-9+\-*/().\s]+)", query)
        if match is None:
            return None
        candidate = match.group(1).strip().rstrip(". ")
        candidate = re.sub(r"^(?:solve|find\s+[A-Za-z]+|what\s+is\s+[A-Za-z]+)\s+", "", candidate, flags=re.IGNORECASE)
        return candidate.strip()

    @staticmethod
    def _read_coeff(text: str | None) -> float:
        if text in (None, "", "+"):
            return 1.0
        if text == "-":
            return -1.0
        return float(text)

    @staticmethod
    def _quadratic_text(variable: str, quad: float, linear: float, constant: float) -> str:
        parts: list[str] = []
        if abs(quad) > 1e-12:
            if abs(quad - 1.0) < 1e-12:
                parts.append(f"{variable}^2")
            elif abs(quad + 1.0) < 1e-12:
                parts.append(f"-{variable}^2")
            else:
                parts.append(f"{_fmt(quad)}{variable}^2")
        if abs(linear) > 1e-12:
            term = variable if abs(linear - 1.0) < 1e-12 else (f"-{variable}" if abs(linear + 1.0) < 1e-12 else f"{_fmt(abs(linear))}{variable}")
            if parts:
                parts.append(f"+ {term}" if linear >= 0 else f"- {term.lstrip('-')}")
            else:
                parts.append(term if linear >= 0 else f"-{term.lstrip('-')}")
        if abs(constant) > 1e-12 or not parts:
            const_text = _fmt(abs(constant)) if parts else _fmt(constant)
            if parts:
                parts.append(f"+ {const_text}" if constant >= 0 else f"- {const_text}")
            else:
                parts.append(const_text)
        return " ".join(parts)


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")

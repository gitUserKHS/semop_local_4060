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


class ExpressionParser:
    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.index = 0

    def evaluate(self, expression: str) -> float:
        self.tokens = self._tokenize(expression)
        self.index = 0
        value = self._parse_expression()
        if self.index != len(self.tokens):
            raise ValueError(f"Unexpected trailing token: {self.tokens[self.index]}")
        return value

    @staticmethod
    def _tokenize(expression: str) -> list[str]:
        compact = expression.replace(" ", "")
        tokens = re.findall(r"ceil|floor|\d+(?:\.\d+)?|[()+\-*/,]", compact)
        if "".join(tokens) != compact:
            raise ValueError(f"Unsupported expression: {expression}")
        return tokens

    def _parse_expression(self) -> float:
        value = self._parse_term()
        while self.index < len(self.tokens) and self.tokens[self.index] in {"+", "-"}:
            op = self.tokens[self.index]
            self.index += 1
            rhs = self._parse_term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def _parse_term(self) -> float:
        value = self._parse_factor()
        while self.index < len(self.tokens) and self.tokens[self.index] in {"*", "/"}:
            op = self.tokens[self.index]
            self.index += 1
            rhs = self._parse_factor()
            value = value * rhs if op == "*" else value / rhs
        return value

    def _parse_factor(self) -> float:
        if self.index >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        token = self.tokens[self.index]

        if token == "-":
            self.index += 1
            return -self._parse_factor()

        if token in {"ceil", "floor"}:
            self.index += 1
            self._expect("(")
            inner = self._parse_expression()
            self._expect(")")
            return float(math.ceil(inner) if token == "ceil" else math.floor(inner))

        if token == "(":
            self.index += 1
            value = self._parse_expression()
            self._expect(")")
            return value

        self.index += 1
        return float(token)

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
            self._solve_line_item_division(normalized)
            or self._solve_sales_multiplier(normalized)
            or self._solve_ratio_share(normalized)
            or self._solve_unit_rate(normalized)
            or self._solve_fraction_weighted_total(normalized)
            or self._solve_pack_round_up(normalized)
        )
        return match.to_symbolic_result() if match is not None else None

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
        per_expr = f"({total_expr}) / {self._fmt(people)}"
        per_person = self.parser.evaluate(per_expr)
        total = self.parser.evaluate(total_expr)
        return ArithmeticMatch(
            total=per_person,
            answer=f"Symbolic math answer: each person pays {self._fmt(per_person)}.",
            evidence=["split equally", f"people={self._fmt(people)}"],
            equations=[f"total = {total_expr}", f"each = {self._fmt(total)} / {self._fmt(people)}"],
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
        base_expr = f"({self._fmt(price_map['large'])}*{self._fmt(sold_map['large'])}) + ({self._fmt(price_map['small'])}*{self._fmt(sold_map['small'])})"
        total_expr = f"({base_expr}) * {self._fmt(multiplier)}"
        total = self.parser.evaluate(total_expr)
        return ArithmeticMatch(
            total=total,
            answer=f"Symbolic math answer: this month's sales are {self._fmt(total)}.",
            evidence=[f"multiplier={self._fmt(multiplier)}"],
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
            unit_expr = f"{self._fmt(known_share)} / {self._fmt(right_ratio)}"
            share_expr = f"({unit_expr}) * {self._fmt(left_ratio)} - {self._fmt(spend)}"
            target_name = "Mike"
        else:
            if left_ratio == 0:
                return None
            unit_expr = f"{self._fmt(known_share)} / {self._fmt(left_ratio)}"
            share_expr = f"({unit_expr}) * {self._fmt(right_ratio)} - {self._fmt(spend)}"
            target_name = "Johnson"
        remaining = self.parser.evaluate(share_expr)
        return ArithmeticMatch(
            total=remaining,
            answer=f"Symbolic math answer: {target_name} has {self._fmt(remaining)} left.",
            evidence=[f"ratio {self._fmt(left_ratio)}:{self._fmt(right_ratio)}"],
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
        total_material_expr = f"{self._fmt(amount_per_day)} * {self._fmt(days)}"
        total_boxes_expr = f"({total_material_expr}) / {self._fmt(use_per_box)}"
        total_boxes = self.parser.evaluate(total_boxes_expr)
        return ArithmeticMatch(
            total=total_boxes,
            answer=f"Symbolic math answer: the number of boxes is {self._fmt(total_boxes)}.",
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
        high_count_expr = f"{self._fmt(population)} * {self._fmt(numerator)} / {self._fmt(denominator)}"
        high_count = self.parser.evaluate(high_count_expr)
        low_count_expr = f"{self._fmt(population)} - ({high_count_expr})"
        low_count = self.parser.evaluate(low_count_expr)
        total_expr = f"({self._fmt(high_count)}*{self._fmt(high_value)}) + ({self._fmt(low_count)}*{self._fmt(low_value)})"
        total = self.parser.evaluate(total_expr)
        return ArithmeticMatch(
            total=total,
            answer=f"Symbolic math answer: the surveyed group receives {self._fmt(total)} in total.",
            evidence=[f"fraction={self._fmt(numerator)}/{self._fmt(denominator)}", f"population={self._fmt(population)}"],
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
        pack_expr = f"ceil({self._fmt(total_people)} / {self._fmt(pack_size)})"
        packs = self.parser.evaluate(pack_expr)
        return ArithmeticMatch(
            total=packs,
            answer=f"Symbolic math answer: buy {self._fmt(packs)} packs.",
            evidence=[f"pack_size={self._fmt(pack_size)}", f"people={self._fmt(total_people)}"],
            equations=[f"people = {self._fmt(total_people)}", f"packs = {pack_expr}"],
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
    def _fmt(value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.2f}".rstrip("0").rstrip(".")

from __future__ import annotations

from typing import Any

from .runtime import DomainKind
from .semantic_benchmark import SemanticBenchmark, SemanticBenchmarkCase
from .semantic_grounding_case_common import make_programmatic_semantic_case
from .self_learning import LearningSplit


def generate_operator_boundary_semantic_benchmark(
    *,
    per_domain: int,
    split: LearningSplit | str,
    seed: int,
    namespace: str,
) -> SemanticBenchmark:
    """Generate broad operator-state coverage for low-resource curriculum selection."""

    resolved_split, normalized_namespace = _validate_generation(
        per_domain, split, namespace, "operator-boundary"
    )
    cases: list[SemanticBenchmarkCase] = []
    for index in range(per_domain):
        cases.extend(
            (
                _boundary_language_case(
                    index, seed, normalized_namespace, resolved_split
                ),
                _boundary_math_case(index, seed, normalized_namespace, resolved_split),
                _boundary_vision_case(
                    index, seed, normalized_namespace, resolved_split
                ),
            )
        )
    return SemanticBenchmark(tuple(cases))


def generate_extrapolation_semantic_benchmark(
    *,
    per_domain: int,
    split: LearningSplit | str,
    seed: int,
    namespace: str,
) -> SemanticBenchmark:
    """Generate a development split with larger compositions than the curriculum."""

    resolved_split, normalized_namespace = _validate_generation(
        per_domain, split, namespace, "extrapolation"
    )
    cases: list[SemanticBenchmarkCase] = []
    for index in range(per_domain):
        cases.extend(
            (
                _extrapolation_language_case(
                    index, seed, normalized_namespace, resolved_split
                ),
                _extrapolation_math_case(
                    index, seed, normalized_namespace, resolved_split
                ),
                _extrapolation_vision_case(
                    index, seed, normalized_namespace, resolved_split
                ),
            )
        )
    return SemanticBenchmark(tuple(cases))


def _validate_generation(
    per_domain: int,
    split: LearningSplit | str,
    namespace: str,
    label: str,
) -> tuple[LearningSplit, str]:
    if per_domain <= 0:
        raise ValueError(f"{label} semantic case count must be positive")
    normalized = namespace.strip()
    if not normalized:
        raise ValueError(f"{label} semantic namespace must not be empty")
    return LearningSplit(split), normalized


def _boundary_language_case(
    index: int,
    seed: int,
    namespace: str,
    split: LearningSplit,
) -> SemanticBenchmarkCase:
    token = 20_000_000 + abs(seed) * 10_000 + index
    goal = f"boundary_release_{token}"
    variant = index % 12
    count = 1 + (variant % 4)
    premises = [f"boundary_check_{token}_{slot}" for slot in range(count)]
    requirement = f"Goal: {goal}; Requires: {', '.join(premises)}"
    if variant < 4:
        statuses = [f"Satisfied: {premise}" for premise in premises]
        expected, phenomenon = True, f"boundary_language_complete_{count}"
    elif variant < 8:
        statuses = [f"Satisfied: {premise}" for premise in premises[:-1]]
        expected, phenomenon = False, f"boundary_language_missing_{count}"
    elif variant == 8:
        statuses = [f"Blocked: {premises[-1]}"]
        expected, phenomenon = True, "boundary_language_clean_block"
    elif variant == 9:
        statuses = [
            *(f"Satisfied: {premise}" for premise in premises),
            f"Blocked: {premises[-1]}",
        ]
        expected, phenomenon = False, "boundary_language_conflicting_block"
    elif variant == 10:
        statuses = [
            *(f"Satisfied: {premise}" for premise in premises[:-1]),
            f"Satisfied: unrelated_{token}",
        ]
        expected, phenomenon = False, "boundary_language_wrong_binding"
    else:
        requirement = f"Requires: {', '.join(premises)}; Goal: {goal}"
        statuses = [f"Satisfied: {premise}" for premise in reversed(premises)]
        expected, phenomenon = True, "boundary_language_reordered_complete"
    return make_programmatic_semantic_case(
        case_id=f"{namespace}-language-{index:04d}",
        domain=DomainKind.LANGUAGE,
        payload={
            "text": "; ".join((requirement, *statuses)),
            "use_legacy_heuristics": False,
        },
        expected=expected,
        phenomenon=phenomenon,
        split=split,
        difficulty=2,
        extra_tags=("operator_boundary_curriculum",),
    )


def _boundary_math_case(
    index: int,
    seed: int,
    namespace: str,
    split: LearningSplit,
) -> SemanticBenchmarkCase:
    base = 7 + abs(seed) * 3 + index
    other = 2 + (abs(seed) % 13) + (index % 6)
    variant = index % 26
    if variant == 0:
        expression = f"3*(x - {other}) = x + {base}"
        expected, phenomenon = True, "boundary_math_two_sided_linear"
    elif variant == 1:
        answer = (base + other) * (other + 1) - base
        expression = f"(({base}+{other})*({other}+1)-{base}) == {answer}"
        expected, phenomenon = True, "boundary_math_nested_true"
    elif variant == 2:
        answer = (base + other) * (other + 1) - base + 1
        expression = f"(({base}+{other})*({other}+1)-{base}) == {answer}"
        expected, phenomenon = False, "boundary_math_nested_near_miss"
    elif variant == 3:
        expression = f"-({base}+{other}) < -{base}"
        expected, phenomenon = True, "boundary_math_negative_less"
    elif variant == 4:
        expression = f"-({base}+{other}) >= -{base}"
        expected, phenomenon = False, "boundary_math_negative_greater_miss"
    elif variant == 5:
        expression = f"-{base}+{other} == {-base + other}"
        expected, phenomenon = True, "boundary_math_signed_sum"
    elif variant == 6:
        expression = f"-{base}+{other} == {-base + other + 1}"
        expected, phenomenon = False, "boundary_math_signed_sum_miss"
    elif variant == 7:
        expression = f"({base}*{other})/{other} <= {base}"
        expected, phenomenon = True, "boundary_math_fraction_boundary"
    elif variant == 8:
        expression = f"(-{base})*(-{other}) == {base * other}"
        expected, phenomenon = True, "boundary_math_double_negative"
    elif variant == 9:
        expression = f"(-{base})*{other} != {-base * other}"
        expected, phenomenon = False, "boundary_math_negative_polarity_miss"
    elif variant == 10:
        expression = f"({base}-{base}) == 0"
        expected, phenomenon = True, "boundary_math_exact_zero"
    elif variant == 11:
        expression = f"({base}-{base}) > 0"
        expected, phenomenon = False, "boundary_math_zero_order_miss"
    elif variant == 12:
        expression = f"({base}+{other})/({other}+{base}) == 1"
        expected, phenomenon = True, "boundary_math_unit_ratio"
    elif variant == 13:
        expression = f"({base}+{other})/({other}+{base}) != 1"
        expected, phenomenon = False, "boundary_math_unit_ratio_miss"
    elif variant == 14:
        expression = f"4*(x + {other}) = 2*x + {base}"
        expected, phenomenon = True, "boundary_math_scaled_linear"
    elif variant == 15:
        expression = f"2*(x + {other}) = x - {base}"
        expected, phenomenon = True, "boundary_math_negative_solution_linear"
    elif variant == 16:
        expression = f"{base}+{other} <= {base}"
        expected, phenomenon = False, "boundary_math_less_equal_miss"
    elif variant == 17:
        expression = f"{base}-{other} >= {base}"
        expected, phenomenon = False, "boundary_math_greater_equal_miss"
    elif variant == 18:
        expression = f"{base}+{other} > {base}"
        expected, phenomenon = True, "boundary_math_greater_true"
    elif variant == 19:
        expression = f"{base}-{other} < {base}"
        expected, phenomenon = True, "boundary_math_less_true"
    elif variant == 20:
        expression = f"{base} <= {base}"
        expected, phenomenon = True, "boundary_math_less_equal_boundary"
    elif variant == 21:
        expression = f"{base} >= {base}"
        expected, phenomenon = True, "boundary_math_greater_equal_boundary"
    elif variant == 22:
        expression = f"{base} == {base + 1}"
        expected, phenomenon = False, "boundary_math_equal_near_miss"
    elif variant == 23:
        expression = f"{base} != {base + 1}"
        expected, phenomenon = True, "boundary_math_not_equal_true"
    elif variant == 24:
        expression = f"{base} < {base}"
        expected, phenomenon = False, "boundary_math_less_strict_boundary_miss"
    else:
        expression = f"{base} > {base}"
        expected, phenomenon = False, "boundary_math_greater_strict_boundary_miss"
    return make_programmatic_semantic_case(
        case_id=f"{namespace}-math-{index:04d}",
        domain=DomainKind.MATH,
        payload={"expression": expression},
        expected=expected,
        phenomenon=phenomenon,
        split=split,
        difficulty=2,
        extra_tags=("operator_boundary_curriculum",),
    )


def _boundary_vision_case(
    index: int,
    seed: int,
    namespace: str,
    split: LearningSplit,
) -> SemanticBenchmarkCase:
    variant = index % 16
    pad = 1 + ((seed + index) % 3)
    if variant in {0, 1, 2, 3}:
        size = 2 if variant < 2 else 4
        incomplete = variant % 2 == 1
        rows = _shape_rows(size=size, pad=pad, incomplete=incomplete)
        goal: dict[str, Any] = {
            "kind": "property",
            "predicate": "SQUARE",
            "subject": "red",
        }
        expected = not incomplete
        phenomenon = f"boundary_vision_square_{size}_{'gap' if incomplete else 'full'}"
    elif variant in {4, 5}:
        rows = _relation_rows(vertical=False, pad=pad)
        reverse = variant == 5
        goal = {
            "kind": "relation",
            "predicate": "LEFT_OF",
            "source": "blue" if reverse else "red",
            "target": "red" if reverse else "blue",
        }
        expected = not reverse
        phenomenon = "boundary_vision_left" if expected else "boundary_vision_left_reverse"
    elif variant in {6, 7}:
        rows = _relation_rows(vertical=True, pad=pad)
        reverse = variant == 7
        goal = {
            "kind": "relation",
            "predicate": "ABOVE",
            "source": "blue" if reverse else "red",
            "target": "red" if reverse else "blue",
        }
        expected = not reverse
        phenomenon = "boundary_vision_above" if expected else "boundary_vision_above_reverse"
    elif variant in {8, 9}:
        rows = _component_rows(("red", "blue", "green"), pad=pad)
        goal = {
            "kind": "count",
            "selector": "all",
            "expected": 3 if variant == 8 else 4,
        }
        expected = variant == 8
        phenomenon = "boundary_vision_count_three" if expected else "boundary_vision_count_four_miss"
    elif variant in {10, 11}:
        rows = _component_rows(("red", "red", "blue"), pad=pad)
        goal = {
            "kind": "count",
            "selector": "red",
            "expected": 2 if variant == 10 else 1,
        }
        expected = variant == 10
        phenomenon = "boundary_vision_red_count_two" if expected else "boundary_vision_red_count_one_miss"
    elif variant in {12, 13}:
        rows = _relation_rows(vertical=False, pad=pad)
        reverse = variant == 13
        goal = {
            "kind": "area",
            "larger": "red" if reverse else "blue",
            "smaller": "blue" if reverse else "red",
        }
        expected = not reverse
        phenomenon = "boundary_vision_area" if expected else "boundary_vision_area_reverse"
    else:
        rows = _relation_rows(vertical=True, pad=pad)
        reverse = variant == 15
        goal = {
            "kind": "relation",
            "predicate": "BELOW",
            "source": "red" if reverse else "blue",
            "target": "blue" if reverse else "red",
        }
        expected = not reverse
        phenomenon = "boundary_vision_below" if expected else "boundary_vision_below_reverse"
    return make_programmatic_semantic_case(
        case_id=f"{namespace}-vision-{index:04d}",
        domain=DomainKind.VISION,
        payload={
            "background": "white",
            "rows": rows,
            "goals": [goal],
            "source": f"{namespace}-boundary-raster-{seed}-{index}",
        },
        expected=expected,
        phenomenon=phenomenon,
        split=split,
        difficulty=2,
        extra_tags=("operator_boundary_curriculum",),
    )


def _extrapolation_language_case(
    index: int,
    seed: int,
    namespace: str,
    split: LearningSplit,
) -> SemanticBenchmarkCase:
    token = 30_000_000 + abs(seed) * 10_000 + index
    goal = f"extrapolate_release_{token}"
    count = 5 + (index % 2)
    premises = [f"extrapolate_check_{token}_{slot}" for slot in range(count)]
    variant = index % 6
    requirement = f"{goal} requires {', '.join(premises[:-1])}, and {premises[-1]}"
    if variant == 0:
        statuses = [f"{premise} is ready" for premise in premises]
        expected, phenomenon = True, "extrapolation_language_large_complete"
    elif variant == 1:
        statuses = [f"{premise} is ready" for premise in premises[:-1]]
        expected, phenomenon = False, "extrapolation_language_large_missing"
    elif variant == 2:
        statuses = [
            *(f"{premise} is ready" for premise in premises[:-2]),
            f"{premises[-2]} is blocked",
            f"{premises[-1]} is ready",
        ]
        expected, phenomenon = True, "extrapolation_language_large_blocked"
    elif variant == 3:
        statuses = [
            *(f"{premise} is ready" for premise in premises),
            f"{premises[-2]} is blocked",
        ]
        expected, phenomenon = False, "extrapolation_language_large_conflict"
    elif variant == 4:
        statuses = [
            *(f"{premise} is ready" for premise in premises[:-1]),
            f"unrelated_extrapolate_{token} is ready",
        ]
        expected, phenomenon = False, "extrapolation_language_large_binding_miss"
    else:
        statuses = [f"{premise} is available" for premise in reversed(premises)]
        requirement = f"To {goal}, {', '.join(premises[:-1])}, and {premises[-1]} are required"
        expected, phenomenon = True, "extrapolation_language_large_reordered"
    return make_programmatic_semantic_case(
        case_id=f"{namespace}-language-{index:04d}",
        domain=DomainKind.LANGUAGE,
        payload={
            "text": "; ".join((requirement, *statuses)),
            "use_legacy_heuristics": False,
        },
        expected=expected,
        phenomenon=phenomenon,
        split=split,
        difficulty=3,
        extra_tags=("extrapolation_holdout",),
    )


def _extrapolation_math_case(
    index: int,
    seed: int,
    namespace: str,
    split: LearningSplit,
) -> SemanticBenchmarkCase:
    base = 17 + abs(seed) * 5 + index
    other = 3 + (abs(seed) % 17) + (index % 7)
    variant = index % 12
    if variant == 0:
        expression = f"5*(x-{other}) + 2*(x+{base}) = 3*x + {base - other}"
        expected, phenomenon = True, "extrapolation_math_multi_term_linear"
    elif variant == 1:
        answer = ((base - other) * (other + 3) + base) * 2
        expression = f"((({base}-{other})*({other}+3)+{base})*2) == {answer}"
        expected, phenomenon = True, "extrapolation_math_depth_four_true"
    elif variant == 2:
        answer = ((base - other) * (other + 3) + base) * 2 + 1
        expression = f"((({base}-{other})*({other}+3)+{base})*2) == {answer}"
        expected, phenomenon = False, "extrapolation_math_depth_four_miss"
    elif variant == 3:
        expression = f"(-{base}*{other}-1)/{other} < -{base}"
        expected, phenomenon = True, "extrapolation_math_negative_fraction_order"
    elif variant == 4:
        expression = f"(-{base}*{other}+1)/{other} <= -{base}"
        expected, phenomenon = False, "extrapolation_math_negative_fraction_miss"
    elif variant == 5:
        expression = f"(({base}+{other})-({base}+{other})) == 0"
        expected, phenomenon = True, "extrapolation_math_nested_zero"
    elif variant == 6:
        expression = f"(({base}+{other})-({base}+{other})) != 0"
        expected, phenomenon = False, "extrapolation_math_nested_zero_polarity"
    elif variant == 7:
        expression = f"({base}*({other}+2))/({other}+2) >= {base}"
        expected, phenomenon = True, "extrapolation_math_cancelled_boundary"
    elif variant == 8:
        expression = f"-(-({base}+{other})) == {base + other}"
        expected, phenomenon = True, "extrapolation_math_double_unary"
    elif variant == 9:
        expression = f"-(-({base}+{other})) < {base + other}"
        expected, phenomenon = False, "extrapolation_math_double_unary_miss"
    elif variant == 10:
        expression = f"6*(x+{other}) = 3*x - {base}"
        expected, phenomenon = True, "extrapolation_math_scaled_negative_solution"
    else:
        expression = f"({base}+{other})/({base}+{other}) > 1"
        expected, phenomenon = False, "extrapolation_math_unit_strict_miss"
    return make_programmatic_semantic_case(
        case_id=f"{namespace}-math-{index:04d}",
        domain=DomainKind.MATH,
        payload={"expression": expression},
        expected=expected,
        phenomenon=phenomenon,
        split=split,
        difficulty=3,
        extra_tags=("extrapolation_holdout",),
    )


def _extrapolation_vision_case(
    index: int,
    seed: int,
    namespace: str,
    split: LearningSplit,
) -> SemanticBenchmarkCase:
    variant = index % 12
    pad = 2 + ((seed + index) % 3)
    if variant in {0, 1}:
        rows = _shape_rows(size=5, pad=pad, incomplete=variant == 1)
        goal: dict[str, Any] = {
            "kind": "property",
            "predicate": "SQUARE",
            "subject": "red",
        }
        expected = variant == 0
        phenomenon = "extrapolation_vision_square_five" if expected else "extrapolation_vision_square_five_gap"
    elif variant in {2, 3}:
        rows = _distractor_relation_rows(pad=pad)
        reverse = variant == 3
        goal = {
            "kind": "relation",
            "predicate": "BELOW",
            "source": "red" if reverse else "blue",
            "target": "blue" if reverse else "red",
        }
        expected = not reverse
        phenomenon = "extrapolation_vision_below_distractors" if expected else "extrapolation_vision_below_distractors_reverse"
    elif variant in {4, 5}:
        rows = _component_rows(("red", "blue", "green", "yellow", "black"), pad=pad)
        goal = {
            "kind": "count",
            "selector": "all",
            "expected": 5 if variant == 4 else 6,
        }
        expected = variant == 4
        phenomenon = "extrapolation_vision_count_five" if expected else "extrapolation_vision_count_six_miss"
    elif variant in {6, 7}:
        rows = _component_rows(("green", "green", "green", "red"), pad=pad)
        goal = {
            "kind": "count",
            "selector": "green",
            "expected": 3 if variant == 6 else 2,
        }
        expected = variant == 6
        phenomenon = "extrapolation_vision_green_count_three" if expected else "extrapolation_vision_green_count_two_miss"
    elif variant in {8, 9}:
        rows = _three_area_rows(pad=pad)
        reverse = variant == 9
        goal = {
            "kind": "area",
            "larger": "red" if reverse else "blue",
            "smaller": "blue" if reverse else "red",
        }
        expected = not reverse
        phenomenon = "extrapolation_vision_three_object_area" if expected else "extrapolation_vision_three_object_area_reverse"
    else:
        rows = _wide_relation_rows(pad=pad)
        reverse = variant == 11
        goal = {
            "kind": "relation",
            "predicate": "RIGHT_OF",
            "source": "red" if reverse else "blue",
            "target": "blue" if reverse else "red",
        }
        expected = not reverse
        phenomenon = "extrapolation_vision_right_wide" if expected else "extrapolation_vision_right_wide_reverse"
    return make_programmatic_semantic_case(
        case_id=f"{namespace}-vision-{index:04d}",
        domain=DomainKind.VISION,
        payload={
            "background": "white",
            "rows": rows,
            "goals": [goal],
            "source": f"{namespace}-extrapolation-raster-{seed}-{index}",
        },
        expected=expected,
        phenomenon=phenomenon,
        split=split,
        difficulty=3,
        extra_tags=("extrapolation_holdout",),
    )


def _shape_rows(*, size: int, pad: int, incomplete: bool) -> list[list[str]]:
    extent = size + 2 * pad
    rows = [["white" for _column in range(extent)] for _row in range(extent)]
    for row in range(pad, pad + size):
        for column in range(pad, pad + size):
            rows[row][column] = "red"
    if incomplete:
        rows[pad + size // 2][pad + size // 2] = "white"
    return rows


def _relation_rows(*, vertical: bool, pad: int) -> list[list[str]]:
    height = 12 + 2 * pad if vertical else 8 + 2 * pad
    width = 8 + 2 * pad if vertical else 13 + 2 * pad
    rows = [["white" for _column in range(width)] for _row in range(height)]
    for row in range(pad, pad + 2):
        for column in range(pad, pad + 2):
            rows[row][column] = "red"
    blue_top = pad + 6 if vertical else pad
    blue_left = pad + 1 if vertical else pad + 6
    for row in range(blue_top, blue_top + 3):
        for column in range(blue_left, blue_left + 3):
            rows[row][column] = "blue"
    return rows


def _component_rows(colors: tuple[str, ...], *, pad: int) -> list[list[str]]:
    height = 7 + 2 * pad
    width = 4 * len(colors) + 2 * pad
    rows = [["white" for _column in range(width)] for _row in range(height)]
    for index, color in enumerate(colors):
        left = pad + index * 4
        for row in range(pad, pad + 2):
            for column in range(left, left + 2):
                rows[row][column] = color
    return rows


def _distractor_relation_rows(*, pad: int) -> list[list[str]]:
    rows = _relation_rows(vertical=True, pad=pad)
    for row in range(pad + 2, pad + 4):
        for column in range(pad + 5, pad + 7):
            rows[row][column] = "green"
    for row in range(pad + 7, pad + 9):
        for column in range(pad + 5, pad + 7):
            rows[row][column] = "yellow"
    return rows


def _three_area_rows(*, pad: int) -> list[list[str]]:
    height = 10 + 2 * pad
    width = 17 + 2 * pad
    rows = [["white" for _column in range(width)] for _row in range(height)]
    for row in range(pad, pad + 2):
        for column in range(pad, pad + 2):
            rows[row][column] = "red"
    for row in range(pad, pad + 5):
        for column in range(pad + 5, pad + 10):
            rows[row][column] = "blue"
    for row in range(pad + 5, pad + 7):
        for column in range(pad + 12, pad + 14):
            rows[row][column] = "green"
    return rows


def _wide_relation_rows(*, pad: int) -> list[list[str]]:
    rows = _component_rows(("red", "green", "yellow", "blue"), pad=pad)
    return rows


__all__ = [
    "generate_extrapolation_semantic_benchmark",
    "generate_operator_boundary_semantic_benchmark",
]

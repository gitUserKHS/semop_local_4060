from __future__ import annotations

from typing import Any, Mapping

from .runtime import DomainKind
from .semantic_benchmark import SemanticBenchmark, SemanticBenchmarkCase
from .self_learning import LearningSplit


PROGRAMMATIC_ORACLE_ID = "semop:controlled-semantic-oracle-v1"


def generate_controlled_semantic_benchmark(
    *,
    per_domain: int,
    split: LearningSplit | str,
    seed: int,
    namespace: str,
) -> SemanticBenchmark:
    """Generate raw controlled LMV cases, not precomputed sensor vectors."""

    if per_domain <= 0:
        raise ValueError("controlled semantic case count must be positive")
    normalized_namespace = namespace.strip()
    if not normalized_namespace:
        raise ValueError("controlled semantic namespace must not be empty")
    resolved_split = LearningSplit(split)
    cases: list[SemanticBenchmarkCase] = []
    for index in range(per_domain):
        cases.append(
            _language_case(index, seed, normalized_namespace, resolved_split)
        )
        cases.append(_math_case(index, seed, normalized_namespace, resolved_split))
        cases.append(_vision_case(index, seed, normalized_namespace, resolved_split))
    return SemanticBenchmark(tuple(cases))


def generate_structural_semantic_benchmark(
    *,
    per_domain: int,
    split: LearningSplit | str,
    seed: int,
    namespace: str,
) -> SemanticBenchmark:
    """Generate composition and raster structures absent from the v1 train generator."""

    if per_domain <= 0:
        raise ValueError("structural semantic case count must be positive")
    normalized_namespace = namespace.strip()
    if not normalized_namespace:
        raise ValueError("structural semantic namespace must not be empty")
    resolved_split = LearningSplit(split)
    cases: list[SemanticBenchmarkCase] = []
    for index in range(per_domain):
        cases.append(
            _structural_language_case(
                index, seed, normalized_namespace, resolved_split
            )
        )
        cases.append(
            _structural_math_case(index, seed, normalized_namespace, resolved_split)
        )
        cases.append(
            _structural_vision_case(
                index, seed, normalized_namespace, resolved_split
            )
        )
    return SemanticBenchmark(tuple(cases))


def _language_case(
    index: int,
    seed: int,
    namespace: str,
    split: LearningSplit,
) -> SemanticBenchmarkCase:
    token = seed * 10_000 + index
    goal = f"deploy_{token}"
    first = f"tests_{token}"
    second = f"approval_{token}"
    variant = index % 6
    if variant == 0:
        text = (
            f"Goal: {goal}; Requires: {first}, {second}; "
            f"Satisfied: {first}; Satisfied: {second}"
        )
        expected, phenomenon = True, "conjunctive_requirements"
    elif variant == 1:
        text = (
            f"Goal: {goal}; Requires: {first}, {second}; Satisfied: {first}"
        )
        expected, phenomenon = False, "missing_conjunct"
    elif variant == 2:
        text = f"Goal: {goal}; Requires: {first}; Blocked: {first}"
        expected, phenomenon = True, "blocked_requirement"
    elif variant == 3:
        text = (
            f"Goal: {goal}; Requires: {first}; Satisfied: {first}; "
            f"Blocked: {first}"
        )
        expected, phenomenon = False, "contradictory_evidence"
    elif variant == 4:
        text = f"Requires: {first}; Goal: {goal}; Satisfied: {first}"
        expected, phenomenon = True, "discourse_order"
    else:
        text = f"Goal: {goal}; Requires: {first}; Satisfied: {second}"
        expected, phenomenon = False, "entity_binding"
    return _programmatic_case(
        case_id=f"{namespace}-language-{index:04d}",
        domain=DomainKind.LANGUAGE,
        payload={"text": text, "use_legacy_heuristics": False},
        expected=expected,
        phenomenon=phenomenon,
        split=split,
    )


def _math_case(
    index: int,
    seed: int,
    namespace: str,
    split: LearningSplit,
) -> SemanticBenchmarkCase:
    # Keep raw numeric inputs disjoint across seeded train/validation/test runs.
    base = 2 + abs(seed) * 3 + index
    other = 2 + (abs(seed) % 19) + (index % 7)
    variant = index % 6
    if variant == 0:
        expression = f"{base}*x + {other} = {base * 3 + other}"
        expected, phenomenon = True, "linear_equation"
    elif variant == 1:
        expression = f"{base} * ({other} + 2) == {base * (other + 2)}"
        expected, phenomenon = True, "operator_precedence"
    elif variant == 2:
        expression = f"{base} * ({other} + 2) == {base * (other + 2) + 1}"
        expected, phenomenon = False, "numeric_near_miss"
    elif variant == 3:
        expression = f"{base * other + 1} / {other} > {base}"
        expected, phenomenon = True, "exact_rational_order"
    elif variant == 4:
        expression = f"({base} + {other}) / 1 != {base + other}"
        expected, phenomenon = False, "relation_polarity"
    else:
        expression = f"{base * other} / {other} <= {base}"
        expected, phenomenon = True, "comparison_boundary"
    return _programmatic_case(
        case_id=f"{namespace}-math-{index:04d}",
        domain=DomainKind.MATH,
        payload={"expression": expression},
        expected=expected,
        phenomenon=phenomenon,
        split=split,
    )


def _vision_case(
    index: int,
    seed: int,
    namespace: str,
    split: LearningSplit,
) -> SemanticBenchmarkCase:
    variant = index % 8
    pad = 1 + ((seed + index) % 2)
    if variant in {0, 1}:
        rows = _single_shape_rows(pad=pad, incomplete=variant == 1)
        goal: dict[str, Any] = {
            "kind": "property",
            "predicate": "SQUARE",
            "subject": "red",
        }
        expected = variant == 0
        phenomenon = "shape_composition" if expected else "missing_pixel_near_miss"
    else:
        rows = _two_object_rows(pad=pad)
        if variant in {2, 3}:
            reversed_relation = variant == 3
            goal = {
                "kind": "relation",
                "predicate": "LEFT_OF",
                "source": "blue" if reversed_relation else "red",
                "target": "red" if reversed_relation else "blue",
            }
            expected = not reversed_relation
            phenomenon = "spatial_relation" if expected else "relation_direction"
        elif variant in {4, 5}:
            goal = {
                "kind": "count",
                "selector": "all",
                "expected": 2 if variant == 4 else 3,
            }
            expected = variant == 4
            phenomenon = "closed_world_count" if expected else "count_near_miss"
        else:
            reversed_area = variant == 7
            goal = {
                "kind": "area",
                "larger": "red" if reversed_area else "blue",
                "smaller": "blue" if reversed_area else "red",
            }
            expected = not reversed_area
            phenomenon = "pixel_area_order" if expected else "area_relation_direction"
    return _programmatic_case(
        case_id=f"{namespace}-vision-{index:04d}",
        domain=DomainKind.VISION,
        payload={
            "background": "white",
            "rows": rows,
            "goals": [goal],
            "source": f"{namespace}-raster-{seed}-{index}",
        },
        expected=expected,
        phenomenon=phenomenon,
        split=split,
    )


def _structural_language_case(
    index: int,
    seed: int,
    namespace: str,
    split: LearningSplit,
) -> SemanticBenchmarkCase:
    token = 10_000_000 + abs(seed) * 10_000 + index
    goal = f"release_{token}"
    first = f"security_{token}"
    second = f"audit_{token}"
    third = f"budget_{token}"
    variant = index % 7
    requirement = (
        f"To {goal}, {first}, {second}, and {third} are required"
    )
    if variant == 0:
        text = (
            f"{requirement}; {first} is ready; {second} is ready; "
            f"{third} is ready"
        )
        expected, phenomenon = True, "structural_language_three_conjuncts"
    elif variant == 1:
        text = f"{requirement}; {first} is ready; {second} is ready"
        expected, phenomenon = False, "structural_language_missing_third"
    elif variant == 2:
        text = (
            f"{requirement}; {first} is ready; {second} is blocked; "
            f"{third} is ready"
        )
        expected, phenomenon = True, "structural_language_block_among_three"
    elif variant == 3:
        text = (
            f"{requirement}; {first} is ready; {second} is ready; "
            f"{second} is blocked; {third} is ready"
        )
        expected, phenomenon = False, "structural_language_conflict_among_three"
    elif variant == 4:
        text = (
            f"{third} is present; {first} is available; {second} is met; "
            f"{goal} requires {first}, {second}, and {third}"
        )
        expected, phenomenon = True, "structural_language_sentence_reordering"
    elif variant == 5:
        wrong = f"budget_other_{token}"
        text = (
            f"{goal} requires {first}, {second}, and {third}; "
            f"{first} is ready; {second} is ready; {wrong} is ready"
        )
        expected, phenomenon = False, "structural_language_binding_near_miss"
    else:
        text = (
            f"{goal} requires {first}, {second}, and also {third}; "
            f"{third} is available; {first} is met; {second} is present; "
            f"{first} is ready"
        )
        expected, phenomenon = True, "structural_language_duplicate_evidence"
    return _programmatic_case(
        case_id=f"{namespace}-language-{index:04d}",
        domain=DomainKind.LANGUAGE,
        payload={"text": text, "use_legacy_heuristics": False},
        expected=expected,
        phenomenon=phenomenon,
        split=split,
        difficulty=2,
        extra_tags=("structural_holdout", "unseen_language_composition"),
    )


def _structural_math_case(
    index: int,
    seed: int,
    namespace: str,
    split: LearningSplit,
) -> SemanticBenchmarkCase:
    base = 5 + abs(seed) * 2 + index
    other = 3 + (abs(seed) % 11) + (index % 5)
    variant = index % 8
    if variant == 0:
        expression = f"3*(x - {other}) = x + {base}"
        expected, phenomenon = True, "structural_math_two_sided_linear"
    elif variant == 1:
        answer = ((base + other) * (other + 2)) - base
        expression = f"(({base} + {other}) * ({other} + 2) - {base}) == {answer}"
        expected, phenomenon = True, "structural_math_deep_exact_true"
    elif variant == 2:
        answer = ((base + other) * (other + 2)) - base + 1
        expression = f"(({base} + {other}) * ({other} + 2) - {base}) == {answer}"
        expected, phenomenon = False, "structural_math_deep_exact_near_miss"
    elif variant == 3:
        expression = f"-({base} + {other}) < -{base}"
        expected, phenomenon = True, "structural_math_negative_order"
    elif variant == 4:
        expression = f"-({base} + {other}) >= -{base}"
        expected, phenomenon = False, "structural_math_negative_order_reversal"
    elif variant == 5:
        expression = f"(({base} * {other}) + 1) / {other} > {base}"
        expected, phenomenon = True, "structural_math_nested_rational_order"
    elif variant == 6:
        expression = f"(({base} - {base}) * ({other} + 3)) == 1"
        expected, phenomenon = False, "structural_math_zero_product_near_miss"
    else:
        expression = (
            f"({base} * ({other} + 1) - {base} * {other}) >= {base}"
        )
        expected, phenomenon = True, "structural_math_derived_boundary"
    return _programmatic_case(
        case_id=f"{namespace}-math-{index:04d}",
        domain=DomainKind.MATH,
        payload={"expression": expression},
        expected=expected,
        phenomenon=phenomenon,
        split=split,
        difficulty=2,
        extra_tags=("structural_holdout", "deeper_exact_math"),
    )


def _structural_vision_case(
    index: int,
    seed: int,
    namespace: str,
    split: LearningSplit,
) -> SemanticBenchmarkCase:
    variant = index % 10
    pad = 2 + ((seed + index) % 2)
    if variant in {0, 1}:
        rows = _large_shape_rows(pad=pad, incomplete=variant == 1)
        goal: dict[str, Any] = {
            "kind": "property",
            "predicate": "SQUARE",
            "subject": "red",
        }
        expected = variant == 0
        phenomenon = (
            "structural_vision_three_by_three_square"
            if expected
            else "structural_vision_three_by_three_pixel_gap"
        )
    elif variant in {2, 3}:
        rows = _vertical_object_rows(pad=pad)
        reversed_relation = variant == 3
        goal = {
            "kind": "relation",
            "predicate": "ABOVE",
            "source": "blue" if reversed_relation else "red",
            "target": "red" if reversed_relation else "blue",
        }
        expected = not reversed_relation
        phenomenon = (
            "structural_vision_vertical_relation"
            if expected
            else "structural_vision_vertical_direction_reversal"
        )
    elif variant in {4, 5}:
        rows = _three_object_rows(pad=pad)
        goal = {
            "kind": "count",
            "selector": "all",
            "expected": 3 if variant == 4 else 4,
        }
        expected = variant == 4
        phenomenon = (
            "structural_vision_three_component_count"
            if expected
            else "structural_vision_three_component_near_miss"
        )
    elif variant in {6, 7}:
        rows = _repeated_color_rows(pad=pad)
        goal = {
            "kind": "count",
            "selector": "red",
            "expected": 2 if variant == 6 else 1,
        }
        expected = variant == 6
        phenomenon = (
            "structural_vision_repeated_color_count"
            if expected
            else "structural_vision_repeated_color_near_miss"
        )
    else:
        rows = _vertical_object_rows(pad=pad)
        reversed_area = variant == 9
        goal = {
            "kind": "area",
            "larger": "red" if reversed_area else "blue",
            "smaller": "blue" if reversed_area else "red",
        }
        expected = not reversed_area
        phenomenon = (
            "structural_vision_larger_canvas_area"
            if expected
            else "structural_vision_larger_canvas_area_reversal"
        )
    return _programmatic_case(
        case_id=f"{namespace}-vision-{index:04d}",
        domain=DomainKind.VISION,
        payload={
            "background": "white",
            "rows": rows,
            "goals": [goal],
            "source": f"{namespace}-structural-raster-{seed}-{index}",
        },
        expected=expected,
        phenomenon=phenomenon,
        split=split,
        difficulty=2,
        extra_tags=("structural_holdout", "unseen_raster_structure"),
    )


def _programmatic_case(
    *,
    case_id: str,
    domain: DomainKind,
    payload: Mapping[str, Any],
    expected: bool,
    phenomenon: str,
    split: LearningSplit,
    difficulty: int = 1,
    extra_tags: tuple[str, ...] = (),
) -> SemanticBenchmarkCase:
    return SemanticBenchmarkCase.from_mapping(
        {
            "case_id": case_id,
            "domain": domain.value,
            "payload": dict(payload),
            "expected_solved": expected,
            "phenomenon": phenomenon,
            "rationale": (
                "controlled generator independently specifies whether the exact "
                "typed goal is supported by the raw input"
            ),
            "split": split.value,
            "difficulty": difficulty,
            "author": PROGRAMMATIC_ORACLE_ID,
            "tags": [
                "programmatic_oracle",
                "controlled_semantics",
                "closed_world_support",
                *extra_tags,
            ],
        }
    )


def _single_shape_rows(*, pad: int, incomplete: bool) -> list[list[str]]:
    size = 5 + pad
    rows = [["white" for _column in range(size)] for _row in range(size)]
    top = pad
    left = pad
    for row in range(top, top + 2):
        for column in range(left, left + 2):
            rows[row][column] = "red"
    if incomplete:
        rows[top + 1][left + 1] = "white"
    return rows


def _two_object_rows(*, pad: int) -> list[list[str]]:
    height = 6 + pad
    width = 11 + pad
    rows = [["white" for _column in range(width)] for _row in range(height)]
    top = pad
    for row in range(top, top + 2):
        for column in range(pad, pad + 2):
            rows[row][column] = "red"
        for column in range(pad + 4, pad + 7):
            rows[row][column] = "blue"
    return rows


def _large_shape_rows(*, pad: int, incomplete: bool) -> list[list[str]]:
    size = 8 + 2 * pad
    rows = [["white" for _column in range(size)] for _row in range(size)]
    for row in range(pad, pad + 3):
        for column in range(pad, pad + 3):
            rows[row][column] = "red"
    if incomplete:
        rows[pad + 1][pad + 1] = "white"
    return rows


def _vertical_object_rows(*, pad: int) -> list[list[str]]:
    height = 11 + 2 * pad
    width = 8 + 2 * pad
    rows = [["white" for _column in range(width)] for _row in range(height)]
    for row in range(pad, pad + 2):
        for column in range(pad, pad + 2):
            rows[row][column] = "red"
    for row in range(pad + 5, pad + 8):
        for column in range(pad + 1, pad + 4):
            rows[row][column] = "blue"
    return rows


def _three_object_rows(*, pad: int) -> list[list[str]]:
    height = 8 + 2 * pad
    width = 14 + 2 * pad
    rows = [["white" for _column in range(width)] for _row in range(height)]
    for offset, color in ((0, "red"), (4, "blue"), (8, "green")):
        for row in range(pad, pad + 2):
            for column in range(pad + offset, pad + offset + 2):
                rows[row][column] = color
    return rows


def _repeated_color_rows(*, pad: int) -> list[list[str]]:
    height = 9 + 2 * pad
    width = 15 + 2 * pad
    rows = [["white" for _column in range(width)] for _row in range(height)]
    for offset, color in ((0, "red"), (5, "red"), (10, "blue")):
        for row in range(pad, pad + 2):
            for column in range(pad + offset, pad + offset + 2):
                rows[row][column] = color
    return rows


__all__ = [
    "PROGRAMMATIC_ORACLE_ID",
    "generate_controlled_semantic_benchmark",
    "generate_structural_semantic_benchmark",
]

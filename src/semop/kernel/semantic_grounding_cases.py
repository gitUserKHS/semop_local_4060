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


def _programmatic_case(
    *,
    case_id: str,
    domain: DomainKind,
    payload: Mapping[str, Any],
    expected: bool,
    phenomenon: str,
    split: LearningSplit,
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
            "difficulty": 1,
            "author": PROGRAMMATIC_ORACLE_ID,
            "tags": [
                "programmatic_oracle",
                "controlled_semantics",
                "closed_world_support",
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


__all__ = ["PROGRAMMATIC_ORACLE_ID", "generate_controlled_semantic_benchmark"]

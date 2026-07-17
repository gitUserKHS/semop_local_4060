from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import random
from typing import Any, Callable

from .domains import (
    DomainInstance,
    RasterImage,
    SceneThresholdAdapter,
    SceneThresholdProblem,
    VisionProblem,
    VisionRelationGoal,
    VisionWorldAdapter,
    make_hidden_premise_instance,
    parse_arithmetic_expression,
    parse_geometry_dsl,
    parse_grid_problem,
)
from .model import Fact, FactStatus, Rule, WorldState


@dataclass(frozen=True)
class SyntheticProblem:
    problem_id: str
    domain: str
    instance: DomainInstance


@dataclass
class _SyntheticEntity:
    id: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class _SyntheticRelation:
    source: str
    relation: str
    target: str
    confidence: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class _SyntheticWorld:
    query: str
    entities: list[_SyntheticEntity]
    relations: list[_SyntheticRelation]
    metadata: dict[str, Any] = field(default_factory=dict)
    inferred_steps: list[str] = field(default_factory=list)
    audit_trace: list[str] = field(default_factory=list)


def generate_symbolic_curriculum(
    examples_per_domain: int,
    *,
    seed: int = 0,
    curriculum: str = "operator-v1",
    domains: Sequence[str] | None = None,
) -> tuple[SyntheticProblem, ...]:
    """Generate a bounded deterministic curriculum from replayable domains."""

    if not 1 <= examples_per_domain <= 5_000:
        raise ValueError("examples_per_domain must be between 1 and 5,000")
    try:
        available = _curriculum_builders()[curriculum]
    except KeyError as exc:
        raise ValueError(f"unknown symbolic curriculum: {curriculum}") from exc
    available_domains = tuple(domain for domain, _builder in available)
    selected = available_domains if domains is None else tuple(domains)
    if not selected:
        raise ValueError("symbolic curriculum requires at least one domain")
    if len(set(selected)) != len(selected):
        raise ValueError("symbolic curriculum domains must be unique")
    unknown = sorted(set(selected) - set(available_domains))
    if unknown:
        raise ValueError(
            f"domains {unknown} are not in {curriculum}; "
            f"available={list(available_domains)}"
        )
    selected_set = set(selected)
    builders = tuple(
        (domain, builder)
        for domain, builder in available
        if domain in selected_set
    )
    rng_by_domain = {
        domain: random.Random(f"{seed}:{curriculum}:{domain}")
        for domain, _builder in builders
    }
    examples: list[SyntheticProblem] = []
    for index in range(examples_per_domain):
        examples.extend(
            builder(index, rng_by_domain[domain])
            for domain, builder in builders
        )
    return tuple(examples)


def symbolic_curriculum_domains(curriculum: str) -> tuple[str, ...]:
    try:
        return tuple(
            domain for domain, _builder in _curriculum_builders()[curriculum]
        )
    except KeyError as exc:
        raise ValueError(f"unknown symbolic curriculum: {curriculum}") from exc


def _curriculum_builders() -> dict[
    str,
    tuple[tuple[str, Callable[[int, random.Random], SyntheticProblem]], ...],
]:
    return {
        "operator-v1": (
            ("geometry", _geometry_problem),
            ("hidden_premise", _premise_problem),
            ("grid", _grid_problem),
        ),
        "language-math-vision": (
            ("language", _language_problem),
            ("math", _arithmetic_problem),
            ("vision", _vision_problem),
        ),
        "language-math-vision-composed": (
            ("language", _language_problem),
            ("math", _arithmetic_problem),
            ("vision", _vision_problem),
            ("composed", _composed_problem),
        ),
    }


def _geometry_problem(index: int, rng: random.Random) -> SyntheticProblem:
    suffix = f"{index}_{rng.randrange(1_000_000)}"
    kind = index % 3
    if kind == 0:
        text = (
            f"point A{suffix}, B{suffix}, M{suffix}\n"
            f"assume midpoint(M{suffix}, A{suffix}, B{suffix})\n"
            f"prove equal_length(segment(A{suffix}, M{suffix}), "
            f"segment(M{suffix}, B{suffix}))\n"
        )
    elif kind == 1:
        text = (
            f"segment S{suffix}, T{suffix}, U{suffix}\n"
            f"assume equal_length(S{suffix}, T{suffix})\n"
            f"assume equal_length(T{suffix}, U{suffix})\n"
            f"prove equal_length(S{suffix}, U{suffix})\n"
        )
    else:
        text = (
            f"line X{suffix}, Y{suffix}, Z{suffix}\n"
            f"assume parallel(X{suffix}, Y{suffix})\n"
            f"assume perpendicular(X{suffix}, Z{suffix})\n"
            f"prove perpendicular(Y{suffix}, Z{suffix})\n"
        )
    return SyntheticProblem(
        f"geometry-{index}", "geometry", parse_geometry_dsl(text)
    )


def _premise_problem(index: int, rng: random.Random) -> SyntheticProblem:
    requirement_count = 1 + index % 3
    requirements = [f"condition_{index}_{slot}" for slot in range(requirement_count)]
    goal = f"goal_{index}_{rng.randrange(1_000_000)}"
    return SyntheticProblem(
        f"hidden-premise-{index}",
        "hidden_premise",
        make_hidden_premise_instance(
            goal, requirements, satisfied=requirements
        ),
    )


def _grid_problem(index: int, rng: random.Random) -> SyntheticProblem:
    length = 2 + index % 5
    if index % 2 == 0:
        grid = "S" + "." * (length - 1) + "G"
    else:
        grid = "\n".join(["S", *("." for _ in range(length - 1)), "G"])
    return SyntheticProblem(f"grid-{index}", "grid", parse_grid_problem(grid))


def _language_problem(index: int, rng: random.Random) -> SyntheticProblem:
    requirement_count = 1 + index % 3
    requirements = [f"premise_{index}_{slot}" for slot in range(requirement_count)]
    goal = f"language_goal_{index}_{rng.randrange(1_000_000)}"
    instance = make_hidden_premise_instance(
        goal, requirements, satisfied=requirements
    )
    return SyntheticProblem(
        f"language-{index}",
        "language",
        _with_hard_negative_distractors(instance, index),
    )


def _arithmetic_problem(index: int, rng: random.Random) -> SyntheticProblem:
    left = 1 + rng.randrange(9)
    right = 1 + rng.randrange(9)
    extra = 1 + rng.randrange(5)
    kind = index % 3
    if kind == 0:
        expression = f"{left} + {right}"
    elif kind == 1:
        expression = f"({left} + {right}) * {extra}"
    else:
        expression = f"{left}/{right} - {extra}"
    instance = parse_arithmetic_expression(expression)
    return SyntheticProblem(
        f"math-{index}",
        "math",
        _with_hard_negative_distractors(instance, index),
    )


def _vision_problem(index: int, rng: random.Random) -> SyntheticProblem:
    relation = ("LEFT_OF", "ABOVE", "INSIDE")[index % 3]
    suffix = f"{index}_{rng.randrange(1_000_000)}"
    names = tuple(f"object_{slot}_{suffix}" for slot in range(3))
    attributes = {"geometry_verified": True}
    world = _SyntheticWorld(
        query=f"Does {relation} compose?",
        entities=[
            _SyntheticEntity(name, {"verified": True}) for name in names
        ],
        relations=[
            _SyntheticRelation(
                names[0], relation, names[1], 0.95, attributes
            ),
            _SyntheticRelation(
                names[1], relation, names[2], 0.95, attributes
            ),
        ],
    )
    instance = VisionWorldAdapter().adapt(
        VisionProblem(
            world,
            (VisionRelationGoal(relation, names[0], names[2]),),
        )
    )
    return SyntheticProblem(
        f"vision-{index}",
        "vision",
        _with_hard_negative_distractors(instance, index),
    )


def _composed_problem(index: int, rng: random.Random) -> SyntheticProblem:
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    image = RasterImage.from_rows(
        (
            [white] * 11,
            [white, red, red, white, green, green, white, blue, blue, white, white],
            [white, red, red, white, green, green, white, blue, blue, white, white],
            [white] * 11,
        )
    )
    property_name = f"verified_scene_{index}_{rng.randrange(1_000_000)}"
    kind = index % 3
    if kind == 0:
        text = (
            "If the count of red objects is at least 1 and the count of blue "
            f"objects is at least 1, the scene is {property_name}. "
            f"Prove: the scene is {property_name}."
        )
    elif kind == 1:
        text = (
            "If the count of green objects is greater than 0 and the count of "
            f"blue objects is at most 1, the scene is {property_name}. "
            f"Prove: the scene is {property_name}."
        )
    else:
        text = (
            "If the count of red objects is at least 1 and the count of red "
            f"objects is at most 1, the scene is {property_name}. "
            f"Prove: the scene is {property_name}."
        )
    instance = SceneThresholdAdapter().adapt(SceneThresholdProblem(image, text))
    return SyntheticProblem(
        f"composed-{index}",
        "composed",
        _with_hard_negative_distractors(instance, index),
    )


def _with_hard_negative_distractors(
    instance: DomainInstance,
    index: int,
    *,
    count: int = 4,
) -> DomainInstance:
    registry = instance.registry
    entity = registry.types.resolve("Entity")
    registry.register_predicate("DISTRACTOR_TRIGGER", (entity,))
    target = instance.goals[0].atom
    facts: list[Fact] = []
    for slot in range(count):
        symbol = registry.symbol(f"distractor_{index}_{slot}", entity)
        trigger = registry.atom("DISTRACTOR_TRIGGER", symbol)
        facts.append(Fact(trigger, FactStatus.OBSERVED, "synthetic_distractor"))
        wrong_arguments = tuple(
            registry.symbol(
                f"wrong_goal_{index}_{slot}_{argument_index}",
                argument.type,
            )
            for argument_index, argument in enumerate(target.arguments)
        )
        registry.register_operator(
            Rule(
                name=f"distractor_action_{index}_{slot}",
                parameters=(),
                preconditions=(trigger,),
                effects=(registry.atom(target.predicate, *wrong_arguments),),
                description_ko="목표와 무관한 합성 distractor를 실행한다.",
            ),
            family="search",
            tags=("hard_negative", "synthetic"),
        )
    return replace(
        instance,
        state=WorldState(
            facts=instance.state.facts + tuple(facts),
            depth=instance.state.depth,
            path_cost=instance.state.path_cost,
        ),
    )

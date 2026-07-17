from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import random
from typing import Any, Callable

from .domains import (
    DomainInstance,
    LanguageLogicAdapter,
    LanguageTextAdapter,
    LanguageTextProblem,
    RasterImage,
    RasterVisionAdapter,
    RasterVisionProblem,
    SceneThresholdAdapter,
    SceneThresholdProblem,
    VisionAreaGoal,
    VisionCountGoal,
    VisionProblem,
    VisionPropertyGoal,
    VisionRelationGoal,
    VisionWorldAdapter,
    make_hidden_premise_instance,
    parse_arithmetic_expression,
    parse_geometry_dsl,
    parse_grid_problem,
    parse_linear_equation,
    parse_numeric_comparison,
)
from .model import Fact, FactStatus, Goal, Rule, WorldState


@dataclass(frozen=True)
class SyntheticProblem:
    problem_id: str
    domain: str
    instance: DomainInstance
    capability: str = ""
    structure_key: str = ""
    difficulty: int = 1

    def __post_init__(self) -> None:
        if not self.problem_id.strip() or not self.domain.strip():
            raise ValueError("synthetic problem id and domain must not be empty")
        if self.difficulty <= 0:
            raise ValueError("synthetic problem difficulty must be positive")


@dataclass(frozen=True)
class SyntheticCurriculumSplit:
    """A capability-level split, not a rename-only random split."""

    training: tuple[SyntheticProblem, ...]
    heldout: tuple[SyntheticProblem, ...]
    negative_controls: tuple[SyntheticProblem, ...]
    training_structures: tuple[str, ...]
    heldout_structures: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.training_structures) & set(self.heldout_structures)
        if overlap:
            raise ValueError(
                "synthetic structural split overlaps: " + ", ".join(sorted(overlap))
            )


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


def generate_lmv_structural_transfer_split(
    examples_per_structure: int,
    *,
    seed: int = 0,
) -> SyntheticCurriculumSplit:
    """Hold out one full adapter/operator composition in each primary domain."""

    if not 1 <= examples_per_structure <= 1_000:
        raise ValueError("examples_per_structure must be between 1 and 1,000")
    builders = _lmv_transfer_builders()
    training: list[SyntheticProblem] = []
    heldout: list[SyntheticProblem] = []
    for domain, variants in builders:
        for variant_index, (capability, builder) in enumerate(variants):
            target = heldout if variant_index == len(variants) - 1 else training
            rng = random.Random(
                f"{seed}:lmv-structural-transfer:{domain}:{capability}"
            )
            target.extend(
                builder(index, rng)
                for index in range(examples_per_structure)
            )
    controls = generate_symbolic_negative_controls(heldout)
    return SyntheticCurriculumSplit(
        training=tuple(training),
        heldout=tuple(heldout),
        negative_controls=controls,
        training_structures=tuple(
            sorted({problem.structure_key for problem in training})
        ),
        heldout_structures=tuple(
            sorted({problem.structure_key for problem in heldout})
        ),
    )


def generate_symbolic_negative_controls(
    problems: Sequence[SyntheticProblem],
) -> tuple[SyntheticProblem, ...]:
    """Create deterministic unsolved controls without adding verifier facts."""

    controls: list[SyntheticProblem] = []
    for problem in problems:
        instance = problem.instance
        if not instance.goals:
            raise ValueError(
                f"negative control requires explicit goals: {problem.problem_id}"
            )
        false_goals = []
        safe_id = problem.problem_id.replace("-", "_")
        for goal_index, goal in enumerate(instance.goals):
            wrong_arguments = tuple(
                instance.registry.symbol(
                    f"negative_{safe_id}_{goal_index}_{argument_index}",
                    argument.type,
                )
                for argument_index, argument in enumerate(goal.atom.arguments)
            )
            false_goals.append(
                Goal(
                    instance.registry.atom(goal.atom.predicate, *wrong_arguments),
                    label="intentionally unreachable typed target",
                )
            )
        control = replace(instance, goals=tuple(false_goals))
        controls.append(
            SyntheticProblem(
                problem_id=f"negative-{problem.problem_id}",
                domain=problem.domain,
                instance=control,
                capability=problem.capability,
                structure_key=problem.structure_key,
                difficulty=problem.difficulty,
            )
        )
    return tuple(controls)


def _lmv_transfer_builders() -> tuple[
    tuple[
        str,
        tuple[
            tuple[str, Callable[[int, random.Random], SyntheticProblem]],
            ...,
        ],
    ],
    ...,
]:
    return (
        (
            "language",
            (
                ("premise_readiness", _transfer_language_readiness),
                ("inheritance_chain", _transfer_language_inheritance),
                ("conjunctive_rule_chain", _transfer_language_conjunction),
            ),
        ),
        (
            "math",
            (
                ("nested_arithmetic", _transfer_math_arithmetic),
                ("linear_equation", _transfer_math_equation),
                ("exact_comparison", _transfer_math_comparison),
            ),
        ),
        (
            "vision",
            (
                ("spatial_transitivity", _transfer_vision_relation),
                ("pixel_shape", _transfer_vision_shape),
                ("pixel_quantification", _transfer_vision_quantification),
            ),
        ),
    )


def _transfer_language_readiness(
    index: int,
    rng: random.Random,
) -> SyntheticProblem:
    suffix = f"{index}{rng.randrange(1_000_000)}"
    goal = f"deploy{suffix}"
    first = f"tests{suffix}"
    second = f"approval{suffix}"
    instance = LanguageTextAdapter().adapt(
        LanguageTextProblem(
            f"Goal: {goal}; Requires: {first}, {second}; "
            f"Satisfied: {first}; Satisfied: {second}",
            use_legacy_heuristics=False,
        )
    )
    return _transfer_problem(
        f"transfer-language-readiness-{index}",
        "language",
        "premise_readiness",
        instance,
        difficulty=3,
        distractor_index=10_000 + index,
    )


def _transfer_language_inheritance(
    index: int,
    rng: random.Random,
) -> SyntheticProblem:
    suffix = f"{index}{rng.randrange(1_000_000)}"
    child = f"coder{suffix}"
    middle = f"person{suffix}"
    parent = f"mortal{suffix}"
    subject = f"ada{suffix}"
    instance = LanguageLogicAdapter().adapt(
        f"Every {child} is a {middle}. Every {middle} is a {parent}. "
        f"{subject} is a {child}. Prove: {subject} is a {parent}."
    )
    return _transfer_problem(
        f"transfer-language-inheritance-{index}",
        "language",
        "inheritance_chain",
        instance,
        difficulty=2,
        distractor_index=20_000 + index,
    )


def _transfer_language_conjunction(
    index: int,
    rng: random.Random,
) -> SyntheticProblem:
    suffix = f"{index}{rng.randrange(1_000_000)}"
    red = f"red{suffix}"
    square = f"square{suffix}"
    marker = f"marker{suffix}"
    visible = f"visible{suffix}"
    target = f"target{suffix}"
    subject = f"tile{suffix}"
    instance = LanguageLogicAdapter().adapt(
        f"Rule: {red} & {square} -> {marker}. "
        f"Rule: {marker} & {visible} -> {target}. "
        f"{subject} is {red}. {subject} is {square}. "
        f"{subject} is {visible}. Prove: {subject} is a {target}."
    )
    return _transfer_problem(
        f"transfer-language-conjunction-{index}",
        "language",
        "conjunctive_rule_chain",
        instance,
        difficulty=3,
        distractor_index=30_000 + index,
    )


def _transfer_math_arithmetic(
    index: int,
    rng: random.Random,
) -> SyntheticProblem:
    left = 1 + rng.randrange(9)
    right = 1 + rng.randrange(9)
    factor = 2 + rng.randrange(4)
    instance = parse_arithmetic_expression(f"({left} + {right}) * {factor}")
    return _transfer_problem(
        f"transfer-math-arithmetic-{index}",
        "math",
        "nested_arithmetic",
        instance,
        difficulty=5,
        distractor_index=40_000 + index,
    )


def _transfer_math_equation(
    index: int,
    rng: random.Random,
) -> SyntheticProblem:
    solution = 1 + rng.randrange(9)
    coefficient = 2 + rng.randrange(5)
    offset = 1 + rng.randrange(7)
    total = coefficient * solution + offset
    instance = parse_linear_equation(
        f"{coefficient}*x + {offset} = {total}"
    )
    return _transfer_problem(
        f"transfer-math-equation-{index}",
        "math",
        "linear_equation",
        instance,
        difficulty=2,
        distractor_index=50_000 + index,
    )


def _transfer_math_comparison(
    index: int,
    rng: random.Random,
) -> SyntheticProblem:
    left = 1 + rng.randrange(9)
    right = 1 + rng.randrange(9)
    instance = parse_numeric_comparison(f"{left} + {right} == {left + right}")
    return _transfer_problem(
        f"transfer-math-comparison-{index}",
        "math",
        "exact_comparison",
        instance,
        difficulty=5,
        distractor_index=60_000 + index,
    )


def _transfer_vision_relation(
    index: int,
    rng: random.Random,
) -> SyntheticProblem:
    suffix = f"{index}_{rng.randrange(1_000_000)}"
    names = tuple(f"visual_{slot}_{suffix}" for slot in range(3))
    attributes = {"geometry_verified": True}
    world = _SyntheticWorld(
        query="Does LEFT_OF compose?",
        entities=[
            _SyntheticEntity(name, {"verified": True}) for name in names
        ],
        relations=[
            _SyntheticRelation(names[0], "LEFT_OF", names[1], 0.99, attributes),
            _SyntheticRelation(names[1], "LEFT_OF", names[2], 0.99, attributes),
        ],
    )
    instance = VisionWorldAdapter().adapt(
        VisionProblem(
            world,
            (VisionRelationGoal("LEFT_OF", names[0], names[2]),),
        )
    )
    return _transfer_problem(
        f"transfer-vision-relation-{index}",
        "vision",
        "spatial_transitivity",
        instance,
        difficulty=2,
        distractor_index=70_000 + index,
    )


def _transfer_vision_shape(
    index: int,
    _rng: random.Random,
) -> SyntheticProblem:
    white = (255, 255, 255)
    red = (255, 0, 0)
    blue = (0, 0, 255)
    image = RasterImage.from_rows(
        (
            [white] * 7,
            [white, red, red, white, blue, white, white],
            [white, red, red, white, white, white, white],
            [white] * 7,
        )
    )
    instance = RasterVisionAdapter().adapt(
        RasterVisionProblem(
            image,
            (VisionPropertyGoal("SQUARE", "red"),),
        )
    )
    return _transfer_problem(
        f"transfer-vision-shape-{index}",
        "vision",
        "pixel_shape",
        instance,
        difficulty=2,
        distractor_index=80_000 + index,
    )


def _transfer_vision_quantification(
    index: int,
    _rng: random.Random,
) -> SyntheticProblem:
    white = (255, 255, 255)
    red = (255, 0, 0)
    blue = (0, 0, 255)
    image = RasterImage.from_rows(
        (
            [white] * 8,
            [white, red, red, white, white, blue, white, white],
            [white, red, red, white, white, blue, white, white],
            [white] * 8,
        )
    )
    instance = RasterVisionAdapter().adapt(
        RasterVisionProblem(
            image,
            (
                VisionCountGoal("all", 2),
                VisionAreaGoal("red", "blue"),
            ),
        )
    )
    return _transfer_problem(
        f"transfer-vision-quantification-{index}",
        "vision",
        "pixel_quantification",
        instance,
        difficulty=2,
        distractor_index=90_000 + index,
    )


def _transfer_problem(
    problem_id: str,
    domain: str,
    capability: str,
    instance: DomainInstance,
    *,
    difficulty: int,
    distractor_index: int,
) -> SyntheticProblem:
    structure_key = f"{domain}:{capability}"
    tagged = replace(
        instance,
        metadata={
            **instance.metadata,
            "capability": capability,
            "structure_key": structure_key,
            "synthetic_transfer": True,
        },
    )
    return SyntheticProblem(
        problem_id=problem_id,
        domain=domain,
        instance=_with_hard_negative_distractors(tagged, distractor_index),
        capability=capability,
        structure_key=structure_key,
        difficulty=difficulty,
    )


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
                description_ko="목표와 무관한 합성 탐색 후보를 실행한다.",
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

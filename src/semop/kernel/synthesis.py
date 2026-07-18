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


_SEMANTIC_FLOW_TRAIN_VARIANTS = (
    "single_threshold",
    "dual_selector_conjunction",
)
_SEMANTIC_FLOW_HELDOUT_VARIANTS = (
    "reused_measurement_bounds",
    "triple_mixed_comparison",
)
_SEMANTIC_FLOW_VARIANTS = (
    _SEMANTIC_FLOW_TRAIN_VARIANTS + _SEMANTIC_FLOW_HELDOUT_VARIANTS
)


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


@dataclass(frozen=True)
class MacroReuseCurriculumSplit:
    """Three-way split for learning reusable primitive operator programs."""

    training: tuple[SyntheticProblem, ...]
    validation: tuple[SyntheticProblem, ...]
    heldout: tuple[SyntheticProblem, ...]
    negative_controls: tuple[SyntheticProblem, ...]

    def __post_init__(self) -> None:
        groups = (self.training, self.validation, self.heldout)
        identifiers = [
            problem.problem_id for group in groups for problem in group
        ]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("macro reuse split problem ids must be disjoint")
        expected_domains = {"language", "math", "vision"}
        for label, group in zip(
            ("training", "validation", "heldout"),
            groups,
            strict=True,
        ):
            domains = {problem.domain for problem in group}
            if domains != expected_domains:
                raise ValueError(
                    f"macro reuse {label} must cover language, math, and vision"
                )


@dataclass(frozen=True)
class HierarchicalBrainCurriculumSplit:
    """Disjoint component and joint splits for a controller-plus-macro brain."""

    controller_training: tuple[SyntheticProblem, ...]
    controller_heldout: tuple[SyntheticProblem, ...]
    controller_negative_controls: tuple[SyntheticProblem, ...]
    macro_training: tuple[SyntheticProblem, ...]
    macro_validation: tuple[SyntheticProblem, ...]
    macro_heldout: tuple[SyntheticProblem, ...]
    macro_negative_controls: tuple[SyntheticProblem, ...]
    joint_heldout: tuple[SyntheticProblem, ...]
    joint_negative_controls: tuple[SyntheticProblem, ...]

    def __post_init__(self) -> None:
        positive_groups = (
            self.controller_training,
            self.controller_heldout,
            self.macro_training,
            self.macro_validation,
            self.macro_heldout,
            self.joint_heldout,
        )
        identifiers = [
            problem.problem_id
            for group in positive_groups
            for problem in group
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("hierarchical brain split problem ids must be disjoint")
        expected_domains = {"language", "math", "vision"}
        if any(
            {problem.domain for problem in group} != expected_domains
            for group in positive_groups
        ):
            raise ValueError(
                "hierarchical brain positive splits must cover all three domains"
            )
        controller_names = {
            str(problem.instance.metadata["completion_operator"])
            for problem in self.controller_training + self.controller_heldout
        }
        joint_names = {
            str(problem.instance.metadata["completion_operator"])
            for problem in self.joint_heldout
        }
        if controller_names.intersection(joint_names):
            raise ValueError(
                "hierarchical controller and heldout operator names must be disjoint"
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


def generate_semantic_flow_transfer_split(
    examples_per_structure: int,
    *,
    seed: int = 0,
) -> SyntheticCurriculumSplit:
    """Hold out new vision -> math -> language operator programs."""

    if not 1 <= examples_per_structure <= 1_000:
        raise ValueError("examples_per_structure must be between 1 and 1,000")
    training: list[SyntheticProblem] = []
    heldout: list[SyntheticProblem] = []
    for target, variants in (
        (training, _SEMANTIC_FLOW_TRAIN_VARIANTS),
        (heldout, _SEMANTIC_FLOW_HELDOUT_VARIANTS),
    ):
        for variant in variants:
            rng = random.Random(f"{seed}:semantic-flow:{variant}")
            target.extend(
                _semantic_flow_problem(index, rng, variant)
                for index in range(examples_per_structure)
            )
    controls = generate_symbolic_negative_controls(heldout)
    return SyntheticCurriculumSplit(
        training=tuple(training),
        heldout=tuple(heldout),
        negative_controls=controls,
        training_structures=tuple(
            f"composed:semantic_flow:{variant}"
            for variant in _SEMANTIC_FLOW_TRAIN_VARIANTS
        ),
        heldout_structures=tuple(
            f"composed:semantic_flow:{variant}"
            for variant in _SEMANTIC_FLOW_HELDOUT_VARIANTS
        ),
    )


def generate_macro_reuse_transfer_split(
    examples_per_domain: int = 3,
    *,
    validation_per_domain: int = 1,
    heldout_per_domain: int = 1,
    seed: int = 0,
    grounding_offset: int = 0,
) -> MacroReuseCurriculumSplit:
    """Generate disjoint grounding splits for retained macro activation."""

    if not 3 <= examples_per_domain <= 1_000:
        raise ValueError("macro training examples_per_domain must be between 3 and 1,000")
    if not 1 <= validation_per_domain <= 1_000:
        raise ValueError("validation_per_domain must be between 1 and 1,000")
    if not 1 <= heldout_per_domain <= 1_000:
        raise ValueError("heldout_per_domain must be between 1 and 1,000")
    if not 0 <= grounding_offset < 100_000:
        raise ValueError("grounding_offset must be between 0 and 99,999")
    builders = (
        ("language", _transfer_language_inheritance),
        ("math", _transfer_math_equation),
        ("vision", _macro_vision_shape),
    )
    partitions = (
        ("training", examples_per_domain, grounding_offset),
        ("validation", validation_per_domain, 100_000 + grounding_offset),
        ("heldout", heldout_per_domain, 200_000 + grounding_offset),
    )
    generated: dict[str, list[SyntheticProblem]] = {
        name: [] for name, _count, _offset in partitions
    }
    for partition, count, offset in partitions:
        for domain, builder in builders:
            rng = random.Random(f"{seed}:macro-reuse:{partition}:{domain}")
            for local_index in range(count):
                source = builder(offset + local_index, rng)
                generated[partition].append(
                    _retag_macro_problem(source, partition, local_index)
                )
    heldout = tuple(generated["heldout"])
    return MacroReuseCurriculumSplit(
        training=tuple(generated["training"]),
        validation=tuple(generated["validation"]),
        heldout=heldout,
        negative_controls=generate_symbolic_negative_controls(heldout),
    )


def generate_hierarchical_brain_transfer_split(
    examples_per_domain: int = 3,
    *,
    validation_per_domain: int = 1,
    heldout_per_domain: int = 1,
    seed: int = 0,
) -> HierarchicalBrainCurriculumSplit:
    """Build component-level and longer joint programs with no name overlap."""

    macro = generate_macro_reuse_transfer_split(
        examples_per_domain,
        validation_per_domain=validation_per_domain,
        heldout_per_domain=heldout_per_domain,
        seed=seed,
    )
    controller_source = generate_macro_reuse_transfer_split(
        examples_per_domain,
        validation_per_domain=validation_per_domain,
        heldout_per_domain=heldout_per_domain,
        seed=seed + 1,
        grounding_offset=10_000,
    )
    joint_source = generate_macro_reuse_transfer_split(
        examples_per_domain,
        validation_per_domain=validation_per_domain,
        heldout_per_domain=heldout_per_domain,
        seed=seed + 2,
        grounding_offset=20_000,
    )
    controller_training = tuple(
        _with_hierarchical_completion_goal(
            problem,
            partition="controller-training",
            local_index=index,
            source_goal_ready=True,
        )
        for index, problem in enumerate(controller_source.training)
    )
    controller_heldout = tuple(
        _with_hierarchical_completion_goal(
            problem,
            partition="controller-heldout",
            local_index=index,
            source_goal_ready=True,
        )
        for index, problem in enumerate(controller_source.heldout)
    )
    joint_heldout = tuple(
        _with_hierarchical_completion_goal(
            problem,
            partition="joint-heldout",
            local_index=index,
            source_goal_ready=False,
        )
        for index, problem in enumerate(joint_source.heldout)
    )
    return HierarchicalBrainCurriculumSplit(
        controller_training=controller_training,
        controller_heldout=controller_heldout,
        controller_negative_controls=generate_symbolic_negative_controls(
            controller_heldout
        ),
        macro_training=macro.training,
        macro_validation=macro.validation,
        macro_heldout=macro.heldout,
        macro_negative_controls=macro.negative_controls,
        joint_heldout=joint_heldout,
        joint_negative_controls=generate_symbolic_negative_controls(
            joint_heldout
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


def _macro_vision_shape(
    index: int,
    rng: random.Random,
) -> SyntheticProblem:
    white = (255, 255, 255)
    palette = (
        ("red", (255, 0, 0)),
        ("blue", (0, 0, 255)),
        ("green", (0, 255, 0)),
    )
    color_name, color = palette[index % len(palette)]
    other_name, other = palette[(index + 1) % len(palette)]
    size = 2 + rng.randrange(2)
    partition_offset = (index // 100_000) % 3
    grounding_layout = (index % 100_000) // 10_000
    target_x = 1 + partition_offset + grounding_layout
    distractor_x = target_x + size + 2
    width = distractor_x + 2
    height = size + 2
    rows = [[white for _x in range(width)] for _y in range(height)]
    for y in range(1, size + 1):
        for x in range(target_x, target_x + size):
            rows[y][x] = color
    rows[1][distractor_x] = other
    if height > 2:
        rows[2][distractor_x] = other
    image = RasterImage.from_rows(rows)
    instance = RasterVisionAdapter().adapt(
        RasterVisionProblem(
            image,
            (VisionPropertyGoal("SQUARE", color_name),),
        )
    )
    problem = _transfer_problem(
        f"macro-vision-shape-{index}",
        "vision",
        "macro_pixel_shape",
        instance,
        difficulty=2,
        distractor_index=180_000 + index,
    )
    return replace(
        problem,
        instance=replace(
            problem.instance,
            metadata={
                **problem.instance.metadata,
                "target_color": color_name,
                "distractor_color": other_name,
                "target_square_size": size,
            },
        ),
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


def _retag_macro_problem(
    problem: SyntheticProblem,
    partition: str,
    local_index: int,
) -> SyntheticProblem:
    structure_key = (
        f"{problem.domain}:macro_reuse:{problem.capability}:{partition}"
    )
    instance = replace(
        problem.instance,
        metadata={
            **problem.instance.metadata,
            "macro_reuse_partition": partition,
            "macro_reuse_source_id": problem.problem_id,
            "structure_key": structure_key,
        },
    )
    return SyntheticProblem(
        problem_id=f"macro-{partition}-{problem.domain}-{local_index}",
        domain=problem.domain,
        instance=instance,
        capability=problem.capability,
        structure_key=structure_key,
        difficulty=problem.difficulty,
    )


def _with_hierarchical_completion_goal(
    problem: SyntheticProblem,
    *,
    partition: str,
    local_index: int,
    source_goal_ready: bool,
) -> SyntheticProblem:
    instance = problem.instance
    registry = instance.registry
    source_goal = instance.goals[0].atom
    predicate = registry.register_predicate(
        "CERTIFIED_RESULT",
        tuple(argument.type for argument in source_goal.arguments),
    )
    certified = registry.atom(predicate, *source_goal.arguments)
    operator_name = (
        f"verify_certified_{partition.replace('-', '_')}_"
        f"{problem.domain}_{local_index}"
    )
    registry.register_operator(
        Rule(
            name=operator_name,
            parameters=(),
            preconditions=(source_goal,),
            effects=(certified,),
            description_ko="검증된 중간 결론을 최종 결과로 승인한다.",
        ),
        family="verify",
        tags=("completion", "shared_operator_family"),
    )
    facts = instance.state.facts
    if source_goal_ready and not instance.state.contains(source_goal):
        facts = facts + (
            Fact(
                source_goal,
                FactStatus.OBSERVED,
                "hierarchical_controller_curriculum",
            ),
        )
    state = WorldState(
        facts=facts,
        depth=instance.state.depth,
        path_cost=instance.state.path_cost,
    )
    structure_key = f"{problem.domain}:hierarchical_completion:{partition}"
    extended = replace(
        instance,
        state=state,
        goals=(Goal(certified, label="certified_result"),),
        metadata={
            **instance.metadata,
            "hierarchical_partition": partition,
            "completion_operator": operator_name,
            "completion_family": "verify",
            "source_goal_ready": source_goal_ready,
            "structure_key": structure_key,
        },
    )
    return SyntheticProblem(
        problem_id=f"hierarchical-{partition}-{problem.domain}-{local_index}",
        domain=problem.domain,
        instance=extended,
        capability="hierarchical_completion",
        structure_key=structure_key,
        difficulty=problem.difficulty + 1,
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


def _semantic_flow_problem(
    index: int,
    rng: random.Random,
    variant: str,
) -> SyntheticProblem:
    suffix = f"{index}_{rng.randrange(1_000_000)}"
    if variant == "single_threshold":
        counts = {"red": 1 + index % 2, "blue": 1}
        property_name = f"occupied_{suffix}"
        text = (
            f"If the count of red objects is at least {counts['red']}, "
            f"the scene is {property_name}. "
            f"Prove: the scene is {property_name}."
        )
        proof_depth = 3
    elif variant == "dual_selector_conjunction":
        counts = {"red": 1 + index % 2, "blue": 1 + (index + 1) % 2}
        property_name = f"balanced_{suffix}"
        text = (
            f"If the count of red objects is equal to {counts['red']} and "
            f"the count of blue objects is at least {counts['blue']}, "
            f"the scene is {property_name}. "
            f"Prove: the scene is {property_name}."
        )
        proof_depth = 5
    elif variant == "reused_measurement_bounds":
        counts = {"red": 3 + index % 2, "blue": 2}
        property_name = f"bounded_{suffix}"
        if index % 2:
            text = (
                f"빨간 물체의 개수가 {counts['red']}보다 크거나 같고 "
                f"빨간 물체의 개수가 {counts['red']}보다 작거나 같으면 "
                f"장면은 {property_name}이다. "
                f"증명: 장면은 {property_name}이다."
            )
        else:
            text = (
                f"If the number of red objects is at least {counts['red']} and "
                f"the number of red objects is at most {counts['red']}, "
                f"the scene is {property_name}. "
                f"Prove: the scene is {property_name}."
            )
        proof_depth = 4
    elif variant == "triple_mixed_comparison":
        counts = {
            "red": 2 + index % 2,
            "blue": 2 + (index + 1) % 2,
            "green": 2 + (index // 2) % 2,
        }
        total = sum(counts.values())
        property_name = f"structured_{suffix}"
        if index % 2:
            text = (
                f"If the number of all objects is greater than {total - 1} and "
                f"the number of green objects is not equal to "
                f"{counts['green'] + 1} and the number of blue objects is at "
                f"most {counts['blue']}, the scene is {property_name}. "
                f"Prove: the scene is {property_name}."
            )
        else:
            text = (
                f"모든 물체의 개수가 {total - 1}보다 크고 초록 물체의 개수가 "
                f"{counts['green'] + 1}보다 다르고 파란 물체의 개수가 "
                f"{counts['blue']}보다 작거나 같으면 장면은 {property_name}이다. "
                f"증명: 장면은 {property_name}이다."
            )
        proof_depth = 7
    else:  # pragma: no cover - private caller validates variants
        raise ValueError(f"unknown semantic flow variant: {variant}")

    instance = SceneThresholdAdapter().adapt(
        SceneThresholdProblem(_semantic_flow_image(counts), text)
    )
    structure_key = f"composed:semantic_flow:{variant}"
    tagged = replace(
        instance,
        metadata={
            **instance.metadata,
            "capability": "semantic_flow",
            "structure_key": structure_key,
            "semantic_flow": ("vision", "math", "language"),
            "semantic_flow_variant": variant,
            "object_counts": tuple(sorted(counts.items())),
            "synthetic_transfer": True,
        },
    )
    distractor_index = (
        100_000 + _SEMANTIC_FLOW_VARIANTS.index(variant) * 10_000 + index
    )
    return SyntheticProblem(
        problem_id=f"semantic-flow-{variant}-{index}",
        domain="composed",
        instance=_with_hard_negative_distractors(tagged, distractor_index),
        capability="semantic_flow",
        structure_key=structure_key,
        difficulty=proof_depth,
    )


def _semantic_flow_image(counts: dict[str, int]) -> RasterImage:
    palette = {
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "green": (0, 255, 0),
    }
    white = (255, 255, 255)
    colors = tuple(
        palette[name]
        for name in ("red", "blue", "green")
        for _slot in range(counts.get(name, 0))
    )
    middle = [white]
    for color in colors:
        middle.extend((color, color, white))
    blank = [white] * len(middle)
    return RasterImage.from_rows((blank, middle, middle, blank))


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

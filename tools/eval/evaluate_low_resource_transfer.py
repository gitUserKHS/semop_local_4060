from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    Fact,
    FactStatus,
    Goal,
    GoalDirectedPolicy,
    LanguageTextAdapter,
    LanguageTextProblem,
    LanguageLogicAdapter,
    NumericComparisonAdapter,
    OperatorKernel,
    RasterImage,
    RasterVisionAdapter,
    RasterVisionProblem,
    Rule,
    SceneThresholdAdapter,
    SceneThresholdProblem,
    SolveBudget,
    StructuredMeaningGraphAdapter,
    VisionProblem,
    VisionAreaGoal,
    VisionCountGoal,
    VisionPropertyGoal,
    VisionRelationGoal,
    VisionWorldAdapter,
    WorldState,
)
from semop.kernel.domains import (
    DomainInstance,
    make_hidden_premise_instance,
    parse_arithmetic_expression,
    parse_geometry_dsl,
    parse_grid_problem,
    parse_linear_equation,
    parse_numeric_comparison,
)
from semop.structures import StructuredMeaningGraph


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    domain: str
    split_axis: str
    size: str
    expected_solved: bool
    build: Callable[[], DomainInstance]


@dataclass(frozen=True)
class CaseMeasurement:
    case_id: str
    domain: str
    split_axis: str
    size: str
    expected_solved: bool
    success: bool
    verified: bool
    false_positive: bool
    expansions: int
    proof_steps: int
    inference_rounds: int
    cpu_seconds: float
    halt_reason: str
    fallback_used: bool


@dataclass
class _SceneEntity:
    id: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class _SceneRelation:
    source: str
    relation: str
    target: str
    confidence: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class _SceneWorld:
    query: str
    entities: list[_SceneEntity]
    relations: list[_SceneRelation]
    metadata: dict[str, Any] = field(default_factory=dict)
    inferred_steps: list[str] = field(default_factory=list)
    audit_trace: list[str] = field(default_factory=list)


def benchmark_cases() -> tuple[EvalCase, ...]:
    return (
        EvalCase(
            "geometry_midpoint_names",
            "geometry",
            "renamed_points",
            "small",
            True,
            lambda: parse_geometry_dsl(
                "point P, Q, X\n"
                "assume midpoint(X, P, Q)\n"
                "prove equal_length(segment(P, X), segment(X, Q))\n"
            ),
        ),
        EvalCase(
            "geometry_length_composition",
            "geometry",
            "operator_composition",
            "small",
            True,
            lambda: parse_geometry_dsl(
                "segment S1, S2, S3\n"
                "assume equal_length(S1, S2)\n"
                "assume equal_length(S2, S3)\n"
                "prove equal_length(S1, S3)\n"
            ),
        ),
        EvalCase(
            "geometry_parallel_perpendicular",
            "geometry",
            "new_relation_chain",
            "small",
            True,
            lambda: parse_geometry_dsl(
                "line L1, L2, L3\n"
                "assume parallel(L1, L2)\n"
                "assume perpendicular(L1, L3)\n"
                "prove perpendicular(L2, L3)\n"
            ),
        ),
        EvalCase(
            "geometry_unsupported",
            "geometry",
            "negative_control",
            "small",
            False,
            lambda: parse_geometry_dsl(
                "line L1, L2, L3\n"
                "assume parallel(L1, L2)\n"
                "prove perpendicular(L2, L3)\n"
            ),
        ),
        EvalCase(
            "premise_one_requirement",
            "hidden_premise",
            "new_wording",
            "small",
            True,
            lambda: make_hidden_premise_instance(
                "publish", ["tests_pass"], satisfied=["tests_pass"]
            ),
        ),
        EvalCase(
            "premise_composed_requirements",
            "hidden_premise",
            "relation_composition",
            "small",
            True,
            lambda: make_hidden_premise_instance(
                "deploy",
                ["tests_pass", "approval"],
                satisfied=["approval", "tests_pass"],
            ),
        ),
        EvalCase(
            "premise_blocked",
            "hidden_premise",
            "blocked_by_transfer",
            "small",
            True,
            lambda: make_hidden_premise_instance(
                "release",
                ["security_review"],
                blocked=["security_review"],
                target="not_ready",
            ),
        ),
        EvalCase(
            "premise_missing_unknown",
            "hidden_premise",
            "negative_control",
            "small",
            False,
            lambda: make_hidden_premise_instance(
                "ship", ["approval"], satisfied=[]
            ),
        ),
        EvalCase(
            "grid_short_path",
            "grid",
            "path_shape",
            "small",
            True,
            lambda: parse_grid_problem("S.G"),
        ),
        EvalCase(
            "grid_turn_path",
            "grid",
            "new_obstacle_layout",
            "small",
            True,
            lambda: parse_grid_problem("S..\n##.\n..G"),
        ),
        EvalCase(
            "grid_larger_corridor",
            "grid",
            "larger_problem_size",
            "large",
            True,
            lambda: parse_grid_problem(
                "S....\n####.\n.....\n.####\n....G"
            ),
        ),
        EvalCase(
            "grid_no_path",
            "grid",
            "negative_control",
            "small",
            False,
            lambda: parse_grid_problem("S#G"),
        ),
    )


def language_math_vision_cases() -> tuple[EvalCase, ...]:
    return (
        EvalCase(
            "language_single_requirement",
            "language",
            "new_wording",
            "small",
            True,
            lambda: _language_instance("publish", ["tests"], ["tests"]),
        ),
        EvalCase(
            "language_composed_requirements",
            "language",
            "relation_composition",
            "small",
            True,
            lambda: _language_instance(
                "deploy", ["tests", "approval"], ["approval", "tests"]
            ),
        ),
        EvalCase(
            "language_blocked",
            "language",
            "blocked_by_transfer",
            "small",
            True,
            lambda: _language_instance(
                "release", ["security_review"], [], ["security_review"]
            ),
        ),
        EvalCase(
            "language_missing_unknown",
            "language",
            "negative_control",
            "small",
            False,
            lambda: _language_instance("ship", ["approval"], []),
        ),
        EvalCase(
            "language_typed_argument_distractors",
            "language",
            "typed_argument_selection",
            "small",
            True,
            _language_distractor_instance,
        ),
        EvalCase(
            "language_goal_effect_distractors",
            "language",
            "cross_domain_goal_binding",
            "small",
            True,
            _language_goal_effect_distractor_instance,
        ),
        EvalCase(
            "language_korean_text_ready",
            "language",
            "natural_wording_transfer",
            "small",
            True,
            lambda: _language_text_instance(
                "배포하려면 테스트 통과와 승인이 필요하다. "
                "테스트 통과가 충족되었다. 승인이 충족되었다."
            ),
        ),
        EvalCase(
            "language_english_text_blocked",
            "language",
            "natural_negation_transfer",
            "small",
            True,
            lambda: _language_text_instance(
                "Goal: release; Requires: security review; "
                "Security review is not satisfied."
            ),
        ),
        EvalCase(
            "language_english_text_missing",
            "language",
            "natural_missing_premise_control",
            "small",
            False,
            lambda: _language_text_instance(
                "Can we deploy? Deploy requires tests."
            ),
        ),
        EvalCase(
            "language_logic_two_hop",
            "language",
            "horn_composition",
            "small",
            True,
            lambda: LanguageLogicAdapter().adapt(
                "Every programmer is a person. Every person is mortal. "
                "Ada is a programmer. Prove: Ada is mortal."
            ),
        ),
        EvalCase(
            "language_logic_binding_distractors",
            "language",
            "cross_domain_goal_binding",
            "small",
            True,
            _language_logic_distractor_instance,
        ),
        EvalCase(
            "language_logic_contradiction",
            "language",
            "contradiction_control",
            "small",
            False,
            lambda: LanguageLogicAdapter().adapt(
                "Ada is a programmer. Ada is not a programmer. "
                "Prove: Ada is a programmer."
            ),
        ),
        EvalCase(
            "language_logic_conjunction",
            "language",
            "conjunctive_horn_composition",
            "small",
            True,
            lambda: LanguageLogicAdapter().adapt(
                "Rule: red & square -> marker. Tile is red. Tile is square. "
                "Prove: Tile is a marker."
            ),
        ),
        EvalCase(
            "language_logic_missing_conjunct",
            "language",
            "conjunction_negative_control",
            "small",
            False,
            lambda: LanguageLogicAdapter().adapt(
                "Rule: red & square -> marker. Tile is red. "
                "Prove: Tile is a marker."
            ),
        ),
        EvalCase(
            "math_exact_fraction",
            "math",
            "numeric_representation",
            "small",
            True,
            lambda: parse_arithmetic_expression("0.1 + 1/5"),
        ),
        EvalCase(
            "math_unseen_composition",
            "math",
            "operator_composition",
            "small",
            True,
            lambda: parse_arithmetic_expression("-(2 + 3) * (7 - 4)"),
        ),
        EvalCase(
            "math_deeper_program",
            "math",
            "larger_program_depth",
            "large",
            True,
            lambda: parse_arithmetic_expression(
                "((1 + 2) * (3 + 4) - (5 / 2)) / (6 - 5)"
            ),
        ),
        EvalCase(
            "math_goal_effect_distractors",
            "math",
            "cross_domain_goal_binding",
            "small",
            True,
            _math_goal_effect_distractor_instance,
        ),
        EvalCase(
            "math_wrong_target",
            "math",
            "negative_control",
            "small",
            False,
            lambda: _wrong_math_target("2 + 3", "6"),
        ),
        EvalCase(
            "math_linear_equation",
            "math",
            "algebraic_transformation",
            "small",
            True,
            lambda: parse_linear_equation("2*x + 3 = 11"),
        ),
        EvalCase(
            "math_fractional_equation",
            "math",
            "exact_fraction_transfer",
            "small",
            True,
            lambda: parse_linear_equation("x/2 + 1 = 5/2"),
        ),
        EvalCase(
            "math_wrong_equation_target",
            "math",
            "negative_control",
            "small",
            False,
            _wrong_linear_equation_target,
        ),
        EvalCase(
            "math_numeric_comparison",
            "math",
            "exact_comparison_composition",
            "small",
            True,
            _numeric_comparison_distractor_instance,
        ),
        EvalCase(
            "math_false_comparison",
            "math",
            "comparison_negative_control",
            "small",
            False,
            lambda: NumericComparisonAdapter().adapt("2 * 3 > 7"),
        ),
        EvalCase(
            "vision_verified_chain",
            "vision",
            "spatial_composition",
            "small",
            True,
            lambda: _vision_instance(verified=True),
        ),
        EvalCase(
            "vision_raster_transitive_pixels",
            "vision",
            "raw_pixel_composition",
            "small",
            True,
            _raster_vision_instance,
        ),
        EvalCase(
            "vision_inverse_relation",
            "vision",
            "canonical_inverse",
            "small",
            True,
            _inverse_vision_instance,
        ),
        EvalCase(
            "vision_confidence_without_verification",
            "vision",
            "negative_control",
            "small",
            False,
            lambda: _vision_instance(verified=False),
        ),
        EvalCase(
            "vision_raster_centroid_only",
            "vision",
            "pixel_uncertainty_control",
            "small",
            False,
            _ambiguous_raster_vision_instance,
        ),
        EvalCase(
            "vision_unknown_relation",
            "vision",
            "unverified_predicate",
            "small",
            False,
            _unknown_vision_instance,
        ),
        EvalCase(
            "vision_spatial_distractor_chains",
            "vision",
            "typed_argument_selection",
            "small",
            True,
            _vision_distractor_instance,
        ),
        EvalCase(
            "vision_raster_square",
            "vision",
            "pixel_shape_composition",
            "small",
            True,
            lambda: _measured_raster_instance(VisionPropertyGoal("SQUARE", "red")),
        ),
        EvalCase(
            "vision_raster_count",
            "vision",
            "closed_component_quantification",
            "small",
            True,
            lambda: _measured_raster_instance(VisionCountGoal("all", 2)),
        ),
        EvalCase(
            "vision_raster_area",
            "vision",
            "pixel_measurement_comparison",
            "small",
            True,
            lambda: _measured_raster_instance(VisionAreaGoal("blue", "red")),
        ),
        EvalCase(
            "vision_raster_wrong_count",
            "vision",
            "quantification_control",
            "small",
            False,
            lambda: _measured_raster_instance(VisionCountGoal("all", 3)),
        ),
        EvalCase(
            "vision_goal_independent_count",
            "vision",
            "goal_independent_observation",
            "small",
            True,
            _goal_independent_raster_count_instance,
        ),
    )


def composed_v3_cases() -> tuple[EvalCase, ...]:
    return (
        EvalCase(
            "composed_scene_english",
            "composed",
            "vision_math_language_composition",
            "small",
            True,
            lambda: _scene_threshold_instance(
                "If the number of all objects is greater than 2, "
                "the scene is crowded. Prove: the scene is crowded."
            ),
        ),
        EvalCase(
            "composed_scene_korean_color",
            "composed",
            "cross_language_color_count",
            "small",
            True,
            lambda: _scene_threshold_instance(
                "파란 물체의 개수가 1보다 크거나 같으면 장면은 균형이다. "
                "증명: 장면은 균형이다."
            ),
        ),
        EvalCase(
            "composed_scene_false_threshold",
            "composed",
            "numeric_negative_control",
            "small",
            False,
            lambda: _scene_threshold_instance(
                "If the number of all objects is greater than 3, "
                "the scene is crowded. Prove: the scene is crowded."
            ),
        ),
        EvalCase(
            "composed_scene_explicit_negation",
            "composed",
            "language_contradiction_control",
            "small",
            False,
            lambda: _scene_threshold_instance(
                "Scene is not crowded. If the number of all objects is greater "
                "than 2, the scene is crowded. Prove: the scene is crowded."
            ),
        ),
    )


def composed_v4_cases() -> tuple[EvalCase, ...]:
    return composed_v3_cases() + (
        EvalCase(
            "composed_scene_two_color_conjunction",
            "composed",
            "multi_measurement_conjunction",
            "small",
            True,
            lambda: _scene_threshold_instance(
                "If the count of red objects is at least 1 and the count of blue "
                "objects is at least 1, the scene is colorful. "
                "Prove: the scene is colorful."
            ),
        ),
        EvalCase(
            "composed_scene_korean_conjunction",
            "composed",
            "cross_language_multi_measurement",
            "small",
            True,
            lambda: _scene_threshold_instance(
                "빨간 물체의 개수가 1보다 크거나 같고 초록 물체의 개수가 "
                "1보다 크거나 같으면 장면은 다채롭다. "
                "증명: 장면은 다채롭다."
            ),
        ),
        EvalCase(
            "composed_scene_reused_measurement_bounds",
            "composed",
            "measurement_reuse",
            "small",
            True,
            lambda: _scene_threshold_instance(
                "If the count of red objects is at least 1 and the count of red "
                "objects is at most 1, the scene is bounded. "
                "Prove: the scene is bounded."
            ),
        ),
        EvalCase(
            "composed_scene_false_conjunct",
            "composed",
            "conjunction_negative_control",
            "small",
            False,
            lambda: _scene_threshold_instance(
                "If the count of red objects is at least 1 and the count of blue "
                "objects is at least 2, the scene is colorful. "
                "Prove: the scene is colorful."
            ),
        ),
    )


def run_suite(
    policy=None,
    budget: SolveBudget | None = None,
    cases: tuple[EvalCase, ...] | None = None,
) -> list[CaseMeasurement]:
    measurements: list[CaseMeasurement] = []
    for case in cases or benchmark_cases():
        instance = case.build()
        started = perf_counter()
        result = OperatorKernel(instance.registry).solve(
            instance.state,
            instance.goals,
            policy=policy,
            budget=budget or SolveBudget(),
        )
        elapsed = perf_counter() - started
        measurements.append(
            CaseMeasurement(
                case_id=case.case_id,
                domain=case.domain,
                split_axis=case.split_axis,
                size=case.size,
                expected_solved=case.expected_solved,
                success=result.success,
                verified=result.verified,
                false_positive=result.success and not case.expected_solved,
                expansions=result.expansions,
                proof_steps=len(result.proof),
                inference_rounds=result.inference_rounds,
                cpu_seconds=elapsed,
                halt_reason=result.halt_reason,
                fallback_used=result.fallback_used,
            )
        )
    return measurements


def _language_instance(
    goal: str,
    required: list[str],
    satisfied: list[str],
    blocked: list[str] | None = None,
) -> DomainInstance:
    graph = StructuredMeaningGraph(
        query=f"Can the system {goal}?",
        intent="decision",
        hidden_goals=[goal],
        required_premises=list(required),
        satisfied_premises=list(satisfied),
        missing_premises=list(blocked or ()),
    )
    return StructuredMeaningGraphAdapter().adapt(graph)


def _language_text_instance(text: str) -> DomainInstance:
    return LanguageTextAdapter().adapt(
        LanguageTextProblem(text, use_legacy_heuristics=False)
    )


def _language_logic_distractor_instance() -> DomainInstance:
    statements = [
        f"Every aa_type_{index:02d} is aa_parent_{index:02d}. "
        f"aa_item_{index:02d} is an aa_type_{index:02d}."
        for index in range(12)
    ]
    statements.extend(
        (
            "Every zz_programmer is a zz_person.",
            "Every zz_person is zz_mortal.",
            "zz_ada is a zz_programmer.",
            "Prove: zz_ada is zz_mortal.",
        )
    )
    return LanguageLogicAdapter().adapt(" ".join(statements))


def _wrong_math_target(expression: str, wrong_answer: str) -> DomainInstance:
    instance = parse_arithmetic_expression(expression)
    root = instance.goals[0].atom.arguments[0]
    wrong = instance.registry.symbol(wrong_answer, "Number")
    metadata = dict(instance.metadata)
    metadata["expected_negative_control"] = wrong_answer
    return replace(
        instance,
        goals=(
            Goal(
                instance.registry.atom("VALUE", root, wrong),
                label=f"{expression} = {wrong_answer}",
            ),
        ),
        metadata=metadata,
    )


def _wrong_linear_equation_target() -> DomainInstance:
    instance = parse_linear_equation("2*x + 3 = 11")
    variable = instance.goals[0].atom.arguments[0]
    wrong = instance.registry.symbol("5", "Number")
    return replace(
        instance,
        goals=(Goal(instance.registry.atom("SOLUTION", variable, wrong)),),
        metadata={**instance.metadata, "expected_negative_control": "5"},
    )


def _language_distractor_instance() -> DomainInstance:
    instance = _language_instance("zz_deploy", ["zz_tests"], ["zz_tests"])
    registry = instance.registry
    facts: list[Fact] = []
    for index in range(12):
        goal = registry.symbol(f"aa_noise_goal_{index:02d}", "Entity")
        premise = registry.symbol(f"aa_noise_premise_{index:02d}", "Entity")
        facts.extend(
            (
                Fact(registry.atom("GOAL", goal), FactStatus.OBSERVED, "benchmark"),
                Fact(
                    registry.atom("REQUIRES", goal, premise),
                    FactStatus.OBSERVED,
                    "benchmark",
                ),
                Fact(
                    registry.atom("SATISFIED", premise),
                    FactStatus.OBSERVED,
                    "benchmark",
                ),
            )
        )
    return replace(
        instance,
        state=WorldState(instance.state.facts + tuple(facts)),
        metadata={**instance.metadata, "distractor_bindings": 12},
    )


def _language_goal_effect_distractor_instance() -> DomainInstance:
    return _with_goal_effect_distractors(
        _language_instance("zz_release", ["zz_review"], ["zz_review"]),
        prefix="language",
    )


def _math_goal_effect_distractor_instance() -> DomainInstance:
    return _with_goal_effect_distractors(
        parse_arithmetic_expression("2 + 3"),
        prefix="math",
    )


def _numeric_comparison_distractor_instance() -> DomainInstance:
    return _with_goal_effect_distractors(
        parse_numeric_comparison("0.1 + 0.2 == 0.3"),
        prefix="numeric_comparison",
    )


def _with_goal_effect_distractors(
    instance: DomainInstance,
    *,
    prefix: str,
    count: int = 12,
) -> DomainInstance:
    registry = instance.registry
    kernel = OperatorKernel(registry)
    reference = kernel.solve(instance.state, instance.goals)
    if not reference.success or not reference.verified:
        raise RuntimeError("benchmark fixture requires a verified reference proof")
    target = instance.goals[0].atom
    producer = next(
        (
            step.action
            for step in reversed(reference.proof)
            if target in step.effects
        ),
        None,
    )
    if producer is None:
        raise RuntimeError("benchmark fixture has no goal-producing action")
    for index in range(count):
        wrong_arguments = tuple(
            registry.symbol(
                f"aa_{prefix}_wrong_{index:02d}_{argument_index}",
                argument.type,
            )
            for argument_index, argument in enumerate(target.arguments)
        )
        registry.register_operator(
            Rule(
                name=f"aa_{prefix}_goal_distractor_{index:02d}",
                parameters=(),
                preconditions=producer.preconditions,
                effects=(registry.atom(target.predicate, *wrong_arguments),),
                description_ko="같은 타입이지만 목표 결속이 다른 benchmark action",
            ),
            family="benchmark",
            tags=("typed_argument_distractor",),
        )
    return replace(
        instance,
        metadata={
            **instance.metadata,
            "goal_effect_distractors": count,
        },
    )


def _vision_instance(*, verified: bool) -> DomainInstance:
    attributes = {"geometry_verified": True} if verified else {}
    world = _scene_world(
        [
            _SceneRelation("A", "LEFT_OF", "B", 0.99, attributes),
            _SceneRelation("B", "LEFT_OF", "C", 0.99, attributes),
        ]
    )
    return VisionWorldAdapter().adapt(
        VisionProblem(world, (VisionRelationGoal("LEFT_OF", "A", "C"),))
    )


def _inverse_vision_instance() -> DomainInstance:
    world = _scene_world(
        [
            _SceneRelation(
                "B",
                "RIGHT_OF",
                "A",
                0.9,
                {"geometry_verified": True},
            )
        ]
    )
    return VisionWorldAdapter().adapt(
        VisionProblem(world, (VisionRelationGoal("LEFT_OF", "A", "B"),))
    )


def _raster_vision_instance() -> DomainInstance:
    return RasterVisionAdapter().adapt(
        RasterVisionProblem(
            _three_object_raster_image(),
            (VisionRelationGoal("LEFT_OF", "red", "blue"),),
        )
    )


def _measured_raster_instance(
    goal: VisionPropertyGoal | VisionCountGoal | VisionAreaGoal,
) -> DomainInstance:
    return RasterVisionAdapter().adapt(
        RasterVisionProblem(_measured_raster_image(), (goal,))
    )


def _goal_independent_raster_count_instance() -> DomainInstance:
    observed = RasterVisionAdapter().adapt(
        RasterVisionProblem(_measured_raster_image())
    )
    registry = observed.registry
    goal = Goal(
        registry.atom(
            "OBJECT_COUNT",
            registry.symbol("all", "Color"),
            registry.symbol("2", "Number"),
        )
    )
    return replace(observed, goals=(goal,))


def _scene_threshold_instance(text: str) -> DomainInstance:
    return SceneThresholdAdapter().adapt(
        SceneThresholdProblem(_three_object_raster_image(), text)
    )


def _measured_raster_image() -> RasterImage:
    white = (255, 255, 255)
    red = (255, 0, 0)
    blue = (0, 0, 255)
    return RasterImage.from_rows(
        (
            [white] * 10,
            [white, red, red, white, blue, blue, blue, white, white, white],
            [white, red, red, white, blue, blue, blue, white, white, white],
            [white] * 10,
        )
    )


def _three_object_raster_image() -> RasterImage:
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    return RasterImage.from_rows(
        (
            [white] * 11,
            [white, red, red, white, green, green, white, blue, blue, white, white],
            [white, red, red, white, green, green, white, blue, blue, white, white],
            [white] * 11,
        )
    )


def _ambiguous_raster_vision_instance() -> DomainInstance:
    white = (255, 255, 255)
    red = (255, 0, 0)
    blue = (0, 0, 255)
    image = RasterImage.from_rows(
        (
            [white] * 6,
            [white, red, red, red, white, white],
            [white] * 6,
            [white, white, blue, blue, blue, white],
            [white] * 6,
        )
    )
    return RasterVisionAdapter().adapt(
        RasterVisionProblem(
            image,
            (VisionRelationGoal("LEFT_OF", "red", "blue"),),
        )
    )


def _unknown_vision_instance() -> DomainInstance:
    world = _scene_world(
        [_SceneRelation("A", "NEARISH", "B", 1.0, {"verified": True})]
    )
    return VisionWorldAdapter().adapt(
        VisionProblem(world, (VisionRelationGoal("NEARISH", "A", "B"),))
    )


def _vision_distractor_instance() -> DomainInstance:
    attributes = {"geometry_verified": True}
    relations = [
        _SceneRelation("Z0", "LEFT_OF", "Z1", 0.99, attributes),
        _SceneRelation("Z1", "LEFT_OF", "Z2", 0.99, attributes),
    ]
    for index in range(12):
        relations.extend(
            (
                _SceneRelation(
                    f"A{index:02d}0",
                    "LEFT_OF",
                    f"A{index:02d}1",
                    0.99,
                    attributes,
                ),
                _SceneRelation(
                    f"A{index:02d}1",
                    "LEFT_OF",
                    f"A{index:02d}2",
                    0.99,
                    attributes,
                ),
            )
        )
    world = _scene_world(relations)
    return VisionWorldAdapter().adapt(
        VisionProblem(world, (VisionRelationGoal("LEFT_OF", "Z0", "Z2"),))
    )


def _scene_world(relations: list[_SceneRelation]) -> _SceneWorld:
    entity_ids = {
        endpoint
        for relation in relations
        for endpoint in (relation.source, relation.target)
    }
    return _SceneWorld(
        query="Is the spatial relation true?",
        entities=[
            _SceneEntity(name, {"verified": True}) for name in sorted(entity_ids)
        ],
        relations=relations,
    )


def summarize(measurements: list[CaseMeasurement]) -> dict[str, Any]:
    expected = [item for item in measurements if item.expected_solved]
    reported_success = [item for item in measurements if item.success]
    latencies = [item.cpu_seconds for item in measurements]
    by_domain: dict[str, Any] = {}
    for domain in sorted({item.domain for item in measurements}):
        items = [item for item in measurements if item.domain == domain]
        domain_expected = [item for item in items if item.expected_solved]
        by_domain[domain] = {
            "verified_solve_rate": _ratio(
                sum(item.success and item.verified for item in domain_expected),
                len(domain_expected),
            ),
            "proof_soundness": _ratio(
                sum(item.verified and not item.false_positive for item in items if item.success),
                sum(item.success for item in items),
                empty=1.0,
            ),
            "median_expansions": statistics.median(
                [item.expansions for item in domain_expected]
            ),
            "median_inference_rounds": statistics.median(
                [item.inference_rounds for item in domain_expected]
            ),
            "p95_cpu_seconds": _percentile(
                [item.cpu_seconds for item in items], 0.95
            ),
        }
    return {
        "verified_solve_rate": _ratio(
            sum(item.success and item.verified for item in expected), len(expected)
        ),
        "proof_soundness": _ratio(
            sum(item.verified and not item.false_positive for item in reported_success),
            len(reported_success),
            empty=1.0,
        ),
        "false_positives": sum(item.false_positive for item in measurements),
        "expansions": sum(item.expansions for item in measurements),
        "median_expansions": statistics.median(
            [item.expansions for item in expected]
        ),
        "median_inference_rounds": statistics.median(
            [item.inference_rounds for item in expected]
        ),
        "p50_cpu_seconds": _percentile(latencies, 0.50),
        "p95_cpu_seconds": _percentile(latencies, 0.95),
        "small_p95_cpu_seconds": _percentile(
            [item.cpu_seconds for item in measurements if item.size == "small"], 0.95
        ),
        "large_p95_cpu_seconds": _percentile(
            [item.cpu_seconds for item in measurements if item.size == "large"], 0.95
        ),
        "by_domain": by_domain,
        "cases": [asdict(item) for item in measurements],
    }


def evaluate(
    model_path: str | None = None,
    *,
    shot_models: dict[int, str] | None = None,
    max_expansions: int = 20_000,
    run_fast_tests: bool = False,
    suite: str = "operator-v1",
) -> dict[str, Any]:
    suites = {
        "operator-v1": benchmark_cases,
        "language-math-vision": language_math_vision_cases,
        "composed-v3": composed_v3_cases,
        "composed-v4": composed_v4_cases,
    }
    try:
        cases = suites[suite]()
    except KeyError as exc:
        raise ValueError(f"unknown evaluation suite: {suite}") from exc
    budget = SolveBudget(max_expansions=max_expansions)
    baseline_measurements = run_suite(None, budget, cases)
    baseline = summarize(baseline_measurements)
    controller_kind = "goal_directed_symbolic"
    parameter_count = 0
    artifact_bytes = 0
    training_metadata: dict[str, Any] = {}
    training_metadata_path: str | None = None
    peak_before = _peak_rss_bytes()
    if model_path:
        from semop.tiny_controller import NumpyTinyController

        policy = NumpyTinyController.load(model_path)
        controller_kind = "numpy_tiny_controller"
        parameter_count = policy.parameter_count
        artifact_bytes = Path(model_path).stat().st_size
        training_metadata, metadata_path = _load_training_metadata(model_path)
        training_metadata_path = str(metadata_path) if metadata_path else None
    else:
        policy = GoalDirectedPolicy()
    guided_measurements = run_suite(policy, budget, cases)
    peak_after = _peak_rss_bytes()
    guided = summarize(guided_measurements)

    reductions: dict[str, float] = {}
    for domain in baseline["by_domain"]:
        old = baseline["by_domain"][domain]["median_expansions"]
        new = guided["by_domain"][domain]["median_expansions"]
        reductions[domain] = 0.0 if old == 0 else (old - new) / old
    baseline_by_case = {
        item.case_id: item for item in baseline_measurements
    }
    guided_by_case = {item.case_id: item for item in guided_measurements}
    case_reductions = {
        case_id: (
            0.0
            if baseline_case.expansions == 0
            else (
                baseline_case.expansions
                - guided_by_case[case_id].expansions
            )
            / baseline_case.expansions
        )
        for case_id, baseline_case in baseline_by_case.items()
        if baseline_case.expected_solved
    }

    shot_curves: dict[str, Any] = {}
    configured_shots = shot_models or {}
    for shots in (0, 5, 20, 100):
        shot_path = configured_shots.get(shots)
        if shot_path:
            shot_metadata, shot_metadata_path = _load_training_metadata(shot_path)
            declared_shots = shot_metadata.get("reviewed_examples_per_domain")
            if declared_shots != shots:
                shot_curves[str(shots)] = {
                    "available": False,
                    "reviewed_per_domain": shots,
                    "verified_solve_rate": None,
                    "artifact": shot_path,
                    "metadata": str(shot_metadata_path) if shot_metadata_path else None,
                    "reason": (
                        "artifact metadata does not verify the requested reviewed "
                        f"shot count (declared={declared_shots!r})"
                    ),
                }
            else:
                from semop.tiny_controller import NumpyTinyController

                shot_summary = summarize(
                    run_suite(NumpyTinyController.load(shot_path), budget, cases)
                )
                shot_curves[str(shots)] = {
                    "available": True,
                    "reviewed_per_domain": shots,
                    "verified_solve_rate": shot_summary["verified_solve_rate"],
                    "artifact": shot_path,
                    "metadata": str(shot_metadata_path),
                    "synthetic_traces": shot_metadata.get(
                        "verified_synthetic_traces", 0
                    ),
                }
        else:
            shot_curves[str(shots)] = {
                "available": False,
                "reviewed_per_domain": shots,
                "verified_solve_rate": None,
                "reason": "no shot-specific verified model supplied",
            }

    shot_20 = shot_curves["20"]
    shot_100 = shot_curves["100"]
    shot_gate = None
    if shot_20["available"] and shot_100["available"]:
        denominator = shot_100["verified_solve_rate"]
        shot_gate = (
            shot_20["verified_solve_rate"] >= 0.9 * denominator
            if denominator > 0
            else True
        )
    soundness_gate = guided["proof_soundness"] == 1.0
    solve_drop_gate = (
        guided["verified_solve_rate"]
        >= baseline["verified_solve_rate"] - 0.01
    )
    expansion_gate = sum(value >= 0.30 for value in reductions.values()) >= 2
    fast_tests = _run_fast_core_tests() if run_fast_tests else {
        "available": False,
        "seconds": None,
        "returncode": None,
        "reason": "run with --run-fast-tests to measure the 30-second gate",
    }
    fast_test_gate = (
        fast_tests["returncode"] == 0 and fast_tests["seconds"] <= 30.0
        if fast_tests["available"]
        else None
    )
    trained_domains = {
        str(domain) for domain in training_metadata.get("trained_domains", ())
    }
    leave_one_domain_out: dict[str, Any] = {}
    for domain, metrics in guided["by_domain"].items():
        if model_path is None:
            leave_one_domain_out[domain] = {
                "available": False,
                "evaluation_mode": "non_learned_policy_baseline",
                "reason": "leave-one-domain-out applies to trained artifacts",
                **metrics,
            }
        elif training_metadata_path is None:
            leave_one_domain_out[domain] = {
                "available": False,
                "evaluation_mode": "training_metadata_missing",
                "reason": "artifact has no adjacent verified training summary",
                **metrics,
            }
        elif "trained_domains" not in training_metadata:
            leave_one_domain_out[domain] = {
                "available": False,
                "evaluation_mode": "training_domains_not_declared",
                "reason": "training summary does not declare trained_domains",
                **metrics,
            }
        elif domain in trained_domains:
            leave_one_domain_out[domain] = {
                "available": False,
                "evaluation_mode": "domain_seen_during_training",
                "reason": f"training metadata includes {domain!r}",
                **metrics,
            }
        else:
            leave_one_domain_out[domain] = {
                "available": True,
                "evaluation_mode": "verified_domain_held_out",
                **metrics,
            }
    report = {
        "schema_version": 1,
        "suite": suite,
        "controller": {
            "kind": controller_kind,
            "parameter_count": parameter_count,
            "artifact_bytes": artifact_bytes,
            "model_cap_pass": parameter_count <= 15_000_000,
            "artifact_cap_pass": artifact_bytes <= 64 * 1024 * 1024,
        },
        "data": {
            "reviewed_examples": training_metadata.get("reviewed_examples", 0),
            "reviewed_examples_per_domain": training_metadata.get(
                "reviewed_examples_per_domain", 0
            ),
            "synthetic_traces": training_metadata.get(
                "verified_synthetic_traces", 0
            ),
            "training_metadata": training_metadata_path,
            "curriculum": training_metadata.get("curriculum"),
            "trained_domains": sorted(trained_domains),
            "synthetic_cap_per_domain": 5_000,
            "operator_depth_cap": 6,
            "hard_negatives_per_positive": 4,
            "shot_curves": shot_curves,
        },
        "unguided": baseline,
        "guided": guided,
        "ab_comparison": {
            "solve_rate_delta": guided["verified_solve_rate"]
            - baseline["verified_solve_rate"],
            "median_expansion_reduction_by_domain": reductions,
            "expansion_reduction_by_case": case_reductions,
        },
        "leave_one_domain_out": leave_one_domain_out,
        "resources": {
            "peak_rss_bytes": peak_after,
            "additional_peak_rss_bytes": max(0, peak_after - peak_before),
            "additional_peak_rss_cap_pass": max(0, peak_after - peak_before)
            <= 512 * 1024 * 1024,
            "fast_core_tests": fast_tests,
        },
        "gates": {
            "proof_soundness_100_percent": soundness_gate,
            "solve_rate_drop_within_1pp": solve_drop_gate,
            "expansion_reduction_30_percent_in_two_domains": expansion_gate,
            "twenty_shot_reaches_90_percent_of_hundred_shot": shot_gate,
            "small_problem_p95_under_2s": guided["small_p95_cpu_seconds"] <= 2.0,
            "large_problem_p95_under_10s": guided["large_p95_cpu_seconds"] <= 10.0,
            "fast_cpu_suite_under_30s": fast_test_gate,
            "overall_status": "not_evaluated"
            if shot_gate is None or fast_test_gate is None
            else "pass"
            if all(
                (
                    soundness_gate,
                    solve_drop_gate,
                    expansion_gate,
                    shot_gate,
                    parameter_count <= 15_000_000,
                    artifact_bytes <= 64 * 1024 * 1024,
                    guided["small_p95_cpu_seconds"] <= 2.0,
                    guided["large_p95_cpu_seconds"] <= 10.0,
                    fast_test_gate,
                )
            )
            else "fail",
        },
    }
    return report


def _ratio(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    return empty if denominator == 0 else numerator / denominator


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _peak_rss_bytes() -> int:
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(Counters),
                wintypes.DWORD,
            )
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            process = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            ):
                return int(counters.PeakWorkingSetSize)
        except Exception:
            return 0
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, OSError):
        return 0


def _run_fast_core_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_typed_operator_*.py",
    ]
    started = perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=35.0,
            check=False,
        )
        elapsed = perf_counter() - started
        output = (completed.stdout + "\n" + completed.stderr).strip().splitlines()
        return {
            "available": True,
            "seconds": elapsed,
            "returncode": completed.returncode,
            "output_tail": output[-12:],
        }
    except subprocess.TimeoutExpired:
        return {
            "available": True,
            "seconds": perf_counter() - started,
            "returncode": -1,
            "reason": "fast core suite exceeded 35-second measurement timeout",
        }


def _parse_shot_models(values: list[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for value in values:
        shot_text, separator, path = value.partition("=")
        if not separator or int(shot_text) not in {0, 5, 20, 100}:
            raise ValueError("--shot-model must use SHOTS=PATH for 0,5,20,100")
        result[int(shot_text)] = path
    return result


def _load_training_metadata(path: str | Path) -> tuple[dict[str, Any], Path | None]:
    model_path = Path(path)
    summary_path = model_path.with_suffix(".summary.json")
    if not summary_path.exists():
        return {}, None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, summary_path
    return payload if isinstance(payload, dict) else {}, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate SemOp low-resource typed-operator transfer"
    )
    parser.add_argument("--model", help="NumPy .npz controller artifact")
    parser.add_argument(
        "--shot-model",
        action="append",
        default=[],
        metavar="SHOTS=PATH",
        help="shot-specific artifact; may be repeated",
    )
    parser.add_argument("--max-expansions", type=int, default=20_000)
    parser.add_argument(
        "--suite",
        choices=(
            "operator-v1",
            "language-math-vision",
            "composed-v3",
            "composed-v4",
        ),
        default="operator-v1",
        help="benchmark domain set",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run-fast-tests",
        action="store_true",
        help="measure the typed-core 30-second CPU test gate",
    )
    args = parser.parse_args()
    report = evaluate(
        args.model,
        shot_models=_parse_shot_models(args.shot_model),
        max_expansions=args.max_expansions,
        run_fast_tests=args.run_fast_tests,
        suite=args.suite,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["gates"]["proof_soundness_100_percent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

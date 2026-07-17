from __future__ import annotations

from collections.abc import Sequence
import re

from .model import OperatorFamily, Rule, Symbol, TypeRef
from .registry import KernelRegistry


def normalize_predicate_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return normalized.upper() or "UNKNOWN_RELATION"


def register_transitive_relation(
    registry: KernelRegistry,
    predicate: str,
    argument_type: str | TypeRef,
    *,
    operator_name: str | None = None,
    tags: Sequence[str] = (),
    description_ko: str | None = None,
) -> None:
    """Register a reusable, typed transitivity operator."""

    name = operator_name or f"{predicate.lower()}_transitivity"
    guard_name = f"{name}__different_endpoints"
    if guard_name not in registry.guards:
        registry.register_guard(
            guard_name,
            lambda binding, _state: binding["x"] != binding["z"],
        )
    x = registry.variable("x", argument_type)
    y = registry.variable("y", argument_type)
    z = registry.variable("z", argument_type)
    registry.register_operator(
        Rule(
            name=name,
            parameters=(x, y, z),
            preconditions=(
                registry.atom(predicate, x, y),
                registry.atom(predicate, y, z),
            ),
            effects=(registry.atom(predicate, x, z),),
            guards=(guard_name,),
            description_ko=description_ko
            or "{x}와 {y}, {y}와 {z}의 관계를 이어 {x}와 {z}의 관계를 얻는다.",
        ),
        family=OperatorFamily.COMPOSE.value,
        tags=("transitive", predicate.lower(), *tags),
    )


def register_requirement_reasoning(
    registry: KernelRegistry,
    goal_type: str | TypeRef,
    premise_type: str | TypeRef,
    *,
    tags: Sequence[str] = ("language",),
) -> None:
    """Install shared hidden-premise predicates and verified operators."""

    registry.register_predicate("GOAL", (goal_type,))
    registry.register_predicate("REQUIRES", (goal_type, premise_type))
    registry.register_predicate("SATISFIED", (premise_type,))
    registry.register_predicate("BLOCKED", (premise_type,))
    registry.register_predicate("REQUIREMENT_MET", (goal_type, premise_type))
    registry.register_predicate("BLOCKED_BY", (goal_type, premise_type))
    registry.register_predicate("READY", (goal_type,))
    registry.register_predicate("NOT_READY", (goal_type,))

    goal = registry.variable("g", goal_type)
    premise = registry.variable("p", premise_type)
    registry.register_operator(
        Rule(
            name="verify_required_premise",
            parameters=(goal, premise),
            preconditions=(
                registry.atom("REQUIRES", goal, premise),
                registry.atom("SATISFIED", premise),
            ),
            effects=(registry.atom("REQUIREMENT_MET", goal, premise),),
            description_ko="목표 {g}의 필수 전제 {p}: 충족 상태를 확인한다.",
        ),
        family=OperatorFamily.VERIFY.value,
        tags=(*tags, "premise"),
    )
    registry.register_operator(
        Rule(
            name="identify_blocking_premise",
            parameters=(goal, premise),
            preconditions=(
                registry.atom("REQUIRES", goal, premise),
                registry.atom("BLOCKED", premise),
            ),
            effects=(registry.atom("BLOCKED_BY", goal, premise),),
            description_ko="전제 {p}: 목표 {g}의 차단 원인임을 확인한다.",
        ),
        family=OperatorFamily.VERIFY.value,
        tags=(*tags, "blocker"),
    )
    registry.register_operator(
        Rule(
            name="blocked_goal_is_not_ready",
            parameters=(goal, premise),
            preconditions=(registry.atom("BLOCKED_BY", goal, premise),),
            effects=(registry.atom("NOT_READY", goal),),
            description_ko="차단 전제 {p} 때문에 목표 {g}: 아직 준비되지 않았다.",
        ),
        family=OperatorFamily.CONTROL.value,
        tags=(*tags, "readiness"),
    )


def register_all_requirements_ready(
    registry: KernelRegistry,
    goal: Symbol,
    premises: Sequence[Symbol],
    *,
    operator_name: str = "all_required_premises_ready",
    tags: Sequence[str] = ("language",),
) -> None:
    """Compile a finite conjunction into one replayable ground operator."""

    preconditions = [registry.atom("GOAL", goal)]
    preconditions.extend(
        registry.atom("REQUIREMENT_MET", goal, premise) for premise in premises
    )
    registry.register_operator(
        Rule(
            name=operator_name,
            parameters=(),
            preconditions=tuple(preconditions),
            effects=(registry.atom("READY", goal),),
            description_ko="명시된 필수 전제가 모두 충족되어 목표를 실행할 수 있다.",
        ),
        family=OperatorFamily.COMPOSE.value,
        tags=(*tags, "all_requirements"),
    )

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..catalog import (
    register_all_requirements_ready,
    register_requirement_reasoning,
)
from ..model import AssertionStatus, EvidenceStatus, Fact, FactStatus, Goal, WorldState
from ..registry import KernelRegistry
from .base import DomainInstance


def create_hidden_premise_registry() -> KernelRegistry:
    registry = KernelRegistry()
    entity = registry.types.register("Entity")
    goal_type = registry.types.register("ReasoningGoal", entity)
    premise_type = registry.types.register("Premise", entity)

    register_requirement_reasoning(registry, goal_type, premise_type)
    return registry


def make_hidden_premise_instance(
    goal_name: str,
    required: Sequence[str],
    *,
    satisfied: Iterable[str] = (),
    blocked: Iterable[str] = (),
    target: str = "ready",
) -> DomainInstance:
    registry = create_hidden_premise_registry()
    goal_symbol = registry.symbol(goal_name, "ReasoningGoal")
    premise_symbols = {
        name: registry.symbol(name, "Premise")
        for name in sorted(set(required) | set(satisfied) | set(blocked))
    }
    facts: list[Fact] = [
        Fact(
            registry.atom("GOAL", goal_symbol),
            FactStatus.OBSERVED,
            "adapter",
            assertion_status=AssertionStatus.EXPLICIT,
            evidence_status=EvidenceStatus.UNVERIFIED,
        )
    ]
    for name in required:
        facts.append(
            Fact(
                registry.atom("REQUIRES", goal_symbol, premise_symbols[name]),
                FactStatus.OBSERVED,
                "adapter",
                assertion_status=AssertionStatus.EXPLICIT,
                evidence_status=EvidenceStatus.UNVERIFIED,
            )
        )
    for name in satisfied:
        facts.append(
            Fact(
                registry.atom("SATISFIED", premise_symbols[name]),
                FactStatus.OBSERVED,
                "adapter",
                assertion_status=AssertionStatus.EXPLICIT,
                evidence_status=EvidenceStatus.UNVERIFIED,
            )
        )
    for name in blocked:
        facts.append(
            Fact(
                registry.atom("BLOCKED", premise_symbols[name]),
                FactStatus.OBSERVED,
                "adapter",
                assertion_status=AssertionStatus.EXPLICIT,
                evidence_status=EvidenceStatus.UNVERIFIED,
            )
        )

    normalized_target = target.lower()
    if normalized_target == "ready":
        register_all_requirements_ready(
            registry,
            goal_symbol,
            tuple(premise_symbols[name] for name in required),
        )
        goal = Goal(registry.atom("READY", goal_symbol))
    elif normalized_target == "not_ready":
        goal = Goal(registry.atom("NOT_READY", goal_symbol))
    elif normalized_target.startswith("blocked_by:"):
        name = target.split(":", 1)[1]
        goal = Goal(
            registry.atom("BLOCKED_BY", goal_symbol, premise_symbols[name])
        )
    elif normalized_target.startswith("requirement_met:"):
        name = target.split(":", 1)[1]
        goal = Goal(
            registry.atom("REQUIREMENT_MET", goal_symbol, premise_symbols[name])
        )
    else:
        raise ValueError(f"unknown hidden-premise target: {target}")
    return DomainInstance(
        registry=registry,
        state=WorldState(tuple(facts)),
        goals=(goal,),
        domain="hidden_premise",
        metadata={
            "goal_name": goal_name,
            "required": tuple(required),
            "reviewed_examples": 0,
        },
    )

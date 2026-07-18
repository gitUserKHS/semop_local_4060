from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import tracemalloc
from typing import TYPE_CHECKING

from .catalog import (
    normalize_predicate_name,
    register_all_requirements_ready,
    register_requirement_reasoning,
)
from .domains.base import DomainInstance
from .engine import ActionPolicy, OperatorKernel, RegistryPolicyProvider
from .grounding import (
    GroundingAuthority,
    GroundingDisposition,
    GroundingTrace,
    grounding_payload_digest,
    make_grounding_record,
)
from .model import (
    AssertionStatus,
    EvidenceStatus,
    Fact,
    FactStatus,
    Goal,
    SolveBudget,
    SolveResult,
    Symbol,
    WorldState,
)
from .registry import KernelRegistry

if TYPE_CHECKING:
    from semop.structures import StructuredMeaningGraph


class MigrationMode(str, Enum):
    LEGACY = "legacy"
    SHADOW = "shadow"
    TYPED = "typed"


@dataclass(frozen=True)
class GraphAdapterResult(DomainInstance):
    symbols: dict[str, Symbol] = field(default_factory=dict, compare=False, hash=False)
    unverified_relations: tuple[str, ...] = ()


@dataclass(frozen=True)
class BridgeResult:
    mode: MigrationMode
    graph: "StructuredMeaningGraph"
    typed_result: SolveResult | None
    unverified_relations: tuple[str, ...] = ()


class StructuredMeaningGraphAdapter:
    """Loss-minimizing adapter from the legacy mutable graph to typed facts."""

    _KNOWN_BINARY = {
        "REQUIRES",
        "BLOCKED_BY",
        "REQUIREMENT_MET",
        "SUPPORTS",
        "CONTRADICTS",
        "CAUSES",
        "BEFORE",
        "AFTER",
        "CONTAINS",
        "PART_OF",
        "PARALLEL",
        "PERPENDICULAR",
        "EQUAL_LENGTH",
    }
    _KNOWN_UNARY = {"GOAL", "SATISFIED", "BLOCKED", "READY", "NOT_READY"}

    def adapt(self, graph: "StructuredMeaningGraph") -> GraphAdapterResult:
        registry = self._create_registry()
        symbols: dict[str, Symbol] = {}

        def symbol(value: str):
            key = value.strip() or "<empty>"
            if key not in symbols:
                symbols[key] = registry.symbol(key, "Entity")
            return symbols[key]

        facts: list[Fact] = []
        unknown_relations: set[str] = set()
        for node in graph.nodes:
            node_symbol = symbol(node.id)
            facts.append(
                Fact(
                    registry.atom("NODE", node_symbol),
                    FactStatus.OBSERVED,
                    source="structured_graph",
                    assertion_status=AssertionStatus.IMPORTED,
                    evidence_status=EvidenceStatus.UNVERIFIED,
                )
            )
        for edge in graph.edges:
            normalized = normalize_predicate_name(edge.relation)
            predicate_name = normalized
            if normalized in self._KNOWN_UNARY:
                predicate_name = f"EDGE_{normalized}"
            if predicate_name not in registry.predicates:
                verified = predicate_name in self._KNOWN_BINARY
                registry.register_predicate(
                    predicate_name,
                    (registry.types.resolve("Entity"),) * 2,
                    verified=verified,
                    metadata=None if verified else {"status": "unverified"},
                )
                if not verified:
                    unknown_relations.add(normalized)
            predicate = registry.predicates[predicate_name]
            status = (
                FactStatus.OBSERVED
                if predicate.verified and edge.confidence >= 0.5
                else FactStatus.PROPOSED
            )
            facts.append(
                Fact(
                    registry.atom(predicate, symbol(edge.source), symbol(edge.target)),
                    status,
                    source="structured_graph_edge",
                    confidence=max(0.0, min(1.0, edge.confidence)),
                    assertion_status=AssertionStatus.IMPORTED,
                    evidence_status=EvidenceStatus.UNVERIFIED,
                )
            )

        hidden_goals = tuple(dict.fromkeys(item for item in graph.hidden_goals if item))
        required_by_goal: dict[str, set[str]] = defaultdict(set)
        status_by_premise: dict[str, str] = {}
        default_goal = hidden_goals[0] if hidden_goals else ""
        for premise in graph.required_premises:
            if default_goal:
                required_by_goal[default_goal].add(premise)
        for validation in graph.premise_validations:
            goal_name = validation.hidden_goal or default_goal
            if goal_name and validation.premise:
                required_by_goal[goal_name].add(validation.premise)
            status_by_premise[validation.premise] = _premise_state(
                validation.status, validation.requirement_state
            )
        for premise in graph.satisfied_premises:
            status_by_premise[premise] = "satisfied"
        for premise in graph.missing_premises:
            status_by_premise.setdefault(premise, "blocked")

        goals: list[Goal] = []
        for goal_name in hidden_goals:
            goal_symbol = symbol(goal_name)
            facts.append(
                Fact(
                    registry.atom("GOAL", goal_symbol),
                    FactStatus.OBSERVED,
                    "legacy_goal",
                    assertion_status=AssertionStatus.IMPORTED,
                    evidence_status=EvidenceStatus.UNVERIFIED,
                )
            )
            required = tuple(sorted(required_by_goal.get(goal_name, ())))
            blocked = False
            for premise in required:
                premise_symbol = symbol(premise)
                facts.append(
                    Fact(
                        registry.atom("REQUIRES", goal_symbol, premise_symbol),
                        FactStatus.OBSERVED,
                        "legacy_premise",
                        assertion_status=AssertionStatus.IMPORTED,
                        evidence_status=EvidenceStatus.UNVERIFIED,
                    )
                )
                state = status_by_premise.get(premise, "unknown")
                if state == "satisfied":
                    facts.append(
                        Fact(
                            registry.atom("SATISFIED", premise_symbol),
                            FactStatus.OBSERVED,
                            "legacy_validation",
                            assertion_status=AssertionStatus.IMPORTED,
                            evidence_status=EvidenceStatus.UNVERIFIED,
                        )
                    )
                elif state == "blocked":
                    blocked = True
                    facts.append(
                        Fact(
                            registry.atom("BLOCKED", premise_symbol),
                            FactStatus.OBSERVED,
                            "legacy_validation",
                            assertion_status=AssertionStatus.IMPORTED,
                            evidence_status=EvidenceStatus.UNVERIFIED,
                        )
                    )
            if blocked:
                goals.append(Goal(registry.atom("NOT_READY", goal_symbol)))
            elif required:
                operator_name = f"all_requirements_ready_{len(goals)}"
                register_all_requirements_ready(
                    registry,
                    goal_symbol,
                    tuple(symbol(premise) for premise in required),
                    operator_name=operator_name,
                )
                goals.append(Goal(registry.atom("READY", goal_symbol)))
        input_digest = grounding_payload_digest(
            "\n".join(
                (
                    graph.query,
                    *(f"{fact.status.value}:{fact.atom}" for fact in facts),
                )
            )
        )
        grounding_trace = GroundingTrace(
            tuple(
                make_grounding_record(
                    domain="language",
                    statement=f"legacy_graph_fact:{index}:{fact.atom}",
                    atom=fact.atom,
                    producer_id="structured_meaning_graph_adapter",
                    source=fact.source,
                    disposition=(
                        GroundingDisposition.OBSERVED
                        if fact.status is FactStatus.OBSERVED
                        else GroundingDisposition.PROPOSED
                    ),
                    authority=(
                        GroundingAuthority.EXPLICIT_INPUT
                        if fact.status is FactStatus.OBSERVED
                        else GroundingAuthority.IMPORTED_PROPOSAL
                    ),
                    assertion_status=fact.assertion_status,
                    evidence_status=fact.evidence_status,
                    rationale=(
                        "legacy graph explicitly supplied a typed relation"
                        if fact.status is FactStatus.OBSERVED
                        else "legacy graph relation remained unverified"
                    ),
                    input_digest=input_digest,
                    evidence=(f"input:{input_digest}",),
                    confidence=fact.confidence,
                )
                for index, fact in enumerate(facts)
            )
        )
        return GraphAdapterResult(
            registry=registry,
            state=WorldState(grounding_trace.facts),
            goals=tuple(goals),
            domain="language",
            metadata={
                "source": "structured_meaning_graph",
                "query": graph.query,
                "reviewed_examples": 0,
                "grounding": grounding_trace.to_dict(include_records=False),
            },
            grounding_trace=grounding_trace,
            symbols=symbols,
            unverified_relations=tuple(sorted(unknown_relations)),
        )

    @staticmethod
    def _create_registry() -> KernelRegistry:
        registry = KernelRegistry()
        entity = registry.types.register("Entity")
        registry.register_predicate("NODE", (entity,))
        register_requirement_reasoning(registry, entity, entity)
        return registry

    @staticmethod
    def project(graph: "StructuredMeaningGraph", result: SolveResult) -> None:
        """Project replay-verified typed results into the legacy report shape."""

        from semop.structures import OperatorExecutionReport

        report = graph.operator_execution or OperatorExecutionReport()
        for outcome in result.goals:
            target = str(outcome.goal)
            collection = (
                report.satisfied_facts if outcome.proven else report.missing_facts
            )
            if target not in collection:
                collection.append(target)
        for step in result.proof:
            trace = (
                f"typed:{step.index}:{step.action.operator.name}:"
                + ",".join(str(effect) for effect in step.effects)
            )
            if trace not in report.support_trace:
                report.support_trace.append(trace)
            for effect in step.effects:
                decision = str(effect)
                if decision not in report.derived_decisions:
                    report.derived_decisions.append(decision)
        graph.operator_execution = report


class TypedKernelBridge:
    def __init__(self, adapter: StructuredMeaningGraphAdapter | None = None) -> None:
        self.adapter = adapter or StructuredMeaningGraphAdapter()

    def run(
        self,
        graph: "StructuredMeaningGraph",
        *,
        mode: str | MigrationMode = MigrationMode.SHADOW,
        policy: ActionPolicy | RegistryPolicyProvider | None = None,
        budget: SolveBudget | None = None,
    ) -> BridgeResult:
        execution_mode = MigrationMode(mode)
        if execution_mode is MigrationMode.LEGACY:
            return BridgeResult(execution_mode, graph, None)
        adapted = self.adapter.adapt(graph)
        if not adapted.goals:
            graph.audit_trace.append(
                "typed_shadow: skipped (no explicit typed goal/premise pair)"
            )
            return BridgeResult(
                execution_mode,
                graph,
                None,
                adapted.unverified_relations,
            )
        already_tracing = tracemalloc.is_tracing()
        if not already_tracing:
            tracemalloc.start()
        before_current, _ = tracemalloc.get_traced_memory()
        try:
            result = OperatorKernel(adapted.registry).solve(
                adapted.state,
                adapted.goals,
                policy=policy,
                budget=budget,
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if not already_tracing:
                tracemalloc.stop()
        legacy_report = graph.operator_execution
        legacy_trace = set(legacy_report.support_trace if legacy_report else ())
        typed_operators = {step.action.operator.name for step in result.proof}
        proof_overlap = sum(
            any(operator in trace for trace in legacy_trace)
            for operator in typed_operators
        )
        graph.audit_trace.append(
            "typed_shadow: "
            f"success={str(result.success).lower()} "
            f"verified={str(result.verified).lower()} "
            f"steps={len(result.proof)} rounds={result.inference_rounds} "
            f"expansions={result.expansions} "
            f"elapsed_ms={result.elapsed_seconds * 1000:.3f} "
            f"python_peak_kb={max(0, peak - before_current) / 1024:.1f} "
            f"proof_operator_overlap={proof_overlap}/{len(typed_operators)} "
            f"unverified_relations={len(adapted.unverified_relations)}"
        )
        if execution_mode is MigrationMode.TYPED:
            self.adapter.project(graph, result)
        return BridgeResult(
            execution_mode,
            graph,
            result,
            adapted.unverified_relations,
        )

def _premise_state(status: str, requirement_state: str) -> str:
    normalized_requirement = requirement_state.strip().lower()
    normalized_status = status.strip().lower()
    if normalized_requirement in {"satisfied", "met", "available"}:
        return "satisfied"
    if normalized_requirement in {"blocked", "missing", "unavailable"}:
        return "blocked"
    if normalized_status in {"satisfied", "met"}:
        return "satisfied"
    if normalized_status in {"blocked", "contradicted", "invalid", "missing"}:
        return "blocked"
    return "unknown"

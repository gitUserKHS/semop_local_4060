from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re

from .contracts import DomainInstance
from .model import (
    Atom,
    Fact,
    Goal,
    KernelError,
    OperatorSpec,
    Rule,
    Symbol,
    Term,
    TermApplication,
    Variable,
    WorldState,
)
from .registry import KernelRegistry


class RegistryCompositionError(KernelError):
    """Raised when independently valid registries cannot share one typed world."""


@dataclass(frozen=True)
class CompositionComponent:
    instance: DomainInstance
    alias: str = ""
    include_goals: bool = True
    namespace_symbol_types: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.instance, DomainInstance):
            raise TypeError("composition component requires a DomainInstance")
        object.__setattr__(
            self,
            "namespace_symbol_types",
            frozenset(self.namespace_symbol_types),
        )


@dataclass(frozen=True)
class RegistryImport:
    alias: str
    domain: str
    term_map: Mapping[Term, Term] = field(
        compare=False,
        hash=False,
        repr=False,
    )
    operator_names: tuple[str, ...] = ()
    guard_names: tuple[tuple[str, str], ...] = ()

    def map_term(self, term: Term) -> Term:
        try:
            return self.term_map[term]
        except KeyError as exc:
            raise RegistryCompositionError(
                f"term {term!s} was not part of component {self.alias!r}"
            ) from exc


@dataclass(frozen=True)
class DomainComposition:
    instance: DomainInstance
    imports: tuple[RegistryImport, ...]

    def import_for(self, alias: str) -> RegistryImport:
        for imported in self.imports:
            if imported.alias == alias:
                return imported
        raise KeyError(alias)


def compose_domain_instances(
    components: Sequence[CompositionComponent | DomainInstance],
    *,
    domain: str = "composed",
) -> DomainComposition:
    """Merge domain instances into one replayable registry and world state.

    Registries share nominal types and compatible predicate/function schemas. Guards
    are always alias-scoped because callable equality cannot establish semantic
    identity. Operator names stay stable unless a collision requires aliasing.
    """

    if not components:
        raise RegistryCompositionError("composition requires at least one component")
    normalized = tuple(
        item if isinstance(item, CompositionComponent) else CompositionComponent(item)
        for item in components
    )
    aliases = _component_aliases(normalized)
    target = KernelRegistry()

    for component in normalized:
        _merge_types(target, component.instance.registry)

    all_facts: list[Fact] = []
    all_goals: list[Goal] = []
    imports: list[RegistryImport] = []
    for component, alias in zip(normalized, aliases, strict=True):
        imported, facts, goals = _import_component(target, component, alias)
        imports.append(imported)
        all_facts.extend(facts)
        all_goals.extend(goals)

    unique_goals: dict[tuple[object, ...], Goal] = {}
    for goal in all_goals:
        key = (goal.atom.canonical_key(), goal.label)
        unique_goals[key] = goal
    state = WorldState(
        tuple(all_facts),
        depth=max(item.instance.state.depth for item in normalized),
        path_cost=sum(item.instance.state.path_cost for item in normalized),
    )
    instance = DomainInstance(
        registry=target,
        state=state,
        goals=tuple(unique_goals[key] for key in sorted(unique_goals, key=str)),
        domain=domain,
        metadata={
            "input_kind": "domain_composition",
            "component_count": len(normalized),
            "components": tuple(
                {
                    "alias": alias,
                    "domain": component.instance.domain,
                    "fact_count": len(component.instance.state.facts),
                    "goal_count": len(component.instance.goals),
                    "included_goal_count": (
                        len(component.instance.goals) if component.include_goals else 0
                    ),
                    "namespaced_symbol_types": tuple(
                        sorted(component.namespace_symbol_types)
                    ),
                }
                for component, alias in zip(normalized, aliases, strict=True)
            ),
            "reviewed_examples": 0,
        },
    )
    return DomainComposition(instance=instance, imports=tuple(imports))


def _import_component(
    target: KernelRegistry,
    component: CompositionComponent,
    alias: str,
) -> tuple[RegistryImport, tuple[Fact, ...], tuple[Goal, ...]]:
    source = component.instance.registry
    function_map = {}
    for name, function in sorted(source.functions.items()):
        existing = target.functions.get(name)
        if existing is None:
            existing = target.register_function(
                name,
                tuple(item.name for item in function.input_types),
                function.output_type.name,
                symmetry_groups=function.symmetry_groups,
                allow_repeated_arguments=function.allow_repeated_arguments,
            )
        elif not _compatible_function(existing, function):
            raise RegistryCompositionError(
                f"function {name!r} has incompatible component signatures"
            )
        function_map[name] = existing

    predicate_map = {}
    for name, predicate in sorted(source.predicates.items()):
        existing = target.predicates.get(name)
        if existing is None:
            existing = target.register_predicate(
                name,
                tuple(item.name for item in predicate.argument_types),
                symmetry_groups=predicate.symmetry_groups,
                verified=predicate.verified,
                metadata=dict(predicate.metadata),
            )
        elif not _compatible_predicate(existing, predicate):
            raise RegistryCompositionError(
                f"predicate {name!r} has incompatible component signatures"
            )
        predicate_map[name] = existing

    guard_map: dict[str, str] = {}
    for name, guard in sorted(source.guards.items()):
        imported_name = _unique_name(target.guards, f"{alias}__{name}")
        target.register_guard(imported_name, guard)
        guard_map[name] = imported_name

    term_map: dict[Term, Term] = {}

    def remap_term(term: Term) -> Term:
        existing = term_map.get(term)
        if existing is not None:
            return existing
        if isinstance(term, Symbol):
            name = (
                f"{alias}::{term.name}"
                if term.type.name in component.namespace_symbol_types
                else term.name
            )
            mapped: Term = target.symbol(name, term.type.name)
        elif isinstance(term, Variable):
            mapped = target.variable(term.name, term.type.name)
        elif isinstance(term, TermApplication):
            mapped = target.apply(
                function_map[term.function.name],
                *(remap_term(argument) for argument in term.arguments),
            )
        else:  # pragma: no cover - Term is closed by the public model contract
            raise TypeError(f"unsupported term type: {type(term).__name__}")
        term_map[term] = mapped
        return mapped

    def remap_atom(atom: Atom) -> Atom:
        return target.atom(
            predicate_map[atom.predicate.name],
            *(remap_term(argument) for argument in atom.arguments),
        )

    operator_names: list[str] = []
    for source_operator in sorted(source.operators.values(), key=lambda item: item.name):
        imported_name = source_operator.name
        if imported_name in target.operators:
            imported_name = _unique_name(
                target.operators,
                f"{alias}__{source_operator.name}",
            )
        rule = Rule(
            name=imported_name,
            parameters=tuple(
                remap_term(parameter) for parameter in source_operator.parameters
            ),
            preconditions=tuple(
                remap_atom(atom) for atom in source_operator.preconditions
            ),
            effects=tuple(remap_atom(atom) for atom in source_operator.effects),
            guards=tuple(guard_map[name] for name in source_operator.guards),
            cost=source_operator.cost,
            description_ko=source_operator.rule.description_ko,
        )
        target.register_operator(
            OperatorSpec(
                rule,
                family=source_operator.family,
                tags=source_operator.tags,
                metadata=source_operator.metadata,
            )
        )
        operator_names.append(imported_name)

    facts = tuple(
        Fact(
            remap_atom(fact.atom),
            fact.status,
            fact.source,
            fact.confidence,
        )
        for fact in component.instance.state.facts
    )
    goals = (
        tuple(
            Goal(remap_atom(goal.atom), label=goal.label)
            for goal in component.instance.goals
        )
        if component.include_goals
        else ()
    )
    return (
        RegistryImport(
            alias=alias,
            domain=component.instance.domain,
            term_map=dict(term_map),
            operator_names=tuple(operator_names),
            guard_names=tuple(sorted(guard_map.items())),
        ),
        facts,
        goals,
    )


def _merge_types(target: KernelRegistry, source: KernelRegistry) -> None:
    target_parents = dict(target.types.snapshot())
    pending = dict(source.types.snapshot())
    while pending:
        progressed = False
        for name, parent in tuple(sorted(pending.items())):
            if name in target_parents:
                if target_parents[name] != parent:
                    raise RegistryCompositionError(
                        f"type {name!r} has incompatible parents "
                        f"{target_parents[name]!r} and {parent!r}"
                    )
                del pending[name]
                progressed = True
            elif parent is None or parent in target_parents:
                target.types.register(name, parent)
                target_parents[name] = parent
                del pending[name]
                progressed = True
        if not progressed:
            raise RegistryCompositionError(
                "component type graph contains an unresolved parent or cycle: "
                + ", ".join(sorted(pending))
            )


def _compatible_function(left, right) -> bool:
    return (
        tuple(item.name for item in left.input_types)
        == tuple(item.name for item in right.input_types)
        and left.output_type.name == right.output_type.name
        and left.symmetry_groups == right.symmetry_groups
        and left.allow_repeated_arguments == right.allow_repeated_arguments
    )


def _compatible_predicate(left, right) -> bool:
    return (
        tuple(item.name for item in left.argument_types)
        == tuple(item.name for item in right.argument_types)
        and left.symmetry_groups == right.symmetry_groups
        and left.verified == right.verified
        and left.metadata == right.metadata
    )


def _component_aliases(
    components: Sequence[CompositionComponent],
) -> tuple[str, ...]:
    aliases: list[str] = []
    used: set[str] = set()
    for index, component in enumerate(components):
        raw = component.alias or f"{component.instance.domain}_{index}"
        base = re.sub(r"[^A-Za-z0-9_]+", "_", raw.strip()).strip("_")
        if not base:
            raise RegistryCompositionError("component alias cannot be empty")
        alias = base
        suffix = 2
        while alias in used:
            alias = f"{base}_{suffix}"
            suffix += 1
        used.add(alias)
        aliases.append(alias)
    return tuple(aliases)


def _unique_name(existing: Mapping[str, object], preferred: str) -> str:
    if preferred not in existing:
        return preferred
    suffix = 2
    while f"{preferred}_{suffix}" in existing:
        suffix += 1
    return f"{preferred}_{suffix}"

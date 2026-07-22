from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re

from .model import (
    Atom,
    Fact,
    FactStatus,
    KernelError,
    OperatorFamily,
    OperatorSpec,
    Predicate,
    Rule,
    Term,
    TypeValidationError,
    WorldState,
    collect_variables,
)
from .registry import KernelRegistry


_COMPARATORS = frozenset({"<", "<=", ">", ">=", "==", "!="})
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class NumericMeasurementRef:
    """A ground reference to one numeric slot of a verified predicate."""

    predicate: Predicate
    fixed_arguments: tuple[Term, ...]
    value_position: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixed_arguments", tuple(self.fixed_arguments))
        if not isinstance(self.predicate, Predicate):
            raise TypeError("numeric measurement predicate must be a Predicate")
        if isinstance(self.value_position, bool) or not isinstance(
            self.value_position, int
        ):
            raise TypeError("numeric measurement value_position must be an integer")
        if not 0 <= self.value_position < self.predicate.arity:
            raise TypeValidationError(
                f"measurement value_position {self.value_position} is outside "
                f"{self.predicate.name}/{self.predicate.arity}"
            )
        if len(self.fixed_arguments) != self.predicate.arity - 1:
            raise TypeValidationError(
                f"measurement {self.predicate.name} requires "
                f"{self.predicate.arity - 1} fixed arguments, got "
                f"{len(self.fixed_arguments)}"
            )
        if any(not isinstance(argument, Term) for argument in self.fixed_arguments):
            raise TypeError("numeric measurement fixed arguments must be Terms")
        if any(collect_variables(argument) for argument in self.fixed_arguments):
            raise TypeValidationError("numeric measurement arguments must be ground")

    def arguments_with(self, value: Term) -> tuple[Term, ...]:
        arguments = list(self.fixed_arguments)
        arguments.insert(self.value_position, value)
        return tuple(arguments)


@dataclass(frozen=True)
class NumericConditionSpec:
    """A declarative comparison over a verified numeric measurement."""

    name: str
    measurement: NumericMeasurementRef
    comparator: str
    threshold: Fraction
    rule_context: Term | None = None
    value_variable_name: str = "value"
    operator_name: str = ""
    guard_name: str = ""
    description_ko: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.measurement, NumericMeasurementRef):
            raise TypeError("numeric condition measurement must be a reference")
        _require_identifier(self.name, "condition name")
        _require_identifier(self.value_variable_name, "value variable name")
        if self.operator_name:
            _require_identifier(self.operator_name, "condition operator name")
        if self.guard_name:
            _require_identifier(self.guard_name, "condition guard name")
        if self.comparator not in _COMPARATORS:
            raise ValueError(f"unsupported numeric comparator: {self.comparator}")
        if not isinstance(self.threshold, Fraction):
            raise TypeError("numeric condition threshold must be a Fraction")
        if self.rule_context is not None:
            if not isinstance(self.rule_context, Term):
                raise TypeError("numeric condition rule_context must be a Term")
            if collect_variables(self.rule_context):
                raise TypeValidationError(
                    "numeric condition rule_context must be ground"
                )


@dataclass(frozen=True)
class NumericDataflowRuleSpec:
    """Compile verified numeric conditions into one typed conclusion program."""

    name: str
    conditions: tuple[NumericConditionSpec, ...]
    conclusion: Atom
    target: Term | None = None
    blocked_by: tuple[Atom, ...] = ()
    rule_predicate_name: str = ""
    met_predicate_name: str = ""
    conclusion_operator_name: str = ""
    conclusion_guard_name: str = ""
    fact_source: str = "typed_numeric_dataflow"
    compare_tags: tuple[str, ...] = ("math", "dataflow", "cross_domain")
    conclusion_tags: tuple[str, ...] = ("compose", "dataflow", "cross_domain")
    conclusion_description_ko: str = ""

    def __post_init__(self) -> None:
        _require_identifier(self.name, "dataflow rule name")
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(self, "blocked_by", tuple(self.blocked_by))
        object.__setattr__(self, "compare_tags", tuple(self.compare_tags))
        object.__setattr__(self, "conclusion_tags", tuple(self.conclusion_tags))
        if not self.conditions:
            raise ValueError("numeric dataflow rule requires at least one condition")
        if any(
            not isinstance(condition, NumericConditionSpec)
            for condition in self.conditions
        ):
            raise TypeError("numeric dataflow conditions must be condition specs")
        if not isinstance(self.conclusion, Atom):
            raise TypeError("numeric dataflow conclusion must be an Atom")
        if self.target is not None and not isinstance(self.target, Term):
            raise TypeError("numeric dataflow target must be a Term")
        if any(not isinstance(blocker, Atom) for blocker in self.blocked_by):
            raise TypeError("numeric dataflow blockers must be Atoms")
        names = [condition.name for condition in self.conditions]
        if len(set(names)) != len(names):
            raise ValueError("numeric dataflow condition names must be unique")
        for value, label in (
            (self.rule_predicate_name, "rule predicate name"),
            (self.met_predicate_name, "met predicate name"),
            (self.conclusion_operator_name, "conclusion operator name"),
            (self.conclusion_guard_name, "conclusion guard name"),
        ):
            if value:
                _require_identifier(value, label)
        if not self.fact_source.strip():
            raise ValueError("numeric dataflow fact_source cannot be empty")
        if collect_variables(self.conclusion):
            raise TypeValidationError("numeric dataflow conclusion must be ground")
        if self.target is not None and collect_variables(self.target):
            raise TypeValidationError("numeric dataflow target must be ground")
        if any(collect_variables(atom) for atom in self.blocked_by):
            raise TypeValidationError("numeric dataflow blockers must be ground")


@dataclass(frozen=True)
class CompiledDataflowRule:
    rule_facts: tuple[Fact, ...]
    rule_atoms: tuple[Atom, ...]
    condition_atoms: tuple[Atom, ...]
    comparison_operators: tuple[OperatorSpec, ...]
    conclusion_operator: OperatorSpec
    target: Term


class TypedDataflowCompiler:
    """Turn declarative numeric flow into verifier-replayable operators."""

    def __init__(self, registry: KernelRegistry) -> None:
        self.registry = registry

    def compile(self, spec: NumericDataflowRuleSpec) -> CompiledDataflowRule:
        registry = self.registry
        entity_type = registry.types.ensure("Entity")
        number_type = registry.types.ensure("Number", entity_type)
        condition_type = registry.types.ensure("Condition", entity_type)
        comparator_type = registry.types.ensure("Comparator", entity_type)
        measurement_type = registry.types.ensure("Measurement", entity_type)
        target_type = registry.types.ensure("RuleTarget", entity_type)

        self._validate_registered_atom(spec.conclusion, "conclusion")
        if not spec.conclusion.predicate.verified:
            raise TypeValidationError("numeric dataflow conclusion must be verified")
        for blocker in spec.blocked_by:
            self._validate_registered_atom(blocker, "blocker")

        target = spec.target or registry.symbol(spec.name, target_type)
        contexts = tuple(
            condition.rule_context
            or registry.symbol(f"measurement_{condition.name}", measurement_type)
            for condition in spec.conditions
        )
        context_type = contexts[0].type
        if any(context.type != context_type for context in contexts[1:]):
            raise TypeValidationError(
                "conditions sharing one dataflow rule predicate must use the same "
                "rule_context type"
            )

        comparison_names = tuple(
            condition.operator_name or f"verify_{spec.name}_{condition.name}"
            for condition in spec.conditions
        )
        comparison_guard_names = tuple(
            condition.guard_name or f"guard_{spec.name}_{condition.name}"
            for condition in spec.conditions
        )
        conclusion_name = spec.conclusion_operator_name or f"conclude_{spec.name}"
        conclusion_guard_name = (
            spec.conclusion_guard_name or f"guard_{spec.name}_not_blocked"
        )
        self._validate_names(
            comparison_names,
            comparison_guard_names,
            conclusion_name,
            conclusion_guard_name if spec.blocked_by else "",
        )

        for condition in spec.conditions:
            self._validate_measurement(condition.measurement, number_type)
        self._validate_contexts(contexts)
        self._validate_target(target)

        rule_predicate_name = (
            spec.rule_predicate_name or f"{spec.name.upper()}_NUMERIC_RULE"
        )
        met_predicate_name = (
            spec.met_predicate_name or f"{spec.name.upper()}_CONDITION_MET"
        )
        rule_predicate = _ensure_verified_predicate(
            registry,
            rule_predicate_name,
            (
                condition_type,
                context_type,
                comparator_type,
                number_type,
                target.type,
            ),
        )
        met_predicate = _ensure_verified_predicate(
            registry,
            met_predicate_name,
            (condition_type, target.type),
        )

        rule_atoms: list[Atom] = []
        condition_atoms: list[Atom] = []
        comparison_operators: list[OperatorSpec] = []
        for condition, context, operator_name, guard_name in zip(
            spec.conditions,
            contexts,
            comparison_names,
            comparison_guard_names,
            strict=True,
        ):
            condition_id = registry.symbol(condition.name, condition_type)
            comparator = registry.symbol(condition.comparator, comparator_type)
            threshold = registry.symbol(str(condition.threshold), number_type)
            rule_atom = registry.atom(
                rule_predicate,
                condition_id,
                context,
                comparator,
                threshold,
                target,
            )
            condition_atom = registry.atom(
                met_predicate,
                condition_id,
                target,
            )
            measurement_value_type = condition.measurement.predicate.argument_types[
                condition.measurement.value_position
            ]
            measured_value = registry.variable(
                condition.value_variable_name,
                measurement_value_type,
            )
            measurement_atom = registry.atom(
                condition.measurement.predicate,
                *condition.measurement.arguments_with(measured_value),
            )
            registry.register_guard(
                guard_name,
                _numeric_condition_guard(
                    condition.comparator,
                    condition.threshold,
                    condition.value_variable_name,
                ),
            )
            comparison_operators.append(
                registry.register_operator(
                    Rule(
                        name=operator_name,
                        parameters=(measured_value,),
                        preconditions=(measurement_atom, rule_atom),
                        effects=(condition_atom,),
                        guards=(guard_name,),
                        description_ko=(
                            condition.description_ko
                            or "측정값 {value}와 기준을 비교해 조건을 검증했다."
                        ).replace(
                            "{value}",
                            "{" + condition.value_variable_name + "}",
                        ),
                    ),
                    family=OperatorFamily.COMPARE.value,
                    tags=spec.compare_tags,
                )
            )
            rule_atoms.append(rule_atom)
            condition_atoms.append(condition_atom)

        guards: tuple[str, ...] = ()
        if spec.blocked_by:
            registry.register_guard(
                conclusion_guard_name,
                _not_blocked_guard(spec.blocked_by),
            )
            guards = (conclusion_guard_name,)
        conclusion_operator = registry.register_operator(
            Rule(
                name=conclusion_name,
                parameters=(),
                preconditions=tuple(condition_atoms),
                effects=(spec.conclusion,),
                guards=guards,
                description_ko=(
                    spec.conclusion_description_ko
                    or "검증된 수치 조건을 모두 만족해 결론을 도출했다."
                ),
            ),
            family=OperatorFamily.COMPOSE.value,
            tags=spec.conclusion_tags,
        )
        rule_facts = tuple(
            Fact(atom, FactStatus.OBSERVED, spec.fact_source) for atom in rule_atoms
        )
        return CompiledDataflowRule(
            rule_facts=rule_facts,
            rule_atoms=tuple(rule_atoms),
            condition_atoms=tuple(condition_atoms),
            comparison_operators=tuple(comparison_operators),
            conclusion_operator=conclusion_operator,
            target=target,
        )

    def _validate_measurement(
        self,
        measurement: NumericMeasurementRef,
        number_type,
    ) -> None:
        registry = self.registry
        predicate = registry.predicates.get(measurement.predicate.name)
        if predicate != measurement.predicate:
            raise KernelError(
                "measurement predicate is not registered here: "
                f"{measurement.predicate.name}"
            )
        if not predicate.verified:
            raise TypeValidationError("numeric measurement predicate must be verified")
        value_type = predicate.argument_types[measurement.value_position]
        if not registry.types.is_assignable(value_type, number_type):
            raise TypeValidationError(
                f"measurement {predicate.name} value position expects {value_type}, "
                "not Number"
            )
        fixed_types = tuple(
            expected
            for index, expected in enumerate(predicate.argument_types)
            if index != measurement.value_position
        )
        for index, (argument, expected) in enumerate(
            zip(measurement.fixed_arguments, fixed_types, strict=True),
            start=1,
        ):
            if not registry.types.is_assignable(argument.type, expected):
                raise TypeValidationError(
                    f"measurement {predicate.name} fixed argument {index} expects "
                    f"{expected}, got {argument.type}"
                )

    def _validate_registered_atom(self, atom: Atom, label: str) -> None:
        predicate = self.registry.predicates.get(atom.predicate.name)
        if predicate != atom.predicate:
            raise KernelError(
                f"numeric dataflow {label} predicate is not registered here: "
                f"{atom.predicate.name}"
            )
        if len(atom.arguments) != predicate.arity:
            raise TypeValidationError(
                f"numeric dataflow {label} has invalid arity for {predicate.name}"
            )
        for argument, expected in zip(
            atom.arguments,
            predicate.argument_types,
            strict=True,
        ):
            if not self.registry.types.is_assignable(argument.type, expected):
                raise TypeValidationError(
                    f"numeric dataflow {label} expects {expected}, got {argument.type}"
                )

    def _validate_contexts(self, contexts: tuple[Term, ...]) -> None:
        for context in contexts:
            self.registry.types.resolve(context.type)
            if collect_variables(context):
                raise TypeValidationError("numeric dataflow rule_context must be ground")

    def _validate_target(self, target: Term) -> None:
        self.registry.types.resolve(target.type)
        if collect_variables(target):
            raise TypeValidationError("numeric dataflow target must be ground")

    def _validate_names(
        self,
        comparison_names: tuple[str, ...],
        guard_names: tuple[str, ...],
        conclusion_name: str,
        conclusion_guard_name: str,
    ) -> None:
        operator_names = comparison_names + (conclusion_name,)
        if len(set(operator_names)) != len(operator_names):
            raise KernelError("numeric dataflow operator names must be unique")
        used_operators = sorted(set(operator_names) & self.registry.operators.keys())
        if used_operators:
            raise KernelError(
                "numeric dataflow operators already registered: "
                + ", ".join(used_operators)
            )
        active_guards = guard_names + (
            (conclusion_guard_name,) if conclusion_guard_name else ()
        )
        if len(set(active_guards)) != len(active_guards):
            raise KernelError("numeric dataflow guard names must be unique")
        used_guards = sorted(set(active_guards) & self.registry.guards.keys())
        if used_guards:
            raise KernelError(
                "numeric dataflow guards already registered: "
                + ", ".join(used_guards)
            )


def _ensure_verified_predicate(
    registry: KernelRegistry,
    name: str,
    argument_types,
) -> Predicate:
    existing = registry.predicates.get(name)
    expected = tuple(registry.types.resolve(item) for item in argument_types)
    if existing is not None:
        if existing.argument_types != expected or not existing.verified:
            raise TypeValidationError(f"predicate {name} has an incompatible schema")
        return existing
    return registry.register_predicate(name, expected)


def _numeric_condition_guard(
    comparator: str,
    threshold: Fraction,
    binding_name: str,
):
    def verify(binding, _state: WorldState) -> bool:
        try:
            measured = Fraction(str(binding[binding_name]))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return False
        if comparator == ">":
            return measured > threshold
        if comparator == ">=":
            return measured >= threshold
        if comparator == "<":
            return measured < threshold
        if comparator == "<=":
            return measured <= threshold
        if comparator == "==":
            return measured == threshold
        if comparator == "!=":
            return measured != threshold
        return False

    return verify


def _not_blocked_guard(blockers: tuple[Atom, ...]):
    def verify(_binding, state: WorldState) -> bool:
        return not any(
            state.contains(blocker, proof_eligible=False) for blocker in blockers
        )

    return verify


def _require_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be an ASCII identifier")

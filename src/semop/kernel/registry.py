from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import permutations
from typing import TYPE_CHECKING

from .model import (
    Atom,
    FunctionSpec,
    KernelError,
    OperatorSpec,
    Predicate,
    Rule,
    Symbol,
    Term,
    TermApplication,
    TypeRef,
    TypeSystem,
    TypeValidationError,
    Variable,
    WorldState,
    collect_variables,
)

if TYPE_CHECKING:
    GuardFn = Callable[[Mapping[str, Term], WorldState], bool]
else:
    GuardFn = Callable


class KernelRegistry:
    """Owns typed functions, predicates, guards, and executable operators."""

    def __init__(self, type_system: TypeSystem | None = None) -> None:
        self.types = type_system or TypeSystem()
        self.functions: dict[str, FunctionSpec] = {}
        self.predicates: dict[str, Predicate] = {}
        self.operators: dict[str, OperatorSpec] = {}
        self.guards: dict[str, GuardFn] = {}

    def register_function(
        self,
        name: str,
        input_types: Sequence[str | TypeRef],
        output_type: str | TypeRef,
        *,
        symmetry_groups: Sequence[Sequence[int]] = (),
        allow_repeated_arguments: bool = True,
    ) -> FunctionSpec:
        if name in self.functions:
            raise KernelError(f"function already registered: {name}")
        inputs = tuple(self.types.resolve(type_ref) for type_ref in input_types)
        groups = _validate_symmetry_groups(symmetry_groups, len(inputs), name)
        function = FunctionSpec(
            name=name,
            input_types=inputs,
            output_type=self.types.resolve(output_type),
            symmetry_groups=groups,
            allow_repeated_arguments=allow_repeated_arguments,
        )
        self.functions[name] = function
        return function

    def register_predicate(
        self,
        name: str,
        argument_types: Sequence[str | TypeRef],
        *,
        symmetry_groups: Sequence[Sequence[int]] = (),
        verified: bool = True,
        metadata: Mapping[str, str] | None = None,
    ) -> Predicate:
        if name in self.predicates:
            raise KernelError(f"predicate already registered: {name}")
        types = tuple(self.types.resolve(type_ref) for type_ref in argument_types)
        predicate = Predicate(
            name=name,
            argument_types=types,
            symmetry_groups=_validate_symmetry_groups(
                symmetry_groups, len(types), name
            ),
            verified=verified,
            metadata=tuple(sorted((metadata or {}).items())),
        )
        self.predicates[name] = predicate
        return predicate

    def ensure_unverified_predicate(
        self,
        name: str,
        argument_types: Sequence[str | TypeRef],
    ) -> Predicate:
        existing = self.predicates.get(name)
        if existing is not None:
            expected = tuple(self.types.resolve(item) for item in argument_types)
            if existing.argument_types != expected:
                raise TypeValidationError(
                    f"predicate {name} has incompatible signatures"
                )
            return existing
        return self.register_predicate(
            name,
            argument_types,
            verified=False,
            metadata={"status": "unverified"},
        )

    def register_guard(self, name: str, guard: GuardFn) -> None:
        if name in self.guards:
            raise KernelError(f"guard already registered: {name}")
        self.guards[name] = guard

    def register_operator(
        self,
        operator: OperatorSpec | Rule,
        *,
        family: str | None = None,
        tags: Sequence[str] = (),
    ) -> OperatorSpec:
        if isinstance(operator, OperatorSpec):
            if family is not None or tags:
                raise KernelError(
                    "family and tags must be set on OperatorSpec when registering a spec"
                )
            spec = operator
        else:
            spec = OperatorSpec(
                operator,
                family=family or "reasoning",
                tags=tuple(sorted(set(tags))),
            )
        if spec.name in self.operators:
            raise KernelError(f"operator already registered: {spec.name}")
        self._validate_operator(spec)
        self.operators[spec.name] = spec
        return spec

    def symbol(self, name: str, type_ref: str | TypeRef) -> Symbol:
        return Symbol(name=name, type=self.types.resolve(type_ref))

    def variable(self, name: str, type_ref: str | TypeRef) -> Variable:
        return Variable(name=name, type=self.types.resolve(type_ref))

    def apply(self, function: str | FunctionSpec, *arguments: Term) -> TermApplication:
        spec = self.functions[function] if isinstance(function, str) else function
        if self.functions.get(spec.name) != spec:
            raise KernelError(f"function is not registered in this registry: {spec.name}")
        self._validate_arguments(spec.name, spec.input_types, arguments)
        if not spec.allow_repeated_arguments:
            keys = [argument.canonical_key() for argument in arguments]
            if len(keys) != len(set(keys)):
                raise TypeValidationError(
                    f"degenerate term {spec.name}: repeated arguments are not allowed"
                )
        canonical = _canonicalize_arguments(tuple(arguments), spec.symmetry_groups)
        return TermApplication(function=spec, arguments=canonical)

    def atom(self, predicate: str | Predicate, *arguments: Term) -> Atom:
        spec = self.predicates[predicate] if isinstance(predicate, str) else predicate
        if self.predicates.get(spec.name) != spec:
            raise KernelError(
                f"predicate is not registered in this registry: {spec.name}"
            )
        self._validate_arguments(spec.name, spec.argument_types, arguments)
        canonical = _canonicalize_arguments(tuple(arguments), spec.symmetry_groups)
        return Atom(predicate=spec, arguments=canonical)

    def evaluate_guards(
        self,
        operator: OperatorSpec,
        bindings: Mapping[str, Term],
        state: WorldState,
    ) -> bool:
        return all(self.guards[name](bindings, state) for name in operator.guards)

    def _validate_arguments(
        self,
        name: str,
        expected_types: Sequence[TypeRef],
        arguments: Sequence[Term],
    ) -> None:
        if len(arguments) != len(expected_types):
            raise TypeValidationError(
                f"{name} expects {len(expected_types)} arguments, got {len(arguments)}"
            )
        for index, (argument, expected) in enumerate(
            zip(arguments, expected_types, strict=True), start=1
        ):
            if not self.types.is_assignable(argument.type, expected):
                raise TypeValidationError(
                    f"{name} argument {index} expects {expected}, got {argument.type}"
                )

    def _validate_operator(self, operator: OperatorSpec) -> None:
        declared = {parameter.name: parameter for parameter in operator.parameters}
        for parameter in operator.parameters:
            self.types.resolve(parameter.type)
        variables = collect_variables(operator.preconditions + operator.effects)
        for variable in variables:
            declared_variable = declared.get(variable.name)
            if declared_variable is None:
                raise TypeValidationError(
                    f"operator {operator.name} uses undeclared variable ?{variable.name}"
                )
            if declared_variable.type != variable.type:
                raise TypeValidationError(
                    f"operator {operator.name} gives ?{variable.name} inconsistent types"
                )
        bound_names = {
            variable.name for variable in collect_variables(operator.preconditions)
        }
        effect_names = {
            variable.name for variable in collect_variables(operator.effects)
        }
        unbound = sorted(effect_names - bound_names)
        if unbound:
            raise TypeValidationError(
                f"operator {operator.name} has effect variables not bound by preconditions: "
                + ", ".join(f"?{name}" for name in unbound)
            )
        for guard_name in operator.guards:
            if guard_name not in self.guards:
                raise KernelError(
                    f"operator {operator.name} references unknown guard: {guard_name}"
                )
        for atom in operator.preconditions + operator.effects:
            registered = self.predicates.get(atom.predicate.name)
            if registered != atom.predicate:
                raise KernelError(
                    f"operator {operator.name} references unregistered predicate "
                    f"{atom.predicate.name}"
                )
            self._validate_arguments(
                atom.predicate.name,
                atom.predicate.argument_types,
                atom.arguments,
            )
            for term in atom.arguments:
                self._validate_term_tree(term)

    def _validate_term_tree(self, term: Term) -> None:
        self.types.resolve(term.type)
        if not isinstance(term, TermApplication):
            return
        if self.functions.get(term.function.name) != term.function:
            raise KernelError(f"unregistered function: {term.function.name}")
        self._validate_arguments(
            term.function.name, term.function.input_types, term.arguments
        )
        for argument in term.arguments:
            self._validate_term_tree(argument)


def _validate_symmetry_groups(
    groups: Sequence[Sequence[int]], arity: int, owner: str
) -> tuple[tuple[int, ...], ...]:
    normalized: list[tuple[int, ...]] = []
    occupied: set[int] = set()
    for group in groups:
        indices = tuple(sorted(set(group)))
        if len(indices) < 2:
            raise KernelError(f"{owner} symmetry groups must contain at least two slots")
        if indices[0] < 0 or indices[-1] >= arity:
            raise KernelError(f"{owner} symmetry group index is out of range")
        overlap = occupied.intersection(indices)
        if overlap:
            raise KernelError(f"{owner} symmetry groups overlap at {sorted(overlap)}")
        occupied.update(indices)
        normalized.append(indices)
    return tuple(normalized)


def _canonicalize_arguments(
    arguments: tuple[Term, ...], groups: Sequence[Sequence[int]]
) -> tuple[Term, ...]:
    result = list(arguments)
    for group in groups:
        ordered = sorted(
            (result[index] for index in group), key=lambda term: term.canonical_key()
        )
        for index, value in zip(group, ordered, strict=True):
            result[index] = value
    return tuple(result)


def atom_variants(atom: Atom) -> tuple[tuple[Term, ...], ...]:
    """Enumerate symmetry-equivalent argument orders for matching only."""

    variants: list[tuple[Term, ...]] = [atom.arguments]
    for group in atom.predicate.symmetry_groups:
        expanded: list[tuple[Term, ...]] = []
        for current in variants:
            group_values = tuple(current[index] for index in group)
            for permuted in permutations(group_values):
                candidate = list(current)
                for index, value in zip(group, permuted, strict=True):
                    candidate[index] = value
                expanded.append(tuple(candidate))
        variants = expanded
    unique = {tuple(term.canonical_key() for term in item): item for item in variants}
    return tuple(unique[key] for key in sorted(unique))


def application_variants(term: TermApplication) -> tuple[tuple[Term, ...], ...]:
    variants: list[tuple[Term, ...]] = [term.arguments]
    for group in term.function.symmetry_groups:
        expanded: list[tuple[Term, ...]] = []
        for current in variants:
            values = tuple(current[index] for index in group)
            for permuted in permutations(values):
                candidate = list(current)
                for index, value in zip(group, permuted, strict=True):
                    candidate[index] = value
                expanded.append(tuple(candidate))
        variants = expanded
    unique = {tuple(value.canonical_key() for value in item): item for item in variants}
    return tuple(unique[key] for key in sorted(unique))

from __future__ import annotations

from collections.abc import Mapping

from .model import Atom, Term, TermApplication, TypeSystem, Variable
from .registry import KernelRegistry, application_variants, atom_variants


Substitution = dict[str, Term]


def unify_atom(
    pattern: Atom,
    ground: Atom,
    type_system: TypeSystem,
    substitution: Mapping[str, Term] | None = None,
) -> tuple[Substitution, ...]:
    if pattern.predicate.name != ground.predicate.name:
        return ()
    initial = dict(substitution or {})
    solutions: list[Substitution] = []
    for arguments in atom_variants(ground):
        partial: tuple[Substitution, ...] = (initial,)
        for pattern_term, ground_term in zip(
            pattern.arguments, arguments, strict=True
        ):
            next_partial: list[Substitution] = []
            for candidate in partial:
                next_partial.extend(
                    unify_term(pattern_term, ground_term, type_system, candidate)
                )
            partial = tuple(next_partial)
            if not partial:
                break
        solutions.extend(partial)
    return _deduplicate_substitutions(solutions)


def unify_term(
    pattern: Term,
    ground: Term,
    type_system: TypeSystem,
    substitution: Mapping[str, Term] | None = None,
) -> tuple[Substitution, ...]:
    current = dict(substitution or {})
    if isinstance(pattern, Variable):
        if not type_system.is_assignable(ground.type, pattern.type):
            return ()
        existing = current.get(pattern.name)
        if existing is None:
            current[pattern.name] = ground
            return (current,)
        return (current,) if existing == ground else ()
    if type(pattern) is not type(ground):
        return ()
    if not isinstance(pattern, TermApplication):
        return (current,) if pattern == ground else ()
    assert isinstance(ground, TermApplication)
    if pattern.function.name != ground.function.name:
        return ()
    solutions: list[Substitution] = []
    for ground_arguments in application_variants(ground):
        partial: tuple[Substitution, ...] = (current,)
        for pattern_argument, ground_argument in zip(
            pattern.arguments, ground_arguments, strict=True
        ):
            next_partial: list[Substitution] = []
            for candidate in partial:
                next_partial.extend(
                    unify_term(
                        pattern_argument,
                        ground_argument,
                        type_system,
                        candidate,
                    )
                )
            partial = tuple(next_partial)
            if not partial:
                break
        solutions.extend(partial)
    return _deduplicate_substitutions(solutions)


def substitute_term(
    term: Term, substitution: Mapping[str, Term], registry: KernelRegistry
) -> Term:
    if isinstance(term, Variable):
        try:
            return substitution[term.name]
        except KeyError as exc:
            raise KeyError(f"unbound variable: ?{term.name}") from exc
    if isinstance(term, TermApplication):
        return registry.apply(
            term.function,
            *(substitute_term(argument, substitution, registry) for argument in term.arguments),
        )
    return term


def substitute_atom(
    atom: Atom, substitution: Mapping[str, Term], registry: KernelRegistry
) -> Atom:
    return registry.atom(
        atom.predicate,
        *(substitute_term(argument, substitution, registry) for argument in atom.arguments),
    )


def _deduplicate_substitutions(
    substitutions: list[Substitution],
) -> tuple[Substitution, ...]:
    unique: dict[tuple[tuple[str, tuple], ...], Substitution] = {}
    for substitution in substitutions:
        key = tuple(
            sorted(
                (name, term.canonical_key())
                for name, term in substitution.items()
            )
        )
        unique[key] = substitution
    return tuple(unique[key] for key in sorted(unique))

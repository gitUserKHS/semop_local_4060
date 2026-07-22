from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Sequence


class KernelError(ValueError):
    """Base exception for invalid typed-kernel declarations or execution."""


class TypeValidationError(KernelError):
    """Raised when a term, predicate, or binding violates its signature."""


@dataclass(frozen=True, order=True)
class TypeRef:
    """Nominal type identity. Inheritance is owned by :class:`TypeSystem`."""

    name: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise TypeValidationError("type name must not be empty")

    def __str__(self) -> str:
        return self.name


class TypeSystem:
    """A small nominal type system with single inheritance."""

    def __init__(self) -> None:
        self._types: dict[str, TypeRef] = {}
        self._parents: dict[str, str | None] = {}

    def register(self, name: str, parent: str | TypeRef | None = None) -> TypeRef:
        clean_name = name.strip()
        if clean_name in self._types:
            raise TypeValidationError(f"type already registered: {clean_name}")
        parent_name = self._type_name(parent) if parent is not None else None
        if parent_name is not None and parent_name not in self._types:
            raise TypeValidationError(f"unknown parent type: {parent_name}")
        type_ref = TypeRef(clean_name)
        self._types[clean_name] = type_ref
        self._parents[clean_name] = parent_name
        return type_ref

    def ensure(self, name: str, parent: str | TypeRef | None = None) -> TypeRef:
        existing = self._types.get(name)
        if existing is not None:
            expected_parent = self._type_name(parent) if parent is not None else None
            if parent is not None and self._parents[name] != expected_parent:
                raise TypeValidationError(
                    f"type {name} already has parent {self._parents[name]!r}, not {expected_parent!r}"
                )
            return existing
        return self.register(name, parent)

    def resolve(self, value: str | TypeRef) -> TypeRef:
        name = self._type_name(value)
        try:
            return self._types[name]
        except KeyError as exc:
            raise TypeValidationError(f"unknown type: {name}") from exc

    def parent_of(self, value: str | TypeRef) -> TypeRef | None:
        type_ref = self.resolve(value)
        parent = self._parents[type_ref.name]
        return self._types[parent] if parent is not None else None

    def is_assignable(self, actual: str | TypeRef, expected: str | TypeRef) -> bool:
        current = self.resolve(actual).name
        target = self.resolve(expected).name
        while current is not None:
            if current == target:
                return True
            current = self._parents[current]
        return False

    def ancestry(self, value: str | TypeRef) -> tuple[TypeRef, ...]:
        current: str | None = self.resolve(value).name
        result: list[TypeRef] = []
        while current is not None:
            result.append(self._types[current])
            current = self._parents[current]
        return tuple(result)

    def snapshot(self) -> tuple[tuple[str, str | None], ...]:
        return tuple(sorted(self._parents.items()))

    @staticmethod
    def _type_name(value: str | TypeRef | None) -> str:
        if isinstance(value, TypeRef):
            return value.name
        if value is None:
            raise TypeValidationError("type must not be None")
        return value.strip()


class Term:
    """Marker base class for immutable typed terms."""

    type: TypeRef

    def canonical_key(self) -> tuple[Any, ...]:  # pragma: no cover - abstract marker
        raise NotImplementedError


@dataclass(frozen=True)
class Symbol(Term):
    name: str
    type: TypeRef

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise TypeValidationError("symbol name must not be empty")

    def canonical_key(self) -> tuple[Any, ...]:
        return ("symbol", self.type.name, self.name)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Variable(Term):
    name: str
    type: TypeRef

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise TypeValidationError("variable name must not be empty")

    def canonical_key(self) -> tuple[Any, ...]:
        return ("variable", self.type.name, self.name)

    def __str__(self) -> str:
        return f"?{self.name}"


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    input_types: tuple[TypeRef, ...]
    output_type: TypeRef
    symmetry_groups: tuple[tuple[int, ...], ...] = ()
    allow_repeated_arguments: bool = True

    @property
    def arity(self) -> int:
        return len(self.input_types)


@dataclass(frozen=True)
class TermApplication(Term):
    function: FunctionSpec
    arguments: tuple[Term, ...]

    @property
    def type(self) -> TypeRef:
        return self.function.output_type

    def canonical_key(self) -> tuple[Any, ...]:
        return (
            "application",
            self.function.name,
            tuple(argument.canonical_key() for argument in self.arguments),
        )

    def __str__(self) -> str:
        return f"{self.function.name}({', '.join(map(str, self.arguments))})"


@dataclass(frozen=True)
class Predicate:
    name: str
    argument_types: tuple[TypeRef, ...]
    symmetry_groups: tuple[tuple[int, ...], ...] = ()
    verified: bool = True
    metadata: tuple[tuple[str, str], ...] = ()

    @property
    def arity(self) -> int:
        return len(self.argument_types)


@dataclass(frozen=True)
class Atom:
    predicate: Predicate
    arguments: tuple[Term, ...]

    def canonical_key(self) -> tuple[Any, ...]:
        return (
            self.predicate.name,
            tuple(argument.canonical_key() for argument in self.arguments),
        )

    def __str__(self) -> str:
        return f"{self.predicate.name}({', '.join(map(str, self.arguments))})"


class FactStatus(str, Enum):
    OBSERVED = "observed"
    ASSUMED = "assumed"
    DERIVED = "derived"
    PROPOSED = "proposed"
    CONTRADICTED = "contradicted"

    @property
    def proof_eligible(self) -> bool:
        return self in {
            FactStatus.OBSERVED,
            FactStatus.ASSUMED,
            FactStatus.DERIVED,
        }


class AssertionStatus(str, Enum):
    """How an atom entered the typed world before logical execution."""

    EXPLICIT = "explicit"
    MEASURED = "measured"
    INFERRED = "inferred"
    IMPORTED = "imported"
    GENERATED = "generated"
    UNKNOWN = "unknown"


class EvidenceStatus(str, Enum):
    """What kind of evidence supports a fact, independently of logical status."""

    EXTERNAL_VERIFIED = "external_verified"
    ADAPTER_VERIFIED = "adapter_verified"
    UNVERIFIED = "unverified"
    ASSUMED = "assumed"
    DERIVED = "derived"


class OperatorFamily(str, Enum):
    """Domain-neutral behavior families exposed to the shared controller."""

    OBSERVE = "observe"
    RELATE = "relate"
    TRANSFORM = "transform"
    COMPARE = "compare"
    QUANTIFY = "quantify"
    CONTROL = "control"
    SEARCH = "search"
    VERIFY = "verify"
    MEMORY = "memory"
    COMPOSE = "compose"
    REASONING = "reasoning"


@dataclass(frozen=True)
class Fact:
    atom: Atom
    status: FactStatus = FactStatus.OBSERVED
    source: str = "input"
    confidence: float = 1.0
    assertion_status: AssertionStatus = AssertionStatus.UNKNOWN
    evidence_status: EvidenceStatus | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", FactStatus(self.status))
        object.__setattr__(
            self,
            "assertion_status",
            AssertionStatus(self.assertion_status),
        )
        evidence_status = self.evidence_status
        if evidence_status is None:
            if self.status is FactStatus.ASSUMED:
                evidence_status = EvidenceStatus.ASSUMED
            elif self.status is FactStatus.DERIVED:
                evidence_status = EvidenceStatus.DERIVED
            else:
                evidence_status = EvidenceStatus.UNVERIFIED
        object.__setattr__(self, "evidence_status", EvidenceStatus(evidence_status))
        if not 0.0 <= self.confidence <= 1.0:
            raise KernelError("fact confidence must be between 0 and 1")

    @property
    def proof_eligible(self) -> bool:
        return self.status.proof_eligible and self.atom.predicate.verified

    @property
    def logical_status(self) -> FactStatus:
        """Compatibility-safe name for the proof engine's fact status axis."""

        return self.status

    def __str__(self) -> str:
        return f"[{self.status.value}] {self.atom}"


@dataclass(frozen=True)
class Goal:
    atom: Atom
    label: str = ""

    def __str__(self) -> str:
        return self.label or str(self.atom)


@dataclass(frozen=True)
class Rule:
    name: str
    parameters: tuple[Variable, ...]
    preconditions: tuple[Atom, ...]
    effects: tuple[Atom, ...]
    guards: tuple[str, ...] = ()
    cost: float = 1.0
    description_ko: str = "{operator}를 적용했다."

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise KernelError("rule name must not be empty")
        if not self.effects:
            raise KernelError(f"rule {self.name} must have at least one effect")
        if self.cost <= 0:
            raise KernelError(f"rule {self.name} must have positive cost")
        names = [parameter.name for parameter in self.parameters]
        if len(set(names)) != len(names):
            raise KernelError(f"rule {self.name} has duplicate parameter names")


@dataclass(frozen=True)
class OperatorSpec:
    rule: Rule
    family: str = OperatorFamily.REASONING.value
    tags: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        family = self.family.value if isinstance(self.family, OperatorFamily) else self.family
        if not family or not family.strip():
            raise KernelError("operator family must not be empty")
        object.__setattr__(self, "family", family.strip().lower())
        object.__setattr__(
            self,
            "tags",
            tuple(sorted({tag.strip().lower() for tag in self.tags if tag.strip()})),
        )

    @property
    def name(self) -> str:
        return self.rule.name

    @property
    def parameters(self) -> tuple[Variable, ...]:
        return self.rule.parameters

    @property
    def preconditions(self) -> tuple[Atom, ...]:
        return self.rule.preconditions

    @property
    def effects(self) -> tuple[Atom, ...]:
        return self.rule.effects

    @property
    def guards(self) -> tuple[str, ...]:
        return self.rule.guards

    @property
    def cost(self) -> float:
        return self.rule.cost


Binding = tuple[tuple[str, Term], ...]


@dataclass(frozen=True)
class GroundAction:
    operator: OperatorSpec
    bindings: Binding
    preconditions: tuple[Atom, ...]
    effects: tuple[Atom, ...]
    cost: float

    def binding_map(self) -> dict[str, Term]:
        return dict(self.bindings)

    def canonical_key(self) -> tuple[Any, ...]:
        return (
            self.operator.name,
            tuple((name, term.canonical_key()) for name, term in self.bindings),
            tuple(effect.canonical_key() for effect in self.effects),
        )

    def __str__(self) -> str:
        values = ", ".join(f"{name}={term}" for name, term in self.bindings)
        return f"{self.operator.name}({values})"


@dataclass(frozen=True)
class WorldState:
    facts: tuple[Fact, ...] = ()
    depth: int = 0
    path_cost: float = 0.0

    def __post_init__(self) -> None:
        unique: dict[tuple[Any, ...], Fact] = {}
        for fact in self.facts:
            key = (
                fact.atom.canonical_key(),
                fact.status.value,
                fact.source,
                fact.confidence,
                fact.assertion_status.value,
                fact.evidence_status.value,
            )
            unique[key] = fact
        ordered = tuple(
            sorted(
                unique.values(),
                key=lambda fact: (
                    fact.atom.canonical_key(),
                    fact.status.value,
                    fact.source,
                    fact.confidence,
                    fact.assertion_status.value,
                    fact.evidence_status.value,
                ),
            )
        )
        object.__setattr__(self, "facts", ordered)
        if self.depth < 0 or self.path_cost < 0:
            raise KernelError("state depth and path cost must not be negative")

    @property
    def eligible_facts(self) -> tuple[Fact, ...]:
        return tuple(fact for fact in self.facts if fact.proof_eligible)

    @property
    def eligible_atoms(self) -> frozenset[Atom]:
        return frozenset(fact.atom for fact in self.eligible_facts)

    def contains(self, atom: Atom, *, proof_eligible: bool = True) -> bool:
        facts = self.eligible_facts if proof_eligible else self.facts
        return any(fact.atom == atom for fact in facts)

    def signature(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(sorted(atom.canonical_key() for atom in self.eligible_atoms))

    def digest(self) -> str:
        trust_signature = tuple(
            (
                fact.atom.canonical_key(),
                fact.status.value,
                fact.source,
                fact.confidence,
                fact.assertion_status.value,
                fact.evidence_status.value,
            )
            for fact in self.facts
        )
        payload = json.dumps(
            trust_signature,
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )
        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    def terms(self) -> tuple[Term, ...]:
        found: dict[tuple[Any, ...], Term] = {}

        def visit(term: Term) -> None:
            found[term.canonical_key()] = term
            if isinstance(term, TermApplication):
                for argument in term.arguments:
                    visit(argument)

        for fact in self.eligible_facts:
            for argument in fact.atom.arguments:
                visit(argument)
        return tuple(found[key] for key in sorted(found))


@dataclass(frozen=True)
class ProofStep:
    index: int
    action: GroundAction
    premises: tuple[Atom, ...]
    effects: tuple[Atom, ...]
    before_digest: str
    after_digest: str


@dataclass(frozen=True)
class ProofDependencies:
    """Initial proof leaves grouped by their logical trust role."""

    observed: tuple[Fact, ...] = ()
    assumed: tuple[Fact, ...] = ()
    derived: tuple[Fact, ...] = ()

    @property
    def all_dependencies(self) -> tuple[Fact, ...]:
        return self.observed + self.assumed + self.derived

    @property
    def unverified(self) -> tuple[Fact, ...]:
        return tuple(
            fact
            for fact in self.all_dependencies
            if fact.evidence_status is EvidenceStatus.UNVERIFIED
        )

    @property
    def conditional(self) -> bool:
        return bool(self.assumed)

    @property
    def evidence_complete(self) -> bool:
        trusted = {
            EvidenceStatus.EXTERNAL_VERIFIED,
            EvidenceStatus.ADAPTER_VERIFIED,
        }
        return not self.assumed and not self.derived and all(
            fact.evidence_status in trusted for fact in self.observed
        )


@dataclass(frozen=True)
class SolveBudget:
    max_steps: int = 32
    max_expansions: int = 20_000
    max_facts: int = 10_000
    timeout_seconds: float = 10.0
    beam_width: int = 4
    top_operators: int = 8
    argument_candidates: int = 4

    def __post_init__(self) -> None:
        integer_fields = (
            self.max_steps,
            self.max_expansions,
            self.max_facts,
            self.beam_width,
            self.top_operators,
            self.argument_candidates,
        )
        if any(value <= 0 for value in integer_fields) or self.timeout_seconds <= 0:
            raise KernelError("all solve budget limits must be positive")


@dataclass(frozen=True)
class GoalOutcome:
    goal: Goal
    proven: bool
    proof_step: int | None = None


@dataclass(frozen=True)
class SolveResult:
    success: bool
    verified: bool
    goals: tuple[GoalOutcome, ...]
    initial_state: WorldState
    final_state: WorldState
    proof: tuple[ProofStep, ...]
    expansions: int
    elapsed_seconds: float
    halt_reason: str
    policy_used: bool = False
    fallback_used: bool = False
    diagnostics: tuple[str, ...] = ()
    inference_rounds: int = 0
    dependencies: ProofDependencies = ProofDependencies()

    @property
    def conditional(self) -> bool:
        return self.dependencies.conditional

    @property
    def assumption_dependencies(self) -> tuple[Fact, ...]:
        return self.dependencies.assumed

    @property
    def observed_dependencies(self) -> tuple[Fact, ...]:
        return self.dependencies.observed

    @property
    def derived_input_dependencies(self) -> tuple[Fact, ...]:
        return self.dependencies.derived

    @property
    def unverified_dependencies(self) -> tuple[Fact, ...]:
        return self.dependencies.unverified

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "verified": self.verified,
            "goals": [
                {
                    "goal": str(outcome.goal),
                    "proven": outcome.proven,
                    "proof_step": outcome.proof_step,
                }
                for outcome in self.goals
            ],
            "facts": [str(fact) for fact in self.final_state.facts],
            "proof": [
                {
                    "index": step.index,
                    "operator": step.action.operator.name,
                    "bindings": {
                        name: str(term) for name, term in step.action.bindings
                    },
                    "premises": [str(atom) for atom in step.premises],
                    "effects": [str(atom) for atom in step.effects],
                }
                for step in self.proof
            ],
            "expansions": self.expansions,
            "elapsed_seconds": self.elapsed_seconds,
            "halt_reason": self.halt_reason,
            "policy_used": self.policy_used,
            "fallback_used": self.fallback_used,
            "diagnostics": list(self.diagnostics),
            "inference_rounds": self.inference_rounds,
            "trust": {
                "conditional": self.conditional,
                "evidence_complete": self.dependencies.evidence_complete,
                "observed_dependencies": [
                    _fact_payload(fact) for fact in self.observed_dependencies
                ],
                "assumption_dependencies": [
                    _fact_payload(fact) for fact in self.assumption_dependencies
                ],
                "derived_input_dependencies": [
                    _fact_payload(fact) for fact in self.derived_input_dependencies
                ],
                "unverified_dependencies": [
                    _fact_payload(fact) for fact in self.unverified_dependencies
                ],
            },
        }


def collect_proof_dependencies(
    initial_state: WorldState,
    goals: Sequence[Goal],
    proof: Sequence[ProofStep],
) -> ProofDependencies:
    """Find the strongest available initial fact for every proof-leaf atom."""

    used_atoms = {premise for step in proof for premise in step.premises}
    used_atoms.update(
        goal.atom for goal in goals if initial_state.contains(goal.atom)
    )
    by_atom: dict[Atom, list[Fact]] = {}
    for fact in initial_state.eligible_facts:
        if fact.atom in used_atoms:
            by_atom.setdefault(fact.atom, []).append(fact)

    selected = tuple(
        min(facts, key=_dependency_rank)
        for _atom, facts in sorted(
            by_atom.items(),
            key=lambda item: item[0].canonical_key(),
        )
    )
    return ProofDependencies(
        observed=tuple(
            fact for fact in selected if fact.status is FactStatus.OBSERVED
        ),
        assumed=tuple(
            fact for fact in selected if fact.status is FactStatus.ASSUMED
        ),
        derived=tuple(
            fact for fact in selected if fact.status is FactStatus.DERIVED
        ),
    )


def _dependency_rank(fact: Fact) -> tuple[int, int, str, float]:
    logical_rank = {
        FactStatus.OBSERVED: 0,
        FactStatus.DERIVED: 1,
        FactStatus.ASSUMED: 2,
    }.get(fact.status, 3)
    evidence_rank = {
        EvidenceStatus.EXTERNAL_VERIFIED: 0,
        EvidenceStatus.ADAPTER_VERIFIED: 1,
        EvidenceStatus.UNVERIFIED: 2,
        EvidenceStatus.DERIVED: 3,
        EvidenceStatus.ASSUMED: 4,
    }[fact.evidence_status]
    return logical_rank, evidence_rank, fact.source, -fact.confidence


def _fact_payload(fact: Fact) -> dict[str, Any]:
    return {
        "atom": str(fact.atom),
        "logical_status": fact.logical_status.value,
        "assertion_status": fact.assertion_status.value,
        "evidence_status": fact.evidence_status.value,
        "source": fact.source,
        "confidence": fact.confidence,
    }


def collect_variables(value: Atom | Term | Sequence[Atom]) -> frozenset[Variable]:
    variables: set[Variable] = set()

    def visit_term(term: Term) -> None:
        if isinstance(term, Variable):
            variables.add(term)
        elif isinstance(term, TermApplication):
            for argument in term.arguments:
                visit_term(argument)

    def visit_atom(atom: Atom) -> None:
        for argument in atom.arguments:
            visit_term(argument)

    if isinstance(value, Atom):
        visit_atom(value)
    elif isinstance(value, Term):
        visit_term(value)
    else:
        for atom in value:
            visit_atom(atom)
    return frozenset(variables)

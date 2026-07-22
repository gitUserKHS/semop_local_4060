from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from .model import (
    Atom,
    OperatorSpec,
    ProofStep,
    SolveResult,
    Symbol,
    Term,
    TermApplication,
    Variable,
)

if TYPE_CHECKING:
    from .registry import KernelRegistry


@dataclass(frozen=True)
class MacroOperatorCandidate:
    name: str
    operator_sequence: tuple[str, ...]
    typed_pattern: tuple[str, ...]
    effect_signature: tuple[str, ...]
    support: int
    old_description_length: float
    new_description_length: float
    reduction_ratio: float
    evidence_trace_ids: tuple[str, ...]
    type_effect_verified: bool
    heldout_no_regression: bool
    operator_schema_fingerprints: tuple[str, ...] = ()

    @property
    def retained(self) -> bool:
        return (
            self.support >= 3
            and self.type_effect_verified
            and self.reduction_ratio >= 0.10
            and self.heldout_no_regression
        )


@dataclass(frozen=True)
class VerifiedMacroProgram:
    """A retained procedure that may only rank its primitive operators."""

    name: str
    primitive_operator_names: tuple[str, ...]
    operator_schema_fingerprints: tuple[str, ...]
    effect_signature: tuple[str, ...]
    support: int
    reduction_ratio: float

    def expand(self) -> tuple[str, ...]:
        return self.primitive_operator_names


@dataclass(frozen=True)
class MacroActivationIssue:
    macro_name: str
    reason: str


@dataclass(frozen=True)
class MacroActivationResult:
    programs: tuple[VerifiedMacroProgram, ...]
    rejected: tuple[MacroActivationIssue, ...]


class MdlMacroLibrary:
    """Bounded DreamCoder-style program compression over verified traces."""

    FORMAT_VERSION = 2

    def __init__(self, records: Iterable[MacroOperatorCandidate] = ()) -> None:
        self.records = tuple(record for record in records if record.retained)

    @classmethod
    def induce(
        cls,
        traces: Mapping[str, SolveResult],
        *,
        heldout_validator: Callable[[MacroOperatorCandidate], bool],
        min_length: int = 2,
        max_length: int = 6,
        min_support: int = 3,
        min_reduction: float = 0.10,
    ) -> "MdlMacroLibrary":
        if not 2 <= min_length <= max_length <= 6:
            raise ValueError("macro lengths must satisfy 2 <= min <= max <= 6")
        groups: dict[
            tuple[str, ...], dict[str, tuple[ProofStep, ...]]
        ] = defaultdict(dict)
        for trace_id, result in sorted(traces.items()):
            if not result.success or not result.verified:
                continue
            proof = result.proof
            for length in range(min_length, min(max_length, len(proof)) + 1):
                for start in range(0, len(proof) - length + 1):
                    sequence = proof[start : start + length]
                    if not _sequence_is_typed_and_chained(sequence):
                        continue
                    signature = _anti_unified_signature(sequence)
                    groups[signature][trace_id] = sequence

        candidates: list[MacroOperatorCandidate] = []
        for candidate_index, (signature, evidence) in enumerate(sorted(groups.items())):
            support = len(evidence)
            if support < min_support:
                continue
            sample = evidence[sorted(evidence)[0]]
            operator_sequence = tuple(step.action.operator.name for step in sample)
            schema_variants = {
                tuple(
                    operator_schema_fingerprint(step.action.operator)
                    for step in sequence
                )
                for sequence in evidence.values()
            }
            effect_variants = {
                tuple(_atom_type_signature(effect) for effect in sequence[-1].effects)
                for sequence in evidence.values()
            }
            type_effect_verified = (
                len(schema_variants) == 1 and len(effect_variants) == 1
            )
            schema_fingerprints = min(schema_variants)
            parameter_slots = len(
                {
                    token.split("slot:", 1)[1]
                    for token in signature
                    if "slot:" in token
                }
            )
            unit_length = len(operator_sequence) + max(1, parameter_slots)
            old_length = float(support * unit_length)
            new_length = float(unit_length + support)
            reduction = (old_length - new_length) / old_length
            effect_signature = tuple(
                _atom_type_signature(effect) for effect in sample[-1].effects
            )
            provisional = MacroOperatorCandidate(
                name=f"macro_{candidate_index:04d}_{'_'.join(operator_sequence)}",
                operator_sequence=operator_sequence,
                typed_pattern=signature,
                effect_signature=effect_signature,
                support=support,
                old_description_length=old_length,
                new_description_length=new_length,
                reduction_ratio=reduction,
                evidence_trace_ids=tuple(sorted(evidence)),
                type_effect_verified=type_effect_verified,
                heldout_no_regression=False,
                operator_schema_fingerprints=schema_fingerprints,
            )
            if reduction < min_reduction or not type_effect_verified:
                continue
            candidates.append(
                MacroOperatorCandidate(
                    **{
                        **asdict(provisional),
                        "heldout_no_regression": bool(
                            heldout_validator(provisional)
                        ),
                    }
                )
            )
        return cls(candidates)

    def expand(self, name: str) -> tuple[str, ...]:
        for record in self.records:
            if record.name == name:
                return record.operator_sequence
        raise KeyError(name)

    def activate(self, registry: "KernelRegistry") -> MacroActivationResult:
        """Validate retained schemas and expose primitive-only macro programs."""

        programs: list[VerifiedMacroProgram] = []
        rejected: list[MacroActivationIssue] = []
        for record in sorted(self.records, key=lambda item: item.name):
            if not record.operator_schema_fingerprints:
                rejected.append(
                    MacroActivationIssue(
                        record.name,
                        "legacy_record_missing_operator_schema_fingerprints",
                    )
                )
                continue
            if len(record.operator_schema_fingerprints) != len(
                record.operator_sequence
            ):
                rejected.append(
                    MacroActivationIssue(
                        record.name,
                        "operator_schema_fingerprint_count_mismatch",
                    )
                )
                continue
            mismatch = None
            for operator_name, expected in zip(
                record.operator_sequence,
                record.operator_schema_fingerprints,
                strict=True,
            ):
                operator = registry.operators.get(operator_name)
                if operator is None:
                    mismatch = f"missing_primitive_operator:{operator_name}"
                    break
                if operator_schema_fingerprint(operator) != expected:
                    mismatch = f"primitive_operator_schema_mismatch:{operator_name}"
                    break
            if mismatch is not None:
                rejected.append(MacroActivationIssue(record.name, mismatch))
                continue
            programs.append(
                VerifiedMacroProgram(
                    name=record.name,
                    primitive_operator_names=record.operator_sequence,
                    operator_schema_fingerprints=(
                        record.operator_schema_fingerprints
                    ),
                    effect_signature=record.effect_signature,
                    support=record.support,
                    reduction_ratio=record.reduction_ratio,
                )
            )
        return MacroActivationResult(tuple(programs), tuple(rejected))

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_payload()
        temporary = destination.with_name(
            f".{destination.name}.{uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "MdlMacroLibrary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_payload(payload)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "records": [asdict(record) for record in self.records],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MdlMacroLibrary":
        format_version = payload.get("format_version")
        if format_version not in {1, cls.FORMAT_VERSION}:
            raise ValueError("unsupported macro library format")
        records = []
        for raw in payload.get("records", []):
            records.append(
                MacroOperatorCandidate(
                    **{
                        **raw,
                        "operator_sequence": tuple(raw["operator_sequence"]),
                        "typed_pattern": tuple(raw["typed_pattern"]),
                        "effect_signature": tuple(raw["effect_signature"]),
                        "evidence_trace_ids": tuple(raw["evidence_trace_ids"]),
                        "operator_schema_fingerprints": tuple(
                            raw.get("operator_schema_fingerprints", ())
                        ),
                    }
                )
            )
        return cls(records)


def _sequence_is_typed_and_chained(sequence: Sequence[ProofStep]) -> bool:
    if not sequence:
        return False
    for step in sequence:
        binding_names = {name for name, _ in step.action.bindings}
        parameter_names = {parameter.name for parameter in step.action.operator.parameters}
        if binding_names != parameter_names:
            return False
        for parameter in step.action.operator.parameters:
            bound = step.action.binding_map()[parameter.name]
            if bound.type != parameter.type:
                return False
    return all(
        left.after_digest == right.before_digest
        for left, right in zip(sequence, sequence[1:])
    )


def _anti_unified_signature(sequence: Sequence[ProofStep]) -> tuple[str, ...]:
    slots: dict[tuple, str] = {}
    type_counts: defaultdict[str, int] = defaultdict(int)
    signature: list[str] = []
    for step in sequence:
        signature.append(f"operator:{step.action.operator.name}")
        for parameter_name, term in step.action.bindings:
            key = term.canonical_key()
            if key not in slots:
                index = type_counts[term.type.name]
                type_counts[term.type.name] += 1
                slots[key] = f"slot:{term.type.name}:{index}"
            signature.append(
                f"binding:{parameter_name}:{slots[key]}"
            )
        signature.extend(
            f"effect:{atom.predicate.name}:"
            + ",".join(term.type.name for term in atom.arguments)
            for atom in step.effects
        )
    return tuple(signature)


def operator_schema_fingerprint(operator: OperatorSpec) -> str:
    """Hash an alpha-normalized primitive schema, excluding ground symbol names."""

    def encode_term(term: Term):
        if isinstance(term, Variable):
            return ("variable", term.name, term.type.name)
        if isinstance(term, Symbol):
            return ("ground_symbol", term.type.name)
        if isinstance(term, TermApplication):
            function = term.function
            return (
                "application",
                function.name,
                tuple(item.name for item in function.input_types),
                function.output_type.name,
                function.symmetry_groups,
                function.allow_repeated_arguments,
                tuple(encode_term(argument) for argument in term.arguments),
            )
        raise TypeError(f"unsupported operator schema term: {type(term).__name__}")

    def encode_atom(atom: Atom):
        predicate = atom.predicate
        return (
            predicate.name,
            tuple(item.name for item in predicate.argument_types),
            predicate.symmetry_groups,
            predicate.verified,
            predicate.metadata,
            tuple(encode_term(argument) for argument in atom.arguments),
        )

    payload = (
        operator.name,
        operator.family,
        operator.tags,
        operator.metadata,
        tuple((item.name, item.type.name) for item in operator.parameters),
        tuple(encode_atom(atom) for atom in operator.preconditions),
        tuple(encode_atom(atom) for atom in operator.effects),
        operator.guards,
        operator.cost,
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _atom_type_signature(atom: Atom) -> str:
    return (
        f"{atom.predicate.name}("
        + ",".join(term.type.name for term in atom.arguments)
        + ")"
    )

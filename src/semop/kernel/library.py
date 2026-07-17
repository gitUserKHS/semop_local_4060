from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .model import ProofStep, SolveResult


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

    @property
    def retained(self) -> bool:
        return (
            self.support >= 3
            and self.type_effect_verified
            and self.reduction_ratio >= 0.10
            and self.heldout_no_regression
        )


class MdlMacroLibrary:
    """Bounded DreamCoder-style program compression over verified traces."""

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
                f"{effect.predicate.name}({','.join(term.type.name for term in effect.arguments)})"
                for effect in sample[-1].effects
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
                type_effect_verified=True,
                heldout_no_regression=False,
            )
            if reduction < min_reduction:
                continue
            candidates.append(
                MacroOperatorCandidate(
                    **{
                        **asdict(provisional),
                        "heldout_no_regression": bool(heldout_validator(provisional)),
                    }
                )
            )
        return cls(candidates)

    def expand(self, name: str) -> tuple[str, ...]:
        for record in self.records:
            if record.name == name:
                return record.operator_sequence
        raise KeyError(name)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "records": [asdict(record) for record in self.records],
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "MdlMacroLibrary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
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

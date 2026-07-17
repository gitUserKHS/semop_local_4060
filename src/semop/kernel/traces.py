from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

from .engine import OperatorKernel
from .model import Goal, GroundAction, SolveResult, WorldState


@dataclass(frozen=True)
class TraceActionRecord:
    operator: str
    bindings: tuple[tuple[str, str, str], ...]
    premises: tuple[str, ...]
    effects: tuple[str, ...]


@dataclass(frozen=True)
class HardNegativeRecord:
    step: int
    operator: str
    bindings: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class VerifiedTraceRecord:
    trace_id: str
    domain: str
    source: str
    reviewed: bool
    initial_facts: tuple[str, ...]
    goals: tuple[str, ...]
    actions: tuple[TraceActionRecord, ...]
    hard_negatives: tuple[HardNegativeRecord, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class DecisionTrainingCase:
    state: WorldState
    goals: tuple[Goal, ...]
    actions: tuple[GroundAction, ...]
    target_action: int
    step: int


class TraceCorpus:
    """Bounded verifier-produced traces for low-resource controller training."""

    VALID_SOURCES = {"verifier", "synthetic", "reviewed"}

    def __init__(
        self,
        *,
        max_synthetic_per_domain: int = 5_000,
        max_synthetic_depth: int = 6,
        hard_negatives_per_positive: int = 4,
    ) -> None:
        self.max_synthetic_per_domain = max_synthetic_per_domain
        self.max_synthetic_depth = max_synthetic_depth
        self.hard_negatives_per_positive = hard_negatives_per_positive
        self.records: list[VerifiedTraceRecord] = []

    def add_result(
        self,
        trace_id: str,
        domain: str,
        result: SolveResult,
        *,
        source: str = "verifier",
        hard_negatives: Sequence[HardNegativeRecord] = (),
        metadata: dict[str, str] | None = None,
    ) -> VerifiedTraceRecord:
        if source not in self.VALID_SOURCES:
            raise ValueError(f"unknown trace source: {source}")
        if not result.success or not result.verified:
            raise ValueError("only replay-verified successful traces may enter the corpus")
        if source == "synthetic":
            if len(result.proof) > self.max_synthetic_depth:
                raise ValueError(
                    f"synthetic proof depth exceeds {self.max_synthetic_depth}"
                )
            count = sum(
                record.domain == domain and record.source == "synthetic"
                for record in self.records
            )
            if count >= self.max_synthetic_per_domain:
                raise ValueError(f"synthetic trace cap reached for domain {domain}")
        limited_negatives = tuple(hard_negatives)[
            : len(result.proof) * self.hard_negatives_per_positive
        ]
        record = VerifiedTraceRecord(
            trace_id=trace_id,
            domain=domain,
            source=source,
            reviewed=source == "reviewed",
            initial_facts=tuple(str(fact) for fact in result.initial_state.facts),
            goals=tuple(str(outcome.goal) for outcome in result.goals),
            actions=tuple(
                TraceActionRecord(
                    operator=step.action.operator.name,
                    bindings=tuple(
                        (name, str(term), term.type.name)
                        for name, term in step.action.bindings
                    ),
                    premises=tuple(map(str, step.premises)),
                    effects=tuple(map(str, step.effects)),
                )
                for step in result.proof
            ),
            hard_negatives=limited_negatives,
            metadata=tuple(sorted((metadata or {}).items())),
        )
        if any(existing.trace_id == trace_id for existing in self.records):
            raise ValueError(f"duplicate trace id: {trace_id}")
        self.records.append(record)
        return record

    def shot_split(
        self, shots: int, *, seed: int = 0
    ) -> tuple[VerifiedTraceRecord, ...]:
        if shots not in {0, 5, 20, 100}:
            raise ValueError("shots must be one of 0, 5, 20, 100")
        by_domain: dict[str, list[VerifiedTraceRecord]] = {}
        for record in self.records:
            by_domain.setdefault(record.domain, []).append(record)
        rng = random.Random(seed)
        selected: list[VerifiedTraceRecord] = []
        for domain in sorted(by_domain):
            reviewed = [record for record in by_domain[domain] if record.reviewed]
            reviewed.sort(key=lambda record: record.trace_id)
            rng.shuffle(reviewed)
            selected.extend(reviewed[:shots])
            selected.extend(
                record for record in by_domain[domain] if not record.reviewed
            )
        return tuple(selected)

    def counts(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for record in self.records:
            domain = result.setdefault(
                record.domain, {"verifier": 0, "synthetic": 0, "reviewed": 0}
            )
            domain[record.source] += 1
        return result

    def save_jsonl(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return destination

    @classmethod
    def load_jsonl(
        cls,
        path: str | Path,
        *,
        max_synthetic_per_domain: int = 5_000,
        max_synthetic_depth: int = 6,
        hard_negatives_per_positive: int = 4,
    ) -> "TraceCorpus":
        """Restore the portable audit records used by a learning checkpoint."""

        corpus = cls(
            max_synthetic_per_domain=max_synthetic_per_domain,
            max_synthetic_depth=max_synthetic_depth,
            hard_negatives_per_positive=hard_negatives_per_positive,
        )
        identifiers: set[str] = set()
        for line_number, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                record = VerifiedTraceRecord(
                    trace_id=str(raw["trace_id"]),
                    domain=str(raw["domain"]),
                    source=str(raw["source"]),
                    reviewed=bool(raw["reviewed"]),
                    initial_facts=tuple(map(str, raw["initial_facts"])),
                    goals=tuple(map(str, raw["goals"])),
                    actions=tuple(
                        TraceActionRecord(
                            operator=str(action["operator"]),
                            bindings=tuple(
                                (str(name), str(term), str(type_name))
                                for name, term, type_name in action["bindings"]
                            ),
                            premises=tuple(map(str, action["premises"])),
                            effects=tuple(map(str, action["effects"])),
                        )
                        for action in raw["actions"]
                    ),
                    hard_negatives=tuple(
                        HardNegativeRecord(
                            step=int(negative["step"]),
                            operator=str(negative["operator"]),
                            bindings=tuple(
                                (str(name), str(term), str(type_name))
                                for name, term, type_name in negative["bindings"]
                            ),
                        )
                        for negative in raw.get("hard_negatives", ())
                    ),
                    metadata=tuple(
                        (str(name), str(value))
                        for name, value in raw.get("metadata", ())
                    ),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid trace JSONL record at line {line_number}"
                ) from exc
            if record.source not in cls.VALID_SOURCES:
                raise ValueError(
                    f"unknown trace source at line {line_number}: {record.source}"
                )
            if record.reviewed != (record.source == "reviewed"):
                raise ValueError(
                    f"review flag/source mismatch at line {line_number}"
                )
            if record.trace_id in identifiers:
                raise ValueError(
                    f"duplicate trace id at line {line_number}: {record.trace_id}"
                )
            identifiers.add(record.trace_id)
            corpus.records.append(record)
        return corpus


def build_decision_training_cases(
    kernel: OperatorKernel,
    result: SolveResult,
    *,
    negatives_per_positive: int = 4,
) -> tuple[DecisionTrainingCase, ...]:
    if not result.success or not result.verified:
        raise ValueError("decision cases require a verified successful result")
    state = result.initial_state
    goals = tuple(outcome.goal for outcome in result.goals)
    cases: list[DecisionTrainingCase] = []
    for proof_offset, step in enumerate(result.proof):
        available = kernel.enumerate_actions(state)
        policy_goals = kernel.policy_goals(state, goals, available)
        gold_key = step.action.canonical_key()
        gold = next(
            (action for action in available if action.canonical_key() == gold_key),
            None,
        )
        if gold is None:
            raise ValueError(f"gold action is not replay-applicable at step {step.index}")
        future_action_keys = {
            future.action.canonical_key()
            for future in result.proof[proof_offset + 1 :]
        }
        negatives = [
            action
            for action in available
            if action.canonical_key() != gold_key
            and action.canonical_key() not in future_action_keys
            and "hard_negative" in action.operator.tags
        ][:negatives_per_positive]
        candidates = tuple(sorted((gold, *negatives), key=GroundAction.canonical_key))
        target = next(
            index
            for index, action in enumerate(candidates)
            if action.canonical_key() == gold_key
        )
        cases.append(
            DecisionTrainingCase(
                state=state,
                goals=policy_goals,
                actions=candidates,
                target_action=target,
                step=step.index,
            )
        )
        state = kernel.execute_action(state, step.action)
    return tuple(cases)


def hard_negative_records(
    cases: Iterable[DecisionTrainingCase],
) -> tuple[HardNegativeRecord, ...]:
    records: list[HardNegativeRecord] = []
    for case in cases:
        gold = case.actions[case.target_action]
        for action in case.actions:
            if action == gold:
                continue
            records.append(
                HardNegativeRecord(
                    step=case.step,
                    operator=action.operator.name,
                    bindings=tuple(
                        (name, str(term), term.type.name)
                        for name, term in action.bindings
                    ),
                )
            )
    return tuple(records)

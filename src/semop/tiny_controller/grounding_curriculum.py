from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Iterable, Sequence

from semop.kernel.grounding import GroundingLabel, GroundingLearningExample

from .grounding_features import encode_grounding_candidate


@dataclass(frozen=True)
class DomainGroundingCurriculumAudit:
    domain: str
    available_examples: int
    selected_examples: int
    accept_examples: int
    reject_examples: int
    available_feature_states: int
    covered_feature_states: int
    feature_state_coverage: float
    target_feature_support: int
    support_eligible_feature_states: int
    supported_feature_states: int
    supported_feature_state_coverage: float


@dataclass(frozen=True)
class GroundingCurriculumAudit:
    available_examples: int
    selected_examples: int
    available_feature_states: int
    covered_feature_states: int
    feature_state_coverage: float
    target_feature_support: int
    support_eligible_feature_states: int
    supported_feature_states: int
    supported_feature_state_coverage: float
    by_domain: tuple[DomainGroundingCurriculumAudit, ...]


@dataclass(frozen=True)
class GroundingCurriculumSelection:
    examples: tuple[GroundingLearningExample, ...] = field(compare=False, repr=False)
    audit: GroundingCurriculumAudit
    selection_digest: str

    def __post_init__(self) -> None:
        if len(self.selection_digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.selection_digest
        ):
            raise ValueError("grounding curriculum selection digest must be SHA-256")


@dataclass(frozen=True)
class _CurriculumCandidate:
    example: GroundingLearningExample = field(repr=False)
    feature_states: frozenset[str]


def select_feature_novel_grounding_examples(
    examples: Sequence[GroundingLearningExample]
    | Iterable[GroundingLearningExample],
    *,
    per_domain: int,
    min_per_label: int = 1,
    target_feature_support: int = 2,
) -> GroundingCurriculumSelection:
    """Select examples that cover and sufficiently repeat label-free states."""

    pool = tuple(examples)
    if per_domain <= 0:
        raise ValueError("grounding curriculum per_domain must be positive")
    if min_per_label <= 0 or 2 * min_per_label > per_domain:
        raise ValueError("grounding curriculum label minimum is incompatible with budget")
    if target_feature_support <= 0:
        raise ValueError("grounding curriculum feature support target must be positive")
    if not pool:
        raise ValueError("grounding curriculum requires a non-empty verified pool")

    selected: list[GroundingLearningExample] = []
    domains = tuple(sorted({example.candidate.domain for example in pool}))
    for domain in domains:
        domain_pool = tuple(
            _CurriculumCandidate(example, _feature_states(example))
            for example in pool
            if example.candidate.domain == domain
        )
        if len(domain_pool) < per_domain:
            raise ValueError(f"grounding curriculum lacks {domain} examples")
        counts = Counter(item.example.label for item in domain_pool)
        for label in GroundingLabel:
            if counts[label] < min_per_label:
                raise ValueError(
                    f"grounding curriculum lacks {label.value} labels for {domain}"
                )
        selected.extend(
            _select_domain_candidates(
                domain_pool,
                budget=per_domain,
                min_per_label=min_per_label,
                target_feature_support=target_feature_support,
            )
        )

    ordered = tuple(
        sorted(
            selected,
            key=lambda example: (
                example.candidate.domain,
                example.candidate.candidate_digest,
                example.record_digest,
            ),
        )
    )
    digest = sha256(
        json.dumps(
            [example.record_digest for example in ordered],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return GroundingCurriculumSelection(
        ordered,
        audit_grounding_curriculum_selection(
            ordered,
            pool,
            target_feature_support=target_feature_support,
        ),
        digest,
    )


def audit_grounding_curriculum_selection(
    selected: Sequence[GroundingLearningExample],
    available: Sequence[GroundingLearningExample],
    *,
    target_feature_support: int = 2,
) -> GroundingCurriculumAudit:
    if target_feature_support <= 0:
        raise ValueError("grounding curriculum feature support target must be positive")
    selected_items = tuple(selected)
    available_items = tuple(available)
    available_records = {item.record_digest for item in available_items}
    if any(item.record_digest not in available_records for item in selected_items):
        raise ValueError("grounding curriculum selection is not a subset of its pool")
    if len({item.record_digest for item in selected_items}) != len(selected_items):
        raise ValueError("grounding curriculum selection contains duplicate records")

    domains = tuple(
        sorted({item.candidate.domain for item in available_items})
    )
    audits: list[DomainGroundingCurriculumAudit] = []
    all_available_states: set[str] = set()
    all_selected_states: set[str] = set()
    all_support_eligible_states: set[str] = set()
    all_supported_states: set[str] = set()
    for domain in domains:
        domain_available = tuple(
            item for item in available_items if item.candidate.domain == domain
        )
        domain_selected = tuple(
            item for item in selected_items if item.candidate.domain == domain
        )
        available_counts = Counter(
            state
            for item in domain_available
            for state in _feature_states(item)
        )
        selected_counts = Counter(
            state
            for item in domain_selected
            for state in _feature_states(item)
        )
        available_states = set(available_counts)
        selected_states = set(selected_counts)
        support_eligible_states = {
            state
            for state, count in available_counts.items()
            if count >= target_feature_support
        }
        supported_states = {
            state
            for state, count in selected_counts.items()
            if count >= target_feature_support
        } & support_eligible_states
        all_available_states.update(f"{domain}:{name}" for name in available_states)
        all_selected_states.update(f"{domain}:{name}" for name in selected_states)
        all_support_eligible_states.update(
            f"{domain}:{name}" for name in support_eligible_states
        )
        all_supported_states.update(
            f"{domain}:{name}" for name in supported_states
        )
        labels = Counter(item.label for item in domain_selected)
        audits.append(
            DomainGroundingCurriculumAudit(
                domain=domain,
                available_examples=len(domain_available),
                selected_examples=len(domain_selected),
                accept_examples=labels[GroundingLabel.ACCEPT],
                reject_examples=labels[GroundingLabel.REJECT],
                available_feature_states=len(available_states),
                covered_feature_states=len(selected_states),
                feature_state_coverage=(
                    len(selected_states) / len(available_states)
                    if available_states
                    else 0.0
                ),
                target_feature_support=target_feature_support,
                support_eligible_feature_states=len(support_eligible_states),
                supported_feature_states=len(supported_states),
                supported_feature_state_coverage=(
                    len(supported_states) / len(support_eligible_states)
                    if support_eligible_states
                    else 0.0
                ),
            )
        )
    return GroundingCurriculumAudit(
        available_examples=len(available_items),
        selected_examples=len(selected_items),
        available_feature_states=len(all_available_states),
        covered_feature_states=len(all_selected_states),
        feature_state_coverage=(
            len(all_selected_states) / len(all_available_states)
            if all_available_states
            else 0.0
        ),
        target_feature_support=target_feature_support,
        support_eligible_feature_states=len(all_support_eligible_states),
        supported_feature_states=len(all_supported_states),
        supported_feature_state_coverage=(
            len(all_supported_states) / len(all_support_eligible_states)
            if all_support_eligible_states
            else 0.0
        ),
        by_domain=tuple(audits),
    )


def _select_domain_candidates(
    pool: tuple[_CurriculumCandidate, ...],
    *,
    budget: int,
    min_per_label: int,
    target_feature_support: int,
) -> tuple[GroundingLearningExample, ...]:
    remaining = list(pool)
    selected: list[_CurriculumCandidate] = []
    covered: Counter[str] = Counter()
    frequencies = Counter(
        state for item in pool for state in item.feature_states
    )

    def choose(candidates: list[_CurriculumCandidate]) -> _CurriculumCandidate:
        if not candidates:
            raise ValueError("grounding curriculum cannot satisfy its label quota")

        def ranking(item: _CurriculumCandidate):
            novel = {
                state for state in item.feature_states if covered[state] == 0
            }
            support_gain = {
                state
                for state in item.feature_states
                if covered[state] < target_feature_support
            }
            reinforcement = support_gain - novel
            rarity = sum(1.0 / frequencies[state] for state in support_gain)
            return (
                -(2 * len(novel) + len(reinforcement)),
                -len(novel),
                -rarity,
                item.example.candidate.candidate_digest,
                item.example.record_digest,
            )

        return min(candidates, key=ranking)

    for _round in range(min_per_label):
        for label in GroundingLabel:
            chosen = choose(
                [item for item in remaining if item.example.label is label]
            )
            remaining.remove(chosen)
            selected.append(chosen)
            covered.update(chosen.feature_states)

    while len(selected) < budget:
        label_counts = Counter(item.example.label for item in selected)

        def ranking(item: _CurriculumCandidate):
            novel = {
                state for state in item.feature_states if covered[state] == 0
            }
            support_gain = {
                state
                for state in item.feature_states
                if covered[state] < target_feature_support
            }
            reinforcement = support_gain - novel
            rarity = sum(1.0 / frequencies[state] for state in support_gain)
            return (
                -(2 * len(novel) + len(reinforcement)),
                -len(novel),
                -rarity,
                label_counts[item.example.label],
                item.example.candidate.candidate_digest,
                item.example.record_digest,
            )

        chosen = min(remaining, key=ranking)
        remaining.remove(chosen)
        selected.append(chosen)
        covered.update(chosen.feature_states)
    return tuple(item.example for item in selected)


def _feature_states(example: GroundingLearningExample) -> frozenset[str]:
    vector = encode_grounding_candidate(example.candidate)
    return frozenset(
        name
        for name, value in vector.values
        if value
        and (
            name.startswith("sensor:shared:")
            or name.startswith("derived:interaction:")
            or name.startswith("derived:decision:")
        )
    )


__all__ = [
    "DomainGroundingCurriculumAudit",
    "GroundingCurriculumAudit",
    "GroundingCurriculumSelection",
    "audit_grounding_curriculum_selection",
    "select_feature_novel_grounding_examples",
]

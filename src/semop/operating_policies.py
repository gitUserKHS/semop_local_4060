from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DomainOperatingPolicy:
    domain: str
    label: str
    promotable_review_reasons: tuple[str, ...] = field(default_factory=tuple)
    blocked_review_reasons: tuple[str, ...] = field(default_factory=tuple)
    override_review_reasons: tuple[str, ...] = field(default_factory=tuple)
    require_answer_text: bool = True
    require_resolution_note_for_manual_review: bool = True
    minimum_unseen_transfer: float = 0.0
    minimum_analogy_usefulness: float = 0.0
    minimum_compiler_validity: float = 0.55
    minimum_grounded_explanation_fidelity: float = 0.5
    minimum_repair_success_rate: float = 0.5
    regression_tolerance: float = 0.03
    require_improvement_if_baseline: bool = True
    scenario_threshold_overrides: dict[str, dict[str, float]] = field(default_factory=dict)
    review_severity_weights: dict[str, float] = field(default_factory=dict)
    scenario_severity_weight_overrides: dict[str, dict[str, float]] = field(default_factory=dict)
    slice_balance_limit: int = 4
    scenario_slice_balance_limits: dict[str, int] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewPromotionDecision:
    item_id: int
    domain: str
    query: str
    promotable: bool
    matched_reasons: list[str] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    override_reasons: list[str] = field(default_factory=list)
    rationale: str = ''

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


_COMMON_PROMOTABLE_REASONS = (
    'approved_training_trace',
    'grounding_review',
    'claim_grounding_review',
    'clarification_need_rate',
    'low_plan_executability',
    'low_audit_usefulness',
    'manual_review_note',
    'compiler_validity_gap',
    'repair_failure',
    'context_misread_review',
    'relation_recovery_gap',
)

DEFAULT_REVIEW_SEVERITY_WEIGHTS: dict[str, float] = {
    'low': 0.75,
    'medium': 1.0,
    'high': 1.35,
    'critical': 1.7,
}

OPERATING_POLICIES: dict[str, DomainOperatingPolicy] = {
    'general': DomainOperatingPolicy(
        domain='general',
        label='General Reasoning Gate',
        promotable_review_reasons=_COMMON_PROMOTABLE_REASONS,
        blocked_review_reasons=('invalid_advice_rate',),
        override_review_reasons=('approved_training_trace',),
        minimum_unseen_transfer=0.05,
        minimum_analogy_usefulness=0.1,
        minimum_compiler_validity=0.6,
        minimum_grounded_explanation_fidelity=0.55,
        minimum_repair_success_rate=0.55,
        regression_tolerance=0.03,
        require_improvement_if_baseline=True,
        scenario_threshold_overrides={
            'qa': {
                'minimum_compiler_validity': 0.62,
                'minimum_grounded_explanation_fidelity': 0.56,
                'minimum_repair_success_rate': 0.55,
            },
        },
        review_severity_weights=DEFAULT_REVIEW_SEVERITY_WEIGHTS,
        slice_balance_limit=4,
    ),
    'warehouse_onboarding': DomainOperatingPolicy(
        domain='warehouse_onboarding',
        label='Warehouse Onboarding Gate',
        promotable_review_reasons=_COMMON_PROMOTABLE_REASONS,
        blocked_review_reasons=('invalid_advice_rate',),
        override_review_reasons=('approved_training_trace',),
        minimum_unseen_transfer=0.08,
        minimum_analogy_usefulness=0.12,
        minimum_compiler_validity=0.68,
        minimum_grounded_explanation_fidelity=0.74,
        minimum_repair_success_rate=0.62,
        regression_tolerance=0.025,
        require_improvement_if_baseline=True,
        scenario_threshold_overrides={
            'onboarding': {
                'minimum_compiler_validity': 0.7,
                'minimum_grounded_explanation_fidelity': 0.78,
                'minimum_repair_success_rate': 0.64,
            },
        },
        review_severity_weights={
            'low': 0.8,
            'medium': 1.0,
            'high': 1.45,
            'critical': 1.85,
        },
        scenario_severity_weight_overrides={
            'onboarding': {
                'critical': 1.95,
            },
        },
        slice_balance_limit=4,
        scenario_slice_balance_limits={
            'onboarding': 5,
        },
    ),
    'warehouse_exception': DomainOperatingPolicy(
        domain='warehouse_exception',
        label='Warehouse Exception Gate',
        promotable_review_reasons=tuple(reason for reason in _COMMON_PROMOTABLE_REASONS if reason != 'clarification_need_rate'),
        blocked_review_reasons=('invalid_advice_rate',),
        override_review_reasons=('approved_training_trace',),
        minimum_unseen_transfer=0.08,
        minimum_analogy_usefulness=0.1,
        minimum_compiler_validity=0.74,
        minimum_grounded_explanation_fidelity=0.68,
        minimum_repair_success_rate=0.74,
        regression_tolerance=0.02,
        require_improvement_if_baseline=True,
        scenario_threshold_overrides={
            'exception_response': {
                'minimum_compiler_validity': 0.8,
                'minimum_grounded_explanation_fidelity': 0.75,
                'minimum_repair_success_rate': 0.78,
            },
            'quality_gate': {
                'minimum_compiler_validity': 0.78,
                'minimum_grounded_explanation_fidelity': 0.72,
                'minimum_repair_success_rate': 0.76,
            },
        },
        review_severity_weights={
            'low': 0.9,
            'medium': 1.1,
            'high': 1.6,
            'critical': 1.95,
        },
        scenario_severity_weight_overrides={
            'exception_response': {
                'high': 1.75,
                'critical': 2.25,
            },
            'quality_gate': {
                'high': 1.7,
                'critical': 2.05,
            },
        },
        slice_balance_limit=4,
        scenario_slice_balance_limits={
            'exception_response': 6,
            'quality_gate': 5,
        },
    ),
}


def resolve_operating_policy(domain: str | None) -> DomainOperatingPolicy:
    key = str(domain or 'general').strip() or 'general'
    return OPERATING_POLICIES.get(key, OPERATING_POLICIES['general'])


def resolve_slice_thresholds(domain: str | None, scenario: str | None) -> dict[str, float]:
    policy = resolve_operating_policy(domain)
    metrics = {
        'minimum_compiler_validity': policy.minimum_compiler_validity,
        'minimum_grounded_explanation_fidelity': policy.minimum_grounded_explanation_fidelity,
        'minimum_repair_success_rate': policy.minimum_repair_success_rate,
    }
    scenario_key = str(scenario or '').strip()
    if scenario_key:
        metrics.update(policy.scenario_threshold_overrides.get(scenario_key, {}))
    return metrics


def resolve_review_severity_weight(domain: str | None, scenario: str | None, severity: str | None) -> float:
    policy = resolve_operating_policy(domain)
    resolved_severity = str(severity or 'medium').strip().lower() or 'medium'
    weights = dict(DEFAULT_REVIEW_SEVERITY_WEIGHTS)
    weights.update(policy.review_severity_weights)
    scenario_key = str(scenario or '').strip()
    if scenario_key:
        weights.update(policy.scenario_severity_weight_overrides.get(scenario_key, {}))
    return float(weights.get(resolved_severity, weights['medium']))


def resolve_slice_balance_limit(domain: str | None, scenario: str | None) -> int:
    policy = resolve_operating_policy(domain)
    scenario_key = str(scenario or '').strip()
    if scenario_key and scenario_key in policy.scenario_slice_balance_limits:
        return int(policy.scenario_slice_balance_limits[scenario_key])
    return int(policy.slice_balance_limit)


def evaluate_review_promotion(detail: dict[str, Any], domain: str | None = None) -> ReviewPromotionDecision:
    resolved_domain = str(domain or detail.get('domain') or 'general').strip() or 'general'
    policy = resolve_operating_policy(resolved_domain)
    reasons = sorted({str(item).strip() for item in detail.get('reasons', []) if str(item).strip()})
    reason_set = set(reasons)
    matched_reasons = sorted(reason_set & set(policy.promotable_review_reasons))
    blocked_reasons = sorted(reason_set & set(policy.blocked_review_reasons))
    override_reasons = sorted(reason_set & set(policy.override_review_reasons))
    answer_text = str(detail.get('answer_text', '')).strip()
    resolution_note = str(detail.get('resolution_note', '')).strip()
    status = str(detail.get('status', '')).strip().lower()

    promotable = False
    rationale = ''
    if status != 'approved':
        rationale = 'review is not approved'
    elif policy.require_answer_text and not answer_text:
        rationale = 'approved review has no corrected answer text'
    elif override_reasons:
        promotable = True
        rationale = 'explicit training-trace override reason present'
    elif blocked_reasons:
        rationale = 'blocked review reasons require explicit override before retraining'
    elif 'manual_review_note' in reason_set and policy.require_resolution_note_for_manual_review and not resolution_note:
        rationale = 'manual review note requires a resolution note before retraining'
    elif matched_reasons:
        promotable = True
        rationale = 'approved review reasons match promotable policy reasons'
    else:
        rationale = 'approved review reasons are outside the promotable policy set'

    return ReviewPromotionDecision(
        item_id=int(detail.get('id', 0) or 0),
        domain=resolved_domain,
        query=str(detail.get('query', '')),
        promotable=promotable,
        matched_reasons=matched_reasons,
        blocked_reasons=blocked_reasons,
        override_reasons=override_reasons,
        rationale=rationale,
    )


def collect_review_promotion_decisions(
    review_store_path: str | Path,
    domain: str | None = None,
    limit: int = 500,
) -> list[tuple[dict[str, Any], ReviewPromotionDecision]]:
    from .review_queue import ReviewQueueStore

    store = ReviewQueueStore(review_store_path)
    approved = store.fetch_items(status='approved', limit=limit)
    decisions: list[tuple[dict[str, Any], ReviewPromotionDecision]] = []
    for item in approved:
        detail = store.fetch_item_detail(item.id)
        if detail is None:
            continue
        item_domain = str(detail.get('domain') or item.domain or 'general').strip() or 'general'
        if domain and item_domain != domain:
            continue
        detail['status'] = str(detail.get('status') or item.status or 'approved')
        decision = evaluate_review_promotion(detail, domain=item_domain)
        decisions.append((detail, decision))
    return decisions


def promoted_review_details(
    review_store_path: str | Path,
    domain: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    return [detail for detail, decision in collect_review_promotion_decisions(review_store_path, domain=domain, limit=limit) if decision.promotable]


def infer_operating_domain(review_store_path: str | Path | None) -> str:
    if not review_store_path:
        return 'general'
    counts: dict[str, int] = {}
    for detail, decision in collect_review_promotion_decisions(review_store_path):
        if not decision.promotable:
            continue
        counts[decision.domain] = counts.get(decision.domain, 0) + 1
    if not counts:
        return 'general'
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

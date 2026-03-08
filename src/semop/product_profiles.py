from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class DomainProfile:
    name: str
    label: str
    description: str
    term_to_concept: Dict[str, str] = field(default_factory=dict)
    clarification_triggers: List[str] = field(default_factory=list)
    risky_advice_markers: List[str] = field(default_factory=list)


DOMAIN_PROFILES: Dict[str, DomainProfile] = {
    "general": DomainProfile(
        name="general",
        label="General Reasoning",
        description="Default profile for generic structured reasoning.",
        clarification_triggers=["이거", "그거", "this one", "that one"],
    ),
    "warehouse_onboarding": DomainProfile(
        name="warehouse_onboarding",
        label="Warehouse Onboarding",
        description="Focus on SOP explanation, task sequencing, and quality gates for new workers.",
        term_to_concept={
            "바코드": "barcode",
            "barcode": "barcode",
            "라벨": "order_label",
            "label": "order_label",
            "피킹": "pick_ticket",
            "pick": "pick_ticket",
            "포장": "package",
            "pack": "package",
            "승인": "supervisor_approval",
            "approval": "supervisor_approval",
        },
        clarification_triggers=["신입", "처음", "onboarding", "교육", "manual", "sop"],
        risky_advice_markers=["그냥 출고", "바로 포장", "skip scan", "ignore mismatch"],
    ),
    "warehouse_exception": DomainProfile(
        name="warehouse_exception",
        label="Warehouse Exception Response",
        description="Focus on blockers, safety stops, escalation, and temporary rerouting in warehouse operations.",
        term_to_concept={
            "지게차": "forklift",
            "forklift": "forklift",
            "팔레트": "pallet",
            "pallet": "pallet",
            "통로": "blocked_aisle",
            "aisle": "blocked_aisle",
            "승인": "supervisor_approval",
            "approval": "supervisor_approval",
            "안전": "safety_clearance",
            "safety": "safety_clearance",
            "스테이징": "staging_area",
            "staging": "staging_area",
            "보고": "incident_report",
            "report": "incident_report",
        },
        clarification_triggers=["긴급", "urgent", "예외", "blocked", "막힘"],
        risky_advice_markers=["그냥 이동", "승인 없이", "force through", "ignore blockage"],
    ),
}


def resolve_domain_profile(name: str) -> DomainProfile:
    return DOMAIN_PROFILES.get(name, DOMAIN_PROFILES["general"])

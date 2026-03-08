from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


CONCEPT_ALIASES = {
    "가방": "bag",
    "백팩": "bag",
    "bag": "bag",
    "책": "book",
    "book": "book",
    "지퍼": "zipper",
    "zip": "zipper",
    "zipper": "zipper",
    "수납공간": "compartment",
    "칸": "compartment",
    "compartment": "compartment",
    "세차장": "car_wash",
    "car wash": "car_wash",
    "세차": "car_wash",
    "차": "car",
    "자동차": "car",
    "차량": "car",
    "car": "car",
    "교통체증": "traffic",
    "교통 정체": "traffic",
    "막힘": "traffic",
    "정체": "traffic",
    "traffic": "traffic",
    "출장 세차": "mobile_detailer",
    "방문 세차": "mobile_detailer",
    "mobile detailer": "mobile_detailer",
    "가까운 세차장": "nearby_car_wash",
    "근처 세차장": "nearby_car_wash",
    "우회 경로": "detour_route",
    "출발 지연": "delay_departure",
    "창고": "warehouse",
    "warehouse": "warehouse",
    "지게차": "forklift",
    "forklift": "forklift",
    "팔레트": "pallet",
    "pallet": "pallet",
    "랙": "rack",
    "선반": "rack",
    "rack": "rack",
    "통로": "aisle",
    "aisle": "aisle",
    "막힌 통로": "blocked_aisle",
    "blocked aisle": "blocked_aisle",
    "승인": "supervisor_approval",
    "approval": "supervisor_approval",
    "관리자 승인": "supervisor_approval",
    "안전 확인": "safety_clearance",
    "안전 점검": "safety_clearance",
    "safety clearance": "safety_clearance",
    "지게차 자격": "forklift_certification",
    "forklift certification": "forklift_certification",
    "스테이징": "staging_area",
    "staging area": "staging_area",
    "사고 보고": "incident_report",
    "incident report": "incident_report",
    "작업 중지": "stop_work",
    "stop work": "stop_work",
    "바코드": "barcode",
    "barcode": "barcode",
    "라벨": "order_label",
    "label": "order_label",
    "불일치": "label_mismatch",
    "라벨 불일치": "label_mismatch",
    "mismatch": "label_mismatch",
    "재스캔": "rescan_item",
    "rescan": "rescan_item",
    "출고 보류": "hold_shipment",
    "hold shipment": "hold_shipment",
    "피킹 티켓": "pick_ticket",
    "pick ticket": "pick_ticket",
    "포장": "package",
    "package": "package",
    "검증된 라벨 일치": "verified_label_match",
    "verified label match": "verified_label_match",
    "worker": "worker",
}


COMMONSENSE_KB: Dict[str, Dict[str, Any]] = {
    "bag": {
        "label": "가방",
        "kind": "container",
        "parts": ["zipper", "compartment"],
        "contains": ["compartment"],
        "affordances": ["open", "close", "contain", "carry"],
        "requires": ["open_access"],
        "scripts": ["open_bag", "insert_item", "close_bag"],
        "attributes": {"portable": True, "has_closure": True},
    },
    "zipper": {
        "label": "지퍼",
        "kind": "closure",
        "affordances": ["open", "close"],
        "attributes": {"controls_access": "compartment"},
    },
    "compartment": {
        "label": "수납공간",
        "kind": "space",
        "affordances": ["contain", "store"],
        "attributes": {"inside": "bag"},
    },
    "book": {
        "label": "책",
        "kind": "object",
        "affordances": ["insert", "remove", "carry", "read"],
        "attributes": {"solid": True},
    },
    "car_wash": {
        "label": "세차장",
        "kind": "service_place",
        "affordances": ["clean_vehicle", "wait", "pay"],
        "requires": ["vehicle_present"],
        "alternatives": ["nearby_car_wash", "mobile_detailer", "delay_departure"],
        "scripts": ["drive_vehicle_to_site", "queue", "wash_vehicle", "exit_site"],
        "attributes": {"walk_only_usage_unusual": True, "accessible_by": "vehicle"},
    },
    "nearby_car_wash": {
        "label": "가까운 세차장",
        "kind": "service_place",
        "affordances": ["clean_vehicle"],
        "requires": ["vehicle_present"],
        "scripts": ["drive_vehicle_to_site", "wash_vehicle"],
        "attributes": {"distance": "low"},
    },
    "mobile_detailer": {
        "label": "출장 세차",
        "kind": "service",
        "affordances": ["travel_to_customer", "clean_vehicle"],
        "scripts": ["book_service", "provider_arrives", "wash_vehicle"],
        "attributes": {"reduces_user_travel": True},
    },
    "delay_departure": {
        "label": "출발 지연",
        "kind": "time_strategy",
        "affordances": ["wait", "reassess"],
    },
    "detour_route": {
        "label": "우회 경로",
        "kind": "route_strategy",
        "affordances": ["avoid_traffic"],
    },
    "car": {
        "label": "차량",
        "kind": "vehicle",
        "affordances": ["drive", "park", "wash", "move_to"],
        "attributes": {"road_vehicle": True},
    },
    "traffic": {
        "label": "교통체증",
        "kind": "obstacle",
        "affordances": ["delay", "block_route"],
        "attributes": {"affects": ["drive", "travel_time"]},
    },
    "warehouse": {
        "label": "창고",
        "kind": "facility",
        "affordances": ["store_goods", "pick_goods", "stage_goods"],
        "scripts": ["receive_goods", "pick_goods", "pack_goods", "ship_goods"],
    },
    "worker": {
        "label": "작업자",
        "kind": "person",
        "affordances": ["scan", "pick", "pack", "report_issue"],
    },
    "forklift": {
        "label": "지게차",
        "kind": "vehicle",
        "affordances": ["lift_pallet", "move_pallet", "stage_pallet"],
        "requires": ["forklift_certification", "safety_clearance"],
        "scripts": ["inspect_forklift", "move_pallet", "park_forklift"],
    },
    "pallet": {
        "label": "팔레트",
        "kind": "container",
        "affordances": ["store_boxes", "move_to_rack", "stage"],
        "scripts": ["wrap_pallet", "move_pallet", "place_pallet"],
    },
    "rack": {
        "label": "랙",
        "kind": "storage_structure",
        "affordances": ["store_pallet", "reserve_slot"],
    },
    "aisle": {
        "label": "통로",
        "kind": "path",
        "affordances": ["allow_travel", "allow_forklift_access"],
    },
    "blocked_aisle": {
        "label": "막힌 통로",
        "kind": "obstacle",
        "affordances": ["block_forklift_access"],
        "attributes": {"severity": "high"},
    },
    "supervisor_approval": {
        "label": "관리자 승인",
        "kind": "precondition",
        "affordances": ["authorize", "release_hold"],
    },
    "safety_clearance": {
        "label": "안전 확인",
        "kind": "precondition",
        "affordances": ["confirm_safe_route", "confirm_safe_handling"],
    },
    "forklift_certification": {
        "label": "지게차 자격",
        "kind": "precondition",
        "affordances": ["allow_forklift_operation"],
    },
    "staging_area": {
        "label": "스테이징 구역",
        "kind": "holding_area",
        "affordances": ["hold_goods", "resequence_work"],
    },
    "incident_report": {
        "label": "사고 보고",
        "kind": "workflow",
        "affordances": ["record_issue", "notify_supervisor"],
    },
    "stop_work": {
        "label": "작업 중지",
        "kind": "workflow",
        "affordances": ["pause_operation", "protect_workers"],
    },
    "barcode": {
        "label": "바코드",
        "kind": "identifier",
        "affordances": ["scan", "verify_item"],
    },
    "order_label": {
        "label": "주문 라벨",
        "kind": "identifier",
        "affordances": ["verify_destination", "verify_item"],
    },
    "label_mismatch": {
        "label": "라벨 불일치",
        "kind": "quality_issue",
        "affordances": ["block_shipment", "trigger_rescan"],
        "attributes": {"severity": "high"},
    },
    "verified_label_match": {
        "label": "검증된 라벨 일치",
        "kind": "state",
    },
    "rescan_item": {
        "label": "재스캔",
        "kind": "workflow",
        "affordances": ["scan", "verify_item"],
    },
    "hold_shipment": {
        "label": "출고 보류",
        "kind": "workflow",
        "affordances": ["hold_goods", "prevent_shipping"],
    },
    "pick_ticket": {
        "label": "피킹 티켓",
        "kind": "document",
        "affordances": ["verify_pick", "check_quantity"],
    },
    "package": {
        "label": "포장",
        "kind": "workflow",
        "affordances": ["pack_goods", "seal_box"],
    },
    "open_access": {
        "label": "열린 접근 상태",
        "kind": "state",
    },
    "available_space": {
        "label": "빈 공간",
        "kind": "state",
    },
    "vehicle_present": {
        "label": "차량 동반",
        "kind": "precondition",
    },
}


def normalize_concept(text: str) -> str:
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    lowered = lowered.strip(".,!?\"'()[]{}")
    return CONCEPT_ALIASES.get(lowered, lowered.replace(" ", "_"))


def lookup_concept(text: str) -> Dict[str, Any]:
    key = normalize_concept(text)
    return COMMONSENSE_KB.get(
        key,
        {
            "label": key.replace("_", " "),
            "kind": "concept",
            "affordances": [],
            "parts": [],
            "contains": [],
            "requires": [],
            "alternatives": [],
            "scripts": [],
            "attributes": {},
        },
    )


def concept_label(concept_id: str) -> str:
    return lookup_concept(concept_id).get("label", concept_id.replace("_", " "))


def kb_relations_for_concept(concept_id: str) -> Iterable[tuple[str, str, str, List[str]]]:
    data = lookup_concept(concept_id)
    for part in data.get("parts", []):
        yield part, "PART_OF", concept_id, [f"kb:{concept_id}:parts"]
    for target in data.get("contains", []):
        yield concept_id, "CONTAINS", target, [f"kb:{concept_id}:contains"]
    for affordance in data.get("affordances", []):
        yield concept_id, "AFFORDS", affordance, [f"kb:{concept_id}:affordances"]
    for requirement in data.get("requires", []):
        yield concept_id, "REQUIRES", requirement, [f"kb:{concept_id}:requires"]
    for alternative in data.get("alternatives", []):
        yield concept_id, "ALTERNATIVE", alternative, [f"kb:{concept_id}:alternatives"]


def scripts_for_concept(concept_id: str) -> List[str]:
    return list(lookup_concept(concept_id).get("scripts", []))


def match_concepts(query: str) -> List[str]:
    lowered = query.lower()
    matches: List[str] = []
    for alias, concept_id in sorted(CONCEPT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in lowered and concept_id not in matches:
            matches.append(concept_id)
    return matches

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DomainTemplate:
    trigger_tokens: tuple[str, ...]
    tags: tuple[str, ...]
    variants: tuple[str, ...]


DOMAIN_TEMPLATES: tuple[DomainTemplate, ...] = (
    DomainTemplate(
        trigger_tokens=("가방", "책"),
        tags=("container_access",),
        variants=(
            "가방에 책을 넣기 전에 무엇부터 확인해야 하나요?",
            "가방이 꽉 차 있을 때 책을 넣으려면 먼저 어떤 조건을 봐야 하나요?",
            "책을 가방 안에 넣을 때 필요한 선행 조건은 무엇인가요?",
            "책이 잘 들어가게 하려면 가방 구조에서 무엇을 먼저 봐야 하나요?",
        ),
    ),
    DomainTemplate(
        trigger_tokens=("세차장",),
        tags=("service_place", "travel_constraint"),
        variants=(
            "교통이 막히고 세차장이 멀면 가장 현실적인 대안은 무엇인가요?",
            "세차장까지 이동이 비효율적이면 어떤 다른 선택지가 있나요?",
            "길이 막힌 상황에서 세차 목표를 유지하려면 어떤 수단을 바꿔야 하나요?",
            "세차가 급하지 않다면 이동 비용을 줄이는 판단 기준은 무엇인가요?",
        ),
    ),
    DomainTemplate(
        trigger_tokens=("주차장",),
        tags=("service_place", "travel_constraint"),
        variants=(
            "주차장이 멀고 막힐 때 지금 이동하는 게 합리적인지 어떻게 판단하나요?",
            "주차장 접근이 어려우면 우회 계획이나 대안을 어떻게 고르나요?",
        ),
    ),
    DomainTemplate(
        trigger_tokens=("주유소",),
        tags=("service_place", "travel_constraint"),
        variants=(
            "주유소가 멀고 길이 막힐 때 급유 목표를 유지하려면 어떤 다른 수단을 찾아야 하나요?",
            "주유소까지 이동이 비효율적일 때 어떤 전제조건을 먼저 확인해야 하나요?",
        ),
    ),
    DomainTemplate(
        trigger_tokens=("옷장",),
        tags=("container_access",),
        variants=(
            "옷장이 꽉 차 있을 때 물건을 넣으려면 어떤 선행 점검이 필요한가요?",
            "옷장 접근이 막혀 있으면 어떤 대안이나 우회 방법을 생각할 수 있나요?",
        ),
    ),
    DomainTemplate(
        trigger_tokens=("서랍",),
        tags=("container_access",),
        variants=(
            "서랍 안에 파일을 넣기 전에 확인해야 할 조건은 무엇인가요?",
            "서랍 문이 잘 안 열릴 때 파일을 넣는 대안은 무엇인가요?",
        ),
    ),
)


def infer_tags(query: str) -> List[str]:
    tags: List[str] = []
    lowered = query.lower()
    if any(token in lowered for token in ["가방", "서랍", "옷장", "책"]):
        tags.append("container_access")
    if any(token in lowered for token in ["세차장", "주차장", "주유소"]):
        tags.append("service_place")
    if any(token in lowered for token in ["막히", "정체", "멀", "traffic"]):
        tags.append("travel_constraint")
    if any(token in lowered for token in ["대안", "우회", "선택지", "alternative"]):
        tags.append("alternative_search")
    return tags


def generate_domain_variants(query: str) -> List[str]:
    lowered = query.lower()
    variants: List[str] = []
    for template in DOMAIN_TEMPLATES:
        if all(token in lowered for token in template.trigger_tokens):
            variants.extend(template.variants)
    return list(dict.fromkeys(variants))

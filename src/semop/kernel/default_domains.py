from __future__ import annotations

from typing import Any, Mapping

from .contracts import TypedDomainAdapter
from .domain_catalog import DomainCatalog, DomainKind, DomainSpec
from .domains.coding import CodingInputAdapter
from .domains.composed_scene import SceneThresholdAdapter
from .domains.language_text import LanguageInputAdapter
from .domains.linear_equation import MathInputAdapter
from .domains.raster_vision import VisionInputAdapter
from .semantic_codec import (
    CODING_SEMANTIC_CODEC,
    LANGUAGE_SEMANTIC_CODEC,
    MATH_SEMANTIC_CODEC,
    VISION_SEMANTIC_CODEC,
)


_SHARED_CAPABILITIES = frozenset({"typed_grounding", "proof_replay"})


def create_default_domain_catalog(
    adapters: Mapping[DomainKind | str, TypedDomainAdapter[Any]] | None = None,
) -> DomainCatalog:
    """Build the canonical coding/language/math/vision/composed boundary."""

    catalog = DomainCatalog(
        (
            DomainSpec(
                kind=DomainKind.CODING,
                adapter=CodingInputAdapter(),
                semantic_codec=CODING_SEMANTIC_CODEC,
                input_contract=(
                    "English or Korean competitive-programming statement requesting "
                    "a C++17 solution"
                ),
                description_ko=(
                    "문제를 알고리즘 템플릿으로 풀고 컴파일과 등록된 실행 테스트로 검증해."
                ),
                capabilities=_SHARED_CAPABILITIES
                | {"code_generation", "compiler_feedback", "execution_validation"},
            ),
            DomainSpec(
                kind=DomainKind.LANGUAGE,
                adapter=LanguageInputAdapter(),
                semantic_codec=LANGUAGE_SEMANTIC_CODEC,
                input_contract=(
                    "controlled requirement text, typed Horn logic, or a legacy "
                    "StructuredMeaningGraph"
                ),
                description_ko=(
                    "명시적 문장이나 논리 규칙을 typed 전제와 목표로 바꾼다."
                ),
                capabilities=_SHARED_CAPABILITIES
                | {"controlled_language", "symbolic_logic"},
            ),
            DomainSpec(
                kind=DomainKind.MATH,
                adapter=MathInputAdapter(),
                semantic_codec=MATH_SEMANTIC_CODEC,
                input_contract=(
                    "exact arithmetic expression, numeric comparison, or one-variable "
                    "linear or quadratic real equation"
                ),
                description_ko=(
                    "정확한 수식과 일차방정식을 typed 계산 프로그램으로 바꾼다."
                ),
                capabilities=_SHARED_CAPABILITIES
                | {"exact_arithmetic", "linear_equation", "quadratic_real_equation"},
            ),
            DomainSpec(
                kind=DomainKind.VISION,
                adapter=VisionInputAdapter(),
                semantic_codec=VISION_SEMANTIC_CODEC,
                input_contract=(
                    "verified symbolic scene or deterministic small RGB raster with "
                    "explicit goals"
                ),
                description_ko=(
                    "객체·관계 장면이나 작은 RGB 격자를 typed 시각 사실로 바꾼다."
                ),
                capabilities=_SHARED_CAPABILITIES
                | {"deterministic_raster", "object_centric"},
            ),
            DomainSpec(
                kind=DomainKind.COMPOSED,
                adapter=SceneThresholdAdapter(),
                input_contract=(
                    "verified vision-to-math-to-language scene threshold problem"
                ),
                description_ko=(
                    "여러 도메인의 검증된 사실을 하나의 operator program으로 잇는다."
                ),
                capabilities=_SHARED_CAPABILITIES | {"cross_domain_composition"},
            ),
        )
    )
    if adapters:
        for kind, adapter in adapters.items():
            catalog = catalog.with_adapter(kind, adapter)
    return catalog

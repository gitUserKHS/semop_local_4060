from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .domain_catalog import DomainCatalog, DomainKind, TypedDomainRequest
from .domains.language_text import LanguageTextProblem
from .domains.raster_vision import (
    RasterImage,
    RasterVisionProblem,
    VisionPropertyGoal,
)
from .engine import OperatorKernel
from .runtime import UnifiedTypedReasoner
from .semantic_codec import (
    canonical_json,
    decode_semantic_request,
    encode_semantic_request,
    semantic_request_digest,
)


LMV_CORE_GATE_SCHEMA_VERSION = 1
LMV_DOMAINS = (
    DomainKind.LANGUAGE,
    DomainKind.MATH,
    DomainKind.VISION,
)
LMV_COMMON_CAPABILITIES = frozenset({"typed_grounding", "proof_replay", "semantic_codec"})


@dataclass(frozen=True)
class LMVGateFixture:
    domain: DomainKind
    positive: TypedDomainRequest
    negative: TypedDomainRequest


@dataclass(frozen=True)
class LMVDomainGateResult:
    domain: DomainKind
    positive_verified: bool
    negative_fail_closed: bool
    proof_replay_verified: bool
    semantic_roundtrip_stable: bool
    semantic_digest_stable: bool
    catalog_contract_valid: bool
    expansions: int
    elapsed_seconds: float
    capabilities: tuple[str, ...]
    diagnostics: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return all(
            (
                self.positive_verified,
                self.negative_fail_closed,
                self.proof_replay_verified,
                self.semantic_roundtrip_stable,
                self.semantic_digest_stable,
                self.catalog_contract_valid,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "passed": self.passed,
            "positive_verified": self.positive_verified,
            "negative_fail_closed": self.negative_fail_closed,
            "proof_replay_verified": self.proof_replay_verified,
            "semantic_roundtrip_stable": self.semantic_roundtrip_stable,
            "semantic_digest_stable": self.semantic_digest_stable,
            "catalog_contract_valid": self.catalog_contract_valid,
            "expansions": self.expansions,
            "elapsed_seconds": self.elapsed_seconds,
            "capabilities": list(self.capabilities),
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class LMVCoreGateReport:
    results: tuple[LMVDomainGateResult, ...]
    schema_version: int = LMV_CORE_GATE_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return (
            tuple(result.domain for result in self.results) == LMV_DOMAINS
            and all(result.passed for result in self.results)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gate": "lmv_core_contract",
            "passed": self.passed,
            "claim_scope": (
                "controlled language, exact math, and deterministic raster fixtures; "
                "not open-domain semantic understanding"
            ),
            "results": [result.to_dict() for result in self.results],
        }


def evaluate_lmv_core_gate(
    reasoner: UnifiedTypedReasoner | None = None,
    *,
    catalog: DomainCatalog | None = None,
) -> LMVCoreGateReport:
    """Run one fail-closed contract gate across language, math, and vision."""

    if reasoner is not None and catalog is not None:
        raise ValueError("pass either reasoner or catalog, not both")
    active = reasoner or UnifiedTypedReasoner(catalog=catalog)
    fixtures = _lmv_gate_fixtures()
    results = tuple(_evaluate_fixture(active, fixture) for fixture in fixtures)
    return LMVCoreGateReport(results)


def _evaluate_fixture(
    reasoner: UnifiedTypedReasoner,
    fixture: LMVGateFixture,
) -> LMVDomainGateResult:
    spec = reasoner.catalog.require(fixture.domain)
    diagnostics: list[str] = []
    started = perf_counter()

    encoded = encode_semantic_request(fixture.positive, catalog=reasoner.catalog)
    decoded = decode_semantic_request(
        fixture.domain,
        encoded["payload"],
        mode=fixture.positive.mode,
        catalog=reasoner.catalog,
    )
    reencoded = encode_semantic_request(decoded, catalog=reasoner.catalog)
    roundtrip_stable = canonical_json(encoded) == canonical_json(reencoded)
    digest_stable = semantic_request_digest(
        fixture.positive,
        catalog=reasoner.catalog,
    ) == semantic_request_digest(decoded, catalog=reasoner.catalog)

    positive = reasoner.run(decoded)
    negative = reasoner.run(fixture.negative)
    positive_verified = positive.success and positive.verified
    negative_fail_closed = not negative.success and not negative.verified

    replay_verified = False
    expansions = 0
    if positive.typed_result is None or positive.instance is None:
        diagnostics.append("positive fixture produced no typed proof")
    else:
        expansions = positive.typed_result.expansions
        replay = OperatorKernel(positive.instance.registry).replay(
            positive.typed_result.initial_state,
            positive.instance.goals,
            positive.typed_result.proof,
        )
        replay_verified = replay.verified
        diagnostics.extend(replay.diagnostics)

    if not negative_fail_closed:
        diagnostics.append("near-miss negative was not rejected")
    required = LMV_COMMON_CAPABILITIES
    catalog_contract_valid = (
        required.issubset(spec.capabilities)
        and tuple(sorted(spec.capabilities)) == positive.domain_capabilities
        and positive.input_contract == spec.input_contract
    )
    if not catalog_contract_valid:
        diagnostics.append("runtime result did not preserve the domain catalog contract")

    return LMVDomainGateResult(
        domain=fixture.domain,
        positive_verified=positive_verified,
        negative_fail_closed=negative_fail_closed,
        proof_replay_verified=replay_verified,
        semantic_roundtrip_stable=roundtrip_stable,
        semantic_digest_stable=digest_stable,
        catalog_contract_valid=catalog_contract_valid,
        expansions=expansions,
        elapsed_seconds=perf_counter() - started,
        capabilities=tuple(sorted(spec.capabilities)),
        diagnostics=tuple(diagnostics),
    )


def _lmv_gate_fixtures() -> tuple[LMVGateFixture, ...]:
    return (
        LMVGateFixture(
            domain=DomainKind.LANGUAGE,
            positive=TypedDomainRequest(
                DomainKind.LANGUAGE,
                LanguageTextProblem(
                    "Goal: deploy\nRequires: tests, approval\n"
                    "Satisfied: tests\nSatisfied: approval",
                    use_legacy_heuristics=False,
                ),
            ),
            negative=TypedDomainRequest(
                DomainKind.LANGUAGE,
                LanguageTextProblem(
                    "Goal: deploy\nRequires: tests, approval\nSatisfied: tests",
                    use_legacy_heuristics=False,
                ),
            ),
        ),
        LMVGateFixture(
            domain=DomainKind.MATH,
            positive=TypedDomainRequest(DomainKind.MATH, "(2 + 3) * 4 == 20"),
            negative=TypedDomainRequest(DomainKind.MATH, "(2 + 3) * 4 == 21"),
        ),
        LMVGateFixture(
            domain=DomainKind.VISION,
            positive=TypedDomainRequest(
                DomainKind.VISION,
                _shape_problem(square=True),
            ),
            negative=TypedDomainRequest(
                DomainKind.VISION,
                _shape_problem(square=False),
            ),
        ),
    )


def _shape_problem(*, square: bool) -> RasterVisionProblem:
    white = (255, 255, 255)
    red = (255, 0, 0)
    shape_row = (white, red, red, white) if square else (white, red, red, red, white)
    blank_row = (white,) * len(shape_row)
    image = RasterImage.from_rows(
        (blank_row, shape_row, shape_row, blank_row),
        background=white,
        source="lmv-core-gate-square" if square else "lmv-core-gate-rectangle",
    )
    return RasterVisionProblem(
        image=image,
        goals=(VisionPropertyGoal("SQUARE", "red", "red object is square"),),
        query="Is the red object square?",
    )

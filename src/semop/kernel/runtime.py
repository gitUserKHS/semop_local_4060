from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .adapters import MigrationMode
from .contracts import DomainInstance, TypedDomainAdapter
from .domains.language_text import LanguageInputAdapter
from .domains.linear_equation import MathInputAdapter
from .domains.composed_scene import SceneThresholdAdapter
from .domains.raster_vision import VisionInputAdapter
from .engine import ActionPolicy, OperatorKernel
from .model import SolveBudget, SolveResult
from .proof import render_proof_ko


class DomainKind(str, Enum):
    LANGUAGE = "language"
    MATH = "math"
    VISION = "vision"
    COMPOSED = "composed"


@dataclass(frozen=True)
class TypedDomainRequest:
    domain: DomainKind | str
    payload: Any
    mode: MigrationMode | str = MigrationMode.SHADOW


@dataclass(frozen=True)
class UnifiedTypedResult:
    domain: DomainKind
    mode: MigrationMode
    instance: DomainInstance | None
    typed_result: SolveResult | None
    proof_ko: str = ""
    projection_applied: bool = False
    diagnostics: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return bool(self.typed_result and self.typed_result.success)

    @property
    def verified(self) -> bool:
        return bool(self.typed_result and self.typed_result.verified)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "mode": self.mode.value,
            "success": self.success,
            "verified": self.verified,
            "projection_applied": self.projection_applied,
            "metadata": dict(self.instance.metadata) if self.instance else {},
            "typed_result": (
                self.typed_result.to_dict() if self.typed_result is not None else None
            ),
            "proof_ko": self.proof_ko,
            "diagnostics": list(self.diagnostics),
        }


class UnifiedTypedReasoner:
    """One verifier-first execution path for language, math, and vision."""

    def __init__(
        self,
        adapters: Mapping[DomainKind | str, TypedDomainAdapter[Any]] | None = None,
    ) -> None:
        defaults: dict[DomainKind, TypedDomainAdapter[Any]] = {
            DomainKind.LANGUAGE: LanguageInputAdapter(),
            DomainKind.MATH: MathInputAdapter(),
            DomainKind.VISION: VisionInputAdapter(),
            DomainKind.COMPOSED: SceneThresholdAdapter(),
        }
        if adapters:
            for key, adapter in adapters.items():
                defaults[DomainKind(key)] = adapter
        self.adapters = defaults

    def run(
        self,
        request: TypedDomainRequest,
        *,
        policy: ActionPolicy | None = None,
        budget: SolveBudget | None = None,
    ) -> UnifiedTypedResult:
        domain = DomainKind(request.domain)
        mode = MigrationMode(request.mode)
        if mode is MigrationMode.LEGACY:
            return UnifiedTypedResult(
                domain=domain,
                mode=mode,
                instance=None,
                typed_result=None,
                diagnostics=("legacy mode: typed kernel was not executed",),
            )

        adapter = self.adapters[domain]
        prebuilt_instance = isinstance(request.payload, DomainInstance)
        instance = request.payload if prebuilt_instance else adapter.adapt(request.payload)
        if not instance.goals:
            return UnifiedTypedResult(
                domain=domain,
                mode=mode,
                instance=instance,
                typed_result=None,
                diagnostics=("adapter produced no explicit typed goals",),
            )
        solved = OperatorKernel(instance.registry).solve(
            instance.state,
            instance.goals,
            policy=policy,
            budget=budget,
        )
        projection_applied = False
        diagnostics: list[str] = []
        if mode is MigrationMode.TYPED:
            project = getattr(adapter, "project", None)
            if callable(project) and not prebuilt_instance:
                projection_result = project(request.payload, solved)
                projection_applied = projection_result is not False
                if not projection_applied:
                    diagnostics.append(
                        "typed result has immutable source payload; proof metadata "
                        "was returned without legacy mutation"
                    )
            else:
                diagnostics.append(
                    "typed result has no mutable legacy projection for this payload"
                )
        else:
            diagnostics.append("shadow mode: source payload was not changed")
        return UnifiedTypedResult(
            domain=domain,
            mode=mode,
            instance=instance,
            typed_result=solved,
            proof_ko=render_proof_ko(solved),
            projection_applied=projection_applied,
            diagnostics=tuple(diagnostics),
        )

    def run_many(
        self,
        requests: Sequence[TypedDomainRequest],
        *,
        policy: ActionPolicy | None = None,
        budget: SolveBudget | None = None,
    ) -> tuple[UnifiedTypedResult, ...]:
        return tuple(
            self.run(request, policy=policy, budget=budget) for request in requests
        )

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .adapters import MigrationMode
from .contracts import DomainInstance, TypedDomainAdapter, TypedInstanceAugmenter
from .domains.language_text import LanguageInputAdapter
from .domains.linear_equation import MathInputAdapter
from .domains.composed_scene import SceneThresholdAdapter
from .domains.raster_vision import VisionInputAdapter
from .engine import ActionPolicy, OperatorKernel, RegistryPolicyProvider
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
            "grounding": (
                self.instance.grounding_trace.to_dict()
                if self.instance is not None
                else {}
            ),
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
        *,
        augmenters: Sequence[TypedInstanceAugmenter] = (),
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
        resolved_augmenters = tuple(augmenters)
        if any(
            not isinstance(augmenter, TypedInstanceAugmenter)
            for augmenter in resolved_augmenters
        ):
            raise TypeError(
                "typed reasoner augmenters must implement augment_instance"
            )
        self.augmenters = resolved_augmenters

    def ground(self, request: TypedDomainRequest) -> DomainInstance:
        """Adapt one raw request without adding learned executable structure."""

        domain = DomainKind(request.domain)
        if isinstance(request.payload, DomainInstance):
            return request.payload
        instance = self.adapters[domain].adapt(request.payload)
        if not isinstance(instance, DomainInstance):
            raise TypeError("typed domain adapter did not return a DomainInstance")
        return instance

    def prepare(self, request: TypedDomainRequest) -> DomainInstance:
        """Ground input and apply only explicitly configured verified augmenters."""

        instance = self.ground(request)
        for augmenter in self.augmenters:
            registry_contract = _registry_contract(instance)
            augmented = augmenter.augment_instance(instance)
            if not isinstance(augmented, DomainInstance):
                raise TypeError(
                    "typed instance augmenter did not return a DomainInstance"
                )
            if augmented is instance or augmented.registry is instance.registry:
                raise ValueError(
                    "typed instance augmenter must return a copied registry"
                )
            _require_operator_only_augmentation(
                instance,
                augmented,
                registry_contract,
            )
            if (
                augmented.state != instance.state
                or augmented.goals != instance.goals
                or augmented.domain != instance.domain
            ):
                raise ValueError(
                    "typed instance augmenter may add operators only; "
                    "facts, goals, and domain are immutable"
                )
            instance = augmented
        return instance

    def run(
        self,
        request: TypedDomainRequest,
        *,
        policy: ActionPolicy | RegistryPolicyProvider | None = None,
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

        prebuilt_instance = isinstance(request.payload, DomainInstance)
        instance = self.prepare(request)
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
            adapter = self.adapters[domain]
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
        policy: ActionPolicy | RegistryPolicyProvider | None = None,
        budget: SolveBudget | None = None,
    ) -> tuple[UnifiedTypedResult, ...]:
        return tuple(
            self.run(request, policy=policy, budget=budget) for request in requests
        )


def _registry_contract(instance: DomainInstance) -> tuple[Any, ...]:
    registry = instance.registry
    return (
        registry.types.snapshot(),
        dict(registry.functions),
        dict(registry.predicates),
        dict(registry.operators),
        dict(registry.guards),
    )


def _require_operator_only_augmentation(
    original: DomainInstance,
    augmented: DomainInstance,
    contract: tuple[Any, ...],
) -> None:
    types, functions, predicates, operators, guards = contract
    if _registry_contract(original) != contract:
        raise ValueError("typed instance augmenter mutated the source registry")
    registry = augmented.registry
    if (
        registry.types.snapshot() != types
        or registry.functions != functions
        or registry.predicates != predicates
        or registry.guards != guards
        or any(registry.operators.get(name) != spec for name, spec in operators.items())
    ):
        raise ValueError(
            "typed instance augmenter may only append operators to the registry"
        )

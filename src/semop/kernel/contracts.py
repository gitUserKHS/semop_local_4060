from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

from .grounding import GroundingBoundaryError, GroundingTrace
from .model import Goal, WorldState
from .registry import KernelRegistry


@dataclass(frozen=True)
class DomainInstance:
    registry: KernelRegistry
    state: WorldState
    goals: tuple[Goal, ...]
    domain: str
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)
    grounding_trace: GroundingTrace = field(default_factory=GroundingTrace)

    def __post_init__(self) -> None:
        audit = self.grounding_trace.audit(self.state.facts)
        if not audit.valid:
            raise GroundingBoundaryError(
                "domain grounding trace contains facts absent from its world state"
            )


InputT = TypeVar("InputT", contravariant=True)


@runtime_checkable
class TypedDomainAdapter(Protocol[InputT]):
    """Boundary contract for language, math, vision, or composed adapters."""

    def adapt(self, value: InputT) -> DomainInstance: ...


@runtime_checkable
class TypedInstanceAugmenter(Protocol):
    """Add verified executable structure to a freshly grounded instance."""

    def augment_instance(self, instance: DomainInstance) -> DomainInstance: ...

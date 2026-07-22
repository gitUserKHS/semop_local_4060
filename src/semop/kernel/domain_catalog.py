from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from .adapters import MigrationMode
from .contracts import TypedDomainAdapter


class DomainKind(str, Enum):
    CODING = "coding"
    LANGUAGE = "language"
    MATH = "math"
    VISION = "vision"
    COMPOSED = "composed"


@dataclass(frozen=True)
class TypedDomainRequest:
    domain: DomainKind | str
    payload: Any
    mode: MigrationMode | str = MigrationMode.SHADOW


@runtime_checkable
class SemanticPayloadCodec(Protocol):
    """Canonical raw-payload boundary used by data and evaluation artifacts."""

    def encode(self, value: Any) -> Mapping[str, Any]: ...

    def decode(self, payload: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True)
class FunctionSemanticCodec:
    name: str
    encoder: Callable[[Any], Mapping[str, Any]]
    decoder: Callable[[Mapping[str, Any]], Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("semantic codec name cannot be empty")
        if not callable(self.encoder) or not callable(self.decoder):
            raise TypeError("semantic codec encoder and decoder must be callable")

    def encode(self, value: Any) -> Mapping[str, Any]:
        encoded = self.encoder(value)
        if not isinstance(encoded, Mapping):
            raise TypeError(f"semantic codec {self.name} did not encode an object")
        return encoded

    def decode(self, payload: Mapping[str, Any]) -> Any:
        if not isinstance(payload, Mapping):
            raise TypeError(f"semantic codec {self.name} requires an object")
        return self.decoder(payload)


@dataclass(frozen=True)
class DomainSpec:
    """One self-describing domain boundary around the shared operator kernel."""

    kind: DomainKind | str
    adapter: TypedDomainAdapter[Any]
    input_contract: str
    description_ko: str
    capabilities: frozenset[str]
    semantic_codec: SemanticPayloadCodec | None = None

    def __post_init__(self) -> None:
        kind = DomainKind(self.kind)
        if not isinstance(self.adapter, TypedDomainAdapter):
            raise TypeError(f"domain {kind.value} adapter must implement adapt")
        input_contract = self.input_contract.strip()
        description = self.description_ko.strip()
        if not input_contract:
            raise ValueError(f"domain {kind.value} input contract cannot be empty")
        if not description:
            raise ValueError(f"domain {kind.value} description cannot be empty")
        capabilities = frozenset(
            str(capability).strip().lower() for capability in self.capabilities
        )
        if not capabilities:
            raise ValueError(f"domain {kind.value} must declare capabilities")
        invalid = tuple(
            sorted(
                capability
                for capability in capabilities
                if re.fullmatch(r"[a-z][a-z0-9_]*", capability) is None
            )
        )
        if invalid:
            raise ValueError(
                f"domain {kind.value} has invalid capabilities: {', '.join(invalid)}"
            )
        if self.semantic_codec is not None and not isinstance(
            self.semantic_codec, SemanticPayloadCodec
        ):
            raise TypeError(
                f"domain {kind.value} semantic codec must implement encode/decode"
            )
        if self.semantic_codec is not None:
            capabilities = capabilities | {"semantic_codec"}

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "input_contract", input_contract)
        object.__setattr__(self, "description_ko", description)
        object.__setattr__(self, "capabilities", capabilities)

    def supports(self, capability: str) -> bool:
        return capability.strip().lower() in self.capabilities

    def with_adapter(self, adapter: TypedDomainAdapter[Any]) -> DomainSpec:
        return replace(self, adapter=adapter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.kind.value,
            "input_contract": self.input_contract,
            "description_ko": self.description_ko,
            "capabilities": sorted(self.capabilities),
            "semantic_codec": (
                getattr(self.semantic_codec, "name", type(self.semantic_codec).__name__)
                if self.semantic_codec is not None
                else None
            ),
        }


@dataclass(frozen=True)
class DomainCatalog:
    """Immutable registry for domain-specific boundaries around one shared kernel."""

    specs: tuple[DomainSpec, ...]

    def __post_init__(self) -> None:
        resolved = tuple(sorted(self.specs, key=lambda item: item.kind.value))
        if not resolved:
            raise ValueError("domain catalog requires at least one domain")
        kinds = tuple(spec.kind for spec in resolved)
        if len(set(kinds)) != len(kinds):
            duplicates = sorted(
                kind.value for kind in set(kinds) if kinds.count(kind) > 1
            )
            raise ValueError(
                f"domain catalog contains duplicate domains: {', '.join(duplicates)}"
            )
        object.__setattr__(self, "specs", resolved)

    @property
    def adapters(self) -> Mapping[DomainKind, TypedDomainAdapter[Any]]:
        return MappingProxyType({spec.kind: spec.adapter for spec in self.specs})

    @property
    def kinds(self) -> tuple[DomainKind, ...]:
        return tuple(spec.kind for spec in self.specs)

    @property
    def semantic_kinds(self) -> tuple[DomainKind, ...]:
        return tuple(
            spec.kind for spec in self.specs if spec.semantic_codec is not None
        )

    def require(self, kind: DomainKind | str) -> DomainSpec:
        resolved = DomainKind(kind)
        for spec in self.specs:
            if spec.kind is resolved:
                return spec
        raise KeyError(f"domain {resolved.value!r} is not registered")

    def with_spec(
        self,
        spec: DomainSpec,
        *,
        replace_existing: bool = False,
    ) -> DomainCatalog:
        existing = {item.kind: item for item in self.specs}
        if spec.kind in existing and not replace_existing:
            raise ValueError(f"domain {spec.kind.value!r} is already registered")
        existing[spec.kind] = spec
        return DomainCatalog(tuple(existing.values()))

    def with_adapter(
        self,
        kind: DomainKind | str,
        adapter: TypedDomainAdapter[Any],
    ) -> DomainCatalog:
        spec = self.require(kind)
        return self.with_spec(spec.with_adapter(adapter), replace_existing=True)

    def describe(self) -> tuple[dict[str, Any], ...]:
        return tuple(spec.to_dict() for spec in self.specs)

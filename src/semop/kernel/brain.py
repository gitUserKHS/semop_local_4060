from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .contracts import DomainInstance
from .engine import ActionPolicy, OperatorKernel
from .library import MdlMacroLibrary
from .macro_policy import PrimitiveMacroPolicy
from .model import SolveBudget, SolveResult
from .registry import KernelRegistry
from .self_learning import PolicyCandidate, PolicyLearner


@dataclass(frozen=True)
class HierarchicalOperatorBrain:
    """Portable controller plus verifier-gated primitive program memory."""

    base_policy: ActionPolicy = field(compare=False, repr=False)
    base_kind: str
    base_artifact_suffix: str
    base_artifact: bytes = field(compare=False, repr=False)
    base_parameter_count: int
    macro_library: MdlMacroLibrary = field(compare=False, repr=False)

    FORMAT_VERSION = 1

    def __post_init__(self) -> None:
        if not self.base_kind.strip():
            raise ValueError("hierarchical brain base kind must not be empty")
        if not self.base_artifact_suffix.startswith("."):
            raise ValueError("hierarchical brain artifact suffix must begin with '.'")
        if any(
            character not in ".-_abcdefghijklmnopqrstuvwxyz0123456789"
            for character in self.base_artifact_suffix.lower()
        ):
            raise ValueError("hierarchical brain artifact suffix is unsafe")
        if not self.base_artifact:
            raise ValueError("hierarchical brain requires a base policy artifact")
        if self.base_parameter_count < 0:
            raise ValueError("hierarchical brain parameter count must not be negative")
        if not self.macro_library.records:
            raise ValueError("hierarchical brain requires promoted macro memory")

    @classmethod
    def from_candidate(
        cls,
        candidate: PolicyCandidate,
        macro_library: MdlMacroLibrary,
    ) -> "HierarchicalOperatorBrain":
        return cls(
            base_policy=candidate.policy,
            base_kind=candidate.kind,
            base_artifact_suffix=candidate.artifact_suffix,
            base_artifact=candidate.artifact,
            base_parameter_count=candidate.parameter_count,
            macro_library=macro_library,
        )

    @property
    def parameter_count(self) -> int:
        return self.base_parameter_count

    @property
    def macro_count(self) -> int:
        return len(self.macro_library.records)

    @property
    def artifact_sha256(self) -> str:
        return sha256(self.to_artifact()).hexdigest()

    def policy_for(self, registry: KernelRegistry) -> ActionPolicy:
        return PrimitiveMacroPolicy(
            registry,
            self.macro_library,
            base_policy=self.base_policy,
        )

    def solve(
        self,
        instance: DomainInstance,
        *,
        budget: SolveBudget | None = None,
    ) -> SolveResult:
        return OperatorKernel(instance.registry).solve(
            instance.state,
            instance.goals,
            policy=self,
            budget=budget,
        )

    def to_artifact(self) -> bytes:
        macro_payload = self.macro_library.to_payload()
        payload = {
            "format_version": self.FORMAT_VERSION,
            "base_policy": {
                "kind": self.base_kind,
                "artifact_suffix": self.base_artifact_suffix,
                "parameter_count": self.base_parameter_count,
                "sha256": sha256(self.base_artifact).hexdigest(),
                "artifact_base64": b64encode(self.base_artifact).decode("ascii"),
            },
            "macro_library": macro_payload,
            "macro_sha256": sha256(
                _canonical_payload_bytes(macro_payload)
            ).hexdigest(),
        }
        return _canonical_payload_bytes(payload)

    @classmethod
    def from_artifact(
        cls,
        artifact: bytes,
        learner: PolicyLearner,
    ) -> "HierarchicalOperatorBrain":
        try:
            payload: dict[str, Any] = json.loads(artifact.decode("utf-8"))
            if payload.get("format_version") != cls.FORMAT_VERSION:
                raise ValueError("unsupported hierarchical brain format")
            base = payload["base_policy"]
            kind = str(base["kind"])
            if kind != learner.name:
                raise ValueError(
                    "hierarchical brain policy kind does not match learner"
                )
            base_artifact = b64decode(
                str(base["artifact_base64"]),
                validate=True,
            )
            if sha256(base_artifact).hexdigest() != str(base["sha256"]):
                raise ValueError("hierarchical brain base artifact hash mismatch")
            macro_payload = payload["macro_library"]
            if sha256(_canonical_payload_bytes(macro_payload)).hexdigest() != str(
                payload["macro_sha256"]
            ):
                raise ValueError("hierarchical brain macro artifact hash mismatch")
            macro_library = MdlMacroLibrary.from_payload(macro_payload)
            base_parameter_count = int(base["parameter_count"])
            base_policy = learner.restore(base_artifact)
            restored_parameter_count = getattr(
                base_policy,
                "parameter_count",
                base_parameter_count,
            )
            if int(restored_parameter_count) != base_parameter_count:
                raise ValueError(
                    "hierarchical brain policy parameter count mismatch"
                )
            return cls(
                base_policy=base_policy,
                base_kind=kind,
                base_artifact_suffix=str(base["artifact_suffix"]),
                base_artifact=base_artifact,
                base_parameter_count=base_parameter_count,
                macro_library=macro_library,
            )
        except (
            KeyError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError("invalid hierarchical brain artifact") from exc

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{uuid4().hex}.tmp"
        )
        try:
            temporary.write_bytes(self.to_artifact())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        learner: PolicyLearner,
    ) -> "HierarchicalOperatorBrain":
        return cls.from_artifact(Path(path).read_bytes(), learner)


def _canonical_payload_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

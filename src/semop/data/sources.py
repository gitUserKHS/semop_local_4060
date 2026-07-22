from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlparse


DATASET_SOURCE_SCHEMA_VERSION = "semop.dataset-sources.v1"
DATASET_RECEIPT_SCHEMA_VERSION = "semop.dataset-receipt.v1"

_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DatasetDomain(str, Enum):
    LANGUAGE = "language"
    MATH = "math"
    VISION = "vision"


class DatasetLabelAuthority(str, Enum):
    """Who supplied the labels before SemOp performs its own verification."""

    PROGRAMMATIC_ORACLE = "programmatic_oracle"
    PUBLISHED_LABEL = "published_label"
    MODEL_PROPOSAL = "model_proposal"
    HUMAN_REVIEW = "human_review"


class DatasetUsePolicy(str, Enum):
    """Gate that must pass before a record can become grounding supervision."""

    PROGRAMMATIC_REPLAY = "programmatic_replay"
    VERIFIER_REQUIRED = "verifier_required"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


def _require_https(value: str, field_name: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute HTTPS URL")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class DatasetSource:
    source_id: str
    domain: DatasetDomain
    title: str
    homepage_url: str
    artifact_url: str
    artifact_filename: str
    revision: str
    license_spdx: str
    license_url: str
    max_bytes: int
    label_authority: DatasetLabelAuthority
    use_policy: DatasetUsePolicy
    adapter_id: str
    expected_sha256: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not _IDENTIFIER_RE.fullmatch(self.source_id):
            raise ValueError("source_id must use lowercase letters, digits, '.', '_' or '-'")
        if not self.title.strip():
            raise ValueError("dataset title cannot be empty")
        _require_https(self.homepage_url, "homepage_url")
        _require_https(self.artifact_url, "artifact_url")
        _require_https(self.license_url, "license_url")
        filename = Path(self.artifact_filename)
        if (
            not self.artifact_filename.strip()
            or filename.name != self.artifact_filename
            or self.artifact_filename in {".", ".."}
        ):
            raise ValueError("artifact_filename must be a safe basename")
        if not self.revision.strip():
            raise ValueError("dataset revision cannot be empty")
        if self.revision not in self.artifact_url:
            raise ValueError("artifact_url must contain the pinned revision")
        if not self.license_spdx.strip():
            raise ValueError("license_spdx cannot be empty")
        if not 0 < self.max_bytes <= 2 * 1024 * 1024 * 1024:
            raise ValueError("max_bytes must be between 1 byte and 2 GiB")
        if not _IDENTIFIER_RE.fullmatch(self.adapter_id):
            raise ValueError("adapter_id must use lowercase letters, digits, '.', '_' or '-'")
        if self.expected_sha256 is not None and not _SHA256_RE.fullmatch(
            self.expected_sha256
        ):
            raise ValueError("expected_sha256 must contain 64 lowercase hexadecimal digits")
        if (
            self.label_authority is DatasetLabelAuthority.MODEL_PROPOSAL
            and self.use_policy is DatasetUsePolicy.PROGRAMMATIC_REPLAY
        ):
            raise ValueError("model-proposed labels cannot use programmatic replay authority")
        if (
            self.label_authority is DatasetLabelAuthority.PUBLISHED_LABEL
            and self.use_policy is DatasetUsePolicy.PROGRAMMATIC_REPLAY
        ):
            raise ValueError("published labels require a SemOp verifier or human review")

    @property
    def integrity_locked(self) -> bool:
        return self.expected_sha256 is not None

    @property
    def direct_training_allowed(self) -> bool:
        """Downloaded or generated rows never bypass their declared semantic gate."""

        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "domain": self.domain.value,
            "title": self.title,
            "homepage_url": self.homepage_url,
            "artifact_url": self.artifact_url,
            "artifact_filename": self.artifact_filename,
            "revision": self.revision,
            "license_spdx": self.license_spdx,
            "license_url": self.license_url,
            "max_bytes": self.max_bytes,
            "label_authority": self.label_authority.value,
            "use_policy": self.use_policy.value,
            "adapter_id": self.adapter_id,
            "expected_sha256": self.expected_sha256,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DatasetSource:
        return cls(
            source_id=str(value["source_id"]),
            domain=DatasetDomain(str(value["domain"])),
            title=str(value["title"]),
            homepage_url=str(value["homepage_url"]),
            artifact_url=str(value["artifact_url"]),
            artifact_filename=str(value["artifact_filename"]),
            revision=str(value["revision"]),
            license_spdx=str(value["license_spdx"]),
            license_url=str(value["license_url"]),
            max_bytes=int(value["max_bytes"]),
            label_authority=DatasetLabelAuthority(str(value["label_authority"])),
            use_policy=DatasetUsePolicy(str(value["use_policy"])),
            adapter_id=str(value["adapter_id"]),
            expected_sha256=(
                str(value["expected_sha256"])
                if value.get("expected_sha256") is not None
                else None
            ),
            notes=str(value.get("notes", "")),
        )


@dataclass(frozen=True)
class DatasetSourceManifest:
    sources: tuple[DatasetSource, ...]
    schema_version: str = DATASET_SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_SOURCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported dataset source schema: {self.schema_version}")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("dataset source ids must be unique")

    @property
    def digest(self) -> str:
        return sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def get(self, source_id: str) -> DatasetSource:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(f"unknown dataset source: {source_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sources": [source.to_dict() for source in self.sources],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DatasetSourceManifest:
        raw_sources = value.get("sources")
        if not isinstance(raw_sources, list):
            raise ValueError("dataset source manifest requires a sources list")
        return cls(
            sources=tuple(DatasetSource.from_dict(item) for item in raw_sources),
            schema_version=str(value.get("schema_version", "")),
        )


@dataclass(frozen=True)
class DatasetArtifactReceipt:
    source_id: str
    source_revision: str
    manifest_digest: str
    artifact_url: str
    artifact_path: str
    sha256: str
    byte_count: int
    fetched_at: str
    integrity_locked: bool
    schema_version: str = DATASET_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_RECEIPT_SCHEMA_VERSION:
            raise ValueError(f"unsupported dataset receipt schema: {self.schema_version}")
        if not _IDENTIFIER_RE.fullmatch(self.source_id):
            raise ValueError("invalid receipt source_id")
        if not _SHA256_RE.fullmatch(self.manifest_digest):
            raise ValueError("manifest_digest must be a SHA-256 digest")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("receipt sha256 must be a SHA-256 digest")
        if self.byte_count < 0:
            raise ValueError("receipt byte_count cannot be negative")
        _require_https(self.artifact_url, "artifact_url")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "manifest_digest": self.manifest_digest,
            "artifact_url": self.artifact_url,
            "artifact_path": self.artifact_path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "fetched_at": self.fetched_at,
            "integrity_locked": self.integrity_locked,
        }


def load_dataset_source_manifest(path: str | Path) -> DatasetSourceManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset source manifest must be a JSON object")
    return DatasetSourceManifest.from_dict(payload)

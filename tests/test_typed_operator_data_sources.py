from __future__ import annotations

from io import BytesIO
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from semop.data import (
    DatasetAcquisitionError,
    DatasetDomain,
    DatasetLabelAuthority,
    DatasetSource,
    DatasetSourceManifest,
    DatasetUsePolicy,
    fetch_dataset_source,
    load_dataset_source_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "sources" / "semantic_sources.v1.json"


class _Response(BytesIO):
    def __init__(self, payload: bytes, url: str) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}
        self._url = url

    def geturl(self) -> str:
        return self._url


def _source(payload: bytes, *, expected: str | None = None, max_bytes: int = 1024):
    revision = "revision-1"
    return DatasetSource(
        source_id="unit-source",
        domain=DatasetDomain.LANGUAGE,
        title="Unit source",
        homepage_url="https://example.test/source",
        artifact_url=f"https://example.test/{revision}/data.bin",
        artifact_filename="data.bin",
        revision=revision,
        license_spdx="Apache-2.0",
        license_url="https://example.test/license",
        max_bytes=max_bytes,
        label_authority=DatasetLabelAuthority.PUBLISHED_LABEL,
        use_policy=DatasetUsePolicy.VERIFIER_REQUIRED,
        adapter_id="unit-adapter",
        expected_sha256=expected,
    )


def test_official_source_catalog_covers_language_math_and_vision() -> None:
    manifest = load_dataset_source_manifest(MANIFEST_PATH)

    assert {source.domain for source in manifest.sources} == set(DatasetDomain)
    assert len(manifest.digest) == 64
    assert all(source.revision in source.artifact_url for source in manifest.sources)
    assert all(source.license_spdx for source in manifest.sources)
    assert all(source.integrity_locked for source in manifest.sources)
    assert all(not source.direct_training_allowed for source in manifest.sources)
    assert all(
        source.use_policy is DatasetUsePolicy.VERIFIER_REQUIRED
        for source in manifest.sources
    )


def test_manifest_rejects_duplicate_source_ids() -> None:
    source = _source(b"payload")

    with pytest.raises(ValueError, match="source ids must be unique"):
        DatasetSourceManifest((source, source))


def test_model_proposal_cannot_claim_programmatic_replay() -> None:
    source = _source(b"payload")
    values = source.to_dict()
    values["label_authority"] = DatasetLabelAuthority.MODEL_PROPOSAL.value
    values["use_policy"] = DatasetUsePolicy.PROGRAMMATIC_REPLAY.value

    with pytest.raises(ValueError, match="model-proposed labels"):
        DatasetSource.from_dict(values)


def test_fetch_verifies_digest_size_and_writes_atomically(tmp_path: Path) -> None:
    payload = b"verified dataset bytes"
    expected = sha256(payload).hexdigest()
    source = _source(payload, expected=expected)

    artifact = fetch_dataset_source(
        source,
        tmp_path,
        manifest_digest="1" * 64,
        fetched_at="2026-07-18T00:00:00Z",
        open_url=lambda url, timeout: _Response(payload, url),
    )

    assert artifact.downloaded
    assert artifact.path.read_bytes() == payload
    assert artifact.receipt.sha256 == expected
    assert artifact.receipt.integrity_locked
    assert not list(tmp_path.rglob("*.part"))


def test_unpinned_fetch_is_fail_closed_without_bootstrap(tmp_path: Path) -> None:
    source = _source(b"payload")

    with pytest.raises(DatasetAcquisitionError, match="bootstrap mode"):
        fetch_dataset_source(
            source,
            tmp_path,
            manifest_digest="2" * 64,
            open_url=lambda url, timeout: _Response(b"payload", url),
        )


def test_fetch_removes_partial_file_on_digest_or_size_failure(tmp_path: Path) -> None:
    payload = b"payload larger than the contract"
    source = _source(payload, expected="0" * 64, max_bytes=8)

    with pytest.raises(DatasetAcquisitionError):
        fetch_dataset_source(
            source,
            tmp_path,
            manifest_digest="3" * 64,
            open_url=lambda url, timeout: _Response(payload, url),
        )

    assert not list(tmp_path.rglob("*.part"))
    assert not list(tmp_path.rglob("data.bin"))


def test_fetch_cli_lists_sources_without_network_access() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "data" / "fetch_semantic_sources.py"),
            "--manifest",
            str(MANIFEST_PATH),
            "--list",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert {item["domain"] for item in payload["sources"]} == {
        "language",
        "math",
        "vision",
    }
    assert all(not item["direct_training_allowed"] for item in payload["sources"])

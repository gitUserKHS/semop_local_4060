from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import BinaryIO, Callable, ContextManager
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .sources import DatasetArtifactReceipt, DatasetSource


class DatasetAcquisitionError(RuntimeError):
    """Raised when a remote artifact violates its declared source contract."""


OpenUrl = Callable[[str, int], ContextManager[BinaryIO]]


@dataclass(frozen=True)
class FetchedDatasetArtifact:
    path: Path
    receipt: DatasetArtifactReceipt
    downloaded: bool


def _utc_now() -> str:
    value = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return value.replace("+00:00", "Z")


def _default_open_url(url: str, timeout: int) -> ContextManager[BinaryIO]:
    request = Request(url, headers={"User-Agent": "SemOp-Dataset-Acquisition/1"})
    return closing(urlopen(request, timeout=timeout))


def _stream_digest(path: Path, chunk_size: int = 64 * 1024) -> tuple[str, int]:
    digest = sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _validate_existing(source: DatasetSource, target: Path) -> tuple[str, int]:
    digest, byte_count = _stream_digest(target)
    if byte_count > source.max_bytes:
        raise DatasetAcquisitionError(
            f"cached artifact exceeds max_bytes for {source.source_id}"
        )
    if source.expected_sha256 is not None and digest != source.expected_sha256:
        raise DatasetAcquisitionError(
            f"cached artifact digest mismatch for {source.source_id}"
        )
    return digest, byte_count


def fetch_dataset_source(
    source: DatasetSource,
    output_dir: str | Path,
    *,
    manifest_digest: str,
    overwrite: bool = False,
    allow_unpinned: bool = False,
    timeout: int = 60,
    fetched_at: str | None = None,
    open_url: OpenUrl = _default_open_url,
) -> FetchedDatasetArtifact:
    if source.expected_sha256 is None and not allow_unpinned:
        raise DatasetAcquisitionError(
            f"{source.source_id} has no expected SHA-256; use bootstrap mode explicitly"
        )

    source_dir = Path(output_dir) / source.source_id
    source_dir.mkdir(parents=True, exist_ok=True)
    target = source_dir / source.artifact_filename
    timestamp = fetched_at or _utc_now()

    if target.exists() and not overwrite:
        digest, byte_count = _validate_existing(source, target)
        receipt = DatasetArtifactReceipt(
            source_id=source.source_id,
            source_revision=source.revision,
            manifest_digest=manifest_digest,
            artifact_url=source.artifact_url,
            artifact_path=f"{source.source_id}/{source.artifact_filename}",
            sha256=digest,
            byte_count=byte_count,
            fetched_at=timestamp,
            integrity_locked=source.expected_sha256 is not None,
        )
        return FetchedDatasetArtifact(target, receipt, downloaded=False)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{source.artifact_filename}.", suffix=".part",
            dir=source_dir, delete=False
        ) as output:
            temp_path = Path(output.name)
            digest = sha256()
            byte_count = 0
            with open_url(source.artifact_url, timeout) as response:
                final_url = getattr(response, "geturl", lambda: source.artifact_url)()
                parsed_final = urlparse(str(final_url))
                if parsed_final.scheme != "https" or not parsed_final.netloc:
                    raise DatasetAcquisitionError("dataset redirect must remain on HTTPS")
                headers = getattr(response, "headers", None)
                content_length = headers.get("Content-Length") if headers is not None else None
                if content_length is not None and int(content_length) > source.max_bytes:
                    raise DatasetAcquisitionError(
                        f"declared artifact size exceeds max_bytes for {source.source_id}"
                    )
                while chunk := response.read(64 * 1024):
                    byte_count += len(chunk)
                    if byte_count > source.max_bytes:
                        raise DatasetAcquisitionError(
                            f"download exceeded max_bytes for {source.source_id}"
                        )
                    digest.update(chunk)
                    output.write(chunk)

        observed_digest = digest.hexdigest()
        if (
            source.expected_sha256 is not None
            and observed_digest != source.expected_sha256
        ):
            raise DatasetAcquisitionError(
                f"downloaded artifact digest mismatch for {source.source_id}"
            )
        os.replace(temp_path, target)
        temp_path = None
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    receipt = DatasetArtifactReceipt(
        source_id=source.source_id,
        source_revision=source.revision,
        manifest_digest=manifest_digest,
        artifact_url=source.artifact_url,
        artifact_path=f"{source.source_id}/{source.artifact_filename}",
        sha256=observed_digest,
        byte_count=byte_count,
        fetched_at=timestamp,
        integrity_locked=source.expected_sha256 is not None,
    )
    return FetchedDatasetArtifact(target, receipt, downloaded=True)


def write_dataset_receipts(
    path: str | Path,
    receipts: tuple[DatasetArtifactReceipt, ...] | list[DatasetArtifactReceipt],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(receipt.to_dict(), ensure_ascii=False, sort_keys=True) for receipt in receipts]
    payload = "\n".join(lines) + ("\n" if lines else "")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{target.name}.", suffix=".part",
        dir=target.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
    os.replace(temp_path, target)

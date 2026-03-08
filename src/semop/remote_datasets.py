from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse
from urllib.request import urlopen
import shutil
import zipfile


SUPPORTED_DATASET_SUFFIXES = {".jsonl", ".json", ".csv", ".txt"}


@dataclass
class DownloadedArtifact:
    url: str
    saved_path: Path
    extracted_paths: List[Path]

    @property
    def dataset_paths(self) -> List[Path]:
        if self.extracted_paths:
            return [path for path in self.extracted_paths if path.suffix.lower() in SUPPORTED_DATASET_SUFFIXES]
        return [self.saved_path] if self.saved_path.suffix.lower() in SUPPORTED_DATASET_SUFFIXES else []


class RemoteDatasetDownloader:
    def download(self, url: str, output_dir: str | Path, filename: str | None = None, extract: bool = False, overwrite: bool = False, timeout: int = 60) -> DownloadedArtifact:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        parsed = urlparse(url)
        inferred_name = filename or Path(parsed.path or url).name or "dataset.bin"
        saved_path = output_root / inferred_name

        if saved_path.exists() and not overwrite:
            extracted_paths = self._extract_if_needed(saved_path, output_root, extract=extract, overwrite=overwrite)
            return DownloadedArtifact(url=url, saved_path=saved_path, extracted_paths=extracted_paths)

        local_source = self._local_source_path(url)
        if local_source is not None:
            shutil.copyfile(local_source, saved_path)
        else:
            with urlopen(url, timeout=timeout) as response, saved_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)

        extracted_paths = self._extract_if_needed(saved_path, output_root, extract=extract, overwrite=overwrite)
        return DownloadedArtifact(url=url, saved_path=saved_path, extracted_paths=extracted_paths)

    def download_many(self, urls: Iterable[str], output_dir: str | Path, extract: bool = False, overwrite: bool = False, timeout: int = 60) -> List[DownloadedArtifact]:
        artifacts: List[DownloadedArtifact] = []
        for url in urls:
            artifacts.append(self.download(url, output_dir=output_dir, extract=extract, overwrite=overwrite, timeout=timeout))
        return artifacts

    def _extract_if_needed(self, saved_path: Path, output_root: Path, extract: bool, overwrite: bool) -> List[Path]:
        if not extract or saved_path.suffix.lower() != ".zip":
            return []

        target_dir = output_root / saved_path.stem
        if target_dir.exists() and overwrite:
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(saved_path) as archive:
            archive.extractall(target_dir)

        extracted: List[Path] = []
        for path in target_dir.rglob("*"):
            if path.is_file():
                extracted.append(path)
        return extracted

    @staticmethod
    def _local_source_path(url: str) -> Path | None:
        if len(url) >= 2 and url[1] == ":":
            candidate = Path(url)
            if candidate.exists():
                return candidate

        candidate = Path(url)
        if candidate.exists():
            return candidate

        parsed = urlparse(url)
        if len(parsed.scheme) == 1:
            drive_candidate = Path(f"{parsed.scheme}:{parsed.path}")
            if drive_candidate.exists():
                return drive_candidate
        if parsed.scheme == "file":
            file_path = Path(parsed.path.lstrip("/"))
            if file_path.exists():
                return file_path
        return None

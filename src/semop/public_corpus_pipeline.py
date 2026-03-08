from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from .corpus_builder import CorpusBuilder
from .corpus_store import CorpusMemoryStore
from .dataset_adapters import PublicDatasetAdapter
from .pipeline import StructuredMeaningPipeline
from .remote_datasets import RemoteDatasetDownloader, SUPPORTED_DATASET_SUFFIXES


@dataclass
class ManifestEntry:
    name: str
    urls: List[str]
    source: str
    extract: bool = False
    augment: bool = True
    train_ratio: float = 0.8
    query_fields: List[str] = field(default_factory=list)
    context_fields: List[str] = field(default_factory=list)
    answer_fields: List[str] = field(default_factory=list)


@dataclass
class ManifestRunSummary:
    name: str
    source: str
    downloaded_files: int
    normalized_examples: int
    built_records: int
    stored_examples: int
    normalized_output: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PublicCorpusIngestor:
    def __init__(self, mode: str = "heuristic", model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.mode = mode
        self.model_id = model_id
        self.downloader = RemoteDatasetDownloader()

    def load_manifest(self, path: str | Path) -> List[ManifestEntry]:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            items = payload.get("datasets", [])
        else:
            items = payload
        entries: List[ManifestEntry] = []
        for item in items:
            entries.append(
                ManifestEntry(
                    name=item["name"],
                    urls=list(item["urls"]),
                    source=item.get("source", item["name"]),
                    extract=bool(item.get("extract", False)),
                    augment=bool(item.get("augment", True)),
                    train_ratio=float(item.get("train_ratio", 0.8)),
                    query_fields=list(item.get("query_fields", [])),
                    context_fields=list(item.get("context_fields", [])),
                    answer_fields=list(item.get("answer_fields", [])),
                )
            )
        return entries

    def run_manifest(
        self,
        manifest_path: str | Path,
        download_root: str | Path,
        normalized_root: str | Path,
        store_path: str | Path,
        overwrite: bool = False,
    ) -> List[ManifestRunSummary]:
        manifest_file = Path(manifest_path)
        summaries: List[ManifestRunSummary] = []
        store = CorpusMemoryStore(store_path)
        pipeline = StructuredMeaningPipeline(mode=self.mode, model_id=self.model_id)

        for entry in self.load_manifest(manifest_file):
            dataset_paths = self._download_entry(entry, manifest_file.parent, download_root, overwrite=overwrite)
            adapter = PublicDatasetAdapter(
                query_fields=entry.query_fields or None,
                context_fields=entry.context_fields or None,
                answer_fields=entry.answer_fields or None,
            )
            examples = adapter.load_paths(dataset_paths)
            builder = CorpusBuilder(train_ratio=entry.train_ratio)
            records = builder.build_from_queries([(example.query, entry.source) for example in examples], augment=entry.augment)

            normalized_path = Path(normalized_root) / f"{entry.name}.jsonl"
            builder.save_jsonl(records, normalized_path)

            for record in records:
                graph = pipeline.run(record.query)
                store.upsert_graph(graph, source=entry.source, split=record.split)

            summaries.append(
                ManifestRunSummary(
                    name=entry.name,
                    source=entry.source,
                    downloaded_files=len(dataset_paths),
                    normalized_examples=len(examples),
                    built_records=len(records),
                    stored_examples=store.count_examples(source=entry.source),
                    normalized_output=str(normalized_path),
                )
            )
        return summaries

    def _download_entry(self, entry: ManifestEntry, manifest_root: Path, download_root: str | Path, overwrite: bool = False) -> List[Path]:
        target_dir = Path(download_root) / entry.name
        resolved_urls = [self._resolve_url(url, manifest_root) for url in entry.urls]
        artifacts = self.downloader.download_many(resolved_urls, output_dir=target_dir, extract=entry.extract, overwrite=overwrite)
        dataset_paths: List[Path] = []
        for artifact in artifacts:
            candidate_paths = artifact.dataset_paths or [artifact.saved_path]
            for path in candidate_paths:
                if path.is_file() and path.suffix.lower() in SUPPORTED_DATASET_SUFFIXES:
                    dataset_paths.append(path)
                elif path.is_dir():
                    dataset_paths.extend(child for child in path.rglob("*") if child.is_file() and child.suffix.lower() in SUPPORTED_DATASET_SUFFIXES)
        return list(dict.fromkeys(dataset_paths))

    @staticmethod
    def _resolve_url(url: str, manifest_root: Path) -> str:
        parsed = urlparse(url)
        if parsed.scheme:
            return url
        candidate = Path(url)
        if not candidate.is_absolute():
            candidate = manifest_root / candidate
        return str(candidate.resolve())

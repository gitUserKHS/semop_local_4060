from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .remote_datasets import RemoteDatasetDownloader


@dataclass
class CpLabeledDatasetEntry:
    name: str
    source_type: str = "url"
    url: str = ""
    extract: bool = False
    hf_dataset: str = ""
    hf_config: str = ""
    hf_revision: str = ""
    hf_splits: List[str] | None = None
    statement_fields: List[str] | None = None
    solution_fields: List[str] | None = None
    id_fields: List[str] | None = None
    tag_fields: List[str] | None = None
    include_any_tags: List[str] | None = None
    include_text_terms: List[str] | None = None
    exclude_text_terms: List[str] | None = None


@dataclass
class CpLabeledDatasetSummary:
    manifest_path: str
    output_path: str
    datasets: int
    examples: int

    def model_dump(self) -> Dict[str, Any]:
        return {
            'manifest_path': self.manifest_path,
            'output_path': self.output_path,
            'datasets': self.datasets,
            'examples': self.examples,
        }


class CpLabeledDatasetDownloader:
    def __init__(self) -> None:
        self.downloader = RemoteDatasetDownloader()

    def load_manifest(self, path: str | Path) -> List[CpLabeledDatasetEntry]:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        entries: List[CpLabeledDatasetEntry] = []
        for item in payload.get('datasets', []):
            if item.get('enabled', True) is False:
                continue
            statement_fields = [str(v) for v in item.get('statement_fields', [])]
            solution_fields = [str(v) for v in item.get('solution_fields', [])]
            id_fields = [str(v) for v in item.get('id_fields', [])]
            if not statement_fields:
                statement_fields = [str(item.get('statement_field', 'statement'))]
            if not solution_fields:
                solution_fields = [str(item.get('solution_field', 'solution'))]
            if not id_fields:
                id_fields = [str(item.get('id_field', 'problem_id'))]
            entries.append(CpLabeledDatasetEntry(
                name=str(item['name']),
                source_type=str(item.get('source_type', 'url')),
                url=str(item.get('url', '')),
                extract=bool(item.get('extract', False)),
                hf_dataset=str(item.get('hf_dataset', '')),
                hf_config=str(item.get('hf_config', '')),
                hf_revision=str(item.get('hf_revision', '')),
                hf_splits=[str(v) for v in item.get('hf_splits', ['train'])],
                statement_fields=statement_fields,
                solution_fields=solution_fields,
                id_fields=id_fields,
                tag_fields=[str(v) for v in item.get('tag_fields', ['tags'])],
                include_any_tags=[str(v).lower() for v in item.get('include_any_tags', [])],
                include_text_terms=[str(v).lower() for v in item.get('include_text_terms', [])],
                exclude_text_terms=[str(v).lower() for v in item.get('exclude_text_terms', [])],
            ))
        return entries

    def download_and_normalize(self, manifest_path: str | Path, download_root: str | Path, output_path: str | Path, overwrite: bool = False) -> CpLabeledDatasetSummary:
        entries = self.load_manifest(manifest_path)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        examples = 0
        with output.open('w', encoding='utf-8') as handle:
            for entry in entries:
                if entry.source_type == 'huggingface':
                    rows = self._iter_huggingface_rows(entry)
                    dataset_paths = [Path(f'hf://{entry.hf_dataset}:{split}') for split in (entry.hf_splits or ['train'])]
                else:
                    artifact = self.downloader.download(entry.url, output_dir=Path(download_root) / entry.name, extract=entry.extract, overwrite=overwrite)
                    dataset_paths = artifact.dataset_paths or [artifact.saved_path]
                    rows = []
                    for dataset_path in dataset_paths:
                        rows.extend(list(self._iter_rows(dataset_path)))
                for row in rows:
                    normalized = self._normalize_row(row, entry)
                    if normalized is None:
                        continue
                    handle.write(json.dumps(normalized, ensure_ascii=False) + '\n')
                    examples += 1
        return CpLabeledDatasetSummary(
            manifest_path=str(manifest_path),
            output_path=str(output_path),
            datasets=len(entries),
            examples=examples,
        )

    def _iter_huggingface_rows(self, entry: CpLabeledDatasetEntry) -> List[Dict[str, Any]]:
        try:
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover
            raise RuntimeError('datasets package is required for Hugging Face dataset downloads') from exc

        last_error: Exception | None = None
        for split in entry.hf_splits or ['train']:
            try:
                kwargs: Dict[str, Any] = {}
                if entry.hf_config:
                    kwargs['name'] = entry.hf_config
                if entry.hf_revision:
                    kwargs['revision'] = entry.hf_revision
                dataset = load_dataset(entry.hf_dataset, split=split, **kwargs)
                return [dict(row) for row in dataset]
            except Exception as exc:  # pragma: no cover
                last_error = exc
                continue
        if last_error is not None:
            raise RuntimeError(f'failed to load huggingface dataset {entry.hf_dataset}') from last_error
        return []

    def _normalize_row(self, row: Dict[str, Any], entry: CpLabeledDatasetEntry) -> Dict[str, Any] | None:
        statement = self._first_text(row, entry.statement_fields or ['statement'])
        if not statement:
            return None
        solution = self._first_text(row, entry.solution_fields or ['solution'])
        problem_id = self._first_text(row, entry.id_fields or ['problem_id'])
        tags = self._collect_tags(row, entry.tag_fields or ['tags'])
        if not self._matches_filters(statement, solution, tags, entry):
            return None
        excluded = set((entry.statement_fields or []) + (entry.solution_fields or []) + (entry.id_fields or []) + (entry.tag_fields or []))
        return {
            'problem_id': problem_id,
            'statement': statement,
            'solution': solution,
            'tags': tags,
            'source_dataset': entry.name,
            'source_type': entry.source_type,
            'metadata': {k: v for k, v in row.items() if k not in excluded},
        }

    def _iter_rows(self, path: Path) -> Iterable[Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == '.jsonl':
            with path.open('r', encoding='utf-8-sig') as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if line:
                        yield json.loads(line)
            return
        if suffix == '.json':
            payload = json.loads(path.read_text(encoding='utf-8-sig'))
            if isinstance(payload, list):
                for row in payload:
                    if isinstance(row, dict):
                        yield row
            elif isinstance(payload, dict):
                rows = payload.get('items') or payload.get('examples') or payload.get('data') or []
                for row in rows:
                    if isinstance(row, dict):
                        yield row
            return
        if suffix == '.csv':
            with path.open('r', encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    yield dict(row)
            return

    @staticmethod
    def _first_text(row: Dict[str, Any], fields: Sequence[str]) -> str:
        for field in fields:
            value = row.get(field)
            if value is None:
                continue
            if isinstance(value, list):
                joined = ' '.join(str(v).strip() for v in value if str(v).strip()).strip()
                if joined:
                    return joined
            else:
                text = str(value).strip()
                if text:
                    return text
        return ''

    @staticmethod
    def _collect_tags(row: Dict[str, Any], fields: Sequence[str]) -> List[str]:
        tags: List[str] = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, list):
                tags.extend(str(v) for v in value if str(v).strip())
            elif value:
                raw = str(value)
                if ',' in raw:
                    tags.extend(part.strip() for part in raw.split(',') if part.strip())
                else:
                    tags.append(raw.strip())
        return list(dict.fromkeys(tag for tag in tags if tag))

    @staticmethod
    def _matches_filters(statement: str, solution: str, tags: Sequence[str], entry: CpLabeledDatasetEntry) -> bool:
        haystack = f"{statement} {solution}".lower()
        tag_set = {tag.lower() for tag in tags}
        if entry.include_any_tags:
            include_tags = set(entry.include_any_tags)
            if not (tag_set & include_tags):
                return False
        if entry.include_text_terms:
            if not any(term in haystack for term in entry.include_text_terms):
                return False
        if entry.exclude_text_terms:
            if any(term in haystack for term in entry.exclude_text_terms):
                return False
        return True

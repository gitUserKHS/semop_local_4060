from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_QUERY_FIELDS = (
    "query",
    "question",
    "instruction",
    "input",
    "prompt",
    "text",
    "utterance",
)
DEFAULT_CONTEXT_FIELDS = (
    "context",
    "scenario",
    "background",
    "description",
)
DEFAULT_ANSWER_FIELDS = (
    "answer",
    "output",
    "response",
    "label",
    "target",
)


@dataclass
class DatasetExample:
    query: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PublicDatasetAdapter:
    def __init__(
        self,
        query_fields: Sequence[str] | None = None,
        context_fields: Sequence[str] | None = None,
        answer_fields: Sequence[str] | None = None,
    ):
        self.query_fields = tuple(query_fields or DEFAULT_QUERY_FIELDS)
        self.context_fields = tuple(context_fields or DEFAULT_CONTEXT_FIELDS)
        self.answer_fields = tuple(answer_fields or DEFAULT_ANSWER_FIELDS)

    def load_paths(self, paths: Sequence[str | Path]) -> List[DatasetExample]:
        examples: List[DatasetExample] = []
        for path in paths:
            examples.extend(self.load_path(path))
        return examples

    def load_path(self, path: str | Path) -> List[DatasetExample]:
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()
        source = path_obj.stem
        if suffix == ".jsonl":
            return self._load_jsonl(path_obj, source)
        if suffix == ".json":
            return self._load_json(path_obj, source)
        if suffix == ".csv":
            return self._load_csv(path_obj, source)
        if suffix == ".txt":
            return self._load_txt(path_obj, source)
        raise ValueError(f"Unsupported dataset file: {path_obj}")

    def _load_jsonl(self, path: Path, source: str) -> List[DatasetExample]:
        examples: List[DatasetExample] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                item = json.loads(line)
                example = self._example_from_item(item, source)
                if example is not None:
                    examples.append(example)
        return examples

    def _load_json(self, path: Path, source: str) -> List[DatasetExample]:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        items: List[Any]
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            for key in ["data", "examples", "items", "records", "train", "questions"]:
                if isinstance(payload.get(key), list):
                    items = payload[key]
                    break
            else:
                items = [payload]
        else:
            return []
        examples: List[DatasetExample] = []
        for item in items:
            example = self._example_from_item(item, source)
            if example is not None:
                examples.append(example)
        return examples

    def _load_csv(self, path: Path, source: str) -> List[DatasetExample]:
        examples: List[DatasetExample] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                example = self._example_from_item(dict(row), source)
                if example is not None:
                    examples.append(example)
        return examples

    def _load_txt(self, path: Path, source: str) -> List[DatasetExample]:
        examples: List[DatasetExample] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                query = line.strip()
                if query:
                    examples.append(DatasetExample(query=query, source=source, metadata={}))
        return examples

    def _example_from_item(self, item: Any, source: str) -> DatasetExample | None:
        if isinstance(item, str):
            query = item.strip()
            if not query:
                return None
            return DatasetExample(query=query, source=source, metadata={})
        if not isinstance(item, dict):
            return None

        query = self._extract_query(item)
        if not query:
            return None

        metadata: Dict[str, Any] = {}
        answer = self._first_value(item, self.answer_fields)
        if answer:
            metadata["answer"] = answer
        for key in ["id", "category", "task", "domain", "language", "split"]:
            if key in item and item[key] not in (None, ""):
                metadata[key] = item[key]
        return DatasetExample(query=query, source=source, metadata=metadata)

    def _extract_query(self, item: Dict[str, Any]) -> str:
        query = self._first_value(item, self.query_fields)
        context = self._first_value(item, self.context_fields)
        if context and query and context not in query:
            return f"{context.strip()} {query.strip()}".strip()
        if query:
            return query.strip()
        # Fallback: use the first non-empty string field.
        for value in item.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _first_value(item: Dict[str, Any], keys: Sequence[str]) -> str:
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
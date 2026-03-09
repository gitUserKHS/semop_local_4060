from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence
import zipfile


@dataclass
class CpCorpusRecord:
    statement: str
    source_path: str
    source_kind: str
    augmented: bool = False

    def model_dump(self) -> dict:
        return {
            "statement": self.statement,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "augmented": self.augmented,
        }


class CpCorpusBuilder:
    TEXT_FIELDS = ("statement", "query", "question", "prompt", "text", "problem", "title")
    SUPPORTED_SUFFIXES = {".txt", ".jsonl", ".json", ".csv", ".md", ".html", ".htm", ".zip"}

    def build_from_inputs(self, inputs: Sequence[str | Path], augment: bool = True) -> List[CpCorpusRecord]:
        records: List[CpCorpusRecord] = []
        for item in inputs:
            records.extend(self._extract_from_path(Path(item)))
        deduped = self._dedupe(records)
        if augment:
            deduped.extend(self._augment(deduped))
            deduped = self._dedupe(deduped)
        return deduped

    def save_jsonl(self, path: str | Path, records: Iterable[CpCorpusRecord]) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")

    def _extract_from_path(self, path: Path) -> List[CpCorpusRecord]:
        if path.is_dir():
            items: List[CpCorpusRecord] = []
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in self.SUPPORTED_SUFFIXES:
                    items.extend(self._extract_from_path(child))
            return items
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return self._from_text_or_markdown(path)
        if suffix == ".jsonl":
            return self._from_jsonl(path)
        if suffix == ".json":
            return self._from_json(path)
        if suffix == ".csv":
            return self._from_csv(path)
        if suffix in {".html", ".htm"}:
            return self._from_html(path)
        if suffix == ".zip":
            return self._from_zip(path)
        return []

    def _from_text_or_markdown(self, path: Path) -> List[CpCorpusRecord]:
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".md":
            records = self._split_markdown_blocks(text, path)
            if records:
                return records
        records: List[CpCorpusRecord] = []
        for line in text.splitlines():
            normalized = self._normalize(line)
            if normalized:
                records.append(CpCorpusRecord(statement=normalized, source_path=str(path), source_kind=path.suffix.lower().lstrip(".")))
        return records

    def _from_jsonl(self, path: Path) -> List[CpCorpusRecord]:
        records: List[CpCorpusRecord] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                statement = self._extract_text_from_payload(payload)
                if statement:
                    records.append(CpCorpusRecord(statement=statement, source_path=str(path), source_kind="jsonl"))
        return records

    def _from_json(self, path: Path) -> List[CpCorpusRecord]:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        records: List[CpCorpusRecord] = []
        if isinstance(payload, list):
            for item in payload:
                statement = self._extract_text_from_payload(item)
                if statement:
                    records.append(CpCorpusRecord(statement=statement, source_path=str(path), source_kind="json"))
        elif isinstance(payload, dict):
            statement = self._extract_text_from_payload(payload)
            if statement:
                records.append(CpCorpusRecord(statement=statement, source_path=str(path), source_kind="json"))
            else:
                for value in payload.values():
                    if isinstance(value, list):
                        for item in value:
                            statement = self._extract_text_from_payload(item)
                            if statement:
                                records.append(CpCorpusRecord(statement=statement, source_path=str(path), source_kind="json"))
        return records

    def _from_csv(self, path: Path) -> List[CpCorpusRecord]:
        records: List[CpCorpusRecord] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                statement = self._extract_text_from_payload(row)
                if statement:
                    records.append(CpCorpusRecord(statement=statement, source_path=str(path), source_kind="csv"))
        return records

    def _from_html(self, path: Path) -> List[CpCorpusRecord]:
        text = path.read_text(encoding="utf-8-sig")
        stripped = re.sub(r"<script.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        stripped = re.sub(r"<style.*?</style>", " ", stripped, flags=re.IGNORECASE | re.DOTALL)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        candidates = re.split(r"(?:input|output|examples?|constraints?)\s*:?", stripped, flags=re.IGNORECASE)
        records: List[CpCorpusRecord] = []
        for item in candidates[:2]:
            normalized = self._normalize(item)
            if len(normalized) >= 30:
                records.append(CpCorpusRecord(statement=normalized, source_path=str(path), source_kind="html"))
        return self._dedupe(records)

    def _from_zip(self, path: Path) -> List[CpCorpusRecord]:
        records: List[CpCorpusRecord] = []
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                suffix = Path(name).suffix.lower()
                if suffix not in self.SUPPORTED_SUFFIXES - {".zip"}:
                    continue
                raw = archive.read(name)
                text = raw.decode("utf-8-sig", errors="ignore")
                pseudo = Path(name)
                if suffix in {".txt", ".md"}:
                    if suffix == ".md":
                        split_records = self._split_markdown_blocks(text, pseudo, source_kind="zip_md")
                        if split_records:
                            records.extend(split_records)
                            continue
                    for line in text.splitlines():
                        normalized = self._normalize(line)
                        if normalized:
                            records.append(CpCorpusRecord(statement=normalized, source_path=f"{path}:{name}", source_kind=f"zip_{suffix.lstrip('.')}") )
                elif suffix == ".jsonl":
                    for line in text.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        payload = json.loads(line)
                        statement = self._extract_text_from_payload(payload)
                        if statement:
                            records.append(CpCorpusRecord(statement=statement, source_path=f"{path}:{name}", source_kind="zip_jsonl"))
                elif suffix == ".json":
                    payload = json.loads(text)
                    source = f"{path}:{name}"
                    if isinstance(payload, list):
                        for item in payload:
                            statement = self._extract_text_from_payload(item)
                            if statement:
                                records.append(CpCorpusRecord(statement=statement, source_path=source, source_kind="zip_json"))
                elif suffix == ".csv":
                    reader = csv.DictReader(text.splitlines())
                    for row in reader:
                        statement = self._extract_text_from_payload(row)
                        if statement:
                            records.append(CpCorpusRecord(statement=statement, source_path=f"{path}:{name}", source_kind="zip_csv"))
                elif suffix in {".html", ".htm"}:
                    html_path = Path(name)
                    temp = self._from_html_like_text(text, f"{path}:{name}")
                    records.extend(temp)
        return self._dedupe(records)

    def _from_html_like_text(self, text: str, source_path: str) -> List[CpCorpusRecord]:
        stripped = re.sub(r"<script.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        stripped = re.sub(r"<style.*?</style>", " ", stripped, flags=re.IGNORECASE | re.DOTALL)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        normalized = self._normalize(stripped)
        if len(normalized) >= 30:
            return [CpCorpusRecord(statement=normalized, source_path=source_path, source_kind="zip_html")]
        return []

    def _split_markdown_blocks(self, text: str, path: Path, source_kind: str | None = None) -> List[CpCorpusRecord]:
        chunks = re.split(r"^#{1,3}\s+", text, flags=re.MULTILINE)
        records: List[CpCorpusRecord] = []
        for chunk in chunks:
            normalized = self._normalize(chunk)
            if len(normalized) >= 30 and any(token in normalized.lower() for token in ("query", "graph", "array", "grid", "weight", "minimum", "maximum", "path")):
                records.append(CpCorpusRecord(statement=normalized, source_path=str(path), source_kind=source_kind or "md"))
        return self._dedupe(records)

    def _extract_text_from_payload(self, payload: object) -> str:
        if isinstance(payload, str):
            return self._normalize(payload)
        if not isinstance(payload, dict):
            return ""
        for field in self.TEXT_FIELDS:
            value = payload.get(field)
            if isinstance(value, str):
                normalized = self._normalize(value)
                if normalized:
                    return normalized
        return ""

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.replace("\ufeff", " ").split()).strip()

    def _dedupe(self, records: Iterable[CpCorpusRecord]) -> List[CpCorpusRecord]:
        seen: set[str] = set()
        deduped: List[CpCorpusRecord] = []
        for record in records:
            key = record.statement.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    def _augment(self, records: Sequence[CpCorpusRecord]) -> List[CpCorpusRecord]:
        augmented: List[CpCorpusRecord] = []
        replacements = (
            ("Given", "You are given"),
            ("output", "print"),
            ("answer", "compute"),
            ("for each query", "for every query"),
            ("minimum", "smallest"),
            ("maximum", "largest"),
            ("Support", "Process"),
            ("Find", "Compute"),
        )
        for record in records:
            statement = record.statement
            for src, dst in replacements:
                if src in statement:
                    variant = statement.replace(src, dst, 1)
                    normalized = self._normalize(variant)
                    if normalized and normalized != statement:
                        augmented.append(
                            CpCorpusRecord(
                                statement=normalized,
                                source_path=record.source_path,
                                source_kind=f"{record.source_kind}_aug",
                                augmented=True,
                            )
                        )
                        break
        return augmented

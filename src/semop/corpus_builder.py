from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Dict, List, Sequence

from .domain_templates import generate_domain_variants, infer_tags


@dataclass
class CorpusRecord:
    query: str
    normalized_query: str
    source: str
    split: str
    tags: List[str] = field(default_factory=list)
    augmented: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class CorpusBuilder:
    def __init__(self, train_ratio: float = 0.8):
        self.train_ratio = train_ratio

    def build_from_paths(self, paths: Sequence[str | Path], augment: bool = True) -> List[CorpusRecord]:
        queries: List[tuple[str, str]] = []
        for path in paths:
            path_obj = Path(path)
            source = path_obj.stem
            queries.extend((query, source) for query in self._load_queries(path_obj))
        return self.build_from_queries(queries, augment=augment)

    def build_from_queries(self, queries: Sequence[tuple[str, str]], augment: bool = True) -> List[CorpusRecord]:
        records: Dict[str, CorpusRecord] = {}
        for query, source in queries:
            self._register_record(records, query, source, augmented=False)
            if augment:
                for variant in self._augment_query(query):
                    self._register_record(records, variant, f"{source}:aug", augmented=True)
        return list(records.values())

    def save_jsonl(self, records: Sequence[CorpusRecord], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def _register_record(self, records: Dict[str, CorpusRecord], query: str, source: str, augmented: bool) -> None:
        normalized = self.normalize_query(query)
        if not normalized or normalized in records:
            return
        records[normalized] = CorpusRecord(
            query=query.strip(),
            normalized_query=normalized,
            source=source,
            split=self._assign_split(normalized),
            tags=infer_tags(query),
            augmented=augmented,
        )

    def _assign_split(self, normalized_query: str) -> str:
        bucket = int(hashlib.sha256(normalized_query.encode("utf-8")).hexdigest(), 16) % 100
        return "train" if bucket < int(self.train_ratio * 100) else "test"

    def _augment_query(self, query: str) -> List[str]:
        variants = list(generate_domain_variants(query))
        if "어떻게" in query and "하나요" in query:
            variants.append(query.replace("어떻게", "어떤 방식으로"))
        if "무엇" in query:
            variants.append(query.replace("무엇", "어떤 조건"))
        if query.endswith("?"):
            variants.append(query.rstrip("?") + " 다른 대안도 있나요?")
        cleaned = [variant.strip() for variant in variants if variant.strip() and variant.strip() != query.strip()]
        return list(dict.fromkeys(cleaned))

    @staticmethod
    def _load_queries(path: Path) -> List[str]:
        queries: List[str] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    item = json.loads(line)
                    query = item.get("query") or item.get("text") or item.get("input")
                    if query:
                        queries.append(query)
                else:
                    queries.append(line)
        return queries

    @staticmethod
    def normalize_query(query: str) -> str:
        compact = re.sub(r"\s+", " ", query.strip().lower())
        return compact.strip(" ?!.,")

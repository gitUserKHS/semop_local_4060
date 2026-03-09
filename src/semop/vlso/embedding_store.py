from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class VisualEmbeddingRecord:
    key: str
    label: str
    vector: List[float]
    metadata: Dict[str, object]

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class VisualEmbeddingMatch:
    key: str
    label: str
    score: float
    metadata: Dict[str, object]

    def model_dump(self) -> dict:
        return asdict(self)


class VisualEmbeddingStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Iterable[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS visual_embeddings (
                    key TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def upsert(self, record: VisualEmbeddingRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO visual_embeddings (key, label, vector_json, metadata_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    label=excluded.label,
                    vector_json=excluded.vector_json,
                    metadata_json=excluded.metadata_json
                """,
                (record.key, record.label, json.dumps(record.vector), json.dumps(record.metadata, ensure_ascii=False)),
            )
            conn.commit()

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM visual_embeddings").fetchone()
        return int(row[0]) if row else 0

    def search(self, query_vector: List[float], limit: int = 5) -> List[VisualEmbeddingMatch]:
        matches: List[VisualEmbeddingMatch] = []
        with self._connect() as conn:
            rows = conn.execute("SELECT key, label, vector_json, metadata_json FROM visual_embeddings").fetchall()
        for key, label, vector_json, metadata_json in rows:
            vector = json.loads(vector_json)
            score = self._cosine(query_vector, vector)
            matches.append(
                VisualEmbeddingMatch(
                    key=key,
                    label=label,
                    score=round(score, 6),
                    metadata=json.loads(metadata_json),
                )
            )
        matches.sort(key=lambda item: (-item.score, item.key))
        return matches[:limit]

    @staticmethod
    def _cosine(left: List[float], right: List[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
        right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
        return dot / (left_norm * right_norm)

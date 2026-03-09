from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class VisualConceptRecord:
    key: str
    label: str
    feature_vector: Dict[str, float]
    metadata: Dict[str, object]

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class VisualConceptMatch:
    key: str
    label: str
    score: float
    metadata: Dict[str, object]

    def model_dump(self) -> dict:
        return asdict(self)


class VisualConceptMemory:
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
                CREATE TABLE IF NOT EXISTS visual_concepts (
                    key TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    feature_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def upsert(self, record: VisualConceptRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO visual_concepts (key, label, feature_json, metadata_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    label=excluded.label,
                    feature_json=excluded.feature_json,
                    metadata_json=excluded.metadata_json
                """,
                (
                    record.key,
                    record.label,
                    json.dumps(record.feature_vector, ensure_ascii=False, sort_keys=True),
                    json.dumps(record.metadata, ensure_ascii=False),
                ),
            )
            conn.commit()

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM visual_concepts").fetchone()
        return int(row[0]) if row else 0

    def search(self, feature_vector: Dict[str, float], limit: int = 5) -> List[VisualConceptMatch]:
        matches: List[VisualConceptMatch] = []
        with self._connect() as conn:
            rows = conn.execute("SELECT key, label, feature_json, metadata_json FROM visual_concepts").fetchall()
        for key, label, feature_json, metadata_json in rows:
            stored = json.loads(feature_json)
            score = self._cosine(feature_vector, stored)
            matches.append(
                VisualConceptMatch(
                    key=key,
                    label=label,
                    score=round(score, 6),
                    metadata=json.loads(metadata_json),
                )
            )
        matches.sort(key=lambda item: (-item.score, item.key))
        return matches[:limit]

    @staticmethod
    def _cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        keys = set(left) | set(right)
        if not keys:
            return 0.0
        dot = sum(float(left.get(key, 0.0)) * float(right.get(key, 0.0)) for key in keys)
        left_norm = math.sqrt(sum(float(left.get(key, 0.0)) ** 2 for key in keys)) or 1.0
        right_norm = math.sqrt(sum(float(right.get(key, 0.0)) ** 2 for key in keys)) or 1.0
        return dot / (left_norm * right_norm)

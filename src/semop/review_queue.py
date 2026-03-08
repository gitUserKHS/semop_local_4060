from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ReviewQueueItem:
    id: int
    domain: str
    scenario: str
    query: str
    reasons: List[str]
    status: str
    answer_text: str = ""
    resolution_note: str = ""


class ReviewQueueStore:
    def __init__(self, db_path: str | Path = "data/ops_review_queue.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    query TEXT NOT NULL,
                    reasons_json TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    kpis_json TEXT NOT NULL,
                    audit_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    resolution_note TEXT NOT NULL DEFAULT '',
                    resolved_at TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(review_items)")}
            if "resolution_note" not in columns:
                conn.execute("ALTER TABLE review_items ADD COLUMN resolution_note TEXT NOT NULL DEFAULT ''")
            if "resolved_at" not in columns:
                conn.execute("ALTER TABLE review_items ADD COLUMN resolved_at TEXT")
            conn.commit()

    def enqueue(
        self,
        *,
        domain: str,
        scenario: str,
        query: str,
        reasons: List[str],
        answer_text: str,
        kpis: Dict[str, Any],
        audit_items: List[Dict[str, Any]],
    ) -> int:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO review_items (domain, scenario, query, reasons_json, answer_text, kpis_json, audit_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    domain,
                    scenario,
                    query,
                    json.dumps(reasons, ensure_ascii=False),
                    answer_text,
                    json.dumps(kpis, ensure_ascii=False),
                    json.dumps(audit_items, ensure_ascii=False),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def fetch_items(self, status: Optional[str] = None, limit: int = 20) -> List[ReviewQueueItem]:
        query = (
            "SELECT id, domain, scenario, query, reasons_json, status, answer_text, resolution_note "
            "FROM review_items"
        )
        params: List[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            ReviewQueueItem(
                id=int(row[0]),
                domain=row[1],
                scenario=row[2],
                query=row[3],
                reasons=json.loads(row[4]),
                status=row[5],
                answer_text=row[6],
                resolution_note=row[7] or "",
            )
            for row in rows
        ]

    def fetch_pending(self, limit: int = 20) -> List[ReviewQueueItem]:
        return self.fetch_items(status="pending", limit=limit)

    def fetch_item_detail(self, item_id: int) -> Dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT id, domain, scenario, query, reasons_json, answer_text, kpis_json, audit_json, status, resolution_note
                FROM review_items WHERE id = ?
                """,
                (item_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row[0]),
            "domain": row[1],
            "scenario": row[2],
            "query": row[3],
            "reasons": json.loads(row[4]),
            "answer_text": row[5],
            "kpis": json.loads(row[6]),
            "audit_items": json.loads(row[7]),
            "status": row[8],
            "resolution_note": row[9] or "",
        }

    def update_status(self, item_id: int, status: str, resolution_note: str = "") -> None:
        if status not in {"pending", "approved", "rejected", "needs_followup"}:
            raise ValueError(f"Unsupported review status: {status}")
        with closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE review_items
                SET status = ?, resolution_note = ?, resolved_at = CASE WHEN ? = 'pending' THEN NULL ELSE CURRENT_TIMESTAMP END
                WHERE id = ?
                """,
                (status, resolution_note, status, item_id),
            )
            conn.commit()

    def fetch_stats(self) -> Dict[str, int]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT status, COUNT(*) FROM review_items GROUP BY status").fetchall()
        stats = {"pending": 0, "approved": 0, "rejected": 0, "needs_followup": 0}
        for status, count in rows:
            stats[str(status)] = int(count)
        stats["total"] = sum(stats.values())
        return stats


def review_reasons_from_kpis(kpis: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if kpis.get("invalid_advice_rate", 0.0) > 0.2:
        reasons.append("invalid_advice_rate")
    if kpis.get("clarification_need_rate", 0.0) > 0.4:
        reasons.append("clarification_need_rate")
    if kpis.get("plan_executability", 1.0) < 0.75:
        reasons.append("low_plan_executability")
    if kpis.get("human_audit_usefulness", 1.0) < 0.7:
        reasons.append("low_audit_usefulness")
    if not reasons and kpis.get("notes"):
        reasons.append("manual_review_note")
    return reasons

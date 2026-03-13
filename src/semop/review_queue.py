from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .operating_policies import resolve_review_severity_weight
from .structures import StructuredMeaningGraph


SEVERITY_LEVELS = ('low', 'medium', 'high', 'critical')


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
    severity: str = 'medium'


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
                    context_text TEXT NOT NULL DEFAULT '',
                    graph_json TEXT NOT NULL DEFAULT '',
                    severity TEXT NOT NULL DEFAULT 'medium',
                    status TEXT NOT NULL DEFAULT 'pending',
                    resolution_note TEXT NOT NULL DEFAULT '',
                    resolved_at TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(review_items)")}
            if "context_text" not in columns:
                conn.execute("ALTER TABLE review_items ADD COLUMN context_text TEXT NOT NULL DEFAULT ''")
            if "graph_json" not in columns:
                conn.execute("ALTER TABLE review_items ADD COLUMN graph_json TEXT NOT NULL DEFAULT ''")
            if "severity" not in columns:
                conn.execute("ALTER TABLE review_items ADD COLUMN severity TEXT NOT NULL DEFAULT 'medium'")
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
        context_text: str = '',
        graph_payload: Dict[str, Any] | None = None,
        severity: str = 'medium',
    ) -> int:
        normalized_severity = normalize_review_severity(severity)
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO review_items (domain, scenario, query, reasons_json, answer_text, kpis_json, audit_json, context_text, graph_json, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    domain,
                    scenario,
                    query,
                    json.dumps(reasons, ensure_ascii=False),
                    answer_text,
                    json.dumps(kpis, ensure_ascii=False),
                    json.dumps(audit_items, ensure_ascii=False),
                    context_text,
                    json.dumps(graph_payload, ensure_ascii=False) if graph_payload is not None else '',
                    normalized_severity,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def fetch_items(self, status: Optional[str] = None, limit: int = 20) -> List[ReviewQueueItem]:
        query = (
            "SELECT id, domain, scenario, query, reasons_json, status, answer_text, resolution_note, severity "
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
                severity=normalize_review_severity(row[8] or 'medium'),
            )
            for row in rows
        ]

    def fetch_pending(self, limit: int = 20) -> List[ReviewQueueItem]:
        return self.fetch_items(status="pending", limit=limit)

    def fetch_item_detail(self, item_id: int) -> Dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT id, domain, scenario, query, reasons_json, answer_text, kpis_json, audit_json, context_text, graph_json, severity, status, resolution_note
                FROM review_items WHERE id = ?
                """,
                (item_id,),
            ).fetchone()
        if row is None:
            return None
        graph_payload = json.loads(row[9]) if row[9] else None
        severity = normalize_review_severity(row[10] or 'medium')
        return {
            "id": int(row[0]),
            "domain": row[1],
            "scenario": row[2],
            "query": row[3],
            "reasons": json.loads(row[4]),
            "answer_text": row[5],
            "kpis": json.loads(row[6]),
            "audit_items": json.loads(row[7]),
            "context_text": row[8] or '',
            "graph": graph_payload,
            "severity": severity,
            "severity_weight": severity_weight(severity, row[1], row[2]),
            "status": row[11],
            "resolution_note": row[12] or "",
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


def normalize_review_severity(value: str | None) -> str:
    normalized = str(value or 'medium').strip().lower()
    if normalized not in SEVERITY_LEVELS:
        return 'medium'
    return normalized


def severity_weight(value: str | None, domain: str | None = None, scenario: str | None = None) -> float:
    return resolve_review_severity_weight(domain, scenario, normalize_review_severity(value))


def infer_review_severity(
    domain: str,
    scenario: str,
    reasons: List[str],
    kpis: Dict[str, Any] | None = None,
) -> str:
    kpis = dict(kpis or {})
    reason_set = {str(item).strip() for item in reasons if str(item).strip()}
    severity_index = 1
    if str(domain).strip() == 'warehouse_exception':
        severity_index = max(severity_index, 2)
    if str(scenario).strip() in {'exception_response', 'quality_gate'}:
        severity_index = max(severity_index, 2)
    if 'invalid_advice_rate' in reason_set or float(kpis.get('invalid_advice_rate', 0.0) or 0.0) >= 0.4:
        severity_index = max(severity_index, 3)
    elif 'repair_failure' in reason_set and ({'grounding_review', 'claim_grounding_review', 'compiler_validity_gap'} & reason_set):
        severity_index = max(severity_index, 3)
    elif {'repair_failure', 'compiler_validity_gap', 'grounding_review', 'claim_grounding_review'} & reason_set:
        severity_index = max(severity_index, 2)
    elif {'context_misread_review', 'relation_recovery_gap', 'low_plan_executability'} & reason_set:
        severity_index = max(severity_index, 1)
    return SEVERITY_LEVELS[severity_index]


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
    if kpis.get("context_misread_rate", 0.0) > 0.25:
        reasons.append("context_misread_review")
    if kpis.get("relation_recovery", 1.0) < 0.7:
        reasons.append("relation_recovery_gap")
    if not reasons and kpis.get("notes"):
        reasons.append("manual_review_note")
    return _unique_reasons(reasons)


def review_reasons_from_graph(graph: StructuredMeaningGraph) -> List[str]:
    reasons: List[str] = []
    report = graph.operator_execution
    grounded = any(edge.source == "question" and edge.relation == "GROUNDED_BY" for edge in graph.edges)
    has_visual_context = any(node.id == 'visual_scene' or node.kind == 'scene' for node in graph.nodes)
    has_document_context = bool(graph.source_context.strip()) or any(result.domain == 'document_grounding' for result in graph.symbolic_results)

    if report is not None:
        findings = [str(item).lower() for item in report.compiler_findings]
        warnings = [str(item).lower() for item in report.warnings]
        unsupported_claims = [item for item in report.claim_groundings if not item.grounded]
        claim_gap = (
            any('claim grounding warning:' in item for item in findings)
            or any('claim grounding risk:' in item for item in warnings)
            or bool(unsupported_claims)
            or (bool(report.claim_groundings) and report.claim_grounding_score < 0.75)
        )
        if any('grounding compiler warning' in item for item in findings + warnings):
            reasons.append('grounding_review')
        if claim_gap:
            reasons.append('claim_grounding_review')
            reasons.append('grounding_review')
        if any(item.startswith('compiler warning:') for item in findings) or report.composition_score < 0.75:
            reasons.append('compiler_validity_gap')
        repairs_applied = any(str(item).startswith('repair_applied:') for item in report.derived_decisions)
        if report.counterexample_repairs and not repairs_applied:
            reasons.append('repair_failure')
    if (has_document_context or has_visual_context) and not grounded:
        reasons.append('grounding_review')
    if graph.clarification_needed and graph.clarification_score >= 0.5:
        reasons.append('clarification_need_rate')
    return _unique_reasons(reasons)


def review_reasons_from_graph_and_kpis(graph: StructuredMeaningGraph, kpis: Dict[str, Any]) -> List[str]:
    return _unique_reasons(review_reasons_from_kpis(kpis) + review_reasons_from_graph(graph))


def _unique_reasons(reasons: List[str]) -> List[str]:
    return list(dict.fromkeys(str(reason).strip() for reason in reasons if str(reason).strip()))





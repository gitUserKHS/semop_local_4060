from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from .structures import StructuredMeaningGraph

if TYPE_CHECKING:
    from .corpus_learning import CorpusLearningResult
    from .operator_hierarchy import OperatorHierarchyResult


@dataclass
class StoredExample:
    query: str
    normalized_query: str
    source: str
    split: str
    intent: str
    graph: StructuredMeaningGraph


class CorpusMemoryStore:
    def __init__(self, db_path: str | Path = "data/semop_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    source TEXT NOT NULL,
                    split TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    graph_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(normalized_query, source, split)
                );

                CREATE TABLE IF NOT EXISTS learning_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    corpus_size INTEGER NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS hierarchy_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    split TEXT NOT NULL,
                    corpus_size INTEGER NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS hierarchy_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    level TEXT NOT NULL,
                    name TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES hierarchy_runs(id)
                );
                """
            )
            conn.commit()

    def upsert_graph(self, graph: StructuredMeaningGraph, source: str = "manual", split: str = "train") -> None:
        normalized = self.normalize_query(graph.query)
        payload = json.dumps(graph.model_dump(), ensure_ascii=False)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO examples (query, normalized_query, source, split, intent, graph_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(normalized_query, source, split)
                DO UPDATE SET intent=excluded.intent, graph_json=excluded.graph_json, query=excluded.query
                """,
                (graph.query, normalized, source, split, graph.intent, payload),
            )
            conn.commit()

    def store_learning_result(self, result: "CorpusLearningResult", source: str = "manual") -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO learning_runs (source, corpus_size, summary_json) VALUES (?, ?, ?)",
                (source, result.corpus_size, result.to_json(indent=2)),
            )
            conn.commit()

    def store_hierarchy_result(self, result: "OperatorHierarchyResult", source: str = "manual", split: str = "train") -> int:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "INSERT INTO hierarchy_runs (source, split, corpus_size, summary_json) VALUES (?, ?, ?, ?)",
                (source, split, result.corpus_size, result.to_json(indent=2)),
            )
            run_id = int(cursor.lastrowid)
            for node in [*result.micro_nodes, *result.family_nodes, *result.abstract_nodes]:
                conn.execute(
                    "INSERT INTO hierarchy_nodes (run_id, level, name, payload_json) VALUES (?, ?, ?, ?)",
                    (run_id, node.level, node.name, json.dumps(node.__dict__, ensure_ascii=False)),
                )
            conn.commit()
        return run_id

    def fetch_graphs(self, split: Optional[str] = None, source: Optional[str] = None) -> List[StructuredMeaningGraph]:
        query = "SELECT graph_json FROM examples"
        clauses = []
        params: List[str] = []
        if split is not None:
            clauses.append("split = ?")
            params.append(split)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id"
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [StructuredMeaningGraph.from_dict(json.loads(row[0])) for row in rows]

    def fetch_queries(self, split: Optional[str] = None, source: Optional[str] = None) -> List[str]:
        query = "SELECT query FROM examples"
        clauses = []
        params: List[str] = []
        if split is not None:
            clauses.append("split = ?")
            params.append(split)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id"
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [row[0] for row in rows]

    def fetch_latest_hierarchy_summary(self, source: Optional[str] = None, split: Optional[str] = None) -> dict | None:
        query = "SELECT summary_json FROM hierarchy_runs"
        clauses = []
        params: List[str] = []
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if split is not None:
            clauses.append("split = ?")
            params.append(split)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT 1"
        with closing(self._connect()) as conn:
            row = conn.execute(query, params).fetchone()
        return json.loads(row[0]) if row else None

    def count_examples(self, split: Optional[str] = None, source: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) FROM examples"
        clauses = []
        params: List[str] = []
        if split is not None:
            clauses.append("split = ?")
            params.append(split)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with closing(self._connect()) as conn:
            return int(conn.execute(query, params).fetchone()[0])

    def search_similar_graphs(self, query: str, split: Optional[str] = "train", source: Optional[str] = None, top_k: int = 3) -> List[StructuredMeaningGraph]:
        from .memory_retrieval import QueryEmbeddingIndex

        graphs = self.fetch_graphs(split=split, source=source)
        if not graphs:
            return []

        index = QueryEmbeddingIndex()
        hits = index.search(query, graphs, top_k=top_k)
        if not hits:
            return []

        graph_map = {graph.query: graph for graph in graphs}
        ordered: List[StructuredMeaningGraph] = []
        for hit in hits:
            graph = graph_map.get(hit.query)
            if graph is not None:
                ordered.append(graph)
        return ordered

    @staticmethod
    def normalize_query(query: str) -> str:
        return " ".join(query.strip().lower().split())

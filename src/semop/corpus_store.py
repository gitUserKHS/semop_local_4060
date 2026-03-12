from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from .commonsense_kb import concept_family, lookup_concept, match_concepts, scripts_for_concept
from .structures import Edge, Node, OperatorCandidate, PremiseCandidate, StructuredMeaningGraph

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

                CREATE TABLE IF NOT EXISTS premise_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    normalized_query TEXT NOT NULL,
                    graph_query TEXT NOT NULL,
                    source TEXT NOT NULL,
                    split TEXT NOT NULL,
                    hidden_goal TEXT NOT NULL,
                    premise TEXT NOT NULL,
                    candidate_type TEXT NOT NULL,
                    support_score REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(normalized_query, source, split, hidden_goal, premise, candidate_type)
                );

                CREATE TABLE IF NOT EXISTS operator_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    normalized_query TEXT NOT NULL,
                    graph_query TEXT NOT NULL,
                    source TEXT NOT NULL,
                    split TEXT NOT NULL,
                    operator_name TEXT NOT NULL,
                    operator_family TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(normalized_query, source, split, operator_name, operator_family)
                );


                CREATE TABLE IF NOT EXISTS script_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    normalized_query TEXT NOT NULL,
                    graph_query TEXT NOT NULL,
                    source TEXT NOT NULL,
                    split TEXT NOT NULL,
                    script_step TEXT NOT NULL,
                    support_score REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(normalized_query, source, split, script_step)
                );

                CREATE TABLE IF NOT EXISTS evolved_operator_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    split TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS evolved_operator_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    operator_name TEXT NOT NULL,
                    retained INTEGER NOT NULL,
                    utility_score REAL NOT NULL,
                    support INTEGER NOT NULL,
                    domain_support INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES evolved_operator_runs(id)
                );

                CREATE TABLE IF NOT EXISTS operator_transfer_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    split TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    evolution_run_id INTEGER,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(evolution_run_id) REFERENCES evolved_operator_runs(id)
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


    def upsert_premise_operator_memory(self, graph: StructuredMeaningGraph, source: str = "manual", split: str = "train") -> None:
        normalized = self.normalize_query(graph.query)
        with closing(self._connect()) as conn:
            for candidate in graph.premise_candidates:
                conn.execute(
                    """
                    INSERT INTO premise_memories (normalized_query, graph_query, source, split, hidden_goal, premise, candidate_type, support_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(normalized_query, source, split, hidden_goal, premise, candidate_type)
                    DO UPDATE SET graph_query=excluded.graph_query, support_score=excluded.support_score
                    """,
                    (
                        normalized,
                        graph.query,
                        source,
                        split,
                        candidate.hidden_goal,
                        candidate.premise,
                        candidate.candidate_type,
                        float(candidate.support_score),
                    ),
                )
            for operator in graph.induced_operators:
                conn.execute(
                    """
                    INSERT INTO operator_memories (normalized_query, graph_query, source, split, operator_name, operator_family, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(normalized_query, source, split, operator_name, operator_family)
                    DO UPDATE SET graph_query=excluded.graph_query, confidence=excluded.confidence
                    """,
                    (
                        normalized,
                        graph.query,
                        source,
                        split,
                        operator.name,
                        operator.family,
                        float(operator.confidence),
                    ),
                )
            for script_step in graph.inferred_scripts:
                conn.execute(
                    """
                    INSERT INTO script_memories (normalized_query, graph_query, source, split, script_step, support_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(normalized_query, source, split, script_step)
                    DO UPDATE SET graph_query=excluded.graph_query, support_score=excluded.support_score
                    """,
                    (
                        normalized,
                        graph.query,
                        source,
                        split,
                        script_step,
                        0.66,
                    ),
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


    def store_operator_evolution_result(self, result, source: str = "manual", split: str = "train", iteration: int = 1) -> int:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "INSERT INTO evolved_operator_runs (source, split, iteration, summary_json) VALUES (?, ?, ?, ?)",
                (source, split, iteration, json.dumps(result.model_dump(), ensure_ascii=False, indent=2)),
            )
            run_id = int(cursor.lastrowid)
            for proposal in result.proposals:
                conn.execute(
                    "INSERT INTO evolved_operator_proposals (run_id, operator_name, retained, utility_score, support, domain_support, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        proposal.name,
                        1 if proposal.retained else 0,
                        float(proposal.utility_score),
                        int(proposal.support),
                        int(proposal.domain_support),
                        json.dumps(proposal.model_dump(), ensure_ascii=False),
                    ),
                )
            conn.commit()
        return run_id

    def store_operator_transfer_summary(self, summary, source: str = "manual", split: str = "train", iteration: int = 1, evolution_run_id: int | None = None) -> int:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "INSERT INTO operator_transfer_runs (source, split, iteration, evolution_run_id, summary_json) VALUES (?, ?, ?, ?, ?)",
                (source, split, iteration, evolution_run_id, json.dumps(summary.model_dump(), ensure_ascii=False, indent=2)),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def seed_evolved_operator_memory(self, proposals: List, source: str = "manual", split: str = "train") -> int:
        inserted = 0
        with closing(self._connect()) as conn:
            for proposal in proposals:
                conn.execute(
                    """
                    INSERT INTO operator_memories (normalized_query, graph_query, source, split, operator_name, operator_family, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(normalized_query, source, split, operator_name, operator_family)
                    DO UPDATE SET confidence=excluded.confidence, graph_query=excluded.graph_query
                    """,
                    (
                        self.normalize_query(proposal.name),
                        f"Evolved operator from basis: {', '.join(proposal.basis_signature)}",
                        source,
                        split,
                        proposal.name,
                        'evolved_operator',
                        float(proposal.utility_score),
                    ),
                )
                inserted += 1
            conn.commit()
        return inserted

    def fetch_latest_operator_evolution_summary(self, source: Optional[str] = None, split: Optional[str] = None) -> dict | None:
        query = "SELECT summary_json FROM evolved_operator_runs"
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

    def fetch_latest_operator_transfer_summary(self, source: Optional[str] = None, split: Optional[str] = None) -> dict | None:
        query = "SELECT summary_json FROM operator_transfer_runs"
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

    def fetch_retained_evolved_operator_rows(
        self,
        source: Optional[str] = None,
        split: Optional[str] = None,
        limit: int = 12,
    ) -> List[dict]:
        sql = """
            SELECT p.operator_name, p.utility_score, p.payload_json
            FROM evolved_operator_proposals AS p
            JOIN evolved_operator_runs AS r ON r.id = p.run_id
            WHERE p.retained = 1
        """
        clauses = []
        params: List[str] = []
        if source is not None:
            clauses.append("r.source = ?")
            params.append(source)
        if split is not None:
            clauses.append("r.split = ?")
            params.append(split)
        if clauses:
            sql += " AND " + " AND ".join(clauses)
        sql += " ORDER BY p.utility_score DESC, p.id DESC LIMIT ?"
        params.append(int(limit))
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        payloads: List[dict] = []
        for operator_name, utility_score, payload_json in rows:
            payload = json.loads(payload_json)
            payload['operator_name'] = operator_name
            payload['utility_score'] = float(utility_score)
            payloads.append(payload)
        return payloads

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

    def fetch_latest_learning_summary(self, source: Optional[str] = None) -> dict | None:
        query = "SELECT summary_json FROM learning_runs"
        params: List[str] = []
        if source is not None:
            query += " WHERE source = ?"
            params.append(source)
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



    def collect_script_compatibility_profile(self, query: str, premise: str, hidden_goal: str = '', source: Optional[str] = None, split: Optional[str] = None) -> dict:
        normalized = self.normalize_query(query)
        query_terms = set(normalized.split())
        query_concepts = set(match_concepts(query))
        query_concept_families = {concept_family(item) for item in query_concepts}
        query_scripts = {step for concept in query_concepts for step in scripts_for_concept(concept)}
        premise_hints = self._premise_script_hints(premise)
        premise_family_hint = self._premise_family_hint(premise, hidden_goal)
        evolved_rows = self.fetch_retained_evolved_operator_rows(source=source, split=split, limit=16)
        with closing(self._connect()) as conn:
            clauses = []
            params: List[str] = []
            premise_sql = "SELECT graph_query, hidden_goal, premise, support_score, normalized_query FROM premise_memories"
            script_sql = "SELECT graph_query, script_step, support_score, normalized_query FROM script_memories"
            operator_sql = "SELECT graph_query, operator_name, operator_family, confidence, normalized_query FROM operator_memories"
            if source is not None:
                clauses.append("source = ?")
                params.append(source)
            if split is not None:
                clauses.append("split = ?")
                params.append(split)
            if clauses:
                where = " WHERE " + " AND ".join(clauses)
                premise_sql += where
                script_sql += where
                operator_sql += where
            premise_rows = conn.execute(premise_sql, params).fetchall()
            script_rows = conn.execute(script_sql, params).fetchall()
            operator_rows = conn.execute(operator_sql, params).fetchall()

        premise_hits = 0
        goal_hits = 0
        concept_scores: List[float] = []
        family_scores: List[float] = []
        script_scores: List[float] = []
        operator_scores: List[float] = []
        evolved_scores: List[float] = []

        for graph_query, stored_goal, stored_premise, support_score, stored_normalized in premise_rows:
            if str(stored_premise) != premise:
                continue
            premise_hits += 1
            if hidden_goal and str(stored_goal) == hidden_goal:
                goal_hits += 1
            stored_terms = set(str(stored_normalized).split())
            lexical = len(query_terms & stored_terms) / float(max(1, len(query_terms | stored_terms)))
            graph_concepts = set(match_concepts(str(graph_query)))
            graph_families = {concept_family(item) for item in graph_concepts}
            concept_overlap = len(query_concepts & graph_concepts) / float(max(1, len(query_concepts | graph_concepts))) if (query_concepts or graph_concepts) else 0.0
            family_overlap = len(query_concept_families & graph_families) / float(max(1, len(query_concept_families | graph_families))) if (query_concept_families or graph_families) else 0.0
            if premise_family_hint and premise_family_hint in graph_families:
                family_overlap = max(family_overlap, 0.75)
            concept_scores.append(min(1.0, lexical * 0.25 + concept_overlap * 0.5 + family_overlap * 0.25) * float(support_score))
            family_scores.append(family_overlap * float(support_score))

        for graph_query, script_step, support_score, stored_normalized in script_rows:
            step = str(script_step)
            matched = step in query_scripts or any(hint in step for hint in premise_hints)
            if not matched:
                continue
            stored_terms = set(str(stored_normalized).split())
            lexical = len(query_terms & stored_terms) / float(max(1, len(query_terms | stored_terms)))
            graph_concepts = set(match_concepts(str(graph_query)))
            graph_families = {concept_family(item) for item in graph_concepts}
            concept_overlap = len(query_concepts & graph_concepts) / float(max(1, len(query_concepts | graph_concepts))) if (query_concepts or graph_concepts) else 0.0
            family_overlap = len(query_concept_families & graph_families) / float(max(1, len(query_concept_families | graph_families))) if (query_concept_families or graph_families) else 0.0
            if premise_family_hint and premise_family_hint in graph_families:
                family_overlap = max(family_overlap, 0.7)
            script_scores.append(min(1.0, 0.45 + lexical * 0.1 + concept_overlap * 0.2 + family_overlap * 0.25) * float(support_score))

        for graph_query, operator_name, operator_family, confidence, stored_normalized in operator_rows:
            family_upper = str(operator_family).upper()
            name_upper = str(operator_name).upper()
            relevant = False
            if premise == 'open_access' and ('ACCESS' in family_upper or 'ACCESS' in name_upper or 'CONTROL' in family_upper or 'CONTROL' in name_upper or 'CAPPED' in family_upper or 'CAPPED' in name_upper):
                relevant = True
            elif premise == 'available_space' and ('CONTAINER' in family_upper or 'CONTAINER' in name_upper):
                relevant = True
            elif premise == 'vehicle_present' and ('SERVICE' in family_upper or 'GOAL' in name_upper):
                relevant = True
            elif premise == 'subquadratic_complexity' and ('GOAL_PRESERVATION' in name_upper or 'EFFICIENCY' in family_upper):
                relevant = True
            if not relevant:
                continue
            stored_terms = set(str(stored_normalized).split())
            lexical = len(query_terms & stored_terms) / float(max(1, len(query_terms | stored_terms)))
            operator_scores.append(min(1.0, 0.5 + lexical * 0.2) * float(confidence))

        for payload in evolved_rows:
            family_name = str(payload.get('family', '')).upper()
            operator_name = str(payload.get('name', payload.get('operator_name', ''))).upper()
            basis = {str(item).upper() for item in payload.get('basis_signature', [])}
            support = 0.0
            if premise == 'open_access':
                if {'ACCESS_PORT_OPERATOR', 'ACCESS_CONTROL_OPERATOR'} & basis:
                    support += 0.75
                if 'CAPPED_ACCESS_OPERATOR' in family_name or 'CAPPED' in operator_name:
                    support += 0.85
                if hidden_goal in {'retrieve_item_from_cabinet_goal', 'pour_from_bottle_goal', 'retrieve_item_from_drawer_goal', 'pass_through_door_goal'}:
                    support += 0.2
            elif premise == 'available_space':
                if 'CONTAINER_BODY_OPERATOR' in basis or 'CONTAINER' in family_name:
                    support += 0.7
            elif premise == 'vehicle_present':
                if 'SERVICE' in family_name or 'SERVICE' in operator_name or 'GOAL_PRESERVATION' in operator_name:
                    support += 0.65
            elif premise == 'subquadratic_complexity':
                if 'GOAL_PRESERVATION' in operator_name or 'EFFICIENCY' in family_name:
                    support += 0.7
            if support <= 0.0:
                continue
            evolved_scores.append(min(1.0, support) * float(payload.get('utility_score', 0.0)))

        concept_overlap = sum(concept_scores) / float(len(concept_scores)) if concept_scores else 0.0
        family_overlap = sum(family_scores) / float(len(family_scores)) if family_scores else 0.0
        script_overlap = sum(script_scores) / float(len(script_scores)) if script_scores else 0.0
        operator_support = sum(operator_scores) / float(len(operator_scores)) if operator_scores else 0.0
        evolved_operator_support = sum(evolved_scores) / float(len(evolved_scores)) if evolved_scores else 0.0
        goal_alignment = min(1.0, goal_hits / float(max(1, premise_hits))) if premise_hits and hidden_goal else 0.0
        score = min(0.99, concept_overlap * 0.28 + family_overlap * 0.16 + script_overlap * 0.24 + operator_support * 0.16 + evolved_operator_support * 0.16)
        if premise_hits and hidden_goal:
            score = min(0.99, score + min(0.15, goal_hits * 0.05))
        return {
            'premise': premise,
            'hidden_goal': hidden_goal,
            'hits': premise_hits + len(script_scores) + len(operator_scores),
            'premise_hits': premise_hits,
            'goal_hits': goal_hits,
            'concept_overlap': round(concept_overlap, 4),
            'script_overlap': round(script_overlap, 4),
            'family_overlap': round(family_overlap, 4),
            'operator_support': round(operator_support, 4),
            'evolved_operator_support': round(evolved_operator_support, 4),
            'goal_alignment': round(goal_alignment, 4),
            'score': round(score, 4),
        }

    def build_script_compatibility_training_examples(self, source: Optional[str] = None, split: Optional[str] = None) -> List[dict]:
        clauses = []
        params: List[str] = []
        sql = "SELECT normalized_query, graph_query, hidden_goal, premise FROM premise_memories"
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if split is not None:
            clauses.append("split = ?")
            params.append(split)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with closing(self._connect()) as conn:
            premise_rows = conn.execute(sql, params).fetchall()
            all_premises = [row[3] for row in premise_rows]
        known_premises = sorted({str(item) for item in all_premises})
        examples: List[dict] = []
        for normalized_query, graph_query, hidden_goal, premise in premise_rows:
            positive_profile = self.collect_script_compatibility_profile(
                str(graph_query),
                premise=str(premise),
                hidden_goal=str(hidden_goal),
                source=source,
                split=split,
            )
            examples.append({
                'query': str(graph_query),
                'premise': str(premise),
                'hidden_goal': str(hidden_goal),
                'label': 1.0,
                'features': positive_profile,
            })
            negative_candidates = []
            hard_negative_map = {
                'open_access': ['available_space', 'vehicle_present', 'subquadratic_complexity'],
                'available_space': ['open_access', 'vehicle_present'],
                'vehicle_present': ['open_access', 'goal_clarity'],
                'subquadratic_complexity': ['goal_clarity', 'vehicle_present', 'open_access'],
                'goal_clarity': ['vehicle_present', 'open_access'],
            }
            for alt in hard_negative_map.get(str(premise), []):
                if alt != premise and alt in known_premises and alt not in negative_candidates:
                    negative_candidates.append(alt)
            for alt in known_premises:
                if alt == premise or alt in negative_candidates:
                    continue
                if alt in scripts_for_concept(str(premise)):
                    continue
                negative_candidates.append(alt)
                if len(negative_candidates) >= 3:
                    break
            concept_data = lookup_concept(str(hidden_goal or premise))
            for alt in concept_data.get('requires', []):
                if alt != premise and alt not in negative_candidates:
                    negative_candidates.append(str(alt))
                if len(negative_candidates) >= 3:
                    break
            for alt in negative_candidates[:3]:
                negative_profile = self.collect_script_compatibility_profile(
                    str(graph_query),
                    premise=str(alt),
                    hidden_goal=str(hidden_goal),
                    source=source,
                    split=split,
                )
                examples.append({
                    'query': str(graph_query),
                    'premise': str(alt),
                    'hidden_goal': str(hidden_goal),
                    'label': 0.0,
                    'features': negative_profile,
                })
        return examples

    def seed_visual_review_memories(
        self,
        rows: List[dict],
        source: str = 'vlso_review',
        split: str = 'train',
    ) -> int:
        count = 0
        for row in rows:
            query = str(row.get('query', '')).strip()
            if not query:
                continue
            graph = StructuredMeaningGraph(query=query, intent='visual_review_memory', domain='vlso')
            concepts = [str(item).strip() for item in row.get('concepts', []) if str(item).strip()]
            for concept in concepts:
                graph.add_node(Node(id=concept, label=concept.replace('_', ' '), kind='concept', provenance=['vlso_review_seed']))
            graph.hidden_goals = [str(item).strip() for item in row.get('hidden_goals', []) if str(item).strip()]
            graph.required_premises = [str(item).strip() for item in row.get('required_premises', []) if str(item).strip()]
            graph.satisfied_premises = [str(item).strip() for item in row.get('satisfied_premises', []) if str(item).strip()]
            graph.missing_premises = [str(item).strip() for item in row.get('missing_premises', []) if str(item).strip()]
            graph.inferred_scripts = [str(item).strip() for item in row.get('scripts', []) if str(item).strip()]
            for goal in graph.hidden_goals:
                graph.add_node(Node(id=goal, label=goal.replace('_', ' '), kind='hidden_goal', provenance=['vlso_review_seed']))
                for premise_name in graph.required_premises:
                    graph.add_edge(Edge(source=goal, relation='REQUIRES', target=premise_name, confidence=0.8, provenance=['vlso_review_seed']))
            for premise_name in graph.required_premises:
                graph.premise_candidates.append(PremiseCandidate(premise=premise_name, hidden_goal=graph.hidden_goals[0] if graph.hidden_goals else '', candidate_type='required', source='vlso_review', support_score=0.78, evidence=['Derived from approved visual review clusters.']))
            for operator_name in [str(item).strip() for item in row.get('operators', []) if str(item).strip()]:
                family = operator_name.lower().replace('_operator', '')
                graph.induced_operators.append(OperatorCandidate(name=operator_name, family=family, arity=2, input_types=['visual_structure'], output_type='operator', description='Approved visual-review structural operator seed.', confidence=0.76, provenance=['vlso_review_seed']))
            self.upsert_graph(graph, source=source, split=split)
            self.upsert_premise_operator_memory(graph, source=source, split=split)
            count += 1
        return count

    @staticmethod
    def _premise_script_hints(premise: str) -> set[str]:
        mapping = {
            'open_access': {'open_', 'access', 'unzip', 'unlock', 'uncap', 'remove_cap', 'lift_lid', 'open_drawer', 'open_cabinet', 'open_box'},
            'available_space': {'insert', 'store', 'space', 'contain'},
            'vehicle_present': {'drive_vehicle_to_site', 'wash_vehicle', 'drop_off'},
            'subquadratic_complexity': {'prefix', 'offline', 'precompute', 'range'},
            'goal_clarity': {'book', 'cancel', 'confirm', 'inquiry'},
        }
        return mapping.get(premise, {premise})


    def collect_script_compatibility_training_stats(self, source: Optional[str] = None, split: Optional[str] = None) -> dict:
        with closing(self._connect()) as conn:
            clauses = []
            params: List[str] = []
            if source is not None:
                clauses.append("source = ?")
                params.append(source)
            if split is not None:
                clauses.append("split = ?")
                params.append(split)
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ''
            premise_hits = int(conn.execute("SELECT COUNT(*) FROM premise_memories" + where, params).fetchone()[0])
            script_hits = int(conn.execute("SELECT COUNT(*) FROM script_memories" + where, params).fetchone()[0])
            operator_hits = int(conn.execute("SELECT COUNT(*) FROM operator_memories" + where, params).fetchone()[0])
            if clauses:
                goal_hits = int(conn.execute("SELECT COUNT(*) FROM premise_memories" + where + " AND hidden_goal != ''", params).fetchone()[0])
            else:
                goal_hits = int(conn.execute("SELECT COUNT(*) FROM premise_memories WHERE hidden_goal != ''").fetchone()[0])
        return {
            'premise_hits': premise_hits,
            'script_hits': script_hits,
            'operator_hits': operator_hits,
            'goal_hits': goal_hits,
        }

    def search_premise_support(self, query: str, top_k: int = 6, source: Optional[str] = None, split: Optional[str] = None) -> List[dict]:
        normalized = self.normalize_query(query)
        query_terms = set(normalized.split())
        query_concepts = set(match_concepts(query))
        query_requirements = {req for concept in query_concepts for req in lookup_concept(concept).get('requires', [])}
        query_scripts = {step for concept in query_concepts for step in scripts_for_concept(concept)}
        with closing(self._connect()) as conn:
            clauses = []
            params: List[str] = []
            sql = "SELECT graph_query, hidden_goal, premise, candidate_type, support_score, normalized_query FROM premise_memories"
            if source is not None:
                clauses.append("source = ?")
                params.append(source)
            if split is not None:
                clauses.append("split = ?")
                params.append(split)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            rows = conn.execute(sql, params).fetchall()
        scored: List[dict] = []
        for graph_query, hidden_goal, premise, candidate_type, support_score, stored_normalized in rows:
            stored_terms = set(str(stored_normalized).split())
            lexical = len(query_terms & stored_terms) / float(max(1, len(query_terms | stored_terms)))
            graph_concepts = set(match_concepts(str(graph_query)))
            concept_overlap = len(query_concepts & graph_concepts) / float(max(1, len(query_concepts | graph_concepts))) if (query_concepts or graph_concepts) else 0.0
            compatibility = 0.0
            if premise in query_requirements:
                compatibility += 0.9
            if premise == 'open_access' and query_concepts & {'bag', 'drawer', 'door', 'cabinet', 'box', 'bottle', 'jar', 'bin', 'pouch', 'suitcase'}:
                compatibility += 0.45
            if premise == 'subquadratic_complexity' and any(token in normalized for token in ('range', 'query', '2e5', '10^5', '100000', '200000')):
                compatibility += 0.7
            if hidden_goal and hidden_goal in query_scripts:
                compatibility += 0.3
            score = round(min(0.99, float(support_score) * 0.5 + lexical * 0.2 + concept_overlap * 0.15 + min(1.0, compatibility) * 0.15), 4)
            scored.append({
                'graph_query': graph_query,
                'hidden_goal': hidden_goal,
                'premise': premise,
                'candidate_type': candidate_type,
                'support_score': score,
            })
        scored.sort(key=lambda item: (-float(item['support_score']), item['premise']))
        return scored[:top_k]

    def search_operator_support(self, query: str, top_k: int = 6, source: Optional[str] = None, split: Optional[str] = None) -> List[dict]:
        normalized = self.normalize_query(query)
        query_terms = set(normalized.split())
        query_concepts = set(match_concepts(query))
        with closing(self._connect()) as conn:
            clauses = []
            params: List[str] = []
            sql = "SELECT graph_query, operator_name, operator_family, confidence, normalized_query FROM operator_memories"
            if source is not None:
                clauses.append("source = ?")
                params.append(source)
            if split is not None:
                clauses.append("split = ?")
                params.append(split)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            rows = conn.execute(sql, params).fetchall()
        scored: List[dict] = []
        for graph_query, operator_name, operator_family, confidence, stored_normalized in rows:
            stored_terms = set(str(stored_normalized).split())
            lexical = len(query_terms & stored_terms) / float(max(1, len(query_terms | stored_terms)))
            graph_concepts = set(match_concepts(str(graph_query)))
            concept_overlap = len(query_concepts & graph_concepts) / float(max(1, len(query_concepts | graph_concepts))) if (query_concepts or graph_concepts) else 0.0
            compatibility = 0.0
            family_upper = str(operator_family).upper()
            if concepts := query_concepts & {'bag', 'drawer', 'door', 'cabinet', 'box', 'bottle', 'jar', 'bin', 'pouch', 'suitcase'}:
                if 'ACCESS' in family_upper:
                    compatibility += 0.7
                if 'CONTROL' in family_upper or 'CAPPED' in family_upper:
                    compatibility += 0.45
                if 'CONTAINER' in family_upper:
                    compatibility += 0.4
            if query_concepts & {'car_wash', 'car'} and 'SERVICE' in family_upper:
                compatibility += 0.6
            score = round(min(0.99, float(confidence) * 0.5 + lexical * 0.2 + concept_overlap * 0.15 + min(1.0, compatibility) * 0.15), 4)
            scored.append({
                'graph_query': graph_query,
                'operator_name': operator_name,
                'operator_family': operator_family,
                'support_score': score,
            })
        scored.sort(key=lambda item: (-float(item['support_score']), item['operator_name']))
        return scored[:top_k]

    def search_script_support(self, query: str, top_k: int = 6, source: Optional[str] = None, split: Optional[str] = None) -> List[dict]:
        normalized = self.normalize_query(query)
        query_terms = set(normalized.split())
        query_concepts = set(match_concepts(query))
        query_scripts = {step for concept in query_concepts for step in scripts_for_concept(concept)}
        with closing(self._connect()) as conn:
            clauses = []
            params: List[str] = []
            sql = "SELECT graph_query, script_step, support_score, normalized_query FROM script_memories"
            if source is not None:
                clauses.append("source = ?")
                params.append(source)
            if split is not None:
                clauses.append("split = ?")
                params.append(split)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            rows = conn.execute(sql, params).fetchall()
        scored: List[dict] = []
        for graph_query, script_step, support_score, stored_normalized in rows:
            stored_terms = set(str(stored_normalized).split())
            lexical = len(query_terms & stored_terms) / float(max(1, len(query_terms | stored_terms)))
            graph_concepts = set(match_concepts(str(graph_query)))
            concept_overlap = len(query_concepts & graph_concepts) / float(max(1, len(query_concepts | graph_concepts))) if (query_concepts or graph_concepts) else 0.0
            compatibility = 1.0 if script_step in query_scripts else 0.0
            if not compatibility and any(token in str(script_step) for token in ('open_', 'retrieve_', 'wash_', 'queue', 'pass_through')):
                compatibility = 0.35
            score = round(min(0.99, float(support_score) * 0.5 + lexical * 0.15 + concept_overlap * 0.15 + compatibility * 0.2), 4)
            scored.append({
                'graph_query': graph_query,
                'script_step': script_step,
                'support_score': score,
            })
        scored.sort(key=lambda item: (-float(item['support_score']), item['script_step']))
        return scored[:top_k]

    def search_similar_graphs(self, query: str, split: Optional[str] = "train", source: Optional[str] = None, top_k: int = 3, query_graph: StructuredMeaningGraph | None = None) -> List[StructuredMeaningGraph]:
        from .memory_retrieval import QueryEmbeddingIndex

        graphs = self.fetch_graphs(split=split, source=source)
        if not graphs:
            return []

        index = QueryEmbeddingIndex()
        hits = index.search(query, graphs, top_k=len(graphs))
        hit_scores = {hit.query: hit.score for hit in hits}
        if query_graph is None:
            ordered = sorted(graphs, key=lambda graph: (-hit_scores.get(graph.query, 0.0), graph.query))
            return ordered[:top_k]

        scored: List[tuple[float, float, float, StructuredMeaningGraph]] = []
        for graph in graphs:
            lexical_score = float(hit_scores.get(graph.query, 0.0))
            structural_score = self._graph_structural_similarity(query_graph, graph)
            total_score = round(lexical_score * 0.35 + structural_score * 0.65, 4)
            scored.append((total_score, structural_score, lexical_score, graph))
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3].query))
        return [item[3] for item in scored[:top_k]]

    @classmethod
    def _graph_structural_similarity(cls, query_graph: StructuredMeaningGraph, stored_graph: StructuredMeaningGraph) -> float:
        query_nodes = set(match_concepts(query_graph.query)) | query_graph.node_ids()
        stored_nodes = set(match_concepts(stored_graph.query)) | stored_graph.node_ids()
        query_families = cls._node_families(query_nodes)
        stored_families = cls._node_families(stored_nodes)
        query_goals = cls._goal_patterns(query_graph.hidden_goals)
        stored_goals = cls._goal_patterns(stored_graph.hidden_goals)
        query_required = set(query_graph.required_premises)
        stored_required = set(stored_graph.required_premises)
        query_missing = set(query_graph.missing_premises)
        stored_missing = set(stored_graph.missing_premises)
        query_relations = {edge.relation for edge in query_graph.edges}
        stored_relations = {edge.relation for edge in stored_graph.edges}
        query_operators = {candidate.family for candidate in query_graph.induced_operators}
        stored_operators = {candidate.family for candidate in stored_graph.induced_operators}
        query_scripts = set(query_graph.inferred_scripts)
        stored_scripts = set(stored_graph.inferred_scripts)
        node_overlap = max(cls._set_overlap(query_nodes, stored_nodes), cls._set_overlap(query_families, stored_families))
        score = (
            0.24 * cls._set_overlap(query_goals, stored_goals)
            + 0.20 * cls._set_overlap(query_required, stored_required)
            + 0.16 * cls._set_overlap(query_missing, stored_missing)
            + 0.14 * node_overlap
            + 0.10 * cls._set_overlap(query_relations, stored_relations)
            + 0.10 * cls._set_overlap(query_operators, stored_operators)
            + 0.06 * cls._set_overlap(query_scripts, stored_scripts)
        )
        return round(min(0.99, score), 4)

    @staticmethod
    def _set_overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / float(len(left | right))

    @staticmethod
    def _node_families(nodes: set[str]) -> set[str]:
        families = {concept_family(node) for node in nodes}
        return {family for family in families if family and family != 'concept'}

    @classmethod
    def _goal_patterns(cls, goals: List[str]) -> set[str]:
        return {cls._goal_pattern(goal) for goal in goals if goal}

    @staticmethod
    def _goal_pattern(goal: str) -> str:
        if goal.startswith('retrieve_item_from_') and goal.endswith('_goal'):
            return 'retrieve_item_from_container_goal'
        if goal.startswith('store_') and '_in_' in goal and goal.endswith('_goal'):
            return 'store_item_in_container_goal'
        if goal.startswith('pass_through_') and goal.endswith('_goal'):
            return 'pass_through_barrier_goal'
        return goal

    @staticmethod
    def normalize_query(query: str) -> str:
        return " ".join(query.strip().lower().split())


    @staticmethod
    def _premise_family_hint(premise: str, hidden_goal: str = '') -> str:
        if premise in {'open_access', 'available_space'}:
            if hidden_goal in {'retrieve_item_from_cabinet_goal', 'retrieve_item_from_drawer_goal', 'retrieve_item_from_box_goal', 'retrieve_item_from_bin_goal', 'retrieve_item_from_pouch_goal', 'retrieve_item_from_suitcase_goal', 'store_book_in_bag_goal', 'pour_from_bottle_goal', 'pass_through_door_goal'}:
                return 'container'
        if premise == 'vehicle_present':
            return 'service_place'
        if premise == 'subquadratic_complexity':
            return 'concept'
        return ''



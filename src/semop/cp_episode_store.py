from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class CpEpisodeRecord:
    statement: str
    normalized_statement: str
    category: str
    approach: str
    cpp_code: str
    time_complexity: str
    memory_complexity: str
    confidence: float
    goal_types: List[str]
    domain_tags: List[str]
    logical_frames: List[str]
    dsl_operators: List[str]
    hidden_concepts: List[str]
    extracted_constraints: List[str]
    candidate_algorithms: List[str]
    memory_projection: dict
    reasoning_steps: List[str]
    compile_ok: bool
    compile_command: str
    compile_stderr: str
    validation_report: dict
    repaired: bool
    repair_attempts: List[dict]
    knowledge_sources: List[str]
    problem_id: str = ""
    editorial_summary: str = ""
    outcome_label: str = ""
    failure_kind: str = ""
    source_kind: str = "generated"

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class CpEpisodeMatch:
    category: str
    statement: str
    similarity: float
    success: bool
    repaired: bool
    logical_frames: List[str]
    dsl_operators: List[str]
    outcome_label: str
    failure_kind: str
    editorial_summary: str

    def model_dump(self) -> dict:
        return asdict(self)


class CpEpisodeStore:
    EXTRA_COLUMNS = {
        "problem_id": "TEXT NOT NULL DEFAULT ''",
        "editorial_summary": "TEXT NOT NULL DEFAULT ''",
        "outcome_label": "TEXT NOT NULL DEFAULT ''",
        "failure_kind": "TEXT NOT NULL DEFAULT ''",
        "source_kind": "TEXT NOT NULL DEFAULT 'generated'",
    }

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
                CREATE TABLE IF NOT EXISTS cp_episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    statement TEXT NOT NULL,
                    normalized_statement TEXT NOT NULL,
                    category TEXT NOT NULL,
                    approach TEXT NOT NULL,
                    cpp_code TEXT NOT NULL,
                    time_complexity TEXT NOT NULL,
                    memory_complexity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    goal_types TEXT NOT NULL,
                    domain_tags TEXT NOT NULL,
                    logical_frames TEXT NOT NULL,
                    dsl_operators TEXT NOT NULL,
                    hidden_concepts TEXT NOT NULL,
                    extracted_constraints TEXT NOT NULL,
                    candidate_algorithms TEXT NOT NULL,
                    memory_projection TEXT NOT NULL,
                    reasoning_steps TEXT NOT NULL,
                    compile_ok INTEGER NOT NULL,
                    compile_command TEXT NOT NULL,
                    compile_stderr TEXT NOT NULL,
                    validation_report TEXT NOT NULL,
                    repaired INTEGER NOT NULL,
                    repair_attempts TEXT NOT NULL,
                    knowledge_sources TEXT NOT NULL,
                    problem_id TEXT NOT NULL DEFAULT '',
                    editorial_summary TEXT NOT NULL DEFAULT '',
                    outcome_label TEXT NOT NULL DEFAULT '',
                    failure_kind TEXT NOT NULL DEFAULT '',
                    source_kind TEXT NOT NULL DEFAULT 'generated',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            existing_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(cp_episodes)").fetchall()
            }
            for column, ddl in self.EXTRA_COLUMNS.items():
                if column not in existing_columns:
                    conn.execute(f"ALTER TABLE cp_episodes ADD COLUMN {column} {ddl}")
            conn.commit()

    @staticmethod
    def _encode(value: object) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _decode(value: str) -> object:
        return json.loads(value)

    def record_episode(self, record: CpEpisodeRecord) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO cp_episodes (
                    statement,
                    normalized_statement,
                    category,
                    approach,
                    cpp_code,
                    time_complexity,
                    memory_complexity,
                    confidence,
                    goal_types,
                    domain_tags,
                    logical_frames,
                    dsl_operators,
                    hidden_concepts,
                    extracted_constraints,
                    candidate_algorithms,
                    memory_projection,
                    reasoning_steps,
                    compile_ok,
                    compile_command,
                    compile_stderr,
                    validation_report,
                    repaired,
                    repair_attempts,
                    knowledge_sources,
                    problem_id,
                    editorial_summary,
                    outcome_label,
                    failure_kind,
                    source_kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.statement,
                    record.normalized_statement,
                    record.category,
                    record.approach,
                    record.cpp_code,
                    record.time_complexity,
                    record.memory_complexity,
                    record.confidence,
                    self._encode(record.goal_types),
                    self._encode(record.domain_tags),
                    self._encode(record.logical_frames),
                    self._encode(record.dsl_operators),
                    self._encode(record.hidden_concepts),
                    self._encode(record.extracted_constraints),
                    self._encode(record.candidate_algorithms),
                    self._encode(record.memory_projection),
                    self._encode(record.reasoning_steps),
                    int(record.compile_ok),
                    record.compile_command,
                    record.compile_stderr,
                    self._encode(record.validation_report),
                    int(record.repaired),
                    self._encode(record.repair_attempts),
                    self._encode(record.knowledge_sources),
                    record.problem_id,
                    record.editorial_summary,
                    record.outcome_label,
                    record.failure_kind,
                    record.source_kind,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def fetch_recent(self, limit: int = 20) -> List[CpEpisodeRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    statement,
                    normalized_statement,
                    category,
                    approach,
                    cpp_code,
                    time_complexity,
                    memory_complexity,
                    confidence,
                    goal_types,
                    domain_tags,
                    logical_frames,
                    dsl_operators,
                    hidden_concepts,
                    extracted_constraints,
                    candidate_algorithms,
                    memory_projection,
                    reasoning_steps,
                    compile_ok,
                    compile_command,
                    compile_stderr,
                    validation_report,
                    repaired,
                    repair_attempts,
                    knowledge_sources,
                    problem_id,
                    editorial_summary,
                    outcome_label,
                    failure_kind,
                    source_kind
                FROM cp_episodes
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute('SELECT COUNT(*) FROM cp_episodes').fetchone()
        return int(row[0]) if row else 0

    def iter_records(self) -> Iterable[CpEpisodeRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    statement,
                    normalized_statement,
                    category,
                    approach,
                    cpp_code,
                    time_complexity,
                    memory_complexity,
                    confidence,
                    goal_types,
                    domain_tags,
                    logical_frames,
                    dsl_operators,
                    hidden_concepts,
                    extracted_constraints,
                    candidate_algorithms,
                    memory_projection,
                    reasoning_steps,
                    compile_ok,
                    compile_command,
                    compile_stderr,
                    validation_report,
                    repaired,
                    repair_attempts,
                    knowledge_sources,
                    problem_id,
                    editorial_summary,
                    outcome_label,
                    failure_kind,
                    source_kind
                FROM cp_episodes
                ORDER BY id ASC
                """
            ).fetchall()
        for row in rows:
            yield self._row_to_record(row)

    def export_jsonl(self, output_path: str | Path) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', encoding='utf-8') as handle:
            for record in self.iter_records():
                handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + '\n')

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z_]+", text.lower()) if len(token) > 2}

    @classmethod
    def _similarity(cls, left: str, right: str) -> float:
        left_tokens = cls._tokenize(left)
        right_tokens = cls._tokenize(right)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))

    def find_similar(self, normalized_statement: str, limit: int = 8) -> List[CpEpisodeMatch]:
        matches: List[CpEpisodeMatch] = []
        for record in self.iter_records():
            similarity = self._similarity(normalized_statement, record.normalized_statement)
            if similarity <= 0:
                continue
            success = bool(record.compile_ok and record.validation_report.get("overall_ok"))
            matches.append(
                CpEpisodeMatch(
                    category=record.category,
                    statement=record.statement,
                    similarity=round(similarity, 4),
                    success=success,
                    repaired=record.repaired,
                    logical_frames=record.logical_frames,
                    dsl_operators=record.dsl_operators,
                    outcome_label=record.outcome_label,
                    failure_kind=record.failure_kind,
                    editorial_summary=record.editorial_summary,
                )
            )
        matches.sort(key=lambda item: (-item.similarity, not item.success, item.category))
        return matches[:limit]

    def build_rerank_priors(self, normalized_statement: str, limit: int = 8) -> Dict[str, object]:
        matches = self.find_similar(normalized_statement, limit=limit)
        category_scores: Dict[str, float] = {}
        success_counts: Dict[str, int] = {}
        failure_counts: Dict[str, int] = {}
        failure_kinds: Dict[str, int] = {}
        for match in matches:
            outcome = match.outcome_label.upper()
            if match.success or outcome == 'AC':
                category_scores[match.category] = category_scores.get(match.category, 0.0) + match.similarity + (0.08 if match.repaired else 0.0)
                success_counts[match.category] = success_counts.get(match.category, 0) + 1
            else:
                penalty = match.similarity + (0.12 if outcome in {'WA', 'TLE', 'RE', 'MLE'} else 0.0)
                category_scores[match.category] = category_scores.get(match.category, 0.0) - penalty
                failure_counts[match.category] = failure_counts.get(match.category, 0) + 1
                if match.failure_kind:
                    failure_kinds[match.failure_kind] = failure_kinds.get(match.failure_kind, 0) + 1
        return {
            'category_scores': {key: round(value, 4) for key, value in category_scores.items()},
            'success_counts': success_counts,
            'failure_counts': failure_counts,
            'failure_kinds': failure_kinds,
            'similar_episodes': [match.model_dump() for match in matches[:5]],
        }

    def _row_to_record(self, row: tuple) -> CpEpisodeRecord:
        return CpEpisodeRecord(
            statement=row[0],
            normalized_statement=row[1],
            category=row[2],
            approach=row[3],
            cpp_code=row[4],
            time_complexity=row[5],
            memory_complexity=row[6],
            confidence=float(row[7]),
            goal_types=list(self._decode(row[8])),
            domain_tags=list(self._decode(row[9])),
            logical_frames=list(self._decode(row[10])),
            dsl_operators=list(self._decode(row[11])),
            hidden_concepts=list(self._decode(row[12])),
            extracted_constraints=list(self._decode(row[13])),
            candidate_algorithms=list(self._decode(row[14])),
            memory_projection=dict(self._decode(row[15])),
            reasoning_steps=list(self._decode(row[16])),
            compile_ok=bool(row[17]),
            compile_command=row[18],
            compile_stderr=row[19],
            validation_report=dict(self._decode(row[20])),
            repaired=bool(row[21]),
            repair_attempts=list(self._decode(row[22])),
            knowledge_sources=list(self._decode(row[23])),
            problem_id=row[24] or '',
            editorial_summary=row[25] or '',
            outcome_label=row[26] or '',
            failure_kind=row[27] or '',
            source_kind=row[28] or 'generated',
        )

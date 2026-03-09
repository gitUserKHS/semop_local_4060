from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from .contest_programmer import CompetitiveProgrammingReasoner
from .cp_episode_store import CpEpisodeRecord, CpEpisodeStore
from .cp_validation import CpSolutionValidator


@dataclass
class CpIncidentCase:
    statement: str
    problem_id: str = ""
    editorial_summary: str = ""
    outcome_label: str = ""
    failure_kind: str = ""
    code: str = ""
    source_kind: str = "incident"

    def model_dump(self) -> dict:
        return {
            "statement": self.statement,
            "problem_id": self.problem_id,
            "editorial_summary": self.editorial_summary,
            "outcome_label": self.outcome_label,
            "failure_kind": self.failure_kind,
            "code": self.code,
            "source_kind": self.source_kind,
        }


class CpIncidentDataset:
    TEXT_FIELDS = ("statement", "query", "question", "problem", "prompt", "text")
    EDITORIAL_FIELDS = ("editorial", "editorial_summary", "solution", "explanation")
    OUTCOME_FIELDS = ("outcome", "verdict", "result", "status")
    FAILURE_FIELDS = ("failure_kind", "failure", "error_kind", "incident_type")
    CODE_FIELDS = ("code", "submission", "cpp_code")
    ID_FIELDS = ("problem_id", "id", "slug")

    def load_inputs(self, inputs: Sequence[str | Path]) -> List[CpIncidentCase]:
        records: List[CpIncidentCase] = []
        for item in inputs:
            records.extend(self._load_path(Path(item)))
        return records

    def _load_path(self, path: Path) -> List[CpIncidentCase]:
        if path.is_dir():
            cases: List[CpIncidentCase] = []
            for child in sorted(path.rglob('*')):
                if child.is_file() and child.suffix.lower() in {'.jsonl', '.json', '.csv'}:
                    cases.extend(self._load_path(child))
            return cases
        if path.suffix.lower() == '.jsonl':
            return self._from_jsonl(path)
        if path.suffix.lower() == '.json':
            return self._from_json(path)
        if path.suffix.lower() == '.csv':
            return self._from_csv(path)
        return []

    def _from_jsonl(self, path: Path) -> List[CpIncidentCase]:
        cases: List[CpIncidentCase] = []
        with path.open('r', encoding='utf-8-sig') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                case = self._from_payload(payload, source_kind='jsonl')
                if case is not None:
                    cases.append(case)
        return cases

    def _from_json(self, path: Path) -> List[CpIncidentCase]:
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(payload, list):
            return [case for item in payload if (case := self._from_payload(item, source_kind='json')) is not None]
        if isinstance(payload, dict):
            case = self._from_payload(payload, source_kind='json')
            if case is not None:
                return [case]
            cases: List[CpIncidentCase] = []
            for value in payload.values():
                if isinstance(value, list):
                    for item in value:
                        case = self._from_payload(item, source_kind='json')
                        if case is not None:
                            cases.append(case)
            return cases
        return []

    def _from_csv(self, path: Path) -> List[CpIncidentCase]:
        cases: List[CpIncidentCase] = []
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                case = self._from_payload(row, source_kind='csv')
                if case is not None:
                    cases.append(case)
        return cases

    def _from_payload(self, payload: object, source_kind: str) -> CpIncidentCase | None:
        if not isinstance(payload, dict):
            return None
        statement = self._pick(payload, self.TEXT_FIELDS)
        if not statement:
            return None
        return CpIncidentCase(
            statement=statement,
            problem_id=self._pick(payload, self.ID_FIELDS),
            editorial_summary=self._pick(payload, self.EDITORIAL_FIELDS),
            outcome_label=self._pick(payload, self.OUTCOME_FIELDS).upper(),
            failure_kind=self._pick(payload, self.FAILURE_FIELDS),
            code=self._pick(payload, self.CODE_FIELDS),
            source_kind=source_kind,
        )

    @staticmethod
    def _pick(payload: dict, fields: Sequence[str]) -> str:
        for field in fields:
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
        return ""


class CpIncidentIngestor:
    def __init__(self, store_path: str | Path, compiler: str = 'g++') -> None:
        self.store = CpEpisodeStore(store_path)
        self.validator = CpSolutionValidator(compiler=compiler)
        self.reasoner = CompetitiveProgrammingReasoner(compiler=compiler)

    def ingest_cases(self, cases: Iterable[CpIncidentCase], solve_missing: bool = True) -> List[int]:
        stored_ids: List[int] = []
        for case in cases:
            stored_ids.append(self.ingest_case(case, solve_missing=solve_missing))
        return stored_ids

    def ingest_case(self, case: CpIncidentCase, solve_missing: bool = True) -> int:
        solution = self.reasoner.solve(case.statement) if solve_missing else None
        code = case.code or (solution.cpp_code if solution is not None else '')
        validation_report = {}
        compile_ok = False
        compile_command = ''
        compile_stderr = ''
        repaired = False
        repair_attempts: List[dict] = []
        category = solution.category if solution is not None else 'incident_memory_only'
        if code and category in self.validator.SUPPORTED_CATEGORIES:
            validation = self.validator.validate(code, category)
            validation_report = validation.model_dump()
            compile_ok = bool(validation.build_ok)
        if solution is not None:
            compile_command = solution.compile_command
            compile_stderr = solution.compile_stderr
            repaired = solution.repaired
            repair_attempts = solution.repair_attempts
        outcome_label = case.outcome_label or ('AC' if validation_report.get('overall_ok') else 'WA' if validation_report else '')
        failure_kind = case.failure_kind or str(validation_report.get('failure_type', ''))
        record = CpEpisodeRecord(
            statement=case.statement,
            normalized_statement=' '.join(case.statement.lower().split()),
            category=category,
            approach=solution.approach if solution is not None else case.editorial_summary or 'Imported CP incident episode.',
            cpp_code=code,
            time_complexity=solution.time_complexity if solution is not None else '',
            memory_complexity=solution.memory_complexity if solution is not None else '',
            confidence=solution.confidence if solution is not None else 0.0,
            goal_types=solution.goal_types if solution is not None else [],
            domain_tags=solution.domain_tags if solution is not None else [],
            logical_frames=solution.logical_frames if solution is not None else [],
            dsl_operators=solution.dsl_operators if solution is not None else [],
            hidden_concepts=solution.hidden_concepts if solution is not None else [],
            extracted_constraints=solution.extracted_constraints if solution is not None else [],
            candidate_algorithms=solution.memory_projection.get('episodic', {}).get('candidate_algorithms', []) if solution is not None else [],
            memory_projection=solution.memory_projection if solution is not None else {},
            reasoning_steps=solution.reasoning_steps if solution is not None else [],
            compile_ok=compile_ok,
            compile_command=compile_command,
            compile_stderr=compile_stderr,
            validation_report=validation_report,
            repaired=repaired,
            repair_attempts=repair_attempts,
            knowledge_sources=solution.knowledge_sources if solution is not None else [],
            problem_id=case.problem_id,
            editorial_summary=case.editorial_summary,
            outcome_label=outcome_label,
            failure_kind=failure_kind,
            source_kind=case.source_kind,
        )
        return self.store.record_episode(record)

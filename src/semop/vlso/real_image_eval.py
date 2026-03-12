from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass
class RealImageEvalCaseCandidate:
    case_id: str
    image_path: str
    query: str
    expected_entities: list[str]
    expected_relations: list[dict[str, str]]
    required_terms: list[str]
    forbidden_terms: list[str]
    expected_operators: list[str]
    expected_operator_bindings: list[dict[str, str]]
    expected_support_premises: list[str]
    title: str
    provider: str
    source_id: str
    needs_review: bool = True

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RealImageEvalBuildSummary:
    num_records: int
    num_existing_images: int
    num_candidates: int
    num_seed_cases: int
    candidate_output: str | None = None
    seed_output: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)



@dataclass
class RealImageEvalFinalizeSummary:
    num_candidates: int
    num_reviewed: int
    num_approved: int
    output_path: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class RealImageEvalBuilder:
    _BAG_WORDS = {'bag', 'handbag', 'pack', 'backpack', 'purse', 'satchel', 'messenger', 'suitcase', 'pouch'}
    _BOX_WORDS = {'box', 'case', 'drawer', 'cabinet', 'bin', 'jar', 'bottle', 'dryer'}
    _HANDLE_WORDS = {'handle', 'strap', 'knob', 'grip', 'pull'}
    _AUTO_REJECT_TOKENS = {'unusable', 'tutorial', 'steps', 'how', 'knife', 'closed'}

    def load_records(self, path: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw in Path(path).read_text(encoding='utf-8-sig').splitlines():
            if raw.strip():
                rows.append(json.loads(raw))
        return rows

    def load_download_manifest(self, path: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw in Path(path).read_text(encoding='utf-8-sig').splitlines():
            if raw.strip():
                rows.append(json.loads(raw))
        return rows

    @staticmethod
    def _tokenize(record: dict[str, Any]) -> set[str]:
        text_parts = [str(record.get('title', '')), str(record.get('query', ''))]
        raw = record.get('raw', {}) if isinstance(record.get('raw'), dict) else {}
        tags = raw.get('tags', [])
        if isinstance(tags, list):
            for item in tags:
                if isinstance(item, dict):
                    text_parts.append(str(item.get('name', '')))
                else:
                    text_parts.append(str(item))
        joined = ' '.join(text_parts).lower().replace('-', ' ').replace('_', ' ')
        return {token for token in joined.split() if token}

    @staticmethod
    def _relpath_for_output(target: Path, output_path: Path) -> str:
        import os
        try:
            return os.path.relpath(str(target.resolve()), str(output_path.parent.resolve()))
        except Exception:
            return str(target.resolve())

    def _join_rows(self, records: Iterable[dict[str, Any]], manifest_rows: Iterable[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any], Path]]:
        manifest_map = {
            (str(row.get('provider', '')), str(row.get('source_id', ''))): row
            for row in manifest_rows
            if isinstance(row, dict)
        }
        joined: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
        for record in records:
            key = (str(record.get('provider', '')), str(record.get('source_id', '')))
            manifest_row = manifest_map.get(key)
            if not manifest_row:
                continue
            target = Path(str(manifest_row.get('target_path', '')))
            if target.exists():
                joined.append((record, manifest_row, target))
        return joined

    def _make_candidate(self, record: dict[str, Any], target_path: Path, output_path: Path) -> RealImageEvalCaseCandidate | None:
        tokens = self._tokenize(record)
        title = str(record.get('title', ''))
        provider = str(record.get('provider', ''))
        source_id = str(record.get('source_id', ''))
        image_path = self._relpath_for_output(target_path, output_path)
        expected_entities: list[str] = []
        required_terms: list[str] = []
        expected_operators: list[str] = []
        expected_bindings: list[dict[str, str]] = []
        support: list[str] = []
        query = 'What objects are visible here?'

        bag_hits = tokens & self._BAG_WORDS
        box_hits = tokens & self._BOX_WORDS
        handle_hits = tokens & self._HANDLE_WORDS
        if bag_hits:
            expected_entities.append('bag')
            expected_operators.append('CONTAINER_BODY_OPERATOR')
        if box_hits:
            expected_entities.append(sorted(box_hits)[0])
            expected_operators.append('CONTAINER_BODY_OPERATOR')
        if 'door' in tokens:
            expected_entities.append('door')
            expected_operators.append('ACCESS_CONTROL_OPERATOR')
        if handle_hits:
            expected_entities.append('handle')
            expected_operators.append('ATTACHED_GRASP_OPERATOR')
            required_terms.append('handle')
            support.append('manipulable_grasp')
            if 'door' in tokens or 'drawer' in tokens or 'cabinet' in tokens:
                expected_operators.append('ACCESS_CONTROL_OPERATOR')
        if 'zipper' in tokens:
            expected_entities.append('zipper')
            expected_operators.append('ACCESS_CONTROL_OPERATOR')
            required_terms.append('zipper')
            support.append('open_access')
            query = 'What opening or access control is visible here?'
        if 'open' in tokens or 'opened' in tokens or 'opening' in tokens:
            expected_operators.extend(['ACCESS_PORT_OPERATOR', 'CONTAINER_ACCESS_OPERATOR'])
            required_terms.append('opening')
            support.append('open_access')
            query = 'Is there an opening or open access visible here?'
        if 'dryer' in tokens and 'open' in tokens:
            expected_entities.append('door')
            expected_operators.extend(['ACCESS_CONTROL_OPERATOR', 'ACCESS_PORT_OPERATOR'])
            support.append('open_access')
        if 'case' in tokens and 'open' in tokens:
            expected_entities.append('case')
            expected_operators.extend(['CONTAINER_BODY_OPERATOR', 'ACCESS_PORT_OPERATOR'])
            support.append('open_access')
        if 'drawer' in tokens and handle_hits:
            query = 'What access-related object is visible here?'
        if 'cabinet' in tokens and handle_hits:
            query = 'What access-related object is visible here?'
        if 'bag' in expected_entities and 'zipper' in expected_entities:
            expected_bindings.append({'operator_name': 'ACCESS_CONTROL_OPERATOR', 'subject': 'zipper', 'parent': 'bag'})
        if not expected_entities and not expected_operators:
            return None
        expected_entities = sorted(dict.fromkeys(expected_entities))
        expected_operators = sorted(dict.fromkeys(expected_operators))
        required_terms = sorted(dict.fromkeys(required_terms))
        support = sorted(dict.fromkeys(support))
        case_id = f"real_{provider}_{source_id}".replace('-', '_')
        return RealImageEvalCaseCandidate(
            case_id=case_id,
            image_path=image_path,
            query=query,
            expected_entities=expected_entities,
            expected_relations=[],
            required_terms=required_terms,
            forbidden_terms=['clearer image'],
            expected_operators=expected_operators,
            expected_operator_bindings=expected_bindings,
            expected_support_premises=support,
            title=title,
            provider=provider,
            source_id=source_id,
            needs_review=True,
        )

    @staticmethod
    def _resolve_candidate_image_path(image_path: str, candidates_path: str | Path) -> str:
        raw = str(image_path or '').strip()
        if not raw:
            return raw
        path = Path(raw)
        if path.is_absolute() and path.exists():
            return str(path)
        candidates_file = Path(candidates_path)
        primary = (candidates_file.parent / path).resolve()
        if primary.exists():
            return str(primary)
        workspace = (Path.cwd() / path).resolve()
        if workspace.exists():
            return str(workspace)
        return raw

    @staticmethod
    def _seed_score(candidate: RealImageEvalCaseCandidate) -> int:
        score = 0
        if candidate.expected_operator_bindings:
            score += 3
        if candidate.required_terms:
            score += 2
        score += len(candidate.expected_operators)
        score += len(candidate.expected_entities)
        if 'open' in candidate.query.lower() or 'access' in candidate.query.lower():
            score += 1
        return score


    def load_candidates(self, path: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw in Path(path).read_text(encoding='utf-8-sig').splitlines():
            if raw.strip():
                rows.append(json.loads(raw))
        return rows

    @staticmethod
    def _is_review_approved(row: dict[str, Any]) -> bool:
        status = str(row.get('review_status', '')).strip().lower()
        if status:
            return status in {'approved', 'gold', 'accepted'}
        needs_review = row.get('needs_review')
        if isinstance(needs_review, bool):
            return not needs_review
        return False

    def finalize_reviewed(self, candidates_path: str | Path, output_path: str | Path) -> RealImageEvalFinalizeSummary:
        rows = self.load_candidates(candidates_path)
        approved_rows = [row for row in rows if self._is_review_approved(row)]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', encoding='utf-8') as handle:
            for row in approved_rows:
                payload = dict(row)
                payload['image_path'] = self._resolve_candidate_image_path(payload.get('image_path', ''), candidates_path)
                payload.pop('needs_review', None)
                payload.pop('review_status', None)
                payload.pop('review_notes', None)
                payload.pop('title', None)
                payload.pop('provider', None)
                payload.pop('source_id', None)
                handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
        return RealImageEvalFinalizeSummary(
            num_candidates=len(rows),
            num_reviewed=sum(1 for row in rows if 'review_status' in row or 'review_notes' in row or 'needs_review' in row),
            num_approved=len(approved_rows),
            output_path=str(output),
        )

    def finalize_auto_selected(
        self,
        candidates_path: str | Path,
        output_path: str | Path,
        auto_approve_limit: int = 12,
        reject_terms: list[str] | None = None,
    ) -> RealImageEvalFinalizeSummary:
        rows = self.load_candidates(candidates_path)
        reject_tokens = {token.strip().lower() for token in (reject_terms or []) if token.strip()} or set(self._AUTO_REJECT_TOKENS)
        scored_rows: list[tuple[int, dict[str, Any]]] = []
        for row in rows:
            title = str(row.get('title', '')).lower()
            if any(token in title for token in reject_tokens):
                continue
            candidate = RealImageEvalCaseCandidate(
                case_id=str(row.get('case_id', '')),
                image_path=str(row.get('image_path', '')),
                query=str(row.get('query', '')),
                expected_entities=[str(item) for item in row.get('expected_entities', [])],
                expected_relations=[dict(item) for item in row.get('expected_relations', [])],
                required_terms=[str(item) for item in row.get('required_terms', [])],
                forbidden_terms=[str(item) for item in row.get('forbidden_terms', [])],
                expected_operators=[str(item) for item in row.get('expected_operators', [])],
                expected_operator_bindings=[dict(item) for item in row.get('expected_operator_bindings', [])],
                expected_support_premises=[str(item) for item in row.get('expected_support_premises', [])],
                title=str(row.get('title', '')),
                provider=str(row.get('provider', '')),
                source_id=str(row.get('source_id', '')),
                needs_review=bool(row.get('needs_review', True)),
            )
            scored_rows.append((self._seed_score(candidate), row))
        scored_rows.sort(key=lambda item: (-item[0], str(item[1].get('case_id', ''))))
        approved_rows = [row for _score, row in scored_rows[:max(0, auto_approve_limit)]]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', encoding='utf-8') as handle:
            for row in approved_rows:
                payload = dict(row)
                payload['image_path'] = self._resolve_candidate_image_path(payload.get('image_path', ''), candidates_path)
                payload.pop('needs_review', None)
                payload.pop('review_status', None)
                payload.pop('review_notes', None)
                payload.pop('title', None)
                payload.pop('provider', None)
                payload.pop('source_id', None)
                handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
        return RealImageEvalFinalizeSummary(
            num_candidates=len(rows),
            num_reviewed=len(scored_rows),
            num_approved=len(approved_rows),
            output_path=str(output),
        )

    def build(self, records_path: str | Path, manifest_path: str | Path, candidate_output: str | Path | None = None, seed_output: str | Path | None = None, limit: int | None = None) -> RealImageEvalBuildSummary:
        records = self.load_records(records_path)
        manifest_rows = self.load_download_manifest(manifest_path)
        joined = self._join_rows(records, manifest_rows)
        out_ref = Path(candidate_output or seed_output or records_path)
        candidates: list[RealImageEvalCaseCandidate] = []
        for record, _manifest, target in joined:
            candidate = self._make_candidate(record, target, out_ref)
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=lambda item: (-self._seed_score(item), item.case_id))
        if limit is not None:
            candidates = candidates[:limit]
        seed_cases = [item for item in candidates if self._seed_score(item) >= 5][: min(12, len(candidates))]
        if candidate_output:
            path = Path(candidate_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w', encoding='utf-8') as handle:
                for item in candidates:
                    handle.write(json.dumps(item.model_dump(), ensure_ascii=False) + '\n')
        if seed_output:
            path = Path(seed_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w', encoding='utf-8') as handle:
                for item in seed_cases:
                    payload = item.model_dump()
                    payload.pop('needs_review', None)
                    payload.pop('title', None)
                    payload.pop('provider', None)
                    payload.pop('source_id', None)
                    handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
        return RealImageEvalBuildSummary(
            num_records=len(records),
            num_existing_images=len(joined),
            num_candidates=len(candidates),
            num_seed_cases=len(seed_cases),
            candidate_output=str(candidate_output) if candidate_output else None,
            seed_output=str(seed_output) if seed_output else None,
        )

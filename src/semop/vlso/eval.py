from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .reasoner import VLSOReasoner


@dataclass
class VlsoEvalCase:
    case_id: str
    query: str
    visual_json: str | None = None
    image_path: str | None = None
    expected_entities: list[str] | None = None
    expected_relations: list[dict[str, str]] | None = None
    required_terms: list[str] | None = None
    forbidden_terms: list[str] | None = None
    expected_operators: list[str] | None = None
    expected_operator_bindings: list[dict[str, str]] | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VlsoEvalResult:
    case_id: str
    object_recall: float
    relation_recall: float
    answer_term_recall: float
    grounded_answer_accuracy: float
    operator_recall: float
    operator_binding_recall: float
    answer_text: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VlsoEvalSummary:
    num_cases: int
    object_recall: float
    relation_recall: float
    answer_term_recall: float
    grounded_answer_accuracy: float
    operator_recall: float
    operator_binding_recall: float
    results: list[dict[str, Any]]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class VlsoGroundedEvaluator:
    def __init__(self, reasoner: VLSOReasoner) -> None:
        self.reasoner = reasoner

    @staticmethod
    def load_cases(path: str | Path) -> list[VlsoEvalCase]:
        base = Path(path).resolve().parent
        output: list[VlsoEvalCase] = []
        for raw in Path(path).read_text(encoding='utf-8-sig').splitlines():
            if not raw.strip():
                continue
            payload = json.loads(raw)
            visual_json = payload.get('visual_json')
            image_path = payload.get('image_path')
            if isinstance(visual_json, str) and visual_json and not Path(visual_json).is_absolute():
                visual_json = str((base / visual_json).resolve())
            if isinstance(image_path, str) and image_path and not Path(image_path).is_absolute():
                image_path = str((base / image_path).resolve())
            output.append(
                VlsoEvalCase(
                    case_id=str(payload.get('case_id', f'case_{len(output)+1}')),
                    query=str(payload.get('query', '')),
                    visual_json=visual_json,
                    image_path=image_path,
                    expected_entities=[str(item) for item in payload.get('expected_entities', [])],
                    expected_relations=[dict(item) for item in payload.get('expected_relations', [])],
                    required_terms=[str(item) for item in payload.get('required_terms', [])],
                    forbidden_terms=[str(item) for item in payload.get('forbidden_terms', [])],
                    expected_operators=[str(item) for item in payload.get('expected_operators', [])],
                    expected_operator_bindings=[dict(item) for item in payload.get('expected_operator_bindings', [])],
                )
            )
        return output

    def evaluate_cases(self, cases: list[VlsoEvalCase]) -> VlsoEvalSummary:
        results: list[VlsoEvalResult] = []
        for case in cases:
            visual_input = self._visual_input_from_case(case)
            world, answer = self.reasoner.answer(case.query, visual_input=visual_input)
            results.append(
                VlsoEvalResult(
                    case_id=case.case_id,
                    object_recall=self._object_recall(case, world.model_dump()),
                    relation_recall=self._relation_recall(case, world.model_dump()),
                    answer_term_recall=self._answer_term_recall(case, answer.answer_text),
                    grounded_answer_accuracy=self._grounded_answer_accuracy(case, answer.answer_text),
                    operator_recall=self._operator_recall(case, world.model_dump()),
                    operator_binding_recall=self._operator_binding_recall(case, world.model_dump()),
                    answer_text=answer.answer_text,
                )
            )
        total = max(1, len(results))
        return VlsoEvalSummary(
            num_cases=len(results),
            object_recall=round(sum(item.object_recall for item in results) / total, 4),
            relation_recall=round(sum(item.relation_recall for item in results) / total, 4),
            answer_term_recall=round(sum(item.answer_term_recall for item in results) / total, 4),
            grounded_answer_accuracy=round(sum(item.grounded_answer_accuracy for item in results) / total, 4),
            operator_recall=round(sum(item.operator_recall for item in results) / total, 4),
            operator_binding_recall=round(sum(item.operator_binding_recall for item in results) / total, 4),
            results=[item.model_dump() for item in results],
        )

    @staticmethod
    def _visual_input_from_case(case: VlsoEvalCase) -> dict[str, Any] | str | None:
        if case.image_path:
            return {'image_path': case.image_path, 'metadata': {'image_path': case.image_path}}
        if case.visual_json:
            path = Path(case.visual_json)
            if path.exists():
                return json.loads(path.read_text(encoding='utf-8-sig'))
            return case.visual_json
        return None

    @staticmethod
    def _entity_tokens(world: dict[str, Any]) -> set[str]:
        tokens: set[str] = set()
        for entity in world.get('entities', []):
            if not isinstance(entity, dict) or entity.get('modality') != 'vision':
                continue
            entity_id = str(entity.get('id', '')).lower()
            label = str(entity.get('label', '')).lower()
            if entity_id:
                tokens.add(entity_id)
            if label:
                tokens.add(label)
            attrs = entity.get('attributes', {}) if isinstance(entity.get('attributes'), dict) else {}
            concept_labels = attrs.get('concept_labels') or []
            if isinstance(concept_labels, list):
                for item in concept_labels:
                    tokens.add(str(item).lower())
        return tokens

    def _object_recall(self, case: VlsoEvalCase, world: dict[str, Any]) -> float:
        expected = [item.lower() for item in (case.expected_entities or [])]
        if not expected:
            return 1.0
        tokens = self._entity_tokens(world)
        hits = 0
        for item in expected:
            if any(item in token or token in item for token in tokens):
                hits += 1
        return round(hits / len(expected), 4)

    @staticmethod
    def _relation_recall(case: VlsoEvalCase, world: dict[str, Any]) -> float:
        expected = case.expected_relations or []
        if not expected:
            return 1.0
        actual = {
            (
                str(item.get('source', '')).lower(),
                str(item.get('relation', '')).upper(),
                str(item.get('target', '')).lower(),
            )
            for item in world.get('relations', [])
            if isinstance(item, dict)
        }
        hits = 0
        for row in expected:
            triple = (
                str(row.get('source', '')).lower(),
                str(row.get('relation', '')).upper(),
                str(row.get('target', '')).lower(),
            )
            if triple in actual:
                hits += 1
        return round(hits / len(expected), 4)

    @staticmethod
    def _answer_term_recall(case: VlsoEvalCase, answer_text: str) -> float:
        required = [item.lower() for item in (case.required_terms or [])]
        if not required:
            return 1.0
        lowered = answer_text.lower()
        hits = sum(1 for item in required if item in lowered)
        return round(hits / len(required), 4)

    @staticmethod
    def _grounded_answer_accuracy(case: VlsoEvalCase, answer_text: str) -> float:
        lowered = answer_text.lower()
        required = [item.lower() for item in (case.required_terms or [])]
        forbidden = [item.lower() for item in (case.forbidden_terms or [])]
        required_ok = all(item in lowered for item in required)
        forbidden_ok = all(item not in lowered for item in forbidden)
        return 1.0 if required_ok and forbidden_ok else 0.0

    @staticmethod
    def _operator_recall(case: VlsoEvalCase, world: dict[str, Any]) -> float:
        expected = [item.upper() for item in (case.expected_operators or [])]
        if not expected:
            return 1.0
        actual = {
            str(item.get('name', '')).upper()
            for item in world.get('operators', [])
            if isinstance(item, dict)
        }
        hits = sum(1 for item in expected if item in actual)
        return round(hits / len(expected), 4)

    @staticmethod
    def _operator_binding_recall(case: VlsoEvalCase, world: dict[str, Any]) -> float:
        expected = case.expected_operator_bindings or []
        if not expected:
            return 1.0
        metadata = world.get('metadata', {}) if isinstance(world.get('metadata'), dict) else {}
        actual_rows = metadata.get('structural_operator_bindings', []) if isinstance(metadata.get('structural_operator_bindings'), list) else []
        actual = {
            (
                str(item.get('operator_name', '')).upper(),
                str(item.get('subject', '')).lower(),
                str(item.get('parent', '')).lower(),
            )
            for item in actual_rows
            if isinstance(item, dict)
        }
        hits = 0
        for row in expected:
            triple = (
                str(row.get('operator_name', '')).upper(),
                str(row.get('subject', '')).lower(),
                str(row.get('parent', '')).lower(),
            )
            if triple in actual:
                hits += 1
        return round(hits / len(expected), 4)

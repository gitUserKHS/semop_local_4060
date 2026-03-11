from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .affordance_features import VisualAffordanceFeatureExtractor
from .concept_dataset import VisualConceptDataset
from .image_parser import RawImageObservationParser


@dataclass
class VisualOperatorRecord:
    key: str
    operator_name: str
    feature_vector: Dict[str, float]
    signature: List[str]
    metadata: Dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VisualOperatorMatch:
    key: str
    operator_name: str
    score: float
    metadata: Dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VisualOperatorLearningSummary:
    input_path: str
    store_path: str
    operator_names: List[str]
    prototype_count: int
    examples: int
    targets: int

    def model_dump(self) -> dict[str, Any]:
        return {
            'input_path': self.input_path,
            'store_path': self.store_path,
            'operator_names': list(self.operator_names),
            'prototype_count': self.prototype_count,
            'examples': self.examples,
            'targets': self.targets,
        }


class VisualOperatorMemory:
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
                CREATE TABLE IF NOT EXISTS visual_operator_prototypes (
                    key TEXT PRIMARY KEY,
                    operator_name TEXT NOT NULL,
                    feature_json TEXT NOT NULL,
                    signature_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def upsert(self, record: VisualOperatorRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO visual_operator_prototypes (key, operator_name, feature_json, signature_json, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    operator_name=excluded.operator_name,
                    feature_json=excluded.feature_json,
                    signature_json=excluded.signature_json,
                    metadata_json=excluded.metadata_json
                """,
                (
                    record.key,
                    record.operator_name,
                    json.dumps(record.feature_vector, ensure_ascii=False, sort_keys=True),
                    json.dumps(record.signature, ensure_ascii=False),
                    json.dumps(record.metadata, ensure_ascii=False),
                ),
            )
            conn.commit()

    def search(self, feature_vector: Dict[str, float], signature: List[str], limit: int = 5) -> List[VisualOperatorMatch]:
        matches: List[VisualOperatorMatch] = []
        with self._connect() as conn:
            rows = conn.execute('SELECT key, operator_name, feature_json, signature_json, metadata_json FROM visual_operator_prototypes').fetchall()
        for key, operator_name, feature_json, signature_json, metadata_json in rows:
            stored_vector = json.loads(feature_json)
            stored_signature = json.loads(signature_json)
            score = 0.7 * self._cosine(feature_vector, stored_vector) + 0.3 * self._signature_overlap(signature, stored_signature)
            matches.append(
                VisualOperatorMatch(
                    key=key,
                    operator_name=operator_name,
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
        normalized_left = {key: VisualOperatorMemory._normalize_value(float(left.get(key, 0.0))) for key in keys}
        normalized_right = {key: VisualOperatorMemory._normalize_value(float(right.get(key, 0.0))) for key in keys}
        dot = sum(normalized_left[key] * normalized_right[key] for key in keys)
        left_norm = math.sqrt(sum(normalized_left[key] ** 2 for key in keys)) or 1.0
        right_norm = math.sqrt(sum(normalized_right[key] ** 2 for key in keys)) or 1.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _normalize_value(value: float) -> float:
        if value == 0.0:
            return 0.0
        return math.tanh(value / 3.0)

    @staticmethod
    def _signature_overlap(left: List[str], right: List[str]) -> float:
        lset = {str(item) for item in left}
        rset = {str(item) for item in right}
        if not lset or not rset:
            return 0.0
        union = lset | rset
        if not union:
            return 0.0
        return len(lset & rset) / len(union)


class VisualOperatorPrototypeTrainer:
    def __init__(self) -> None:
        self.dataset = VisualConceptDataset()
        self.image_parser = RawImageObservationParser()
        self.feature_extractor = VisualAffordanceFeatureExtractor()

    def train_jsonl(self, labels_path: str | Path, store_path: str | Path, summary_output: str | Path | None = None) -> VisualOperatorLearningSummary:
        examples = self.dataset.load_jsonl(labels_path)
        rows_by_operator: dict[str, list[dict[str, Any]]] = {}
        target_count = 0
        for example in examples:
            observation = self.image_parser.parse_image(example.image_path).observation
            candidates = self.feature_extractor.extract(observation)
            candidate_map = {item.subject: item for item in candidates}
            for target in example.targets:
                target_count += 1
                labels = [str(item) for item in target.positive_labels]
                if not labels:
                    continue
                candidate = self._resolve_candidate(candidates, candidate_map, target.subject_id, labels)
                if candidate is None:
                    continue
                signature = self._signature(candidate.features, candidate.parent)
                for operator_name in self._derive_operator_names(labels, signature):
                    rows_by_operator.setdefault(operator_name, []).append(
                        {
                            'features': dict(candidate.features),
                            'signature': signature,
                            'labels': labels,
                            'image_path': example.image_path,
                            'subject_id': candidate.subject,
                        }
                    )
        store = VisualOperatorMemory(store_path)
        operator_names = sorted(rows_by_operator)
        for operator_name in operator_names:
            rows = rows_by_operator[operator_name]
            mean_vector = self._mean_vector([item['features'] for item in rows])
            signature = self._mean_signature([item['signature'] for item in rows])
            implied_labels = self._mean_labels([item['labels'] for item in rows])
            store.upsert(
                VisualOperatorRecord(
                    key=f'operator:{operator_name}',
                    operator_name=operator_name,
                    feature_vector=mean_vector,
                    signature=signature,
                    metadata={
                        'record_type': 'operator_prototype',
                        'support': len(rows),
                        'implied_labels': implied_labels,
                        'examples': [
                            {
                                'image_path': item['image_path'],
                                'subject_id': item['subject_id'],
                            }
                            for item in rows[:8]
                        ],
                    },
                )
            )
        summary = VisualOperatorLearningSummary(
            input_path=str(labels_path),
            store_path=str(store_path),
            operator_names=operator_names,
            prototype_count=len(operator_names),
            examples=len(examples),
            targets=target_count,
        )
        if summary_output:
            Path(summary_output).write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

    def _resolve_candidate(self, candidates, candidate_map, subject_id: str, labels: List[str]):
        if subject_id and subject_id in candidate_map:
            return candidate_map[subject_id]
        for label in labels:
            chosen = self.feature_extractor.choose_default_subject(candidates, label)
            if chosen and chosen in candidate_map:
                return candidate_map[chosen]
        return None

    @staticmethod
    def _signature(features: Dict[str, float], has_parent: str | bool) -> List[str]:
        signature: list[str] = []
        if has_parent:
            signature.append('HAS_PARENT')
        if features.get('inside_parent', 0.0) >= 1.0:
            signature.append('INSIDE_PARENT')
        if features.get('boundary_attached', 0.0) >= 1.0:
            signature.append('BOUNDARY_ATTACHED')
        if features.get('near_top_band', 0.0) >= 1.0:
            signature.append('TOP_BAND')
        if features.get('near_side_band', 0.0) >= 1.0:
            signature.append('SIDE_BAND')
        if features.get('horizontal_elongation', 0.0) >= 2.3:
            signature.append('HORIZONTAL_ELONGATION')
        if features.get('vertical_elongation', 0.0) >= 2.3:
            signature.append('VERTICAL_ELONGATION')
        if features.get('touches_border', 0.0) >= 1.0:
            signature.append('TOUCHES_BORDER')
        if features.get('is_dominant', 0.0) >= 1.0:
            signature.append('DOMINANT_OBJECT')
        return sorted(signature)

    @staticmethod
    def _derive_operator_names(labels: List[str], signature: List[str]) -> List[str]:
        upper = {item.upper() for item in labels}
        signature_set = set(signature)
        operators: list[str] = []

        container_labels = {
            item for item in upper
            if any(token in item for token in ('CONTAINER', 'INTERIOR', 'BOX', 'DRAWER', 'CABINET', 'BOTTLE', 'CASE', 'BIN'))
        }
        opening_labels = {
            item for item in upper
            if any(token in item for token in ('OPENING', 'ZIPPER', 'HINGE', 'CAP', 'LID', 'DOOR'))
        }
        grasp_labels = {
            item for item in upper
            if any(token in item for token in ('STRAP', 'HANDLE', 'GRASP', 'KNOB', 'PULL', 'HOOK'))
        }
        tool_labels = {
            item for item in upper
            if 'TOOL' in item or 'BLADE' in item or 'HEAD' in item
        }

        if container_labels:
            operators.append('CONTAINER_BODY_OPERATOR')
        if opening_labels or {'TOP_BAND', 'BOUNDARY_ATTACHED'} <= signature_set:
            operators.append('OPENING_CONTROL_OPERATOR')
        if grasp_labels:
            operators.append('ATTACHED_GRASP_OPERATOR')
        if container_labels and opening_labels:
            operators.append('CONTAINER_ACCESS_OPERATOR')
        if any('DRAWER' in item for item in upper):
            operators.append('SLIDING_ACCESS_OPERATOR')
        if any('DOOR' in item for item in upper) or any('HINGE' in item for item in upper):
            operators.append('HINGED_ACCESS_OPERATOR')
        if any('BOTTLE' in item for item in upper) or any('CAP' in item for item in upper):
            operators.append('CAPPED_ACCESS_OPERATOR')
        if tool_labels and grasp_labels:
            operators.append('TOOL_GRASP_OPERATOR')
        if container_labels and grasp_labels:
            operators.append('CARRIABLE_CONTAINER_OPERATOR')
        if not operators and upper:
            operators.append(f'VISUAL_PATTERN_OPERATOR:{sorted(upper)[0]}')
        deduped: list[str] = []
        seen: set[str] = set()
        for item in operators:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    @staticmethod
    def _mean_vector(rows: List[Dict[str, float]]) -> Dict[str, float]:
        if not rows:
            return {}
        keys = sorted({key for row in rows for key in row})
        return {key: round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 6) for key in keys}

    @staticmethod
    def _mean_signature(rows: List[List[str]]) -> List[str]:
        counts: dict[str, int] = {}
        for row in rows:
            for item in set(row):
                counts[item] = counts.get(item, 0) + 1
        threshold = max(1, math.ceil(len(rows) * 0.5))
        return sorted(item for item, count in counts.items() if count >= threshold)

    @staticmethod
    def _mean_labels(rows: List[List[str]]) -> List[dict[str, Any]]:
        counts: dict[str, int] = {}
        for row in rows:
            for item in set(row):
                counts[item] = counts.get(item, 0) + 1
        total = max(1, len(rows))
        return [
            {'label': item, 'support': count, 'confidence': round(count / total, 4)}
            for item, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        ]

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .cluster_review import VisualClusterReviewStore
from .concept_memory import VisualConceptMemory, VisualConceptRecord
from .operator_learning import VisualOperatorPrototypeTrainer
from ..corpus_store import CorpusMemoryStore


@dataclass
class VisualReviewRetrainSummary:
    summary_path: str
    review_path: str
    labels_path: str
    concept_store_path: str
    operator_store_path: str
    approved_clusters: int
    approved_targets: int
    approved_labels: list[str]
    operator_summary: dict[str, Any]
    semop_memory_store_path: str = ''
    seeded_semop_memories: int = 0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class VisualApprovedReviewRetrainer:
    def export_and_retrain(
        self,
        summary_path: str | Path,
        review_path: str | Path,
        labels_path: str | Path,
        concept_store_path: str | Path,
        operator_store_path: str | Path,
        operator_summary_output: str | Path | None = None,
        semop_memory_store_path: str | Path | None = None,
        semop_memory_source: str = "vlso_review",
    ) -> VisualReviewRetrainSummary:
        summary_payload = json.loads(Path(summary_path).read_text(encoding='utf-8-sig'))
        decisions = VisualClusterReviewStore(review_path).load_decisions()
        rows_by_image: dict[str, dict[str, Any]] = {}
        concept_store = VisualConceptMemory(concept_store_path)
        approved_clusters = 0
        approved_targets = 0
        approved_labels: set[str] = set()

        for cluster in summary_payload.get('clusters', []):
            if not isinstance(cluster, dict):
                continue
            cluster_id = str(cluster.get('cluster_id', ''))
            decision = decisions.get(cluster_id, {})
            if str(decision.get('status', 'pending')) != 'approved':
                continue
            labels = [str(item).strip() for item in decision.get('approved_labels', []) if str(item).strip()]
            if not labels:
                labels = [
                    str(item.get('label', '')).strip()
                    for item in cluster.get('accepted_labels', [])
                    if isinstance(item, dict) and str(item.get('label', '')).strip()
                ]
            labels = list(dict.fromkeys(labels))
            if not labels:
                continue
            approved_clusters += 1
            approved_labels.update(labels)
            centroid = {str(k): float(v) for k, v in dict(cluster.get('centroid', {})).items()}
            support = int(cluster.get('support', 0) or 0)
            for label in labels:
                concept_store.upsert(
                    VisualConceptRecord(
                        key=f"approved:{label}:{cluster_id}",
                        label=label,
                        feature_vector=centroid,
                        metadata={
                            'record_type': 'approved_cluster_prototype',
                            'cluster_id': cluster_id,
                            'support': support,
                            'source': 'cluster_review',
                        },
                    )
                )
            for member in cluster.get('members', []):
                if not isinstance(member, dict):
                    continue
                image_path = str(member.get('image_path', '')).strip()
                subject_id = str(member.get('subject_id', '')).strip()
                if not image_path or not subject_id:
                    continue
                row = rows_by_image.setdefault(image_path, {'image_path': image_path, 'targets': []})
                existing = None
                for target in row['targets']:
                    if str(target.get('subject_id', '')) == subject_id:
                        existing = target
                        break
                if existing is None:
                    existing = {'subject_id': subject_id, 'positive_labels': [], 'negative_labels': []}
                    row['targets'].append(existing)
                    approved_targets += 1
                merged = list(dict.fromkeys([*existing['positive_labels'], *labels]))
                existing['positive_labels'] = merged

        labels_out = Path(labels_path)
        labels_out.parent.mkdir(parents=True, exist_ok=True)
        with labels_out.open('w', encoding='utf-8') as handle:
            for row in rows_by_image.values():
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')

        operator_summary = VisualOperatorPrototypeTrainer().train_jsonl(labels_out, operator_store_path, summary_output=operator_summary_output).model_dump()
        seeded_semop_memories = 0
        if semop_memory_store_path is not None:
            seeded_semop_memories = CorpusMemoryStore(semop_memory_store_path).seed_visual_review_memories(
                self._build_semop_memory_rows(summary_payload, decisions),
                source=semop_memory_source,
                split='train',
            )
        return VisualReviewRetrainSummary(
            summary_path=str(summary_path),
            review_path=str(review_path),
            labels_path=str(labels_path),
            concept_store_path=str(concept_store_path),
            operator_store_path=str(operator_store_path),
            approved_clusters=approved_clusters,
            approved_targets=approved_targets,
            approved_labels=sorted(approved_labels),
            operator_summary=operator_summary,
            semop_memory_store_path=str(semop_memory_store_path or ''),
            seeded_semop_memories=seeded_semop_memories,
        )


    @staticmethod
    def _build_semop_memory_rows(summary_payload: dict[str, Any], decisions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cluster in summary_payload.get('clusters', []):
            if not isinstance(cluster, dict):
                continue
            cluster_id = str(cluster.get('cluster_id', ''))
            decision = decisions.get(cluster_id, {})
            if str(decision.get('status', 'pending')) != 'approved':
                continue
            labels = [str(item).strip() for item in decision.get('approved_labels', []) if str(item).strip()]
            if not labels:
                labels = [str(item.get('label', '')).strip() for item in cluster.get('accepted_labels', []) if isinstance(item, dict) and str(item.get('label', '')).strip()]
            labels = list(dict.fromkeys(labels))
            if not labels:
                continue
            lowered = {label.lower() for label in labels}
            concepts: list[str] = []
            hidden_goals: list[str] = []
            premises: list[str] = []
            scripts: list[str] = []
            operators: list[str] = []
            if any('bag_like_container' in label for label in lowered):
                concepts.append('bag')
                hidden_goals.append('store_book_in_bag_goal')
                premises.extend(['open_access', 'available_space'])
                scripts.append('open_bag')
                operators.extend(['CONTAINER_ACCESS_OPERATOR', 'ATTACHED_GRASP_OPERATOR'])
            if any('drawer_like_container' in label for label in lowered):
                concepts.append('drawer')
                hidden_goals.append('retrieve_item_from_drawer_goal')
                premises.append('open_access')
                scripts.append('open_drawer')
                operators.extend(['ACCESS_PORT_OPERATOR', 'ACCESS_CONTROL_OPERATOR'])
            if any('door_panel' in label for label in lowered):
                concepts.append('door')
                hidden_goals.append('pass_through_door_goal')
                premises.append('open_access')
                scripts.append('open_door')
                operators.extend(['ACCESS_CONTROL_OPERATOR'])
            if any('access_opening_candidate' in label or 'zipper_like_part' in label for label in lowered):
                premises.append('open_access')
                scripts.append('open_access_path')
                operators.append('ACCESS_PORT_OPERATOR')
            if any('strap_like_part' in label or 'handle_like_part' in label or 'graspable_part' in label for label in lowered):
                operators.append('ATTACHED_GRASP_OPERATOR')
                scripts.append('grasp_then_open')
            if any('box_like_container' in label for label in lowered):
                concepts.append('box')
                hidden_goals.append('access_contents_goal')
                premises.append('open_access')
                scripts.append('open_box')
                operators.append('CONTAINER_ACCESS_OPERATOR')
            if any('cabinet_like_container' in label for label in lowered):
                concepts.append('cabinet')
                hidden_goals.append('access_contents_goal')
                premises.append('open_access')
                scripts.append('open_cabinet')
                operators.extend(['CONTAINER_ACCESS_OPERATOR', 'ACCESS_CONTROL_OPERATOR'])
            if any('bottle_like_container' in label for label in lowered) or any('jar_like_container' in label for label in lowered):
                concepts.extend(['bottle'] if any('bottle_like_container' in label for label in lowered) else ['jar'])
                hidden_goals.append('access_contents_goal')
                premises.append('open_access')
                scripts.append('uncap_container')
                operators.extend(['CONTAINER_ACCESS_OPERATOR', 'ACCESS_CONTROL_OPERATOR'])
            if any('bin_like_container' in label for label in lowered) or any('pouch_like_container' in label for label in lowered) or any('suitcase_like_container' in label for label in lowered):
                if any('bin_like_container' in label for label in lowered):
                    concepts.append('bin')
                    scripts.append('open_bin')
                if any('pouch_like_container' in label for label in lowered):
                    concepts.append('pouch')
                    scripts.append('open_pouch')
                if any('suitcase_like_container' in label for label in lowered):
                    concepts.append('suitcase')
                    scripts.append('open_suitcase')
                hidden_goals.append('access_contents_goal')
                premises.append('open_access')
                operators.extend(['CONTAINER_ACCESS_OPERATOR', 'ATTACHED_GRASP_OPERATOR'])
            if any('tool_grip_part' in label for label in lowered) or any('tool_like_object' in label for label in lowered):
                concepts.append('tool')
                hidden_goals.append('manipulate_tool_goal')
                scripts.append('grasp_tool')
                operators.append('ATTACHED_GRASP_OPERATOR')
            if any('lid_like_part' in label for label in lowered) or any('capped_opening' in label for label in lowered):
                premises.append('open_access')
                scripts.append('lift_lid_or_cap')
                operators.extend(['ACCESS_PORT_OPERATOR', 'ACCESS_CONTROL_OPERATOR'])
            if any('knob_like_part' in label for label in lowered) or any('hinge_like_part' in label for label in lowered):
                scripts.append('operate_access_control')
                operators.append('ACCESS_CONTROL_OPERATOR')
            concepts = list(dict.fromkeys(concepts))
            hidden_goals = list(dict.fromkeys(hidden_goals))
            premises = list(dict.fromkeys(premises))
            scripts = list(dict.fromkeys(scripts))
            operators = list(dict.fromkeys(operators))
            if not concepts and not premises and not operators:
                continue
            query_bits = [*concepts, *premises, *scripts]
            rows.append({
                'query': 'visual review ' + ' '.join(query_bits),
                'concepts': concepts,
                'hidden_goals': hidden_goals,
                'required_premises': premises,
                'scripts': scripts,
                'operators': operators,
            })
        return rows

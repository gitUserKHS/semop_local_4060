from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .affordance_classifier import WeakAffordanceClassifier
from .concept_memory import VisualConceptMemory
from .detector_adapters import DetectorOutputAdapter
from .geometry_backbones import GeometryPrimitiveBackbone
from .geometry_reasoner import VisualGeometryReasoner
from .geometry_topology import GeometryTopologyExtractor
from .image_parser import RawImageObservationParser
from .object_reasoner import VisualObjectReasoner
from .operator_learning import VisualOperatorMemory
from .types import SharedWorldModel, VisualObservation, VLSOEntity, VLSOOperator, VLSORelation


class VLSOVisualParser:
    def __init__(self, affordance_weights_path: str | None = None, concept_store_path: str | None = None, operator_store_path: str | None = None) -> None:
        self.detector_adapter = DetectorOutputAdapter()
        self.geometry_extractor = GeometryTopologyExtractor()
        self.geometry_reasoner = VisualGeometryReasoner()
        self.geometry_backbone = GeometryPrimitiveBackbone()
        self.image_parser = RawImageObservationParser()
        classifier = WeakAffordanceClassifier(affordance_weights_path) if affordance_weights_path else None
        concept_memory = VisualConceptMemory(concept_store_path) if concept_store_path else None
        operator_memory = VisualOperatorMemory(operator_store_path) if operator_store_path else None
        self.object_reasoner = VisualObjectReasoner(classifier=classifier, concept_memory=concept_memory, operator_memory=operator_memory)

    def parse(self, payload: str | dict[str, Any] | VisualObservation) -> tuple[SharedWorldModel, VisualObservation]:
        observation = self._coerce(payload)
        self.geometry_backbone.enrich_observation(observation)
        observation = self.object_reasoner.enrich_observation(observation)
        model = SharedWorldModel(query='visual_input')
        for item in observation.objects:
            entity_id = item.get('id') or item.get('label') or item.get('name')
            if not entity_id:
                continue
            label = item.get('label') or entity_id
            concept_labels = item.get('concept_labels') or []
            if concept_labels:
                label = f"{label} [{' / '.join(concept_labels[:2])}]"
            entity_type = item.get('kind') or item.get('type') or 'object'
            attributes = {key: value for key, value in item.items() if key not in {'id', 'label', 'name', 'kind', 'type', 'parts'}}
            model.add_entity(
                VLSOEntity(
                    id=str(entity_id),
                    label=str(label),
                    modality='vision',
                    entity_type=str(entity_type),
                    attributes=attributes,
                )
            )
            for part in item.get('parts', []):
                part_id = str(part)
                model.add_entity(VLSOEntity(id=part_id, label=part_id, modality='vision', entity_type='part'))
                model.add_relation(VLSORelation(source=part_id, relation='PART_OF', target=str(entity_id), modality='vision'))
        for item in observation.relations:
            source = item.get('source')
            relation = item.get('relation')
            target = item.get('target')
            if source and relation and target:
                model.add_relation(
                    VLSORelation(
                        source=str(source),
                        relation=str(relation),
                        target=str(target),
                        modality='vision',
                        confidence=float(item.get('confidence', 1.0) or 1.0),
                        attributes={k: v for k, v in item.items() if k not in {'source', 'relation', 'target', 'confidence'}},
                    )
                )
        for item in observation.affordances:
            subject = item.get('subject') or item.get('source')
            value = item.get('value') or item.get('affordance')
            if subject and value:
                affordance_name = str(value)
                model.add_operator(VLSOOperator(name=affordance_name, axis='action', description=f'visual affordance for {subject}', source_modality='vision', confidence=0.75))
                model.add_entity(VLSOEntity(id=str(subject), label=str(subject), modality='vision', entity_type='object'))
                for action_id, action_label in self._affordance_actions(affordance_name):
                    model.add_entity(VLSOEntity(id=action_id, label=action_label, modality='shared', entity_type='action'))
                    model.add_relation(
                        VLSORelation(
                            source=str(subject),
                            relation='AFFORDS',
                            target=action_id,
                            modality='shared',
                            confidence=0.74,
                            attributes={'source_affordance': affordance_name},
                        )
                    )
        for item in observation.states:
            subject = item.get('subject') or item.get('source')
            value = item.get('value') or item.get('state')
            if subject and value:
                state_id = f"state:{subject}:{value}".lower()
                model.add_entity(VLSOEntity(id=state_id, label=str(value), modality='vision', entity_type='state'))
                model.add_relation(VLSORelation(source=str(subject), relation='STATE', target=state_id, modality='vision'))
        structural_bindings = observation.metadata.get('structural_operators', [])
        if isinstance(structural_bindings, list):
            model.metadata['structural_operator_bindings'] = []
            for row in structural_bindings:
                if not isinstance(row, dict):
                    continue
                operator_name = str(row.get('operator_name', ''))
                subject = str(row.get('subject', ''))
                parent = str(row.get('parent', ''))
                confidence = float(row.get('confidence', 0.72) or 0.72)
                if not operator_name or not subject:
                    continue
                model.add_operator(VLSOOperator(name=operator_name, axis='structural', description=f'structural operator for {subject}', source_modality='vision', confidence=confidence))
                model.metadata['structural_operator_bindings'].append(row)
                if parent:
                    model.add_relation(VLSORelation(source=subject, relation='STRUCTURAL_PART_OF', target=parent, modality='vision', confidence=confidence))
        for item in observation.geometry:
            source = item.get('source')
            relation = item.get('relation')
            target = item.get('target')
            if source and relation and target:
                model.add_relation(
                    VLSORelation(
                        source=str(source),
                        relation=str(relation),
                        target=str(target),
                        modality='vision',
                        confidence=float(item.get('confidence', 1.0) or 1.0),
                        attributes={k: v for k, v in item.items() if k not in {'source', 'relation', 'target', 'confidence'}},
                    )
                )
        derived = self.geometry_extractor.extract(observation)
        for item in derived.derived_relations:
            model.add_relation(VLSORelation(source=str(item['source']), relation=str(item['relation']), target=str(item['target']), modality='vision', confidence=0.7))
        model.constraints.extend(item for item in observation.constraints if item not in model.constraints)
        model.constraints.extend(item for item in derived.derived_constraints if item not in model.constraints)
        if observation.metadata.get('image_path'):
            model.audit_trace.append('visual image attached')
        if observation.metadata.get('source') == 'raw_image':
            model.audit_trace.append('raw image parser generated visual observation')
        if observation.constraints:
            model.audit_trace.append('visual constraints observed')
        for item in observation.metadata.get('image_preprocess_audit', []):
            if item not in model.audit_trace:
                model.audit_trace.append(item)
        for item in observation.metadata.get('geometry_backbone_audit', []):
            if item not in model.audit_trace:
                model.audit_trace.append(item)
        for item in observation.metadata.get('object_reasoner_audit', []):
            if item not in model.audit_trace:
                model.audit_trace.append(item)
        model.audit_trace.extend(item for item in derived.audit_trace if item not in model.audit_trace)
        self.geometry_reasoner.enrich_world(model, observation)
        return model, observation

    def _coerce(self, payload: str | dict[str, Any] | VisualObservation) -> VisualObservation:
        if isinstance(payload, VisualObservation):
            return payload
        if isinstance(payload, str):
            stripped = payload.strip()
            image_path = Path(stripped)
            if image_path.exists() and image_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.ppm'}:
                return self.image_parser.parse_image(str(image_path)).observation
            if stripped.startswith('{'):
                parsed = json.loads(stripped)
                return self._coerce(parsed)
            return self._from_text_description(stripped)
        if payload.get('image_path') and not any(payload.get(key) for key in ('objects', 'relations', 'geometry', 'detections', 'predictions', 'instances', 'yolo', 'segments', 'annotations')):
            return self.image_parser.parse_image(str(payload['image_path'])).observation
        observation = self.detector_adapter.to_observation(payload)
        if payload.get('image_path') and 'image_path' not in observation.metadata:
            observation.metadata['image_path'] = payload['image_path']
        return observation

    def _from_text_description(self, text: str) -> VisualObservation:
        lowered = text.lower()
        observation = VisualObservation(metadata={'source': 'scene_text'})
        if 'bag' in lowered:
            observation.objects.append({'id': 'bag', 'label': 'bag', 'kind': 'container', 'bbox': [0, 0, 10, 10]})
        if 'zipper' in lowered:
            observation.objects.append({'id': 'zipper', 'label': 'zipper', 'kind': 'part', 'bbox': [2, 1, 8, 2]})
            observation.relations.append({'source': 'zipper', 'relation': 'PART_OF', 'target': 'bag'})
            observation.affordances.append({'subject': 'zipper', 'value': 'OPENABLE'})
        if 'closed' in lowered:
            observation.states.append({'subject': 'zipper', 'value': 'CLOSED'})
        if 'open' in lowered:
            observation.states.append({'subject': 'zipper', 'value': 'OPEN'})
        if 'triangle' in lowered:
            observation.objects.append({'id': 'triangle_shape', 'kind': 'shape', 'polygon': [[0, 0], [2, 0], [1, 2]]})
        if 'rectangle' in lowered:
            observation.objects.append({'id': 'rectangle_shape', 'kind': 'shape', 'polygon': [[0, 0], [4, 0], [4, 2], [0, 2]]})
        if 'blocked' in lowered:
            observation.constraints.append('path_blocked')
        return observation

    @staticmethod
    def _affordance_actions(value: str) -> list[tuple[str, str]]:
        upper = str(value or '').upper()
        actions: list[tuple[str, str]] = []
        if any(token in upper for token in {'OPEN', 'ACCESS', 'ZIPPER', 'EDGE_OPENING', 'CONTROL_PART', 'PORT'}):
            actions.append(('open_access_action', 'open access'))
        if any(token in upper for token in {'HANDLE', 'GRASP', 'STRAP', 'GRIP', 'KNOB', 'PULL'}):
            actions.append(('grasp_or_carry_action', 'grasp or carry'))
        if any(token in upper for token in {'INTERIOR', 'CONTAINER', 'PORTABLE_CONTAINER'}):
            actions.append(('store_items_action', 'store items'))
        if any(token in upper for token in {'TOOL', 'WEAPON'}):
            actions.append(('control_tool_action', 'control tool'))
        if not actions:
            normalized = ''.join(ch if ch.isalnum() else '_' for ch in str(value or '').lower()).strip('_') or 'visual_action'
            actions.append((f'afford:{normalized}', normalized.replace('_', ' ')))
        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for action_id, action_label in actions:
            if action_id in seen:
                continue
            seen.add(action_id)
            deduped.append((action_id, action_label))
        return deduped

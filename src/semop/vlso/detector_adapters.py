from __future__ import annotations

from typing import Any, Iterable

from .types import VisualObservation


class DetectorOutputAdapter:
    def to_observation(self, payload: dict[str, Any]) -> VisualObservation:
        metadata = self._metadata(payload)
        if any(key in payload for key in {'objects', 'relations', 'affordances', 'states', 'geometry'}):
            return VisualObservation(
                objects=list(payload.get('objects', [])),
                relations=list(payload.get('relations', [])),
                affordances=list(payload.get('affordances', [])),
                states=list(payload.get('states', [])),
                geometry=list(payload.get('geometry', [])),
                constraints=list(payload.get('constraints', [])),
                metadata=metadata,
            )
        if 'detections' in payload:
            return self._from_detection_list(payload.get('detections', []), payload, metadata)
        if 'predictions' in payload:
            return self._from_detection_list(payload.get('predictions', []), payload, metadata)
        if 'segments' in payload:
            return self._from_segment_list(payload.get('segments', []), payload, metadata)
        if 'annotations' in payload:
            return self._from_annotation_list(payload.get('annotations', []), payload, metadata)
        if 'instances' in payload and isinstance(payload['instances'], dict):
            return self._from_instances(payload['instances'], payload, metadata)
        if 'yolo' in payload and isinstance(payload['yolo'], dict):
            return self._from_detection_list(payload['yolo'].get('boxes', []), payload, metadata)
        return VisualObservation(constraints=list(payload.get('constraints', [])), metadata=metadata)

    def _from_detection_list(self, items: Iterable[dict[str, Any]], payload: dict[str, Any], metadata: dict[str, Any]) -> VisualObservation:
        item_list = list(items)
        observation = VisualObservation(constraints=list(payload.get('constraints', [])), metadata=metadata)
        label_counts: dict[str, int] = {}
        for item in item_list:
            label = str(item.get('label') or item.get('class_name') or item.get('name') or item.get('class') or '').strip().lower()
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1
        for index, item in enumerate(item_list):
            label = str(item.get('label') or item.get('class_name') or item.get('name') or item.get('class') or '').strip().lower()
            resolved_id = label if label and label_counts.get(label, 0) == 1 else None
            entry = self._normalize_item(item, index=index, bbox_mode=str(item.get('bbox_mode') or payload.get('bbox_mode') or 'xyxy'), resolved_id=resolved_id)
            observation.objects.append(entry)
            self._append_item_hints(observation, entry['id'], item)
        return observation

    def _from_segment_list(self, items: Iterable[dict[str, Any]], payload: dict[str, Any], metadata: dict[str, Any]) -> VisualObservation:
        observation = VisualObservation(constraints=list(payload.get('constraints', [])), metadata=metadata)
        for index, item in enumerate(items):
            entry = self._normalize_item(item, index=index, bbox_mode='xyxy')
            entry['kind'] = item.get('kind') or 'segment'
            if item.get('mask_area'):
                entry['pixel_count'] = int(item['mask_area'])
            observation.objects.append(entry)
            self._append_item_hints(observation, entry['id'], item)
        return observation

    def _from_annotation_list(self, items: Iterable[dict[str, Any]], payload: dict[str, Any], metadata: dict[str, Any]) -> VisualObservation:
        category_map = self._category_map(payload.get('categories', []))
        observation = VisualObservation(constraints=list(payload.get('constraints', [])), metadata=metadata)
        for index, item in enumerate(items):
            category_name = category_map.get(item.get('category_id')) or item.get('category_name') or item.get('label')
            normalized = dict(item)
            if category_name:
                normalized['label'] = category_name
            entry = self._normalize_item(normalized, index=index, bbox_mode='xywh')
            entry['kind'] = normalized.get('kind') or 'annotation'
            observation.objects.append(entry)
            self._append_item_hints(observation, entry['id'], normalized)
        return observation

    def _from_instances(self, instances: dict[str, Any], payload: dict[str, Any], metadata: dict[str, Any]) -> VisualObservation:
        boxes = instances.get('boxes') or instances.get('bboxes') or []
        labels = instances.get('labels') or instances.get('classes') or []
        scores = instances.get('scores') or []
        polygons = instances.get('polygons') or instances.get('masks') or []
        items = []
        for index, box in enumerate(boxes):
            item = {
                'label': labels[index] if index < len(labels) else f'instance_{index}',
                'bbox': box,
                'score': scores[index] if index < len(scores) else 0.0,
            }
            if index < len(polygons):
                item['segmentation'] = polygons[index]
            items.append(item)
        return self._from_detection_list(items, payload, metadata)

    def _normalize_item(self, item: dict[str, Any], index: int, bbox_mode: str = 'xyxy', resolved_id: str | None = None) -> dict[str, Any]:
        label = item.get('label') or item.get('class_name') or item.get('name') or item.get('class') or f'det_{index}'
        entity_id = item.get('id') or resolved_id or f"{str(label).lower().replace(' ', '_')}_{index}"
        bbox = self._normalize_bbox(item.get('bbox') or item.get('box') or item.get('xyxy'), bbox_mode=bbox_mode)
        polygon = self._normalize_polygon(item.get('polygon') or item.get('points') or item.get('segmentation') or item.get('mask'))
        entry = {
            'id': entity_id,
            'label': str(label),
            'kind': item.get('kind') or item.get('type') or 'object',
            'score': item.get('score') or item.get('confidence') or 0.0,
        }
        parent = item.get('parent_id') or item.get('part_of') or item.get('belongs_to')
        if parent:
            entry['parent_id'] = str(parent)
        attached_to = item.get('attached_to')
        if attached_to:
            entry['attached_to'] = str(attached_to)
        structural_role = item.get('structural_role') or item.get('role_hint')
        if structural_role:
            entry['structural_role'] = str(structural_role)
        relation_hints = item.get('relation_hints')
        if isinstance(relation_hints, list):
            entry['relation_hints'] = [str(v) for v in relation_hints]
        if bbox is not None:
            entry['bbox'] = bbox
        if polygon is not None:
            entry['polygon'] = polygon
        if item.get('parts'):
            entry['parts'] = list(item.get('parts', []))
        if item.get('mask_area'):
            entry['pixel_count'] = int(item['mask_area'])
        if item.get('area') and 'pixel_count' not in entry:
            entry['pixel_count'] = int(item['area'])
        if item.get('hole_count') is not None:
            entry['hole_count'] = int(item.get('hole_count') or 0)
        elif item.get('has_hole'):
            entry['hole_count'] = 1
        if item.get('bbox_fill_ratio') is not None:
            entry['bbox_fill_ratio'] = float(item.get('bbox_fill_ratio') or 0.0)
        if item.get('hull_fill_ratio') is not None:
            entry['hull_fill_ratio'] = float(item.get('hull_fill_ratio') or 0.0)
        if item.get('segmentation_confidence') is not None:
            entry['segmentation_confidence'] = float(item.get('segmentation_confidence') or 0.0)
        if item.get('part_of_confidence') is not None:
            entry['part_of_confidence'] = float(item.get('part_of_confidence') or 0.0)
        if item.get('attributes') and isinstance(item.get('attributes'), dict):
            entry.update({k: v for k, v in item['attributes'].items() if k not in entry})
        return entry

    def _append_item_hints(self, observation: VisualObservation, entity_id: str, item: dict[str, Any]) -> None:
        for value in item.get('affordances', []) or []:
            observation.affordances.append({'subject': entity_id, 'value': value})
        if item.get('state'):
            observation.states.append({'subject': entity_id, 'value': item.get('state')})
        parent = item.get('parent_id') or item.get('part_of') or item.get('belongs_to')
        if parent:
            observation.relations.append({'source': entity_id, 'relation': 'PART_OF', 'target': str(parent), 'confidence': float(item.get('part_of_confidence') or item.get('score') or 0.7)})
        attached_to = item.get('attached_to')
        if attached_to:
            observation.relations.append({'source': entity_id, 'relation': 'ATTACHED_TO', 'target': str(attached_to), 'confidence': float(item.get('part_of_confidence') or item.get('score') or 0.7)})
        structural_role = str(item.get('structural_role') or item.get('role_hint') or '').lower()
        if structural_role in {'opening', 'access_port', 'opening_band'}:
            observation.affordances.append({'subject': entity_id, 'value': 'ACCESS_PORT_CANDIDATE'})
        if structural_role in {'access_control', 'control', 'zipper', 'lid', 'door', 'cap'}:
            observation.affordances.append({'subject': entity_id, 'value': 'ACCESS_CONTROL_PART'})
        if structural_role in {'handle', 'grasp', 'strap', 'grip', 'knob'}:
            observation.affordances.append({'subject': entity_id, 'value': 'HANDLE_CANDIDATE'})
        for relation in item.get('relations', []) or []:
            if not isinstance(relation, dict):
                continue
            source = relation.get('source') or entity_id
            rel = relation.get('relation') or relation.get('type')
            target = relation.get('target')
            if source and rel and target:
                observation.relations.append({'source': str(source), 'relation': str(rel), 'target': str(target)})

    def _normalize_bbox(self, bbox: Any, bbox_mode: str = 'xyxy') -> list[float] | None:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        if bbox_mode == 'xywh':
            x, y, w, h = bbox
            return [float(x), float(y), float(x) + float(w), float(y) + float(h)]
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

    def _normalize_polygon(self, segmentation: Any) -> list[list[float]] | None:
        if isinstance(segmentation, list) and segmentation and all(isinstance(item, list) for item in segmentation):
            if segmentation and segmentation and all(isinstance(value, (int, float)) for value in segmentation[0]):
                flat = segmentation[0] if len(segmentation) == 1 else segmentation
                if flat and isinstance(flat[0], list):
                    return [[float(point[0]), float(point[1])] for point in flat if isinstance(point, list) and len(point) == 2]
                if len(flat) >= 6 and len(flat) % 2 == 0:
                    return [[float(flat[i]), float(flat[i + 1])] for i in range(0, len(flat), 2)]
            if segmentation and all(isinstance(point, list) and len(point) == 2 for point in segmentation):
                return [[float(point[0]), float(point[1])] for point in segmentation]
        if isinstance(segmentation, list) and len(segmentation) >= 6 and len(segmentation) % 2 == 0 and all(isinstance(value, (int, float)) for value in segmentation):
            return [[float(segmentation[i]), float(segmentation[i + 1])] for i in range(0, len(segmentation), 2)]
        return None

    def _category_map(self, categories: Iterable[dict[str, Any]]) -> dict[Any, str]:
        mapping: dict[Any, str] = {}
        for item in categories:
            if item.get('id') is not None and item.get('name'):
                mapping[item['id']] = str(item['name'])
        return mapping

    def _metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(payload.get('metadata', {}))
        for key in ('image_path', 'label', 'scene_id', 'source_path', 'detector', 'backbone'):
            if payload.get(key) and key not in metadata:
                metadata[key] = payload.get(key)
        return metadata

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class OpenImagesPayloadSummary:
    boxes_path: str
    labels_path: str
    output_path: str
    num_images: int
    num_annotations: int

    def model_dump(self) -> Dict[str, Any]:
        return {
            'boxes_path': self.boxes_path,
            'labels_path': self.labels_path,
            'output_path': self.output_path,
            'num_images': self.num_images,
            'num_annotations': self.num_annotations,
        }


class OpenImagesAnnotationAdapter:
    """Converts Open Images-style box annotations into VLSO detector payloads."""

    def load_label_map(self, path: str | Path) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        with Path(path).open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = str(row.get('LabelName') or row.get('label_name') or '').strip()
                value = str(row.get('DisplayName') or row.get('display_name') or key).strip()
                if key:
                    mapping[key] = value or key
        return mapping

    def build_payloads(
        self,
        boxes_path: str | Path,
        labels_path: str | Path,
        segmentation_path: str | Path | None = None,
        image_ids: Iterable[str] | None = None,
        limit_images: int | None = None,
    ) -> List[Dict[str, Any]]:
        label_map = self.load_label_map(labels_path)
        allowed = {str(item) for item in image_ids} if image_ids else None
        masks = self._load_masks(segmentation_path) if segmentation_path else {}
        grouped: Dict[str, Dict[str, Any]] = {}
        with Path(boxes_path).open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                image_id = str(row.get('ImageID') or row.get('image_id') or '').strip()
                if not image_id:
                    continue
                if allowed is not None and image_id not in allowed:
                    continue
                if limit_images is not None and image_id not in grouped and len(grouped) >= limit_images:
                    continue
                label_name = str(row.get('LabelName') or row.get('label_name') or '').strip()
                category_name = label_map.get(label_name, label_name or 'unknown')
                bbox = self._bbox_xywh(row)
                annotation = {
                    'id': f'{image_id}:{index}',
                    'category_name': category_name,
                    'label': category_name,
                    'source_label': label_name,
                    'bbox': bbox,
                    'kind': 'annotation',
                    'score': 1.0,
                    'attributes': self._attributes(row),
                }
                box_id = str(row.get('BoxID') or row.get('BoxId') or '').strip()
                if box_id and box_id in masks:
                    annotation['mask_path'] = masks[box_id]
                payload = grouped.setdefault(
                    image_id,
                    {
                        'image_id': image_id,
                        'metadata': {
                            'source': 'open_images',
                            'image_id': image_id,
                            'bbox_normalized': True,
                        },
                        'annotations': [],
                    },
                )
                payload['annotations'].append(annotation)
        return list(grouped.values())

    def save_payloads(self, payloads: List[Dict[str, Any]], output_path: str | Path) -> OpenImagesPayloadSummary:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        annotation_count = 0
        with output.open('w', encoding='utf-8') as handle:
            for row in payloads:
                annotation_count += len(row.get('annotations', []))
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        labels_path = ''
        boxes_path = ''
        if payloads:
            metadata = payloads[0].get('metadata', {})
            labels_path = str(metadata.get('labels_path', ''))
            boxes_path = str(metadata.get('boxes_path', ''))
        return OpenImagesPayloadSummary(
            boxes_path=boxes_path,
            labels_path=labels_path,
            output_path=str(output),
            num_images=len(payloads),
            num_annotations=annotation_count,
        )

    def build_and_save(
        self,
        boxes_path: str | Path,
        labels_path: str | Path,
        output_path: str | Path,
        segmentation_path: str | Path | None = None,
        image_ids: Iterable[str] | None = None,
        limit_images: int | None = None,
    ) -> OpenImagesPayloadSummary:
        payloads = self.build_payloads(boxes_path, labels_path, segmentation_path=segmentation_path, image_ids=image_ids, limit_images=limit_images)
        for item in payloads:
            item.setdefault('metadata', {})['boxes_path'] = str(boxes_path)
            item.setdefault('metadata', {})['labels_path'] = str(labels_path)
            if segmentation_path:
                item['metadata']['segmentation_path'] = str(segmentation_path)
        return self.save_payloads(payloads, output_path)

    def _load_masks(self, segmentation_path: str | Path) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        with Path(segmentation_path).open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                box_id = str(row.get('BoxID') or row.get('BoxId') or '').strip()
                mask_path = str(row.get('MaskPath') or row.get('mask_path') or '').strip()
                if box_id and mask_path:
                    mapping[box_id] = mask_path
        return mapping

    @staticmethod
    def _bbox_xywh(row: Dict[str, Any]) -> List[float]:
        xmin = float(row.get('XMin') or row.get('x_min') or 0.0)
        xmax = float(row.get('XMax') or row.get('x_max') or 0.0)
        ymin = float(row.get('YMin') or row.get('y_min') or 0.0)
        ymax = float(row.get('YMax') or row.get('y_max') or 0.0)
        return [
            round(xmin, 6),
            round(ymin, 6),
            round(max(0.0, xmax - xmin), 6),
            round(max(0.0, ymax - ymin), 6),
        ]

    @staticmethod
    def _attributes(row: Dict[str, Any]) -> Dict[str, Any]:
        keys = ['IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']
        result: Dict[str, Any] = {}
        for key in keys:
            if row.get(key) is not None and row.get(key) != '':
                result[key] = row.get(key)
        return result

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .types import VisualObservation


@dataclass
class VisualAffordanceCandidate:
    subject: str
    parent: str
    features: Dict[str, float]
    object_data: dict


class VisualAffordanceFeatureExtractor:
    def extract(self, observation: VisualObservation) -> List[VisualAffordanceCandidate]:
        objects = [item for item in observation.objects if isinstance(item.get("bbox"), list) and len(item.get("bbox")) == 4]
        if not objects:
            return []
        dominant = self._dominant_object(objects)
        dominant_id = str(dominant.get("id") or dominant.get("label") or "") if dominant else ""
        dominant_box = self._bbox(dominant) if dominant else None
        frame_area = self._frame_area(observation)
        rows: list[VisualAffordanceCandidate] = []
        for item in objects:
            subject = str(item.get("id") or item.get("label") or item.get("name") or "")
            if not subject:
                continue
            box = self._bbox(item)
            parent = dominant_id if dominant_id and subject != dominant_id and dominant_box and self._overlaps(dominant_box, box) else ""
            rows.append(
                VisualAffordanceCandidate(
                    subject=subject,
                    parent=parent,
                    features=self._feature_vector(item, box, dominant_id, dominant_box, frame_area, objects),
                    object_data=item,
                )
            )
        return rows

    def choose_default_subject(self, candidates: List[VisualAffordanceCandidate], label: str) -> str | None:
        if not candidates:
            return None
        if label == "BAG_LIKE_CONTAINER":
            return max(candidates, key=lambda item: item.features.get("is_dominant", 0.0)).subject
        scored = []
        for item in candidates:
            features = item.features
            score = 0.0
            if label in {"ZIPPER_LIKE_PART", "ACCESS_OPENING_CANDIDATE"}:
                score += 2.0 * features.get("near_top_band", 0.0)
                score += 2.0 * features.get("horizontal_elongation", 0.0)
                score += 1.0 * features.get("boundary_attached", 0.0)
                score -= 0.5 * features.get("relative_area", 0.0)
            elif label == "STRAP_LIKE_PART":
                score += 2.0 * features.get("near_side_band", 0.0)
                score += 2.0 * features.get("vertical_elongation", 0.0)
                score += 0.5 * features.get("boundary_attached", 0.0)
            elif label == "GRASPABLE_PART":
                score += max(features.get("vertical_elongation", 0.0), features.get("horizontal_elongation", 0.0))
                score += 0.5 * features.get("inside_parent", 0.0)
            scored.append((score, item.subject))
        scored.sort(reverse=True)
        return scored[0][1] if scored else None

    def _feature_vector(
        self,
        item: dict,
        box: tuple[float, float, float, float],
        dominant_id: str,
        dominant_box: tuple[float, float, float, float] | None,
        frame_area: float,
        objects: List[dict],
    ) -> Dict[str, float]:
        width = max(1.0, box[2] - box[0])
        height = max(1.0, box[3] - box[1])
        area = self._area(box)
        overlaps = sum(1 for other in objects if other is not item and self._overlaps(box, self._bbox(other)))
        inside_parent = 0.0
        boundary_attached = 0.0
        relative_area = 0.0
        near_top_band = 0.0
        near_side_band = 0.0
        if dominant_box is not None and dominant_id != str(item.get("id") or item.get("label") or item.get("name") or ""):
            inside_parent = 1.0 if self._contains(dominant_box, box) else 0.0
            boundary_attached = 1.0 if self._is_boundary_attached(dominant_box, box) else 0.0
            relative_area = area / max(1.0, self._area(dominant_box))
            near_top_band = 1.0 if box[1] <= dominant_box[1] + (dominant_box[3] - dominant_box[1]) * 0.28 else 0.0
            near_side_band = 1.0 if (box[0] <= dominant_box[0] + (dominant_box[2] - dominant_box[0]) * 0.2 or box[2] >= dominant_box[2] - (dominant_box[2] - dominant_box[0]) * 0.2) else 0.0
        return {
            "is_dominant": 1.0 if dominant_id == str(item.get("id") or item.get("label") or item.get("name") or "") else 0.0,
            "height_over_width": height / width,
            "horizontal_elongation": width / height,
            "vertical_elongation": height / width,
            "vertex_count_norm": min(1.0, self._vertex_count(item) / 12.0),
            "shape_complexity": min(1.0, max(0.0, (self._vertex_count(item) - 4) / 12.0)),
            "overlap_children": min(1.0, overlaps / 4.0),
            "area_ratio": area / max(1.0, frame_area),
            "touches_border": 1.0 if self._touches_frame_border(box, observation=None, frame_area=frame_area) else 0.0,
            "inside_parent": inside_parent,
            "boundary_attached": boundary_attached,
            "relative_area": relative_area,
            "near_top_band": near_top_band,
            "near_side_band": near_side_band,
        }

    def _frame_area(self, observation: VisualObservation) -> float:
        size = observation.metadata.get("image_size") or [0, 0]
        if isinstance(size, list) and len(size) == 2:
            return max(1.0, float(size[0]) * float(size[1]))
        boxes = [self._bbox(item) for item in observation.objects if isinstance(item.get("bbox"), list) and len(item.get("bbox")) == 4]
        if not boxes:
            return 1.0
        max_x = max(item[2] for item in boxes)
        max_y = max(item[3] for item in boxes)
        return max(1.0, max_x * max_y)

    def _touches_frame_border(self, box: tuple[float, float, float, float], observation: VisualObservation | None, frame_area: float) -> bool:
        if observation is not None:
            size = observation.metadata.get("image_size") or [0, 0]
            if isinstance(size, list) and len(size) == 2 and size[0] and size[1]:
                return box[0] <= 0 or box[1] <= 0 or box[2] >= float(size[0]) - 1 or box[3] >= float(size[1]) - 1
        side = frame_area ** 0.5
        margin = max(2.0, side * 0.015)
        return box[0] <= margin or box[1] <= margin

    def _dominant_object(self, objects: List[dict]) -> dict | None:
        if not objects:
            return None
        return max(objects, key=lambda item: float(item.get("pixel_count") or self._area(self._bbox(item))))

    @staticmethod
    def _vertex_count(item: dict) -> int:
        polygon = item.get("polygon") or []
        return len([point for point in polygon if isinstance(point, list) and len(point) == 2])

    @staticmethod
    def _bbox(item: dict) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = item.get("bbox")
        return float(x1), float(y1), float(x2), float(y2)

    @staticmethod
    def _area(bbox: tuple[float, float, float, float]) -> float:
        return max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    @staticmethod
    def _contains(parent: tuple[float, float, float, float], child: tuple[float, float, float, float]) -> bool:
        return parent[0] <= child[0] and parent[1] <= child[1] and parent[2] >= child[2] and parent[3] >= child[3]

    @staticmethod
    def _overlaps(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
        return not (left[2] <= right[0] or right[2] <= left[0] or left[3] <= right[1] or right[3] <= left[1])

    @staticmethod
    def _is_boundary_attached(parent: tuple[float, float, float, float], child: tuple[float, float, float, float]) -> bool:
        margin_x = max(2.0, (parent[2] - parent[0]) * 0.12)
        margin_y = max(2.0, (parent[3] - parent[1]) * 0.12)
        return abs(child[0] - parent[0]) <= margin_x or abs(child[1] - parent[1]) <= margin_y or abs(parent[2] - child[2]) <= margin_x or abs(parent[3] - child[3]) <= margin_y

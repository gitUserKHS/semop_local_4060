from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .types import VisualObservation


@dataclass
class GeometryTopologyResult:
    derived_relations: List[dict]
    derived_constraints: List[str]
    audit_trace: List[str]


class GeometryTopologyExtractor:
    def extract(self, observation: VisualObservation) -> GeometryTopologyResult:
        relations: List[dict] = []
        constraints: List[str] = []
        audit: List[str] = []
        objects = [item for item in observation.objects if isinstance(item.get("bbox"), list) and len(item.get("bbox")) == 4]
        for index, left in enumerate(objects):
            for right in objects[index + 1:]:
                left_box = self._bbox(left)
                right_box = self._bbox(right)
                left_id = str(left.get("id") or left.get("label") or left.get("name"))
                right_id = str(right.get("id") or right.get("label") or right.get("name"))
                if not left_id or not right_id:
                    continue
                if left_box[2] <= right_box[0]:
                    relations.append({"source": left_id, "relation": "LEFT_OF", "target": right_id})
                if right_box[2] <= left_box[0]:
                    relations.append({"source": left_id, "relation": "RIGHT_OF", "target": right_id})
                if left_box[3] <= right_box[1]:
                    relations.append({"source": left_id, "relation": "ABOVE", "target": right_id})
                if right_box[3] <= left_box[1]:
                    relations.append({"source": left_id, "relation": "BELOW", "target": right_id})
                if self._overlaps(left_box, right_box):
                    relations.append({"source": left_id, "relation": "OVERLAPS", "target": right_id})
                if self._contains(left_box, right_box):
                    relations.append({"source": left_id, "relation": "CONTAINS", "target": right_id})
                if self._contains(right_box, left_box):
                    relations.append({"source": right_id, "relation": "CONTAINS", "target": left_id})
        segments = [item for item in observation.geometry if item.get("points")]
        for index, left in enumerate(segments):
            for right in segments[index + 1:]:
                left_points = self._segment(left.get("points"))
                right_points = self._segment(right.get("points"))
                if not left_points or not right_points:
                    continue
                left_id = str(left.get("id") or left.get("label") or f"segment_{index}")
                right_id = str(right.get("id") or right.get("label") or f"segment_{index + 1}")
                if self._parallel(left_points, right_points):
                    relations.append({"source": left_id, "relation": "PARALLEL", "target": right_id})
                if self._intersects(left_points, right_points):
                    relations.append({"source": left_id, "relation": "INTERSECTS", "target": right_id})
        if relations:
            audit.append("geometry-topology extractor derived spatial relations")
        if any(item.get("relation") == "CONTAINS" for item in relations):
            constraints.append("containment_structure_detected")
        return GeometryTopologyResult(derived_relations=relations, derived_constraints=constraints, audit_trace=audit)

    @staticmethod
    def _bbox(item: dict) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = item.get("bbox")
        return float(x1), float(y1), float(x2), float(y2)

    @staticmethod
    def _overlaps(left: Tuple[float, float, float, float], right: Tuple[float, float, float, float]) -> bool:
        return not (left[2] <= right[0] or right[2] <= left[0] or left[3] <= right[1] or right[3] <= left[1])

    @staticmethod
    def _contains(left: Tuple[float, float, float, float], right: Tuple[float, float, float, float]) -> bool:
        return left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]

    @staticmethod
    def _segment(points: object) -> Tuple[Tuple[float, float], Tuple[float, float]] | None:
        if not isinstance(points, list) or len(points) != 2:
            return None
        a, b = points
        if not isinstance(a, list) or not isinstance(b, list) or len(a) != 2 or len(b) != 2:
            return None
        return (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))

    @staticmethod
    def _parallel(left, right) -> bool:
        lx = left[1][0] - left[0][0]
        ly = left[1][1] - left[0][1]
        rx = right[1][0] - right[0][0]
        ry = right[1][1] - right[0][1]
        return abs(lx * ry - ly * rx) < 1e-6

    @staticmethod
    def _intersects(left, right) -> bool:
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        a, b = left
        c, d = right
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, List

from .types import VisualObservation


@dataclass
class GeometryPrimitiveResult:
    primitive_count: int
    enriched_objects: int
    audit_trace: List[str]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class GeometryPrimitiveBackbone:
    """Extract geometry/topology primitives before semantic labels are used.

    This is a structural backbone, not a classifier. It converts polygons and
    segmentation-like regions into reusable geometric evidence such as edge
    segments, corner counts, right angles, parallel pairs, and equal-length pairs.
    """

    def enrich_observation(self, observation: VisualObservation) -> GeometryPrimitiveResult:
        primitive_count = 0
        enriched_objects = 0
        audit: list[str] = []
        existing_geometry = list(observation.geometry)
        for item in observation.objects:
            polygon = item.get('polygon')
            if not isinstance(polygon, list):
                continue
            points = [point for point in polygon if isinstance(point, list) and len(point) == 2]
            if len(points) < 3:
                continue
            primitive = self._analyze_polygon(str(item.get('id') or item.get('label') or 'shape'), points)
            item['corner_count'] = primitive['corner_count']
            item['right_angle_count'] = primitive['right_angle_count']
            item['parallel_edge_pair_count'] = primitive['parallel_edge_pair_count']
            item['equal_length_pair_count'] = primitive['equal_length_pair_count']
            item['convex_like'] = primitive['convex_like']
            item['dominant_orientation'] = primitive['dominant_orientation']
            item['symmetry_score'] = primitive['symmetry_score']
            item['axis_alignment_score'] = primitive['axis_alignment_score']
            item['closure_score'] = primitive['closure_score']
            item['geometry_signature'] = primitive['geometry_signature']
            existing_geometry.extend(primitive['edges'])
            primitive_count += len(primitive['edges'])
            enriched_objects += 1
        if enriched_objects:
            observation.geometry = existing_geometry
            audit.append('geometry primitive backbone derived edge and angle primitives')
            observation.metadata.setdefault('geometry_backbone_audit', [])
            for row in audit:
                if row not in observation.metadata['geometry_backbone_audit']:
                    observation.metadata['geometry_backbone_audit'].append(row)
        return GeometryPrimitiveResult(
            primitive_count=primitive_count,
            enriched_objects=enriched_objects,
            audit_trace=audit,
        )

    def _analyze_polygon(self, item_id: str, points: list[list[float]]) -> dict[str, Any]:
        edges: list[dict[str, Any]] = []
        vectors: list[tuple[float, float]] = []
        lengths: list[float] = []
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            vector = (float(next_point[0]) - float(point[0]), float(next_point[1]) - float(point[1]))
            length = math.hypot(vector[0], vector[1])
            vectors.append(vector)
            lengths.append(length)
            edges.append({
                'id': f'geom_edge:{item_id}:{index}',
                'kind': 'edge_segment',
                'owner': item_id,
                'points': [[float(point[0]), float(point[1])], [float(next_point[0]), float(next_point[1])]],
                'length': round(length, 4),
            })
        right_angles = 0
        for index in range(len(vectors)):
            left = vectors[index - 1]
            right = vectors[index]
            if self._is_perpendicular(left, right):
                right_angles += 1
        parallel_pairs = 0
        equal_length_pairs = 0
        for index, left in enumerate(vectors):
            for other_index in range(index + 1, len(vectors)):
                right = vectors[other_index]
                if self._is_parallel(left, right):
                    parallel_pairs += 1
                if abs(lengths[index] - lengths[other_index]) <= 1e-3:
                    equal_length_pairs += 1
        dominant_orientation = self._dominant_orientation(vectors)
        symmetry_score = self._symmetry_score(points)
        axis_alignment_score = self._axis_alignment_score(vectors)
        closure_score = self._closure_score(points)
        geometry_signature: list[str] = []
        if right_angles >= 2:
            geometry_signature.append('RIGHT_ANGLE_STRUCTURE')
        if parallel_pairs >= 1:
            geometry_signature.append('PARALLEL_EDGE_STRUCTURE')
        if equal_length_pairs >= 1:
            geometry_signature.append('EQUAL_LENGTH_STRUCTURE')
        if symmetry_score >= 0.7:
            geometry_signature.append('SYMMETRIC_STRUCTURE')
        if axis_alignment_score >= 0.7:
            geometry_signature.append('AXIS_ALIGNED_STRUCTURE')
        if closure_score >= 0.9:
            geometry_signature.append('CLOSED_BOUNDARY_STRUCTURE')
        if dominant_orientation != 'mixed':
            geometry_signature.append(f'{dominant_orientation.upper()}_ORIENTATION')
        return {
            'edges': edges,
            'corner_count': len(points),
            'right_angle_count': right_angles,
            'parallel_edge_pair_count': parallel_pairs,
            'equal_length_pair_count': equal_length_pairs,
            'convex_like': 1.0,
            'dominant_orientation': dominant_orientation,
            'symmetry_score': round(symmetry_score, 4),
            'axis_alignment_score': round(axis_alignment_score, 4),
            'closure_score': round(closure_score, 4),
            'geometry_signature': geometry_signature,
        }

    @staticmethod
    def _is_parallel(left: tuple[float, float], right: tuple[float, float]) -> bool:
        return abs(left[0] * right[1] - left[1] * right[0]) <= 1e-6

    @staticmethod
    def _is_perpendicular(left: tuple[float, float], right: tuple[float, float]) -> bool:
        return abs(left[0] * right[0] + left[1] * right[1]) <= 1e-6

    @staticmethod
    def _dominant_orientation(vectors: list[tuple[float, float]]) -> str:
        horizontal = 0
        vertical = 0
        for dx, dy in vectors:
            if abs(dx) >= abs(dy) * 1.5:
                horizontal += 1
            elif abs(dy) >= abs(dx) * 1.5:
                vertical += 1
        if horizontal and not vertical:
            return 'horizontal'
        if vertical and not horizontal:
            return 'vertical'
        if horizontal > vertical:
            return 'horizontal'
        if vertical > horizontal:
            return 'vertical'
        return 'mixed'


    @staticmethod
    def _symmetry_score(points: list[list[float]]) -> float:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        center_x = (min(xs) + max(xs)) / 2.0
        center_y = (min(ys) + max(ys)) / 2.0
        point_cloud = [(float(point[0]), float(point[1])) for point in points]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

        def _coverage(mirrored: list[tuple[float, float]]) -> float:
            matches = 0
            for mx, my in mirrored:
                if min(math.hypot(mx - px, my - py) for px, py in point_cloud) <= span * 0.08:
                    matches += 1
            return matches / float(len(point_cloud) or 1)

        mirrored_x = [(2.0 * center_x - px, py) for px, py in point_cloud]
        mirrored_y = [(px, 2.0 * center_y - py) for px, py in point_cloud]
        return max(_coverage(mirrored_x), _coverage(mirrored_y))

    @staticmethod
    def _axis_alignment_score(vectors: list[tuple[float, float]]) -> float:
        aligned = 0
        for dx, dy in vectors:
            if abs(dx) <= 1e-6 or abs(dy) <= 1e-6:
                aligned += 1
                continue
            ratio = min(abs(dx), abs(dy)) / max(abs(dx), abs(dy))
            if ratio <= 0.2:
                aligned += 1
        return aligned / float(len(vectors) or 1)

    @staticmethod
    def _closure_score(points: list[list[float]]) -> float:
        if len(points) < 3:
            return 0.0
        first = (float(points[0][0]), float(points[0][1]))
        last = (float(points[-1][0]), float(points[-1][1]))
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        gap = math.hypot(first[0] - last[0], first[1] - last[1])
        return max(0.0, 1.0 - (gap / span))

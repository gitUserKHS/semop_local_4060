from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List

from .types import SharedWorldModel, VLSOEntity, VLSOOperator, VLSORelation, VisualObservation


@dataclass
class GeometryReasoningResult:
    shape_hypotheses: List[dict]
    inferred_steps: List[str]
    audit_trace: List[str]


class VisualGeometryReasoner:
    def analyze(self, observation: VisualObservation) -> GeometryReasoningResult:
        hypotheses: List[dict] = []
        inferred_steps: List[str] = []
        audit_trace: List[str] = []
        for item in observation.objects:
            polygon = item.get("polygon")
            if not isinstance(polygon, list) or not polygon:
                continue
            points = [point for point in polygon if isinstance(point, list) and len(point) == 2]
            if len(points) < 3:
                continue
            hypothesis = self._geometry_from_polygon(str(item.get("id") or item.get("label")), points)
            if hypothesis:
                hypotheses.append(hypothesis)
        if hypotheses:
            audit_trace.append("geometry reasoner inferred shape hypotheses")
        for hypothesis in hypotheses:
            shape = hypothesis["shape"]
            if shape in {"rectangle", "square"}:
                inferred_steps.append(f"{hypothesis['id']} has opposite edges that behave as parallel pairs.")
            if hypothesis.get("triangle_kind") == "right_triangle":
                inferred_steps.append(f"{hypothesis['id']} contains a right angle between two edges.")
            if hypothesis.get("triangle_kind") == "isosceles_triangle":
                inferred_steps.append(f"{hypothesis['id']} has at least two equal-length edges.")
            if hypothesis.get("parallel_pairs"):
                inferred_steps.append(f"{hypothesis['id']} exposes parallel edge structure useful for geometric reasoning.")
        return GeometryReasoningResult(shape_hypotheses=hypotheses, inferred_steps=inferred_steps, audit_trace=audit_trace)

    def enrich_world(self, world: SharedWorldModel, observation: VisualObservation) -> None:
        result = self.analyze(observation)
        for item in result.shape_hypotheses:
            shape_id = f"shape:{item['id']}:{item['shape']}"
            world.add_entity(VLSOEntity(id=shape_id, label=item['shape'], modality="vision", entity_type="shape", attributes={
                "triangle_kind": item.get("triangle_kind", ""),
                "parallel_pairs": item.get("parallel_pairs", []),
                "perpendicular_pairs": item.get("perpendicular_pairs", []),
            }))
            world.add_relation(VLSORelation(source=str(item['id']), relation="SHAPE_IS", target=shape_id, modality="vision", confidence=0.8))
            world.add_operator(VLSOOperator(name=f"SHAPE_{item['shape'].upper()}", axis="geometry", description=f"shape hypothesis for {item['id']}", source_modality="vision", confidence=0.8))
            if item.get("triangle_kind"):
                kind = str(item["triangle_kind"])
                kind_id = f"shape:{item['id']}:{kind}"
                world.add_entity(VLSOEntity(id=kind_id, label=kind, modality="vision", entity_type="shape_property"))
                world.add_relation(VLSORelation(source=str(item['id']), relation="SHAPE_PROPERTY", target=kind_id, modality="vision", confidence=0.76))
                world.add_operator(VLSOOperator(name=f"TRIANGLE_KIND_{kind.upper()}", axis="geometry", description=f"triangle subtype for {item['id']}", source_modality="vision", confidence=0.76))
            for edge in item.get("edges", []):
                edge_id = edge["id"]
                world.add_entity(VLSOEntity(id=edge_id, label=edge_id, modality="vision", entity_type="line_segment", attributes={
                    "points": edge["points"],
                    "length": edge["length"],
                }))
                world.add_relation(VLSORelation(source=edge_id, relation="PART_OF", target=str(item['id']), modality="vision", confidence=0.74))
            for left, right in item.get("parallel_pairs", []):
                world.add_relation(VLSORelation(source=left, relation="PARALLEL", target=right, modality="vision", confidence=0.78))
                world.add_operator(VLSOOperator(name="PARALLEL_EDGE_PAIR", axis="geometry", description=f"{left} parallel to {right}", source_modality="vision", confidence=0.78))
            for left, right in item.get("perpendicular_pairs", []):
                world.add_relation(VLSORelation(source=left, relation="PERPENDICULAR", target=right, modality="vision", confidence=0.78))
                world.add_operator(VLSOOperator(name="PERPENDICULAR_EDGE_PAIR", axis="geometry", description=f"{left} perpendicular to {right}", source_modality="vision", confidence=0.78))
            for left, right in item.get("equal_length_pairs", []):
                world.add_relation(VLSORelation(source=left, relation="EQUAL_LENGTH", target=right, modality="vision", confidence=0.74))
        world.inferred_steps.extend(item for item in result.inferred_steps if item not in world.inferred_steps)
        world.audit_trace.extend(item for item in result.audit_trace if item not in world.audit_trace)

    def _geometry_from_polygon(self, item_id: str, points: list[list[float]]) -> dict | None:
        edges = []
        for index in range(len(points)):
            a = points[index]
            b = points[(index + 1) % len(points)]
            edge_id = f"edge:{item_id}:{index}"
            edges.append({
                "id": edge_id,
                "points": [a, b],
                "vector": (float(b[0]) - float(a[0]), float(b[1]) - float(a[1])),
                "length": round(self._distance(a, b), 4),
            })
        shape = self._shape_from_points(points, edges)
        if shape is None:
            return None
        parallel_pairs = self._edge_pairs(edges, self._parallel)
        perpendicular_pairs = self._edge_pairs(edges, self._perpendicular)
        equal_length_pairs = self._equal_length_pairs(edges)
        triangle_kind = ""
        if len(points) == 3:
            if perpendicular_pairs:
                triangle_kind = "right_triangle"
            elif equal_length_pairs:
                triangle_kind = "isosceles_triangle"
        return {
            "id": item_id,
            "shape": shape,
            "triangle_kind": triangle_kind,
            "edges": edges,
            "parallel_pairs": parallel_pairs,
            "perpendicular_pairs": perpendicular_pairs,
            "equal_length_pairs": equal_length_pairs,
        }

    def _shape_from_points(self, points: list[list[float]], edges: list[dict]) -> str | None:
        if len(points) == 3:
            return "triangle"
        if len(points) == 4:
            right_angles = len(self._edge_pairs(edges, self._perpendicular)) >= 3
            lengths = [float(edge["length"]) for edge in edges]
            all_equal = max(lengths) - min(lengths) <= 1e-3
            opposite_parallel = len(self._edge_pairs(edges, self._parallel)) >= 2
            if right_angles and all_equal:
                return "square"
            if right_angles:
                return "rectangle"
            if opposite_parallel:
                return "parallelogram"
            return "quadrilateral"
        return None

    @staticmethod
    def _edge_pairs(edges: list[dict], predicate) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for index, left in enumerate(edges):
            for right in edges[index + 1:]:
                if predicate(left["vector"], right["vector"]):
                    pairs.append((left["id"], right["id"]))
        return pairs

    @staticmethod
    def _equal_length_pairs(edges: list[dict]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for index, left in enumerate(edges):
            for right in edges[index + 1:]:
                if abs(float(left["length"]) - float(right["length"])) <= 1e-3:
                    pairs.append((left["id"], right["id"]))
        return pairs

    @staticmethod
    def _parallel(left: tuple[float, float], right: tuple[float, float]) -> bool:
        return abs(left[0] * right[1] - left[1] * right[0]) <= 1e-6

    @staticmethod
    def _perpendicular(left: tuple[float, float], right: tuple[float, float]) -> bool:
        return abs(left[0] * right[0] + left[1] * right[1]) <= 1e-6

    @staticmethod
    def _distance(a: list[float], b: list[float]) -> float:
        return math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))

from __future__ import annotations

from dataclasses import dataclass
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
            if isinstance(polygon, list) and polygon:
                shape = self._shape_from_polygon(polygon)
                if shape:
                    hypotheses.append({"id": str(item.get("id") or item.get("label")), "shape": shape})
        if hypotheses:
            audit_trace.append("geometry reasoner inferred shape hypotheses")
        for hypothesis in hypotheses:
            if hypothesis["shape"] == "rectangle":
                inferred_steps.append(f"{hypothesis['id']} behaves like a rectangle with parallel opposite sides.")
            if hypothesis["shape"] == "triangle":
                inferred_steps.append(f"{hypothesis['id']} behaves like a triangle with three boundary edges.")
        return GeometryReasoningResult(shape_hypotheses=hypotheses, inferred_steps=inferred_steps, audit_trace=audit_trace)

    def enrich_world(self, world: SharedWorldModel, observation: VisualObservation) -> None:
        result = self.analyze(observation)
        for item in result.shape_hypotheses:
            shape_id = f"shape:{item['id']}:{item['shape']}"
            world.add_entity(VLSOEntity(id=shape_id, label=item['shape'], modality="vision", entity_type="shape"))
            world.add_relation(VLSORelation(source=str(item['id']), relation="SHAPE_IS", target=shape_id, modality="vision", confidence=0.75))
            world.add_operator(VLSOOperator(name=f"SHAPE_{item['shape'].upper()}", axis="geometry", description=f"shape hypothesis for {item['id']}", source_modality="vision", confidence=0.75))
        world.inferred_steps.extend(item for item in result.inferred_steps if item not in world.inferred_steps)
        world.audit_trace.extend(item for item in result.audit_trace if item not in world.audit_trace)

    def _shape_from_polygon(self, polygon: list) -> str | None:
        points = [point for point in polygon if isinstance(point, list) and len(point) == 2]
        if len(points) == 3:
            return "triangle"
        if len(points) == 4 and self._axis_aligned_rectangle(points):
            side_x = abs(points[1][0] - points[0][0])
            side_y = abs(points[2][1] - points[1][1])
            if abs(side_x - side_y) < 1e-6:
                return "square"
            return "rectangle"
        return None

    @staticmethod
    def _axis_aligned_rectangle(points: list[list[float]]) -> bool:
        xs = sorted({float(point[0]) for point in points})
        ys = sorted({float(point[1]) for point in points})
        if len(xs) != 2 or len(ys) != 2:
            return False
        corners = {(xs[0], ys[0]), (xs[0], ys[1]), (xs[1], ys[0]), (xs[1], ys[1])}
        return corners == {(float(x), float(y)) for x, y in points}

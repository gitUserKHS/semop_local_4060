from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Iterable, List, Sequence

from .image_preprocess import ImageMaskPreprocessor
from .types import VisualObservation


@dataclass
class ImageParseResult:
    observation: VisualObservation
    width: int
    height: int
    component_count: int
    backend: str
    warnings: List[str] = field(default_factory=list)


class RawImageObservationParser:
    def __init__(
        self,
        background_threshold: int = 48,
        min_component_pixels: int = 8,
        max_components: int = 24,
        max_image_dimension: int = 960,
    ) -> None:
        self.background_threshold = background_threshold
        self.min_component_pixels = min_component_pixels
        self.max_components = max_components
        self.max_image_dimension = max_image_dimension
        self.preprocessor = ImageMaskPreprocessor()

    def parse_image(self, image_path: str) -> ImageParseResult:
        try:
            from PIL import Image
        except Exception as exc:
            return ImageParseResult(
                observation=VisualObservation(metadata={"image_path": image_path, "source": "raw_image", "parse_error": str(exc)}),
                width=0,
                height=0,
                component_count=0,
                backend="unavailable",
                warnings=[f"Pillow unavailable: {exc}"],
            )
        path = Path(image_path)
        if not path.exists():
            return ImageParseResult(
                observation=VisualObservation(metadata={"image_path": image_path, "source": "raw_image", "parse_error": "file_not_found"}),
                width=0,
                height=0,
                component_count=0,
                backend="missing_file",
                warnings=["raw image path was not found"],
            )
        try:
            image = Image.open(path).convert("RGBA")
        except Exception as exc:
            return ImageParseResult(
                observation=VisualObservation(metadata={"image_path": image_path, "source": "raw_image", "parse_error": str(exc)}),
                width=0,
                height=0,
                component_count=0,
                backend="image_open_failed",
                warnings=[f"raw image could not be opened: {exc}"],
            )

        original_width, original_height = image.size
        image, resize_metadata = self._resize_if_needed(image)
        width, height = image.size
        pixels = image.load()
        background = self._estimate_background(pixels, width, height)
        effective_threshold = self._adaptive_foreground_threshold(pixels, width, height, background)
        mask = [[False for _ in range(width)] for _ in range(height)]
        for y in range(height):
            for x in range(width):
                rgba = pixels[x, y]
                if self._is_foreground(rgba, background, effective_threshold):
                    mask[y][x] = True
        preprocess_result = self.preprocessor.refine_mask(mask)
        components = self._connected_components(preprocess_result.mask)
        dynamic_min_pixels = max(self.min_component_pixels, int(width * height * 0.0009))
        components = [item for item in components if len(item) >= dynamic_min_pixels]
        components, suppressed_components, preprocess_audit = self.preprocessor.suppress_border_components(components, width, height)
        observation = VisualObservation(
            metadata={
                "image_path": str(path),
                "source": "raw_image",
                "image_size": [width, height],
                "original_image_size": [original_width, original_height],
                "dynamic_min_pixels": dynamic_min_pixels,
                "effective_background_threshold": effective_threshold,
                "suppressed_components": suppressed_components,
            }
        )
        observation.metadata.update(resize_metadata)
        warnings: list[str] = []
        for item in preprocess_result.audit_trace + preprocess_audit:
            observation.metadata.setdefault("image_preprocess_audit", [])
            if item not in observation.metadata["image_preprocess_audit"]:
                observation.metadata["image_preprocess_audit"].append(item)
        for index, component in enumerate(components[: self.max_components], start=1):
            bbox = self._bbox(component)
            boundary = self._boundary_points(component)
            hull = self._convex_hull(boundary)
            polygon = self._simplify_polygon(hull)
            shape = self._shape_name(polygon, bbox, len(component))
            obj = {
                "id": f"shape_{index}",
                "label": shape,
                "kind": "shape",
                "bbox": list(bbox),
                "polygon": [[int(x), int(y)] for x, y in polygon],
                "pixel_count": len(component),
                "shape_hint": shape,
            }
            observation.objects.append(obj)
            if shape in {"rectangle", "square", "triangle", "circle"}:
                observation.affordances.append({"subject": obj["id"], "value": f"SHAPE_{shape.upper()}"})
        if len(components) > self.max_components:
            warnings.append("raw image parser truncated connected components")
        if suppressed_components:
            warnings.append("raw image parser suppressed dominant border components")
        if resize_metadata.get("resized"):
            warnings.append("raw image parser resized the source image for normalized analysis")
        if not observation.objects:
            warnings.append("raw image parser did not find foreground components")
        return ImageParseResult(
            observation=observation,
            width=width,
            height=height,
            component_count=len(components),
            backend="raw_image_connected_components",
            warnings=warnings,
        )

    def _resize_if_needed(self, image):
        width, height = image.size
        max_dim = max(width, height)
        if max_dim <= self.max_image_dimension:
            return image, {"resized": False, "resize_scale": 1.0}
        scale = self.max_image_dimension / max_dim
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = image.resize(new_size)
        return resized, {
            "resized": True,
            "resize_scale": round(scale, 6),
            "resized_from": [width, height],
        }

    def _estimate_background(self, pixels, width: int, height: int) -> tuple[int, int, int]:
        corners = [
            pixels[0, 0],
            pixels[max(0, width - 1), 0],
            pixels[0, max(0, height - 1)],
            pixels[max(0, width - 1), max(0, height - 1)],
        ]
        return tuple(int(sum(item[i] for item in corners) / len(corners)) for i in range(3))

    def _adaptive_foreground_threshold(self, pixels, width: int, height: int, background: tuple[int, int, int]) -> int:
        samples: list[float] = []
        step_x = max(1, width // 48)
        step_y = max(1, height // 48)
        for y in range(0, height, step_y):
            for x in range(0, width, step_x):
                rgba = pixels[x, y]
                if len(rgba) >= 4 and rgba[3] < 32:
                    continue
                distance = math.sqrt(sum((int(rgba[i]) - background[i]) ** 2 for i in range(3)))
                samples.append(distance)
        if not samples:
            return self.background_threshold
        samples.sort()
        p75 = samples[int(0.75 * (len(samples) - 1))]
        p90 = samples[int(0.90 * (len(samples) - 1))]
        adaptive = int(max(self.background_threshold * 0.65, p75 * 0.55, p90 * 0.33))
        return max(18, min(96, adaptive))

    def _is_foreground(self, rgba: Sequence[int], background: tuple[int, int, int], threshold: int) -> bool:
        if len(rgba) >= 4 and rgba[3] < 32:
            return False
        distance = math.sqrt(sum((int(rgba[i]) - background[i]) ** 2 for i in range(3)))
        return distance >= threshold

    def _connected_components(self, mask: list[list[bool]]) -> list[list[tuple[int, int]]]:
        height = len(mask)
        width = len(mask[0]) if height else 0
        seen = [[False for _ in range(width)] for _ in range(height)]
        components: list[list[tuple[int, int]]] = []
        for y in range(height):
            for x in range(width):
                if not mask[y][x] or seen[y][x]:
                    continue
                queue = deque([(x, y)])
                seen[y][x] = True
                component: list[tuple[int, int]] = []
                while queue:
                    cx, cy = queue.popleft()
                    component.append((cx, cy))
                    for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                        if 0 <= nx < width and 0 <= ny < height and mask[ny][nx] and not seen[ny][nx]:
                            seen[ny][nx] = True
                            queue.append((nx, ny))
                if len(component) >= self.min_component_pixels:
                    components.append(component)
        components.sort(key=len, reverse=True)
        return components

    def _bbox(self, component: Sequence[tuple[int, int]]) -> tuple[int, int, int, int]:
        xs = [point[0] for point in component]
        ys = [point[1] for point in component]
        return min(xs), min(ys), max(xs), max(ys)

    def _boundary_points(self, component: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        point_set = set(component)
        boundary = []
        for x, y in component:
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if (nx, ny) not in point_set:
                    boundary.append((x, y))
                    break
        return boundary or list(component)

    def _convex_hull(self, points: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
        pts = sorted(set(points))
        if len(pts) <= 1:
            return pts

        def cross(o: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: list[tuple[int, int]] = []
        for point in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)
        upper: list[tuple[int, int]] = []
        for point in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)
        return lower[:-1] + upper[:-1]

    def _simplify_polygon(self, points: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(points) <= 4:
            return list(points)
        simplified = list(points)
        changed = True
        while changed and len(simplified) > 6:
            changed = False
            next_points: list[tuple[int, int]] = []
            for index, point in enumerate(simplified):
                prev_point = simplified[index - 1]
                next_point = simplified[(index + 1) % len(simplified)]
                if self._is_nearly_collinear(prev_point, point, next_point):
                    changed = True
                    continue
                next_points.append(point)
            simplified = next_points or simplified
        return simplified

    def _is_nearly_collinear(self, a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> bool:
        area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        return area <= 2

    def _shape_name(self, polygon: Sequence[tuple[int, int]], bbox: tuple[int, int, int, int], pixel_count: int) -> str:
        width = max(1, bbox[2] - bbox[0] + 1)
        height = max(1, bbox[3] - bbox[1] + 1)
        area = width * height
        fill_ratio = pixel_count / max(1, area)
        vertex_count = len(polygon)
        if vertex_count == 3:
            return "triangle"
        if vertex_count == 4:
            if abs(width - height) <= 2 and fill_ratio >= 0.72:
                return "square"
            return "rectangle"
        if 0.65 <= fill_ratio <= 0.9 and abs(width - height) <= max(2, int(0.15 * max(width, height))):
            return "circle"
        if vertex_count > 4:
            return f"polygon_{vertex_count}"
        return "shape"

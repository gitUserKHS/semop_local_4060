from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class ImagePreprocessResult:
    mask: list[list[bool]]
    suppressed_components: list[dict]
    audit_trace: list[str]


class ImageMaskPreprocessor:
    def __init__(
        self,
        fill_neighbor_threshold: int = 5,
        isolated_neighbor_threshold: int = 1,
        border_area_ratio: float = 0.22,
        border_touch_threshold: int = 3,
    ) -> None:
        self.fill_neighbor_threshold = fill_neighbor_threshold
        self.isolated_neighbor_threshold = isolated_neighbor_threshold
        self.border_area_ratio = border_area_ratio
        self.border_touch_threshold = border_touch_threshold

    def refine_mask(self, mask: list[list[bool]]) -> ImagePreprocessResult:
        if not mask or not mask[0]:
            return ImagePreprocessResult(mask=mask, suppressed_components=[], audit_trace=[])
        filled = self._fill_small_holes(mask)
        cleaned = self._remove_isolated_foreground(filled)
        audit = []
        if cleaned != mask:
            audit.append("image preprocessor refined the raw foreground mask")
        return ImagePreprocessResult(mask=cleaned, suppressed_components=[], audit_trace=audit)

    def suppress_border_components(
        self,
        components: Sequence[Sequence[tuple[int, int]]],
        width: int,
        height: int,
    ) -> tuple[list[list[tuple[int, int]]], list[dict], list[str]]:
        if len(components) <= 1 or width <= 0 or height <= 0:
            return [list(component) for component in components], [], []
        image_area = max(1, width * height)
        kept: list[list[tuple[int, int]]] = []
        suppressed: list[dict] = []
        audit: list[str] = []
        for index, component in enumerate(components):
            bbox = self._bbox(component)
            border_touch_count = self._border_touch_count(bbox, width, height)
            area_ratio = len(component) / image_area
            dominant = index == 0 and len(components) > 1
            if dominant and area_ratio >= self.border_area_ratio and border_touch_count >= self.border_touch_threshold:
                suppressed.append(
                    {
                        "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                        "pixel_count": len(component),
                        "area_ratio": round(area_ratio, 4),
                        "border_touch_count": border_touch_count,
                        "reason": "dominant_border_frame",
                    }
                )
                continue
            kept.append(list(component))
        if suppressed:
            audit.append("image preprocessor suppressed dominant border-connected components")
        return kept, suppressed, audit

    def _fill_small_holes(self, mask: list[list[bool]]) -> list[list[bool]]:
        height = len(mask)
        width = len(mask[0]) if height else 0
        output = [row[:] for row in mask]
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if output[y][x]:
                    continue
                neighbors = self._foreground_neighbors(mask, x, y)
                if neighbors >= self.fill_neighbor_threshold:
                    output[y][x] = True
        return output

    def _remove_isolated_foreground(self, mask: list[list[bool]]) -> list[list[bool]]:
        height = len(mask)
        width = len(mask[0]) if height else 0
        output = [row[:] for row in mask]
        for y in range(height):
            for x in range(width):
                if not output[y][x]:
                    continue
                neighbors = self._foreground_neighbors(mask, x, y)
                if neighbors <= self.isolated_neighbor_threshold:
                    output[y][x] = False
        return output

    def _foreground_neighbors(self, mask: list[list[bool]], x: int, y: int) -> int:
        height = len(mask)
        width = len(mask[0]) if height else 0
        count = 0
        for ny in range(max(0, y - 1), min(height, y + 2)):
            for nx in range(max(0, x - 1), min(width, x + 2)):
                if nx == x and ny == y:
                    continue
                if mask[ny][nx]:
                    count += 1
        return count

    @staticmethod
    def _bbox(component: Sequence[tuple[int, int]]) -> tuple[int, int, int, int]:
        xs = [point[0] for point in component]
        ys = [point[1] for point in component]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _border_touch_count(bbox: tuple[int, int, int, int], width: int, height: int) -> int:
        left = 1 if bbox[0] <= 0 else 0
        top = 1 if bbox[1] <= 0 else 0
        right = 1 if bbox[2] >= width - 1 else 0
        bottom = 1 if bbox[3] >= height - 1 else 0
        return left + top + right + bottom

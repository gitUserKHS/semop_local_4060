from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class SyntheticGeometryScene:
    scene_id: str
    image_path: str
    visual_json_path: str
    query: str
    expected_entities: list[str]
    expected_relations: list[dict[str, str]]
    required_terms: list[str]
    forbidden_terms: list[str]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class SyntheticGeometrySceneBuilder:
    def build(self, output_dir: str | Path) -> list[SyntheticGeometryScene]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        scenes = [
            self._parallel_perpendicular_scene(root),
            self._square_scene(root),
            self._triangle_scene(root),
            self._right_triangle_scene(root),
            self._parallelogram_scene(root),
        ]
        return scenes

    def write_eval_jsonl(self, scenes: list[SyntheticGeometryScene], output_path: str | Path) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for scene in scenes:
                visual_json_path = Path(scene.visual_json_path)
                try:
                    visual_json_value = str(visual_json_path.relative_to(output.parent))
                except ValueError:
                    visual_json_value = str(visual_json_path)
                handle.write(
                    json.dumps(
                        {
                            "case_id": scene.scene_id,
                            "query": scene.query,
                            "visual_json": visual_json_value,
                            "expected_entities": scene.expected_entities,
                            "expected_relations": scene.expected_relations,
                            "required_terms": scene.required_terms,
                            "forbidden_terms": scene.forbidden_terms,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _parallel_perpendicular_scene(self, root: Path) -> SyntheticGeometryScene:
        image_path = root / "parallel_perpendicular_scene.png"
        visual_path = root / "parallel_perpendicular_scene.json"
        payload = {
            "label": "synthetic parallel and perpendicular line scene",
            "metadata": {"image_path": str(image_path)},
            "objects": [
                {"id": "region_outer", "kind": "region", "bbox": [0, 0, 12, 12]},
                {"id": "region_inner", "kind": "region", "bbox": [2, 2, 5, 5]},
                {"id": "region_right", "kind": "region", "bbox": [14, 2, 18, 6]},
            ],
            "geometry": [
                {"id": "line_ab", "points": [[0, 0], [6, 0]]},
                {"id": "line_cd", "points": [[1, 3], [7, 3]]},
                {"id": "line_ef", "points": [[3, -1], [3, 5]]},
            ],
        }
        self._write_visual_json(visual_path, payload)
        self._draw_image(
            image_path,
            lines=[((20, 40), (180, 40)), ((40, 100), (200, 100)), ((100, 10), (100, 150))],
        )
        return SyntheticGeometryScene(
            scene_id="geometry_parallel_perpendicular_synth",
            image_path=str(image_path),
            visual_json_path=str(visual_path),
            query="What geometric structure is visible here?",
            expected_entities=["region_outer", "region_inner", "region_right"],
            expected_relations=[
                {"source": "line_ab", "relation": "PARALLEL", "target": "line_cd"},
                {"source": "line_ab", "relation": "PERPENDICULAR", "target": "line_ef"},
            ],
            required_terms=["parallel", "perpendicular"],
            forbidden_terms=["clearer image"],
        )

    def _square_scene(self, root: Path) -> SyntheticGeometryScene:
        image_path = root / "square_scene.png"
        visual_path = root / "square_scene.json"
        payload = {
            "label": "synthetic square scene",
            "metadata": {"image_path": str(image_path)},
            "objects": [
                {
                    "id": "square_panel",
                    "kind": "shape",
                    "bbox": [2, 2, 8, 8],
                    "polygon": [[2, 2], [8, 2], [8, 8], [2, 8]],
                }
            ],
        }
        self._write_visual_json(visual_path, payload)
        self._draw_image(
            image_path,
            polygons=[[(40, 40), (160, 40), (160, 160), (40, 160)]],
        )
        return SyntheticGeometryScene(
            scene_id="geometry_square_synth",
            image_path=str(image_path),
            visual_json_path=str(visual_path),
            query="What shape is visible here?",
            expected_entities=["square_panel", "square"],
            expected_relations=[],
            required_terms=["square"],
            forbidden_terms=["clearer image"],
        )

    def _triangle_scene(self, root: Path) -> SyntheticGeometryScene:
        image_path = root / "triangle_scene.png"
        visual_path = root / "triangle_scene.json"
        payload = {
            "label": "synthetic triangle scene",
            "metadata": {"image_path": str(image_path)},
            "objects": [
                {
                    "id": "triangle_panel",
                    "kind": "shape",
                    "bbox": [1, 1, 9, 9],
                    "polygon": [[5, 1], [1, 9], [9, 9]],
                }
            ],
        }
        self._write_visual_json(visual_path, payload)
        self._draw_image(
            image_path,
            polygons=[[(100, 20), (20, 180), (180, 180)]],
        )
        return SyntheticGeometryScene(
            scene_id="geometry_triangle_synth",
            image_path=str(image_path),
            visual_json_path=str(visual_path),
            query="What shape is visible here?",
            expected_entities=["triangle_panel", "triangle"],
            expected_relations=[],
            required_terms=["triangle"],
            forbidden_terms=["clearer image"],
        )


    def _right_triangle_scene(self, root: Path) -> SyntheticGeometryScene:
        image_path = root / "right_triangle_scene.png"
        visual_path = root / "right_triangle_scene.json"
        payload = {
            "label": "synthetic right triangle scene",
            "metadata": {"image_path": str(image_path)},
            "objects": [
                {
                    "id": "right_triangle_panel",
                    "kind": "shape",
                    "bbox": [1, 1, 9, 9],
                    "polygon": [[1, 1], [1, 9], [9, 9]],
                }
            ],
        }
        self._write_visual_json(visual_path, payload)
        self._draw_image(
            image_path,
            polygons=[[(30, 30), (30, 180), (180, 180)]],
        )
        return SyntheticGeometryScene(
            scene_id="geometry_right_triangle_synth",
            image_path=str(image_path),
            visual_json_path=str(visual_path),
            query="What shape is visible here?",
            expected_entities=["right_triangle_panel", "triangle", "right_triangle"],
            expected_relations=[],
            required_terms=["triangle"],
            forbidden_terms=["clearer image"],
        )

    def _parallelogram_scene(self, root: Path) -> SyntheticGeometryScene:
        image_path = root / "parallelogram_scene.png"
        visual_path = root / "parallelogram_scene.json"
        payload = {
            "label": "synthetic parallelogram scene",
            "metadata": {"image_path": str(image_path)},
            "objects": [
                {
                    "id": "parallelogram_panel",
                    "kind": "shape",
                    "bbox": [1, 1, 10, 8],
                    "polygon": [[3, 1], [9, 1], [7, 7], [1, 7]],
                }
            ],
        }
        self._write_visual_json(visual_path, payload)
        self._draw_image(
            image_path,
            polygons=[[(70, 30), (180, 30), (140, 170), (30, 170)]],
        )
        return SyntheticGeometryScene(
            scene_id="geometry_parallelogram_synth",
            image_path=str(image_path),
            visual_json_path=str(visual_path),
            query="What shape is visible here?",
            expected_entities=["parallelogram_panel", "parallelogram"],
            expected_relations=[],
            required_terms=["parallelogram"],
            forbidden_terms=["clearer image"],
        )

    @staticmethod
    def _write_visual_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _draw_image(path: Path, lines: list[tuple[tuple[int, int], tuple[int, int]]] | None = None, polygons: list[list[tuple[int, int]]] | None = None) -> None:
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return
        image = Image.new("RGB", (220, 220), "white")
        drawer = ImageDraw.Draw(image)
        for line in lines or []:
            drawer.line(line, fill="black", width=5)
        for polygon in polygons or []:
            drawer.polygon(polygon, outline="black")
        image.save(path)

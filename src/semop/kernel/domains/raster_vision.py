from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any, TypeAlias

from ..catalog import normalize_predicate_name
from ..model import (
    Fact,
    FactStatus,
    Goal,
    OperatorFamily,
    Rule,
    SolveResult,
    Symbol,
    WorldState,
)
from .base import DomainInstance
from .vision import VisionProblem, VisionRelationGoal, VisionWorldAdapter


Rgb: TypeAlias = tuple[int, int, int]
PixelValue: TypeAlias = int | Sequence[int]
PixelPoint: TypeAlias = tuple[int, int]


@dataclass(frozen=True)
class RasterImage:
    """Small immutable RGB raster with dependency-free Netpbm loading."""

    rows: tuple[tuple[Rgb, ...], ...]
    background: Rgb | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("raster image requires at least one row")
        normalized = tuple(
            tuple(_normalize_pixel(pixel) for pixel in row) for row in self.rows
        )
        width = len(normalized[0])
        if width == 0:
            raise ValueError("raster image rows cannot be empty")
        if any(len(row) != width for row in normalized):
            raise ValueError("raster image rows must have equal width")
        background = (
            _normalize_pixel(self.background)
            if self.background is not None
            else _infer_background(normalized)
        )
        object.__setattr__(self, "rows", normalized)
        object.__setattr__(self, "background", background)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Sequence[PixelValue]],
        *,
        background: PixelValue | None = None,
        source: str = "",
    ) -> RasterImage:
        return cls(
            tuple(tuple(_normalize_pixel(pixel) for pixel in row) for row in rows),
            None if background is None else _normalize_pixel(background),
            source,
        )

    @classmethod
    def from_pnm(cls, path: str | Path) -> RasterImage:
        """Load ASCII PBM/PGM/PPM (P1/P2/P3) without Pillow or OpenCV."""

        source = Path(path)
        try:
            text = source.read_text(encoding="ascii")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "binary Netpbm is not supported; use ASCII P1, P2, or P3"
            ) from exc
        tokens: list[str] = []
        for line in text.splitlines():
            tokens.extend(line.split("#", 1)[0].split())
        if len(tokens) < 3:
            raise ValueError("invalid Netpbm header")
        magic = tokens[0]
        if magic not in {"P1", "P2", "P3"}:
            raise ValueError("supported Netpbm formats are ASCII P1, P2, and P3")
        try:
            width = int(tokens[1])
            height = int(tokens[2])
        except ValueError as exc:
            raise ValueError("Netpbm width and height must be integers") from exc
        if width <= 0 or height <= 0:
            raise ValueError("Netpbm width and height must be positive")

        cursor = 3
        if magic == "P1":
            max_value = 1
        else:
            if len(tokens) <= cursor:
                raise ValueError("Netpbm maximum value is missing")
            try:
                max_value = int(tokens[cursor])
            except ValueError as exc:
                raise ValueError("Netpbm maximum value must be an integer") from exc
            cursor += 1
            if not 1 <= max_value <= 65_535:
                raise ValueError("Netpbm maximum value must be between 1 and 65535")

        channels = 3 if magic == "P3" else 1
        expected = width * height * channels
        samples = tokens[cursor:]
        if len(samples) != expected:
            raise ValueError(
                f"Netpbm sample count mismatch: expected {expected}, got {len(samples)}"
            )
        try:
            values = [int(sample) for sample in samples]
        except ValueError as exc:
            raise ValueError("Netpbm samples must be integers") from exc
        if any(value < 0 or value > max_value for value in values):
            raise ValueError("Netpbm sample is outside the declared range")

        pixels: list[Rgb] = []
        if magic == "P1":
            pixels = [
                (0, 0, 0) if value == 1 else (255, 255, 255)
                for value in values
            ]
        elif magic == "P2":
            pixels = [
                (scaled, scaled, scaled)
                for scaled in (_scale_sample(value, max_value) for value in values)
            ]
        else:
            for index in range(0, len(values), 3):
                pixels.append(
                    tuple(
                        _scale_sample(value, max_value)
                        for value in values[index : index + 3]
                    )
                )
        rows = tuple(
            tuple(pixels[offset : offset + width])
            for offset in range(0, len(pixels), width)
        )
        return cls(rows, source=str(source))

    @classmethod
    def from_file(cls, path: str | Path) -> RasterImage:
        """Load Netpbm natively or common image formats through optional Pillow."""

        source = Path(path)
        if source.suffix.lower() in {".pbm", ".pgm", ".ppm", ".pnm"}:
            return cls.from_pnm(source)
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "PNG/JPEG loading requires optional Pillow; install the vision "
                "extra or use RasterImage.from_rows/from_pnm"
            ) from exc
        try:
            with Image.open(source) as opened:
                image = opened.convert("RGB")
                width, height = image.size
                pixels = list(image.getdata())
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot load raster image {source}: {exc}") from exc
        rows = tuple(
            tuple(pixels[offset : offset + width])
            for offset in range(0, width * height, width)
        )
        return cls(rows, source=str(source))

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def height(self) -> int:
        return len(self.rows)

    def digest(self) -> str:
        payload = bytearray()
        payload.extend(self.width.to_bytes(4, "big"))
        payload.extend(self.height.to_bytes(4, "big"))
        for row in self.rows:
            for pixel in row:
                payload.extend(pixel)
        return sha256(payload).hexdigest()


@dataclass(frozen=True)
class RasterVisionConfig:
    minimum_component_area: int = 2
    color_tolerance: int = 24
    background_tolerance: int = 12
    proposal_min_centroid_gap: float = 1.0
    max_pixels: int = 1_000_000
    max_components: int = 256

    def __post_init__(self) -> None:
        if self.minimum_component_area <= 0:
            raise ValueError("minimum component area must be positive")
        if not 0 <= self.color_tolerance <= 441:
            raise ValueError("color tolerance must be between 0 and 441")
        if not 0 <= self.background_tolerance <= 441:
            raise ValueError("background tolerance must be between 0 and 441")
        if self.proposal_min_centroid_gap < 0:
            raise ValueError("proposal centroid gap cannot be negative")
        if self.max_pixels <= 0:
            raise ValueError("max pixels must be positive")
        if self.max_components <= 0:
            raise ValueError("max components must be positive")


@dataclass(frozen=True)
class RasterObject:
    id: str
    color_name: str
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    mean_rgb: Rgb
    pixels: frozenset[PixelPoint] = field(repr=False)

    @property
    def left(self) -> int:
        return self.bbox[0]

    @property
    def top(self) -> int:
        return self.bbox[1]

    @property
    def right(self) -> int:
        return self.bbox[2]

    @property
    def bottom(self) -> int:
        return self.bbox[3]


@dataclass(frozen=True)
class RasterRelation:
    source: str
    predicate: str
    target: str
    confidence: float
    verified: bool


@dataclass(frozen=True)
class RasterAnalysis:
    width: int
    height: int
    background: Rgb
    image_digest: str
    objects: tuple[RasterObject, ...]
    relations: tuple[RasterRelation, ...]
    ignored_component_count: int = 0


@dataclass(frozen=True)
class RasterVisionProblem:
    image: RasterImage
    goals: tuple[
        VisionRelationGoal | VisionPropertyGoal | VisionCountGoal | VisionAreaGoal,
        ...,
    ] = ()
    query: str = ""
    config: RasterVisionConfig | None = None
    count_selectors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.image, RasterImage):
            raise TypeError("raster vision problem requires a RasterImage")
        object.__setattr__(
            self,
            "count_selectors",
            tuple(
                dict.fromkeys(
                    _normalize_count_selector(item) for item in self.count_selectors
                )
            ),
        )


@dataclass(frozen=True)
class VisionPropertyGoal:
    predicate: str
    subject: str
    label: str = ""


@dataclass(frozen=True)
class VisionCountGoal:
    selector: str
    expected: int
    label: str = ""

    def __post_init__(self) -> None:
        if self.expected < 0:
            raise ValueError("vision count goal cannot be negative")


@dataclass(frozen=True)
class VisionAreaGoal:
    larger: str
    smaller: str
    label: str = ""


@dataclass(frozen=True)
class _RawComponent:
    pixels: frozenset[PixelPoint]
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    mean_rgb: Rgb


@dataclass
class _RasterEntityRecord:
    id: str
    attributes: dict[str, Any]


@dataclass
class _RasterRelationRecord:
    source: str
    relation: str
    target: str
    confidence: float
    attributes: dict[str, Any]


@dataclass
class _RasterWorld:
    query: str
    entities: list[_RasterEntityRecord]
    relations: list[_RasterRelationRecord]


class RasterVisionAdapter:
    """Extract small, replayable spatial programs directly from RGB pixels."""

    def __init__(
        self,
        config: RasterVisionConfig | None = None,
        *,
        symbolic_adapter: VisionWorldAdapter | None = None,
    ) -> None:
        self.config = config or RasterVisionConfig()
        self.symbolic_adapter = symbolic_adapter or VisionWorldAdapter()

    def analyze(
        self,
        image: RasterImage,
        *,
        config: RasterVisionConfig | None = None,
    ) -> RasterAnalysis:
        active = config or self.config
        if image.width * image.height > active.max_pixels:
            raise ValueError(
                f"raster has {image.width * image.height} pixels; "
                f"limit is {active.max_pixels}"
            )
        components, ignored = _connected_components(image, active)
        objects = _name_components(components)
        if len(objects) > active.max_components:
            raise ValueError(
                f"raster segmentation found {len(objects)} components; "
                f"limit is {active.max_components}"
            )
        relations = _spatial_relations(objects, image, active)
        return RasterAnalysis(
            width=image.width,
            height=image.height,
            background=image.background or (255, 255, 255),
            image_digest=image.digest(),
            objects=objects,
            relations=relations,
            ignored_component_count=ignored,
        )

    def adapt(self, value: RasterVisionProblem) -> DomainInstance:
        if not isinstance(value, RasterVisionProblem):
            raise TypeError("raster vision payload must be a RasterVisionProblem")
        active = value.config or self.config
        analysis = self.analyze(value.image, config=active)
        if not analysis.objects:
            raise ValueError("raster segmentation found no eligible foreground object")

        resolved_goals: list[VisionRelationGoal] = []
        property_goals: list[VisionPropertyGoal] = []
        count_goals: list[VisionCountGoal] = []
        area_goals: list[VisionAreaGoal] = []
        selector_resolution: list[tuple[str, str]] = []
        for goal in value.goals:
            if isinstance(goal, VisionRelationGoal):
                source = _resolve_selector(goal.source, analysis.objects)
                target = _resolve_selector(goal.target, analysis.objects)
                resolved_goals.append(
                    VisionRelationGoal(goal.predicate, source, target, goal.label)
                )
                selector_resolution.extend(
                    ((goal.source, source), (goal.target, target))
                )
            elif isinstance(goal, VisionPropertyGoal):
                subject = _resolve_selector(goal.subject, analysis.objects)
                predicate = _canonical_property(goal.predicate)
                property_goals.append(
                    VisionPropertyGoal(predicate, subject, goal.label)
                )
                selector_resolution.append((goal.subject, subject))
            elif isinstance(goal, VisionCountGoal):
                count_goals.append(
                    VisionCountGoal(
                        _normalize_count_selector(goal.selector),
                        goal.expected,
                        goal.label,
                    )
                )
            elif isinstance(goal, VisionAreaGoal):
                larger = _resolve_selector(goal.larger, analysis.objects)
                smaller = _resolve_selector(goal.smaller, analysis.objects)
                area_goals.append(VisionAreaGoal(larger, smaller, goal.label))
                selector_resolution.extend(
                    ((goal.larger, larger), (goal.smaller, smaller))
                )
            else:  # pragma: no cover - frozen public union guards this boundary
                raise TypeError(f"unsupported raster vision goal: {type(goal).__name__}")

        entities = [
            _RasterEntityRecord(
                item.id,
                {
                    "verified": True,
                    "geometry_verified": True,
                    "source": "deterministic_raster_component",
                    "color_name": item.color_name,
                    "mean_rgb": item.mean_rgb,
                    "bbox": item.bbox,
                    "centroid": item.centroid,
                    "area": item.area,
                },
            )
            for item in analysis.objects
        ]
        relations = [
            _RasterRelationRecord(
                item.source,
                item.predicate,
                item.target,
                item.confidence,
                (
                    {
                        "geometry_verified": True,
                        "source": "deterministic_pixel_geometry",
                    }
                    if item.verified
                    else {
                        "perception_proposal": True,
                        "source": "centroid_only_proposal",
                    }
                ),
            )
            for item in analysis.relations
        ]
        world = _RasterWorld(value.query, entities, relations)
        symbolic = self.symbolic_adapter.adapt_world(
            world,
            tuple(resolved_goals),
        )
        symbolic = _extend_raster_reasoning(
            symbolic,
            analysis,
            tuple(property_goals),
            tuple(count_goals),
            tuple(area_goals),
            value.count_selectors,
        )
        object_metadata = tuple(
            {
                "id": item.id,
                "color_name": item.color_name,
                "area": item.area,
                "bbox": item.bbox,
                "centroid": item.centroid,
                "mean_rgb": item.mean_rgb,
                "bbox_width": item.right - item.left + 1,
                "bbox_height": item.bottom - item.top + 1,
                "fills_bounding_box": item.area
                == (item.right - item.left + 1) * (item.bottom - item.top + 1),
            }
            for item in analysis.objects
        )
        relation_metadata = tuple(asdict(item) for item in analysis.relations)
        return DomainInstance(
            registry=symbolic.registry,
            state=symbolic.state,
            goals=symbolic.goals,
            domain="vision",
            metadata={
                **symbolic.metadata,
                "input_kind": "raster",
                "raster_width": analysis.width,
                "raster_height": analysis.height,
                "background_rgb": analysis.background,
                "image_digest": analysis.image_digest,
                "image_source": value.image.source,
                "detected_objects": object_metadata,
                "detected_relations": relation_metadata,
                "ignored_component_count": analysis.ignored_component_count,
                "selector_resolution": tuple(selector_resolution),
                "property_goal_count": len(property_goals),
                "count_goal_count": len(count_goals),
                "area_goal_count": len(area_goals),
                "observation_count_selectors": value.count_selectors,
                "segmentation_config": asdict(active),
            },
        )

    @staticmethod
    def project(_value: RasterVisionProblem, _result: SolveResult) -> bool:
        # Raster payloads are immutable; the typed result already carries provenance.
        return False


def _extend_raster_reasoning(
    instance: DomainInstance,
    analysis: RasterAnalysis,
    property_goals: tuple[VisionPropertyGoal, ...],
    count_goals: tuple[VisionCountGoal, ...],
    area_goals: tuple[VisionAreaGoal, ...],
    requested_count_selectors: tuple[str, ...] = (),
) -> DomainInstance:
    registry = instance.registry
    entity_type = registry.types.resolve("Entity")
    visual_type = registry.types.resolve("VisualEntity")
    number_type = registry.types.ensure("Number", entity_type)
    color_type = registry.types.ensure("Color", entity_type)

    predicate_specs = (
        ("PIXEL_AREA", (visual_type, number_type)),
        ("HAS_COLOR", (visual_type, color_type)),
        ("FILLS_BOUNDING_BOX", (visual_type,)),
        ("EQUAL_EXTENT", (visual_type,)),
        ("FILLED_RECTANGLE", (visual_type,)),
        ("SQUARE", (visual_type,)),
        ("OBJECT_COUNT", (color_type, number_type)),
        ("LARGER_AREA", (visual_type, visual_type)),
    )
    for name, signature in predicate_specs:
        if name not in registry.predicates:
            registry.register_predicate(name, signature)

    item = registry.variable("item", visual_type)
    registry.register_operator(
        Rule(
            name="filled_bbox_is_rectangle",
            parameters=(item,),
            preconditions=(registry.atom("FILLS_BOUNDING_BOX", item),),
            effects=(registry.atom("FILLED_RECTANGLE", item),),
            description_ko="대상 {item}의 모든 bounding-box 픽셀이 채워져 직사각형임을 확인한다.",
        ),
        family=OperatorFamily.VERIFY.value,
        tags=("vision", "shape", "rectangle"),
    )
    registry.register_operator(
        Rule(
            name="equal_extent_rectangle_is_square",
            parameters=(item,),
            preconditions=(
                registry.atom("FILLED_RECTANGLE", item),
                registry.atom("EQUAL_EXTENT", item),
            ),
            effects=(registry.atom("SQUARE", item),),
            description_ko="직사각형 {item}의 가로와 세로가 같아 정사각형임을 확인한다.",
        ),
        family=OperatorFamily.COMPOSE.value,
        tags=("vision", "shape", "square"),
    )

    object_symbols = {
        obj.id: registry.symbol(obj.id, visual_type) for obj in analysis.objects
    }
    number_symbols: dict[int, Symbol] = {}
    color_symbols: dict[str, Symbol] = {}

    def number(value: int) -> Symbol:
        if value not in number_symbols:
            number_symbols[value] = registry.symbol(str(value), number_type)
        return number_symbols[value]

    def color(value: str) -> Symbol:
        if value not in color_symbols:
            color_symbols[value] = registry.symbol(value, color_type)
        return color_symbols[value]

    extra_facts: list[Fact] = []
    for obj in analysis.objects:
        symbol = object_symbols[obj.id]
        extra_facts.extend(
            (
                Fact(
                    registry.atom("PIXEL_AREA", symbol, number(obj.area)),
                    FactStatus.OBSERVED,
                    "deterministic_pixel_measurement",
                ),
                Fact(
                    registry.atom("HAS_COLOR", symbol, color(obj.color_name)),
                    FactStatus.OBSERVED,
                    "deterministic_component_color",
                ),
            )
        )
        width = obj.right - obj.left + 1
        height = obj.bottom - obj.top + 1
        if width >= 2 and height >= 2 and obj.area == width * height:
            extra_facts.append(
                Fact(
                    registry.atom("FILLS_BOUNDING_BOX", symbol),
                    FactStatus.OBSERVED,
                    "deterministic_pixel_geometry",
                )
            )
        if width == height:
            extra_facts.append(
                Fact(
                    registry.atom("EQUAL_EXTENT", symbol),
                    FactStatus.OBSERVED,
                    "deterministic_pixel_geometry",
                )
            )

    base_state = WorldState(instance.state.facts + tuple(extra_facts))
    extra_goals: list[Goal] = []
    for goal in property_goals:
        extra_goals.append(
            Goal(
                registry.atom(goal.predicate, object_symbols[goal.subject]),
                label=goal.label,
            )
        )

    color_counts = Counter(obj.color_name for obj in analysis.objects)
    count_selectors = tuple(
        dict.fromkeys(
            (
                "all",
                *sorted(color_counts),
                *requested_count_selectors,
                *(goal.selector for goal in count_goals),
            )
        )
    )
    for index, selector in enumerate(count_selectors):
        actual = (
            len(analysis.objects)
            if selector == "all"
            else sum(obj.color_name == selector for obj in analysis.objects)
        )
        selector_symbol = color(selector)
        effect = registry.atom("OBJECT_COUNT", selector_symbol, number(actual))
        guard_name = f"verify_object_count_{index:03d}"
        registry.register_guard(
            guard_name,
            _object_count_guard(selector, actual),
        )
        preconditions = tuple(
            [
                registry.atom("VISUAL_ENTITY", object_symbols[obj.id])
                for obj in analysis.objects
            ]
            + [
                registry.atom(
                    "HAS_COLOR",
                    object_symbols[obj.id],
                    color(obj.color_name),
                )
                for obj in analysis.objects
            ]
        )
        registry.register_operator(
            Rule(
                name=f"count_raster_objects_{index:03d}",
                parameters=(),
                preconditions=preconditions,
                effects=(effect,),
                guards=(guard_name,),
                description_ko=(
                    f"검증된 component 집합에서 {selector} 대상의 개수가 "
                    f"{actual}임을 센다."
                ),
            ),
            family=OperatorFamily.QUANTIFY.value,
            tags=("vision", "count", "closed_world"),
        )
    for goal in count_goals:
        extra_goals.append(
            Goal(
                registry.atom(
                    "OBJECT_COUNT",
                    color(goal.selector),
                    number(goal.expected),
                ),
                label=goal.label,
            )
        )

    prepared_area_pairs: list[tuple[str, str]] = []
    for first, second in combinations(analysis.objects, 2):
        if first.area == second.area:
            continue
        larger, smaller = (
            (first, second) if first.area > second.area else (second, first)
        )
        index = len(prepared_area_pairs)
        prepared_area_pairs.append((larger.id, smaller.id))
        larger_symbol = object_symbols[larger.id]
        smaller_symbol = object_symbols[smaller.id]
        guard_name = f"verify_larger_pixel_area_{index:03d}"
        registry.register_guard(
            guard_name,
            _larger_area_guard(larger.id, smaller.id),
        )
        registry.register_operator(
            Rule(
                name=f"compare_raster_area_{index:03d}",
                parameters=(),
                preconditions=(
                    registry.atom("PIXEL_AREA", larger_symbol, number(larger.area)),
                    registry.atom("PIXEL_AREA", smaller_symbol, number(smaller.area)),
                ),
                effects=(
                    registry.atom("LARGER_AREA", larger_symbol, smaller_symbol),
                ),
                guards=(guard_name,),
                description_ko=(
                    f"픽셀 면적 {larger.area}와 {smaller.area}를 비교해 "
                    f"{larger.id}의 면적이 더 큰지 검산한다."
                ),
            ),
            family=OperatorFamily.COMPARE.value,
            tags=("vision", "area", "compare"),
        )
    for goal in area_goals:
        extra_goals.append(
            Goal(
                registry.atom(
                    "LARGER_AREA",
                    object_symbols[goal.larger],
                    object_symbols[goal.smaller],
                ),
                label=goal.label,
            )
        )

    return DomainInstance(
        registry=registry,
        state=base_state,
        goals=instance.goals + tuple(extra_goals),
        domain=instance.domain,
        metadata={
            **instance.metadata,
            "raster_reasoning": {
                "closed_world_component_count": len(analysis.objects),
                "color_counts": tuple(sorted(color_counts.items())),
                "exact_pixel_areas": tuple(
                    sorted((obj.id, obj.area) for obj in analysis.objects)
                ),
                "prepared_count_selectors": count_selectors,
                "prepared_area_comparisons": tuple(prepared_area_pairs),
            },
        },
    )


class VisionInputAdapter:
    """Dispatch symbolic scenes and raw rasters through one vision boundary."""

    def __init__(
        self,
        *,
        symbolic: VisionWorldAdapter | None = None,
        raster: RasterVisionAdapter | None = None,
    ) -> None:
        self.symbolic = symbolic or VisionWorldAdapter()
        self.raster = raster or RasterVisionAdapter(symbolic_adapter=self.symbolic)

    def adapt(self, value: VisionProblem | RasterVisionProblem) -> DomainInstance:
        if isinstance(value, RasterVisionProblem):
            return self.raster.adapt(value)
        if isinstance(value, VisionProblem):
            return self.symbolic.adapt(value)
        raise TypeError(
            "vision payload must be VisionProblem or RasterVisionProblem"
        )

    def project(
        self,
        value: VisionProblem | RasterVisionProblem,
        result: SolveResult,
    ) -> bool:
        if isinstance(value, RasterVisionProblem):
            return self.raster.project(value, result)
        self.symbolic.project(value, result)
        return True


def _connected_components(
    image: RasterImage, config: RasterVisionConfig
) -> tuple[tuple[_RawComponent, ...], int]:
    background = image.background or (255, 255, 255)
    visited: set[PixelPoint] = set()
    components: list[_RawComponent] = []
    ignored = 0

    for y, row in enumerate(image.rows):
        for x, pixel in enumerate(row):
            point = (x, y)
            if point in visited or _rgb_distance(pixel, background) <= config.background_tolerance:
                continue
            seed = pixel
            queue: deque[PixelPoint] = deque((point,))
            visited.add(point)
            points: set[PixelPoint] = set()
            while queue:
                current_x, current_y = queue.popleft()
                points.add((current_x, current_y))
                for next_x, next_y in _neighbors4(
                    current_x, current_y, image.width, image.height
                ):
                    neighbor = (next_x, next_y)
                    if neighbor in visited:
                        continue
                    candidate = image.rows[next_y][next_x]
                    if _rgb_distance(candidate, background) <= config.background_tolerance:
                        continue
                    if _rgb_distance(candidate, seed) > config.color_tolerance:
                        continue
                    visited.add(neighbor)
                    queue.append(neighbor)

            if len(points) < config.minimum_component_area:
                ignored += 1
                continue
            ordered = sorted(points, key=lambda item: (item[1], item[0]))
            xs = [item[0] for item in ordered]
            ys = [item[1] for item in ordered]
            colors = [image.rows[item_y][item_x] for item_x, item_y in ordered]
            components.append(
                _RawComponent(
                    pixels=frozenset(points),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    centroid=(sum(xs) / len(xs), sum(ys) / len(ys)),
                    mean_rgb=tuple(
                        round(sum(color[channel] for color in colors) / len(colors))
                        for channel in range(3)
                    ),
                )
            )
    return tuple(components), ignored


def _name_components(
    components: Sequence[_RawComponent],
) -> tuple[RasterObject, ...]:
    ordered = sorted(
        components,
        key=lambda item: (
            _color_name(item.mean_rgb),
            item.bbox[1],
            item.bbox[0],
            item.bbox,
        ),
    )
    counts: Counter[str] = Counter()
    objects: list[RasterObject] = []
    for component in ordered:
        color_name = _color_name(component.mean_rgb)
        counts[color_name] += 1
        objects.append(
            RasterObject(
                id=f"{color_name}_{counts[color_name]}",
                color_name=color_name,
                area=len(component.pixels),
                bbox=component.bbox,
                centroid=component.centroid,
                mean_rgb=component.mean_rgb,
                pixels=component.pixels,
            )
        )
    return tuple(sorted(objects, key=lambda item: item.id))


def _spatial_relations(
    objects: Sequence[RasterObject],
    image: RasterImage,
    config: RasterVisionConfig,
) -> tuple[RasterRelation, ...]:
    relations: dict[tuple[str, str, str], RasterRelation] = {}
    by_id = {item.id: item for item in objects}

    horizontal = _strict_order_pairs(objects, axis="x")
    vertical = _strict_order_pairs(objects, axis="y")
    for source, target in _transitive_reduction(horizontal):
        relation = RasterRelation(source, "LEFT_OF", target, 1.0, True)
        relations[(source, relation.predicate, target)] = relation
    for source, target in _transitive_reduction(vertical):
        relation = RasterRelation(source, "ABOVE", target, 1.0, True)
        relations[(source, relation.predicate, target)] = relation

    for first, second in combinations(sorted(objects, key=lambda item: item.id), 2):
        if _touching(first.pixels, second.pixels):
            relation = RasterRelation(first.id, "TOUCHING", second.id, 1.0, True)
            relations[(first.id, relation.predicate, second.id)] = relation

        if (first.id, second.id) not in horizontal and (second.id, first.id) not in horizontal:
            delta_x = second.centroid[0] - first.centroid[0]
            if abs(delta_x) >= config.proposal_min_centroid_gap:
                source, target = (
                    (first.id, second.id) if delta_x > 0 else (second.id, first.id)
                )
                confidence = min(0.99, 0.5 + abs(delta_x) / max(1, image.width) / 2)
                relation = RasterRelation(source, "LEFT_OF", target, confidence, False)
                relations.setdefault((source, relation.predicate, target), relation)

        if (first.id, second.id) not in vertical and (second.id, first.id) not in vertical:
            delta_y = second.centroid[1] - first.centroid[1]
            if abs(delta_y) >= config.proposal_min_centroid_gap:
                source, target = (
                    (first.id, second.id) if delta_y > 0 else (second.id, first.id)
                )
                confidence = min(0.99, 0.5 + abs(delta_y) / max(1, image.height) / 2)
                relation = RasterRelation(source, "ABOVE", target, confidence, False)
                relations.setdefault((source, relation.predicate, target), relation)

    return tuple(
        relations[key]
        for key in sorted(relations)
        if key[0] in by_id and key[2] in by_id
    )


def _strict_order_pairs(
    objects: Sequence[RasterObject], *, axis: str
) -> frozenset[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for first, second in combinations(objects, 2):
        if axis == "x":
            if first.right < second.left:
                pairs.add((first.id, second.id))
            elif second.right < first.left:
                pairs.add((second.id, first.id))
        elif axis == "y":
            if first.bottom < second.top:
                pairs.add((first.id, second.id))
            elif second.bottom < first.top:
                pairs.add((second.id, first.id))
        else:
            raise ValueError(f"unsupported order axis: {axis}")
    return frozenset(pairs)


def _transitive_reduction(
    pairs: frozenset[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    nodes = {endpoint for pair in pairs for endpoint in pair}
    reduced = {
        pair
        for pair in pairs
        if not any(
            middle not in pair
            and (pair[0], middle) in pairs
            and (middle, pair[1]) in pairs
            for middle in nodes
        )
    }
    return tuple(sorted(reduced))


def _resolve_selector(selector: str, objects: Sequence[RasterObject]) -> str:
    normalized = selector.strip().lower().replace("-", "_").replace(" ", "_")
    by_id = {item.id: item for item in objects}
    if normalized in by_id:
        return normalized
    matches = [item.id for item in objects if item.color_name == normalized]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"unknown raster object selector {selector!r}; "
            f"available ids: {', '.join(sorted(by_id))}"
        )
    raise ValueError(
        f"ambiguous raster object selector {selector!r}; use one of "
        + ", ".join(sorted(matches))
    )


def _canonical_property(value: str) -> str:
    normalized = normalize_predicate_name(value)
    aliases = {
        "RECTANGLE": "FILLED_RECTANGLE",
        "FILLED_RECT": "FILLED_RECTANGLE",
    }
    canonical = aliases.get(normalized, normalized)
    if canonical not in {"FILLED_RECTANGLE", "SQUARE"}:
        raise ValueError(
            "supported raster property goals are FILLED_RECTANGLE and SQUARE"
        )
    return canonical


def _normalize_count_selector(value: str) -> str:
    normalized = "_".join(value.strip().lower().split())
    if not normalized:
        raise ValueError("vision count selector cannot be empty")
    if normalized in {"all", "all_objects", "objects", "전체"}:
        return "all"
    return normalized


def _object_count_guard(selector: str, expected: int):
    def verify(_binding, state: WorldState) -> bool:
        entities = {
            atom.arguments[0]
            for atom in state.eligible_atoms
            if atom.predicate.name == "VISUAL_ENTITY"
        }
        if selector == "all":
            return len(entities) == expected
        matching = {
            atom.arguments[0]
            for atom in state.eligible_atoms
            if atom.predicate.name == "HAS_COLOR"
            and str(atom.arguments[1]) == selector
        }
        return matching <= entities and len(matching) == expected

    return verify


def _larger_area_guard(larger: str, smaller: str):
    def verify(_binding, state: WorldState) -> bool:
        areas = {
            str(atom.arguments[0]): int(str(atom.arguments[1]))
            for atom in state.eligible_atoms
            if atom.predicate.name == "PIXEL_AREA"
        }
        return larger in areas and smaller in areas and areas[larger] > areas[smaller]

    return verify


def _touching(first: frozenset[PixelPoint], second: frozenset[PixelPoint]) -> bool:
    smaller, larger = (first, second) if len(first) <= len(second) else (second, first)
    return any(
        (x + dx, y + dy) in larger
        for x, y in smaller
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )


def _neighbors4(x: int, y: int, width: int, height: int) -> tuple[PixelPoint, ...]:
    return tuple(
        (next_x, next_y)
        for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        if 0 <= next_x < width and 0 <= next_y < height
    )


def _normalize_pixel(value: PixelValue | Rgb) -> Rgb:
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid pixel value")
    if isinstance(value, int):
        channels = (value, value, value)
    else:
        try:
            channels = tuple(int(channel) for channel in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid pixel value: {value!r}") from exc
        if len(channels) != 3:
            raise ValueError("RGB pixels require exactly three channels")
    if any(channel < 0 or channel > 255 for channel in channels):
        raise ValueError("pixel channels must be between 0 and 255")
    return channels


def _infer_background(rows: tuple[tuple[Rgb, ...], ...]) -> Rgb:
    height = len(rows)
    width = len(rows[0])
    border = [*rows[0], *rows[-1]]
    if height > 2:
        border.extend(rows[y][0] for y in range(1, height - 1))
        if width > 1:
            border.extend(rows[y][-1] for y in range(1, height - 1))
    counts = Counter(border)
    return min(counts, key=lambda color: (-counts[color], color))


def _rgb_distance(first: Rgb, second: Rgb) -> float:
    return sum((left - right) ** 2 for left, right in zip(first, second, strict=True)) ** 0.5


def _color_name(color: Rgb) -> str:
    red, green, blue = color
    high = max(color)
    low = min(color)
    if high - low <= 24:
        if high < 64:
            return "black"
        if low > 224:
            return "white"
        return "gray"
    if red > blue + 40 and green > blue + 40:
        return "yellow" if abs(red - green) <= 96 else "orange"
    if green > red + 40 and blue > red + 40:
        return "cyan"
    if red > green + 40 and blue > green + 40:
        return "magenta"
    if red > green + 40 and red > blue + 40:
        return "red"
    if green > red + 40 and green > blue + 40:
        return "green"
    if blue > red + 40 and blue > green + 40:
        return "blue"
    return f"rgb_{red:02x}{green:02x}{blue:02x}"


def _scale_sample(value: int, max_value: int) -> int:
    return round(value * 255 / max_value)

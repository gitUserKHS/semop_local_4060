from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from .adapters import MigrationMode
from .domains.language_text import LanguageTextProblem
from .domains.linear_equation import LinearEquationProblem
from .domains.numeric_comparison import NumericComparisonProblem
from .domains.raster_vision import (
    RasterImage,
    RasterVisionConfig,
    RasterVisionProblem,
    VisionAreaGoal,
    VisionCountGoal,
    VisionPropertyGoal,
)
from .domains.vision import VisionRelationGoal
from .runtime import DomainKind, TypedDomainRequest


_SUPPORTED_DOMAINS = frozenset(
    {DomainKind.LANGUAGE, DomainKind.MATH, DomainKind.VISION}
)
_COLOR_NAMES = {
    "black": (0, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "red": (255, 0, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
}


def canonical_json(value: Any) -> str:
    """Return the single JSON representation used by semantic artifacts."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"semantic value is not canonical JSON: {exc}") from exc


def encode_semantic_request(request: TypedDomainRequest) -> dict[str, Any]:
    """Encode one supported raw request without passing through typed IR."""

    if not isinstance(request, TypedDomainRequest):
        raise TypeError("semantic request codec requires TypedDomainRequest")
    domain = DomainKind(request.domain)
    if domain not in _SUPPORTED_DOMAINS:
        raise ValueError(f"semantic request codec does not support {domain.value}")
    if MigrationMode(request.mode) is MigrationMode.LEGACY:
        raise ValueError("legacy requests cannot enter semantic experience data")
    if domain is DomainKind.LANGUAGE:
        payload = _encode_language(request.payload)
    elif domain is DomainKind.MATH:
        payload = _encode_math(request.payload)
    else:
        payload = _encode_vision(request.payload)
    return {"domain": domain.value, "payload": payload}


def decode_semantic_request(
    domain: DomainKind | str,
    payload: Mapping[str, Any],
    *,
    mode: MigrationMode | str = MigrationMode.SHADOW,
) -> TypedDomainRequest:
    """Decode the canonical LMV payload used by benchmarks and experience data."""

    resolved_domain = DomainKind(domain)
    if resolved_domain not in _SUPPORTED_DOMAINS:
        raise ValueError(
            f"semantic request codec does not support {resolved_domain.value}"
        )
    if not isinstance(payload, Mapping):
        raise TypeError("semantic request payload must be an object")
    resolved_mode = MigrationMode(mode)
    if resolved_mode is MigrationMode.LEGACY:
        raise ValueError("legacy requests cannot enter semantic experience data")
    if resolved_domain is DomainKind.LANGUAGE:
        decoded = _decode_language(payload)
    elif resolved_domain is DomainKind.MATH:
        decoded = _decode_math(payload)
    else:
        decoded = _decode_vision(payload)
    return TypedDomainRequest(resolved_domain, decoded, resolved_mode)


def semantic_request_digest(request: TypedDomainRequest) -> str:
    encoded = encode_semantic_request(request)
    return sha256(canonical_json(encoded).encode("utf-8")).hexdigest()


def _encode_language(value: str | LanguageTextProblem) -> dict[str, Any]:
    if isinstance(value, LanguageTextProblem):
        problem = value
    elif isinstance(value, str):
        problem = LanguageTextProblem(value)
    else:
        raise TypeError("language semantic payload must be text")
    return {
        "text": problem.text,
        "source_context": problem.source_context,
        "use_legacy_heuristics": problem.use_legacy_heuristics,
    }


def _decode_language(payload: Mapping[str, Any]) -> LanguageTextProblem:
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("language semantic payload requires non-empty text")
    source_context = payload.get("source_context", "")
    if not isinstance(source_context, str):
        raise TypeError("language semantic source_context must be a string")
    heuristics = payload.get("use_legacy_heuristics", False)
    if type(heuristics) is not bool:
        raise TypeError("language semantic heuristics flag must be boolean")
    return LanguageTextProblem(text, source_context, heuristics)


def _encode_math(
    value: str | LinearEquationProblem | NumericComparisonProblem,
) -> dict[str, Any]:
    if isinstance(value, LinearEquationProblem):
        expression = value.equation
    elif isinstance(value, NumericComparisonProblem):
        expression = value.expression
    elif isinstance(value, str):
        expression = value
    else:
        raise TypeError(
            "math semantic payload must be a string, linear equation, or comparison"
        )
    if not expression.strip():
        raise ValueError("math semantic payload requires a non-empty expression")
    return {"expression": expression}


def _decode_math(payload: Mapping[str, Any]) -> str:
    expression = payload.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("math semantic payload requires non-empty expression")
    return expression


def _encode_vision(value: RasterVisionProblem) -> dict[str, Any]:
    if not isinstance(value, RasterVisionProblem):
        raise TypeError("vision semantic payload requires RasterVisionProblem")
    return {
        "rows": [
            [list(pixel) for pixel in row]
            for row in value.image.rows
        ],
        "background": list(value.image.background),
        "source": value.image.source,
        "goals": [_encode_vision_goal(goal) for goal in value.goals],
        "query": value.query,
        "config": asdict(value.config) if value.config is not None else None,
        "count_selectors": list(value.count_selectors),
    }


def _encode_vision_goal(goal: object) -> dict[str, Any]:
    if isinstance(goal, VisionRelationGoal):
        return {
            "kind": "relation",
            "predicate": goal.predicate,
            "source": goal.source,
            "target": goal.target,
            "label": goal.label,
        }
    if isinstance(goal, VisionPropertyGoal):
        return {
            "kind": "property",
            "predicate": goal.predicate,
            "subject": goal.subject,
            "label": goal.label,
        }
    if isinstance(goal, VisionCountGoal):
        return {
            "kind": "count",
            "selector": goal.selector,
            "expected": goal.expected,
            "label": goal.label,
        }
    if isinstance(goal, VisionAreaGoal):
        return {
            "kind": "area",
            "larger": goal.larger,
            "smaller": goal.smaller,
            "label": goal.label,
        }
    raise TypeError(f"unsupported raster vision goal: {type(goal).__name__}")


def _decode_vision(payload: Mapping[str, Any]) -> RasterVisionProblem:
    rows = payload.get("rows")
    if isinstance(rows, str) or not isinstance(rows, Sequence) or not rows:
        raise TypeError("vision semantic rows must be a non-empty list")
    decoded_rows: list[list[tuple[int, int, int]]] = []
    for row in rows:
        if isinstance(row, str) or not isinstance(row, Sequence) or not row:
            raise TypeError("each semantic raster row must be a non-empty list")
        decoded_rows.append([_decode_color(pixel) for pixel in row])
    background_value = payload.get("background")
    background = (
        _decode_color(background_value) if background_value is not None else None
    )
    source = payload.get("source", "semantic-codec")
    if not isinstance(source, str):
        raise TypeError("vision semantic source must be a string")
    query = payload.get("query", "")
    if not isinstance(query, str):
        raise TypeError("vision semantic query must be a string")
    image = RasterImage.from_rows(
        decoded_rows,
        background=background,
        source=source,
    )
    raw_goals = payload.get("goals")
    if (
        isinstance(raw_goals, str)
        or not isinstance(raw_goals, Sequence)
        or not raw_goals
    ):
        raise TypeError("vision semantic goals must be a non-empty list")
    goals = tuple(_decode_vision_goal(raw_goal) for raw_goal in raw_goals)
    count_selectors = payload.get("count_selectors", ())
    if isinstance(count_selectors, str) or not isinstance(
        count_selectors, Sequence
    ):
        raise TypeError("vision semantic count_selectors must be a list")
    if any(not isinstance(item, str) for item in count_selectors):
        raise TypeError("vision semantic count_selectors must contain strings")
    raw_config = payload.get("config")
    if raw_config is None:
        config = None
    elif isinstance(raw_config, Mapping):
        config = RasterVisionConfig(**dict(raw_config))
    else:
        raise TypeError("vision semantic config must be an object or null")
    return RasterVisionProblem(
        image=image,
        goals=goals,
        query=query,
        config=config,
        count_selectors=tuple(count_selectors),
    )


def _decode_vision_goal(
    raw_goal: Any,
) -> VisionRelationGoal | VisionPropertyGoal | VisionCountGoal | VisionAreaGoal:
    if not isinstance(raw_goal, Mapping):
        raise TypeError("vision semantic goal must be an object")
    kind = _string_field(raw_goal, "kind").strip().lower()
    label = _string_field(raw_goal, "label", default="")
    if kind == "relation":
        return VisionRelationGoal(
            _string_field(raw_goal, "predicate"),
            _string_field(raw_goal, "source"),
            _string_field(raw_goal, "target"),
            label,
        )
    if kind == "property":
        return VisionPropertyGoal(
            _string_field(raw_goal, "predicate"),
            _string_field(raw_goal, "subject"),
            label,
        )
    if kind == "count":
        expected = raw_goal.get("expected")
        if type(expected) is not int:
            raise TypeError("vision count goal expected must be an integer")
        return VisionCountGoal(
            _string_field(raw_goal, "selector"),
            expected,
            label,
        )
    if kind == "area":
        return VisionAreaGoal(
            _string_field(raw_goal, "larger"),
            _string_field(raw_goal, "smaller"),
            label,
        )
    raise ValueError(f"unsupported vision semantic goal kind: {kind or '<empty>'}")


def _decode_color(value: Any) -> tuple[int, int, int]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _COLOR_NAMES:
            return _COLOR_NAMES[normalized]
        if len(normalized) == 7 and normalized.startswith("#"):
            try:
                return tuple(
                    int(normalized[index : index + 2], 16)
                    for index in (1, 3, 5)
                )
            except ValueError as exc:
                raise ValueError(f"invalid RGB hex color: {value}") from exc
        raise ValueError(f"unknown raster color: {value}")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        channels = tuple(value)
        if len(channels) != 3 or any(
            type(channel) is not int for channel in channels
        ):
            raise TypeError("RGB colors must contain three integer channels")
        if any(channel < 0 or channel > 255 for channel in channels):
            raise ValueError("RGB channels must be between 0 and 255")
        return channels
    raise TypeError("raster color must be a name, #RRGGBB, or RGB list")


def _string_field(
    value: Mapping[str, Any],
    name: str,
    *,
    default: str | None = None,
) -> str:
    resolved = value.get(name, default)
    if not isinstance(resolved, str):
        raise TypeError(f"vision semantic {name} must be a string")
    return resolved

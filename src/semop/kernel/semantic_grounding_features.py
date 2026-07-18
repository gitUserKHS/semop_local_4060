from __future__ import annotations

from collections import Counter
from enum import Enum
from fractions import Fraction
import json

from .contracts import DomainInstance
from .grounding import (
    GroundingCandidate,
    GroundingLearningExample,
    grounding_payload_digest,
    make_grounding_candidate,
)
from .runtime import DomainKind
from .semantic_benchmark import SemanticBenchmarkCase


class SemanticGroundingFeatureProfile(str, Enum):
    """Label-independent views used to audit grounding-policy shortcuts."""

    FULL = "full"
    DOMAIN_LOCAL_MARGIN = "domain_local_margin"
    NO_SHARED_MARGIN = "no_shared_margin"
    NO_MARGIN = "no_margin"
    NO_SURFACE = "no_surface"
    PRIMITIVES_ONLY = "primitives_only"


def semantic_sensor_features(
    case: SemanticBenchmarkCase,
    instance: DomainInstance,
) -> tuple[tuple[str, float], ...]:
    """Return label-free primitive measurements for one typed goal candidate."""

    audit = instance.grounding_trace.audit(instance.state.facts)
    features: dict[str, float] = {
        "context.record_count": _bounded_count(audit.record_count),
        "context.fact_count": _bounded_count(len(instance.state.facts)),
        "context.proof_fact_count": _bounded_count(audit.proof_eligible_count),
        "context.goal_count": _bounded_count(len(instance.goals)),
    }
    if case.domain is DomainKind.LANGUAGE:
        _language_sensor_features(instance, features)
    elif case.domain is DomainKind.MATH:
        _math_sensor_features(instance, features)
    elif case.domain is DomainKind.VISION:
        _vision_sensor_features(instance, features)
    return tuple(sorted(features.items()))


def semantic_candidate_statement(
    case: SemanticBenchmarkCase,
    instance: DomainInstance,
) -> str:
    """Expose useful surface syntax without hashing an entire raster JSON blob."""

    goal = str(instance.goals[0].atom)
    if case.domain is DomainKind.LANGUAGE:
        surface = str(case.payload["text"])
    elif case.domain is DomainKind.MATH:
        surface = str(case.payload["expression"])
    else:
        surface = (
            f"raster {instance.metadata.get('raster_width', 0)}x"
            f"{instance.metadata.get('raster_height', 0)} with "
            f"{len(instance.metadata.get('detected_objects', ()))} components"
        )
    return f"{surface} -> proposed typed goal {goal}"


def profile_semantic_grounding_candidate(
    candidate: GroundingCandidate,
    profile: SemanticGroundingFeatureProfile | str,
) -> GroundingCandidate:
    """Create one deterministic, label-free ablation view of a candidate."""

    selected = SemanticGroundingFeatureProfile(profile)
    if selected is SemanticGroundingFeatureProfile.FULL:
        return candidate

    features = candidate.sensor_features
    if selected is SemanticGroundingFeatureProfile.DOMAIN_LOCAL_MARGIN:
        features = tuple(
            (
                f"{candidate.domain}.support_margin"
                if name == "operator.support_margin"
                else name,
                value,
            )
            for name, value in features
        )
    elif selected in {
        SemanticGroundingFeatureProfile.NO_SHARED_MARGIN,
        SemanticGroundingFeatureProfile.PRIMITIVES_ONLY,
    }:
        features = tuple(
            (name, value)
            for name, value in features
            if name != "operator.support_margin"
        )
    elif selected is SemanticGroundingFeatureProfile.NO_MARGIN:
        features = tuple(
            (name, value)
            for name, value in features
            if not _is_margin_feature(name)
        )

    statement = candidate.statement
    if selected in {
        SemanticGroundingFeatureProfile.NO_SURFACE,
        SemanticGroundingFeatureProfile.PRIMITIVES_ONLY,
    }:
        statement = "typed grounding candidate"

    return make_grounding_candidate(
        domain=candidate.domain,
        statement=statement,
        atom=candidate.atom,
        producer_id=candidate.producer_id,
        source=candidate.source,
        input_digest=candidate.input_digest,
        evidence=candidate.evidence,
        assertion_status=candidate.assertion_status,
        confidence=candidate.confidence,
        sensor_features=features,
    )


def profile_semantic_grounding_examples(
    examples: tuple[GroundingLearningExample, ...]
    | list[GroundingLearningExample],
    profile: SemanticGroundingFeatureProfile | str,
) -> tuple[GroundingLearningExample, ...]:
    """Apply a feature profile while retaining verifier/reviewer label lineage."""

    selected = SemanticGroundingFeatureProfile(profile)
    if selected is SemanticGroundingFeatureProfile.FULL:
        return tuple(examples)
    transformed: list[GroundingLearningExample] = []
    for example in examples:
        candidate = profile_semantic_grounding_candidate(example.candidate, selected)
        record_digest = grounding_payload_digest(
            json.dumps(
                {
                    "candidate_digest": candidate.candidate_digest,
                    "original_record_digest": example.record_digest,
                    "profile": selected.value,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        transformed.append(
            GroundingLearningExample(
                candidate=candidate,
                label=example.label,
                authority=example.authority,
                record_digest=record_digest,
                weight=example.weight,
            )
        )
    return tuple(transformed)


def _language_sensor_features(
    instance: DomainInstance,
    features: dict[str, float],
) -> None:
    claims = tuple(instance.metadata.get("claims", ()))
    relation_counts = Counter(str(claim.get("relation", "")) for claim in claims)
    for relation in ("GOAL", "REQUIRES", "SATISFIED", "BLOCKED"):
        features[f"language.{relation.lower()}_count"] = _bounded_count(
            relation_counts[relation]
        )
    contradictions = set(instance.metadata.get("contradictions", ()))
    required = {
        str(claim["arguments"][1])
        for claim in claims
        if claim.get("verified")
        and claim.get("relation") == "REQUIRES"
        and len(claim.get("arguments", ())) == 2
    }
    satisfied = {
        str(claim["arguments"][0])
        for claim in claims
        if claim.get("verified")
        and claim.get("relation") == "SATISFIED"
        and claim.get("arguments")
    }
    blocked = {
        str(claim["arguments"][0])
        for claim in claims
        if claim.get("verified")
        and claim.get("relation") == "BLOCKED"
        and claim.get("arguments")
    }
    eligible_satisfied = satisfied - contradictions
    eligible_blocked = blocked - contradictions
    denominator = max(1, len(required))
    features["language.premise_coverage"] = len(
        required & eligible_satisfied
    ) / denominator
    features["language.premise_block_fraction"] = len(
        required & eligible_blocked
    ) / denominator
    features["language.premise_conflict_fraction"] = len(
        required & contradictions
    ) / denominator
    predicate = instance.goals[0].atom.predicate.name
    if predicate == "NOT_READY":
        support_margin = features["language.premise_block_fraction"] - 1.0
    else:
        support_margin = (
            features["language.premise_coverage"]
            - 1.0
            - features["language.premise_block_fraction"]
            - features["language.premise_conflict_fraction"]
        )
    features["operator.support_margin"] = max(-1.0, min(1.0, support_margin))
    features["language.unparsed_count"] = _bounded_count(
        len(instance.metadata.get("unparsed_statements", ()))
    )


def _math_sensor_features(
    instance: DomainInstance,
    features: dict[str, float],
) -> None:
    metadata = instance.metadata
    input_kind = str(metadata.get("input_kind", "unknown"))
    features[f"math.kind.{input_kind}"] = 1.0
    features["math.operator_depth"] = min(
        float(metadata.get("operator_depth", 0)) / 16.0,
        1.0,
    )
    features["math.node_count"] = _bounded_count(
        int(metadata.get("node_count", 0))
    )
    if input_kind != "numeric_comparison":
        return
    left = Fraction(str(metadata["left_value"]))
    right = Fraction(str(metadata["right_value"]))
    operator = str(metadata["comparison_operator"])
    delta = left - right
    scale = max(abs(left), abs(right), Fraction(1))
    normalized_delta = float(delta / scale)
    normalized_gap = abs(normalized_delta)
    relation_margin = {
        "==": -normalized_gap,
        "!=": normalized_gap,
        ">": normalized_delta,
        ">=": normalized_delta,
        "<": -normalized_delta,
        "<=": -normalized_delta,
    }[operator]
    features[f"math.relation.{_relation_name(operator)}"] = 1.0
    features["math.relation_margin"] = max(-1.0, min(1.0, relation_margin))
    features["operator.support_margin"] = max(
        -1.0,
        min(1.0, relation_margin),
    )
    features["math.absolute_gap"] = min(1.0, normalized_gap)
    features["math.boundary"] = float(delta == 0)


def _vision_sensor_features(
    instance: DomainInstance,
    features: dict[str, float],
) -> None:
    metadata = instance.metadata
    width = max(1, int(metadata.get("raster_width", 1)))
    height = max(1, int(metadata.get("raster_height", 1)))
    objects = {
        str(item["id"]): item for item in metadata.get("detected_objects", ())
    }
    features["vision.width"] = min(width / 32.0, 1.0)
    features["vision.height"] = min(height / 32.0, 1.0)
    features["vision.object_count"] = _bounded_count(len(objects))
    goal = instance.goals[0].atom
    predicate = goal.predicate.name
    features[f"vision.goal.{predicate.lower()}"] = 1.0
    arguments = goal.arguments
    if predicate == "SQUARE" and arguments:
        item = objects.get(str(arguments[0]))
        if item is not None:
            box_width = max(1, int(item["bbox_width"]))
            box_height = max(1, int(item["bbox_height"]))
            area = int(item["area"])
            features["vision.fill_ratio"] = area / (box_width * box_height)
            features["vision.extent_similarity"] = min(
                box_width,
                box_height,
            ) / max(box_width, box_height)
            features["operator.support_margin"] = min(
                features["vision.fill_ratio"],
                features["vision.extent_similarity"],
            ) - 1.0
    elif predicate in {"LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW"} and len(arguments) == 2:
        first = objects.get(str(arguments[0]))
        second = objects.get(str(arguments[1]))
        if first is not None and second is not None:
            first_x, first_y = first["centroid"]
            second_x, second_y = second["centroid"]
            directional_margin = {
                "LEFT_OF": (second_x - first_x) / width,
                "RIGHT_OF": (first_x - second_x) / width,
                "ABOVE": (second_y - first_y) / height,
                "BELOW": (first_y - second_y) / height,
            }[predicate]
            features["vision.directional_margin"] = max(
                -1.0,
                min(1.0, float(directional_margin)),
            )
            features["operator.support_margin"] = features[
                "vision.directional_margin"
            ]
    elif predicate == "OBJECT_COUNT" and len(arguments) == 2:
        selector = str(arguments[0])
        expected = int(str(arguments[1]))
        counts = dict(metadata.get("raster_reasoning", {}).get("color_counts", ()))
        actual = len(objects) if selector == "all" else int(counts.get(selector, 0))
        scale = max(1, actual, expected)
        features["vision.count_delta"] = (actual - expected) / scale
        features["vision.count_gap"] = abs(actual - expected) / scale
        features["operator.support_margin"] = -features["vision.count_gap"]
    elif predicate == "LARGER_AREA" and len(arguments) == 2:
        first = objects.get(str(arguments[0]))
        second = objects.get(str(arguments[1]))
        if first is not None and second is not None:
            first_area = int(first["area"])
            second_area = int(second["area"])
            features["vision.area_margin"] = (
                (first_area - second_area) / max(1, first_area, second_area)
            )
            features["operator.support_margin"] = features["vision.area_margin"]


def _bounded_count(value: int) -> float:
    return min(max(value, 0), 64) / 64.0


def _relation_name(operator: str) -> str:
    return {
        "==": "equal",
        "!=": "not_equal",
        ">": "greater",
        ">=": "greater_equal",
        "<": "less",
        "<=": "less_equal",
    }[operator]


def _is_margin_feature(name: str) -> bool:
    return "margin" in name.lower()


__all__ = [
    "SemanticGroundingFeatureProfile",
    "profile_semantic_grounding_candidate",
    "profile_semantic_grounding_examples",
    "semantic_candidate_statement",
    "semantic_sensor_features",
]

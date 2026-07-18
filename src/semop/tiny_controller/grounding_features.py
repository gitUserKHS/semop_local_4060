from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata

from semop.kernel.grounding import GroundingCandidate, grounding_payload_digest
from semop.kernel.model import Symbol, Term, TermApplication

from .features import stable_bucket


GROUNDING_FEATURE_VERSION = 1
DEFAULT_SURFACE_BUCKETS = 512
MAX_SURFACE_TOKENS = 64
MAX_SENSOR_ABS_VALUE = 16.0


@dataclass(frozen=True)
class GroundingFeatureVector:
    """Canonical sparse input shared by all grounding policies."""

    values: tuple[tuple[str, float], ...]
    digest: str

    def __post_init__(self) -> None:
        names = [name for name, _value in self.values]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("grounding features must be unique and sorted")
        if any(not name for name in names):
            raise ValueError("grounding feature names cannot be empty")
        if any(not math.isfinite(value) for _name, value in self.values):
            raise ValueError("grounding feature values must be finite")

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)


def encode_grounding_candidate(
    candidate: GroundingCandidate,
    *,
    surface_buckets: int = DEFAULT_SURFACE_BUCKETS,
) -> GroundingFeatureVector:
    """Encode a typed candidate without retaining entity or symbol names.

    Domain adapters may add numeric ``sensor_features``. Their names describe a
    sensor contract, not a label, and their values are clipped before training.
    Surface text is anonymized and hashed into a bounded private namespace.
    """

    if surface_buckets <= 0:
        raise ValueError("surface bucket count must be positive")
    features: dict[str, float] = {
        "shared:bias": 1.0,
        f"shared:assertion:{candidate.assertion_status.value}": 1.0,
        f"shared:arity:{min(candidate.atom.predicate.arity, 8)}": 1.0,
        "shared:predicate_verified": float(candidate.atom.predicate.verified),
        "shared:candidate_confidence": candidate.confidence,
        "shared:evidence_count": min(len(candidate.evidence), 8) / 8.0,
        f"route:domain:{candidate.domain}": 1.0,
        f"route:source:{stable_bucket(candidate.source, 128):03d}": 1.0,
        f"route:producer:{stable_bucket(candidate.producer_id, 128):03d}": 1.0,
        (
            "shared:predicate_bucket:"
            f"{stable_bucket(candidate.atom.predicate.name.lower(), 256):03d}"
        ): 1.0,
    }
    for index, argument in enumerate(candidate.atom.arguments):
        features[f"shared:argument:{index}:type:{argument.type.name}"] = 1.0
        _encode_term_shape(argument, f"shared:argument:{index}", features, depth=0)

    for name, value in candidate.sensor_features:
        clipped = max(-MAX_SENSOR_ABS_VALUE, min(MAX_SENSOR_ABS_VALUE, value))
        features[f"sensor:{candidate.domain}:{name}"] = clipped
        features[f"sensor:shared:{name}"] = clipped

    anonymized = _anonymize_statement(candidate)
    tokens = _surface_tokens(anonymized)[:MAX_SURFACE_TOKENS]
    features[f"surface:{candidate.domain}:token_count"] = min(len(tokens), 32) / 32.0
    for token in tokens:
        bucket = stable_bucket(token, surface_buckets)
        key = f"surface:{candidate.domain}:unigram:{bucket:04d}"
        features[key] = features.get(key, 0.0) + 1.0 / max(1, len(tokens))
    for left, right in zip(tokens, tokens[1:]):
        bucket = stable_bucket(f"{left}|{right}", surface_buckets)
        key = f"surface:{candidate.domain}:bigram:{bucket:04d}"
        features[key] = features.get(key, 0.0) + 1.0 / max(1, len(tokens) - 1)

    if candidate.domain == "language":
        _encode_language_shape(anonymized, features)
    elif candidate.domain == "math":
        _encode_math_shape(anonymized, features)
    elif candidate.domain == "vision":
        _encode_vision_shape(candidate, features)
    else:
        features["surface:other_domain"] = 1.0

    values = tuple(sorted((name, value) for name, value in features.items() if value))
    payload = "\n".join(f"{name}={value:.12g}" for name, value in values)
    return GroundingFeatureVector(values, grounding_payload_digest(payload))


def grounding_feature_support(
    values: tuple[tuple[str, float], ...],
    support: dict[str, int],
) -> int:
    """Return conservative support for selective prediction.

    If an adapter supplies a sensor contract, every active shared sensor feature
    must have appeared in verified training data. This prevents familiar type or
    bias features from hiding an out-of-distribution sensor channel.
    """

    sensor_names = [
        name
        for name, value in values
        if value and name.startswith("sensor:shared:")
    ]
    if sensor_names:
        return min(support.get(name, 0) for name in sensor_names)
    return max(
        (
            support.get(name, 0)
            for name, value in values
            if value and name != "shared:bias"
        ),
        default=0,
    )


def _encode_term_shape(
    term: Term,
    prefix: str,
    features: dict[str, float],
    *,
    depth: int,
) -> None:
    if isinstance(term, Symbol):
        features[f"{prefix}:symbol"] = 1.0
        return
    if isinstance(term, TermApplication):
        features[f"{prefix}:application"] = 1.0
        features[f"{prefix}:function_arity:{min(len(term.arguments), 8)}"] = 1.0
        features[
            f"{prefix}:function_bucket:{stable_bucket(term.function.name, 128):03d}"
        ] = 1.0
        if depth < 3:
            for index, child in enumerate(term.arguments):
                _encode_term_shape(
                    child,
                    f"{prefix}:child:{index}",
                    features,
                    depth=depth + 1,
                )
        return
    features[f"{prefix}:other_term"] = 1.0


def _anonymize_statement(candidate: GroundingCandidate) -> str:
    normalized = unicodedata.normalize("NFKC", candidate.statement).lower()
    names = sorted(
        {symbol.name for argument in candidate.atom.arguments for symbol in _symbols(argument)},
        key=lambda value: (-len(value), value),
    )
    for name in names:
        normalized = re.sub(re.escape(name.lower()), " <entity> ", normalized)
    normalized = re.sub(r"(?<![\w.])-?\d+(?:\.\d+)?(?:/\d+)?", " <number> ", normalized)
    return " ".join(normalized.split())


def _symbols(term: Term) -> tuple[Symbol, ...]:
    if isinstance(term, Symbol):
        return (term,)
    if isinstance(term, TermApplication):
        return tuple(symbol for child in term.arguments for symbol in _symbols(child))
    return ()


def _surface_tokens(value: str) -> list[str]:
    return re.findall(r"[^\W_]+|[+*/=<>:-]", value, flags=re.UNICODE)


def _encode_language_shape(value: str, features: dict[str, float]) -> None:
    features["surface:language:has_colon"] = float(":" in value)
    features["surface:language:has_negation"] = float(
        any(token in value for token in (" not ", " no ", "아니", "없"))
    )
    features["surface:language:controlled_prefix"] = float(
        bool(
            re.match(
                r"^(goal|requires|satisfied|blocked|fact|rule|class|instance)\s*:",
                value,
            )
        )
    )


def _encode_math_shape(value: str, features: dict[str, float]) -> None:
    for name, token in (
        ("plus", "+"),
        ("minus", "-"),
        ("multiply", "*"),
        ("divide", "/"),
        ("equals", "="),
        ("compare", "<"),
        ("compare", ">"),
    ):
        if token in value:
            features[f"surface:math:has_{name}"] = 1.0
    features["surface:math:parenthesis_balance"] = float(
        value.count("(") == value.count(")")
    )


def _encode_vision_shape(
    candidate: GroundingCandidate,
    features: dict[str, float],
) -> None:
    features["surface:vision:is_relation"] = float(candidate.atom.predicate.arity == 2)
    features["surface:vision:is_property"] = float(candidate.atom.predicate.arity == 1)
    features["surface:vision:has_image_evidence"] = float(
        any(item.startswith("image:") for item in candidate.evidence)
    )


__all__ = [
    "DEFAULT_SURFACE_BUCKETS",
    "GROUNDING_FEATURE_VERSION",
    "GroundingFeatureVector",
    "encode_grounding_candidate",
    "grounding_feature_support",
]

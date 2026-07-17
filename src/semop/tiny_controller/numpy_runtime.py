from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import json
from math import sqrt
from pathlib import Path
from typing import Any, Sequence

from semop.kernel import Goal, GroundAction, PolicyDecision, WorldState

from .features import (
    ACTION_STRUCTURAL_FEATURE_COUNT,
    canonicalize_problem,
    stable_bucket,
)


@dataclass(frozen=True)
class TinyControllerConfig:
    d_model: int = 192
    token_buckets: int = 24_576
    relation_buckets: int = 2_048
    operator_buckets: int = 2_048
    message_blocks: int = 2
    recursion_steps: int = 4
    action_limit_score_margin: float = 0.1
    max_parameters: int = 15_000_000

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.message_blocks <= 0 or self.recursion_steps <= 0:
            raise ValueError("controller dimensions and iteration counts must be positive")
        if min(self.token_buckets, self.relation_buckets, self.operator_buckets) <= 0:
            raise ValueError("hash bucket counts must be positive")
        if self.action_limit_score_margin <= 0:
            raise ValueError("action-limit score margin must be positive")
        if self.estimated_parameter_count() > self.max_parameters:
            raise ValueError(
                f"controller would exceed {self.max_parameters:,} parameters: "
                f"{self.estimated_parameter_count():,}"
            )

    def estimated_parameter_count(self) -> int:
        d = self.d_model
        embeddings = (
            self.token_buckets + self.relation_buckets + self.operator_buckets
        ) * d
        blocks = self.message_blocks * (3 * d * d + d)
        heads = 3 * d * d + 2 * d + 2 + ACTION_STRUCTURAL_FEATURE_COUNT
        return embeddings + blocks + heads


class NumpyTinyController:
    """Relation-aware recurrent policy runtime with no PyTorch dependency."""

    FORMAT_VERSION = 5

    def __init__(self, config: TinyControllerConfig, weights: dict[str, Any]) -> None:
        import numpy as np

        self.config = config
        self.weights = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in weights.items()
        }
        self._validate_weights()

    @classmethod
    def random(
        cls, config: TinyControllerConfig | None = None, *, seed: int = 0
    ) -> "NumpyTinyController":
        import numpy as np

        resolved = config or TinyControllerConfig()
        rng = np.random.default_rng(seed)
        d = resolved.d_model

        def normal(shape, scale: float | None = None):
            return rng.normal(
                0.0, scale or (1.0 / sqrt(d)), size=shape
            ).astype(np.float32)

        weights = {
            "token_embedding": normal((resolved.token_buckets, d), 0.02),
            "relation_embedding": normal((resolved.relation_buckets, d), 0.02),
            "operator_embedding": normal((resolved.operator_buckets, d), 0.02),
            "block_self": normal((resolved.message_blocks, d, d)),
            "block_message": normal((resolved.message_blocks, d, d)),
            "block_relation": normal((resolved.message_blocks, d, d)),
            "block_bias": np.zeros((resolved.message_blocks, d), dtype=np.float32),
            "operator_query": normal((d, d)),
            "argument_query": normal((d, d)),
            "action_state": normal((d, d)),
            "action_structure_weight": np.zeros(
                (ACTION_STRUCTURAL_FEATURE_COUNT,), dtype=np.float32
            ),
            "halt_weight": normal((d,)),
            "halt_bias": np.zeros((1,), dtype=np.float32),
            "value_weight": normal((d,)),
            "value_bias": np.zeros((1,), dtype=np.float32),
        }
        return cls(resolved, weights)

    @classmethod
    def load(cls, path: str | Path) -> "NumpyTinyController":
        return cls.from_artifact(Path(path).read_bytes())

    @classmethod
    def from_artifact(cls, artifact: bytes) -> "NumpyTinyController":
        import numpy as np

        try:
            with np.load(BytesIO(artifact), allow_pickle=False) as payload:
                version = int(payload["format_version"].item())
                if version not in {2, 3, 4, cls.FORMAT_VERSION}:
                    raise ValueError(
                        f"unsupported tiny-controller format: {version}"
                    )
                config = TinyControllerConfig(
                    **json.loads(str(payload["config_json"].item()))
                )
                weights = {
                    name: payload[name]
                    for name in payload.files
                    if name not in {"format_version", "config_json"}
                }
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise ValueError("invalid tiny-controller artifact") from exc
        if version == 2 and "action_structure_weight" not in weights:
            weights["action_structure_weight"] = np.zeros(
                (ACTION_STRUCTURAL_FEATURE_COUNT,), dtype=np.float32
            )
        elif version == 3:
            weights["action_structure_weight"] = (
                np.asarray(weights["action_structure_weight"], dtype=np.float32)
                / sqrt(config.d_model)
            )
        return cls(config, weights)

    def save(self, path: str | Path, *, compressed: bool = True) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.to_artifact(compressed=compressed))
        return destination

    def to_artifact(self, *, compressed: bool = True) -> bytes:
        import numpy as np

        payload = {
            "format_version": np.asarray(self.FORMAT_VERSION, dtype=np.int32),
            "config_json": np.asarray(json.dumps(asdict(self.config), sort_keys=True)),
            **self.weights,
        }
        writer = np.savez_compressed if compressed else np.savez
        buffer = BytesIO()
        writer(buffer, **payload)
        return buffer.getvalue()

    @property
    def parameter_count(self) -> int:
        return sum(int(value.size) for value in self.weights.values())

    @property
    def artifact_bytes_uncompressed(self) -> int:
        return sum(int(value.nbytes) for value in self.weights.values())

    def score_actions(
        self,
        state: WorldState,
        goals: Sequence[Goal],
        actions: Sequence[GroundAction],
    ) -> PolicyDecision:
        import numpy as np

        graph = canonicalize_problem(state, goals, actions)
        node_vectors = self._initial_node_vectors(graph.node_tokens)
        relation_vectors = self._relation_vectors(graph.relations, graph.goals)
        d = self.config.d_model

        for _ in range(self.config.recursion_steps):
            for block in range(self.config.message_blocks):
                aggregate = np.zeros_like(node_vectors)
                counts = np.zeros((len(node_vectors), 1), dtype=np.float32)
                for relation_index, relation in enumerate(graph.relations + graph.goals):
                    if not relation.arguments:
                        continue
                    context = node_vectors[list(relation.arguments)].mean(axis=0)
                    message = (
                        context @ self.weights["block_message"][block]
                        + relation_vectors[relation_index]
                        @ self.weights["block_relation"][block]
                    )
                    for node_index in relation.arguments:
                        aggregate[node_index] += message
                        counts[node_index] += 1.0
                aggregate /= np.maximum(counts, 1.0)
                update = (
                    node_vectors @ self.weights["block_self"][block]
                    + aggregate
                    + self.weights["block_bias"][block]
                )
                node_vectors = np.tanh(update).astype(np.float32, copy=False)

        relation_pool = (
            relation_vectors.mean(axis=0)
            if len(relation_vectors)
            else np.zeros((d,), dtype=np.float32)
        )
        state_vector = node_vectors.mean(axis=0) + relation_pool
        norm = float(np.linalg.norm(state_vector))
        if norm > 0:
            state_vector = state_vector / norm
        operator_query = state_vector @ self.weights["operator_query"]
        argument_query = state_vector @ self.weights["argument_query"]
        state_action = state_vector @ self.weights["action_state"]
        scores: list[float] = []
        scale = sqrt(d)
        for action in graph.actions:
            operator_indices = [
                stable_bucket(feature, self.config.operator_buckets)
                for feature in action.operator_features
            ]
            operator_vector = self.weights["operator_embedding"][
                operator_indices
            ].mean(axis=0)
            operator_score = float(operator_vector @ operator_query) / scale
            if action.argument_nodes:
                argument_vectors = node_vectors[list(action.argument_nodes)]
                type_bias = np.stack(
                    [
                        self.weights["token_embedding"][
                            stable_bucket(
                                f"argument_type:{type_name}", self.config.token_buckets
                            )
                        ]
                        for type_name in action.argument_types
                    ]
                )
                pointer_score = float(
                    ((argument_vectors + type_bias) @ argument_query).mean()
                ) / scale
                action_vector = argument_vectors.mean(axis=0) + operator_vector
            else:
                pointer_score = 0.0
                action_vector = operator_vector
            compatibility = float(action_vector @ state_action) / scale
            structural_score = float(
                np.asarray(action.structural_values, dtype=np.float32)
                @ self.weights["action_structure_weight"]
            ) * sqrt(d)
            latent_score = float(np.tanh(operator_score + compatibility))
            bounded_pointer = float(np.tanh(pointer_score))
            scores.append(latent_score + bounded_pointer + structural_score)
        halt_logit = float(state_vector @ self.weights["halt_weight"])
        halt_logit += float(self.weights["halt_bias"][0])
        value_logit = float(state_vector @ self.weights["value_weight"])
        value_logit += float(self.weights["value_bias"][0])
        halt_probability = 1.0 / (1.0 + float(np.exp(-np.clip(halt_logit, -30, 30))))
        ordered_scores = sorted(scores, reverse=True)
        clear_preference = (
            len(ordered_scores) <= 1
            or ordered_scores[0] - ordered_scores[1]
            >= self.config.action_limit_score_margin
        )
        return PolicyDecision(
            action_scores=tuple(scores),
            halt_probability=halt_probability,
            state_value=float(np.tanh(value_logit)),
            action_limit=1 if clear_preference else None,
        )

    def _initial_node_vectors(self, node_tokens: Sequence[Sequence[str]]):
        import numpy as np

        vectors = np.zeros(
            (len(node_tokens), self.config.d_model), dtype=np.float32
        )
        for node_index, tokens in enumerate(node_tokens):
            indices = [
                stable_bucket(token, self.config.token_buckets) for token in tokens
            ]
            vectors[node_index] = self.weights["token_embedding"][indices].mean(axis=0)
        return vectors

    def _relation_vectors(self, relations, goals):
        import numpy as np

        all_relations = relations + goals
        if not all_relations:
            return np.zeros((0, self.config.d_model), dtype=np.float32)
        indices = [
            stable_bucket(
                f"relation:{relation.name}:role:{relation.role}",
                self.config.relation_buckets,
            )
            for relation in all_relations
        ]
        return self.weights["relation_embedding"][indices]

    def _validate_weights(self) -> None:
        d = self.config.d_model
        expected = {
            "token_embedding": (self.config.token_buckets, d),
            "relation_embedding": (self.config.relation_buckets, d),
            "operator_embedding": (self.config.operator_buckets, d),
            "block_self": (self.config.message_blocks, d, d),
            "block_message": (self.config.message_blocks, d, d),
            "block_relation": (self.config.message_blocks, d, d),
            "block_bias": (self.config.message_blocks, d),
            "operator_query": (d, d),
            "argument_query": (d, d),
            "action_state": (d, d),
            "action_structure_weight": (ACTION_STRUCTURAL_FEATURE_COUNT,),
            "halt_weight": (d,),
            "halt_bias": (1,),
            "value_weight": (d,),
            "value_bias": (1,),
        }
        missing = sorted(set(expected) - set(self.weights))
        extra = sorted(set(self.weights) - set(expected))
        if missing or extra:
            raise ValueError(f"invalid weight set: missing={missing}, extra={extra}")
        for name, shape in expected.items():
            if self.weights[name].shape != shape:
                raise ValueError(
                    f"weight {name} has shape {self.weights[name].shape}, expected {shape}"
                )
        if self.parameter_count > self.config.max_parameters:
            raise ValueError(
                f"loaded model exceeds parameter cap: {self.parameter_count:,}"
            )

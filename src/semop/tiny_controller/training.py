from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from semop.kernel import Goal, GroundAction, WorldState

from .features import (
    ACTION_STRUCTURAL_FEATURE_COUNT,
    canonicalize_problem,
    stable_bucket,
)
from .numpy_runtime import NumpyTinyController, TinyControllerConfig


@dataclass(frozen=True)
class ControllerLossWeights:
    action: float = 1.0
    argument: float = 1.0
    halt: float = 0.5
    value: float = 0.5
    recursive_consistency: float = 0.2


@dataclass(frozen=True)
class ControllerTrainingExample:
    node_token_ids: Tensor
    node_token_mask: Tensor
    relation_ids: Tensor
    relation_arguments: Tensor
    action_operator_ids: Tensor
    action_operator_mask: Tensor
    action_argument_nodes: Tensor
    action_argument_type_ids: Tensor
    action_structural_values: Tensor
    target_action: Tensor
    target_halt: Tensor
    target_value: Tensor

    def to(self, device: str | torch.device) -> "ControllerTrainingExample":
        return ControllerTrainingExample(
            **{
                name: value.to(device)
                for name, value in self.__dict__.items()
            }
        )


@dataclass(frozen=True)
class ControllerLoss:
    total: Tensor
    action: Tensor
    argument: Tensor
    halt: Tensor
    value: Tensor
    recursive_consistency: Tensor


class TorchTinyController(nn.Module):
    """Training mirror of the NumPy relation-aware recurrent controller."""

    def __init__(self, config: TinyControllerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyControllerConfig()
        d = self.config.d_model
        self.token_embedding = nn.Parameter(
            torch.empty(self.config.token_buckets, d)
        )
        self.relation_embedding = nn.Parameter(
            torch.empty(self.config.relation_buckets, d)
        )
        self.operator_embedding = nn.Parameter(
            torch.empty(self.config.operator_buckets, d)
        )
        self.block_self = nn.Parameter(
            torch.empty(self.config.message_blocks, d, d)
        )
        self.block_message = nn.Parameter(
            torch.empty(self.config.message_blocks, d, d)
        )
        self.block_relation = nn.Parameter(
            torch.empty(self.config.message_blocks, d, d)
        )
        self.block_bias = nn.Parameter(
            torch.zeros(self.config.message_blocks, d)
        )
        self.operator_query = nn.Parameter(torch.empty(d, d))
        self.argument_query = nn.Parameter(torch.empty(d, d))
        self.action_state = nn.Parameter(torch.empty(d, d))
        self.action_structure_weight = nn.Parameter(
            torch.zeros(ACTION_STRUCTURAL_FEATURE_COUNT)
        )
        self.halt_weight = nn.Parameter(torch.empty(d))
        self.halt_bias = nn.Parameter(torch.zeros(1))
        self.value_weight = nn.Parameter(torch.empty(d))
        self.value_bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d = self.config.d_model
        nn.init.normal_(self.token_embedding, std=0.02)
        nn.init.normal_(self.relation_embedding, std=0.02)
        nn.init.normal_(self.operator_embedding, std=0.02)
        for parameter in (
            self.block_self,
            self.block_message,
            self.block_relation,
            self.operator_query,
            self.argument_query,
            self.action_state,
        ):
            nn.init.normal_(parameter, std=1.0 / sqrt(d))
        nn.init.normal_(self.halt_weight, std=1.0 / sqrt(d))
        nn.init.normal_(self.value_weight, std=1.0 / sqrt(d))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, example: ControllerTrainingExample) -> dict[str, Tensor]:
        token_vectors = self.token_embedding[example.node_token_ids]
        token_mask = example.node_token_mask.unsqueeze(-1)
        node_vectors = (token_vectors * token_mask).sum(dim=1) / token_mask.sum(
            dim=1
        ).clamp_min(1.0)
        relation_vectors = self.relation_embedding[example.relation_ids]
        recursive_states: list[Tensor] = []
        for _ in range(self.config.recursion_steps):
            for block in range(self.config.message_blocks):
                aggregate = torch.zeros_like(node_vectors)
                counts = torch.zeros(
                    node_vectors.shape[0],
                    1,
                    device=node_vectors.device,
                    dtype=node_vectors.dtype,
                )
                for relation_index in range(example.relation_arguments.shape[0]):
                    arguments = example.relation_arguments[relation_index]
                    arguments = arguments[arguments >= 0]
                    if arguments.numel() == 0:
                        continue
                    context = node_vectors[arguments].mean(dim=0)
                    message = (
                        context @ self.block_message[block]
                        + relation_vectors[relation_index]
                        @ self.block_relation[block]
                    )
                    aggregate = aggregate.index_add(
                        0, arguments, message.unsqueeze(0).expand(arguments.numel(), -1)
                    )
                    counts = counts.index_add(
                        0,
                        arguments,
                        torch.ones(
                            arguments.numel(),
                            1,
                            device=node_vectors.device,
                            dtype=node_vectors.dtype,
                        ),
                    )
                aggregate = aggregate / counts.clamp_min(1.0)
                node_vectors = torch.tanh(
                    node_vectors @ self.block_self[block]
                    + aggregate
                    + self.block_bias[block]
                )
            relation_pool = (
                relation_vectors.mean(dim=0)
                if relation_vectors.numel()
                else torch.zeros_like(node_vectors[0])
            )
            recursive_states.append(
                F.normalize(node_vectors.mean(dim=0) + relation_pool, dim=0)
            )
        state_vector = recursive_states[-1]
        operator_features = self.operator_embedding[example.action_operator_ids]
        operator_mask = example.action_operator_mask.unsqueeze(-1)
        operator_vectors = (operator_features * operator_mask).sum(
            dim=1
        ) / operator_mask.sum(dim=1).clamp_min(1.0)
        operator_scores = (
            operator_vectors @ (state_vector @ self.operator_query)
        ) / sqrt(self.config.d_model)

        pointer_scores: list[Tensor] = []
        action_vectors: list[Tensor] = []
        argument_query = state_vector @ self.argument_query
        for action_index in range(example.action_argument_nodes.shape[0]):
            nodes = example.action_argument_nodes[action_index]
            types = example.action_argument_type_ids[action_index]
            valid = nodes >= 0
            if valid.any():
                arguments = node_vectors[nodes[valid]]
                type_vectors = self.token_embedding[types[valid]]
                pointer_scores.append(
                    ((arguments + type_vectors) @ argument_query).mean()
                    / sqrt(self.config.d_model)
                )
                action_vectors.append(
                    arguments.mean(dim=0) + operator_vectors[action_index]
                )
            else:
                pointer_scores.append(torch.zeros((), device=node_vectors.device))
                action_vectors.append(operator_vectors[action_index])
        if pointer_scores:
            argument_scores = torch.stack(pointer_scores)
            compatibility_scores = (
                torch.stack(action_vectors) @ (state_vector @ self.action_state)
            ) / sqrt(self.config.d_model)
        else:
            argument_scores = torch.empty(0, device=node_vectors.device)
            compatibility_scores = torch.empty(0, device=node_vectors.device)
        action_logits = torch.tanh(operator_scores + compatibility_scores)
        if argument_scores.numel():
            action_logits = action_logits + torch.tanh(argument_scores)
        if example.action_structural_values.numel():
            action_logits = (
                action_logits
                + example.action_structural_values @ self.action_structure_weight
                * sqrt(self.config.d_model)
            )
        halt_logit = state_vector @ self.halt_weight + self.halt_bias[0]
        value = torch.tanh(state_vector @ self.value_weight + self.value_bias[0])
        return {
            "action_logits": action_logits,
            "argument_logits": argument_scores,
            "halt_logit": halt_logit,
            "value": value,
            "recursive_states": torch.stack(recursive_states),
        }

    @classmethod
    def from_numpy(
        cls, runtime: NumpyTinyController
    ) -> "TorchTinyController":
        model = cls(runtime.config)
        with torch.no_grad():
            for name, value in runtime.weights.items():
                getattr(model, name).copy_(torch.from_numpy(value))
        return model

    def export_numpy(self, path: str | Path) -> Path:
        weights = {
            name: parameter.detach().cpu().numpy()
            for name, parameter in self.named_parameters()
        }
        return NumpyTinyController(self.config, weights).save(path)


def encode_training_example(
    state: WorldState,
    goals: Sequence[Goal],
    actions: Sequence[GroundAction],
    *,
    target_action: int,
    target_halt: bool,
    target_value: float,
    config: TinyControllerConfig | None = None,
) -> ControllerTrainingExample:
    resolved = config or TinyControllerConfig()
    if actions and not 0 <= target_action < len(actions):
        raise ValueError("target_action is outside the candidate action range")
    if not actions and target_action != -1:
        raise ValueError("terminal examples must use target_action=-1")
    graph = canonicalize_problem(state, goals, actions)
    max_tokens = max(len(tokens) for tokens in graph.node_tokens)
    node_ids = torch.zeros((len(graph.node_tokens), max_tokens), dtype=torch.long)
    node_mask = torch.zeros((len(graph.node_tokens), max_tokens), dtype=torch.float32)
    for node_index, tokens in enumerate(graph.node_tokens):
        for token_index, token in enumerate(tokens):
            node_ids[node_index, token_index] = stable_bucket(
                token, resolved.token_buckets
            )
            node_mask[node_index, token_index] = 1.0

    relations = graph.relations + graph.goals
    max_arity = max((len(relation.arguments) for relation in relations), default=1)
    relation_ids = torch.tensor(
        [
            stable_bucket(
                f"relation:{relation.name}:role:{relation.role}",
                resolved.relation_buckets,
            )
            for relation in relations
        ],
        dtype=torch.long,
    )
    relation_arguments = torch.full(
        (len(relations), max_arity), -1, dtype=torch.long
    )
    for relation_index, relation in enumerate(relations):
        relation_arguments[relation_index, : len(relation.arguments)] = torch.tensor(
            relation.arguments, dtype=torch.long
        )

    max_parameters = max((len(action.argument_nodes) for action in graph.actions), default=1)
    action_nodes = torch.full(
        (len(graph.actions), max_parameters), -1, dtype=torch.long
    )
    action_types = torch.zeros(
        (len(graph.actions), max_parameters), dtype=torch.long
    )
    for action_index, action in enumerate(graph.actions):
        action_nodes[action_index, : len(action.argument_nodes)] = torch.tensor(
            action.argument_nodes, dtype=torch.long
        )
        action_types[action_index, : len(action.argument_types)] = torch.tensor(
            [
                stable_bucket(
                    f"argument_type:{type_name}", resolved.token_buckets
                )
                for type_name in action.argument_types
            ],
            dtype=torch.long,
        )
    action_structural_values = torch.tensor(
        [action.structural_values for action in graph.actions],
        dtype=torch.float32,
    ).reshape(-1, ACTION_STRUCTURAL_FEATURE_COUNT)
    max_operator_features = max(
        (len(action.operator_features) for action in graph.actions), default=1
    )
    operator_ids = torch.zeros(
        (len(graph.actions), max_operator_features), dtype=torch.long
    )
    operator_mask = torch.zeros(
        (len(graph.actions), max_operator_features), dtype=torch.float32
    )
    for action_index, action in enumerate(graph.actions):
        for feature_index, feature in enumerate(action.operator_features):
            operator_ids[action_index, feature_index] = stable_bucket(
                feature, resolved.operator_buckets
            )
            operator_mask[action_index, feature_index] = 1.0
    return ControllerTrainingExample(
        node_token_ids=node_ids,
        node_token_mask=node_mask,
        relation_ids=relation_ids,
        relation_arguments=relation_arguments,
        action_operator_ids=operator_ids,
        action_operator_mask=operator_mask,
        action_argument_nodes=action_nodes,
        action_argument_type_ids=action_types,
        action_structural_values=action_structural_values,
        target_action=torch.tensor(target_action, dtype=torch.long),
        target_halt=torch.tensor(float(target_halt), dtype=torch.float32),
        target_value=torch.tensor(float(target_value), dtype=torch.float32),
    )


def controller_loss(
    output: dict[str, Tensor],
    example: ControllerTrainingExample,
    weights: ControllerLossWeights | None = None,
) -> ControllerLoss:
    resolved = weights or ControllerLossWeights()
    if int(example.target_action.item()) >= 0:
        action = F.cross_entropy(
            output["action_logits"].unsqueeze(0), example.target_action.unsqueeze(0)
        )
        argument = F.cross_entropy(
            output["argument_logits"].unsqueeze(0), example.target_action.unsqueeze(0)
        )
    else:
        action = output["halt_logit"] * 0.0
        argument = output["halt_logit"] * 0.0
    halt = F.binary_cross_entropy_with_logits(
        output["halt_logit"], example.target_halt
    )
    value = F.mse_loss(output["value"], example.target_value)
    recursive_states = output["recursive_states"]
    if recursive_states.shape[0] > 1:
        target_state = recursive_states[-1].detach().expand_as(recursive_states[:-1])
        consistency = F.mse_loss(recursive_states[:-1], target_state)
    else:
        consistency = torch.zeros((), device=recursive_states.device)
    total = (
        resolved.action * action
        + resolved.argument * argument
        + resolved.halt * halt
        + resolved.value * value
        + resolved.recursive_consistency * consistency
    )
    return ControllerLoss(total, action, argument, halt, value, consistency)


def train_controller(
    model: TorchTinyController,
    examples: Iterable[ControllerTrainingExample],
    *,
    epochs: int = 1,
    learning_rate: float = 3e-4,
    max_grad_norm: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[float, ...]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    cached = tuple(examples)
    if not cached:
        raise ValueError("training requires at least one example")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        for raw_example in cached:
            example = raw_example.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = controller_loss(model(example), example)
            loss.total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += float(loss.total.detach().cpu())
        history.append(total_loss / len(cached))
    return tuple(history)

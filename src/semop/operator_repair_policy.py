from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .operator_runtime import compile_and_execute
from .structures import StructuredMeaningGraph


REPAIR_ACTIONS = [
    'add_goal_preservation_decomposition',
    'bind_requires_edges',
    'attach_document_context_nodes',
    'attach_visual_scene',
    'rebind_functor_object_map',
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9a-z_]+", text.lower())


@dataclass
class OperatorRepairPolicyModel:
    action_feature_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    bias: dict[str, float] = field(default_factory=dict)
    min_score: float = 0.25
    trained_on_examples: int = 0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: str | Path | None) -> OperatorRepairPolicyModel | None:
        if not path:
            return None
        payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if isinstance(payload, dict) and 'weights' in payload:
            payload = payload['weights']
        return cls(**payload)


@dataclass
class OperatorRepairPolicyTrainingSummary:
    output_path: str
    trained_on_examples: int
    model: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class OperatorRepairPolicyTrainer:
    def train_from_graphs(
        self,
        graphs: Sequence[StructuredMeaningGraph],
        output_path: str | Path,
    ) -> OperatorRepairPolicyTrainingSummary:
        examples: list[tuple[list[str], str]] = []
        for graph in graphs:
            examples.extend(self._synthetic_examples(graph))
        action_feature_weights: dict[str, dict[str, float]] = {action: {} for action in REPAIR_ACTIONS}
        bias: dict[str, float] = {action: 0.0 for action in REPAIR_ACTIONS}
        for features, action in examples:
            bias[action] = round(bias.get(action, 0.0) + 1.0, 4)
            for feature in features:
                bucket = action_feature_weights.setdefault(action, {})
                bucket[feature] = round(bucket.get(feature, 0.0) + 1.0, 4)
        model = OperatorRepairPolicyModel(
            action_feature_weights=action_feature_weights,
            bias=bias,
            trained_on_examples=len(examples),
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'weights': model.model_dump()}, ensure_ascii=False, indent=2), encoding='utf-8')
        return OperatorRepairPolicyTrainingSummary(output_path=str(output), trained_on_examples=len(examples), model=model.model_dump())

    def _synthetic_examples(self, graph: StructuredMeaningGraph) -> list[tuple[list[str], str]]:
        examples: list[tuple[list[str], str]] = []
        if graph.hidden_goals and graph.required_premises:
            broken = self._clone_graph(graph)
            broken.operator_decompositions = []
            compile_and_execute(broken)
            examples.append((self._features(broken), 'add_goal_preservation_decomposition'))
        if graph.required_premises or graph.satisfied_premises or graph.missing_premises:
            broken = self._clone_graph(graph)
            broken.edges = [edge for edge in broken.edges if edge.relation != 'REQUIRES']
            compile_and_execute(broken)
            examples.append((self._features(broken), 'bind_requires_edges'))
        if graph.source_context.strip():
            broken = self._clone_graph(graph)
            broken.nodes = [node for node in broken.nodes if node.id not in {'source_document'} and node.kind != 'evidence']
            broken.edges = [edge for edge in broken.edges if edge.relation not in {'USES_CONTEXT', 'HAS_EVIDENCE', 'GROUNDED_BY'}]
            compile_and_execute(broken)
            examples.append((self._features(broken), 'attach_document_context_nodes'))
        if any(any(tag.startswith('multimodal:vision') for tag in node.provenance) for node in graph.nodes):
            broken = self._clone_graph(graph)
            broken.nodes = [node for node in broken.nodes if node.id != 'visual_scene']
            broken.edges = [edge for edge in broken.edges if not (edge.source == 'question' and edge.relation == 'CONDITIONS_ON' and edge.target == 'visual_scene')]
            compile_and_execute(broken)
            examples.append((self._features(broken), 'attach_visual_scene'))
        if graph.functor_hypotheses:
            broken = self._clone_graph(graph)
            for functor in broken.functor_hypotheses:
                functor.object_map = {}
            compile_and_execute(broken)
            examples.append((self._features(broken), 'rebind_functor_object_map'))
        return examples

    @staticmethod
    def _clone_graph(graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        return StructuredMeaningGraph.from_dict(graph.model_dump())

    @staticmethod
    def _features(graph: StructuredMeaningGraph) -> list[str]:
        report = graph.operator_execution
        findings = ' '.join(report.compiler_findings if report is not None else [])
        features = set(_tokenize(findings))
        if graph.hidden_goals:
            features.add('has_hidden_goal')
        if graph.required_premises or graph.satisfied_premises or graph.missing_premises:
            features.add('has_premise_state')
        if graph.source_context.strip():
            features.add('has_document_context')
        if any(any(tag.startswith('multimodal:vision') for tag in node.provenance) for node in graph.nodes):
            features.add('has_visual_signal')
        if graph.functor_hypotheses:
            features.add('has_functor')
        return sorted(features)


class OperatorRepairPolicyScorer:
    def __init__(self, model: OperatorRepairPolicyModel | None = None, model_path: str | Path | None = None) -> None:
        self.model = model or OperatorRepairPolicyModel.from_path(model_path) or OperatorRepairPolicyModel()

    def rank_actions(self, graph: StructuredMeaningGraph, findings: list[str]) -> list[str]:
        features = set(_tokenize(' '.join(findings)))
        if graph.hidden_goals:
            features.add('has_hidden_goal')
        if graph.required_premises or graph.satisfied_premises or graph.missing_premises:
            features.add('has_premise_state')
        if graph.source_context.strip():
            features.add('has_document_context')
        if any(any(tag.startswith('multimodal:vision') for tag in node.provenance) for node in graph.nodes):
            features.add('has_visual_signal')
        if graph.functor_hypotheses:
            features.add('has_functor')
        scored: list[tuple[str, float]] = []
        for action in REPAIR_ACTIONS:
            score = float(self.model.bias.get(action, 0.0))
            for feature in features:
                score += float(self.model.action_feature_weights.get(action, {}).get(feature, 0.0))
            scored.append((action, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        filtered = [action for action, score in scored if score >= self.model.min_score]
        return filtered or [action for action, _ in scored]


def train_repair_policy_from_graphs(graphs: Sequence[StructuredMeaningGraph], output_path: str | Path) -> OperatorRepairPolicyTrainingSummary:
    return OperatorRepairPolicyTrainer().train_from_graphs(graphs, output_path)

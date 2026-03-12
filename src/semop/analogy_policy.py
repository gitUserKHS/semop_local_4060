from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, List

from .commonsense_kb import concept_family, match_concepts
from .memory_retrieval import QueryEmbeddingIndex
from .structures import StructuredMeaningGraph

if TYPE_CHECKING:
    from .corpus_store import CorpusMemoryStore


@dataclass
class AnalogyPolicyModel:
    bias_weight: float = 0.0
    lexical_weight: float = 0.08
    goal_weight: float = 0.26
    requirement_weight: float = 0.24
    missing_weight: float = 0.18
    relation_weight: float = 0.1
    operator_weight: float = 0.08
    node_family_weight: float = 0.06
    script_weight: float = 0.05
    support_floor: float = 0.02
    plan_guard_weight: float = 1.0
    verifier_weight: float = 1.0
    operator_priority_weight: float = 0.7
    trained_on_pairs: int = 0
    training_examples: int = 0
    training_loss: float = 0.0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: str | Path | None) -> AnalogyPolicyModel | None:
        if not path:
            return None
        payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if isinstance(payload, dict) and 'weights' in payload:
            payload = {**payload.get('weights', {}), 'trained_on_pairs': payload.get('trained_on_pairs', 0)}
        return cls(**payload)


@dataclass
class AnalogyPolicyTrainingSummary:
    output_path: str
    trained_on_pairs: int
    model: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class AnalogyPolicyTrainer:
    def train_from_memory(
        self,
        memory_store: CorpusMemoryStore,
        output_path: str | Path,
        source: str | None = None,
        split: str = 'train',
        epochs: int = 120,
        learning_rate: float = 0.18,
    ) -> AnalogyPolicyTrainingSummary:
        graphs = memory_store.fetch_graphs(split=split, source=source)
        training_examples = self._build_training_examples(graphs)
        weights = {
            'bias_weight': 0.0,
            'lexical_weight': 0.05,
            'goal_weight': 0.1,
            'requirement_weight': 0.1,
            'missing_weight': 0.1,
            'relation_weight': 0.1,
            'operator_weight': 0.1,
            'node_family_weight': 0.1,
            'script_weight': 0.05,
        }
        loss = 0.0
        if training_examples:
            for _ in range(max(1, epochs)):
                epoch_loss = 0.0
                for row in training_examples:
                    features = row.get('features', {})
                    x = {
                        'bias_weight': 1.0,
                        'lexical_weight': float(features.get('lexical_overlap', 0.0)),
                        'goal_weight': float(features.get('goal_overlap', 0.0)),
                        'requirement_weight': float(features.get('requirement_overlap', 0.0)),
                        'missing_weight': float(features.get('missing_overlap', 0.0)),
                        'relation_weight': float(features.get('relation_overlap', 0.0)),
                        'operator_weight': float(features.get('operator_overlap', 0.0)),
                        'node_family_weight': float(features.get('node_family_overlap', 0.0)),
                        'script_weight': float(features.get('script_overlap', 0.0)),
                    }
                    y = float(row.get('label', 0.0))
                    z = sum(weights[key] * value for key, value in x.items())
                    pred = 1.0 / (1.0 + math.exp(-z))
                    error = pred - y
                    epoch_loss += -(y * self._safe_log(pred) + (1.0 - y) * self._safe_log(1.0 - pred))
                    for key, value in x.items():
                        weights[key] -= learning_rate * error * value
                loss = epoch_loss / float(max(1, len(training_examples)))

        model = AnalogyPolicyModel(
            bias_weight=round(weights['bias_weight'], 4),
            lexical_weight=round(weights['lexical_weight'], 4),
            goal_weight=round(weights['goal_weight'], 4),
            requirement_weight=round(weights['requirement_weight'], 4),
            missing_weight=round(weights['missing_weight'], 4),
            relation_weight=round(weights['relation_weight'], 4),
            operator_weight=round(weights['operator_weight'], 4),
            node_family_weight=round(weights['node_family_weight'], 4),
            script_weight=round(weights['script_weight'], 4),
            support_floor=0.02,
            plan_guard_weight=round(max(0.5, min(1.8, 0.55 + weights['requirement_weight'] + weights['missing_weight'])), 4),
            verifier_weight=round(max(0.6, min(2.0, 0.6 + weights['goal_weight'] + weights['requirement_weight'] + weights['missing_weight'])), 4),
            operator_priority_weight=round(max(0.3, min(1.6, 0.35 + weights['operator_weight'] + weights['node_family_weight'])), 4),
            trained_on_pairs=max(0, len(graphs) * max(0, len(graphs) - 1) // 2),
            training_examples=len(training_examples),
            training_loss=round(loss, 6),
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    'weights': model.model_dump(),
                    'trained_on_pairs': model.trained_on_pairs,
                    'training_examples': len(training_examples),
                    'training_loss': loss,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
        return AnalogyPolicyTrainingSummary(output_path=str(output), trained_on_pairs=model.trained_on_pairs, model=model.model_dump())

    def _build_training_examples(self, graphs: List[StructuredMeaningGraph]) -> List[dict[str, Any]]:
        examples: List[dict[str, Any]] = []
        scorer = AnalogyPolicyScorer(model=AnalogyPolicyModel())
        for index, left in enumerate(graphs):
            for right in graphs[index + 1:]:
                features = scorer.pair_features(left, right, self._lexical_overlap(left.query, right.query))
                positive = self._is_positive_pair(features)
                dense = sum(float(value) for value in features.values())
                if not positive and dense < 0.2:
                    continue
                examples.append({'features': features, 'label': 1.0 if positive else 0.0})
        return examples

    @staticmethod
    def _is_positive_pair(features: dict[str, float]) -> bool:
        if features.get('goal_overlap', 0.0) >= 0.5 and features.get('requirement_overlap', 0.0) >= 0.5:
            return True
        if features.get('missing_overlap', 0.0) >= 0.5 and features.get('relation_overlap', 0.0) > 0.0:
            return True
        if features.get('node_family_overlap', 0.0) >= 0.5 and features.get('goal_overlap', 0.0) > 0.0:
            return True
        return False

    @staticmethod
    def _lexical_overlap(left: str, right: str) -> float:
        tokens_left = set(re.findall(r"[A-Za-z가-힣0-9]+", left.lower()))
        tokens_right = set(re.findall(r"[A-Za-z가-힣0-9]+", right.lower()))
        if not tokens_left or not tokens_right:
            return 0.0
        return len(tokens_left & tokens_right) / float(len(tokens_left | tokens_right))

    @staticmethod
    def _safe_log(value: float) -> float:
        clipped = max(1e-6, min(1.0 - 1e-6, value))
        return math.log(clipped)


class AnalogyPolicyScorer:
    def __init__(self, model: AnalogyPolicyModel | None = None, model_path: str | Path | None = None) -> None:
        self.model = model or AnalogyPolicyModel.from_path(model_path) or AnalogyPolicyModel()

    def score_pair(self, query_graph: StructuredMeaningGraph, candidate_graph: StructuredMeaningGraph, lexical_score: float = 0.0) -> float:
        features = self.pair_features(query_graph, candidate_graph, lexical_score)
        linear = (
            self.model.bias_weight
            + features['lexical_overlap'] * self.model.lexical_weight
            + features['goal_overlap'] * self.model.goal_weight
            + features['requirement_overlap'] * self.model.requirement_weight
            + features['missing_overlap'] * self.model.missing_weight
            + features['relation_overlap'] * self.model.relation_weight
            + features['operator_overlap'] * self.model.operator_weight
            + features['node_family_overlap'] * self.model.node_family_weight
            + features['script_overlap'] * self.model.script_weight
        )
        score = 1.0 / (1.0 + math.exp(-linear))
        return round(max(self.model.support_floor, min(0.99, score)), 4)

    def rank_graphs(self, query: str, query_graph: StructuredMeaningGraph, graphs: List[StructuredMeaningGraph], top_k: int = 6) -> List[StructuredMeaningGraph]:
        if not graphs:
            return []
        index = QueryEmbeddingIndex()
        hits = index.search(query, graphs, top_k=len(graphs))
        hit_scores = {hit.query: hit.score for hit in hits}
        scored: List[tuple[float, float, StructuredMeaningGraph]] = []
        for graph in graphs:
            lexical_score = float(hit_scores.get(graph.query, 0.0))
            policy_score = self.score_pair(query_graph, graph, lexical_score)
            scored.append((policy_score, lexical_score, graph))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2].query))
        return [item[2] for item in scored[:top_k]]

    @classmethod
    def pair_features(cls, query_graph: StructuredMeaningGraph, candidate_graph: StructuredMeaningGraph, lexical_score: float = 0.0) -> dict[str, float]:
        query_nodes = set(match_concepts(query_graph.query)) | query_graph.node_ids()
        candidate_nodes = set(match_concepts(candidate_graph.query)) | candidate_graph.node_ids()
        query_goals = cls._goal_patterns(query_graph.hidden_goals)
        candidate_goals = cls._goal_patterns(candidate_graph.hidden_goals)
        query_required = set(query_graph.required_premises)
        candidate_required = set(candidate_graph.required_premises)
        query_missing = set(query_graph.missing_premises)
        candidate_missing = set(candidate_graph.missing_premises)
        query_relations = {edge.relation for edge in query_graph.edges}
        candidate_relations = {edge.relation for edge in candidate_graph.edges}
        query_operators = {candidate.family for candidate in query_graph.induced_operators}
        candidate_operators = {candidate.family for candidate in candidate_graph.induced_operators}
        query_scripts = set(query_graph.inferred_scripts)
        candidate_scripts = set(candidate_graph.inferred_scripts)
        return {
            'lexical_overlap': round(max(0.0, lexical_score), 4),
            'goal_overlap': round(cls._overlap(query_goals, candidate_goals), 4),
            'requirement_overlap': round(cls._overlap(query_required, candidate_required), 4),
            'missing_overlap': round(cls._overlap(query_missing, candidate_missing), 4),
            'relation_overlap': round(cls._overlap(query_relations, candidate_relations), 4),
            'operator_overlap': round(cls._overlap(query_operators, candidate_operators), 4),
            'node_family_overlap': round(cls._overlap(cls._node_families(query_nodes), cls._node_families(candidate_nodes)), 4),
            'script_overlap': round(cls._overlap(query_scripts, candidate_scripts), 4),
        }

    @staticmethod
    def _overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / float(len(left | right))

    @staticmethod
    def _node_families(nodes: set[str]) -> set[str]:
        families = {concept_family(node) for node in nodes}
        return {family for family in families if family and family != 'concept'}

    @classmethod
    def _goal_patterns(cls, goals: List[str]) -> set[str]:
        return {cls._goal_pattern(goal) for goal in goals if goal}

    @staticmethod
    def _goal_pattern(goal: str) -> str:
        if re.match(r'^retrieve_item_from_.+_goal$', goal):
            return 'retrieve_item_from_container_goal'
        if re.match(r'^store_.+_in_.+_goal$', goal):
            return 'store_item_in_container_goal'
        if re.match(r'^pass_through_.+_goal$', goal):
            return 'pass_through_barrier_goal'
        return goal

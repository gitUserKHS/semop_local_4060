from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .corpus_store import CorpusMemoryStore
    from .structures import PremiseCandidate, StructuredMeaningGraph


@dataclass
class ScriptCompatibilityBreakdown:
    symbolic_score: float
    learned_score: float
    combined_score: float
    memory_hits: int
    reasons: list[str]


@dataclass
class ScriptCompatibilityModel:
    symbolic_weight: float = 0.55
    learned_weight: float = 0.45
    concept_weight: float = 0.0
    script_weight: float = 0.0
    family_weight: float = 0.0
    operator_weight: float = 0.0
    evolved_operator_weight: float = 0.0
    premise_family_weight: float = 0.0
    goal_weight: float = 0.0
    hit_weight: float = 0.0
    support_floor: float = 0.0
    bias_weight: float = 0.0
    trained_on_hits: int = 0
    training_examples: int = 0
    training_loss: float = 0.0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: str | Path | None) -> ScriptCompatibilityModel | None:
        if not path:
            return None
        payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if isinstance(payload, dict) and 'weights' in payload:
            payload = {**payload.get('weights', {}), 'trained_on_hits': payload.get('trained_on_hits', 0)}
        return cls(**payload)


@dataclass
class ScriptCompatibilityTrainingSummary:
    output_path: str
    trained_on_hits: int
    model: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class ScriptCompatibilityTrainer:
    def train_from_memory(
        self,
        memory_store: CorpusMemoryStore,
        output_path: str | Path,
        source: str | None = None,
        split: str = 'train',
        epochs: int = 120,
        learning_rate: float = 0.18,
    ) -> ScriptCompatibilityTrainingSummary:
        stats = memory_store.collect_script_compatibility_training_stats(source=source, split=split)
        training_examples = memory_store.build_script_compatibility_training_examples(source=source, split=split)
        premise_hits = max(1, int(stats.get('premise_hits', 0)))
        total_hits = premise_hits + max(0, int(stats.get('script_hits', 0))) + max(0, int(stats.get('operator_hits', 0)))

        weights = {
            'bias_weight': 0.0,
            'concept_weight': 0.1,
            'script_weight': 0.1,
            'family_weight': 0.1,
            'operator_weight': 0.1,
            'evolved_operator_weight': 0.1,
            'premise_family_weight': 0.1,
            'goal_weight': 0.1,
            'hit_weight': 0.1,
        }
        loss = 0.0
        if training_examples:
            for _ in range(max(1, epochs)):
                epoch_loss = 0.0
                for row in training_examples:
                    features = row.get('features', {})
                    x = {
                        'bias_weight': 1.0,
                        'concept_weight': float(features.get('concept_overlap', 0.0)),
                        'script_weight': float(features.get('script_overlap', 0.0)),
                        'family_weight': float(features.get('family_overlap', 0.0)),
                        'operator_weight': float(features.get('operator_support', 0.0)),
                        'evolved_operator_weight': float(features.get('evolved_operator_support', 0.0)),
                        'premise_family_weight': 1.0 if float(features.get('family_overlap', 0.0)) >= 0.65 else 0.0,
                        'goal_weight': float(features.get('goal_alignment', 0.0)),
                        'hit_weight': min(1.0, float(features.get('hits', 0.0)) / 8.0),
                    }
                    y = float(row.get('label', 0.0))
                    z = sum(weights[key] * value for key, value in x.items())
                    pred = 1.0 / (1.0 + (2.718281828459045 ** (-z)))
                    error = pred - y
                    epoch_loss += -(y * self._safe_log(pred) + (1.0 - y) * self._safe_log(1.0 - pred))
                    for key, value in x.items():
                        weights[key] -= learning_rate * error * value
                loss = epoch_loss / float(max(1, len(training_examples)))

        model = ScriptCompatibilityModel(
            symbolic_weight=0.48,
            learned_weight=0.52,
            concept_weight=round(weights['concept_weight'], 4),
            script_weight=round(weights['script_weight'], 4),
            family_weight=round(weights['family_weight'], 4),
            operator_weight=round(weights['operator_weight'], 4),
            evolved_operator_weight=round(weights['evolved_operator_weight'], 4),
            premise_family_weight=round(weights['premise_family_weight'], 4),
            goal_weight=round(weights['goal_weight'], 4),
            hit_weight=round(weights['hit_weight'], 4),
            bias_weight=round(weights['bias_weight'], 4),
            support_floor=0.02,
            trained_on_hits=int(total_hits),
            training_examples=len(training_examples),
            training_loss=round(loss, 6),
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'weights': model.model_dump(), 'trained_on_hits': total_hits, 'training_examples': len(training_examples), 'training_loss': loss}, ensure_ascii=False, indent=2), encoding='utf-8')
        return ScriptCompatibilityTrainingSummary(output_path=str(output), trained_on_hits=int(total_hits), model=model.model_dump())

    @staticmethod
    def _safe_log(value: float) -> float:
        clipped = max(1e-6, min(1.0 - 1e-6, value))
        import math
        return math.log(clipped)


class ScriptCompatibilityScorer:
    """Blend symbolic compatibility with lightweight memory-learned priors."""

    def __init__(
        self,
        memory_store: CorpusMemoryStore | None = None,
        memory_source: str | None = None,
        model: ScriptCompatibilityModel | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.memory_source = memory_source
        self.model = model or ScriptCompatibilityModel.from_path(model_path) or ScriptCompatibilityModel()

    def score(
        self,
        graph: StructuredMeaningGraph,
        candidate: PremiseCandidate,
        support_profile: dict[str, set[str]],
        symbolic_score: float,
    ) -> ScriptCompatibilityBreakdown:
        learned_score = 0.0
        memory_hits = 0
        reasons: list[str] = []
        if self.memory_store is not None:
            learned = self.memory_store.collect_script_compatibility_profile(
                graph.query,
                premise=candidate.premise,
                hidden_goal=candidate.hidden_goal,
                source=self.memory_source,
                split='train',
            )
            memory_hits = int(learned.get('hits', 0))
            hit_strength = min(1.0, memory_hits / 8.0)
            linear_score = (
                self.model.bias_weight
                + float(learned.get('concept_overlap', 0.0)) * self.model.concept_weight
                + float(learned.get('script_overlap', 0.0)) * self.model.script_weight
                + float(learned.get('family_overlap', 0.0)) * self.model.family_weight
                + float(learned.get('operator_support', 0.0)) * self.model.operator_weight
                + float(learned.get('evolved_operator_support', 0.0)) * self.model.evolved_operator_weight
                + (1.0 if float(learned.get('family_overlap', 0.0)) >= 0.65 else 0.0) * self.model.premise_family_weight
                + float(learned.get('goal_alignment', 0.0)) * self.model.goal_weight
                + hit_strength * self.model.hit_weight
            )
            import math
            learned_score = 1.0 / (1.0 + math.exp(-linear_score)) if memory_hits else 0.0
            learned_score = min(1.0, max(self.model.support_floor, learned_score if memory_hits else 0.0))
            if memory_hits:
                reasons.append(
                    f"memory profile hits={memory_hits} concepts={learned.get('concept_overlap', 0.0):.2f} families={learned.get('family_overlap', 0.0):.2f} scripts={learned.get('script_overlap', 0.0):.2f} operators={learned.get('operator_support', 0.0):.2f} evolved={learned.get('evolved_operator_support', 0.0):.2f}"
                )

        combined = symbolic_score
        if learned_score > 0.0:
            combined = min(1.0, symbolic_score * self.model.symbolic_weight + learned_score * self.model.learned_weight)
        if candidate.premise in support_profile.get('requirements', set()):
            reasons.append('query concepts directly imply this requirement')
        elif candidate.hidden_goal and candidate.hidden_goal in support_profile.get('goal_hints', set()):
            reasons.append('query profile already supports the hidden goal')
        return ScriptCompatibilityBreakdown(
            symbolic_score=round(min(1.0, symbolic_score), 4),
            learned_score=round(min(1.0, learned_score), 4),
            combined_score=round(min(1.0, combined), 4),
            memory_hits=memory_hits,
            reasons=reasons[:4],
        )

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence


_VISUAL_NOUNS = {
    'image', 'photo', 'picture', 'screenshot', 'scene', 'frame', 'camera', 'visible',
    '\uc0ac\uc9c4', '\uc774\ubbf8\uc9c0', '\ud654\uba74', '\uc7a5\uba74', '\ucea1\ucc98', '\uc2a4\ud06c\ub9b0\uc0f7',
}


def _is_korean_char(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    return 0xAC00 <= code <= 0xD7A3



def _normalize_visual_query(text: str) -> str:
    lowered = str(text or '').lower()
    filtered: list[str] = []
    for char in lowered:
        if char.isalnum() or char == '_' or char.isspace() or _is_korean_char(char):
            filtered.append(char)
        else:
            filtered.append(' ')
    return ' '.join(''.join(filtered).split())



def _tokenize_visual_query(text: str) -> list[str]:
    normalized = _normalize_visual_query(text)
    tokens: list[str] = []
    current: list[str] = []
    current_kind = ''
    for char in normalized:
        if char.isspace():
            if current:
                tokens.append(''.join(current))
                current = []
                current_kind = ''
            continue
        if _is_korean_char(char):
            kind = 'ko'
        elif char.isascii() and (char.isalnum() or char == '_'):
            kind = 'ascii'
        else:
            kind = ''
        if not kind:
            if current:
                tokens.append(''.join(current))
                current = []
                current_kind = ''
            continue
        if current and current_kind != kind:
            tokens.append(''.join(current))
            current = []
        current.append(char)
        current_kind = kind
    if current:
        tokens.append(''.join(current))
    return tokens


@dataclass
class VisualQuestionUnderstandingExample:
    query: str
    intent: str
    weight: float = 1.0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VisualQuestionUnderstandingModel:
    token_votes: dict[str, dict[str, float]] = field(default_factory=dict)
    phrase_votes: dict[str, dict[str, float]] = field(default_factory=dict)
    min_vote: float = 0.55
    dominance_threshold: float = 0.58
    trained_examples: int = 0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: str | Path | None) -> 'VisualQuestionUnderstandingModel | None':
        if not path:
            return None
        target = Path(path)
        if not target.exists():
            return None
        payload = json.loads(target.read_text(encoding='utf-8-sig'))
        if isinstance(payload, dict) and 'weights' in payload:
            payload = payload['weights']
        return cls(**payload)


@dataclass
class VisualQuestionUnderstandingPrediction:
    intent: str = ''
    confidence: float = 0.0
    scores: list[tuple[str, float]] = field(default_factory=list)
    matched_features: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VisualQuestionUnderstandingTrainingSummary:
    output_path: str
    trained_examples: int
    model: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class VisualQuestionUnderstandingTrainer:
    def train_from_examples(
        self,
        examples: Sequence[VisualQuestionUnderstandingExample | dict[str, Any]],
        output_path: str | Path,
    ) -> VisualQuestionUnderstandingTrainingSummary:
        model = self.train_to_model(examples)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({'weights': model.model_dump()}, ensure_ascii=False, indent=2), encoding='utf-8')
        return VisualQuestionUnderstandingTrainingSummary(
            output_path=str(target),
            trained_examples=model.trained_examples,
            model=model.model_dump(),
        )

    def train_to_model(
        self,
        examples: Sequence[VisualQuestionUnderstandingExample | dict[str, Any]],
    ) -> VisualQuestionUnderstandingModel:
        token_votes: dict[str, dict[str, float]] = {}
        phrase_votes: dict[str, dict[str, float]] = {}
        normalized_examples: list[VisualQuestionUnderstandingExample] = []
        for item in examples:
            if isinstance(item, VisualQuestionUnderstandingExample):
                example = item
            else:
                example = VisualQuestionUnderstandingExample(
                    query=str(item.get('query') or ''),
                    intent=str(item.get('intent') or ''),
                    weight=float(item.get('weight') or 1.0),
                )
            if example.query.strip() and example.intent.strip():
                normalized_examples.append(example)
        for example in normalized_examples:
            weight = max(0.3, float(example.weight or 1.0))
            phrase = _normalize_visual_query(example.query)
            if phrase:
                self._bump(phrase_votes, phrase, example.intent, weight * 1.35)
            for token in set(_tokenize_visual_query(example.query)):
                self._bump(token_votes, token, example.intent, weight)
        return VisualQuestionUnderstandingModel(
            token_votes=token_votes,
            phrase_votes=phrase_votes,
            trained_examples=len(normalized_examples),
        )

    @staticmethod
    def _bump(table: dict[str, dict[str, float]], key: str, intent: str, weight: float) -> None:
        if not key or not intent:
            return
        bucket = table.setdefault(key, {})
        bucket[intent] = round(float(bucket.get(intent, 0.0)) + float(weight), 4)


class VisualQuestionUnderstandingEngine:
    DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / 'data' / 'knowledge' / 'visual_question_understanding.json'

    def __init__(self, model: VisualQuestionUnderstandingModel | None = None, model_path: str | Path | None = None) -> None:
        resolved_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self.model = model or VisualQuestionUnderstandingModel.from_path(resolved_path) or self._seed_model()

    def predict(self, query: str) -> VisualQuestionUnderstandingPrediction:
        normalized = _normalize_visual_query(query)
        tokens = _tokenize_visual_query(query)
        if not normalized and not tokens:
            return VisualQuestionUnderstandingPrediction()
        votes: dict[str, float] = {}
        matched_features: list[str] = []
        phrase_matches = [phrase for phrase in self.model.phrase_votes if phrase and phrase in normalized]
        for phrase in phrase_matches:
            matched_features.append(f'phrase:{phrase}')
            for intent, weight in self.model.phrase_votes.get(phrase, {}).items():
                votes[intent] = float(votes.get(intent, 0.0)) + float(weight)
        for token in tokens:
            bucket = self.model.token_votes.get(token, {})
            if not bucket:
                continue
            matched_features.append(f'token:{token}')
            for intent, weight in bucket.items():
                votes[intent] = float(votes.get(intent, 0.0)) + float(weight)
        ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        visualish = bool(set(tokens) & _VISUAL_NOUNS or phrase_matches)
        if not ranked:
            fallback_intent = 'generic_visual' if visualish else ''
            return VisualQuestionUnderstandingPrediction(intent=fallback_intent, confidence=0.0, scores=[], matched_features=[])
        top_intent, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        score_total = sum(score for _, score in ranked[:3]) or top_score
        share = float(top_score) / float(score_total or 1.0)
        margin = 1.0 if top_score <= 0.0 else max(0.0, (float(top_score) - float(second_score)) / float(top_score))
        coverage = len(dict.fromkeys(matched_features)) / float(max(1, len(set(tokens)) + len(phrase_matches)))
        confidence = round((0.45 * share) + (0.35 * margin) + (0.2 * min(1.0, coverage)), 4)
        intent = top_intent
        if float(top_score) < float(self.model.min_vote):
            intent = 'generic_visual' if visualish else ''
        elif share < float(self.model.dominance_threshold) and visualish:
            intent = 'generic_visual'
        return VisualQuestionUnderstandingPrediction(
            intent=intent,
            confidence=confidence,
            scores=[(name, round(float(score), 4)) for name, score in ranked[:5]],
            matched_features=list(dict.fromkeys(matched_features))[:8],
        )

    @classmethod
    def _seed_model(cls) -> VisualQuestionUnderstandingModel:
        trainer = VisualQuestionUnderstandingTrainer()
        return trainer.train_to_model(cls._seed_examples())

    @staticmethod
    def _seed_examples() -> list[VisualQuestionUnderstandingExample]:
        return [
            VisualQuestionUnderstandingExample('Describe this image.', 'scene_description', 1.0),
            VisualQuestionUnderstandingExample('Describe this photo.', 'scene_description', 1.0),
            VisualQuestionUnderstandingExample('What is happening in this image?', 'scene_description', 1.0),
            VisualQuestionUnderstandingExample('What does this screenshot show?', 'scene_description', 0.9),
            VisualQuestionUnderstandingExample('\uc774 \uc0ac\uc9c4\uc744 \uc124\uba85\ud574\ubd10.', 'scene_description', 1.1),
            VisualQuestionUnderstandingExample('\uc774 \uc774\ubbf8\uc9c0\ub97c \uc124\uba85\ud574\uc918.', 'scene_description', 1.1),
            VisualQuestionUnderstandingExample('\ubb34\uc2a8 \uc7a5\uba74\uc778\uac00\uc694?', 'scene_description', 1.0),
            VisualQuestionUnderstandingExample('\ubb34\uc2a8 \uc0c1\ud669\uc778\uac00\uc694?', 'scene_description', 1.0),
            VisualQuestionUnderstandingExample('What do you see?', 'object_inventory', 1.0),
            VisualQuestionUnderstandingExample('What objects are visible here?', 'object_inventory', 1.0),
            VisualQuestionUnderstandingExample('What is visible in this image?', 'object_inventory', 1.0),
            VisualQuestionUnderstandingExample('\uc774 \uc0ac\uc9c4\uc5d0\uc11c \ubb50\uac00 \ubcf4\uc5ec?', 'object_inventory', 1.2),
            VisualQuestionUnderstandingExample('\uc774 \uc0ac\uc9c4\uc5d0\uc11c \ubb50\uac00 \ubcf4\uc774\ub098\uc694?', 'object_inventory', 1.3),
            VisualQuestionUnderstandingExample('\uc774 \uc0ac\uc9c4\uc5d0 \ubb50\uac00 \uc788\ub098\uc694?', 'object_inventory', 1.2),
            VisualQuestionUnderstandingExample('\ubb34\uc5c7\uc774 \ubcf4\uc774\ub098\uc694?', 'object_inventory', 1.0),
            VisualQuestionUnderstandingExample('\ubb50\uac00 \ubcf4\uc774\ub098\uc694?', 'object_inventory', 1.0),
            VisualQuestionUnderstandingExample('Where is the object?', 'spatial_relation', 1.0),
            VisualQuestionUnderstandingExample('What is the position of the handle?', 'spatial_relation', 0.9),
            VisualQuestionUnderstandingExample('\uc5b4\ub514\uc5d0 \uc788\ub098\uc694?', 'spatial_relation', 1.0),
            VisualQuestionUnderstandingExample('\uc704\uce58\uac00 \uc5b4\ub514\uc778\uac00\uc694?', 'spatial_relation', 1.0),
            VisualQuestionUnderstandingExample('How can I access the opening?', 'access_reasoning', 1.1),
            VisualQuestionUnderstandingExample('How do I open this bag?', 'access_reasoning', 1.0),
            VisualQuestionUnderstandingExample('\uc5b4\ub5bb\uac8c \uc5f4 \uc218 \uc788\ub098\uc694?', 'access_reasoning', 1.0),
            VisualQuestionUnderstandingExample('\uc5b4\ub514\ub85c \uc811\uadfc\ud558\ub098\uc694?', 'access_reasoning', 1.0),
            VisualQuestionUnderstandingExample('What shape is this?', 'geometry', 1.0),
            VisualQuestionUnderstandingExample('Is this triangle parallel to that line?', 'geometry', 1.0),
            VisualQuestionUnderstandingExample('\uc774 \ub3c4\ud615\uc740 \ubb34\uc5c7\uc778\uac00\uc694?', 'geometry', 1.0),
            VisualQuestionUnderstandingExample('\ud3c9\ud589\ud55c\uac00\uc694?', 'geometry', 1.0),
            VisualQuestionUnderstandingExample('Is the door open?', 'state_check', 1.0),
            VisualQuestionUnderstandingExample('What state is the drawer in?', 'state_check', 1.0),
            VisualQuestionUnderstandingExample('\ubb38\uc774 \uc5f4\ub824 \uc788\ub098\uc694?', 'state_check', 1.0),
            VisualQuestionUnderstandingExample('\ub2eb\ud600 \uc788\ub098\uc694?', 'state_check', 0.9),
        ]

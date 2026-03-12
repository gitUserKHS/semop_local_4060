from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, List

from .contest_programmer import CompetitiveProgrammingReasoner
from .cp_dataset import CpDslExample, load_cp_dsl_examples


@dataclass
class CpParserPrediction:
    statement: str
    goal_types: List[str]
    domain_tags: List[str]
    logical_frames: List[str]
    dsl_operators: List[str]
    target_algorithm: str
    reasoning_sketch: str
    raw_output: str = ""

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class CpParserEvalSummary:
    num_examples: int
    algorithm_exact_match: float
    goal_jaccard: float
    domain_jaccard: float
    frame_jaccard: float
    operator_jaccard: float
    reasoning_nonempty_rate: float

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class CpParserComparisonSummary:
    num_examples: int
    heuristic_summary: dict
    model_summary: dict
    deltas: dict

    def model_dump(self) -> dict:
        return asdict(self)


class CpLearnedParser:
    def __init__(self, model_name_or_path: str, local_files_only: bool = False, max_new_tokens: int = 256) -> None:
        self.model_name_or_path = model_name_or_path
        self.local_files_only = local_files_only
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    @staticmethod
    def _detect_adapter_base_model(model_name_or_path: str) -> str | None:
        path = Path(model_name_or_path)
        adapter_config = path / 'adapter_config.json'
        if not adapter_config.exists():
            return None
        payload = json.loads(adapter_config.read_text(encoding='utf-8'))
        return payload.get('base_model_name_or_path')

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model_name = self._detect_adapter_base_model(self.model_name_or_path)
        tokenizer_source = self.model_name_or_path
        if base_model_name:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=self.local_files_only)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, local_files_only=self.local_files_only)
            try:
                from peft import PeftModel
            except Exception as exc:
                raise RuntimeError('PEFT adapter detected but `peft` is not available for loading the trained parser.') from exc
            self._model = PeftModel.from_pretrained(base_model, self.model_name_or_path)
            if hasattr(self._model, 'merge_and_unload'):
                try:
                    self._model = self._model.merge_and_unload()
                except Exception:
                    pass
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=self.local_files_only)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, local_files_only=self.local_files_only)
        if hasattr(self._model, 'eval'):
            self._model.eval()

    @staticmethod
    def _build_prompt(statement: str) -> str:
        return (
            "Read the contest problem and output JSON with keys: goal_types, domain_tags, logical_frames, "
            "dsl_operators, target_algorithm, reasoning_sketch.\n\n"
            f"Problem:\n{statement}"
        )

    @staticmethod
    def _parse_payload(text: str) -> dict:
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            raise ValueError('No JSON object found in model output.')
        return json.loads(text[start : end + 1])

    def predict(self, statement: str) -> CpParserPrediction:
        self._load()
        import torch

        prompt = self._build_prompt(statement)
        encoded = self._tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        text = self._tokenizer.decode(generated[0], skip_special_tokens=True)
        payload = self._parse_payload(text)
        return CpParserPrediction(
            statement=statement,
            goal_types=list(payload.get('goal_types', [])),
            domain_tags=list(payload.get('domain_tags', [])),
            logical_frames=list(payload.get('logical_frames', [])),
            dsl_operators=list(payload.get('dsl_operators', [])),
            target_algorithm=str(payload.get('target_algorithm', '')),
            reasoning_sketch=str(payload.get('reasoning_sketch', '')),
            raw_output=text,
        )


class CpParserEvaluator:
    def __init__(self, heuristic_reasoner: CompetitiveProgrammingReasoner | None = None) -> None:
        self.heuristic_reasoner = heuristic_reasoner or CompetitiveProgrammingReasoner()

    @staticmethod
    def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
        left_set = set(left)
        right_set = set(right)
        if not left_set and not right_set:
            return 1.0
        if not left_set or not right_set:
            return 0.0
        return len(left_set & right_set) / len(left_set | right_set)

    def predict_heuristic(self, statement: str) -> CpParserPrediction:
        structure = self.heuristic_reasoner.parse_problem(statement)
        if structure is None:
            return CpParserPrediction(statement=statement, goal_types=[], domain_tags=[], logical_frames=[], dsl_operators=[], target_algorithm='', reasoning_sketch='')
        top_algorithm = structure.candidate_algorithms[0] if structure.candidate_algorithms else ''
        reasoning_sketch = (
            'Reduce the statement into frames, operators, hidden constraints, and only then choose an algorithm family.'
            if (structure.logical_frames or structure.dsl_operators)
            else 'No strong parser structure match was found.'
        )
        return CpParserPrediction(
            statement=statement,
            goal_types=structure.goal_types,
            domain_tags=structure.domain_tags,
            logical_frames=structure.logical_frames,
            dsl_operators=structure.dsl_operators,
            target_algorithm=top_algorithm,
            reasoning_sketch=reasoning_sketch,
        )

    def evaluate_examples(self, examples: Iterable[CpDslExample], predictor: str = 'heuristic', model: CpLearnedParser | None = None) -> CpParserEvalSummary:
        examples = list(examples)
        if not examples:
            return CpParserEvalSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        algorithm_hits = 0
        goal_scores: List[float] = []
        domain_scores: List[float] = []
        frame_scores: List[float] = []
        operator_scores: List[float] = []
        reasoning_hits = 0
        for example in examples:
            if predictor == 'heuristic':
                prediction = self.predict_heuristic(example.statement)
            else:
                if model is None:
                    raise ValueError('A learned parser model must be provided for predictor="model".')
                prediction = model.predict(example.statement)
            algorithm_hits += int(prediction.target_algorithm == example.target_algorithm)
            goal_scores.append(self._jaccard(prediction.goal_types, example.goal_types))
            domain_scores.append(self._jaccard(prediction.domain_tags, example.domain_tags))
            frame_scores.append(self._jaccard(prediction.logical_frames, example.logical_frames))
            operator_scores.append(self._jaccard(prediction.dsl_operators, example.dsl_operators))
            reasoning_hits += int(bool(prediction.reasoning_sketch.strip()))
        total = len(examples)
        return CpParserEvalSummary(
            num_examples=total,
            algorithm_exact_match=round(algorithm_hits / total, 4),
            goal_jaccard=round(sum(goal_scores) / total, 4),
            domain_jaccard=round(sum(domain_scores) / total, 4),
            frame_jaccard=round(sum(frame_scores) / total, 4),
            operator_jaccard=round(sum(operator_scores) / total, 4),
            reasoning_nonempty_rate=round(reasoning_hits / total, 4),
        )

    def compare_examples(self, examples: Iterable[CpDslExample], model: CpLearnedParser | None = None) -> CpParserComparisonSummary:
        examples = list(examples)
        heuristic_summary = self.evaluate_examples(examples, predictor='heuristic').model_dump()
        model_summary = self.evaluate_examples(examples, predictor='model', model=model).model_dump() if model is not None else {}
        deltas = {}
        if model_summary:
            for key in ['algorithm_exact_match', 'goal_jaccard', 'domain_jaccard', 'frame_jaccard', 'operator_jaccard', 'reasoning_nonempty_rate']:
                deltas[key + '_delta'] = round(float(model_summary.get(key, 0.0)) - float(heuristic_summary.get(key, 0.0)), 4)
        return CpParserComparisonSummary(
            num_examples=len(examples),
            heuristic_summary=heuristic_summary,
            model_summary=model_summary,
            deltas=deltas,
        )

    @staticmethod
    def load_examples(path: str | Path) -> List[CpDslExample]:
        return load_cp_dsl_examples(path)

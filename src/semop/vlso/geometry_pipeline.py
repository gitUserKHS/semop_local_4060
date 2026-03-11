from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .affordance_features import VisualAffordanceFeatureExtractor
from .affordance_classifier import WeakAffordanceClassifier
from .eval import VlsoGroundedEvaluator
from .image_parser import RawImageObservationParser
from .operator_learning import VisualOperatorPrototypeTrainer
from .reasoner import VLSOReasoner
from .self_training import PseudoLabelAcceptanceConfig, VisualConceptSelfTrainer
from .visual_parser import VLSOVisualParser


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
VISUAL_SUFFIXES = IMAGE_SUFFIXES | {'.json'}


@dataclass
class GeometryPipelineSummary:
    inputs: list[str]
    image_count: int
    candidates_path: str
    pseudo_labels_path: str
    concept_store_path: str
    operator_store_path: str
    candidate_images: int
    pseudo_targets: int
    concept_summary: dict[str, Any]
    operator_summary: dict[str, Any]
    cluster_summary_path: str | None = None
    eval_summary: dict[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class VisualGeometryBootstrapPipeline:
    def __init__(self, weights_path: str | None = None) -> None:
        self.image_parser = RawImageObservationParser()
        self.visual_parser = VLSOVisualParser(affordance_weights_path=weights_path)
        self.feature_extractor = VisualAffordanceFeatureExtractor()
        self.classifier = WeakAffordanceClassifier(weights_path) if weights_path else WeakAffordanceClassifier()
        self.weights_path = weights_path

    def collect_input_paths(self, inputs: Iterable[str | Path]) -> list[str]:
        source_paths: list[str] = []
        for item in inputs:
            path = Path(item)
            if path.is_dir():
                for child in sorted(path.rglob('*')):
                    if child.is_file() and child.suffix.lower() in VISUAL_SUFFIXES:
                        source_paths.append(str(child))
            elif path.is_file() and path.suffix.lower() in VISUAL_SUFFIXES:
                source_paths.append(str(path))
        deduped: list[str] = []
        seen: set[str] = set()
        for item in source_paths:
            resolved = str(Path(item).resolve())
            if resolved in seen:
                continue
            deduped.append(resolved)
            seen.add(resolved)
        return deduped

    def build_candidates(self, source_paths: Iterable[str], output_path: str | Path) -> int:
        rows = []
        for source_path in source_paths:
            observation = self._load_observation(source_path)
            candidates = self.feature_extractor.extract(observation)
            rows.append(
                {
                    'image_path': source_path,
                    'source_kind': Path(source_path).suffix.lower().lstrip('.'),
                    'metadata': observation.metadata,
                    'targets': [
                        {
                            'subject_id': item.subject,
                            'parent': item.parent,
                            'bbox': item.object_data.get('bbox'),
                            'shape_hint': item.object_data.get('shape_hint'),
                            'feature_vector': {key: round(float(value), 6) for key, value in item.features.items()},
                            'prediction_details': [
                                {'label': pred.label, 'confidence': pred.confidence, 'score': pred.score}
                                for pred in self.classifier.predict(item.features, limit=5, threshold=0.0)
                            ],
                            'suggested_labels': [pred.label for pred in self.classifier.predict(item.features, limit=5, threshold=0.45)],
                            'positive_labels': [],
                            'negative_labels': [],
                            'notes': '',
                        }
                        for item in candidates
                    ],
                }
            )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        return len(rows)

    def build_pseudo_labels(self, candidates_path: str | Path, output_path: str | Path, confidence_threshold: float = 0.62, max_labels: int = 3) -> int:
        rows_out = []
        target_count = 0
        for raw_line in Path(candidates_path).read_text(encoding='utf-8-sig').splitlines():
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            targets_out = []
            for target in row.get('targets', []):
                predictions = target.get('prediction_details') or []
                positives = [str(item.get('label', '')) for item in predictions if float(item.get('confidence', 0.0)) >= confidence_threshold and str(item.get('label', '')).strip()]
                if not positives:
                    positives = [str(item) for item in (target.get('suggested_labels') or [])[:max_labels] if str(item).strip()]
                positives = list(dict.fromkeys(positives))[:max_labels]
                if not positives:
                    continue
                targets_out.append(
                    {
                        'subject_id': str(target.get('subject_id', '')),
                        'positive_labels': positives,
                        'negative_labels': [],
                    }
                )
                target_count += 1
            if targets_out:
                rows_out.append({'image_path': row.get('image_path', ''), 'targets': targets_out})
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', encoding='utf-8') as handle:
            for row in rows_out:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        return target_count

    def run(self, inputs: Iterable[str | Path], candidates_path: str | Path, pseudo_labels_path: str | Path, concept_store_path: str | Path, operator_store_path: str | Path, eval_input: str | None = None, eval_mode: str = 'heuristic', answer_mode: str = 'structured', config: PseudoLabelAcceptanceConfig | None = None, concept_summary_output: str | Path | None = None) -> GeometryPipelineSummary:
        source_paths = self.collect_input_paths(inputs)
        candidate_images = self.build_candidates(source_paths, candidates_path)
        pseudo_targets = self.build_pseudo_labels(candidates_path, pseudo_labels_path)
        trainer = VisualConceptSelfTrainer(weights_path=self.weights_path)
        concept_summary = trainer.train_candidates_jsonl(candidates_path, concept_store_path, summary_output=concept_summary_output, config=config or PseudoLabelAcceptanceConfig()).model_dump()
        operator_summary = VisualOperatorPrototypeTrainer().train_jsonl(pseudo_labels_path, operator_store_path).model_dump()
        eval_summary = None
        if eval_input:
            reasoner = VLSOReasoner(
                mode=eval_mode,
                answer_mode=answer_mode,
                affordance_weights_path=self.weights_path,
                concept_store_path=str(concept_store_path),
                operator_store_path=str(operator_store_path),
            )
            eval_summary = VlsoGroundedEvaluator(reasoner).evaluate_cases(VlsoGroundedEvaluator.load_cases(eval_input)).model_dump()
        return GeometryPipelineSummary(
            inputs=[str(item) for item in inputs],
            image_count=len(source_paths),
            candidates_path=str(candidates_path),
            pseudo_labels_path=str(pseudo_labels_path),
            concept_store_path=str(concept_store_path),
            operator_store_path=str(operator_store_path),
            candidate_images=candidate_images,
            pseudo_targets=pseudo_targets,
            concept_summary=concept_summary,
            operator_summary=operator_summary,
            cluster_summary_path=str(concept_summary_output) if concept_summary_output else None,
            eval_summary=eval_summary,
        )

    def _load_observation(self, source_path: str):
        path = Path(source_path)
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return self.image_parser.parse_image(source_path).observation
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        return self.visual_parser._coerce(payload)

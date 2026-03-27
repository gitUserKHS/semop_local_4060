from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..hardware_profiles import detect_local_hardware
from .vision_backbones import DEFAULT_VISION_MODEL_ROOT, resolve_local_vision_model_path


SCENE_LABELS = [
    'video game screenshot',
    'first-person shooter game screenshot',
    'combat video game scene',
    'third-person action game scene',
    'urban street',
    'market street',
    'shopfront',
    'street market',
    'alley',
    'outdoor daytime scene',
    'warehouse aisle',
    'indoor room',
    'construction site',
    'parking lot',
    'close-up object photo',
    'product photo',
    'bag or backpack photo',
    'street scene',
]

OBJECT_LABELS = [
    'person',
    'human character',
    'female game character',
    'soldier',
    'handgun',
    'pistol',
    'rifle',
    'gun held in first person view',
    'weapon',
    'player hands',
    'hands',
    'arms',
    'backpack',
    'bag',
    'travel bag',
    'suitcase',
    'market stall',
    'shop awning',
    'cart',
    'street cart',
    'building',
    'building facade',
    'signboard',
    'dome',
    'store counter',
    'kiosk',
    'doorway',
    'bicycle',
    'car',
]

OVERLAY_LABELS = [
    'game HUD',
    'mini-map overlay',
    'crosshair overlay',
    'scoreboard overlay',
    'timer overlay',
    'ammo counter overlay',
    'kill feed overlay',
    'chat overlay',
]

SCENE_PROMPT_TEMPLATES = [
    'a photo of {}',
    'an image of {}',
    'a screenshot of {}',
    'this scene shows {}',
]

OBJECT_PROMPT_TEMPLATES = [
    'a photo of {}',
    'an image of {}',
    'a close-up of {}',
    'this region shows {}',
]

OVERLAY_PROMPT_TEMPLATES = [
    'a screenshot with {}',
    'a game interface with {}',
    'an image that includes {}',
]


@dataclass
class SemanticSceneHypothesis:
    label: str
    score: float
    category: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticRegionHypothesis:
    entity_id: str
    label: str
    score: float
    candidates: list[SemanticSceneHypothesis] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            'entity_id': self.entity_id,
            'label': self.label,
            'score': self.score,
            'candidates': [item.model_dump() for item in self.candidates],
        }


@dataclass
class SemanticSceneSummary:
    backend: str
    backend_ready: bool
    semantic_level: str
    caption: str
    scene_hypotheses: list[SemanticSceneHypothesis] = field(default_factory=list)
    object_hypotheses: list[SemanticSceneHypothesis] = field(default_factory=list)
    overlay_hypotheses: list[SemanticSceneHypothesis] = field(default_factory=list)
    region_hypotheses: list[SemanticRegionHypothesis] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            'backend': self.backend,
            'backend_ready': self.backend_ready,
            'semantic_level': self.semantic_level,
            'caption': self.caption,
            'scene_hypotheses': [item.model_dump() for item in self.scene_hypotheses],
            'object_hypotheses': [item.model_dump() for item in self.object_hypotheses],
            'overlay_hypotheses': [item.model_dump() for item in self.overlay_hypotheses],
            'region_hypotheses': [item.model_dump() for item in self.region_hypotheses],
            'notes': list(self.notes),
        }


class SemanticSceneAnalyzer:
    _MODEL_CACHE: dict[tuple[str, str], tuple[object, object]] = {}
    _TEXT_CACHE_BY_MODEL: dict[tuple[str, str], dict[tuple[str, tuple[str, ...], str], Any]] = {}

    def __init__(self, model_path: str | None = None, hardware_profile: str = 'auto') -> None:
        self.hardware = detect_local_hardware(hardware_profile)
        self.model_path = model_path or resolve_local_vision_model_path('openclip_adapter')
        if not self.model_path:
            fallback = DEFAULT_VISION_MODEL_ROOT / 'openclip' / 'clip-vit-base-patch32'
            if fallback.exists():
                self.model_path = str(fallback)
        self._model = None
        self._processor = None
        self._device = 'cuda' if self.hardware.cuda_available else 'cpu'
        self._load_error = ''
        cache_key = (str(self.model_path or ''), self._device)
        self._text_cache = self._TEXT_CACHE_BY_MODEL.setdefault(cache_key, {})

    def analyze(self, image_path: str, query: str = '', candidate_regions: list[dict[str, Any]] | None = None) -> SemanticSceneSummary:
        path = Path(image_path)
        if not path.exists():
            return SemanticSceneSummary(
                backend='openclip_local',
                backend_ready=False,
                semantic_level='weak',
                caption='',
                notes=['semantic scene analyzer could not find the image path'],
            )
        if not self.model_path or not Path(self.model_path).exists():
            return SemanticSceneSummary(
                backend='openclip_local',
                backend_ready=False,
                semantic_level='weak',
                caption='',
                notes=['no local semantic vision model is available under models/vision/openclip'],
            )
        try:
            from PIL import Image
            import torch
            from transformers import AutoModel, AutoProcessor
        except Exception as exc:
            return SemanticSceneSummary(
                backend='openclip_local',
                backend_ready=False,
                semantic_level='weak',
                caption='',
                notes=[f'semantic scene dependencies unavailable: {exc}'],
            )
        try:
            if self._model is None or self._processor is None:
                cache_key = (str(self.model_path), self._device)
                cached = self._MODEL_CACHE.get(cache_key)
                if cached is None:
                    processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True, use_fast=False)
                    model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
                    model.to(self._device)
                    model.eval()
                    self._MODEL_CACHE[cache_key] = (processor, model)
                    cached = (processor, model)
                self._processor, self._model = cached
            image = Image.open(path).convert('RGB')
            scene_scores = self._rank_image_text(image, SCENE_LABELS, 'scene', prompt_templates=SCENE_PROMPT_TEMPLATES, torch_module=torch)
            object_scores = self._rank_image_text(image, OBJECT_LABELS, 'object', prompt_templates=OBJECT_PROMPT_TEMPLATES, torch_module=torch)
            overlay_scores = self._rank_regions_for_overlay(image, torch)
            region_scores = self._classify_regions(image, candidate_regions or [], torch)
            caption = self._compose_caption(scene_scores, object_scores, overlay_scores, region_scores)
            semantic_level = self._semantic_level(scene_scores, object_scores, overlay_scores, region_scores)
            notes = [f'local semantic backbone: {self.model_path}']
            if query:
                notes.append('semantic scene retrieval was conditioned only lightly on the prompt; structural grounding still verifies the final answer')
            if region_scores:
                notes.append(f'region-level semantic labels recovered for {len(region_scores)} visual regions')
            return SemanticSceneSummary(
                backend='openclip_local',
                backend_ready=True,
                semantic_level=semantic_level,
                caption=caption,
                scene_hypotheses=scene_scores,
                object_hypotheses=object_scores,
                overlay_hypotheses=overlay_scores,
                region_hypotheses=region_scores,
                notes=notes,
            )
        except Exception as exc:
            self._load_error = str(exc)
            return SemanticSceneSummary(
                backend='openclip_local',
                backend_ready=False,
                semantic_level='weak',
                caption='',
                notes=[f'semantic scene analysis failed: {exc}'],
            )

    def _rank_regions_for_overlay(self, image, torch_module) -> list[SemanticSceneHypothesis]:
        width, height = image.size
        regions = [
            image,
            image.crop((0, 0, max(1, int(width * 0.28)), max(1, int(height * 0.28)))),
            image.crop((max(0, int(width * 0.72)), 0, width, max(1, int(height * 0.28)))),
            image.crop((max(0, int(width * 0.2)), max(0, int(height * 0.55)), min(width, int(width * 0.85)), height)),
        ]
        best: dict[str, float] = {}
        for region in regions:
            for item in self._rank_image_text(region, OVERLAY_LABELS, 'overlay', prompt_templates=OVERLAY_PROMPT_TEMPLATES, torch_module=torch_module):
                best[item.label] = max(best.get(item.label, -1.0), item.score)
        ranked = sorted(best.items(), key=lambda row: row[1], reverse=True)[:4]
        return [SemanticSceneHypothesis(label=label, score=round(score, 4), category='overlay') for label, score in ranked if score >= 0.22]

    def _rank_image_text(self, image, labels: list[str], category: str, *, prompt_templates: list[str], torch_module) -> list[SemanticSceneHypothesis]:
        image_features = self._image_features(image, torch_module)
        scores = None
        for prompt_template in prompt_templates:
            text_prompts = [prompt_template.format(label) for label in labels]
            text_features = self._text_features(text_prompts, torch_module, cache_key=prompt_template)
            next_scores = text_features @ image_features
            scores = next_scores if scores is None else scores + next_scores
        scores = scores / max(1, len(prompt_templates))
        ranked: list[SemanticSceneHypothesis] = []
        for index in scores.argsort(descending=True)[:5].tolist():
            score = float(scores[index].item())
            ranked.append(SemanticSceneHypothesis(label=labels[index], score=round(score, 4), category=category))
        return ranked

    def _image_features(self, image, torch_module):
        inputs = self._processor(images=image, return_tensors='pt')
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch_module.no_grad():
            if hasattr(self._model, 'vision_model') and hasattr(self._model, 'visual_projection'):
                vision_outputs = self._model.vision_model(**inputs)
                pooled = getattr(vision_outputs, 'pooler_output', None)
                if pooled is None and isinstance(vision_outputs, tuple) and len(vision_outputs) > 1:
                    pooled = vision_outputs[1]
                if pooled is None:
                    raise RuntimeError('semantic scene model did not produce vision pooler output')
                features = self._model.visual_projection(pooled)
            elif hasattr(self._model, 'get_image_features'):
                features = self._coerce_feature_tensor(self._model.get_image_features(**inputs), feature_key='image_embeds')
            else:
                outputs = self._model(**inputs)
                features = self._coerce_feature_tensor(outputs, feature_key='image_embeds')
        return self._normalize(features[0], torch_module)

    def _text_features(self, texts: list[str], torch_module, *, cache_key: str):
        key = (cache_key, tuple(texts), self._device)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        inputs = self._processor(text=texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch_module.no_grad():
            if hasattr(self._model, 'text_model') and hasattr(self._model, 'text_projection'):
                text_outputs = self._model.text_model(**inputs)
                pooled = getattr(text_outputs, 'pooler_output', None)
                if pooled is None and isinstance(text_outputs, tuple) and len(text_outputs) > 1:
                    pooled = text_outputs[1]
                if pooled is None:
                    raise RuntimeError('semantic scene model did not produce text pooler output')
                features = self._model.text_projection(pooled)
            elif hasattr(self._model, 'get_text_features'):
                features = self._coerce_feature_tensor(self._model.get_text_features(**inputs), feature_key='text_embeds')
            else:
                outputs = self._model(**inputs)
                features = self._coerce_feature_tensor(outputs, feature_key='text_embeds')
        normalized = self._normalize_batch(features, torch_module)
        self._text_cache[key] = normalized
        return normalized

    @staticmethod
    def _normalize(vector, torch_module):
        denom = torch_module.norm(vector, p=2).clamp(min=1e-8)
        return vector / denom

    @staticmethod
    def _normalize_batch(matrix, torch_module):
        denom = torch_module.norm(matrix, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        return matrix / denom

    @staticmethod
    def _coerce_feature_tensor(outputs, *, feature_key: str):
        if hasattr(outputs, feature_key):
            tensor = getattr(outputs, feature_key)
            if tensor is not None:
                return tensor
        if hasattr(outputs, 'pooler_output'):
            tensor = getattr(outputs, 'pooler_output')
            if tensor is not None:
                return tensor
        if isinstance(outputs, tuple):
            for item in outputs:
                if hasattr(item, 'shape'):
                    return item
        if hasattr(outputs, 'shape'):
            return outputs
        raise RuntimeError(f'semantic scene model did not produce usable {feature_key} features')

    def _semantic_level(
        self,
        scenes: list[SemanticSceneHypothesis],
        objects: list[SemanticSceneHypothesis],
        overlays: list[SemanticSceneHypothesis],
        regions: list[SemanticRegionHypothesis],
    ) -> str:
        top_scene = scenes[0].score if scenes else 0.0
        top_object = objects[0].score if objects else 0.0
        top_overlay = overlays[0].score if overlays else 0.0
        top_region = regions[0].score if regions else 0.0
        if top_scene >= 0.26 and (top_object >= 0.24 or top_region >= 0.24):
            return 'semantic_grounded'
        if top_scene >= 0.23 or top_object >= 0.24 or top_overlay >= 0.23 or top_region >= 0.23:
            return 'semantic_candidate'
        return 'weak'

    def _compose_caption(
        self,
        scenes: list[SemanticSceneHypothesis],
        objects: list[SemanticSceneHypothesis],
        overlays: list[SemanticSceneHypothesis],
        regions: list[SemanticRegionHypothesis],
    ) -> str:
        top_scene = scenes[0] if scenes else None
        top_objects = [item.label for item in objects[:4] if item.score >= 0.2]
        top_overlays = [item.label for item in overlays[:3] if item.score >= 0.22]
        top_regions = [item.label for item in regions[:4] if item.score >= 0.2]
        top_object_score = objects[0].score if objects else 0.0
        top_region_score = regions[0].score if regions else 0.0
        top_overlay_score = overlays[0].score if overlays else 0.0
        top_scene_score = top_scene.score if top_scene else 0.0
        top_scene_label = top_scene.label if top_scene else ''
        game_like = top_scene_label in {'first-person shooter game screenshot', 'combat video game scene', 'video game screenshot'} and top_scene_score >= 0.24 and (top_overlay_score >= 0.22 or top_object_score >= 0.24 or top_region_score >= 0.24)
        bag_like = any(item in top_objects for item in ['backpack', 'bag', 'travel bag', 'suitcase']) or any(item in top_regions for item in ['backpack', 'bag', 'travel bag', 'suitcase'])
        market_like = top_scene_label in {'market street', 'street market', 'shopfront', 'street scene'} and top_scene_score >= 0.23
        sentences: list[str] = []
        if game_like:
            sentences.append('This appears to be a first-person or combat game screenshot.')
            if any(item in top_objects for item in ['handgun', 'pistol', 'rifle', 'weapon', 'gun held in first person view']) or any(item in top_regions for item in ['handgun', 'pistol', 'rifle', 'weapon', 'gun held in first person view']):
                sentences.append('A weapon is visible in the foreground from the player viewpoint.')
            if any(item in top_objects for item in ['person', 'human character', 'soldier']) or any(item in top_regions for item in ['female game character', 'human character', 'soldier']):
                sentences.append('At least one human-like character is visible in front of the camera.')
            if top_overlays:
                sentences.append('Overlay elements likely include ' + ', '.join(top_overlays) + '.')
            return ' '.join(sentences).strip()
        if bag_like and max(top_object_score, top_region_score) >= 0.24:
            sentences.append('This image most likely centers on a bag or backpack.')
        elif market_like and (top_object_score >= 0.22 or top_region_score >= 0.22):
            sentences.append('The scene most likely looks like a street market or shopfront area.')
        elif top_scene_label and top_scene_score >= 0.24 and top_scene_label not in {'video game screenshot', 'first-person shooter game screenshot', 'combat video game scene'}:
            sentences.append(f'The scene most likely looks like a {top_scene_label}.')
        if not sentences and top_regions and top_region_score >= 0.24:
            sentences.append('Likely semantic regions include ' + ', '.join(top_regions[:4]) + '.')
        if not sentences and top_objects and top_object_score >= 0.24:
            sentences.append('Top semantic object hypotheses: ' + ', '.join(top_objects[:4]) + '.')
        if not sentences and top_scene_label and top_scene_score >= 0.23:
            sentences.append('Top semantic scene hypothesis: ' + top_scene_label + '.')
        return ' '.join(sentences).strip()

    def _classify_regions(self, image, candidate_regions: list[dict[str, Any]], torch_module) -> list[SemanticRegionHypothesis]:
        width, height = image.size
        region_summaries: list[SemanticRegionHypothesis] = []
        full_area = max(1, width * height)
        for region in candidate_regions[:16]:
            entity_id = str(region.get('entity_id') or '').strip()
            bbox = region.get('bbox')
            if not entity_id or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [max(0, int(value)) for value in bbox]
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            x2 = min(max(x2, x1 + 1), width)
            y2 = min(max(y2, y1 + 1), height)
            area_ratio = ((x2 - x1) * (y2 - y1)) / full_area
            if area_ratio >= 0.72:
                continue
            crop = image.crop((x1, y1, x2, y2))
            ranked = self._rank_image_text(crop, OBJECT_LABELS, 'region', prompt_templates=OBJECT_PROMPT_TEMPLATES, torch_module=torch_module)
            if not ranked:
                continue
            top = ranked[0]
            next_best = ranked[1].score if len(ranked) > 1 else 0.0
            if top.score < 0.18 or (top.score - next_best) < 0.015:
                continue
            region_summaries.append(
                SemanticRegionHypothesis(
                    entity_id=entity_id,
                    label=top.label,
                    score=round(top.score, 4),
                    candidates=ranked[:3],
                )
            )
        region_summaries.sort(key=lambda item: item.score, reverse=True)
        return region_summaries[:8]

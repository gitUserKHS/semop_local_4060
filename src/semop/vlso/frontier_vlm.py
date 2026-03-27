from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..hardware_profiles import detect_local_hardware, recommended_generation_tokens, should_force_4bit
from .vision_backbones import DEFAULT_VISION_MODEL_ROOT


@dataclass(frozen=True)
class FrontierVisionSpec:
    model_id: str
    family: str
    description: str
    path_tokens: tuple[str, ...]


FRONTIER_VISION_SPECS: dict[str, FrontierVisionSpec] = {
    'qwen2_5_vl': FrontierVisionSpec(
        model_id='Qwen/Qwen2.5-VL-3B-Instruct',
        family='qwen2_5_vl',
        description='Frontier open VLM for image/video reasoning and localization.',
        path_tokens=('qwen2.5-vl', 'qwen2_5_vl', '3b-instruct', '3b'),
    ),
    'florence2': FrontierVisionSpec(
        model_id='microsoft/Florence-2-base-ft',
        family='florence2',
        description='Unified promptable vision model for captioning and grounding.',
        path_tokens=('florence-2', 'florence2'),
    ),
    'molmo2': FrontierVisionSpec(
        model_id='allenai/Molmo-7B-D-0924',
        family='molmo',
        description='Frontier open multimodal model with strong scene understanding.',
        path_tokens=('molmo-7b', 'molmo_7b', 'molmo'),
    ),
}


@dataclass
class FrontierVisualSummary:
    backend: str
    backend_ready: bool
    semantic_level: str
    answer_text: str
    family: str = ''
    model_id: str = ''
    notes: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VisualSceneAdjudication:
    preferred_answer: str
    stack_level: str
    confidence: float
    used_frontier: bool
    notes: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class VisualSceneAdjudicator:
    def adjudicate(self, query: str, frontier_summary: dict[str, Any], semantic_summary: dict[str, Any], world: Any) -> VisualSceneAdjudication:
        frontier_ready = bool(frontier_summary.get('backend_ready')) if isinstance(frontier_summary, dict) else False
        frontier_answer = str(frontier_summary.get('answer_text') or '').strip() if isinstance(frontier_summary, dict) else ''
        semantic_caption = str(semantic_summary.get('caption') or '').strip() if isinstance(semantic_summary, dict) else ''
        semantic_level = str(semantic_summary.get('semantic_level') or '').strip() if isinstance(semantic_summary, dict) else ''
        region_labels = []
        if isinstance(semantic_summary, dict) and isinstance(semantic_summary.get('region_hypotheses'), list):
            region_labels = [str(item.get('label') or '') for item in semantic_summary.get('region_hypotheses', []) if isinstance(item, dict)]
        structure_strength = self._structure_strength(world)
        notes: list[str] = []
        if frontier_ready and frontier_answer:
            overlap = self._token_overlap(frontier_answer, ' '.join([semantic_caption] + region_labels))
            confidence = 0.62 + (0.16 if semantic_level in {'semantic_grounded', 'semantic_candidate'} else 0.0) + min(0.12, overlap * 0.2) + min(0.1, structure_strength * 0.1)
            if confidence >= 0.74:
                notes.append('frontier answer accepted after semantic/structural cross-check')
                return VisualSceneAdjudication(
                    preferred_answer=frontier_answer,
                    stack_level='frontier_adjudicated',
                    confidence=round(confidence, 4),
                    used_frontier=True,
                    notes=notes,
                )
            notes.append('frontier answer was available but confidence stayed below the adjudication threshold')
        if semantic_caption and semantic_level in {'semantic_grounded', 'semantic_candidate'}:
            notes.append('semantic OpenCLIP lane selected as the safest grounded scene answer')
            return VisualSceneAdjudication(
                preferred_answer=semantic_caption,
                stack_level='semantic_grounded',
                confidence=0.68 if semantic_level == 'semantic_grounded' else 0.58,
                used_frontier=False,
                notes=notes,
            )
        if frontier_ready and frontier_answer:
            notes.append('frontier answer downgraded to tentative use because no stronger grounded lane was available')
            return VisualSceneAdjudication(
                preferred_answer=frontier_answer,
                stack_level='frontier_tentative',
                confidence=0.56,
                used_frontier=True,
                notes=notes,
            )
        return VisualSceneAdjudication(
            preferred_answer='',
            stack_level='structural_only',
            confidence=0.0,
            used_frontier=False,
            notes=['no semantic lane passed adjudication; keep structural grounding only'],
        )

    @staticmethod
    def _token_overlap(left: str, right: str) -> float:
        left_tokens = {item.lower() for item in str(left).replace('.', ' ').replace(',', ' ').split() if len(item) >= 4}
        right_tokens = {item.lower() for item in str(right).replace('.', ' ').replace(',', ' ').split() if len(item) >= 4}
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    @staticmethod
    def _structure_strength(world: Any) -> float:
        entities = [item for item in getattr(world, 'entities', []) if getattr(item, 'modality', '') == 'vision']
        relations = list(getattr(world, 'relations', []))
        return min(1.0, (len(entities) / 12.0) + (len(relations) / 40.0))


class FrontierVisionAdapter:
    _MODEL_CACHE: dict[tuple[str, str], tuple[object, object, str]] = {}

    def __init__(self, preferred_model: str = 'auto', model_path: str | None = None, hardware_profile: str = 'auto') -> None:
        self.hardware = detect_local_hardware(hardware_profile)
        self.preferred_model = preferred_model
        self.model_path = model_path or self._resolve_model_path(preferred_model)
        self._device = 'cuda' if self.hardware.cuda_available else 'cpu'
        self._processor = None
        self._model = None
        self._family = ''
        self._load_error = ''

    def describe(self, image_path: str, query: str) -> FrontierVisualSummary:
        path = Path(str(image_path or '').strip())
        if not path.exists():
            return FrontierVisualSummary(
                backend='frontier_vlm',
                backend_ready=False,
                semantic_level='weak',
                answer_text='',
                notes=['frontier VLM could not find the image path'],
            )
        if not self.model_path or not Path(self.model_path).exists():
            return FrontierVisualSummary(
                backend='frontier_vlm',
                backend_ready=False,
                semantic_level='weak',
                answer_text='',
                notes=['no local frontier VLM checkpoint is available; using the semantic OpenCLIP lane instead'],
            )
        try:
            self._lazy_load()
            from PIL import Image
            image = Image.open(path).convert('RGB')
            answer_text = self._generate_description(image, query)
            if not answer_text:
                return FrontierVisualSummary(
                    backend='frontier_vlm',
                    backend_ready=False,
                    semantic_level='weak',
                    answer_text='',
                    family=self._family,
                    model_id=self.model_path,
                    notes=['frontier VLM loaded, but it did not produce a usable description'],
                )
            return FrontierVisualSummary(
                backend='frontier_vlm',
                backend_ready=True,
                semantic_level='frontier_vlm',
                answer_text=answer_text.strip(),
                family=self._family,
                model_id=self.model_path,
                notes=[f'local frontier VLM family: {self._family}', f'local frontier checkpoint: {self.model_path}'],
            )
        except Exception as exc:
            self._load_error = str(exc)
            return FrontierVisualSummary(
                backend='frontier_vlm',
                backend_ready=False,
                semantic_level='weak',
                answer_text='',
                family=self._family,
                model_id=self.model_path or '',
                notes=[f'frontier VLM failed: {exc}'],
            )

    def _lazy_load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(f'frontier VLM dependencies unavailable: {exc}') from exc

        family = self._infer_family_from_path(self.model_path)
        cache_key = (str(self.model_path), self._device)
        cached = self._MODEL_CACHE.get(cache_key)
        if cached is not None:
            self._processor, self._model, self._family = cached
            return

        model_kwargs: dict[str, Any] = {'device_map': 'auto', 'low_cpu_mem_usage': True, 'trust_remote_code': True, 'local_files_only': True}
        use_4bit = bool(should_force_4bit(self.hardware, True) and torch.cuda.is_available())
        if use_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except Exception:
                use_4bit = False
            else:
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                )
        elif torch.cuda.is_available():
            model_kwargs['torch_dtype'] = torch.float16

        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
        model = None
        loaders = []
        if family == 'florence2':
            loaders = [AutoModelForVision2Seq, AutoModelForImageTextToText, AutoModelForCausalLM]
        elif family in {'qwen2_5_vl', 'molmo'}:
            loaders = [AutoModelForVision2Seq, AutoModelForImageTextToText, AutoModelForCausalLM]
        else:
            loaders = [AutoModelForVision2Seq, AutoModelForImageTextToText, AutoModelForCausalLM]
        last_error = None
        for loader in loaders:
            try:
                model = loader.from_pretrained(self.model_path, **model_kwargs)
                break
            except Exception as exc:
                last_error = exc
        if model is None:
            raise RuntimeError(f'could not load frontier VLM from {self.model_path}: {last_error}')
        self._processor = processor
        self._model = model
        self._family = family or getattr(config, 'model_type', '') or 'frontier_vlm'
        self._MODEL_CACHE[cache_key] = (self._processor, self._model, self._family)

    def _generate_description(self, image, query: str) -> str:
        family = self._family
        if family == 'florence2':
            return self._generate_florence2(image, query)
        if family == 'molmo':
            return self._generate_molmo2(image, query)
        return self._generate_chat_vlm(image, query)

    def _generate_florence2(self, image, query: str) -> str:
        task_prompt = '<MORE_DETAILED_CAPTION>' if self._looks_like_scene_description(query) else query
        inputs = self._processor(text=task_prompt, images=image, return_tensors='pt')
        inputs = self._to_model_device(inputs)
        generated = self._model.generate(**inputs, max_new_tokens=min(256, recommended_generation_tokens(self.hardware, 256)))
        decoded = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        if hasattr(self._processor, 'post_process_generation'):
            try:
                post = self._processor.post_process_generation(decoded, task=task_prompt, image_size=(image.width, image.height))
                if isinstance(post, dict):
                    detailed = post.get(task_prompt)
                    if isinstance(detailed, str) and detailed.strip():
                        return detailed.strip()
            except Exception:
                pass
        return decoded.strip()

    def _generate_chat_vlm(self, image, query: str) -> str:
        prompt = query.strip() or 'Describe this image.'
        if hasattr(self._processor, 'apply_chat_template'):
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': image},
                        {'type': 'text', 'text': prompt},
                    ],
                }
            ]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=[image], padding=True, return_tensors='pt')
        else:
            inputs = self._processor(text=prompt, images=image, return_tensors='pt')
        inputs = self._to_model_device(inputs)
        generated = self._model.generate(**inputs, max_new_tokens=min(256, recommended_generation_tokens(self.hardware, 256)))
        input_ids = inputs.get('input_ids')
        if input_ids is not None and generated.shape[1] > input_ids.shape[1]:
            generated = generated[:, input_ids.shape[1]:]
        return self._processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def _generate_molmo2(self, image, query: str) -> str:
        prompt = query.strip() or 'Describe this image.'
        if hasattr(self._processor, 'process') and hasattr(self._model, 'generate_from_batch'):
            from transformers import GenerationConfig
            inputs = self._processor.process(images=[image], text=prompt)
            inputs = {key: value.to(self._model.device).unsqueeze(0) if hasattr(value, 'to') else value for key, value in inputs.items()}
            output = self._model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=min(256, recommended_generation_tokens(self.hardware, 256)), stop_strings='<|endoftext|>'),
                tokenizer=self._processor.tokenizer,
            )
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            return self._processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return self._generate_chat_vlm(image, prompt)

    @staticmethod
    def _looks_like_scene_description(query: str) -> bool:
        lowered = str(query or '').lower()
        return any(token in lowered for token in ('describe', '??', 'scene', 'photo', 'image', 'picture', 'screenshot'))

    def _to_model_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        device = getattr(self._model, 'device', None)
        if device is None:
            return inputs
        output: dict[str, Any] = {}
        for key, value in inputs.items():
            output[key] = value.to(device) if hasattr(value, 'to') else value
        return output

    @staticmethod
    def _infer_family_from_path(path_value: str | None) -> str:
        normalized = str(path_value or '').lower()
        for spec in FRONTIER_VISION_SPECS.values():
            if any(token in normalized for token in spec.path_tokens):
                return spec.family
        return ''

    @staticmethod
    def _resolve_model_path(preferred_model: str = 'auto') -> str | None:
        env_path = str(os.getenv('SEMOP_FRONTIER_VLM_PATH', '')).strip()
        if env_path and Path(env_path).exists():
            return env_path
        roots = [DEFAULT_VISION_MODEL_ROOT / 'frontier', DEFAULT_VISION_MODEL_ROOT]
        preferred = str(preferred_model or 'auto').strip().lower()
        spec_order: list[FrontierVisionSpec] = []
        if preferred and preferred != 'auto':
            for spec in FRONTIER_VISION_SPECS.values():
                if preferred in {spec.family, spec.model_id.lower()}:
                    spec_order.append(spec)
        if not spec_order:
            spec_order = [
                FRONTIER_VISION_SPECS['qwen2_5_vl'],
                FRONTIER_VISION_SPECS['molmo2'],
                FRONTIER_VISION_SPECS['florence2'],
            ]
        for root in roots:
            if not root.exists():
                continue
            candidates = [item.parent for item in root.rglob('config.json')]
            for spec in spec_order:
                for candidate in sorted(set(candidates), key=lambda item: str(item)):
                    lowered = str(candidate).lower()
                    if any(token in lowered for token in spec.path_tokens):
                        return str(candidate)
        return None

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict

from .hardware_profiles import detect_local_hardware, recommended_generation_tokens, should_force_4bit
from .prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_TEMPLATE, JSON_REPAIR_SYSTEM_PROMPT, JSON_REPAIR_USER_TEMPLATE


@dataclass
class LocalLLMConfig:
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    max_new_tokens: int = 700
    temperature: float = 0.1
    use_4bit: bool = True
    repair_attempts: int = 2
    hardware_profile: str = 'auto'


class LocalTransformersExtractor:
    def __init__(self, config: LocalLLMConfig):
        self.config = config
        self._loaded = False
        self._tokenizer = None
        self._model = None
        self._hardware_profile = detect_local_hardware(config.hardware_profile)

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("llm 모드를 사용하려면 torch와 transformers가 설치되어 있어야 합니다.") from exc

        model_kwargs: Dict[str, Any] = {"device_map": "auto", "low_cpu_mem_usage": True}
        use_4bit = bool(should_force_4bit(self._hardware_profile, self.config.use_4bit) and torch.cuda.is_available())
        if use_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError:
                use_4bit = False
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

        if not use_4bit and torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            **model_kwargs,
        )
        self._loaded = True

    @staticmethod
    def _extract_json_blob(text: str) -> str:
        fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
        if fence_match:
            return fence_match.group(1)

        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last == -1 or first >= last:
            raise ValueError("모델 출력에서 JSON 객체를 찾지 못했습니다.")
        return text[first : last + 1]

    @staticmethod
    def _heuristic_repair(text: str) -> str:
        repaired = text.strip()
        repaired = re.sub(r"```(?:json)?", "", repaired)
        repaired = re.sub(r"(^|[{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', repaired)
        repaired = re.sub(r":\s*'([^'\\]*(?:\\.[^'\\]*)*)'", lambda m: ': "' + m.group(1).replace('"', '\\"') + '"', repaired)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(r"\bNone\b", "null", repaired)
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        return repaired

    @staticmethod
    def _normalize_payload(data: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(data)
        payload.setdefault("intent", "generic_reasoning")
        payload.setdefault("entities", [])
        payload.setdefault("relations", [])
        payload.setdefault("constraints", [])
        payload.setdefault("scripts", [])
        payload.setdefault("candidate_actions", [])
        payload.setdefault("missing_knowledge", [])

        if not isinstance(payload["intent"], str):
            raise TypeError("intent must be a string")
        for key in ["entities", "relations", "constraints", "scripts", "candidate_actions", "missing_knowledge"]:
            if not isinstance(payload[key], list):
                raise TypeError(f"{key} must be a list")
        return payload

    def _generate_text(self, system_prompt: str, user_prompt: str) -> str:
        self._lazy_load()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=recommended_generation_tokens(self._hardware_profile, self.config.max_new_tokens),
            temperature=self.config.temperature,
            do_sample=self.config.temperature > 0,
        )
        return self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def _parse_with_repair(self, decoded: str) -> Dict[str, Any]:
        errors: list[str] = []
        candidates = [decoded]
        try:
            candidates.append(self._extract_json_blob(decoded))
        except Exception as exc:
            errors.append(str(exc))

        checked: set[str] = set()
        for candidate in candidates:
            for variant in [candidate, self._heuristic_repair(candidate)]:
                if variant in checked:
                    continue
                checked.add(variant)
                try:
                    return self._normalize_payload(json.loads(variant))
                except Exception as exc:
                    errors.append(str(exc))

        if self._loaded:
            raw_text = decoded[:4000]
            for _ in range(self.config.repair_attempts):
                repaired = self._generate_text(
                    JSON_REPAIR_SYSTEM_PROMPT,
                    JSON_REPAIR_USER_TEMPLATE.format(error=" | ".join(errors[-3:]) or "parse failed", raw_text=raw_text),
                )
                try:
                    candidate = self._extract_json_blob(repaired)
                except Exception:
                    candidate = repaired
                try:
                    return self._normalize_payload(json.loads(self._heuristic_repair(candidate)))
                except Exception as exc:
                    errors.append(str(exc))
                    raw_text = repaired[:4000]

        raise RuntimeError("LLM JSON self-repair failed: " + " | ".join(errors[-4:]))

    def extract(self, query: str) -> Dict[str, Any]:
        decoded = self._generate_text(EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_TEMPLATE.format(query=query))
        return self._parse_with_repair(decoded)

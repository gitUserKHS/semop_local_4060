from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..llm_client import LocalLLMConfig
from .types import SharedWorldModel


@dataclass
class VLSOAnswer:
    answer_text: str
    answer_mode: str
    evidence: list[str]
    warnings: list[str]

    def model_dump(self) -> dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "answer_mode": self.answer_mode,
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
        }


class LocalTextGenerator:
    def __init__(self, config: LocalLLMConfig | None = None) -> None:
        self.config = config or LocalLLMConfig()
        self._loaded = False
        self._tokenizer = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("llm answer mode requires torch and transformers.") from exc

        model_kwargs: dict[str, Any] = {"device_map": "auto"}
        use_4bit = bool(self.config.use_4bit and torch.cuda.is_available())
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

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._lazy_load()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=min(512, self.config.max_new_tokens),
            temperature=self.config.temperature,
            do_sample=self.config.temperature > 0,
        )
        return self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class VLSOQuestionAnswerer:
    SYSTEM_PROMPT = (
        "You answer image-grounded questions from a structured world model. "
        "Use only the provided entities, relations, operators, constraints, and audit trace. "
        "If evidence is weak, say so directly and avoid inventing unseen details."
    )

    def __init__(self, generator: LocalTextGenerator | None = None) -> None:
        self.generator = generator

    def answer(self, query: str, world: SharedWorldModel, answer_mode: str = "structured") -> VLSOAnswer:
        evidence = self._evidence_lines(world)
        if answer_mode == "llm":
            if self.generator is None:
                raise RuntimeError("llm answer mode requested but no text generator is configured.")
            answer_text = self.generator.generate(
                self.SYSTEM_PROMPT,
                self._llm_user_prompt(query, world, evidence),
            )
            return VLSOAnswer(
                answer_text=answer_text,
                answer_mode="llm",
                evidence=evidence,
                warnings=list(world.warnings),
            )
        return VLSOAnswer(
            answer_text=self._structured_answer(query, world, evidence),
            answer_mode="structured",
            evidence=evidence,
            warnings=list(world.warnings),
        )

    def _structured_answer(self, query: str, world: SharedWorldModel, evidence: list[str]) -> str:
        lowered = query.lower()
        operator_names = {item.name for item in world.operators}
        constraint_set = set(world.constraints)
        entity_labels = [item.label for item in world.entities if item.modality == "vision"]
        entity_ids = [item.id for item in world.entities if item.modality == "vision"]
        shape_mentions = [item.label for item in world.entities if item.modality == "vision" and item.entity_type == "shape"]
        state_lines = [f"{item.source} is {item.target.split(':')[-1]}" for item in world.relations if item.relation == "STATE"]
        spatial_lines = [
            f"{item.source} {item.relation.lower()} {item.target}"
            for item in world.relations
            if item.relation in {"LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "CONTAINS", "PART_OF", "INTERSECTS", "PARALLEL"}
        ]

        if self._looks_like_object_inventory_question(lowered):
            if entity_labels:
                return "Visible entities in the current world model: " + ", ".join(entity_labels[:8])
            return self._weak_evidence_answer()

        if "shape" in lowered:
            if shape_mentions:
                return "Visible shape hypotheses: " + ", ".join(shape_mentions[:8])
            return self._weak_evidence_answer()

        if "state" in lowered or lowered.startswith("is ") or lowered.startswith("are "):
            matched_entities = self._query_entity_matches(query, entity_ids, entity_labels)
            if matched_entities and state_lines:
                relevant = [line for line in state_lines if any(name.lower() in line.lower() for name in matched_entities)]
                if relevant:
                    return "Observed state evidence: " + " | ".join(relevant[:4])

        if "where" in lowered or "relation" in lowered or "position" in lowered:
            if spatial_lines:
                return "Key grounded relations: " + " | ".join(spatial_lines[:5])

        if any(token in lowered for token in ["open", "access", "inside", "interior"]):
            if "ACCESS_OPENING_CANDIDATE" in operator_names or "EDGE_OPENING" in operator_names or "opening_candidate_detected" in constraint_set:
                return "Likely access path: use the opening candidate near the boundary or top band before inserting or reaching inside."
            if "HAS_INTERIOR" in operator_names or "bag_like_container_detected" in constraint_set:
                return "The object appears to behave like a container. Look for a top or edge opening before interacting with the interior."

        if any(token in lowered for token in ["strap", "carry", "grasp", "handle"]):
            if "STRAP_LIKE_PART" in operator_names or "GRASPABLE_PART" in operator_names:
                return "A strap-like or graspable part is present. That is the safest visible handle candidate."

        if lowered.startswith("can ") or lowered.startswith("is ") or lowered.startswith("are "):
            matched_entities = self._query_entity_matches(query, entity_ids, entity_labels)
            if matched_entities:
                return "Grounded evidence mentions: " + ", ".join(matched_entities[:4])

        if evidence:
            return "Best grounded answer from the current world model: " + evidence[0]
        return self._weak_evidence_answer()

    def _evidence_lines(self, world: SharedWorldModel) -> list[str]:
        lines: list[str] = []
        for relation in world.relations[:16]:
            lines.append(f"{relation.source} {relation.relation} {relation.target}")
        for operator in world.operators[:12]:
            lines.append(f"operator:{operator.name}")
        for constraint in world.constraints[:8]:
            lines.append(f"constraint:{constraint}")
        return lines

    def _llm_user_prompt(self, query: str, world: SharedWorldModel, evidence: list[str]) -> str:
        payload = {
            "query": query,
            "entities": [item.model_dump() for item in world.entities[:32]],
            "relations": [item.model_dump() for item in world.relations[:48]],
            "operators": [item.model_dump() for item in world.operators[:24]],
            "constraints": list(world.constraints),
            "inferred_steps": list(world.inferred_steps),
            "warnings": list(world.warnings),
            "evidence": evidence,
        }
        return (
            "Answer the user question from this structured world model.\n"
            "Keep the answer concise and grounded.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    @staticmethod
    def _query_entity_matches(query: str, entity_ids: list[str], entity_labels: list[str]) -> list[str]:
        query_terms = {item for item in re.findall(r"[A-Za-z_][A-Za-z0-9_\-]+", query.lower()) if len(item) > 2}
        matches: list[str] = []
        for candidate in list(entity_ids) + list(entity_labels):
            lowered = candidate.lower()
            if any(term in lowered or lowered in term for term in query_terms):
                matches.append(candidate)
        ordered: list[str] = []
        seen: set[str] = set()
        for item in matches:
            if item not in seen:
                ordered.append(item)
                seen.add(item)
        return ordered

    @staticmethod
    def _looks_like_object_inventory_question(lowered: str) -> bool:
        triggers = [
            "what is visible",
            "what objects",
            "which objects",
            "what can you see",
            "visible here",
            "what do you see",
        ]
        return any(token in lowered for token in triggers)

    @staticmethod
    def _weak_evidence_answer() -> str:
        return "The current visual evidence is weak. I need a clearer image, detector output, or a more specific question."

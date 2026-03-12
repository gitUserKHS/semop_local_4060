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
        structural_bindings = world.metadata.get('structural_operator_bindings', []) if isinstance(world.metadata, dict) else []
        hidden_premises = world.metadata.get('hidden_premises', []) if isinstance(world.metadata, dict) else []
        goal_checks = world.metadata.get('goal_preservation_checks', []) if isinstance(world.metadata, dict) else []
        functors = world.metadata.get('functor_hypotheses', []) if isinstance(world.metadata, dict) else []
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

        if any(token in lowered for token in ["geometric", "geometry", "parallel", "perpendicular", "equal length", "shape", "triangle", "rectangle", "square", "quadrilateral", "parallelogram"]):
            geometry_lines = [
                f"{item.source} {item.relation.lower()} {item.target}"
                for item in world.relations
                if item.relation in {"PARALLEL", "PERPENDICULAR", "EQUAL_LENGTH"}
            ]
            explicit_shape_query = any(token in lowered for token in ["what shape", "which shape", "triangle", "rectangle", "square", "quadrilateral", "parallelogram"])
            if explicit_shape_query and shape_mentions:
                return "Visible shape hypotheses: " + ", ".join(shape_mentions[:8])
            if geometry_lines:
                return "Key geometry relations: " + " | ".join(geometry_lines[:6])
            if shape_mentions:
                return "Visible shape hypotheses: " + ", ".join(shape_mentions[:8])
            return self._weak_evidence_answer()

        if self._looks_like_object_inventory_question(lowered):
            container_entities = []
            part_entities = []
            structural_containers = [str(item.get('subject', '')) for item in structural_bindings if isinstance(item, dict) and item.get('operator_name') in {'CONTAINER_BODY_OPERATOR', 'MANIPULABLE_CONTAINER_OPERATOR'}]
            structural_access = [str(item.get('subject', '')) for item in structural_bindings if isinstance(item, dict) and item.get('operator_name') in {'ACCESS_PORT_OPERATOR', 'ACCESS_CONTROL_OPERATOR', 'ATTACHED_GRASP_OPERATOR'}]
            for entity in world.entities:
                if entity.modality != "vision":
                    continue
                labels = entity.attributes.get("concept_labels") or []
                text_label = entity.label or entity.id
                upper_labels = {str(item).upper() for item in labels} if isinstance(labels, list) else set()
                is_container = any(any(token in label for token in ["CONTAINER", "BOX", "DRAWER", "CABINET", "BOTTLE", "JAR", "BIN", "POUCH", "SUITCASE"]) for label in upper_labels)
                if is_container:
                    container_entities.append(text_label)
                elif upper_labels:
                    part_entities.append(text_label)
            if structural_containers or structural_access or container_entities or part_entities:
                parts = []
                if structural_containers:
                    parts.append('structural containers: ' + ', '.join(self._describe_entity(world, item) for item in structural_containers[:5]))
                if structural_access:
                    parts.append('structural access/grasp parts: ' + ', '.join(self._describe_entity(world, item) for item in structural_access[:6]))
                if container_entities:
                    parts.append("containers: " + ", ".join(container_entities[:5]))
                if part_entities:
                    parts.append("parts/openings: " + ", ".join(part_entities[:6]))
                return "Visible entities in the current world model: " + " | ".join(parts)
            if entity_labels:
                return "Visible entities in the current world model: " + ", ".join(entity_labels[:8])
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
            opening_candidates = self._opening_candidates(world)
            handle_candidates = self._handle_candidates(world)
            structural_openings = [str(item.get('subject', '')) for item in structural_bindings if isinstance(item, dict) and item.get('operator_name') == 'ACCESS_PORT_OPERATOR']
            structural_controls = [str(item.get('subject', '')) for item in structural_bindings if isinstance(item, dict) and item.get('operator_name') in {'ACCESS_CONTROL_OPERATOR', 'ATTACHED_GRASP_OPERATOR'}]
            structural_containers = [str(item.get('subject', '')) for item in structural_bindings if isinstance(item, dict) and item.get('operator_name') == 'CONTAINER_BODY_OPERATOR']
            if structural_openings and structural_controls:
                container_text = self._describe_entity(world, structural_containers[0]) if structural_containers else 'the container body'
                chosen_control = next((item for item in structural_controls if item not in structural_openings), structural_controls[0])
                return 'Structural access path: use ' + self._describe_entity(world, chosen_control) + ' to reach opening region ' + self._describe_entity(world, structural_openings[0]) + ' on ' + container_text + '.'
            if structural_openings:
                return 'Most likely structural opening region: ' + ', '.join(self._describe_entity(world, item) for item in structural_openings[:2])
            if opening_candidates and handle_candidates:
                return "Most likely access route: interact with " + self._describe_entity(world, handle_candidates[0]) + " to reach " + self._describe_entity(world, opening_candidates[0]) + "."
            if handle_candidates and structural_containers:
                return "Likely access-related evidence: " + self._describe_entity(world, handle_candidates[0]) + " is attached to " + self._describe_entity(world, structural_containers[0]) + ", so an opening or access control is likely present nearby."
            if opening_candidates:
                return "Most likely opening-related part: " + ", ".join(self._describe_entity(world, item) for item in opening_candidates[:2])
            if "ACCESS_OPENING_CANDIDATE" in operator_names or "EDGE_OPENING" in operator_names or "opening_candidate_detected" in constraint_set:
                return "Likely access path: use the opening candidate near the boundary or top band before inserting or reaching inside."
            if "HAS_INTERIOR" in operator_names or "container_like_object_detected" in constraint_set:
                return "The object appears to behave like a container. Look for a lid, zipper, hinge, cap, or edge opening before interacting with the interior."

        if any(token in lowered for token in ["strap", "carry", "grasp", "handle"]):
            if "STRAP_LIKE_PART" in operator_names or "GRASPABLE_PART" in operator_names:
                return "A strap-like or graspable part is present. That is the safest visible handle candidate."

        if lowered.startswith("can ") or lowered.startswith("is ") or lowered.startswith("are "):
            matched_entities = self._query_entity_matches(query, entity_ids, entity_labels)
            if matched_entities:
                return "Grounded evidence mentions: " + ", ".join(matched_entities[:4])

        if goal_checks and any(item.get('status') == 'risk_high' for item in goal_checks if isinstance(item, dict)):
            top = next(item for item in goal_checks if isinstance(item, dict) and item.get('status') == 'risk_high')
            if top.get('hidden_goal') == 'retrieve_item_from_cabinet_goal':
                return 'Hidden-goal risk detected: open the cabinet access-control part before retrieving the item inside.'
            if top.get('hidden_goal') == 'retrieve_item_from_box_goal':
                return 'Hidden-goal risk detected: open the box or lid before retrieving the item inside.'
            if top.get('hidden_goal') == 'retrieve_item_from_bin_goal':
                return 'Hidden-goal risk detected: open or lift the bin cover before retrieving the item inside.'
            if top.get('hidden_goal') == 'retrieve_item_from_pouch_goal':
                return 'Hidden-goal risk detected: open the pouch closure before retrieving the item inside.'
            if top.get('hidden_goal') == 'retrieve_item_from_suitcase_goal':
                return 'Hidden-goal risk detected: open the suitcase closure before retrieving the item inside.'
            if top.get('hidden_goal') == 'pour_from_bottle_goal':
                return 'Hidden-goal risk detected: remove or open the cap or lid before pouring from the container.'
            return 'Hidden-goal risk detected: ' + top.get('rationale', 'The proposed action may fail the real goal.')

        if hidden_premises and any(token in lowered for token in ['why', 'should', 'walk', 'go', 'insert', 'open']):
            return 'Relevant hidden premises: ' + ' | '.join(str(item) for item in hidden_premises[:3])

        if functors and any(token in lowered for token in ['why', 'relation', 'structure']):
            names = [str(item.get('name')) for item in functors if isinstance(item, dict)]
            return 'Cross-modal operator alignments: ' + ', '.join(names[:4])

        if evidence:
            return "Best grounded answer from the current world model: " + evidence[0]
        if constraint_set & {'container_like_object_detected', 'opening_candidate_detected', 'handle_like_part_detected'}:
            return 'There is weak but usable structural evidence of a container, handle, or opening-related part. The scene likely contains an access path, but confidence is limited.'
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
        focus_hints: dict[str, Any] = {}
        lowered = query.lower()
        if any(token in lowered for token in ["open", "access", "inside", "interior"]):
            focus_hints["preferred_opening_candidates"] = self._opening_candidates(world)
            focus_hints["instruction"] = "Prioritize actual opening or zipper candidates. Ignore weak border fragments and avoid listing unrelated parts unless evidence is strong."
        payload = {
            "query": query,
            "entities": [item.model_dump() for item in world.entities[:32]],
            "relations": [item.model_dump() for item in world.relations[:48]],
            "operators": [item.model_dump() for item in world.operators[:24]],
            "constraints": list(world.constraints),
            "inferred_steps": list(world.inferred_steps),
            "warnings": list(world.warnings),
            "evidence": evidence,
            "focus_hints": focus_hints,
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

    @staticmethod
    def _describe_entity(world: SharedWorldModel, entity_id: str) -> str:
        for entity in world.entities:
            if entity.id != entity_id:
                continue
            labels = entity.attributes.get('concept_labels') or []
            upper = {str(item).upper() for item in labels} if isinstance(labels, list) else set()
            if 'ZIPPER_LIKE_PART' in upper:
                return f'zipper-like opening control ({entity_id})'
            if 'HANDLE_LIKE_PART' in upper or 'HANDLE_CANDIDATE' in upper or 'KNOB_LIKE_PART' in upper:
                return f'handle-like grasp part ({entity_id})'
            if 'ACCESS_OPENING_CANDIDATE' in upper or 'EDGE_OPENING' in upper:
                return f'opening region ({entity_id})'
            if 'STRAP_LIKE_PART' in upper:
                return f'strap-like grasp part ({entity_id})'
            if 'GRASPABLE_PART' in upper:
                return f'graspable part ({entity_id})'
            if 'HAS_INTERIOR' in upper or 'STRUCTURAL_CONTAINER_CANDIDATE' in upper or 'BAG_LIKE_CONTAINER' in upper:
                return f'container body ({entity_id})'
            return entity.label or entity_id
        return entity_id

    @staticmethod
    def _handle_candidates(world: SharedWorldModel) -> list[str]:
        ranked: list[tuple[int, str]] = []
        for entity in world.entities:
            if entity.modality != "vision":
                continue
            labels = entity.attributes.get("concept_labels") or []
            if not isinstance(labels, list):
                continue
            upper_labels = {str(item).upper() for item in labels}
            if not ({"HANDLE_LIKE_PART", "GRASPABLE_PART", "STRAP_LIKE_PART", "KNOB_LIKE_PART", "TOOL_GRIP_PART"} & upper_labels):
                continue
            score = 0
            if "HANDLE_LIKE_PART" in upper_labels:
                score += 3
            if "TOOL_GRIP_PART" in upper_labels:
                score += 2
            if "KNOB_LIKE_PART" in upper_labels:
                score += 2
            if "GRASPABLE_PART" in upper_labels:
                score += 1
            ranked.append((score, entity.id))
        ranked.sort(key=lambda item: item[0], reverse=True)
        seen: set[str] = set()
        output: list[str] = []
        for score, entity_id in ranked:
            if score <= 0 or entity_id in seen:
                continue
            output.append(entity_id)
            seen.add(entity_id)
        return output

    @staticmethod
    def _opening_candidates(world: SharedWorldModel) -> list[str]:
        ranked: list[tuple[int, str]] = []
        for entity in world.entities:
            if entity.modality != "vision":
                continue
            labels = entity.attributes.get("concept_labels") or []
            if not isinstance(labels, list):
                continue
            upper_labels = {str(item).upper() for item in labels}
            if "ACCESS_OPENING_CANDIDATE" not in upper_labels and "ZIPPER_LIKE_PART" not in upper_labels and "EDGE_OPENING" not in upper_labels:
                continue
            score = 0
            if "ACCESS_OPENING_CANDIDATE" in upper_labels:
                score += 3
            if "ZIPPER_LIKE_PART" in upper_labels:
                score += 2
            if "EDGE_OPENING" in upper_labels:
                score += 2
            if "STRAP_LIKE_PART" in upper_labels and "ACCESS_OPENING_CANDIDATE" not in upper_labels:
                score -= 2
            bbox = entity.attributes.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                area = max(1, (x2 - x1) * (y2 - y1))
                if (x1 <= 2 or y1 <= 2) and area <= 600:
                    continue
                if x1 <= 2 or y1 <= 2:
                    score -= 4
                if area <= 600:
                    score -= 2
            ranked.append((score, entity.id))
        ranked.sort(key=lambda item: item[0], reverse=True)
        output: list[str] = []
        seen: set[str] = set()
        for score, entity_id in ranked:
            if score <= 0 or entity_id in seen:
                continue
            output.append(entity_id)
            seen.add(entity_id)
        return output

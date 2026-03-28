from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..hardware_profiles import detect_local_hardware, recommended_generation_tokens, should_force_4bit
from ..llm_client import LocalLLMConfig
from .question_understanding import VisualQuestionUnderstandingEngine
from .types import SharedWorldModel
from .world_model_narrator import WorldModelNarrator


@dataclass
class VLSOAnswer:
    answer_text: str
    answer_mode: str
    evidence: list[str]
    warnings: list[str]
    scene_semantic_level: str = ''
    prompt_understanding: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "answer_mode": self.answer_mode,
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "scene_semantic_level": self.scene_semantic_level,
            "prompt_understanding": dict(self.prompt_understanding),
        }


class LocalTextGenerator:
    def __init__(self, config: LocalLLMConfig | None = None) -> None:
        self.config = config or LocalLLMConfig()
        self._loaded = False
        self._tokenizer = None
        self._model = None
        self._hardware_profile = detect_local_hardware(self.config.hardware_profile)

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("llm answer mode requires torch and transformers.") from exc

        model_kwargs: dict[str, Any] = {"device_map": "auto", "low_cpu_mem_usage": True}
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
            max_new_tokens=min(512, recommended_generation_tokens(self._hardware_profile, self.config.max_new_tokens)),
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
        self.narrator = WorldModelNarrator()
        self.question_understanding = VisualQuestionUnderstandingEngine()

    def narrate_world_model(self, query: str, world: SharedWorldModel) -> str:
        return self.narrator.describe(query, world)

    def answer(self, query: str, world: SharedWorldModel, answer_mode: str = "structured") -> VLSOAnswer:
        evidence = self._evidence_lines(world)
        semantic_level = self._scene_semantic_level(query, world)
        prompt_understanding = world.metadata.get('prompt_understanding', {}) if isinstance(world.metadata, dict) and isinstance(world.metadata.get('prompt_understanding'), dict) else {}
        if answer_mode == "llm":
            if self.generator is None:
                raise RuntimeError("llm answer mode requested but no text generator is configured.")
            answer_text = self.generator.generate(
                self.SYSTEM_PROMPT,
                self._llm_user_prompt(query, world, evidence),
            )
            warnings = list(world.warnings)
            if semantic_level == 'structural_only':
                warnings.append('Scene semantic grounding is still structural-only for this image.')
            return VLSOAnswer(
                answer_text=answer_text,
                answer_mode="llm",
                evidence=evidence,
                warnings=warnings,
                scene_semantic_level=semantic_level,
                prompt_understanding=dict(prompt_understanding),
            )
        answer_text = self._structured_answer(query, world, evidence)
        warnings = list(world.warnings)
        if semantic_level == 'structural_only':
            warnings.append('Scene semantic grounding is still structural-only for this image.')
        return VLSOAnswer(
            answer_text=answer_text,
            answer_mode="structured",
            evidence=evidence,
            warnings=warnings,
            scene_semantic_level=semantic_level,
            prompt_understanding=dict(prompt_understanding),
        )

    def _structured_answer(self, query: str, world: SharedWorldModel, evidence: list[str]) -> str:
        lowered = query.lower()
        question_prediction = self.question_understanding.predict(query)
        predicted_intent = question_prediction.intent
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

        if predicted_intent == 'geometry' or any(token in lowered for token in ["geometric", "geometry", "parallel", "perpendicular", "equal length", "shape", "triangle", "rectangle", "square", "quadrilateral", "parallelogram"]):
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

        if predicted_intent == 'object_inventory' or self._looks_like_object_inventory_question(lowered):
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
            prefers_structural_inventory = any(
                token in lowered
                for token in [
                    "opening",
                    "openings",
                    "access",
                    "inside",
                    "interior",
                    "container",
                    "containers",
                    "part",
                    "parts",
                    "개구부",
                    "입구",
                    "내부",
                    "부품",
                ]
            )
            visual_scene_answer = self._scene_description_answer(query, world, spatial_lines)
            if not prefers_structural_inventory and visual_scene_answer != self._weak_evidence_answer():
                return visual_scene_answer
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
            if visual_scene_answer != self._weak_evidence_answer():
                return visual_scene_answer
            if entity_labels:
                return "Visible entities in the current world model: " + ", ".join(entity_labels[:8])
            if any(item.modality == 'vision' for item in world.entities):
                return self._structural_visual_answer(query, world)
            return self._weak_evidence_answer()

        if predicted_intent == 'scene_description' or self._looks_like_scene_description_question(lowered):
            return self._scene_description_answer(query, world, spatial_lines)

        if predicted_intent == 'state_check' or "state" in lowered or lowered.startswith("is ") or lowered.startswith("are "):
            matched_entities = self._query_entity_matches(query, entity_ids, entity_labels)
            if matched_entities and state_lines:
                relevant = [line for line in state_lines if any(name.lower() in line.lower() for name in matched_entities)]
                if relevant:
                    return "Observed state evidence: " + " | ".join(relevant[:4])

        if predicted_intent == 'spatial_relation' or "where" in lowered or "relation" in lowered or "position" in lowered:
            if spatial_lines:
                return "Key grounded relations: " + " | ".join(spatial_lines[:5])

        if predicted_intent == 'access_reasoning' or any(token in lowered for token in ["open", "access", "inside", "interior"]):
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

        if (predicted_intent in {'generic_visual', 'scene_description', 'object_inventory'} or self._looks_like_general_visual_question(lowered)) and any(item.modality == 'vision' for item in world.entities):
            return self._scene_description_answer(query, world, spatial_lines)

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
            "\uc774 \uc0ac\uc9c4\uc5d0\uc11c \ubb50\uac00 \ubcf4\uc5ec",
            "\ubb50\uac00 \ubcf4\uc5ec",
            "\ubb34\uc5c7\uc774 \ubcf4\uc5ec",
            "\ubcf4\uc774\ub294 \ubb3c\uccb4",
            "\uc5b4\ub5a4 \ubb3c\uccb4",
            "\ubb34\uc2a8 \ubb3c\uccb4",
        ]
        return any(token in lowered for token in triggers)

    @staticmethod
    def _looks_like_scene_description_question(lowered: str) -> bool:
        triggers = [
            "describe this image",
            "describe this photo",
            "describe this picture",
            "describe this screenshot",
            "describe the scene",
            "what is happening",
            "what's happening",
            "what is in this image",
            "what does this show",
            "explain this image",
            "scene description",
            "\uc124\uba85",
            "\uc0ac\uc9c4\uc744 \uc124\uba85",
            "\uc774\ubbf8\uc9c0\ub97c \uc124\uba85",
            "\ud654\uba74\uc744 \uc124\uba85",
            "\ubb34\uc2a8 \uc7a5\uba74",
            "\ubb34\uc2a8 \uc0c1\ud669",
        ]
        return any(token in lowered for token in triggers)

    @classmethod
    def _looks_like_general_visual_question(cls, lowered: str) -> bool:
        return cls._looks_like_scene_description_question(lowered) or cls._looks_like_object_inventory_question(lowered)

    @staticmethod
    def _is_korean_query(query: str) -> bool:
        return bool(re.search(r'[\uac00-\ud7a3]', str(query or '')))

    @staticmethod
    def _dedupe_labels(items: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = str(item or '').strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            ordered.append(normalized)
            seen.add(key)
        return ordered

    @classmethod
    def _translate_visual_label(cls, label: str, language: str) -> str:
        normalized = str(label or '').strip()
        if not normalized or language != 'ko':
            return normalized
        translations = {
            'video game screenshot': '게임 스크린샷',
            'first-person shooter game screenshot': '1인칭 슈팅 게임 화면',
            'combat video game scene': '전투 게임 장면',
            'third-person action game scene': '3인칭 액션 게임 장면',
            'urban street': '도시 거리',
            'market street': '시장 거리',
            'shopfront': '상점 앞',
            'street market': '노점 거리',
            'alley': '골목',
            'outdoor daytime scene': '야외 낮 장면',
            'warehouse aisle': '창고 통로',
            'indoor room': '실내 공간',
            'construction site': '공사 현장',
            'parking lot': '주차장',
            'close-up object photo': '사물 근접 사진',
            'product photo': '제품 사진',
            'bag or backpack photo': '가방 또는 백팩 사진',
            'street scene': '거리 장면',
            'person': '사람',
            'human character': '사람형 캐릭터',
            'female game character': '여성 게임 캐릭터',
            'soldier': '병사',
            'handgun': '권총',
            'pistol': '권총',
            'rifle': '소총',
            'gun held in first person view': '플레이어가 들고 있는 총',
            'weapon': '무기',
            'player hands': '플레이어 손',
            'hands': '손',
            'arms': '팔',
            'backpack': '백팩',
            'bag': '가방',
            'travel bag': '여행 가방',
            'suitcase': '여행용 가방',
            'market stall': '노점',
            'shop awning': '상점 차양',
            'cart': '수레',
            'street cart': '거리 수레',
            'building': '건물',
            'building facade': '건물 외벽',
            'signboard': '간판',
            'dome': '돔 지붕',
            'store counter': '매대',
            'kiosk': '가판대',
            'doorway': '출입구',
            'bicycle': '자전거',
            'car': '자동차',
            'game HUD': '게임 HUD',
            'mini-map overlay': '미니맵',
            'crosshair overlay': '조준선',
            'scoreboard overlay': '점수판',
            'timer overlay': '타이머',
            'ammo counter overlay': '탄약 표시',
            'kill feed overlay': '킬 피드',
            'chat overlay': '채팅창',
        }
        return translations.get(normalized, normalized)

    def _localized_visual_scene_answer(self, query: str, world: SharedWorldModel, semantic_summary: dict[str, Any]) -> str | None:
        if not self._is_korean_query(query):
            return None
        scene_hypotheses = semantic_summary.get('scene_hypotheses', []) if isinstance(semantic_summary, dict) and isinstance(semantic_summary.get('scene_hypotheses'), list) else []
        object_hypotheses = semantic_summary.get('object_hypotheses', []) if isinstance(semantic_summary, dict) and isinstance(semantic_summary.get('object_hypotheses'), list) else []
        overlay_hypotheses = semantic_summary.get('overlay_hypotheses', []) if isinstance(semantic_summary, dict) and isinstance(semantic_summary.get('overlay_hypotheses'), list) else []
        region_hypotheses = semantic_summary.get('region_hypotheses', []) if isinstance(semantic_summary, dict) and isinstance(semantic_summary.get('region_hypotheses'), list) else []
        scene_label = ''
        if scene_hypotheses and float(scene_hypotheses[0].get('score', 0.0) or 0.0) >= 0.22:
            scene_label = str(scene_hypotheses[0].get('label') or '').strip()
        semantic_entities, structural_entities = self._scene_description_profile(world)
        object_labels = self._dedupe_labels(
            list(semantic_entities)
            + [str(item.get('label') or '').strip() for item in region_hypotheses if isinstance(item, dict) and float(item.get('score', 0.0) or 0.0) >= 0.18]
            + [str(item.get('label') or '').strip() for item in object_hypotheses if isinstance(item, dict) and float(item.get('score', 0.0) or 0.0) >= 0.22]
        )
        overlay_labels = self._dedupe_labels(
            [str(item.get('label') or '').strip() for item in overlay_hypotheses if isinstance(item, dict) and float(item.get('score', 0.0) or 0.0) >= 0.22]
        )
        translated_scene = self._translate_visual_label(scene_label, 'ko')
        translated_objects = self._dedupe_labels([self._translate_visual_label(item, 'ko') for item in object_labels])
        translated_overlays = self._dedupe_labels([self._translate_visual_label(item, 'ko') for item in overlay_labels])
        scene_game_like = scene_label in {'video game screenshot', 'first-person shooter game screenshot', 'combat video game scene'}
        has_weapon = any(item in {'handgun', 'pistol', 'rifle', 'gun held in first person view', 'weapon'} for item in object_labels)
        has_person = any(item in {'person', 'human character', 'female game character', 'soldier'} for item in object_labels)
        if scene_game_like or (has_weapon and (has_person or translated_overlays)):
            sentences = ['1인칭 전투 게임 화면처럼 보입니다.']
            if has_weapon:
                sentences.append('화면 앞쪽에는 플레이어가 들고 있는 총이나 무기가 보입니다.')
            if has_person:
                sentences.append('앞쪽에는 사람형 게임 캐릭터가 적어도 한 명 보입니다.')
            extras = [item for item in translated_objects if item not in {'권총', '소총', '플레이어가 들고 있는 총', '무기', '사람', '사람형 캐릭터', '여성 게임 캐릭터', '병사'}]
            if extras:
                sentences.append('주변에는 ' + ', '.join(extras[:4]) + ' 같은 요소가 보입니다.')
            if translated_overlays:
                sentences.append('화면 UI로는 ' + ', '.join(translated_overlays[:4]) + ' 등이 보입니다.')
            return ' '.join(sentences)
        if translated_scene or translated_objects or translated_overlays:
            sentences: list[str] = []
            if translated_scene:
                sentences.append('이 이미지는 ' + translated_scene + '처럼 보입니다.')
            if translated_objects:
                sentences.append('보이는 요소로는 ' + ', '.join(translated_objects[:6]) + ' 등이 있습니다.')
            if translated_overlays:
                sentences.append('화면에는 ' + ', '.join(translated_overlays[:4]) + ' 같은 UI도 보입니다.')
            return ' '.join(sentences)
        if structural_entities or any(item.modality == 'vision' for item in world.entities):
            return self._structural_visual_answer(query, world, structural_entities=structural_entities)
        return None

    def _structural_visual_answer(
        self,
        query: str,
        world: SharedWorldModel,
        *,
        structural_entities: list[str] | None = None,
        semantic_caption: str = '',
    ) -> str:
        structural = list(structural_entities or [])
        if not structural:
            _, structural = self._scene_description_profile(world)
        vision_entities = [item for item in world.entities if item.modality == 'vision']
        if self._is_korean_query(query):
            if structural:
                answer = (
                    '현재 로컬 비전 스택은 이 이미지를 아직 사람처럼 의미적으로 해석하지는 못했고, '
                    + ', '.join(structural[:4])
                    + ' 같은 거친 구조 단서만 비교적 확실하게 잡았습니다.'
                )
            else:
                answer = (
                    f'현재 로컬 비전 스택은 보이는 영역 {len(vision_entities)}개와 관계 {len(world.relations)}개 정도를 잡았지만, '
                    '믿을 만한 의미 라벨까지는 복원하지 못했습니다.'
                )
            if semantic_caption:
                answer += ' 의미 가설은 있지만 아직 최종 답으로 쓰기엔 검증이 부족합니다.'
            if world.warnings:
                answer += ' 경고: ' + ' | '.join(str(item) for item in list(world.warnings)[:1])
            return answer
        if structural:
            answer = (
                'This image is not yet being semantically understood at a human level by the current local vision stack. '
                'Right now I can only ground coarse structural regions such as '
                + ', '.join(structural[:4])
                + '.'
            )
        else:
            answer = (
                f'The current local visual parser recovered {len(vision_entities)} visible regions and {len(world.relations)} grounded relations, '
                'but it did not recover trustworthy semantic object labels for this image. '
                'I can ground coarse structural evidence, but not a reliable natural-language scene description yet.'
            )
        if semantic_caption:
            answer += ' Semantic hypothesis exists, but it is not yet trustworthy enough to present as the final answer.'
        if world.warnings:
            answer += ' Warning: ' + ' | '.join(str(item) for item in list(world.warnings)[:1])
        return answer

    def _scene_description_answer(self, query: str, world: SharedWorldModel, spatial_lines: list[str]) -> str:
        narrated = self.narrate_world_model(query, world)
        if narrated:
            return narrated
        semantic_entities, structural_entities = self._scene_description_profile(world)
        if semantic_entities:
            summary = 'Visible scene summary: ' + ', '.join(semantic_entities[:6]) + '.'
            if spatial_lines:
                summary += ' Key grounded relations: ' + ' | '.join(spatial_lines[:3])
            return summary
        if structural_entities or any(item.modality == 'vision' for item in world.entities):
            return self._structural_visual_answer(query, world, structural_entities=structural_entities)
        return self._weak_evidence_answer()

    def _scene_description_profile(self, world: SharedWorldModel) -> tuple[list[str], list[str]]:
        semantic: list[str] = []
        structural: list[str] = []
        for entity in world.entities:
            if entity.modality != 'vision':
                continue
            semantic_label = str(entity.attributes.get('semantic_label') or '').strip()
            if semantic_label:
                semantic.append(semantic_label)
                continue
            label = str(getattr(entity, 'label', '') or '').strip()
            if self._is_semantic_scene_label(label):
                semantic.append(label)
                continue
            description = self._structural_scene_description(entity)
            if description:
                structural.append(description)
        return self._unique_descriptions(semantic), self._unique_descriptions(structural)

    @classmethod
    def _is_semantic_scene_label(cls, label: str) -> bool:
        normalized = str(label or '').strip()
        if not normalized:
            return False
        if cls._is_generic_visual_label(normalized):
            return False
        if '[' in normalized and ']' in normalized:
            return False
        lowered = normalized.lower()
        if lowered in {'shape', 'part', 'object', 'container', 'opening'}:
            return False
        return True

    @classmethod
    def _structural_scene_description(cls, entity: Any) -> str | None:
        labels = getattr(entity, 'attributes', {}).get('concept_labels') or []
        upper = {str(item).upper() for item in labels} if isinstance(labels, list) else set()
        if {'ACCESS_OPENING_CANDIDATE', 'ACCESS_CONTROL_PART', 'ACCESS_PORT_CANDIDATE', 'EDGE_OPENING'} & upper:
            return 'opening or access-control region'
        if {'HANDLE_CANDIDATE', 'HANDLE_LIKE_PART', 'GRASPABLE_PART', 'STRAP_LIKE_PART', 'KNOB_LIKE_PART', 'TOOL_GRIP_PART'} & upper:
            return 'handle or graspable region'
        if {'HAS_INTERIOR', 'STRUCTURAL_CONTAINER_CANDIDATE', 'MANIPULABLE_CONTAINER', 'BAG_LIKE_CONTAINER', 'DRAWER_LIKE_CONTAINER', 'BOTTLE_LIKE_CONTAINER', 'ACCESSIBLE_INTERIOR_PATH'} & upper:
            return 'container-like region'
        entity_type = str(getattr(entity, 'entity_type', '') or '').lower()
        if entity_type == 'opening':
            return 'opening region'
        if entity_type == 'container':
            return 'container-like region'
        if entity_type == 'person':
            return 'person-like region'
        if entity_type == 'vehicle':
            return 'vehicle-like region'
        return None

    def _scene_semantic_level(self, query: str, world: SharedWorldModel) -> str:
        visual_intent = self.question_understanding.predict(query).intent
        if visual_intent not in {'generic_visual', 'scene_description', 'object_inventory'} and not self._looks_like_general_visual_question(str(query or '').lower()):
            return ''
        adjudication = world.metadata.get('scene_adjudication', {}) if isinstance(world.metadata, dict) else {}
        adjudicated_level = str(adjudication.get('stack_level') or '').strip() if isinstance(adjudication, dict) else ''
        if adjudicated_level and adjudicated_level != 'structural_only':
            return adjudicated_level
        frontier_summary = world.metadata.get('frontier_scene_summary', {}) if isinstance(world.metadata, dict) else {}
        frontier_level = str(frontier_summary.get('semantic_level') or '').strip() if isinstance(frontier_summary, dict) else ''
        if frontier_level == 'frontier_vlm' and frontier_summary.get('backend_ready'):
            return 'frontier_vlm'
        semantic_summary = world.metadata.get('semantic_scene_summary', {}) if isinstance(world.metadata, dict) else {}
        summary_level = str(semantic_summary.get('semantic_level') or '').strip() if isinstance(semantic_summary, dict) else ''
        if summary_level:
            if summary_level == 'semantic_grounded':
                return 'semantic_grounded'
            if summary_level == 'semantic_candidate':
                return 'semantic_candidate'
        semantic_entities, structural_entities = self._scene_description_profile(world)
        if semantic_entities:
            return 'semantic_grounded'
        if structural_entities or any(item.modality == 'vision' for item in world.entities):
            return 'structural_only'
        return 'weak'

    @staticmethod
    def _is_generic_visual_label(label: str) -> bool:
        lowered = str(label or '').strip().lower()
        return bool(re.match(r'^(shape|polygon)_\d+', lowered))

    @staticmethod
    def _unique_descriptions(items: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = str(item or '').strip()
            if not normalized or normalized in seen:
                continue
            ordered.append(normalized)
            seen.add(normalized)
        return ordered

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

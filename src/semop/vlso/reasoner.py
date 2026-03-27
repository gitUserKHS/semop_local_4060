from __future__ import annotations

import json
import re

from .aligner import VLSOAligner
from .embedding_store import VisualEmbeddingRecord, VisualEmbeddingStore
from .frontier_vlm import FrontierVisionAdapter, VisualSceneAdjudicator
from ..llm_client import LocalLLMConfig
from .language_parser import VLSOLanguageParser
from .qa import LocalTextGenerator, VLSOAnswer, VLSOQuestionAnswerer
from .semantic_scene import SemanticSceneAnalyzer
from .types import SharedWorldModel, VisualObservation, VLSOEntity, VLSORelation
from .vision_backbones import VisionEmbeddingExtractor
from .visual_parser import VLSOVisualParser


class VLSOReasoner:
    ACTION_LABELS = {
        'open_access_action': 'open access',
        'retrieve_item_action': 'retrieve item',
        'inspect_scene_action': 'inspect scene',
        'move_through_scene_action': 'move through scene',
        'engage_visible_target_action': 'engage visible target',
        'aim_or_shoot_action': 'aim or shoot',
        'grasp_or_carry_action': 'grasp or carry',
        'store_items_action': 'store items',
        'control_tool_action': 'control tool',
        'observe_scene_action': 'observe scene',
    }

    def __init__(
        self,
        mode: str = 'hybrid',
        language_mode: str = 'heuristic',
        visual_store_path: str | None = None,
        vision_model_id: str | None = None,
        vision_model_path: str | None = None,
        affordance_weights_path: str | None = None,
        concept_store_path: str | None = None,
        operator_store_path: str | None = None,
        answer_mode: str = 'structured',
        answer_model_id: str = 'Qwen/Qwen2.5-3B-Instruct',
    ) -> None:
        self.mode = mode
        self.language_parser = VLSOLanguageParser(mode=language_mode)
        self.visual_parser = VLSOVisualParser(
            affordance_weights_path=affordance_weights_path,
            concept_store_path=concept_store_path,
            operator_store_path=operator_store_path,
        )
        self.aligner = VLSOAligner()
        resolved_model_id = self._resolve_model_id(mode, vision_model_id)
        auto_resolve = mode in {'deep', 'hybrid'}
        self.embedding_extractor = VisionEmbeddingExtractor(
            model_id=resolved_model_id,
            local_model_path=vision_model_path,
            auto_resolve=auto_resolve,
        )
        self.visual_store = VisualEmbeddingStore(visual_store_path) if visual_store_path else None
        self.answer_mode = answer_mode
        self.answerer = VLSOQuestionAnswerer(
            generator=LocalTextGenerator(LocalLLMConfig(model_id=answer_model_id)) if answer_mode == 'llm' else None
        )
        self.semantic_scene_analyzer = SemanticSceneAnalyzer()
        self.frontier_vision_adapter = FrontierVisionAdapter()
        self.scene_adjudicator = VisualSceneAdjudicator()

    def _resolve_model_id(self, mode: str, vision_model_id: str | None) -> str:
        if vision_model_id:
            return vision_model_id
        if mode == 'deep':
            return 'dinov2_adapter'
        if mode == 'hybrid':
            return 'dinov2_adapter'
        return 'token_geometry_v1'

    def run(self, query: str, visual_input: str | dict | None = None, remember_visual: bool = False, visual_key: str = '') -> SharedWorldModel:
        language_model, _ = self.language_parser.parse(query)
        language_model.metadata['reasoning_mode'] = self.mode
        if visual_input is None:
            return self.aligner.align(language_model, None)
        visual_model, observation = self.visual_parser.parse(visual_input)
        visual_model.metadata['reasoning_mode'] = self.mode
        world = self.aligner.align(language_model, visual_model)
        world.metadata['reasoning_mode'] = self.mode
        self._attach_visual_memory(world, observation, remember_visual=remember_visual, visual_key=visual_key)
        self._attach_semantic_scene_summary(world, observation, query)
        self._attach_frontier_scene_summary(world, observation, query)
        self._attach_scene_adjudication(world, query)
        self._attach_scene_operator_reasoning(world, query)
        return world

    def _attach_semantic_scene_summary(self, world: SharedWorldModel, observation: VisualObservation, query: str) -> None:
        image_path = str(observation.metadata.get('image_path') or '').strip()
        if not image_path:
            return
        candidate_regions = self._semantic_candidate_regions(world)
        summary = self.semantic_scene_analyzer.analyze(image_path, query=query, candidate_regions=candidate_regions)
        world.metadata['semantic_scene_summary'] = summary.model_dump()
        self._attach_region_semantics(world, summary)
        if summary.backend:
            world.audit_trace.append(f'semantic scene backend: {summary.backend}')
        if summary.semantic_level:
            world.audit_trace.append(f'scene semantic level: {summary.semantic_level}')
        if summary.caption:
            world.audit_trace.append('semantic scene caption available')
        for item in summary.region_hypotheses[:3]:
            world.audit_trace.append(f'semantic region: {item.entity_id} -> {item.label} ({item.score:.2f})')
        for note in summary.notes[:2]:
            if note not in world.audit_trace:
                world.audit_trace.append(str(note))

    @staticmethod
    def _semantic_candidate_regions(world: SharedWorldModel) -> list[dict[str, object]]:
        regions: list[dict[str, object]] = []
        for entity in world.entities:
            if entity.modality != 'vision':
                continue
            bbox = entity.attributes.get('bbox')
            if isinstance(bbox, list) and len(bbox) == 4:
                regions.append({'entity_id': entity.id, 'bbox': list(bbox)})
        return regions

    @staticmethod
    def _semantic_entity_type(label: str) -> str:
        lowered = str(label or '').lower()
        if any(token in lowered for token in ['person', 'character', 'soldier']):
            return 'person'
        if any(token in lowered for token in ['handgun', 'pistol', 'rifle', 'weapon', 'gun']):
            return 'tool'
        if 'cart' in lowered:
            return 'vehicle'
        if any(token in lowered for token in ['awning', 'stall', 'shopfront', 'kiosk', 'counter']):
            return 'structure'
        return 'unknown'

    def _attach_region_semantics(self, world: SharedWorldModel, summary) -> None:
        if not getattr(summary, 'region_hypotheses', None):
            return
        by_id = {item.entity_id: item for item in summary.region_hypotheses}
        for entity in world.entities:
            region = by_id.get(entity.id)
            if region is None:
                continue
            entity.attributes['semantic_label'] = region.label
            entity.attributes['semantic_score'] = region.score
            entity.attributes['semantic_candidates'] = [item.model_dump() for item in region.candidates]
            if entity.label == entity.id or entity.label.lower().startswith('shape_') or entity.label.lower().startswith('polygon_'):
                entity.attributes['original_label'] = entity.label
                entity.label = region.label
            inferred_type = self._semantic_entity_type(region.label)
            if entity.entity_type == 'unknown' and inferred_type != 'unknown':
                entity.entity_type = inferred_type

    def _attach_frontier_scene_summary(self, world: SharedWorldModel, observation: VisualObservation, query: str) -> None:
        image_path = str(observation.metadata.get('image_path') or '').strip()
        if not image_path:
            return
        summary = self.frontier_vision_adapter.describe(image_path, query)
        world.metadata['frontier_scene_summary'] = summary.model_dump()
        if summary.backend_ready:
            world.audit_trace.append(f'frontier scene backend: {summary.family or summary.backend}')
            world.audit_trace.append('frontier scene summary available')
        for note in summary.notes[:2]:
            if note not in world.audit_trace:
                world.audit_trace.append(str(note))

    def _attach_scene_adjudication(self, world: SharedWorldModel, query: str) -> None:
        frontier_summary = world.metadata.get('frontier_scene_summary', {}) if isinstance(world.metadata, dict) else {}
        semantic_summary = world.metadata.get('semantic_scene_summary', {}) if isinstance(world.metadata, dict) else {}
        adjudication = self.scene_adjudicator.adjudicate(query, frontier_summary, semantic_summary, world)
        world.metadata['scene_adjudication'] = adjudication.model_dump()
        if adjudication.stack_level:
            world.audit_trace.append(f'scene adjudication: {adjudication.stack_level} ({adjudication.confidence:.2f})')
        for note in adjudication.notes[:2]:
            if note not in world.audit_trace:
                world.audit_trace.append(str(note))

    def _attach_scene_operator_reasoning(self, world: SharedWorldModel, query: str) -> None:
        vision_entities = [entity for entity in world.entities if entity.modality == 'vision']
        if not vision_entities:
            return
        world.add_entity(VLSOEntity(id='observer_agent', label='observer agent', modality='shared', entity_type='agent'))
        self._attach_scene_affordances(world)
        focus = self._select_focus_target(world, query)
        if focus:
            world.add_relation(VLSORelation(source='observer_agent', relation='TARGET_OF_ATTENTION', target=focus, modality='shared', confidence=0.74))
        intents = self._infer_scene_intents(world, query, focus)
        for action_id in intents:
            self._ensure_action_entity(world, action_id)
            world.add_relation(VLSORelation(source='observer_agent', relation='INTENT_OF_AGENT', target=action_id, modality='shared', confidence=0.72))
        self._attach_scene_blockers(world, intents)
        if focus:
            world.audit_trace.append(f'scene operator reasoning focus: {focus}')
        if intents:
            world.audit_trace.append('scene operator reasoning intents: ' + ', '.join(intents[:3]))

    def _select_focus_target(self, world: SharedWorldModel, query: str) -> str:
        candidates = [entity for entity in world.entities if entity.modality == 'vision']
        if not candidates:
            return ''
        area_by_id: dict[str, float] = {}
        for entity in candidates:
            bbox = entity.attributes.get('bbox')
            if isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = [float(value) for value in bbox]
                area_by_id[entity.id] = max(1.0, (x2 - x1) * (y2 - y1))
        max_area = max(area_by_id.values(), default=1.0)
        lowered = str(query or '').lower()
        query_terms = {item for item in re.findall(r'[a-z_][a-z0-9_\-]+', lowered) if len(item) > 2}
        access_query = self._contains_any(lowered, ['open', 'access', 'inside', 'interior', 'retrieve', 'take', 'get', '\uc5f4', '\uc811\uadfc', '\uaebc', '\ub0b4\ubd80'])
        combat_query = self._contains_any(lowered, ['shoot', 'aim', 'fire', 'weapon', 'gun', 'target', '\uc3d8', '\uc870\uc900', '\ubb34\uae30'])
        best = candidates[0]
        best_score = float('-inf')
        for entity in candidates:
            score = 0.0
            semantic_label = str(entity.attributes.get('semantic_label') or '').strip().lower()
            label = str(entity.label or '').strip().lower()
            if semantic_label:
                score += 0.8 + min(0.4, float(entity.attributes.get('semantic_score', 0.0) or 0.0))
            if query_terms:
                candidate_tokens = set(re.findall(r'[a-z_][a-z0-9_\-]+', ' '.join([entity.id.lower(), label, semantic_label])))
                if query_terms & candidate_tokens:
                    score += 1.4
            score += 0.25 * (area_by_id.get(entity.id, 0.0) / max_area)
            entity_type = str(entity.entity_type or '').lower()
            labels = {str(item).upper() for item in (entity.attributes.get('concept_labels') or []) if str(item).strip()}
            if access_query:
                if entity_type in {'opening', 'container'}:
                    score += 1.2
                if {'ACCESS_OPENING_CANDIDATE', 'ACCESS_CONTROL_PART', 'ACCESS_PORT_CANDIDATE', 'EDGE_OPENING'} & labels:
                    score += 1.5
                if {'HANDLE_CANDIDATE', 'HANDLE_LIKE_PART', 'GRASPABLE_PART', 'STRAP_LIKE_PART', 'KNOB_LIKE_PART'} & labels:
                    score += 0.8
            elif combat_query:
                if entity_type in {'tool', 'person'}:
                    score += 1.0
                if any(token in semantic_label for token in ['weapon', 'gun', 'pistol', 'rifle', 'soldier', 'character']):
                    score += 1.1
            else:
                if entity_type in {'person', 'tool', 'vehicle'}:
                    score += 0.5
                if semantic_label:
                    score += 0.3
            if score > best_score:
                best = entity
                best_score = score
        return best.id

    def _infer_scene_intents(self, world: SharedWorldModel, query: str, focus: str) -> list[str]:
        lowered = str(query or '').lower()
        intents: list[str] = []
        if self._contains_any(lowered, ['open', 'access', 'inside', 'interior', '\uc5f4', '\uc811\uadfc', '\ub0b4\ubd80']):
            intents.append('open_access_action')
        if self._contains_any(lowered, ['retrieve', 'take', 'pull', 'get', '\uaebc', '\uac00\uc838', '\ubc1b\uc544']):
            intents.append('retrieve_item_action')
        if self._contains_any(lowered, ['put', 'store', 'insert', 'pack', '\ub123', '\ub2f4', '\ubcf4\uad00']):
            intents.append('store_items_action')
        if self._contains_any(lowered, ['shoot', 'aim', 'fire', 'weapon', 'target', '\uc3d8', '\uc870\uc900', '\uc0ac\uaca9']):
            intents.extend(['aim_or_shoot_action', 'engage_visible_target_action'])
        if self._contains_any(lowered, ['carry', 'grasp', 'hold', 'handle', 'pick', '\uc7a1', '\ub4e4', '\ud734\ub300']):
            intents.append('grasp_or_carry_action')
        if self._contains_any(lowered, ['move', 'walk', 'go', 'route', '\uc774\ub3d9', '\uac00', '\uacbd\ub85c']) or 'path_blocked' in world.constraints:
            intents.append('move_through_scene_action')
        if self._contains_any(lowered, ['describe', 'what is happening', "what's happening", 'what do you see', 'what is visible', '\uc124\uba85', '\ubb50\uac00 \ubcf4\uc5ec', '\ubb34\uc5c7\uc774 \ubcf4\uc5ec']):
            intents.append('observe_scene_action')
        if focus:
            focused_entity = next((entity for entity in world.entities if entity.id == focus), None)
            semantic_label = str(getattr(focused_entity, 'attributes', {}).get('semantic_label') or '').lower() if focused_entity else ''
            if focused_entity is not None and (str(focused_entity.entity_type).lower() == 'tool' or any(token in semantic_label for token in ['weapon', 'gun', 'pistol', 'rifle'])):
                intents.append('control_tool_action')
        if not intents:
            focus_affordances = [relation.target for relation in world.relations if relation.relation == 'AFFORDS' and relation.source == focus]
            intents.extend(str(item) for item in focus_affordances[:2])
        if not intents:
            intents.append('observe_scene_action')
        return self._dedupe(intents)

    def _attach_scene_affordances(self, world: SharedWorldModel) -> None:
        for entity in [item for item in world.entities if item.modality == 'vision']:
            labels = {str(item).upper() for item in (entity.attributes.get('concept_labels') or []) if str(item).strip()}
            semantic_label = str(entity.attributes.get('semantic_label') or entity.label or '').lower()
            entity_type = str(entity.entity_type or '').lower()
            actions: list[str] = []
            if entity_type in {'opening', 'container'} or {'ACCESS_OPENING_CANDIDATE', 'ACCESS_CONTROL_PART', 'ACCESS_PORT_CANDIDATE', 'EDGE_OPENING'} & labels:
                actions.append('open_access_action')
            if entity_type == 'container' or {'HAS_INTERIOR', 'STRUCTURAL_CONTAINER_CANDIDATE', 'MANIPULABLE_CONTAINER'} & labels:
                actions.append('store_items_action')
            if {'HANDLE_CANDIDATE', 'HANDLE_LIKE_PART', 'GRASPABLE_PART', 'STRAP_LIKE_PART', 'KNOB_LIKE_PART', 'TOOL_GRIP_PART'} & labels:
                actions.append('grasp_or_carry_action')
            if entity_type == 'tool' or any(token in semantic_label for token in ['weapon', 'gun', 'pistol', 'rifle']):
                actions.extend(['control_tool_action', 'aim_or_shoot_action'])
            for action_id in self._dedupe(actions):
                self._ensure_action_entity(world, action_id)
                world.add_relation(
                    VLSORelation(
                        source=entity.id,
                        relation='AFFORDS',
                        target=action_id,
                        modality='shared',
                        confidence=0.68,
                        attributes={'source': 'scene_operator_reasoning'},
                    )
                )

    def _attach_scene_blockers(self, world: SharedWorldModel, intents: list[str]) -> None:
        intent_set = set(intents)
        closed_subjects = [relation.source for relation in world.relations if relation.relation == 'STATE' and str(relation.target).lower().endswith(':closed')]
        blocked_subjects = [relation.source for relation in world.relations if relation.relation == 'STATE' and str(relation.target).lower().endswith(':blocked')]
        if 'path_blocked' in world.constraints:
            world.add_entity(VLSOEntity(id='path_blocker', label='blocked path', modality='shared', entity_type='constraint'))
            self._ensure_action_entity(world, 'move_through_scene_action')
            world.add_relation(VLSORelation(source='move_through_scene_action', relation='BLOCKED_BY', target='path_blocker', modality='shared', confidence=0.8))
        for subject in self._dedupe(closed_subjects):
            if {'retrieve_item_action', 'store_items_action'} & intent_set:
                if 'retrieve_item_action' in intent_set:
                    world.add_relation(VLSORelation(source='retrieve_item_action', relation='BLOCKED_BY', target=subject, modality='shared', confidence=0.72))
                if 'store_items_action' in intent_set:
                    world.add_relation(VLSORelation(source='store_items_action', relation='BLOCKED_BY', target=subject, modality='shared', confidence=0.7))
            if 'move_through_scene_action' in intent_set:
                world.add_relation(VLSORelation(source='move_through_scene_action', relation='BLOCKED_BY', target=subject, modality='shared', confidence=0.66))
        for subject in self._dedupe(blocked_subjects):
            for action_id in self._dedupe(list(intent_set) or ['move_through_scene_action']):
                self._ensure_action_entity(world, action_id)
                world.add_relation(VLSORelation(source=action_id, relation='BLOCKED_BY', target=subject, modality='shared', confidence=0.7))

    def _ensure_action_entity(self, world: SharedWorldModel, action_id: str) -> None:
        world.add_entity(
            VLSOEntity(
                id=action_id,
                label=self.ACTION_LABELS.get(action_id, action_id.replace('_', ' ')),
                modality='shared',
                entity_type='action',
            )
        )

    def _attach_visual_memory(self, world: SharedWorldModel, observation: VisualObservation, remember_visual: bool, visual_key: str) -> None:
        vector = self.embedding_extractor.embed_observation(observation)
        summary = self.embedding_extractor.backend_summary()
        world.metadata['vision_backend'] = summary
        world.audit_trace.append(f'reasoning mode: {self.mode}')
        world.audit_trace.append(f"vision backend: {summary['active_backend']}")
        if summary.get('auto_resolved'):
            world.audit_trace.append(f"vision model auto-resolved: {summary['local_model_path']}")
        if summary.get('load_error'):
            world.audit_trace.append(f"vision backend note: {summary['load_error']}")
        if self.visual_store is not None:
            matches = self.visual_store.search(vector, limit=3)
            if matches:
                world.warnings.append('visual memory matches were found in the embedding store')
                world.audit_trace.append('visual embedding retrieval executed')
                world.metadata['visual_memory_matches'] = [item.model_dump() for item in matches]
                for match in matches:
                    world.audit_trace.append(f'visual memory match: {match.label} ({match.score:.3f})')
        if remember_visual and self.visual_store is not None:
            key = visual_key or f'visual:{len(world.entities)}:{len(world.relations)}'
            metadata = {
                'query': world.query,
                'constraints': world.constraints,
                'entity_ids': [item.id for item in world.entities],
            }
            metadata.update(observation.metadata)
            self.visual_store.upsert(
                VisualEmbeddingRecord(
                    key=key,
                    label=world.query,
                    vector=vector,
                    metadata=metadata,
                )
            )
            world.audit_trace.append('visual observation stored in embedding memory')

    @staticmethod
    def to_text(model: SharedWorldModel) -> str:
        lines = [
            f'Query: {model.query}',
            f"Entities: {', '.join(item.id for item in model.entities) or 'none'}",
            f"Relations: {', '.join(f'{item.source}-{item.relation}-{item.target}' for item in model.relations) or 'none'}",
            f"Operators: {', '.join(item.name for item in model.operators) or 'none'}",
            f"Constraints: {', '.join(model.constraints) or 'none'}",
            f"Inferred steps: {' | '.join(model.inferred_steps) or 'none'}",
        ]
        if model.warnings:
            lines.append(f"Warnings: {' | '.join(model.warnings)}")
        if model.audit_trace:
            lines.append(f"Audit: {' | '.join(model.audit_trace)}")
        return "\n".join(lines)

    @staticmethod
    def to_json(model: SharedWorldModel) -> str:
        return json.dumps(model.model_dump(), ensure_ascii=False, indent=2)

    def answer(self, query: str, visual_input: str | dict | None = None, remember_visual: bool = False, visual_key: str = '') -> tuple[SharedWorldModel, VLSOAnswer]:
        world = self.run(query, visual_input=visual_input, remember_visual=remember_visual, visual_key=visual_key)
        answer = self.answerer.answer(query, world, answer_mode=self.answer_mode)
        world.metadata['answer'] = answer.model_dump()
        return world, answer

    @staticmethod
    def _contains_any(text: str, needles: list[str]) -> bool:
        normalized = str(text or '')
        return any(str(item) in normalized for item in needles)

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = str(item or '').strip()
            if not normalized or normalized in seen:
                continue
            ordered.append(normalized)
            seen.add(normalized)
        return ordered

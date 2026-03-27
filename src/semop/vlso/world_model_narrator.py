from __future__ import annotations

import re
from typing import Any

from .types import SharedWorldModel


class WorldModelNarrator:
    VISUAL_LABEL_TRANSLATIONS_KO = {
        'video game screenshot': '\uac8c\uc784 \uc2a4\ud06c\ub9b0\uc0f7',
        'first-person shooter game screenshot': '1\uc778\uce6d \uc288\ud305 \uac8c\uc784 \ud654\uba74',
        'combat video game scene': '\uc804\ud22c \uac8c\uc784 \uc7a5\uba74',
        'third-person action game scene': '3\uc778\uce6d \uc561\uc158 \uac8c\uc784 \uc7a5\uba74',
        'urban street': '\ub3c4\uc2dc \uae38\uac70\ub9ac',
        'market street': '\uc2dc\uc7a5 \uac70\ub9ac',
        'shopfront': '\uac00\uac8c \uc55e\uba74',
        'street market': '\ub178\uc810 \uac70\ub9ac',
        'alley': '\uace8\ubaa9',
        'outdoor daytime scene': '\ub0ae \uc2e4\uc678 \uc7a5\uba74',
        'warehouse aisle': '\ucc3d\uace0 \ud1b5\ub85c',
        'indoor room': '\uc2e4\ub0b4 \uacf5\uac04',
        'construction site': '\uacf5\uc0ac \ud604\uc7a5',
        'parking lot': '\uc8fc\ucc28\uc7a5',
        'close-up object photo': '\uc0ac\ubb3c \uadfc\uc811 \uc0ac\uc9c4',
        'product photo': '\uc81c\ud488 \uc0ac\uc9c4',
        'bag or backpack photo': '\uac00\ubc29 \ub610\ub294 \ubc31\ud329 \uc0ac\uc9c4',
        'street scene': '\uae38\uac70\ub9ac \uc7a5\uba74',
        'person': '\uc0ac\ub78c',
        'human character': '\uc0ac\ub78c\ud615 \uce90\ub9ad\ud130',
        'female game character': '\uc5ec\uc131 \uac8c\uc784 \uce90\ub9ad\ud130',
        'soldier': '\ubcd1\uc0ac',
        'handgun': '\uad8c\ucd1d',
        'pistol': '\uad8c\ucd1d',
        'rifle': '\uc18c\ucd1d',
        'gun held in first person view': '1\uc778\uce6d \uc2dc\uc810\uc758 \ucd1d',
        'weapon': '\ubb34\uae30',
        'player hands': '\ud50c\ub808\uc774\uc5b4 \uc190',
        'hands': '\uc190',
        'arms': '\ud314',
        'backpack': '\ubc31\ud329',
        'bag': '\uac00\ubc29',
        'travel bag': '\uc5ec\ud589 \uac00\ubc29',
        'suitcase': '\uc5ec\ud589\uc6a9 \uac00\ubc29',
        'market stall': '\ub9e4\ub300',
        'shop awning': '\uac00\ub9bc\ub9c9',
        'cart': '\uce74\ud2b8',
        'street cart': '\ub178\uc810 \uce74\ud2b8',
        'building': '\uac74\ubb3c',
        'building facade': '\uac74\ubb3c \uc678\uad00',
        'signboard': '\uac04\ud310',
        'dome': '\ub3d4',
        'store counter': '\ub9e4\ub300',
        'kiosk': '\ud0a4\uc624\uc2a4\ud06c',
        'doorway': '\ucd9c\uc785\uad6c',
        'bicycle': '\uc790\uc804\uac70',
        'car': '\uc790\ub3d9\ucc28',
        'game HUD': '\uac8c\uc784 HUD',
        'mini-map overlay': '\ubbf8\ub2c8\ub9f5',
        'crosshair overlay': '\uc870\uc900\uc810',
        'scoreboard overlay': '\uc2a4\ucf54\uc5b4\ubcf4\ub4dc',
        'timer overlay': '\ud0c0\uc774\uba38',
        'ammo counter overlay': '\ud0c4\uc57d \ud45c\uc2dc',
        'kill feed overlay': '\ud0ac \ub85c\uadf8',
        'chat overlay': '\ucc44\ud305\ucc3d',
        'tool': '\ub3c4\uad6c',
        'vehicle': '\ud0c8\uac83',
        'opening region': '\uc5f4\ub9bc \uc601\uc5ed',
        'container-like region': '\uc6a9\uae30\ucc98\ub7fc \ubcf4\uc774\ub294 \uc601\uc5ed',
        'access-opening region': '\uc811\uadfc \uac1c\uad6c\ubd80 \uc601\uc5ed',
        'graspable part': '\uc7a1\uc744 \uc218 \uc788\ub294 \ubd80\uc704',
        'blocked path': '\ub9c9\ud78c \uacbd\ub85c',
    }

    ACTION_TRANSLATIONS_KO = {
        'open_access_action': '\uc5f4\uac70\ub098 \uc811\uadfc\ud558\ub294 \ud589\ub3d9',
        'retrieve_item_action': '\ubb3c\uac74\uc744 \uaebc\ub0b4\ub294 \ud589\ub3d9',
        'inspect_scene_action': '\uc7a5\uba74\uc744 \uc0b4\ud53c\ub294 \ud589\ub3d9',
        'move_through_scene_action': '\uc7a5\uba74 \uc548\uc73c\ub85c \uc774\ub3d9\ud558\ub294 \ud589\ub3d9',
        'engage_visible_target_action': '\ubcf4\uc774\ub294 \ub300\uc0c1\uc5d0 \ub300\uc751\ud558\ub294 \ud589\ub3d9',
        'aim_or_shoot_action': '\uc870\uc900\ud558\uac70\ub098 \uc0ac\uaca9\ud558\ub294 \ud589\ub3d9',
        'grasp_or_carry_action': '\uc7a1\uac70\ub098 \ub4dc\ub294 \ud589\ub3d9',
        'store_items_action': '\ubb3c\uac74\uc744 \ub123\uac70\ub098 \ubcf4\uad00\ud558\ub294 \ud589\ub3d9',
        'control_tool_action': '\ub3c4\uad6c\ub97c \uc870\uc791\ud558\ub294 \ud589\ub3d9',
        'observe_scene_action': '\uc7a5\uba74\uc744 \uad00\ucc30\ud558\ub294 \ud589\ub3d9',
    }

    ACTION_TRANSLATIONS_EN = {
        'open_access_action': 'opening or gaining access',
        'retrieve_item_action': 'retrieving an item',
        'inspect_scene_action': 'inspecting the scene',
        'move_through_scene_action': 'moving through the scene',
        'engage_visible_target_action': 'engaging the visible target',
        'aim_or_shoot_action': 'aiming or shooting',
        'grasp_or_carry_action': 'grasping or carrying',
        'store_items_action': 'storing items',
        'control_tool_action': 'controlling a tool',
        'observe_scene_action': 'observing the scene',
    }

    STATE_TRANSLATIONS_KO = {
        'open': '\uc5f4\ub9bc',
        'opened': '\uc5f4\ub9bc',
        'closed': '\ub2eb\ud798',
        'blocked': '\ub9c9\ud798',
        'visible': '\ubcf4\uc784',
    }

    def describe(self, query: str, world: SharedWorldModel) -> str:
        return self._describe_temporal(query, world) if self._is_temporal_world(world) else self._describe_scene(query, world)

    def _describe_scene(self, query: str, world: SharedWorldModel) -> str:
        language = self._language(query)
        base = self._scene_base_summary(world, language)
        focus = self._focus_target(world, language)
        intents = self._intent_labels(world, language)
        blockers = self._blocker_labels(world, language)
        reasons = self._scene_reasons(world, language)
        if language == 'ko':
            parts = [base] if base else []
            if focus:
                parts.append(f'\uc2dc\uc120\uc740 \uc8fc\ub85c {focus}\uc5d0 \ubaa8\uc785\ub2c8\ub2e4.')
            if intents:
                parts.append('\ucd94\ub860\ub41c \uc758\ub3c4\ub294 ' + ', '.join(intents[:2]) + ' \ucabd\uc785\ub2c8\ub2e4.')
            if blockers:
                parts.append('\ub2e4\ub9cc ' + ', '.join(blockers[:2]) + ' \uac19\uc740 \uc81c\uc57d\uc774 \ubcf4\uc785\ub2c8\ub2e4.')
            if reasons:
                parts.append('\uc774\ub807\uac8c \ubcf8 \uc774\uc720\ub294 ' + ' / '.join(reasons[:3]) + ' \ub54c\ubb38\uc785\ub2c8\ub2e4.')
            return ' '.join(part for part in parts if part).strip() or '\ud604\uc7ac \uc6d4\ub4dc \ubaa8\ub378\ub9cc\uc73c\ub85c\ub294 \uc7a5\uba74\uc744 \uc548\uc815\uc801\uc73c\ub85c \uc124\uba85\ud558\uae30 \uc5b4\ub835\uc2b5\ub2c8\ub2e4.'
        parts = [base] if base else []
        if focus:
            parts.append(f'The main focus of attention is {focus}.')
        if intents:
            parts.append('Recovered intent points to ' + ', '.join(intents[:2]) + '.')
        if blockers:
            parts.append('Immediate constraints include ' + ', '.join(blockers[:2]) + '.')
        if reasons:
            parts.append('Why: ' + ' / '.join(reasons[:3]) + '.')
        return ' '.join(part for part in parts if part).strip() or 'The current visual world model is too weak for a reliable answer.'

    def _describe_temporal(self, query: str, world: SharedWorldModel) -> str:
        language = self._language(query)
        temporal = world.metadata.get('temporal_scene_summary', {}) if isinstance(world.metadata, dict) else {}
        frame_count = int(temporal.get('frame_count') or 0)
        stable_entities = [self._display_label(world, item, language) for item in temporal.get('stable_entities', [])[:4]]
        changed_entities = [self._display_label(world, item, language) for item in temporal.get('changed_entities', [])[:4]]
        event_lines = [self._event_text(event, world, language) for event in getattr(world, 'events', [])[:4]]
        focus = self._focus_target(world, language)
        intents = self._intent_labels(world, language)
        reasons = [item for item in [self._stable_relation_reason(temporal.get('stable_relations', [])[:2], world, language)] if item]
        reasons.extend(item for item in event_lines[:2] if item)
        if language == 'ko':
            parts: list[str] = []
            if frame_count > 0:
                parts.append(f'{frame_count}\uac1c \ud504\ub808\uc784\uc744 \ubb36\uc5b4 \ubcf4\uba74 \ud558\ub098\uc758 \uc2dc\uac04\uc801 \uc7a5\uba74\uc73c\ub85c \uc815\ub9ac\ub429\ub2c8\ub2e4.')
            if stable_entities:
                parts.append('\uacc4\uc18d \uc720\uc9c0\ub418\ub294 \ud575\uc2ec \uc694\uc18c\ub294 ' + ', '.join(stable_entities) + '\uc785\ub2c8\ub2e4.')
            if changed_entities:
                parts.append('\ud504\ub808\uc784 \uc0ac\uc774\uc5d0\uc11c \ub2ec\ub77c\uc9c0\ub294 \ub300\uc0c1\uc740 ' + ', '.join(changed_entities) + '\uc785\ub2c8\ub2e4.')
            if focus:
                parts.append(f'\ub9c8\uc9c0\ub9c9\uc5d0\uc11c \uc2dc\uc120\uc774 \ubaa8\uc778 \ub300\uc0c1\uc740 {focus}\uc785\ub2c8\ub2e4.')
            if intents:
                parts.append('\ucd94\ub860\ub41c \ud589\ub3d9 \uc758\ub3c4\ub294 ' + ', '.join(intents[:2]) + ' \ucabd\uc785\ub2c8\ub2e4.')
            if reasons:
                parts.append('\uc2dc\uac04\uc801 \uadfc\uac70\ub294 ' + ' / '.join(reasons[:3]) + ' \uc785\ub2c8\ub2e4.')
            return ' '.join(part for part in parts if part).strip() or '\ud504\ub808\uc784 \uac04 \ubcc0\ud654 \uadfc\uac70\uac00 \uc544\uc9c1 \uc57d\ud569\ub2c8\ub2e4.'
        parts = []
        if frame_count > 0:
            parts.append(f'The scene was aggregated across {frame_count} frame(s).')
        if stable_entities:
            parts.append('Stable anchors: ' + ', '.join(stable_entities) + '.')
        if changed_entities:
            parts.append('Changing entities: ' + ', '.join(changed_entities) + '.')
        if focus:
            parts.append(f'The latest focus of attention is {focus}.')
        if intents:
            parts.append('Recovered intent: ' + ', '.join(intents[:2]) + '.')
        if reasons:
            parts.append('Temporal evidence: ' + ' / '.join(reasons[:3]) + '.')
        return ' '.join(part for part in parts if part).strip() or 'Temporal evidence is still too weak for a reliable answer.'

    def _scene_base_summary(self, world: SharedWorldModel, language: str) -> str:
        preferred = self._preferred_scene_answer(world)
        semantic = world.metadata.get('semantic_scene_summary', {}) if isinstance(world.metadata, dict) else {}
        scene_hypotheses = semantic.get('scene_hypotheses', []) if isinstance(semantic, dict) else []
        object_hypotheses = semantic.get('object_hypotheses', []) if isinstance(semantic, dict) else []
        overlay_hypotheses = semantic.get('overlay_hypotheses', []) if isinstance(semantic, dict) else []
        region_hypotheses = semantic.get('region_hypotheses', []) if isinstance(semantic, dict) else []
        scene_label = ''
        if scene_hypotheses:
            top = scene_hypotheses[0]
            if float(top.get('score', 0.0) or 0.0) >= 0.22:
                scene_label = str(top.get('label') or '').strip()
        object_labels = self._dedupe(
            [str(item.get('label') or '').strip() for item in object_hypotheses if isinstance(item, dict) and float(item.get('score', 0.0) or 0.0) >= 0.22]
            + [str(item.get('label') or '').strip() for item in region_hypotheses if isinstance(item, dict) and float(item.get('score', 0.0) or 0.0) >= 0.18]
            + [self._entity_semantic_label(entity) for entity in world.entities if entity.modality == 'vision']
        )
        overlay_labels = self._dedupe([str(item.get('label') or '').strip() for item in overlay_hypotheses if isinstance(item, dict) and float(item.get('score', 0.0) or 0.0) >= 0.22])
        if language == 'en' and preferred:
            additions: list[str] = []
            if object_labels:
                missing_objects = [item for item in object_labels if item and item.lower() not in preferred.lower()]
                if missing_objects:
                    additions.append('Likely semantic regions include ' + ', '.join(missing_objects[:3]) + '.')
            if overlay_labels:
                missing_overlays = [item for item in overlay_labels if item and item.lower() not in preferred.lower()]
                if missing_overlays:
                    additions.append('Likely overlays include ' + ', '.join(missing_overlays[:3]) + '.')
            return ' '.join([preferred] + additions).strip()
        if language == 'ko':
            translated_scene = self._translate(scene_label, language)
            translated_objects = [self._translate(item, language) for item in object_labels if item]
            translated_overlays = [self._translate(item, language) for item in overlay_labels if item]
            game_like = scene_label in {'video game screenshot', 'first-person shooter game screenshot', 'combat video game scene'}
            if game_like:
                parts = ['1\uc778\uce6d \uc2c8\ud305 \ub610\ub294 \uc804\ud22c \uac8c\uc784 \ud654\uba74\ucc98\ub7fc \ubcf4\uc785\ub2c8\ub2e4.']
                if translated_objects:
                    parts.append('\ubcf4\uc774\ub294 \uc694\uc18c\ub85c\ub294 ' + ', '.join(translated_objects[:5]) + ' \uc815\ub3c4\uac00 \uc7a1\ud799\ub2c8\ub2e4.')
                if translated_overlays:
                    parts.append('\ud654\uba74 UI\ub85c\ub294 ' + ', '.join(translated_overlays[:4]) + ' \uac19\uc740 \uc624\ubc84\ub808\uc774\uac00 \ubcf4\uc785\ub2c8\ub2e4.')
                return ' '.join(parts)
            if translated_scene:
                parts = ['\uc774 \uc7a5\uba74\uc740 ' + translated_scene + '\ucc98\ub7fc \ubcf4\uc785\ub2c8\ub2e4.']
                if translated_objects:
                    parts.append('\ubcf4\uc774\ub294 \uc694\uc18c\ub85c\ub294 ' + ', '.join(translated_objects[:5]) + ' \uc815\ub3c4\uac00 \uc7a1\ud799\ub2c8\ub2e4.')
                if translated_overlays:
                    parts.append('\ud654\uba74 UI\ub85c\ub294 ' + ', '.join(translated_overlays[:4]) + ' \uac19\uc740 \uc624\ubc84\ub808\uc774\uac00 \ubcf4\uc785\ub2c8\ub2e4.')
                return ' '.join(parts)
            if translated_objects:
                return '\ubcf4\uc774\ub294 \uc694\uc18c\ub85c\ub294 ' + ', '.join(translated_objects[:5]) + ' \uc815\ub3c4\uac00 \uc7a1\ud799\ub2c8\ub2e4.'
            return self._structural_scene_summary(world, language)
        if scene_label:
            parts = ['This scene most likely looks like ' + scene_label + '.']
            if object_labels:
                parts.append('Visible elements include ' + ', '.join(object_labels[:5]) + '.')
            if overlay_labels:
                parts.append('Likely overlays include ' + ', '.join(overlay_labels[:4]) + '.')
            return ' '.join(parts)
        if object_labels:
            return 'Visible elements include ' + ', '.join(object_labels[:5]) + '.'
        return self._structural_scene_summary(world, language)

    def _preferred_scene_answer(self, world: SharedWorldModel) -> str:
        adjudication = world.metadata.get('scene_adjudication', {}) if isinstance(world.metadata, dict) else {}
        if isinstance(adjudication, dict):
            preferred = str(adjudication.get('preferred_answer') or '').strip()
            if preferred:
                return preferred
        frontier = world.metadata.get('frontier_scene_summary', {}) if isinstance(world.metadata, dict) else {}
        if isinstance(frontier, dict):
            answer = str(frontier.get('answer_text') or '').strip()
            if answer:
                return answer
        semantic = world.metadata.get('semantic_scene_summary', {}) if isinstance(world.metadata, dict) else {}
        if isinstance(semantic, dict):
            caption = str(semantic.get('caption') or '').strip()
            if caption:
                return caption
        return ''

    def _structural_scene_summary(self, world: SharedWorldModel, language: str) -> str:
        structural = self._dedupe([self._structural_label(entity, language) for entity in world.entities if entity.modality == 'vision'])
        if language == 'ko':
            if structural:
                return '\ud604\uc7ac \ub85c\uceec \ube44\uc804 \uc2a4\ud0dd\uc740 \uc774 \uc7a5\uba74\uc744 \uc0ac\ub78c\ucc98\ub7fc \uc758\ubbf8\uc801\uc73c\ub85c \ud574\uc11d\ud558\uc9c0\ub294 \ubabb\ud588\uace0, ' + ', '.join(structural[:4]) + ' \uac19\uc740 \uac70\uce5c \uad6c\uc870 \uc2e0\ud638\ub9cc \uc7a1\uc558\uc2b5\ub2c8\ub2e4.'
            return '\ud604\uc7ac \ube44\uc804 \uc2a4\ud0dd\uc740 \uac70\uce5c \uad6c\uc870 \uc218\uc900\uc5d0\uc11c\ub9cc \uc7a5\uba74\uc744 \uc7a1\uace0 \uc788\uc2b5\ub2c8\ub2e4.'
        if structural:
            return 'This image is not yet being semantically understood at a human level by the current local vision stack. Right now I can only ground coarse structural regions such as ' + ', '.join(structural[:4]) + '.'
        return 'The current local visual stack remains at a coarse structural grounding level.'

    def _scene_reasons(self, world: SharedWorldModel, language: str) -> list[str]:
        reasons: list[str] = []
        focus = self._target_relation(world, 'TARGET_OF_ATTENTION')
        if focus is not None:
            reasons.append((self._display_label(world, focus[2], language) + '\uc5d0 \uc8fc\uc758\uac00 \ubaa8\uc785\ub2c8\ub2e4') if language == 'ko' else ('attention converges on ' + self._display_label(world, focus[2], language)))
        for source, _, target in self._relations(world, 'AFFORDS')[:2]:
            subject = self._display_label(world, source, language)
            action = self._display_label(world, target, language)
            reasons.append((subject + '\uac00 ' + action + '\uc744 \uac00\ub2a5\ud558\uac8c \ud569\ub2c8\ub2e4') if language == 'ko' else (subject + ' affords ' + action))
        for source, _, target in self._relations(world, 'BLOCKED_BY')[:2]:
            action = self._display_label(world, source, language)
            blocker = self._display_label(world, target, language)
            reasons.append((action + '\uc740 ' + blocker + '\ub54c\ubb38\uc5d0 \uc81c\uc57d\ub429\ub2c8\ub2e4') if language == 'ko' else (action + ' is blocked by ' + blocker))
        for source, _, target in self._relations(world, 'STATE')[:2]:
            subject = self._display_label(world, source, language)
            state = self._display_label(world, target, language)
            reasons.append((subject + '\uc758 \uc0c1\ud0dc\uac00 ' + state + '\uc785\ub2c8\ub2e4') if language == 'ko' else (subject + ' is ' + state))
        return self._dedupe(reasons)

    def _stable_relation_reason(self, relations: list[str], world: SharedWorldModel, language: str) -> str:
        relation_texts: list[str] = []
        for item in relations:
            parts = str(item or '').split(':', 2)
            if len(parts) != 3:
                continue
            source, relation, target = parts
            source_label = self._display_label(world, source, language)
            target_label = self._display_label(world, target, language)
            if language == 'ko':
                relation_texts.append(source_label + '\uacfc ' + target_label + ' \uc0ac\uc774\uc758 ' + relation + ' \uad00\uacc4\uac00 \uc720\uc9c0\ub429\ub2c8\ub2e4')
            else:
                relation_texts.append(relation + ' between ' + source_label + ' and ' + target_label + ' stays stable')
        return ' / '.join(relation_texts[:2])

    def _event_text(self, event: Any, world: SharedWorldModel, language: str) -> str:
        if hasattr(event, 'model_dump'):
            payload = event.model_dump()
        elif isinstance(event, dict):
            payload = event
        else:
            payload = {}
        event_type = str(payload.get('event_type') or '').strip()
        participants = payload.get('participants') or {}
        attributes = payload.get('attributes') or {}
        entity = self._display_label(world, str(participants.get('entity') or ''), language)
        previous_state = self._display_label(world, str(participants.get('previous_state') or ''), language) if participants.get('previous_state') else ''
        current_state = self._display_label(world, str(participants.get('current_state') or ''), language) if participants.get('current_state') else ''
        frame_label = str(attributes.get('frame_label') or '').strip()
        if language == 'ko':
            if event_type == 'appearance' and entity:
                return (frame_label + '\uc5d0 ' if frame_label else '') + entity + '\uac00 \uc0c8\ub85c \ub4f1\uc7a5\ud588\uc2b5\ub2c8\ub2e4'
            if event_type == 'disappearance' and entity:
                return (frame_label + '\uc5d0 ' if frame_label else '') + entity + '\uac00 \uc0ac\ub77c\uc84c\uc2b5\ub2c8\ub2e4'
            if event_type == 'state_change' and entity:
                return entity + '\uc758 \uc0c1\ud0dc\uac00 ' + previous_state + '\uc5d0\uc11c ' + current_state + '\ub85c \ubc14\ub00c\uc5c8\uc2b5\ub2c8\ub2e4'
        else:
            if event_type == 'appearance' and entity:
                return (frame_label + ': ' if frame_label else '') + entity + ' appeared'
            if event_type == 'disappearance' and entity:
                return (frame_label + ': ' if frame_label else '') + entity + ' disappeared'
            if event_type == 'state_change' and entity:
                return entity + ' changed from ' + previous_state + ' to ' + current_state
        return str(payload.get('label') or '').strip()

    def _focus_target(self, world: SharedWorldModel, language: str) -> str:
        relation = self._target_relation(world, 'TARGET_OF_ATTENTION')
        return self._display_label(world, relation[2], language) if relation else ''

    def _intent_labels(self, world: SharedWorldModel, language: str) -> list[str]:
        return self._dedupe([self._display_label(world, target, language) for _, _, target in self._relations(world, 'INTENT_OF_AGENT')])

    def _blocker_labels(self, world: SharedWorldModel, language: str) -> list[str]:
        labels: list[str] = []
        for source, _, target in self._relations(world, 'BLOCKED_BY'):
            action = self._display_label(world, source, language)
            blocker = self._display_label(world, target, language)
            labels.append((action + '\uc740 ' + blocker + '\uc5d0 \uac00\ub85c\ub9c9\ud799\ub2c8\ub2e4') if language == 'ko' else (action + ' blocked by ' + blocker))
        return self._dedupe(labels)

    def _relations(self, world: SharedWorldModel, name: str) -> list[tuple[str, str, str]]:
        return [(item.source, item.relation, item.target) for item in world.relations if item.relation == name]

    def _target_relation(self, world: SharedWorldModel, name: str) -> tuple[str, str, str] | None:
        rows = self._relations(world, name)
        return rows[0] if rows else None

    def _display_label(self, world: SharedWorldModel, entity_id: str, language: str) -> str:
        normalized = str(entity_id or '').strip()
        if not normalized:
            return ''
        if normalized.startswith('state:'):
            return self._translate_state(normalized.split(':')[-1], language)
        if normalized in self.ACTION_TRANSLATIONS_KO or normalized in self.ACTION_TRANSLATIONS_EN:
            return self._translate_action(normalized, language)
        for entity in world.entities:
            if entity.id != normalized:
                continue
            semantic_label = self._entity_semantic_label(entity)
            if semantic_label:
                return self._translate(semantic_label, language)
            label = str(entity.label or entity.id).strip()
            if label and not self._is_generic_visual_label(label):
                return self._translate(label, language)
            return self._translate(self._fallback_entity_label(entity), language)
        return self._translate(normalized.replace('_', ' '), language)

    def _entity_semantic_label(self, entity: Any) -> str:
        semantic_label = str(getattr(entity, 'attributes', {}).get('semantic_label') or '').strip()
        if semantic_label:
            return semantic_label
        label = str(getattr(entity, 'label', '') or '').strip()
        if label and not self._is_generic_visual_label(label) and '[' not in label:
            return label
        return ''

    def _fallback_entity_label(self, entity: Any) -> str:
        entity_type = str(getattr(entity, 'entity_type', '') or '').lower()
        if entity_type == 'person':
            return 'human character'
        if entity_type == 'tool':
            return 'tool'
        if entity_type == 'vehicle':
            return 'vehicle'
        if entity_type == 'opening':
            return 'opening region'
        if entity_type == 'container':
            return 'container-like region'
        return str(getattr(entity, 'id', 'object')).replace('_', ' ')

    def _structural_label(self, entity: Any, language: str) -> str:
        labels = getattr(entity, 'attributes', {}).get('concept_labels') or []
        upper = {str(item).upper() for item in labels} if isinstance(labels, list) else set()
        if {'ACCESS_OPENING_CANDIDATE', 'ACCESS_CONTROL_PART', 'ACCESS_PORT_CANDIDATE', 'EDGE_OPENING'} & upper:
            return self._translate('access-opening region', language)
        if {'HANDLE_CANDIDATE', 'HANDLE_LIKE_PART', 'GRASPABLE_PART', 'STRAP_LIKE_PART', 'KNOB_LIKE_PART', 'TOOL_GRIP_PART'} & upper:
            return self._translate('graspable part', language)
        if {'HAS_INTERIOR', 'STRUCTURAL_CONTAINER_CANDIDATE', 'MANIPULABLE_CONTAINER'} & upper:
            return self._translate('container-like region', language)
        entity_type = str(getattr(entity, 'entity_type', '') or '').lower()
        if entity_type == 'person':
            return self._translate('human character', language)
        if entity_type == 'vehicle':
            return self._translate('vehicle', language)
        return ''

    def _translate(self, label: str, language: str) -> str:
        normalized = str(label or '').strip()
        if not normalized or language != 'ko':
            return normalized
        return self.VISUAL_LABEL_TRANSLATIONS_KO.get(normalized, normalized)

    def _translate_action(self, action_id: str, language: str) -> str:
        normalized = str(action_id or '').strip()
        if language == 'ko':
            return self.ACTION_TRANSLATIONS_KO.get(normalized, normalized.replace('_', ' '))
        return self.ACTION_TRANSLATIONS_EN.get(normalized, normalized.replace('_', ' '))

    def _translate_state(self, state_value: str, language: str) -> str:
        normalized = str(state_value or '').strip().lower()
        if language == 'ko':
            return self.STATE_TRANSLATIONS_KO.get(normalized, normalized)
        return normalized

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            normalized = str(item or '').strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(normalized)
        return ordered

    @staticmethod
    def _is_temporal_world(world: SharedWorldModel) -> bool:
        metadata = world.metadata if isinstance(world.metadata, dict) else {}
        return bool(metadata.get('temporal_scene_summary') or getattr(world, 'events', []))

    @staticmethod
    def _language(text: str) -> str:
        return 'ko' if re.search(r'[\uac00-\ud7a3]', str(text or '')) else 'en'

    @staticmethod
    def _is_generic_visual_label(label: str) -> bool:
        lowered = str(label or '').strip().lower()
        return bool(re.match(r'^(shape|polygon)_\d+', lowered))

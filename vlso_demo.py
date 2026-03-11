from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from semop import VLSOReasoner


def load_json_or_path(value: str) -> dict[str, Any] | str:
    if os.path.exists(value):
        return json.loads(Path(value).read_text(encoding='utf-8-sig'))
    stripped = value.strip()
    if stripped.startswith('{'):
        return json.loads(stripped)
    return value


def build_visual_input(args: argparse.Namespace) -> dict[str, Any] | str | None:
    payload: dict[str, Any] | str | None = None
    if args.detector_json:
        loaded = load_json_or_path(args.detector_json)
        payload = loaded if isinstance(loaded, dict) else {'constraints': [str(loaded)]}
    elif args.visual_json:
        payload = load_json_or_path(args.visual_json)
    elif args.visual_text:
        payload = args.visual_text

    if args.image_path:
        if payload is None:
            payload = {'image_path': args.image_path, 'metadata': {'image_path': args.image_path}}
        elif isinstance(payload, dict):
            payload.setdefault('metadata', {})
            payload['image_path'] = args.image_path
            payload['metadata'].setdefault('image_path', args.image_path)
        else:
            payload = {'image_path': args.image_path, 'metadata': {'image_path': args.image_path}, 'constraints': [payload]}
    return payload


def build_reasoner(args: argparse.Namespace, concept_store_path: str | None = None) -> VLSOReasoner:
    return VLSOReasoner(
        mode=args.mode,
        language_mode=args.language_mode,
        visual_store_path=args.visual_store,
        vision_model_id=args.vision_backbone,
        vision_model_path=args.vision_model_path,
        affordance_weights_path=args.affordance_weights,
        concept_store_path=concept_store_path,
        operator_store_path=args.operator_store,
        answer_mode=args.answer_mode,
        answer_model_id=args.answer_model_id,
    )


def run_reasoner_payload(reasoner: VLSOReasoner, args: argparse.Namespace, visual_input: dict[str, Any] | str | None) -> dict[str, Any]:
    model, answer = reasoner.answer(args.query, visual_input=visual_input, remember_visual=args.remember_visual, visual_key=args.visual_key)
    payload = model.model_dump()
    payload['answer'] = answer.model_dump()
    return payload


def opening_candidates_from_payload(payload: dict[str, Any]) -> list[str]:
    world = payload.get('world', payload)
    entities = world.get('entities', []) if isinstance(world, dict) else []
    ranked: list[tuple[int, str]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        if entity.get('modality') != 'vision':
            continue
        attrs = entity.get('attributes', {}) if isinstance(entity.get('attributes'), dict) else {}
        labels = attrs.get('concept_labels') or []
        if not isinstance(labels, list):
            continue
        upper = {str(item).upper() for item in labels}
        if not ({'ACCESS_OPENING_CANDIDATE', 'ZIPPER_LIKE_PART', 'EDGE_OPENING'} & upper):
            continue
        score = 0
        if 'ACCESS_OPENING_CANDIDATE' in upper:
            score += 3
        if 'ZIPPER_LIKE_PART' in upper:
            score += 2
        if 'EDGE_OPENING' in upper:
            score += 2
        if 'STRAP_LIKE_PART' in upper and 'ACCESS_OPENING_CANDIDATE' not in upper:
            score -= 2
        bbox = attrs.get('bbox')
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            area = max(1, (x2 - x1) * (y2 - y1))
            if (x1 <= 2 or y1 <= 2) and area <= 600:
                continue
            if x1 <= 2 or y1 <= 2:
                score -= 4
            if area <= 600:
                score -= 2
        ranked.append((score, str(entity.get('id', ''))))
    ranked.sort(key=lambda item: item[0], reverse=True)
    output: list[str] = []
    seen: set[str] = set()
    for score, entity_id in ranked:
        if score <= 0 or not entity_id or entity_id in seen:
            continue
        output.append(entity_id)
        seen.add(entity_id)
    return output


def compare_payloads(primary: dict[str, Any], comparison: dict[str, Any], primary_store: str | None, comparison_store: str | None) -> dict[str, Any]:
    primary_answer = str(primary.get('answer', {}).get('answer_text', ''))
    comparison_answer = str(comparison.get('answer', {}).get('answer_text', ''))
    return {
        'primary_concept_store': primary_store or '',
        'comparison_concept_store': comparison_store or '',
        'primary_answer_text': primary_answer,
        'comparison_answer_text': comparison_answer,
        'answers_match': primary_answer == comparison_answer,
        'primary_opening_candidates': opening_candidates_from_payload(primary),
        'comparison_opening_candidates': opening_candidates_from_payload(comparison),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Demo the Vision-Language Semantic Operator prototype')
    parser.add_argument('--query', required=True, help='language task or instruction')
    parser.add_argument('--mode', choices=['heuristic', 'hybrid', 'deep'], default='deep')
    parser.add_argument('--language-mode', choices=['heuristic', 'llm'], default='heuristic')
    parser.add_argument('--visual-json', help='JSON string or a path to a JSON file with structured visual observations')
    parser.add_argument('--detector-json', help='detector output JSON string or file path')
    parser.add_argument('--visual-text', help="simple scene text such as 'closed zipper on bag'")
    parser.add_argument('--image-path', help='optional local image path for a vision backbone embedding pass')
    parser.add_argument('--visual-store', help='optional SQLite visual memory store path')
    parser.add_argument('--remember-visual', action='store_true', help='store the current visual input in the embedding memory')
    parser.add_argument('--visual-key', default='', help='optional key when storing the visual observation')
    parser.add_argument('--vision-backbone', choices=['token_geometry_v1', 'dinov2_adapter', 'openclip_adapter'], help='override the default backbone selected by --mode')
    parser.add_argument('--vision-model-path', help='local checkpoint path for dinov2_adapter or openclip_adapter')
    parser.add_argument('--affordance-weights', help='optional learned affordance weights JSON')
    parser.add_argument('--concept-store', help='optional few-shot visual concept memory SQLite path')
    parser.add_argument('--compare-concept-store', help='optional second concept store path for side-by-side comparison')
    parser.add_argument('--operator-store', help='optional learned visual operator prototype SQLite path')
    parser.add_argument('--answer-mode', choices=['structured', 'llm'], default='structured')
    parser.add_argument('--answer-model-id', default='Qwen/Qwen2.5-3B-Instruct', help='text LLM model id for --answer-mode llm')
    parser.add_argument('--format', choices=['text', 'json'], default='text')
    args = parser.parse_args()

    visual_input = build_visual_input(args)
    primary_reasoner = build_reasoner(args, concept_store_path=args.concept_store)
    primary_payload = run_reasoner_payload(primary_reasoner, args, visual_input)

    if args.compare_concept_store:
        comparison_reasoner = build_reasoner(args, concept_store_path=args.compare_concept_store)
        comparison_payload = run_reasoner_payload(comparison_reasoner, args, visual_input)
        comparison_summary = compare_payloads(primary_payload, comparison_payload, args.concept_store, args.compare_concept_store)
        if args.format == 'json':
            print(json.dumps({
                'primary': primary_payload,
                'comparison': comparison_payload,
                'comparison_summary': comparison_summary,
            }, ensure_ascii=False, indent=2))
            return
        print(json.dumps(comparison_summary, ensure_ascii=False, indent=2))
        return

    if args.format == 'json':
        print(json.dumps(primary_payload, ensure_ascii=False, indent=2))
        return
    world = primary_reasoner.run(args.query, visual_input=visual_input)
    print(VLSOReasoner.to_text(world))
    answer = primary_payload.get('answer', {})
    print()
    print(f"Answer ({answer.get('answer_mode', 'structured')}): {answer.get('answer_text', '')}")
    evidence = answer.get('evidence', [])
    if isinstance(evidence, list) and evidence:
        print('Evidence: ' + ' | '.join(str(item) for item in evidence[:6]))


if __name__ == '__main__':
    main()

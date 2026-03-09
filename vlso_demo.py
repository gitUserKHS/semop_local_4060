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
    parser.add_argument('--answer-mode', choices=['structured', 'llm'], default='structured')
    parser.add_argument('--answer-model-id', default='Qwen/Qwen2.5-3B-Instruct', help='text LLM model id for --answer-mode llm')
    parser.add_argument('--format', choices=['text', 'json'], default='text')
    args = parser.parse_args()

    visual_input = build_visual_input(args)
    reasoner = VLSOReasoner(
        mode=args.mode,
        language_mode=args.language_mode,
        visual_store_path=args.visual_store,
        vision_model_id=args.vision_backbone,
        vision_model_path=args.vision_model_path,
        affordance_weights_path=args.affordance_weights,
        concept_store_path=args.concept_store,
        answer_mode=args.answer_mode,
        answer_model_id=args.answer_model_id,
    )
    model, answer = reasoner.answer(args.query, visual_input=visual_input, remember_visual=args.remember_visual, visual_key=args.visual_key)
    if args.format == 'json':
        payload = model.model_dump()
        payload['answer'] = answer.model_dump()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(VLSOReasoner.to_text(model))
    print()
    print(f'Answer ({answer.answer_mode}): {answer.answer_text}')
    if answer.evidence:
        print('Evidence: ' + ' | '.join(answer.evidence[:6]))


if __name__ == '__main__':
    main()

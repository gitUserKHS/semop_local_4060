from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from semop import VisualGeometry3DWorkbench


def _workbench(args: argparse.Namespace) -> VisualGeometry3DWorkbench:
    return VisualGeometry3DWorkbench(
        concept_store_path=args.concept_store,
        operator_store_path=args.operator_store,
        affordance_weights_path=args.affordance_weights,
        mode=args.mode,
        answer_mode=args.answer_mode,
    )


def _normalize_inputs(args: argparse.Namespace) -> list[str] | str | None:
    if getattr(args, 'input_dir', None):
        return args.input_dir
    inputs = [item for item in getattr(args, 'inputs', []) or [] if item]
    if inputs:
        return inputs
    return None


def _print(payload: object) -> None:
    if hasattr(payload, 'model_dump'):
        payload = payload.model_dump()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_collect_starter(args: argparse.Namespace) -> None:
    summary = _workbench(args).collect_starter_scenes(args.output_dir)
    _print(summary)


def cmd_collect_inputs(args: argparse.Namespace) -> None:
    inputs = _normalize_inputs(args)
    if not inputs:
        raise SystemExit('Provide --input-dir or --inputs for collect-inputs.')
    summary = _workbench(args).collect_input_manifest(inputs, args.output_dir)
    _print(summary)


def cmd_train(args: argparse.Namespace) -> None:
    workbench = _workbench(args)
    inputs = _normalize_inputs(args)
    if inputs:
        summary = workbench.train_bundle_from_inputs(inputs, args.output_dir, limit_scenes=args.limit, eval_input=args.eval_input)
    else:
        summary = workbench.train_starter_bundle(args.output_dir, limit_scenes=args.limit)
    _print(summary)


def cmd_reconstruct(args: argparse.Namespace) -> None:
    summary = _workbench(args).reconstruct_scene(args.query, args.visual_input, args.output_dir)
    _print(summary)


def cmd_reconstruct_batch(args: argparse.Namespace) -> None:
    inputs = _normalize_inputs(args)
    if not inputs:
        raise SystemExit('Provide --input-dir or --inputs for reconstruct-batch.')
    summary = _workbench(args).reconstruct_batch(args.query, inputs, args.output_dir, limit=args.limit)
    _print(summary)


def cmd_bootcamp(args: argparse.Namespace) -> None:
    workbench = _workbench(args)
    inputs = _normalize_inputs(args)
    if inputs:
        collection = workbench.collect_input_manifest(inputs, args.output_dir)
        training = workbench.train_bundle_from_inputs(inputs, args.output_dir, limit_scenes=args.limit, eval_input=args.eval_input)
        visual_input = args.visual_input or (collection.input_paths[0] if collection.input_paths else '')
    else:
        collection = workbench.collect_starter_scenes(args.output_dir)
        training = workbench.train_starter_bundle(args.output_dir, limit_scenes=args.limit)
        starter = Path(args.output_dir) / 'starter_scenes'
        visual_input = args.visual_input or str(starter / 'parallel_perpendicular_scene.json')
    if not visual_input:
        raise SystemExit('No visual input was available for reconstruction.')
    reconstruction = workbench.reconstruct_scene(args.query, visual_input, args.output_dir)
    _print(
        {
            'bootcamp': True,
            'collection': collection.model_dump(),
            'training': training.model_dump(),
            'reconstruction': reconstruction.model_dump(),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Collect, train, and reconstruct visual geometry into 3D bundles.')
    parser.add_argument('--concept-store', default=None)
    parser.add_argument('--operator-store', default=None)
    parser.add_argument('--affordance-weights', default='data/vlso_samples/trained_affordance_weights.json')
    parser.add_argument('--mode', default='deep')
    parser.add_argument('--answer-mode', default='structured')
    subparsers = parser.add_subparsers(dest='command', required=True)

    collect_starter = subparsers.add_parser('collect-starter', help='Generate starter geometry scenes and eval JSONL.')
    collect_starter.add_argument('--output-dir', default='data/visual_geometry_3d_cli_run')
    collect_starter.set_defaults(func=cmd_collect_starter)

    collect_inputs = subparsers.add_parser('collect-inputs', help='Collect a custom folder or file into a visual input manifest.')
    collect_inputs.add_argument('--output-dir', default='data/visual_geometry_3d_cli_run')
    collect_inputs.add_argument('--input-dir', default='')
    collect_inputs.add_argument('--inputs', nargs='*', default=[])
    collect_inputs.set_defaults(func=cmd_collect_inputs)

    train = subparsers.add_parser('train', help='Train visual geometry concept/operator stores.')
    train.add_argument('--output-dir', default='data/visual_geometry_3d_cli_run')
    train.add_argument('--input-dir', default='')
    train.add_argument('--inputs', nargs='*', default=[])
    train.add_argument('--limit', type=int, default=None)
    train.add_argument('--eval-input', default='')
    train.set_defaults(func=cmd_train)

    reconstruct = subparsers.add_parser('reconstruct', help='Reconstruct one image or diagram into a 3D bundle.')
    reconstruct.add_argument('--query', default='Reconstruct the visible geometry and topology into a simple 3D scene.')
    reconstruct.add_argument('--visual-input', required=True)
    reconstruct.add_argument('--output-dir', default='data/visual_geometry_3d_cli_run')
    reconstruct.set_defaults(func=cmd_reconstruct)

    batch = subparsers.add_parser('reconstruct-batch', help='Batch reconstruct a folder or list of visual inputs.')
    batch.add_argument('--query', default='Reconstruct the visible geometry and topology into a simple 3D scene.')
    batch.add_argument('--output-dir', default='data/visual_geometry_3d_cli_run')
    batch.add_argument('--input-dir', default='')
    batch.add_argument('--inputs', nargs='*', default=[])
    batch.add_argument('--limit', type=int, default=None)
    batch.set_defaults(func=cmd_reconstruct_batch)

    bootcamp = subparsers.add_parser('bootcamp', help='Collect, train, and reconstruct in one pass.')
    bootcamp.add_argument('--query', default='Reconstruct the visible geometry and topology into a simple 3D scene.')
    bootcamp.add_argument('--output-dir', default='data/visual_geometry_3d_cli_run')
    bootcamp.add_argument('--input-dir', default='')
    bootcamp.add_argument('--inputs', nargs='*', default=[])
    bootcamp.add_argument('--visual-input', default='')
    bootcamp.add_argument('--limit', type=int, default=3)
    bootcamp.add_argument('--eval-input', default='')
    bootcamp.set_defaults(func=cmd_bootcamp)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

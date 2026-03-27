from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from semop import WorldModelMathTrainer, ensure_starter_math_cases


def main() -> None:
    parser = argparse.ArgumentParser(description='Train and evaluate the local math world-model bundle')
    parser.add_argument('--input', default='examples/math_world_model_starter.jsonl')
    parser.add_argument('--output-dir', default='data/math_world_model_gui_run')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--bootstrap-starter', action='store_true')
    parser.add_argument('--concept-store', default='data/vlso_visual_prototypes.db')
    parser.add_argument('--operator-store', default='data/vlso_visual_operators.db')
    parser.add_argument('--affordance-weights', default='data/vlso_samples/trained_affordance_weights.json')
    parser.add_argument('--hardware-profile', default='auto')
    args = parser.parse_args()

    input_path = ensure_starter_math_cases(args.input) if args.bootstrap_starter else args.input
    trainer = WorldModelMathTrainer(
        concept_store_path=args.concept_store,
        operator_store_path=args.operator_store,
        affordance_weights_path=args.affordance_weights,
        hardware_profile=args.hardware_profile,
    )
    summary = trainer.train_from_cases(
        input_path,
        args.output_dir,
        epochs=max(1, args.epochs),
        bootstrap_starter=args.bootstrap_starter,
    )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

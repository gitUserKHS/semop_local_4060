from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from semop import AffordanceWeightTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description='Retrain VLSO affordance classifier weights from a JSONL label set')
    parser.add_argument('--labels', required=True, help='label JSONL path')
    parser.add_argument('--weights', help='starting weights JSON path')
    parser.add_argument('--output', required=True, help='output weights JSON path')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.18)
    args = parser.parse_args()

    trainer = AffordanceWeightTrainer(weights_path=args.weights, epochs=args.epochs, learning_rate=args.learning_rate)
    summary = trainer.train_jsonl(args.labels, output_path=args.output)
    print(f'examples={summary.examples}')
    print(f'targets={summary.targets}')
    print(f'classes_updated={summary.classes_updated}')
    print(f'output={summary.output_path}')


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import CorpusMemoryStore, ScriptCompatibilityTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a lightweight script compatibility model from SemOp premise/script/operator memories.')
    parser.add_argument('--memory-store', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source')
    parser.add_argument('--split', default='train')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--learning-rate', type=float, default=0.18)
    args = parser.parse_args()

    summary = ScriptCompatibilityTrainer().train_from_memory(
        CorpusMemoryStore(args.memory_store),
        output_path=args.output,
        source=args.source,
        split=args.split,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

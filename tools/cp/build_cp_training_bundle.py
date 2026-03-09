from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import CpTrainingBundleBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged train/val bundles for CP parser training")
    parser.add_argument("--inputs", nargs="*", default=[], help="existing CP DSL jsonl files")
    parser.add_argument("--episode-store", help="optional SQLite episode store to merge")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--train-sft-output")
    parser.add_argument("--val-sft-output")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--allow-failed-episodes", action="store_true")
    args = parser.parse_args()

    builder = CpTrainingBundleBuilder()
    bundle = builder.build(
        dataset_paths=args.inputs,
        episode_store_path=args.episode_store,
        allow_failed_episodes=args.allow_failed_episodes,
        train_ratio=args.train_ratio,
    )
    builder.save_bundle(
        bundle,
        train_output=args.train_output,
        val_output=args.val_output,
        train_sft_output=args.train_sft_output,
        val_sft_output=args.val_sft_output,
    )
    print(json.dumps({
        "train_examples": len(bundle.train_examples),
        "val_examples": len(bundle.val_examples),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


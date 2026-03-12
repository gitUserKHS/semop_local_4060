from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import OperatorTrainConfig, OperatorTrainingScaffold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train or dry-run a generic operator student model on teacher trace SFT data.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--train-jsonl', help='defaults to <workspace>/operator_train_sft.jsonl')
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=768)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--use-lora', action='store_true')
    parser.add_argument('--use-qlora', action='store_true')
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--resume-from-checkpoint')
    parser.add_argument('--save-steps', type=int, default=25)
    parser.add_argument('--save-total-limit', type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace)
    train_jsonl = args.train_jsonl or str(workspace / 'operator_train_sft.jsonl')
    summary = OperatorTrainingScaffold().run(OperatorTrainConfig(
        model_name_or_path=args.model,
        output_dir=str(workspace / 'training_run'),
        train_jsonl=train_jsonl,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        dry_run=args.dry_run,
        local_files_only=args.local_files_only,
        use_lora=args.use_lora,
        use_qlora=args.use_qlora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    ))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop.cp_experiment import CpLoraExperimentConfig, CpLoraExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a bundle -> train -> compare CP LoRA experiment')
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--inputs', nargs='*', default=[])
    parser.add_argument('--episode-store')
    parser.add_argument('--eval-inputs', nargs='*', default=[])
    parser.add_argument('--train-ratio', type=float, default=0.9)
    parser.add_argument('--allow-failed-episodes', action='store_true')
    parser.add_argument('--execute-train', action='store_true')
    parser.add_argument('--dry-run-train', action='store_true')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-lora', action='store_true')
    parser.add_argument('--resume-from-checkpoint')
    parser.add_argument('--save-steps', type=int, default=25)
    parser.add_argument('--save-total-limit', type=int, default=2)
    args = parser.parse_args()

    summary = CpLoraExperimentRunner().run(CpLoraExperimentConfig(
        workspace=args.workspace,
        model_name_or_path=args.model,
        dataset_paths=args.inputs,
        episode_store_path=args.episode_store,
        eval_dataset_paths=args.eval_inputs or None,
        train_ratio=args.train_ratio,
        allow_failed_episodes=args.allow_failed_episodes,
        execute_train=args.execute_train,
        dry_run_train=args.dry_run_train,
        local_files_only=args.local_files_only,
        use_lora=not args.no_lora,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    ))
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

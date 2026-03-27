from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import CpParserTrainConfig, CpParserTrainingScaffold


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or dry-run a small CP parser model for statement -> DSL/frame/sketch")
    parser.add_argument("--model", required=True, help="local model path or Hugging Face model id")
    parser.add_argument("--output-dir", required=True, help="training output directory")
    parser.add_argument("--train-jsonl", help="CP DSL jsonl produced by build_cp_dsl_dataset.py")
    parser.add_argument("--episode-store", help="optional SQLite episode store created by solve_contest.py")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="prepare the dataset and training plan without loading the model")
    parser.add_argument("--local-files-only", action="store_true", help="only load tokenizer/model from local cache or local path")
    parser.add_argument("--allow-failed-episodes", action="store_true", help="include failed compile/validation episodes in the training set")
    parser.add_argument("--use-lora", action="store_true", help="wrap the parser model with a LoRA adapter")
    parser.add_argument("--use-qlora", action="store_true", help="request 4-bit QLoRA when bitsandbytes and CUDA are available")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", help="optional LoRA target modules such as q_proj k_proj v_proj o_proj")
    parser.add_argument("--resume-from-checkpoint", help="resume training from a checkpoint directory")
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--hardware-profile", default="auto", help="auto, rtx_4060_8gb, cuda_low_vram, cuda_general, or cpu_only")
    parser.add_argument("--disable-auto-hw-tune", action="store_true", help="disable automatic RTX 4060 / low-VRAM training safeguards")
    args = parser.parse_args()

    if not args.train_jsonl and not args.episode_store:
        parser.error("Provide either --train-jsonl or --episode-store.")

    config = CpParserTrainConfig(
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        train_jsonl=args.train_jsonl,
        episode_store_path=args.episode_store,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        dry_run=args.dry_run,
        local_files_only=args.local_files_only,
        allow_failed_episodes=args.allow_failed_episodes,
        use_lora=args.use_lora,
        use_qlora=args.use_qlora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        hardware_profile=args.hardware_profile,
        auto_configure_for_local_gpu=not args.disable_auto_hw_tune,
    )
    summary = CpParserTrainingScaffold().run(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


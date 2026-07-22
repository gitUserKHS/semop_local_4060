from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.semantic_student_training import train_semantic_student


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LoRA-train Qwen3.5-0.8B on independently reviewed, replay-verified "
            "SemOp semantic traces. This writes a candidate only; it never activates it."
        )
    )
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=4060)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        summary = train_semantic_student(
            args.corpus,
            args.output,
            epochs=args.epochs,
            max_records=args.max_records,
            device=args.device,
            seed=args.seed,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"semantic student training failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


# Backward-compatible import for experiments that used the old script function.
train_student = train_semantic_student


if __name__ == "__main__":
    raise SystemExit(main())

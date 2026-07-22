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

from semop.verified_semantic_curriculum import build_verified_arithmetic_curriculum


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a development-only Korean arithmetic grounding curriculum "
            "whose expressions are replayed by the exact typed executor."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/distillation/verified-arithmetic-v1"),
    )
    parser.add_argument("--seed", type=int, default=4060)
    parser.add_argument("--train-per-template", type=int, default=8)
    parser.add_argument("--eval-per-template", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        curriculum = build_verified_arithmetic_curriculum(
            seed=args.seed,
            train_examples_per_template=args.train_per_template,
            evaluation_examples_per_template=args.eval_per_template,
        )
        manifest = curriculum.save(args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"verified semantic curriculum generation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

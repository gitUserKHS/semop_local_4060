from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import VlsoGroundedEvaluator, VLSOReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grounded VLSO QA cases")
    parser.add_argument("--input", required=True, help="VLSO eval jsonl path")
    parser.add_argument("--mode", choices=["heuristic", "hybrid", "deep"], default="deep")
    parser.add_argument("--language-mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--visual-store")
    parser.add_argument("--vision-backbone")
    parser.add_argument("--vision-model-path")
    parser.add_argument("--affordance-weights")
    parser.add_argument("--concept-store")
    parser.add_argument("--operator-store")
    parser.add_argument("--answer-mode", choices=["structured", "llm"], default="structured")
    parser.add_argument("--answer-model-id", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    reasoner = VLSOReasoner(
        mode=args.mode,
        language_mode=args.language_mode,
        visual_store_path=args.visual_store,
        vision_model_id=args.vision_backbone,
        vision_model_path=args.vision_model_path,
        affordance_weights_path=args.affordance_weights,
        concept_store_path=args.concept_store,
        operator_store_path=args.operator_store,
        answer_mode=args.answer_mode,
        answer_model_id=args.answer_model_id,
    )
    evaluator = VlsoGroundedEvaluator(reasoner)
    cases = evaluator.load_cases(args.input)
    summary = evaluator.evaluate_cases(cases)
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

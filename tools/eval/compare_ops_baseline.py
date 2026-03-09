from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import BASELINE_SPECS, DomainCopilot, LabeledOpsEvaluator, load_labeled_ops_cases


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Compare SemOp domain copilot against configurable baselines on labeled ops cases")
    parser.add_argument("--input", required=True, help="jsonl labeled cases")
    parser.add_argument("--baseline", choices=sorted(BASELINE_SPECS.keys()), default="lexical_rag")
    parser.add_argument("--baseline-config", help="optional JSON config path for configurable baselines")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--memory-store")
    parser.add_argument("--memory-source")
    parser.add_argument("--feedback-rules")
    parser.add_argument("--output")
    args = parser.parse_args()

    copilot = DomainCopilot(
        mode=args.mode,
        model_id=args.model_id,
        memory_store_path=args.memory_store,
        memory_source=args.memory_source,
        feedback_rules_path=args.feedback_rules,
    )
    evaluator = LabeledOpsEvaluator(copilot=copilot, baseline_name=args.baseline, baseline_config_path=args.baseline_config)
    cases = load_labeled_ops_cases(args.input)

    comparisons = []
    systems = {"semop": [], args.baseline: []}
    for case in cases:
        pair = evaluator.evaluate_case(case)
        comparisons.append({name: result.model_dump() for name, result in pair.items()})
        for name in systems:
            systems[name].append(pair[name].model_dump())

    summary = {
        "case_count": len(cases),
        "baseline": args.baseline,
        "baseline_config": args.baseline_config,
        "systems": {
            name: {
                "avg_relation_recall": round(mean(item["relation_recall"] for item in rows), 3),
                "avg_answer_term_recall": round(mean(item["answer_term_recall"] for item in rows), 3),
                "forbidden_phrase_hit_rate": round(sum(1 for item in rows if item["forbidden_phrase_hit"]) / len(rows), 3),
                "clarification_alignment": round(mean(item["clarification_alignment"] for item in rows), 3),
            }
            for name, rows in systems.items()
        },
        "comparisons": comparisons,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary["systems"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop import StructuredMeaningPipeline


def evaluate_case(graph, expected_relations, forbidden_advice):
    relation_set = {(e.source, e.relation, e.target) for e in graph.edges}
    recovered = sum(1 for rel in expected_relations if tuple(rel) in relation_set)
    relation_recovery = recovered / max(1, len(expected_relations))

    final_answer_text = " ".join(step.action for step in graph.plan if step.status == "valid")
    forbidden_hit = any(text in final_answer_text for text in forbidden_advice)
    invalid_advice_rate = 1.0 if forbidden_hit else 0.0

    executable = sum(1 for step in graph.plan if step.status == "valid")
    plan_executability = executable / max(1, len(graph.plan))
    return {
        "relation_recovery": relation_recovery,
        "invalid_advice_rate": invalid_advice_rate,
        "plan_executability": plan_executability,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--cases", default="examples/eval_cases_ko.jsonl")
    args = parser.parse_args()

    pipeline = StructuredMeaningPipeline(mode=args.mode, model_id=args.model_id)
    cases_path = Path(args.cases)
    metrics = []

    with cases_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            graph = pipeline.run(case["query"])
            result = evaluate_case(graph, case["expected_relations"], case["forbidden_advice"])
            result["query"] = case["query"]
            metrics.append(result)

    mean = {
        "relation_recovery": sum(m["relation_recovery"] for m in metrics) / len(metrics),
        "invalid_advice_rate": sum(m["invalid_advice_rate"] for m in metrics) / len(metrics),
        "plan_executability": sum(m["plan_executability"] for m in metrics) / len(metrics),
        "cases": len(metrics),
    }
    print(json.dumps({"summary": mean, "details": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


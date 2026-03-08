from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from semop import CopilotRequest, DomainCopilot


KPI_FIELDS = [
    "invalid_advice_rate",
    "plan_executability",
    "missing_prerequisite_rate",
    "context_misread_rate",
    "relation_recovery",
    "human_audit_usefulness",
    "clarification_need_rate",
]


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_cases(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Evaluate operations KPI metrics on SOP QA cases")
    parser.add_argument("--input", required=True, help="jsonl file with query/context/domain fields")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--memory-store")
    parser.add_argument("--memory-source")
    parser.add_argument("--output", help="optional json output path")
    args = parser.parse_args()

    copilot = DomainCopilot(
        mode=args.mode,
        model_id=args.model_id,
        memory_store_path=args.memory_store,
        memory_source=args.memory_source,
    )

    cases = load_cases(args.input)
    results = []
    for case in cases:
        request = CopilotRequest(
            query=case["query"],
            context=case.get("context", ""),
            domain=case.get("domain", "general"),
            scenario=case.get("scenario", "qa"),
        )
        result = copilot.run(request)
        payload = result.model_dump()
        payload["case_id"] = case.get("id", request.query[:40])
        results.append(payload)

    summary = {
        "case_count": len(results),
        "averages": {field: round(mean(item["kpis"][field] for item in results), 3) for field in KPI_FIELDS},
        "results": results,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary["averages"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

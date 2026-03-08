from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from semop import DOMAIN_PROFILES, CopilotRequest, DomainCopilot


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _load_context(args: argparse.Namespace) -> str:
    if args.context_file:
        with open(args.context_file, "r", encoding="utf-8-sig") as handle:
            return handle.read()
    return args.context or ""


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Operations and logistics SOP copilot")
    parser.add_argument("--query", required=True, help="question to ask the copilot")
    parser.add_argument("--domain", choices=sorted(DOMAIN_PROFILES.keys()), default="warehouse_exception")
    parser.add_argument("--scenario", default="qa", help="scenario tag for audit trace")
    parser.add_argument("--context", help="inline SOP or manual context")
    parser.add_argument("--context-file", help="path to a text or markdown SOP document")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--memory-store", help="optional sqlite memory store path")
    parser.add_argument("--memory-source", help="optional memory source filter")
    parser.add_argument("--review-queue", help="optional sqlite review queue path")
    parser.add_argument("--feedback-rules", help="optional learned feedback rules json path")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    copilot = DomainCopilot(
        mode=args.mode,
        model_id=args.model_id,
        memory_store_path=args.memory_store,
        memory_source=args.memory_source,
        review_queue_path=args.review_queue,
        feedback_rules_path=args.feedback_rules,
    )
    request = CopilotRequest(
        query=args.query,
        context=_load_context(args),
        domain=args.domain,
        scenario=args.scenario,
    )
    result = copilot.run(request)

    if args.format == "json":
        print(result.model_dump_json(indent=2, ensure_ascii=False))
        return

    print(result.to_text())


if __name__ == "__main__":
    main()

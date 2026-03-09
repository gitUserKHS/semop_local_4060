from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from semop import HardProblemEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a hard problem with structured reasoning, verification, and pattern learning")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--memory-store")
    parser.add_argument("--memory-source")
    parser.add_argument("--logical-weight-path", default="data/logical_pattern_weights.json")
    parser.add_argument("--learn", action="store_true", help="update logical pattern weights from the result")
    parser.add_argument("--success", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    engine = HardProblemEngine(
        mode=args.mode,
        memory_store_path=args.memory_store,
        memory_source=args.memory_source,
        logical_weight_path=args.logical_weight_path,
    )
    report = engine.solve(args.query)

    if args.learn:
        success = None if args.success == "auto" else (args.success == "true")
        engine.learn_from_report(report, success=success)

    if args.format == "json":
        print(report.to_json(indent=2))
        return

    print(f"Solved: {report.solved}")
    print(f"Verification score: {report.verification_score:.2f}")
    print(f"Chosen answer: {report.chosen_answer}")
    if report.matched_patterns:
        print("Matched patterns:")
        for item in report.matched_patterns:
            print(f"- {item}")
    print("Checks:")
    for check in report.checks:
        print(f"- {check.name}: passed={check.passed} score={check.score:.2f} detail={check.detail}")


if __name__ == "__main__":
    main()

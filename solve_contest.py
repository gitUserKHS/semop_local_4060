from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from semop import CompetitiveProgrammingReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a competitive-programming approach, complexity analysis, and C++17 code")
    parser.add_argument("--query", required=True, help="contest problem statement or summary")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--episode-store", help="optional SQLite path for storing solved CP episodes")
    args = parser.parse_args()

    result = CompetitiveProgrammingReasoner(episode_store_path=args.episode_store).solve(args.query)
    if result is None:
        print("No contest-programming plan was generated.")
        return

    if args.format == "json":
        print(result.model_dump_json(indent=2, ensure_ascii=False))
        return

    print(f"Category: {result.category}")
    print(f"Time complexity: {result.time_complexity}")
    print(f"Memory complexity: {result.memory_complexity}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Compile check: {'ok' if result.compile_ok else 'failed'}")
    if result.repaired:
        print("Repair loop: applied")
        if result.repair_attempts:
            print("Repair attempts:")
            for item in result.repair_attempts:
                print(f"- {item['rule_id']}: applied={item['applied']}")
    if result.cues:
        print("Cues:")
        for cue in result.cues:
            print(f"- {cue}")
    if result.hidden_concepts:
        print("Hidden concepts:")
        for item in result.hidden_concepts:
            print(f"- {item}")
    if result.logical_frames:
        print("Logical frames:")
        for item in result.logical_frames:
            print(f"- {item}")
    if result.goal_types:
        print("Goal types:")
        for item in result.goal_types:
            print(f"- {item}")
    if result.domain_tags:
        print("Domain tags:")
        for item in result.domain_tags:
            print(f"- {item}")
    if result.dsl_operators:
        print("DSL operators:")
        for item in result.dsl_operators:
            print(f"- {item}")
    if result.extracted_constraints:
        print("Extracted constraints:")
        for item in result.extracted_constraints:
            print(f"- {item}")
    if result.reasoning_steps:
        print("Reasoning steps:")
        for step in result.reasoning_steps:
            print(f"- {step}")
    if result.memory_projection:
        print("Memory projection:")
        for layer, payload in result.memory_projection.items():
            print(f"- {layer}: {payload}")
    if result.validation_report:
        print("Validation report:")
        for key, value in result.validation_report.items():
            print(f"- {key}: {value}")
    print()
    print("Approach:")
    print(result.approach)
    if result.knowledge_sources:
        print()
        print("Knowledge sources:")
        for source in result.knowledge_sources:
            print(f"- {source}")
    if not result.compile_ok and result.compile_stderr:
        print()
        print("Compiler stderr:")
        print(result.compile_stderr)
    print()
    print("C++17 Code:")
    print(result.cpp_code)


if __name__ == "__main__":
    main()

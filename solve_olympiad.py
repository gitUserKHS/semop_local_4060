from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from semop import OlympiadReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the symbolic olympiad proof-search prototype")
    parser.add_argument("--query", required=True, help="proof-style math question")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    result = OlympiadReasoner().solve(args.query)
    if result is None:
        print("No olympiad-style proof sketch was produced for this query yet.")
        return

    if args.format == "json":
        print(result.model_dump_json(indent=2, ensure_ascii=False))
        return

    print(result.answer)
    if result.evidence:
        print()
        print("Proof outline:")
        for item in result.evidence:
            print(f"- {item}")
    if result.equations:
        print()
        print("Operator trace:")
        for item in result.equations:
            print(f"- {item}")


if __name__ == "__main__":
    main()

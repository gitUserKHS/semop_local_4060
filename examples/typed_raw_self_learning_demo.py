from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    LearningSplit,
    RawLearningExample,
    RawSelfLearningLoop,
    SelfLearningLoop,
    TypedDomainRequest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Learn a verified operator policy directly from raw text"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    training = (
        _example(
            "deploy",
            "Goal: deploy; Requires: tests, approval; "
            "Satisfied: tests; Satisfied: approval",
        ),
        _example(
            "publish",
            "Goal: publish; Requires: review, license; "
            "Satisfied: review; Satisfied: license",
        ),
    )
    heldout = (
        _example(
            "launch",
            "Goal: launch; Requires: audit, signoff; "
            "Satisfied: audit; Satisfied: signoff",
            split=LearningSplit.HELDOUT,
        ),
        _example(
            "release-control",
            "Goal: release; Requires: audit, signoff; Satisfied: audit",
            split=LearningSplit.HELDOUT,
            expected_solved=False,
        ),
    )
    result = RawSelfLearningLoop(
        SelfLearningLoop(store=args.output) if args.output else SelfLearningLoop()
    ).run(training, heldout, namespace="raw-demo")
    iteration = result.learning.iterations[-1]

    print(f"grounding_complete={result.grounding.training.complete}")
    print(f"leakage_free={result.grounding.leakage_free}")
    print(f"verified_traces={iteration.verified_training_traces}")
    print(f"decision_cases={iteration.decision_cases}")
    print(f"promoted={result.promoted}")
    print(f"expansion_reduction={iteration.expansion_reduction:.3f}")
    if iteration.rejection_reasons:
        print("rejection_reasons=" + ",".join(iteration.rejection_reasons))
    return 0 if result.promoted else 2


def _example(
    example_id: str,
    text: str,
    *,
    split: LearningSplit = LearningSplit.TRAIN,
    expected_solved: bool = True,
) -> RawLearningExample:
    return RawLearningExample(
        example_id,
        TypedDomainRequest("language", text, "shadow"),
        expected_solved=expected_solved,
        split=split,
        capability="raw-requirement-reasoning",
    )


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    LearningSplit,
    SelfLearningBudget,
    SelfLearningLoop,
    SelfLearningStore,
    SolveBudget,
    generate_symbolic_curriculum,
    generate_symbolic_negative_controls,
    learning_tasks_from_synthetic,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one verifier-gated language/math/vision learning cycle."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/typed_self_learning_demo"),
    )
    parser.add_argument("--examples-per-domain", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--min-expansion-reduction", type=float, default=0.10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    training_problems = generate_symbolic_curriculum(
        args.examples_per_domain,
        seed=args.seed,
        curriculum="language-math-vision",
    )
    heldout_problems = generate_symbolic_curriculum(
        args.examples_per_domain,
        seed=args.seed + 10_000,
        curriculum="language-math-vision",
    )
    negative_controls = generate_symbolic_negative_controls(heldout_problems)
    training = learning_tasks_from_synthetic(
        training_problems,
        split=LearningSplit.TRAIN,
        namespace=f"train-seed-{args.seed}",
    )
    heldout = learning_tasks_from_synthetic(
        heldout_problems,
        split=LearningSplit.HELDOUT,
        namespace=f"heldout-seed-{args.seed}",
    ) + learning_tasks_from_synthetic(
        negative_controls,
        split=LearningSplit.HELDOUT,
        namespace=f"control-seed-{args.seed}",
        expected_solved=False,
    )

    result = SelfLearningLoop(
        budget=SelfLearningBudget(
            solve_budget=SolveBudget(
                max_expansions=2_000,
                timeout_seconds=5.0,
            ),
            min_expansion_reduction=args.min_expansion_reduction,
        ),
        store=SelfLearningStore(args.output),
    ).run(training, heldout, resume=args.resume)
    iteration = result.iterations[-1]
    candidate = iteration.candidate_metrics
    summary = {
        "promoted": iteration.accepted,
        "generation": iteration.generation_after,
        "rejection_reasons": iteration.rejection_reasons,
        "verified_training_traces": iteration.verified_training_traces,
        "decision_cases": iteration.decision_cases,
        "policy_parameters": iteration.parameter_count,
        "policy_artifact_bytes": iteration.artifact_bytes,
        "baseline_positive_expansions": (
            iteration.baseline_metrics.positive_expansions
        ),
        "candidate_positive_expansions": (
            candidate.positive_expansions if candidate is not None else None
        ),
        "expansion_reduction": iteration.expansion_reduction,
        "verified_solve_rate": (
            candidate.verified_solve_rate if candidate is not None else None
        ),
        "proof_soundness": (
            candidate.proof_soundness if candidate is not None else None
        ),
        "false_positives": (
            candidate.false_positives if candidate is not None else None
        ),
        "checkpoint": (
            str(result.checkpoint.manifest_path)
            if result.checkpoint is not None
            else None
        ),
        "macro_candidates": len(result.macro_library.records),
        "macros_active": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    unchanged_resume = (
        args.resume
        and iteration.rejection_reasons == ("candidate_is_identical_to_incumbent",)
    )
    return 0 if iteration.accepted or unchanged_resume else 2


if __name__ == "__main__":
    raise SystemExit(main())

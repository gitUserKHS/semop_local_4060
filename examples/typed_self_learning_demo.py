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
    ActiveCurriculumConfig,
    ActiveCurriculumScheduler,
    ActiveSelfLearningLoop,
    LearningSplit,
    SelfLearningBudget,
    SelfLearningStore,
    SolveBudget,
    generate_lmv_structural_transfer_split,
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
    parser.add_argument(
        "--examples-per-structure",
        "--examples-per-domain",
        dest="examples_per_structure",
        type=int,
        default=3,
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--min-expansion-reduction", type=float, default=0.10)
    args = parser.parse_args()

    split = generate_lmv_structural_transfer_split(
        args.examples_per_structure,
        seed=args.seed,
    )
    training = learning_tasks_from_synthetic(
        split.training,
        split=LearningSplit.TRAIN,
        namespace=f"train-seed-{args.seed}",
    )
    heldout = learning_tasks_from_synthetic(
        split.heldout,
        split=LearningSplit.HELDOUT,
        namespace=f"heldout-seed-{args.seed}",
    ) + learning_tasks_from_synthetic(
        split.negative_controls,
        split=LearningSplit.HELDOUT,
        namespace=f"control-seed-{args.seed}",
        expected_solved=False,
    )

    result = ActiveSelfLearningLoop(
        scheduler=ActiveCurriculumScheduler(
            ActiveCurriculumConfig(max_tasks=6)
        ),
        self_learning_budget=SelfLearningBudget(
            solve_budget=SolveBudget(
                max_expansions=2_000,
                timeout_seconds=5.0,
            ),
            min_expansion_reduction=args.min_expansion_reduction,
        ),
        store=SelfLearningStore(args.output),
    ).run(training, heldout)
    learning_round = result.rounds[-1]
    iteration = learning_round.learning_iteration
    candidate = iteration.candidate_metrics
    summary = {
        "promoted": iteration.accepted,
        "generation": iteration.generation_after,
        "rejection_reasons": iteration.rejection_reasons,
        "verified_training_traces": iteration.verified_training_traces,
        "decision_cases": iteration.decision_cases,
        "training_structures": result.split_audit.training_structures,
        "heldout_structures": result.split_audit.heldout_structures,
        "overlapping_structures": result.split_audit.overlapping_structures,
        "selected_capabilities": [
            item.capability for item in learning_round.selection.decisions
        ],
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
    return 0 if iteration.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())

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
    SelfDiscoveringLearningLoop,
    SelfLearningBudget,
    SelfLearningStore,
    SolveBudget,
    generate_lmv_structural_transfer_split,
    learning_tasks_from_synthetic,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover and learn replay-verified typed operator tasks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/typed_self_discovery_demo"),
    )
    parser.add_argument("--examples-per-structure", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--min-expansion-reduction", type=float, default=0.10)
    args = parser.parse_args()

    split = generate_lmv_structural_transfer_split(
        args.examples_per_structure,
        seed=args.seed,
    )
    training = learning_tasks_from_synthetic(
        split.training,
        split=LearningSplit.TRAIN,
        namespace=f"discovery-train-{args.seed}",
    )
    heldout = learning_tasks_from_synthetic(
        split.heldout,
        split=LearningSplit.HELDOUT,
        namespace=f"discovery-heldout-{args.seed}",
    ) + learning_tasks_from_synthetic(
        split.negative_controls,
        split=LearningSplit.HELDOUT,
        namespace=f"discovery-negative-{args.seed}",
        expected_solved=False,
    )
    result = SelfDiscoveringLearningLoop(
        self_learning_budget=SelfLearningBudget(
            solve_budget=SolveBudget(
                max_expansions=5_000,
                timeout_seconds=10.0,
            ),
            min_expansion_reduction=args.min_expansion_reduction,
        ),
        store=SelfLearningStore(args.output),
    ).run(training, heldout)

    learning_round = result.learning.rounds[0]
    iteration = learning_round.learning_iteration
    candidate = iteration.candidate_metrics
    summary = {
        "training_seeds": len(training),
        "discovered_training_positive": len(
            result.training_discovery.positive_tasks
        ),
        "discovered_training_counterfactual": len(
            result.training_discovery.negative_tasks
        ),
        "discovered_heldout_positive": len(
            result.heldout_discovery.positive_tasks
        ),
        "max_heldout_discovered_depth": max(
            record.proof_depth
            for record in result.heldout_discovery.accepted_records
            if record.expected_solved
        ),
        "selected_domain_counts": dict(learning_round.selection.domain_counts),
        "structural_overlap": result.learning.split_audit.overlapping_structures,
        "program_overlap": result.learning.split_audit.overlapping_programs,
        "promoted": iteration.accepted,
        "rejection_reasons": iteration.rejection_reasons,
        "positive_expansions_before": (
            iteration.baseline_metrics.positive_expansions
        ),
        "positive_expansions_after": (
            candidate.positive_expansions if candidate is not None else None
        ),
        "expansion_reduction": iteration.expansion_reduction,
        "proof_soundness": (
            candidate.proof_soundness if candidate is not None else None
        ),
        "false_positives": (
            candidate.false_positives if candidate is not None else None
        ),
        "checkpoint": (
            str(result.learning.checkpoint.manifest_path)
            if result.learning.checkpoint is not None
            else None
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if iteration.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())

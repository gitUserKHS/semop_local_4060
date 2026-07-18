from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "tools" / "eval"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from evaluate_semantic_grounding_ablation import (  # noqa: E402
    evaluate_semantic_grounding_ablation,
)


def test_ablation_uses_structural_holdout_and_fail_closed_retention() -> None:
    report = evaluate_semantic_grounding_ablation(
        profiles=("full", "no_shared_margin", "primitives_only"),
        label_budgets=(6,),
        validation_per_domain=6,
        seen_test_per_domain=6,
        structural_test_per_domain=10,
        seed=211,
    )

    assert report["profiles"] == ("full", "no_shared_margin", "primitives_only")
    assert report["structural_test_examples_per_domain"] == 10
    assert len(report["runs"]) == 3
    assert report["gates"]["fail_closed_retention"]
    assert report["gates"]["no_checkpoints_written"]
    for run in report["runs"]:
        assert run["checkpoint"] is None
        assert run["gates"]["train_structural_test_disjoint"]
        assert run["gates"]["validation_structural_test_disjoint"]
        assert (
            run["retained_after_structural_gate"]
            == run["gates"]["all_passed"]
        )
        if not run["retained_after_structural_gate"]:
            assert run["deployment_parameters"] == 0
            assert run["structural_test_deployment"]["coverage"] == 0.0

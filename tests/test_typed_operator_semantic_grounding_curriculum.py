from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "tools" / "eval"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from evaluate_semantic_grounding_curriculum import (  # noqa: E402
    evaluate_semantic_grounding_curriculum,
)


def test_curriculum_evaluation_is_disjoint_and_fail_closed() -> None:
    report = evaluate_semantic_grounding_curriculum(
        label_budgets=(4,),
        training_pool_per_domain=16,
        validation_per_domain=12,
        extrapolation_per_domain=12,
        seed=263,
    )

    assert report["profile"] == "primitives_only"
    assert report["strategies"] == ("prefix", "feature_novel")
    assert len(report["runs"]) == 2
    assert report["split_structure"]["training_final_phenomena_disjoint"]
    assert report["gates"]["split_structure_clean"]
    assert report["gates"]["fail_closed_retention"]
    assert report["gates"]["development_split_not_used_for_promotion"]
    assert report["evaluation_status"] == "development_extrapolation_not_sealed"
    assert not report["sealed_final_evaluated"]
    assert report["comparisons"][0]["feature_state_coverage_nonregression"]
    assert isinstance(
        report["comparisons"][0][
            "supported_feature_state_coverage_nonregression"
        ],
        bool,
    )
    for run in report["runs"]:
        assert run["checkpoint"] is None
        assert run["gates"]["train_final_disjoint"]
        assert (
            run["retained_after_final_gate"] == run["gates"]["all_passed"]
        )
        if not run["retained_after_final_gate"]:
            assert run["deployment_parameters"] == 0
            assert run["final_extrapolation_deployment"]["coverage"] == 0.0

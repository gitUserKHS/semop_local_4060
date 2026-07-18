from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "tools" / "eval"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from evaluate_semantic_grounding_learning import (  # noqa: E402
    evaluate_semantic_grounding_learning,
)


def test_real_adapter_curve_keeps_human_claims_withheld() -> None:
    report = evaluate_semantic_grounding_learning(
        label_budgets=(0, 20),
        validation_per_domain=12,
        test_per_domain=18,
        seed=73,
    )

    assert report["domains"] == ("language", "math", "vision")
    assert "production adapters" in report["benchmark_scope"]
    assert not report["human_semantic_gold_evaluated"]
    assert report["gates"]["zero_shot_abstains"]
    assert report["gates"]["candidate_reviews_digest_clean"]
    assert report["gates"]["human_semantic_gate"] == "not_evaluated"
    assert report["gates"]["curated_labels_excluded_from_promotion"]
    learned = report["runs"][1]
    assert learned["promoted"]
    assert learned["programmatic_verified_labels"] == 60
    assert learned["human_reviewed_labels"] == 0
    assert learned["untouched_test"]["false_accepts"] == 0
    assert learned["untouched_test"]["oracle_correct_completion_rate"] >= 0.85
    assert learned["semantic_test"]["used_for_promotion"] is False
    assert learned["semantic_test"]["human_reviewed_targets"] == 0


def test_cli_writes_semantic_grounding_report() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "report.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(EVAL / "evaluate_semantic_grounding_learning.py"),
                "--label-budgets",
                "0,20",
                "--validation-per-domain",
                "12",
                "--test-per-domain",
                "18",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["runs"][1]["promoted"]
        assert payload["candidate_review_audit"]["reviews"] == 0


def test_review_cli_shows_target_without_writing_a_label() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(EVAL / "review_semantic_grounding.py"),
            "--case-id",
            "language-ready-two-requirements",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["case_id"] == "language-ready-two-requirements"
    assert payload["candidate_atom"].startswith("READY(")
    assert len(payload["candidate_digest"]) == 64

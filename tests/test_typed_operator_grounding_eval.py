from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "tools" / "eval"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from evaluate_grounding_self_learning import evaluate_grounding_self_learning


class GroundingSelfLearningEvalTests(unittest.TestCase):
    def test_report_keeps_contract_and_semantic_claims_separate(self) -> None:
        report = evaluate_grounding_self_learning(
            label_budgets=(0, 5),
            validation_per_domain=5,
            test_per_domain=10,
            seed=17,
        )

        self.assertEqual(report["domains"], ("language", "math", "vision"))
        self.assertFalse(report["human_semantic_gold_evaluated"])
        self.assertIn("not open-domain", report["benchmark_scope"])
        self.assertTrue(report["gates"]["zero_shot_abstains"])
        self.assertTrue(report["gates"]["all_learned_budgets_pass"])
        self.assertTrue(report["gates"]["production_adapter_contracts_present"])
        self.assertTrue(report["gates"]["production_adapter_grounding_coverage"])
        self.assertEqual(
            set(report["adapter_probe"]),
            {"language", "math", "vision"},
        )
        learned = report["runs"][1]
        self.assertEqual(learned["human_reviewed_labels"], 0)
        self.assertEqual(learned["synthetic_verified_labels"], 15)
        self.assertEqual(learned["untouched_test"]["false_accepts"], 0)
        self.assertGreaterEqual(learned["untouched_test"]["coverage"], 0.75)

    def test_cli_writes_checkpointed_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            checkpoint = Path(directory) / "checkpoint"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(EVAL / "evaluate_grounding_self_learning.py"),
                    "--label-budgets",
                    "0,5",
                    "--validation-per-domain",
                    "5",
                    "--test-per-domain",
                    "10",
                    "--checkpoint-root",
                    str(checkpoint),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["gates"]["all_learned_budgets_pass"])
            manifest = Path(payload["runs"][1]["checkpoint"])
            self.assertTrue(manifest.is_file())


if __name__ == "__main__":
    unittest.main()

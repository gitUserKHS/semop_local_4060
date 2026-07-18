from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EVAL = ROOT / "tools" / "eval"
for path in (SRC, EVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_lodo_controller_experiment import run_lodo_experiment
from semop.tiny_controller import ControllerFeatureProfile


class LodoControllerExperimentTests(unittest.TestCase):
    def test_debug_artifact_is_metadata_verified_but_not_promotion_ready(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch training profile is not installed")

        with tempfile.TemporaryDirectory() as directory:
            report = run_lodo_experiment(
                directory,
                held_out_domains=("vision",),
                examples_per_domain=1,
                epochs=1,
                debug_small=True,
                max_expansions=2_000,
                feature_profile=ControllerFeatureProfile.TYPED_STRUCTURE,
            )
            saved_report = Path(report["report_path"])
            self.assertTrue(saved_report.exists())

        run = report["runs"]["vision"]
        self.assertEqual(run["trained_domains"], ["language", "math"])
        self.assertEqual(run["evaluation_mode"], "verified_domain_held_out")
        self.assertEqual(run["feature_profile"], "typed_structure")
        self.assertEqual(run["proof_soundness"], 1.0)
        self.assertEqual(run["composed_v4"]["proof_soundness"], 1.0)
        self.assertEqual(run["composed_v4"]["false_positives"], 0)
        self.assertEqual(report["evaluation_status"], "smoke_only")
        self.assertTrue(
            report["aggregate"]["all_holdouts_metadata_verified"]
        )
        self.assertTrue(
            report["aggregate"]["model_artifact_latency_caps_pass"]
        )
        self.assertTrue(
            report["aggregate"]["composed_v4_verified_for_all_holdouts"]
        )
        self.assertFalse(report["aggregate"]["promotion_ready"])

    def test_duplicate_holdout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be unique"):
            run_lodo_experiment(
                ".unused_lodo_test",
                held_out_domains=("vision", "vision"),
                examples_per_domain=1,
                epochs=1,
            )


if __name__ == "__main__":
    unittest.main()

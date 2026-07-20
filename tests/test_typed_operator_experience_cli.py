from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS_EVAL = ROOT / "tools" / "eval"
for path in (SRC, TOOLS_EVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from review_typed_experience import main as review_cli  # noqa: E402
from semop.kernel import (  # noqa: E402
    ExperiencePartitionConfig,
    TypedDomainRequest,
    TypedExperienceStore,
    observation_from_request,
)


class TypedExperienceReviewCliTests(unittest.TestCase):
    def test_custom_partition_queue_can_be_inspected_reviewed_and_exported(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            db = Path(directory) / "typed-experience.db"
            output = Path(directory) / "approved.json"
            store = TypedExperienceStore(
                db,
                partition=ExperiencePartitionConfig(seed="cli-custom-partition"),
            )
            request = TypedDomainRequest(
                "language",
                "Goal: deploy; Requires: tests",
            )
            observation = observation_from_request(
                request,
                event_id="cli-event",
                observed_success=False,
                observed_verified=False,
                proposed_expected_solved=False,
                proposal_authority="frontier_judge",
                trigger="unsolved",
                rationale="The judge proposed that missing tests block deployment.",
                source="cli-test",
                grounded_fingerprint=_digest("cli-grounding"),
                halt_reason="no_solution",
                expansions=2,
                created_at="2026-07-18T11:00:00Z",
            )
            store.add_observation(observation)

            stats_code, stats_out, stats_err = _run_cli(
                "--db",
                str(db),
                "stats",
                "--format",
                "json",
            )
            list_code, list_out, _ = _run_cli(
                "--db",
                str(db),
                "list",
                "--status",
                "pending",
                "--format",
                "json",
            )
            rejected_code, _, rejected_err = _run_cli(
                "--db",
                str(db),
                "review",
                observation.request_digest,
                "--expected",
                "unsolved",
                "--phenomenon",
                "missing_requirement",
                "--rationale",
                "The exact request lacks required test evidence.",
                "--reviewer",
                "human:cli-test",
                "--decision",
                "approved",
            )
            review_code, review_out, review_err = _run_cli(
                "--db",
                str(db),
                "review",
                observation.request_digest,
                "--expected",
                "unsolved",
                "--phenomenon",
                "missing_requirement",
                "--rationale",
                "The exact request lacks required test evidence.",
                "--reviewer",
                "human:cli-test",
                "--decision",
                "approved",
                "--attest-human-review",
            )
            export_code, export_out, export_err = _run_cli(
                "--db",
                str(db),
                "export",
                "--output",
                str(output),
            )
            artifact = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(stats_code, 0, stats_err)
        self.assertEqual(json.loads(stats_out)["requests"], 1)
        self.assertEqual(list_code, 0)
        listed = json.loads(list_out)[0]
        self.assertEqual(listed["status"], "pending")
        self.assertEqual(listed["grounding_uncertainties"], 0)
        self.assertIn("missing tests", listed["latest_rationale"])
        self.assertEqual(rejected_code, 1)
        self.assertIn("--attest-human-review is required", rejected_err)
        self.assertEqual(review_code, 0, review_err)
        self.assertIn("human_reviewed_input_and_expected_outcome", review_out)
        self.assertEqual(export_code, 0, export_err)
        self.assertIn("exported: 1", export_out)
        self.assertEqual(len(artifact["records"]), 1)
        self.assertEqual(
            artifact["partition_fingerprint"],
            store.partition.fingerprint,
        )


def _run_cli(*arguments: str) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = review_cli(arguments)
    return code, stdout.getvalue().strip(), stderr.getvalue().strip()


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
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

from evaluate_semantic_benchmark import main as evaluate_cli  # noqa: E402
from review_semantic_benchmark import main as review_cli  # noqa: E402
from semop.kernel import (  # noqa: E402
    DomainKind,
    LearningSplit,
    RawLearningExample,
    SemanticBenchmark,
    SemanticBenchmarkCase,
    SemanticLabelAuthority,
    SemanticReviewDecision,
    SelfLearningBudget,
    create_semantic_review,
    evaluate_semantic_benchmark,
    load_semantic_benchmark,
    semantic_promotion_rejections,
    write_semantic_review,
)


CASES = ROOT / "data" / "semantic_benchmark" / "v1" / "cases.jsonl"
REVIEWS = ROOT / "data" / "semantic_benchmark" / "v1" / "reviews.jsonl"


class SemanticBenchmarkSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = load_semantic_benchmark(CASES, REVIEWS)

    def test_seed_corpus_covers_language_math_and_vision_without_gold_claim(self) -> None:
        audit = self.benchmark.audit()

        self.assertEqual(audit.cases, 20)
        self.assertEqual(
            dict(audit.domain_case_counts),
            {"language": 6, "math": 6, "vision": 8},
        )
        self.assertEqual(audit.approved_cases, ())
        self.assertEqual(len(audit.pending_cases), 20)
        self.assertEqual(audit.review_coverage, 0.0)
        self.assertTrue(audit.clean)
        self.assertTrue(
            all(
                example.label_authority
                is SemanticLabelAuthority.CURATED_UNREVIEWED
                for example in self.benchmark.raw_examples()
            )
        )

    def test_seed_outcomes_match_all_three_adapters_but_semantic_gold_is_none(self) -> None:
        evaluation = evaluate_semantic_benchmark(self.benchmark)

        self.assertTrue(evaluation.passed)
        self.assertEqual(evaluation.metrics.tasks, 20)
        self.assertEqual(evaluation.metrics.labeled_outcome_accuracy, 1.0)
        self.assertIsNone(evaluation.metrics.programmatic_outcome_accuracy)
        self.assertEqual(evaluation.metrics.curated_unreviewed_accuracy, 1.0)
        self.assertEqual(evaluation.metrics.curated_unreviewed_tasks, 20)
        self.assertIsNone(evaluation.metrics.semantic_correctness)
        self.assertEqual(evaluation.metrics.semantic_gold_tasks, 0)
        self.assertEqual(evaluation.metrics.primitive_replay_integrity, 1.0)
        self.assertEqual(
            {item.domain for item in evaluation.metrics.by_domain},
            {"language", "math", "vision"},
        )
        self.assertTrue(all(outcome.correct for outcome in evaluation.outcomes))


class SemanticReviewBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = load_semantic_benchmark(CASES, REVIEWS)
        self.case = self.seed.case_by_id["language-ready-two-requirements"]

    def test_review_makes_only_the_exact_digest_human_reviewed(self) -> None:
        review = create_semantic_review(
            self.case,
            reviewer="human:test-reviewer",
            decision=SemanticReviewDecision.APPROVED,
            reviewed_at="2026-07-18T12:00:00Z",
        )
        benchmark = SemanticBenchmark(self.seed.cases, (review,))

        self.assertEqual(
            benchmark.label_authority(self.case),
            SemanticLabelAuthority.HUMAN_REVIEWED,
        )
        self.assertEqual(benchmark.audit().approved_cases, (self.case.case_id,))
        evaluation = evaluate_semantic_benchmark(benchmark)
        self.assertEqual(evaluation.metrics.semantic_gold_tasks, 1)
        self.assertEqual(evaluation.metrics.semantic_correctness, 1.0)

    def test_bare_human_review_enum_cannot_manufacture_semantic_gold(self) -> None:
        with self.assertRaisesRegex(ValueError, "digest-bound label evidence"):
            RawLearningExample(
                "forged-human-label",
                self.case.to_request(),
                label_authority=SemanticLabelAuthority.HUMAN_REVIEWED,
            )

    def test_case_change_invalidates_existing_review(self) -> None:
        review = create_semantic_review(
            self.case,
            reviewer="human:test-reviewer",
            decision="approved",
            reviewed_at="2026-07-18T12:00:00+00:00",
        )
        changed = replace(
            self.case,
            rationale=self.case.rationale + " Changed after review.",
        )
        cases = tuple(
            changed if case.case_id == changed.case_id else case
            for case in self.seed.cases
        )
        benchmark = SemanticBenchmark(cases, (review,))
        audit = benchmark.audit()

        self.assertNotEqual(changed.digest, review.case_digest)
        self.assertEqual(audit.stale_review_cases, (changed.case_id,))
        self.assertIn(changed.case_id, audit.pending_cases)
        self.assertEqual(
            benchmark.label_authority(changed),
            SemanticLabelAuthority.CURATED_UNREVIEWED,
        )

    def test_rejected_review_never_becomes_semantic_gold(self) -> None:
        review = create_semantic_review(
            self.case,
            reviewer="human:test-reviewer",
            decision="rejected",
            reviewed_at="2026-07-18T12:00:00Z",
        )
        benchmark = SemanticBenchmark(self.seed.cases, (review,))

        self.assertEqual(benchmark.audit().rejected_cases, (self.case.case_id,))
        self.assertEqual(
            benchmark.label_authority(self.case),
            SemanticLabelAuthority.CURATED_UNREVIEWED,
        )

    def test_review_writer_replaces_one_case_atomically(self) -> None:
        first = create_semantic_review(
            self.case,
            reviewer="human:first",
            decision="rejected",
            reviewed_at="2026-07-18T12:00:00Z",
        )
        second = create_semantic_review(
            self.case,
            reviewer="human:second",
            decision="approved",
            reviewed_at="2026-07-18T13:00:00Z",
        )
        with TemporaryDirectory() as directory:
            target = Path(directory) / "reviews.jsonl"
            write_semantic_review(target, first)
            write_semantic_review(target, second)
            loaded = load_semantic_benchmark(CASES, target)

        self.assertEqual(len(loaded.reviews), 1)
        self.assertEqual(loaded.reviews[0].reviewer, "human:second")
        self.assertEqual(loaded.audit().approved_cases, (self.case.case_id,))

    def test_require_all_reviewed_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unapproved cases"):
            evaluate_semantic_benchmark(self.seed, require_all_reviewed=True)


class SemanticPayloadCodecTests(unittest.TestCase):
    def test_payload_json_is_canonical_and_digest_is_order_independent(self) -> None:
        left = SemanticBenchmarkCase.from_mapping(
            {
                "case_id": "canonical",
                "domain": "math",
                "payload": {"expression": "2 + 2 == 4"},
                "expected_solved": True,
                "phenomenon": "equality",
                "rationale": "Both sides are four.",
            }
        )
        right = SemanticBenchmarkCase.from_mapping(
            {
                "rationale": "Both sides are four.",
                "phenomenon": "equality",
                "expected_solved": True,
                "payload": {"expression": "2 + 2 == 4"},
                "domain": "math",
                "case_id": "canonical",
            }
        )

        self.assertEqual(left, right)
        self.assertEqual(left.digest, right.digest)
        self.assertEqual(left.split, LearningSplit.HELDOUT)

    def test_vision_codec_accepts_names_hex_and_rgb_channels(self) -> None:
        case = SemanticBenchmarkCase.from_mapping(
            {
                "case_id": "vision-color-codec",
                "domain": "vision",
                "payload": {
                    "background": "#ffffff",
                    "rows": [
                        ["white", [255, 255, 255], "#ffffff"],
                        ["white", "red", "white"],
                        ["white", "white", "white"],
                    ],
                    "goals": [
                        {
                            "kind": "count",
                            "selector": "red",
                            "expected": 1,
                        }
                    ],
                },
                "expected_solved": True,
                "phenomenon": "color_codec",
                "rationale": "The raster contains one red component.",
            }
        )

        request = case.to_request()
        self.assertEqual(request.domain, DomainKind.VISION)
        self.assertEqual(request.payload.image.rows[1][1], (255, 0, 0))


class SemanticPromotionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = load_semantic_benchmark(CASES, REVIEWS)
        self.budget = SelfLearningBudget(
            required_semantic_correctness=1.0,
            min_semantic_gold_tasks=3,
            min_semantic_gold_tasks_per_domain=1,
            required_semantic_domains=("language", "math", "vision"),
        )

    def test_unreviewed_seed_fails_global_and_each_domain_gate(self) -> None:
        metrics = evaluate_semantic_benchmark(self.seed).metrics
        reasons = semantic_promotion_rejections(metrics, self.budget)

        self.assertEqual(metrics.semantic_review_coverage, 0.0)
        self.assertIn("semantic_gold_tasks_below_gate: 0 < 3", reasons)
        self.assertIn("semantic_correctness_unavailable", reasons)
        for domain in ("language", "math", "vision"):
            self.assertIn(
                f"semantic_gold_tasks_below_gate[{domain}]: 0 < 1",
                reasons,
            )
            self.assertIn(
                f"semantic_correctness_unavailable[{domain}]",
                reasons,
            )

    def test_one_digest_bound_review_per_domain_passes_gate(self) -> None:
        benchmark = _approve_one_case_per_domain(self.seed)
        metrics = evaluate_semantic_benchmark(benchmark).metrics

        self.assertEqual(semantic_promotion_rejections(metrics, self.budget), ())
        self.assertEqual(metrics.semantic_gold_tasks, 3)
        self.assertEqual(metrics.semantic_correctness, 1.0)
        self.assertEqual(metrics.semantic_review_coverage, 3 / 20)
        self.assertEqual(
            metrics.to_dict()["semantic_review_coverage"],
            3 / 20,
        )
        for domain in metrics.by_domain:
            self.assertEqual(domain.semantic_gold_tasks, 1)
            self.assertEqual(domain.semantic_correctness, 1.0)

    def test_reviewed_wrong_label_fails_global_and_domain_correctness(self) -> None:
        original = self.seed.case_by_id["language-ready-two-requirements"]
        wrong = replace(original, expected_solved=False)
        cases = tuple(
            wrong if case.case_id == wrong.case_id else case
            for case in self.seed.cases
        )
        altered = SemanticBenchmark(cases)
        benchmark = _approve_one_case_per_domain(altered)
        metrics = evaluate_semantic_benchmark(benchmark).metrics
        reasons = semantic_promotion_rejections(metrics, self.budget)

        self.assertEqual(metrics.semantic_correctness, 2 / 3)
        self.assertTrue(
            any(
                reason.startswith("semantic_correctness_below_gate: ")
                for reason in reasons
            )
        )
        self.assertTrue(
            any(
                reason.startswith("semantic_correctness_below_gate[language]: ")
                for reason in reasons
            )
        )

    def test_budget_rejects_invalid_semantic_gate_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "semantic correctness"):
            SelfLearningBudget(required_semantic_correctness=1.01)
        with self.assertRaisesRegex(ValueError, "non-empty and unique"):
            SelfLearningBudget(required_semantic_domains=("language", "language"))


class SemanticBenchmarkCliTests(unittest.TestCase):
    def test_evaluator_json_reports_curated_not_gold(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            code = evaluate_cli(("--format", "json"))
        payload = json.loads(output.getvalue())

        self.assertEqual(code, 0)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["audit"]["review_coverage"], 0.0)
        self.assertIsNone(payload["metrics"]["semantic_correctness"])
        self.assertFalse(payload["promotion_gate"]["enabled"])

    def test_evaluator_promotion_gate_fails_closed_on_unreviewed_seed(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            code = evaluate_cli(
                (
                    "--format",
                    "json",
                    "--gate-semantic-correctness",
                    "1.0",
                    "--gate-min-gold",
                    "3",
                    "--gate-min-gold-per-domain",
                    "1",
                    "--gate-domains",
                    "language,math,vision",
                )
            )
        payload = json.loads(output.getvalue())

        self.assertEqual(code, 2)
        self.assertTrue(payload["promotion_gate"]["enabled"])
        self.assertFalse(payload["promotion_gate"]["passed"])
        self.assertIn(
            "semantic_correctness_unavailable",
            payload["promotion_gate"]["rejection_reasons"],
        )

    def test_evaluator_rejects_negative_gate_counts(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            code = evaluate_cli(
                ("--format", "json", "--gate-min-gold", "-1")
            )
        payload = json.loads(output.getvalue())

        self.assertEqual(code, 1)
        self.assertEqual(payload["error_type"], "ValueError")
        self.assertIn("must not be negative", payload["error"])

    def test_review_cli_requires_explicit_human_attestation(self) -> None:
        error = StringIO()
        with TemporaryDirectory() as directory, redirect_stderr(error):
            target = Path(directory) / "reviews.jsonl"
            code = review_cli(
                (
                    "--reviews",
                    str(target),
                    "--case-id",
                    "language-ready-two-requirements",
                    "--reviewer",
                    "human:test-reviewer",
                    "--decision",
                    "approved",
                )
            )

        self.assertEqual(code, 1)
        self.assertIn("--attest-human-review is required", error.getvalue())

    def test_review_cli_writes_digest_bound_record_after_attestation(self) -> None:
        output = StringIO()
        with TemporaryDirectory() as directory, redirect_stdout(output):
            target = Path(directory) / "reviews.jsonl"
            code = review_cli(
                (
                    "--reviews",
                    str(target),
                    "--case-id",
                    "language-ready-two-requirements",
                    "--reviewer",
                    "human:test-reviewer",
                    "--decision",
                    "approved",
                    "--attest-human-review",
                )
            )
            loaded = load_semantic_benchmark(CASES, target)

        self.assertEqual(code, 0)
        self.assertEqual(len(loaded.audit().approved_cases), 1)
        self.assertIn("digest:", output.getvalue())


def _approve_one_case_per_domain(
    benchmark: SemanticBenchmark,
) -> SemanticBenchmark:
    reviews = []
    for domain in (DomainKind.LANGUAGE, DomainKind.MATH, DomainKind.VISION):
        case = next(item for item in benchmark.cases if item.domain is domain)
        reviews.append(
            create_semantic_review(
                case,
                reviewer=f"human:test-{domain.value}",
                decision="approved",
                reviewed_at="2026-07-18T12:00:00Z",
            )
        )
    return SemanticBenchmark(benchmark.cases, tuple(reviews))


if __name__ == "__main__":
    unittest.main()

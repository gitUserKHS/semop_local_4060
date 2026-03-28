from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop.grounding_self_evolution import GroundingSelfEvolutionRunner
from semop.unified_world_solver_guidance import UnifiedWorldSolverGuidanceEngine
from semop.unified_benchmark import GroundedExplanationEvalCase


class GroundingSelfEvolutionTests(unittest.TestCase):
    def test_runner_writes_report_and_approves_corrected_trace(self) -> None:
        workspace = Path('data') / 'grounding_self_evolution_test'
        shutil.rmtree(workspace, ignore_errors=True)
        try:
            store_path = workspace / 'store.db'
            review_path = workspace / 'reviews.db'
            output_dir = workspace / 'out'
            output_dir.mkdir(parents=True, exist_ok=True)
            case = GroundedExplanationEvalCase(
                query='Which sentence justifies stopping before movement?',
                source_context='If approval is missing, stop first and verify manager approval before moving.',
                expected_evidence_terms=['stop first', 'verify manager approval'],
                expected_claim_terms=['stop first', 'verify manager approval'],
                domain='warehouse_exception',
                scenario='exception_response',
                severity='high',
                case_weight=1.0,
            )
            score_values = iter([0.15, 0.92])
            def score_side_effect(*args, **kwargs):
                return next(score_values, 0.88)
            with mock.patch.object(
                GroundingSelfEvolutionRunner,
                '_score_graph_against_case',
                side_effect=score_side_effect,
            ):
                summary = GroundingSelfEvolutionRunner().run(
                    store_path,
                    review_path,
                    output_dir,
                    split='train',
                    rounds=1,
                    cases_per_round=1,
                    cases=[case],
                )
            self.assertGreaterEqual(summary.improved_cases, 1)
            self.assertGreaterEqual(summary.approved_reviews, 1)
            self.assertGreaterEqual(summary.stored_graphs, 1)
            self.assertTrue(Path(summary.output_path).exists())
            self.assertTrue(Path(summary.reflection_memory_path).exists())
            self.assertTrue(Path(summary.integrated_world_path).exists())
            self.assertTrue(summary.integrated_world)
            self.assertTrue(summary.integrated_reasoning)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_rank_cases_prefers_shared_world_guidance_matches(self) -> None:
        guided_case = GroundedExplanationEvalCase(
            query='Which check proves approval before movement?',
            source_context='Approval confirmed before movement keeps the route safe.',
            expected_evidence_terms=['approval confirmed'],
            expected_claim_terms=['approval confirmed'],
            domain='warehouse_exception',
            scenario='exception_response',
            case_weight=1.0,
        )
        unrelated_case = GroundedExplanationEvalCase(
            query='What decorative color is visible?',
            source_context='A decorative banner hangs nearby.',
            expected_evidence_terms=['decorative banner'],
            expected_claim_terms=['decorative banner'],
            domain='warehouse_exception',
            scenario='exception_response',
            case_weight=1.0,
        )
        guidance = UnifiedWorldSolverGuidanceEngine().build(
            plan={
                'primary_goal': 'safe movement',
                'hidden_constraints': ['approval confirmed'],
                'operator_algebra_targets': [{'label': 'VERIFY_APPROVAL', 'related_operators': ['REQUIRES', 'BLOCKED_BY']}],
                'retained_operator_candidates': [{'name': 'VERIFY_APPROVAL', 'basis_signature': ['REQUIRES', 'BLOCKED_BY']}],
            },
            reasoning={'evidence': ['approval confirmed'], 'blockers': ['route blocked']},
        )

        ranked = GroundingSelfEvolutionRunner._rank_cases([unrelated_case, guided_case], guidance)

        self.assertEqual(ranked[0].query, guided_case.query)
        variants = GroundingSelfEvolutionRunner._variant_queries(guided_case.query, guidance=guidance)
        self.assertTrue(any('approval confirmed' in item.lower() for item in variants))


if __name__ == '__main__':
    unittest.main()

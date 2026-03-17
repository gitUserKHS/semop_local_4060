from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop.corpus_store import CorpusMemoryStore
from semop.generalization_proof import GeneralizationProofHarness
from semop.operator_evolution import OperatorTransferEvalCase
from semop.pipeline import StructuredMeaningPipeline
from semop.premise_eval import HiddenPremiseEvalCase


class GeneralizationProofHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path('tests/tmp_generalization_proof_test')
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_harness_writes_report_for_single_round(self) -> None:
        store_path = self.root / 'memory.db'
        output_dir = self.root / 'output'
        store = CorpusMemoryStore(store_path)
        pipeline = StructuredMeaningPipeline(mode='heuristic')
        for query, domain in [
            ('The aisle is blocked and approval is still missing. What should I do?', 'warehouse_exception'),
            ('Before I open the drawer and take the tool, what do I verify first?', 'general'),
        ]:
            graph = pipeline.run(query)
            graph.domain = domain
            store.upsert_graph(graph, source='proof_source_round_01', split='train')
            store.upsert_premise_operator_memory(graph, source='proof_source_round_01', split='train')

        hidden_cases = [
            HiddenPremiseEvalCase(
                query='The aisle is blocked and approval is still missing. What should I do?',
                expected_hidden_goals=['maintain_access_goal'],
                expected_required_premises=['manager_approval', 'safe_route_available'],
                expected_satisfied_premises=[],
                expected_missing_premises=['manager_approval'],
                expected_risky_actions=['continue_task'],
                forbidden_premises=[],
                expected_clarification_needed=False,
                domain='warehouse_exception',
                scenario='exception_response',
            )
        ]
        transfer_cases = [
            OperatorTransferEvalCase(query='Open the drawer before taking the tool.', domain='general', expected_operator_names=[], split='train'),
            OperatorTransferEvalCase(query='Check approval before moving through the blocked aisle.', domain='warehouse_exception', expected_operator_names=[], split='test'),
        ]

        summary = GeneralizationProofHarness().run(
            store_path,
            output_dir,
            source_schedule=['proof_source_round_01'],
            rounds=1,
            hidden_premise_cases=hidden_cases,
            transfer_cases=transfer_cases,
        )

        self.assertEqual(len(summary.rounds), 1)
        self.assertTrue(Path(summary.report_path).exists())
        self.assertIn('headline', summary.evidence.model_dump())
        self.assertIn('strong_model_score', summary.evidence.model_dump())
        self.assertIn('goal_tracker', summary.model_dump())

    def test_harness_auto_derives_proof_cases_and_goal_tracker(self) -> None:
        store_path = self.root / 'memory_auto.db'
        output_dir = self.root / 'output_auto'
        store = CorpusMemoryStore(store_path)
        pipeline = StructuredMeaningPipeline(mode='heuristic')
        rows = [
            ('The aisle is blocked and approval is still missing. What should I do?', 'warehouse_exception', 'exception_response', 'proof_auto'),
            ('Before I open the drawer and take the tool, what do I verify first?', 'general', 'access_reasoning', 'proof_auto'),
            ('I am new to this aisle. What should I confirm before opening the cabinet?', 'warehouse_onboarding', 'guided_walkthrough', 'proof_auto'),
        ]
        for query, domain, scenario, source in rows:
            graph = pipeline.run(query)
            graph.domain = domain
            graph.scenario = scenario
            graph.source_context = graph.source_context or 'Confirm access and approval before acting.'
            store.upsert_graph(graph, source=source, split='train')
            store.upsert_premise_operator_memory(graph, source=source, split='train')

        summary = GeneralizationProofHarness().run(
            store_path,
            output_dir,
            source_schedule=['proof_auto'],
            rounds=1,
        )

        self.assertEqual(len(summary.rounds), 1)
        self.assertGreaterEqual(summary.goal_tracker.total_axes, 4)
        self.assertTrue(summary.goal_tracker.domains_seen)
        self.assertIn('domain_coverage_score', summary.evidence.model_dump())
        self.assertGreater(summary.rounds[0].derived_case_counts.get('hidden_premise', 0), 0)
        self.assertGreater(summary.rounds[0].derived_case_counts.get('transfer', 0), 0)


if __name__ == '__main__':
    unittest.main()

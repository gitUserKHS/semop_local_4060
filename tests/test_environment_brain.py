from __future__ import annotations

import os
import shutil
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop.domain_copilot import AuditItem, CopilotRequest, CopilotResult
from semop.environment_brain import EnvironmentBrainRunner
from semop.ops_kpi import OpsKpiReport
from semop.structures import ClaimGrounding, Edge, Node, OperatorCandidate, OperatorExecutionReport, PlanStep, StructuredMeaningGraph


class EnvironmentBrainRunnerTests(unittest.TestCase):
    def test_runner_builds_environment_specific_summary(self) -> None:
        temp_dir = os.path.join('tests', 'environment_brain_runner_case')
        os.makedirs(temp_dir, exist_ok=True)
        try:
            store_path = os.path.join(temp_dir, 'env_memory.db')
            output_dir = os.path.join(temp_dir, 'environment_output')

            def fake_run(self, request: CopilotRequest) -> CopilotResult:
                graph = StructuredMeaningGraph(
                    query=request.query,
                    intent='environment_reasoning',
                    domain=request.domain,
                    scenario=request.scenario,
                    source_context=request.context,
                    nodes=[
                        Node(id='approval', label='approval', kind='constraint'),
                        Node(id='route', label='blocked route', kind='path'),
                    ],
                    edges=[Edge(source='question', relation='GROUNDED_BY', target='approval')],
                    inferred_scripts=['verify approval before movement'],
                    induced_operators=[
                        OperatorCandidate(
                            name='VERIFY_BEFORE_MOVE',
                            family='safety_guard',
                            arity=1,
                            input_types=['state'],
                            output_type='decision',
                            description='Verify approval before movement.',
                            confidence=0.88,
                        )
                    ],
                    required_premises=['approval confirmed'],
                    missing_premises=['route clear'],
                    plan=[PlanStep(id='1', action='verify approval before movement', rationale='guard safety')],
                    warnings=['blocked path requires check'],
                    operator_execution=OperatorExecutionReport(
                        composition_score=0.82,
                        claim_groundings=[ClaimGrounding(claim='verify approval before movement', grounded=True, support_kind='edge', supports=['question:GROUNDED_BY:approval'], score=0.84)],
                        claim_grounding_score=0.81,
                    ),
                )
                return CopilotResult(
                    request=request,
                    graph=graph,
                    answer_text='Verify approval before movement.',
                    kpis=OpsKpiReport(
                        invalid_advice_rate=0.0,
                        plan_executability=0.84,
                        missing_prerequisite_rate=0.2,
                        context_misread_rate=0.1,
                        relation_recovery=0.8,
                        human_audit_usefulness=0.77,
                        clarification_need_rate=0.1,
                        notes=['good local fit'],
                    ),
                    audit_items=[AuditItem(stage='graph', detail='environment seeded')],
                )

            fake_training = SimpleNamespace(
                augmented_graph_count=6,
                model_dump=lambda: {
                    'augmented_graph_count': 6,
                    'artifacts': {'continuous_learning_bundle_dir': os.path.join(output_dir, 'environment_bundle', 'continuous_learning_bundle')},
                },
            )

            with patch('semop.environment_brain.DomainCopilot.run', new=fake_run),                  patch('semop.environment_brain.UnifiedSemOpTrainer.train_from_store', return_value=fake_training),                  patch('semop.environment_brain.EnvironmentBrainRunner._auto_review_environment_queries', return_value=(3, 1)):
                summary = EnvironmentBrainRunner().run(
                    environment_name='warehouse aisle',
                    domain='warehouse_exception',
                    scenario='exception_response',
                    context='If the route is blocked, verify approval before movement.',
                    store_path=store_path,
                    review_queue_path=os.path.join(temp_dir, 'reviews.db'),
                    output_dir=output_dir,
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertGreater(summary.mastery_scores.get('environment_mastery', 0.0), 0.0)
        self.assertTrue(summary.stable_concepts)
        self.assertTrue(summary.routine_patterns)
        self.assertEqual(summary.auto_approved_review_count, 3)
        self.assertIn('approval', [item.label for item in summary.stable_concepts])

    def test_runner_uses_shared_world_guidance_queries(self) -> None:
        temp_dir = os.path.join('tests', 'environment_brain_guidance_case')
        os.makedirs(temp_dir, exist_ok=True)
        captured_queries: list[str] = []
        try:
            store_path = os.path.join(temp_dir, 'env_memory.db')
            output_dir = os.path.join(temp_dir, 'environment_output')

            def fake_run(self, request: CopilotRequest) -> CopilotResult:
                captured_queries.append(request.query)
                graph = StructuredMeaningGraph(
                    query=request.query,
                    intent='environment_reasoning',
                    domain=request.domain,
                    scenario=request.scenario,
                    source_context=request.context,
                    nodes=[Node(id='approval', label='approval', kind='constraint')],
                    edges=[Edge(source='question', relation='GROUNDED_BY', target='approval')],
                    required_premises=['approval confirmed'],
                    plan=[PlanStep(id='1', action='verify approval before movement', rationale='guard safety')],
                    operator_execution=OperatorExecutionReport(
                        composition_score=0.82,
                        claim_groundings=[ClaimGrounding(claim='verify approval before movement', grounded=True, support_kind='edge', supports=['question:GROUNDED_BY:approval'], score=0.84)],
                        claim_grounding_score=0.81,
                    ),
                )
                return CopilotResult(
                    request=request,
                    graph=graph,
                    answer_text='Verify approval before movement.',
                    kpis=OpsKpiReport(
                        invalid_advice_rate=0.0,
                        plan_executability=0.84,
                        missing_prerequisite_rate=0.2,
                        context_misread_rate=0.1,
                        relation_recovery=0.8,
                        human_audit_usefulness=0.77,
                        clarification_need_rate=0.1,
                        notes=['good local fit'],
                    ),
                    audit_items=[AuditItem(stage='graph', detail='environment seeded')],
                )

            fake_training = SimpleNamespace(
                augmented_graph_count=4,
                model_dump=lambda: {'augmented_graph_count': 4, 'artifacts': {'continuous_learning_bundle_dir': os.path.join(output_dir, 'environment_bundle', 'continuous_learning_bundle')}},
            )
            with patch('semop.environment_brain.DomainCopilot.run', new=fake_run), patch('semop.environment_brain.UnifiedSemOpTrainer.train_from_store', return_value=fake_training), patch('semop.environment_brain.EnvironmentBrainRunner._auto_review_environment_queries', return_value=(1, 0)):
                summary = EnvironmentBrainRunner().run(
                    environment_name='warehouse aisle',
                    domain='warehouse_exception',
                    scenario='exception_response',
                    context='If the route is blocked, verify approval before movement.',
                    store_path=store_path,
                    review_queue_path=os.path.join(temp_dir, 'reviews.db'),
                    output_dir=output_dir,
                    self_learning_plan={
                        'primary_goal': 'safe movement',
                        'hidden_constraints': ['approval confirmed'],
                        'operator_algebra_targets': [{'label': 'VERIFY_APPROVAL', 'related_operators': ['REQUIRES', 'BLOCKED_BY']}],
                        'retained_operator_candidates': [{'name': 'VERIFY_APPROVAL', 'basis_signature': ['REQUIRES', 'BLOCKED_BY']}],
                        'next_learning_queries': ['Which exact evidence verifies approval confirmed before movement?'],
                    },
                    integrated_reasoning={
                        'blockers': ['route --BLOCKED_BY--> blocked route'],
                        'prerequisites': ['route --REQUIRES--> approval confirmed'],
                        'evidence': ['worker --TARGET_OF_ATTENTION--> approval'],
                    },
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertTrue(any('approval confirmed' in query.lower() for query in captured_queries))
        self.assertTrue(any('VERIFY_APPROVAL' in query for query in captured_queries))
        self.assertTrue(any('Shared-world solver guidance:' in note for note in summary.notes))


if __name__ == '__main__':
    unittest.main()

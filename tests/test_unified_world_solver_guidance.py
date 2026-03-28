from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop.operator_evolution import EvolvedOperatorProposal, OperatorSelfEvolutionEngine
from semop.structures import StructuredMeaningGraph
from semop.unified_world_solver_guidance import UnifiedWorldSolverGuidanceEngine


class UnifiedWorldSolverGuidanceTests(unittest.TestCase):
    def test_build_collects_constraints_queries_and_operator_focus(self) -> None:
        guidance = UnifiedWorldSolverGuidanceEngine().build(
            plan={
                'primary_goal': 'safe movement',
                'hidden_constraints': ['approval confirmed'],
                'operator_algebra_targets': [{'label': 'VERIFY_APPROVAL', 'related_operators': ['REQUIRES', 'BLOCKED_BY']}],
                'operator_functor_targets': [{'label': 'ConstraintFunctor', 'related_operators': ['constraint_binding']}],
                'retained_operator_candidates': [{'name': 'VERIFY_APPROVAL', 'basis_signature': ['REQUIRES', 'BLOCKED_BY']}],
                'next_learning_queries': ['Which check proves approval before movement?'],
            },
            reasoning={
                'blockers': ['route --BLOCKED_BY--> blocked zone'],
                'prerequisites': ['route --REQUIRES--> approval confirmed'],
                'evidence': ['worker --TARGET_OF_ATTENTION--> approval panel'],
            },
            context='If the route is blocked, verify approval before movement.',
            domain='warehouse_exception',
            scenario='exception_response',
        )

        self.assertIn('approval confirmed', guidance.hidden_constraints)
        self.assertIn('VERIFY_APPROVAL', guidance.operator_focus)
        self.assertTrue(any('Which check proves approval before movement?' in item for item in guidance.priority_queries))
        self.assertTrue(any('REQUIRES' in item for item in guidance.basis_focus))
        self.assertTrue(guidance.summary)

    def test_operator_evolution_uses_shared_world_guidance_boost(self) -> None:
        graphs = [StructuredMeaningGraph(query='verify approval', intent='operator_learning', domain='warehouse_exception')]
        proposals = [
            EvolvedOperatorProposal(
                name='VERIFY_APPROVAL',
                basis_signature=['REQUIRES', 'BLOCKED_BY'],
                basis_operators=['REQUIRES', 'BLOCKED_BY'],
                source_operator_names=['VERIFY_APPROVAL'],
                source_domains=['warehouse_exception'],
                support=1,
                domain_support=1,
                confidence=0.8,
                rationale='approval pattern',
            ),
            EvolvedOperatorProposal(
                name='GENERIC_SCAN',
                basis_signature=['SCAN', 'OBSERVE'],
                basis_operators=['SCAN', 'OBSERVE'],
                source_operator_names=['GENERIC_SCAN'],
                source_domains=['warehouse_exception'],
                support=1,
                domain_support=1,
                confidence=0.8,
                rationale='scan pattern',
            ),
        ]
        with patch.object(OperatorSelfEvolutionEngine, '_propose', return_value=proposals):
            summary = OperatorSelfEvolutionEngine().evolve(
                graphs,
                min_support=1,
                utility_threshold=0.0,
                self_learning_plan={
                    'primary_goal': 'safe movement',
                    'hidden_constraints': ['approval confirmed'],
                    'operator_algebra_targets': [{'label': 'VERIFY_APPROVAL', 'related_operators': ['REQUIRES', 'BLOCKED_BY']}],
                    'retained_operator_candidates': [{'name': 'VERIFY_APPROVAL', 'basis_signature': ['REQUIRES', 'BLOCKED_BY']}],
                },
                integrated_reasoning={'prerequisites': ['approval confirmed']},
            )

        self.assertEqual(summary.proposals[0].name, 'VERIFY_APPROVAL')
        self.assertGreater(summary.proposals[0].utility_score, summary.proposals[1].utility_score)


if __name__ == '__main__':
    unittest.main()

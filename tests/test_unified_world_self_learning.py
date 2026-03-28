from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop import UnifiedWorldModelEngine, UnifiedWorldSelfLearningEngine
from semop.structures import Edge, Node, PlanStep, StructuredMeaningGraph


class UnifiedWorldSelfLearningTests(unittest.TestCase):
    def test_engine_builds_operator_algebra_learning_plan_from_shared_world(self) -> None:
        graph = StructuredMeaningGraph(query='blocked aisle', intent='exception_response', domain='warehouse_exception')
        graph.hidden_goals = ['move pallet safely']
        graph.required_premises = ['supervisor approval']
        graph.add_node(Node(id='move_pallet', label='move_pallet', kind='task'))
        graph.add_node(Node(id='blocked_aisle', label='blocked_aisle', kind='constraint'))
        graph.add_node(Node(id='supervisor_approval', label='supervisor_approval', kind='requirement'))
        graph.add_edge(Edge(source='move_pallet', relation='BLOCKED_BY', target='blocked_aisle'))
        graph.add_edge(Edge(source='move_pallet', relation='REQUIRES', target='supervisor_approval'))
        graph.plan = [PlanStep(id='step_1', action='verify approval first', rationale='blocked aisle', requires=['supervisor_approval'])]

        world_engine = UnifiedWorldModelEngine()
        world = world_engine.from_graph(graph, source='ops')
        reasoning = world_engine.reason(world)
        report = UnifiedWorldSelfLearningEngine().build_report(
            world=world,
            reasoning=reasoning.model_dump(),
            graphs=[graph],
            context='If the route is blocked, verify approval before movement.',
            domain='warehouse_exception',
            scenario='exception_response',
        )

        self.assertTrue(report.objective)
        self.assertTrue(report.plan)
        self.assertTrue(report.summary_text)
        self.assertTrue(report.plan.get('operator_algebra_targets'))
        self.assertTrue(report.plan.get('next_learning_queries'))


if __name__ == '__main__':
    unittest.main()

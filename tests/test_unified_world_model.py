from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop import (
    CommonWorldTaskEngine,
    ContestProblemStructure,
    ContestSolution,
    ContextFrame,
    ContinuousLearningBundleBuilder,
    SharedWorldModel,
    UnifiedResponder,
    UnifiedResponderConfig,
    UnifiedWorldModelEngine,
)
from semop.structures import Edge, Node, PlanStep, StructuredMeaningGraph
from semop.vlso.types import VLSOEntity, VLSORelation


class UnifiedWorldModelTests(unittest.TestCase):
    def test_engine_merges_ops_and_vision_worlds_into_one_shared_reasoning_state(self) -> None:
        graph = StructuredMeaningGraph(query='blocked aisle', intent='exception_response')
        graph.add_node(Node(id='move_through_blocked_aisle', label='move_through_blocked_aisle', kind='task'))
        graph.add_node(Node(id='blocked_aisle', label='blocked_aisle', kind='constraint'))
        graph.add_node(Node(id='supervisor_approval', label='supervisor_approval', kind='requirement'))
        graph.add_edge(Edge(source='move_through_blocked_aisle', relation='BLOCKED_BY', target='blocked_aisle'))
        graph.add_edge(Edge(source='move_through_blocked_aisle', relation='REQUIRES', target='supervisor_approval'))
        graph.plan = [PlanStep(id='step_1', action='??? ?? ????', rationale='blocked aisle', requires=['supervisor_approval'])]
        graph.context_frame = ContextFrame(frame_type='exception_response', primary_goal='move_through_blocked_aisle', active_constraints=['blocked aisle'], missing_requirements=['supervisor approval'])

        vision = SharedWorldModel(query='blocked aisle')
        vision.add_entity(VLSOEntity(id='aisle_photo', label='aisle photo', modality='vision', entity_type='scene'))
        vision.add_entity(VLSOEntity(id='barrier', label='barrier', modality='vision', entity_type='object'))
        vision.add_relation(VLSORelation(source='barrier', relation='PART_OF', target='aisle_photo', modality='vision'))

        engine = UnifiedWorldModelEngine()
        merged = engine.merge(engine.from_graph(graph, source='ops'), vision, query=graph.query)
        reasoning = engine.reason(merged)

        self.assertIn(('move_through_blocked_aisle', 'BLOCKED_BY', 'blocked_aisle'), merged.relation_tuples())
        self.assertIn(('barrier', 'PART_OF', 'aisle_photo'), merged.relation_tuples())
        self.assertIn('move through blocked aisle', reasoning.primary_goal)
        self.assertTrue(any('blocked aisle' in item.lower() for item in reasoning.blockers))
        self.assertTrue(any('approval' in item.lower() for item in reasoning.prerequisites))

    def test_engine_projects_competitive_programming_solution_into_common_world_model(self) -> None:
        structure = ContestProblemStructure(
            normalized_query='range sum queries',
            goal_types=['range_sum'],
            domain_tags=['range_data_structure'],
            extracted_constraints=['n <= 200000', 'q <= 200000'],
            detected_phrases=['range sum', 'queries'],
            hidden_concepts=['prefix accumulation'],
            logical_frames=['offline_range_query'],
            dsl_operators=['PREFIX_SUM'],
            candidate_algorithms=['prefix_sum'],
            episode_priors={},
            memory_projection={},
        )
        solution = ContestSolution(
            category='prefix_sum',
            approach='Use prefix sums and answer each query in O(1).',
            cpp_code='int main() { return 0; }',
            time_complexity='O(n + q)',
            memory_complexity='O(n)',
            confidence=0.82,
            reasoning_steps=['Extract constraints.', 'Build prefix sums.'],
            goal_types=['range_sum'],
            domain_tags=['range_data_structure'],
            dsl_operators=['PREFIX_SUM'],
            extracted_constraints=['n <= 200000', 'q <= 200000'],
            compile_ok=True,
            validation_report={'overall_ok': True},
        )

        engine = UnifiedWorldModelEngine()
        world = engine.from_cp_solution('Given an array and many range sum queries.', solution, structure=structure)
        reasoning = engine.reason(world)

        self.assertIn(('contest_problem', 'USES_ALGORITHM', 'prefix_sum'), world.relation_tuples())
        self.assertTrue(any(operator.name == 'PREFIX_SUM' for operator in world.operators))
        self.assertTrue(any('range sum' in goal.lower() for goal in world.goals))
        self.assertTrue(any('prefix sum' in item.lower() for item in reasoning.evidence))

    def test_engine_projects_environment_brain_summary_into_common_world_model(self) -> None:
        from semop.environment_brain import EnvironmentBrainSummary, EnvironmentConceptStat, EnvironmentProbe, EnvironmentRoutineStat

        summary = EnvironmentBrainSummary(
            environment_name='warehouse exception',
            environment_source='environment_brain_warehouse_exception',
            domain='warehouse_exception',
            scenario='exception_response',
            source_context='If the route is blocked, verify approval before movement.',
            report_path='report.json',
            mastery_scores={'environment_mastery': 0.8},
            stable_concepts=[EnvironmentConceptStat(label='approval', kind='constraint', support_count=4)],
            stable_constraints=['approval confirmed'],
            routine_patterns=[EnvironmentRoutineStat(label='verify approval before movement', support_count=4, required_premises=['approval confirmed'], operator_families=['CHECK'])],
            hazard_patterns=['blocked route'],
            visual_entities=['door'],
            next_probes=[EnvironmentProbe(query='Which exact evidence should ground the next answer?', purpose='grounding')],
        )
        engine = UnifiedWorldModelEngine()
        world = engine.from_environment_brain_summary(summary)
        reasoning = engine.reason(world)

        self.assertTrue(any(relation.relation == 'REQUIRES' for relation in world.relations))
        self.assertTrue(any(relation.relation == 'BLOCKED_BY' for relation in world.relations))
        self.assertTrue(any('blocked route' in item.lower() for item in reasoning.blockers))
        self.assertTrue(any('approval confirmed' in item.lower() for item in reasoning.prerequisites))

    def test_continuous_learning_bundle_exports_integrated_world_traces(self) -> None:
        output_dir = Path('tests') / 'continuous_learning_unified_world_case'
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            graph = StructuredMeaningGraph(query='If the route is blocked, verify approval before movement.', intent='exception_response', domain='warehouse_exception')
            graph.hidden_goals = ['move safely']
            graph.required_premises = ['approval confirmed']
            graph.audit_trace = ['approval gate']
            graph.add_node(Node(id='move_safely', label='move safely', kind='goal'))
            graph.add_node(Node(id='approval_confirmed', label='approval confirmed', kind='requirement'))
            graph.add_edge(Edge(source='move_safely', relation='REQUIRES', target='approval_confirmed'))

            summary = ContinuousLearningBundleBuilder().build_from_graphs([graph], output_dir)
            manifest = (output_dir / 'bundle_manifest.json').read_text(encoding='utf-8')
            integrated_lines = (output_dir / 'integrated_world_traces.jsonl').read_text(encoding='utf-8').strip().splitlines()

            self.assertEqual(summary.integrated_trace_count, 1)
            self.assertIn('integrated_trace_path', manifest)
            self.assertEqual(len(integrated_lines), 1)
            self.assertIn('integrated_reasoning', integrated_lines[0])
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_common_world_task_engine_turns_shared_world_into_decision_text(self) -> None:
        graph = StructuredMeaningGraph(query='blocked aisle', intent='exception_response')
        graph.add_node(Node(id='move_through_blocked_aisle', label='move_through_blocked_aisle', kind='task'))
        graph.add_node(Node(id='blocked_aisle', label='blocked_aisle', kind='constraint'))
        graph.add_node(Node(id='supervisor_approval', label='supervisor_approval', kind='requirement'))
        graph.add_edge(Edge(source='move_through_blocked_aisle', relation='BLOCKED_BY', target='blocked_aisle'))
        graph.add_edge(Edge(source='move_through_blocked_aisle', relation='REQUIRES', target='supervisor_approval'))
        graph.plan = [PlanStep(id='step_1', action='verify approval first', rationale='blocked aisle', requires=['supervisor_approval'])]

        engine = UnifiedWorldModelEngine()
        world = engine.from_graph(graph, source='ops')
        reasoning = engine.reason(world)
        task = CommonWorldTaskEngine().run('How should I respond if the aisle is blocked and approval is missing?', world, reasoning.model_dump(), route='ops')

        self.assertEqual(task.task_type, 'decide')
        self.assertIn('verify approval first', task.answer_text)
        self.assertIn('move through blocked aisle', task.answer_text)

    def test_unified_responder_routes_cp_prompts_and_attaches_integrated_world_payload(self) -> None:
        structure = ContestProblemStructure(
            normalized_query='range sum queries',
            goal_types=['range_sum'],
            domain_tags=['range_data_structure'],
            extracted_constraints=['n <= 200000', 'q <= 200000'],
            detected_phrases=['range sum', 'queries'],
            hidden_concepts=['prefix accumulation'],
            logical_frames=['offline_range_query'],
            dsl_operators=['PREFIX_SUM'],
            candidate_algorithms=['prefix_sum'],
            episode_priors={},
            memory_projection={},
        )
        solution = ContestSolution(
            category='prefix_sum',
            approach='Use prefix sums and answer each query in O(1).',
            cpp_code='int main() { return 0; }',
            time_complexity='O(n + q)',
            memory_complexity='O(n)',
            confidence=0.82,
            reasoning_steps=['Extract constraints.', 'Build prefix sums.'],
            goal_types=['range_sum'],
            domain_tags=['range_data_structure'],
            dsl_operators=['PREFIX_SUM'],
            extracted_constraints=['n <= 200000', 'q <= 200000'],
            compile_ok=True,
            validation_report={'overall_ok': True},
        )
        responder = UnifiedResponder(
            UnifiedResponderConfig(),
            cp_solver=lambda prompt: solution,
            cp_problem_parser=lambda prompt: structure,
        )

        result = responder.respond('Given an array of N integers and Q range sum queries, which algorithm and C++ strategy should I use?')
        payload = result.model_dump()

        self.assertEqual(result.route, 'cp')
        self.assertEqual(result.prompt_understanding.get('likely_domain'), 'competitive_programming')
        self.assertIn('integrated_world', payload)
        self.assertIn('integrated_reasoning', payload)
        self.assertIn('common_world_task', payload)
        self.assertEqual(payload.get('common_world_task', {}).get('task_type'), 'solve')
        self.assertIn('O(n + q)', result.answer_text)


if __name__ == '__main__':
    unittest.main()

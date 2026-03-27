from __future__ import annotations

import os
import shutil
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop.adaptive_environment_learning import AdaptiveEnvironmentLearningRunner
from semop.environment_brain import EnvironmentBrainSummary, EnvironmentConceptStat, EnvironmentProbe, EnvironmentRoutineStat
from semop.grounding_self_evolution import GroundingReflection, GroundingSelfEvolutionRound, GroundingSelfEvolutionSummary
from semop.multimodal_scene_understanding import TemporalSituationSummary
from semop.structures import StructuredMeaningGraph
from semop.unified_benchmark import GroundedExplanationEvalCase


class AdaptiveEnvironmentLearningRunnerTests(unittest.TestCase):
    def test_runner_builds_local_self_improvement_summary(self) -> None:
        temp_dir = os.path.join('tests', 'adaptive_environment_learning_case')
        os.makedirs(temp_dir, exist_ok=True)
        try:
            fake_brain = EnvironmentBrainSummary(
                environment_name='warehouse_exception_exception_response',
                environment_source='environment_brain_warehouse_exception_exception_response',
                domain='warehouse_exception',
                scenario='exception_response',
                source_context='If the route is blocked, verify approval before movement.',
                report_path=os.path.join(temp_dir, 'environment_brain_report.json'),
                seeded_query_count=6,
                stored_graph_count=6,
                auto_approved_review_count=3,
                pending_review_count=1,
                mastery_scores={'environment_mastery': 0.78, 'grounding_strength': 0.62, 'safety_alignment': 0.84},
                stable_concepts=[EnvironmentConceptStat(label='approval', kind='constraint', support_count=4)],
                stable_constraints=['approval confirmed'],
                routine_patterns=[EnvironmentRoutineStat(label='verify approval before movement', support_count=4)],
                hazard_patterns=['blocked route'],
                visual_entities=['door'],
                next_probes=[EnvironmentProbe(query='Which exact evidence should ground the next answer?', purpose='grounding')],
                training={'artifacts': {'continuous_learning_bundle_dir': os.path.join(temp_dir, 'environment_bundle')}},
            )
            fake_self_evolution = GroundingSelfEvolutionSummary(
                output_path=os.path.join(temp_dir, 'grounding_self_evolution_report.json'),
                reflection_memory_path=os.path.join(temp_dir, 'reflection_memory.json'),
                rounds=[GroundingSelfEvolutionRound(round_index=1, attempted_cases=4, improved_cases=3, approved_reviews=2)],
                reflections=[GroundingReflection(query='q', domain='warehouse_exception', scenario='exception_response', weakness='weak grounding', strategy='cite_local_evidence', score_before=0.4, score_after=0.76)],
                improved_cases=3,
                approved_reviews=2,
                stored_graphs=5,
                final_grounding_score=0.76,
                strategy_labels=['cite_local_evidence'],
            )
            fake_training = SimpleNamespace(
                model_dump=lambda: {
                    'artifacts': {'continuous_learning_bundle_dir': os.path.join(temp_dir, 'adaptive_bundle')},
                    'trained_on_graphs': 8,
                }
            )
            fake_graphs = [StructuredMeaningGraph(query='verify approval before movement', intent='environment_reasoning') for _ in range(8)]
            fake_cases = [
                GroundedExplanationEvalCase(
                    query='What should be verified before movement?',
                    source_context='Verify approval before movement.',
                    expected_evidence_terms=['verify approval'],
                    domain='warehouse_exception',
                    scenario='exception_response',
                )
            ]
            fake_temporal = TemporalSituationSummary(
                input_path='frames.json',
                frame_count=3,
                stable_entities=['door'],
                changed_entities=['box'],
                temporal_events=['frame_2: appeared -> box'],
                situation_summary='Door stays stable while a box appears and disappears.',
                answer_text='Door stays stable while a box appears and disappears.',
            )
            with patch('semop.adaptive_environment_learning.EnvironmentBrainRunner.run', return_value=fake_brain), \
                 patch('semop.adaptive_environment_learning.GroundingSelfEvolutionRunner.run', return_value=fake_self_evolution), \
                 patch('semop.adaptive_environment_learning.UnifiedSemOpTrainer.train_from_store', return_value=fake_training), \
                 patch.object(AdaptiveEnvironmentLearningRunner, '_environment_graphs', return_value=fake_graphs), \
                 patch.object(AdaptiveEnvironmentLearningRunner, '_derive_local_grounding_cases', return_value=fake_cases), \
                 patch.object(AdaptiveEnvironmentLearningRunner, '_merge_refined_graphs', return_value=5), \
                 patch.object(AdaptiveEnvironmentLearningRunner, '_summarize_temporal_scene', return_value=fake_temporal):
                summary = AdaptiveEnvironmentLearningRunner().run(
                    environment_name='warehouse_exception_exception_response',
                    domain='warehouse_exception',
                    scenario='exception_response',
                    context='If the route is blocked, verify approval before movement.',
                    store_path=os.path.join(temp_dir, 'store.db'),
                    review_queue_path=os.path.join(temp_dir, 'reviews.db'),
                    output_dir=os.path.join(temp_dir, 'output'),
                    visual_input='frames.json',
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertGreater(summary.capability_scores.get('local_intelligence', 0.0), 0.0)
        self.assertGreater(summary.capability_scores.get('embodied_planning', 0.0), 0.0)
        self.assertEqual(summary.improved_cases, 3)
        self.assertEqual(summary.refined_graph_copies, 5)
        self.assertTrue(summary.completed_skills)
        self.assertTrue(summary.next_actions)
        self.assertTrue(summary.action_rehearsals)
        self.assertEqual(summary.grounding_cases_used, 1)


if __name__ == '__main__':
    unittest.main()

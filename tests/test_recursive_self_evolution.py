from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop.adaptive_environment_learning import (
    AdaptiveActionRehearsal,
    AdaptiveActionStep,
    AdaptiveEnvironmentAxis,
    AdaptiveEnvironmentLearningSummary,
)
from semop.recursive_self_evolution import RecursiveSelfEvolutionRunner


class RecursiveSelfEvolutionRunnerTests(unittest.TestCase):
    def test_runner_evolves_toward_better_local_program(self) -> None:
        temp_dir = Path('tests') / 'recursive_self_evolution_case'
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            def fake_run(self, **kwargs):
                queries = ' '.join(kwargs.get('seed_queries') or [])
                grounded = 0.78 if 'exact evidence' in queries else 0.5
                embodied = 0.76 if 'action order' in queries or 'rehearsed' in queries else 0.46
                contextual = 0.77 if 'hidden constraint' in queries or 'reused' in queries or 'repeated routine' in queries else 0.62
                multimodal = 0.6 if 'changes over time' in queries else 0.34
                self_reflection = 0.72 if kwargs.get('self_evolution_rounds', 2) >= 3 or 'assumption is probably wrong' in queries else 0.53
                local = round(min(1.0, grounded * 0.28 + embodied * 0.22 + contextual * 0.18 + multimodal * 0.1 + self_reflection * 0.22), 4)
                axes = [
                    AdaptiveEnvironmentAxis(key='local_intelligence', label='Local autonomous intelligence', score=local, target=0.7, ready=local >= 0.7, summary='combined', next_step=''),
                    AdaptiveEnvironmentAxis(key='embodied_planning', label='Embodied action rehearsal in this environment', score=embodied, target=0.64, ready=embodied >= 0.64, summary='embodied', next_step=''),
                ]
                report_path = Path(kwargs['output_dir']) / 'adaptive_environment_learning_report.json'
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text('{}', encoding='utf-8')
                return AdaptiveEnvironmentLearningSummary(
                    environment_name=kwargs['environment_name'],
                    environment_source=f"source_{kwargs['environment_name']}",
                    domain=kwargs['domain'],
                    scenario=kwargs['scenario'],
                    output_dir=str(kwargs['output_dir']),
                    report_path=str(report_path),
                    approved_review_count=4 if grounded >= 0.78 else 1,
                    grounding_cases_used=4,
                    improved_cases=3 if self_reflection >= 0.72 else 1,
                    capability_scores={
                        'local_intelligence': local,
                        'grounded_reasoning': grounded,
                        'self_reflection': self_reflection,
                        'embodied_planning': embodied,
                        'contextual_reasoning': contextual,
                        'multimodal_understanding': multimodal,
                    },
                    ready_axes=sum(1 for item in axes if item.ready),
                    total_axes=len(axes),
                    completed_skills=[item.label for item in axes if item.ready],
                    remaining_gaps=[item.label for item in axes if not item.ready],
                    next_actions=['keep evolving'],
                    axes=axes,
                    integrated_reasoning={
                        'active_domains': ['environment_brain', 'self_evolution', 'video'],
                        'blockers': ['route blocked'],
                        'prerequisites': ['approval confirmed'],
                        'evidence': ['door observed', 'action rehearsal'],
                        'next_steps': ['verify approval before movement'],
                    },
                    integrated_reasoning_text='route blocked -> verify approval before movement',
                    action_rehearsals=[
                        AdaptiveActionRehearsal(
                            label='Routine rehearsal 1',
                            trigger='blocked route',
                            safety_goal='stay safe',
                            confidence=0.74,
                            steps=[
                                AdaptiveActionStep(order=1, action='approval confirmed', rationale='guard'),
                                AdaptiveActionStep(order=2, action='verify approval before movement', rationale='execute'),
                            ],
                        )
                    ],
                )

            with patch('semop.recursive_self_evolution.AdaptiveEnvironmentLearningRunner.run', new=fake_run):
                summary = RecursiveSelfEvolutionRunner().run(
                    environment_name='warehouse_exception_exception_response',
                    domain='warehouse_exception',
                    scenario='exception_response',
                    context='If the route is blocked, verify approval before movement.',
                    store_path=str(temp_dir / 'store.db'),
                    review_queue_path=str(temp_dir / 'reviews.db'),
                    output_dir=str(temp_dir / 'output'),
                    visual_input='frames.json',
                    generations=3,
                    population_size=3,
                    children_per_generation=4,
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertGreaterEqual(summary.final_best_score, summary.initial_best_score)
        self.assertTrue(summary.best_program)
        self.assertTrue(summary.generations)
        self.assertTrue(summary.research_principles)
        self.assertTrue(summary.deployed_summary)
        self.assertIn('action_rehearsal', summary.best_program.get('focus_tags', []))


if __name__ == '__main__':
    unittest.main()

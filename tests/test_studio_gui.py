from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import semop_studio_gui


class StudioGuiProofRenderTests(unittest.TestCase):
    def test_home_page_shows_autopilot_coach_controls(self) -> None:
        app = semop_studio_gui.StudioApp()
        html = app.handle({})
        self.assertIn('Autopilot coach', html)
        self.assertIn('Do everything for me', html)
        self.assertIn('Re-check proof on current data', html)
        self.assertIn('#autopilot-lab', html)

    def test_generalization_proof_result_mentions_goal_tracker(self) -> None:
        html = semop_studio_gui.render_result(
            'generalization_proof',
            {
                'proof': {
                    'accepted_rounds': 1,
                    'final_gate_accepted': False,
                    'rounds': [
                        {
                            'output_dir': 'data/unified_semop_gui_run/generalization_proof/round_01',
                            'benchmark': {'unseen_transfer': 0.4, 'analogy_usefulness': 0.3, 'compiler_validity': 0.6, 'grounded_explanation_fidelity': 0.4, 'repair_success_rate': 0.7},
                            'understanding': {'progress': {'robust_general_intelligence_overall': 0.5}},
                        }
                    ],
                    'evidence': {
                        'headline': 'proof headline',
                        'learned_generalization_score': 0.4,
                        'reviewed_corpus_growth_score': 0.5,
                        'multimodal_transfer_score': 0.45,
                        'domain_coverage_score': 0.5,
                        'strong_model_score': 0.46,
                        'strengths': ['one'],
                        'risks': ['two'],
                        'next_steps': ['three'],
                    },
                    'goal_tracker': {
                        'readiness_percent': 72.5,
                        'ready_axes': 2,
                        'total_axes': 5,
                        'priority_focus': 'Wide domain coverage',
                        'completed_items': ['Reviewed corpus improvement proof'],
                        'remaining_items': ['Wide domain coverage'],
                        'domains_seen': ['general', 'warehouse_exception'],
                        'scenarios_seen': ['qa', 'exception_response'],
                        'axes': [
                            {'label': 'Wide domain coverage', 'score': 0.5, 'target': 0.6, 'verified': False},
                        ],
                    },
                    'report_path': 'data/unified_semop_gui_run/generalization_proof/generalization_proof_report.json',
                },
                'curriculum_rounds': [],
            },
        )
        self.assertIn('Ultimate goal tracker', html)
        self.assertIn('Goal readiness', html)
        self.assertIn('Wide domain coverage', html)


if __name__ == '__main__':
    unittest.main()

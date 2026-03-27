from __future__ import annotations

import os
import sys
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop.ultimate_agi_readiness import UltimateAGIReadinessRunner


class UltimateAGIReadinessTests(unittest.TestCase):
    def test_runner_builds_multiaxis_readiness_summary(self) -> None:
        fake_audit = SimpleNamespace(
            overall_readiness_percent=76.5,
            report_path='data/unified_semop_gui_run/capability_audit_report.json',
            axes=[
                SimpleNamespace(name='semop_reasoning', score=0.78),
                SimpleNamespace(name='generalization_proof', score=0.71),
            ],
            model_dump=lambda: {
                'benchmark_gate': {
                    'benchmark': {
                        'grounded_explanation_fidelity': 0.58,
                        'compiler_validity': 0.81,
                        'analogy_usefulness': 0.55,
                    },
                    'gate': {
                        'blocking_reasons': ['grounded_explanation_fidelity below threshold'],
                    },
                },
                'generalization_proof': {
                    'evidence': {
                        'learned_generalization_score': 0.73,
                        'reviewed_corpus_growth_score': 0.61,
                        'multimodal_transfer_score': 0.69,
                        'domain_coverage_score': 0.57,
                        'strong_model_score': 0.62,
                    },
                    'goal_tracker': {
                        'readiness_percent': 74.0,
                    },
                },
                'math_self_test': {'passed_count': 3, 'total_count': 4},
                'math_training': {'final_average_score': 0.76},
                'visual_reconstruction': {
                    'primitives': [{}, {}, {}, {}],
                    'relations': [{}, {}, {}],
                },
                'runtime_doctor': {
                    'dependency_status': {
                        'llm_ready': True,
                        'training_ready': True,
                    }
                },
            },
        )
        temp_dir = tempfile.mkdtemp(dir='tests')
        try:
            output_path = os.path.join(temp_dir, 'ultimate_agi_readiness.json')
            with patch('semop.ultimate_agi_readiness.CapabilityAuditRunner.run', return_value=fake_audit), patch('pathlib.Path.write_text', return_value=None):
                summary = UltimateAGIReadinessRunner().run(workspace='.', output_path=output_path, bootstrap_missing=False)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        self.assertGreater(summary.overall_readiness_percent, 0.0)
        self.assertTrue(summary.axes)
        self.assertTrue(summary.concept_fusion_preview)
        self.assertIn('Commercialization readiness', [item.label for item in summary.axes])


if __name__ == '__main__':
    unittest.main()

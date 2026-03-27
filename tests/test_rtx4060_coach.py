from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from semop import CapabilityAxisStatus, CapabilityAuditSummary, RTX4060ReasoningCoach


class RTX4060CoachTests(unittest.TestCase):
    def _audit_summary(self, readiness: float = 83.2) -> CapabilityAuditSummary:
        return CapabilityAuditSummary(
            workspace='.',
            report_path='data/unified_semop_gui_run/capability_audit_report.json',
            runtime_doctor={
                'hardware_profile': {'detected_profile': 'rtx_4060_8gb'},
                'operator_algebra_mode': 'symbolic_first_gpu_assist',
            },
            benchmark_gate={'gate': {'blocking_reasons': ['grounded_explanation_fidelity below threshold']}},
            generalization_proof={'goal_tracker': {'readiness_percent': 77.4, 'remaining_items': ['Reviewed corpus improvement proof']}},
            math_self_test={},
            math_training={'total_cases': 12},
            visual_reconstruction={'topology_summary': {'primitive_count': 5}},
            generated_artifacts=[],
            axes=[
                CapabilityAxisStatus(name='local_runtime', score=1.0, status='strong'),
                CapabilityAxisStatus(name='semop_reasoning', score=0.8248, status='usable'),
                CapabilityAxisStatus(name='generalization_proof', score=0.774, status='usable'),
                CapabilityAxisStatus(name='math_world_model', score=0.7625, status='usable'),
                CapabilityAxisStatus(name='visual_3d_reconstruction', score=0.8, status='usable'),
            ],
            overall_readiness_percent=readiness,
            overall_status='usable',
            priority_improvements=['grounded fidelity'],
        )

    def test_assess_builds_4060_data_collection_plan(self) -> None:
        with mock.patch('semop.rtx4060_coach.CapabilityAuditRunner.run', return_value=self._audit_summary()), mock.patch('semop.rtx4060_coach.ReviewQueueStore') as store_cls:
            store_cls.return_value.fetch_items.return_value = [
                SimpleNamespace(domain='general', scenario='qa', reasons=['grounding_review']),
                SimpleNamespace(domain='warehouse_exception', scenario='exception_response', reasons=['approved_training_trace']),
            ]
            summary = RTX4060ReasoningCoach().assess(output_path='tests/tmp_rtx4060_coach_assess.json')
        self.assertEqual(summary.detected_profile, 'rtx_4060_8gb')
        self.assertEqual(summary.operator_algebra_mode, 'symbolic_first_gpu_assist')
        self.assertTrue(summary.data_collection_lanes)
        self.assertGreaterEqual(summary.grounded_review_total, 1)

    def test_improve_reports_readiness_delta(self) -> None:
        before = self._audit_summary(80.0)
        after = self._audit_summary(86.0)
        improvement = SimpleNamespace(output_path='data/unified_semop_gui_run/capability_improvement_report.json', model_dump=lambda: {'delta_readiness_percent': 6.0})
        with mock.patch('semop.rtx4060_coach.CapabilityAuditRunner.run', side_effect=[before, after]), mock.patch('semop.rtx4060_coach.CapabilityImprovementRunner.run', return_value=improvement), mock.patch('semop.rtx4060_coach.ReviewQueueStore') as store_cls:
            store_cls.return_value.fetch_items.return_value = []
            summary = RTX4060ReasoningCoach().improve(output_path='tests/tmp_rtx4060_coach_improve.json')
        self.assertEqual(summary.mode, 'improvement')
        self.assertEqual(summary.before_readiness_percent, 80.0)
        self.assertEqual(summary.after_readiness_percent, 86.0)
        self.assertEqual(summary.delta_readiness_percent, 6.0)


if __name__ == '__main__':
    unittest.main()

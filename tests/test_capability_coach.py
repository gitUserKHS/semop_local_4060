from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop.capability_audit import CapabilityAuditSummary, CapabilityAxisStatus
from semop.capability_coach import CapabilityCurriculumRound, CapabilityImprovementRunner


class CapabilityCoachTests(unittest.TestCase):
    def test_improvement_runner_writes_delta_report(self) -> None:
        workspace = Path('data') / 'capability_coach_test'
        shutil.rmtree(workspace, ignore_errors=True)
        try:
            unified_dir = workspace / 'unified'
            math_dir = workspace / 'math'
            visual_dir = math_dir / 'visual_3d'
            unified_dir.mkdir(parents=True, exist_ok=True)
            math_dir.mkdir(parents=True, exist_ok=True)
            visual_dir.mkdir(parents=True, exist_ok=True)
            before = CapabilityAuditSummary(
                workspace=str(workspace),
                report_path=str(unified_dir / 'before.json'),
                runtime_doctor={},
                axes=[CapabilityAxisStatus(name='semop_reasoning', score=0.3, status='weak')],
                overall_readiness_percent=48.0,
                overall_status='partial',
                priority_improvements=['seed more reviewed traces'],
            )
            after = CapabilityAuditSummary(
                workspace=str(workspace),
                report_path=str(unified_dir / 'capability_audit_report.json'),
                runtime_doctor={},
                axes=[CapabilityAxisStatus(name='semop_reasoning', score=0.72, status='usable')],
                overall_readiness_percent=71.0,
                overall_status='usable',
                priority_improvements=[],
            )

            def proof_side_effect(*args, **kwargs):
                callback = kwargs.get('round_setup_callback')
                if callback is not None:
                    callback(1, 'capability_coach_round_01', 'train')
                return SimpleNamespace(model_dump=lambda: {'goal_tracker': {'readiness_percent': 60.0}, 'report_path': str(unified_dir / 'generalization_proof' / 'generalization_proof_report.json')})

            with mock.patch('semop.capability_coach.CapabilityAuditRunner.run', side_effect=[before, after]), mock.patch('semop.capability_coach.GroundingSelfEvolutionRunner') as self_evolution_cls, mock.patch.object(CapabilityImprovementRunner, '_seed_round', return_value=CapabilityCurriculumRound(round_index=1, source_used='capability_coach_round_01', domain='general', scenario='qa', guided_seeded=3, approved_reviews=2, average_novelty=0.8)), mock.patch('semop.capability_coach.GeneralizationProofHarness') as proof_cls, mock.patch('semop.capability_coach.BenchmarkGatedContinuousTrainer') as gate_cls:
                self_evolution_cls.return_value.run.return_value.model_dump.return_value = {'improved_cases': 2, 'approved_reviews': 2, 'stored_graphs': 4, 'final_grounding_score': 0.66, 'strategy_labels': ['self_refine_trim_unsupported_claims'], 'output_path': str(unified_dir / 'grounding_self_evolution_report.json')}
                proof_cls.return_value.run.side_effect = proof_side_effect
                gate_cls.return_value.train_evaluate_and_gate.return_value.model_dump.return_value = {'gate': {'accepted': False}, 'benchmark': {'unseen_transfer': 0.2}}
                summary = CapabilityImprovementRunner().run(
                    workspace=str(workspace),
                    output_path=str(unified_dir / 'capability_improvement_report.json'),
                    unified_output_dir=str(unified_dir),
                    unified_store_path=str(workspace / 'store.db'),
                    unified_review_queue_path=str(workspace / 'reviews.db'),
                    unified_benchmark_corpus_path=str(unified_dir / 'persistent_benchmark_corpus.json'),
                    math_output_dir=str(math_dir),
                    visual_output_dir=str(visual_dir),
                )
            self.assertEqual(summary.delta_readiness_percent, 23.0)
            self.assertTrue(Path(summary.output_path).exists())
            payload = json.loads(Path(summary.output_path).read_text(encoding='utf-8'))
            self.assertEqual(payload['after']['overall_readiness_percent'], 71.0)
            self.assertTrue(any('TurboQuant-inspired' in item for item in payload['actions_taken']))
            self.assertEqual(payload['self_evolution']['improved_cases'], 2)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

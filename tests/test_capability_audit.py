from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop import CapabilityAuditRunner


class CapabilityAuditTests(unittest.TestCase):
    def test_audit_uses_existing_artifacts_without_bootstrap(self) -> None:
        workspace = Path('data') / 'capability_audit_test_existing'
        shutil.rmtree(workspace, ignore_errors=True)
        try:
            unified_dir = workspace / 'unified'
            math_dir = workspace / 'math'
            visual_dir = math_dir / 'visual_3d'
            unified_dir.mkdir(parents=True, exist_ok=True)
            math_dir.mkdir(parents=True, exist_ok=True)
            visual_dir.mkdir(parents=True, exist_ok=True)
            (unified_dir / 'benchmark_gate.json').write_text(json.dumps({
                'gate': {
                    'accepted': True,
                    'candidate_metrics': {
                        'unseen_transfer': 0.8,
                        'analogy_usefulness': 0.7,
                        'compiler_validity': 0.9,
                        'grounded_explanation_fidelity': 0.75,
                        'repair_success_rate': 0.85,
                    },
                    'blocking_reasons': [],
                }
            }, ensure_ascii=False), encoding='utf-8')
            proof_dir = unified_dir / 'generalization_proof'
            proof_dir.mkdir(parents=True, exist_ok=True)
            (proof_dir / 'generalization_proof_report.json').write_text(json.dumps({
                'goal_tracker': {
                    'readiness_percent': 78.0,
                    'priority_focus': 'Wide domain coverage',
                    'remaining_items': ['wide transfer'],
                    'domains_seen': ['general'],
                }
            }, ensure_ascii=False), encoding='utf-8')
            (math_dir / 'math_training_summary.json').write_text(json.dumps({
                'final_average_score': 0.82,
                'accepted_cases': 6,
                'exact_match_cases': 4,
            }, ensure_ascii=False), encoding='utf-8')
            (visual_dir / 'scene_3d_reconstruction.json').write_text(json.dumps({
                'primitives': [{'id': 'p1'}],
                'warnings': [],
                'export_paths': {'obj_path': 'scene.obj'},
            }, ensure_ascii=False), encoding='utf-8')
            fake_self_test = {'passed_count': 2, 'total_count': 2, 'readiness': {'recommended_actions': ['none']}}
            with mock.patch('semop.capability_audit.build_runtime_doctor_report') as doctor, mock.patch('semop.capability_audit.WorldModelMathProductionService') as service_cls:
                doctor.return_value.model_dump.return_value = {
                    'hardware_profile': {'detected_profile': 'rtx_4060_8gb'},
                    'dependency_status': {'llm_ready': True, 'training_ready': True, 'qlora_ready': True},
                    'surfaces': [{'ready': True}, {'ready': True}],
                    'recommended_actions': ['ok'],
                }
                service_cls.return_value.self_test.return_value.model_dump.return_value = fake_self_test
                summary = CapabilityAuditRunner().run(
                    workspace=str(workspace),
                    bootstrap_missing=False,
                    output_path=str(workspace / 'audit.json'),
                    unified_output_dir=str(unified_dir),
                    unified_store_path=str(workspace / 'store.db'),
                    unified_review_queue_path=str(workspace / 'reviews.db'),
                    math_output_dir=str(math_dir),
                    visual_output_dir=str(visual_dir),
                )
            self.assertGreater(summary.overall_readiness_percent, 60.0)
            self.assertIn(summary.overall_status, {'usable', 'strong'})
            self.assertFalse(summary.generated_artifacts)
            self.assertTrue(any(axis.name == 'math_world_model' for axis in summary.axes))
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_audit_bootstraps_missing_artifacts(self) -> None:
        workspace = Path('data') / 'capability_audit_test_bootstrap'
        shutil.rmtree(workspace, ignore_errors=True)
        try:
            unified_dir = workspace / 'unified'
            math_dir = workspace / 'math'
            visual_dir = math_dir / 'visual_3d'
            unified_dir.mkdir(parents=True, exist_ok=True)
            math_dir.mkdir(parents=True, exist_ok=True)
            visual_dir.mkdir(parents=True, exist_ok=True)
            store_path = workspace / 'store.db'
            store_path.write_text('', encoding='utf-8')
            fake_self_test = {'passed_count': 1, 'total_count': 2, 'readiness': {'recommended_actions': ['train more']}}
            with mock.patch('semop.capability_audit.build_runtime_doctor_report') as doctor, mock.patch('semop.capability_audit.GeneralizationProofHarness') as proof_cls, mock.patch('semop.capability_audit.WorldModelMathTrainer') as trainer_cls, mock.patch('semop.capability_audit.WorldModelMathProductionService') as service_cls, mock.patch('semop.capability_audit.VisualGeometry3DWorkbench') as workbench_cls:
                doctor.return_value.model_dump.return_value = {
                    'hardware_profile': {'detected_profile': 'cpu_only'},
                    'dependency_status': {'llm_ready': True, 'training_ready': False, 'qlora_ready': False},
                    'surfaces': [{'ready': True}],
                    'recommended_actions': ['install peft'],
                }
                proof_cls.return_value.run.return_value.model_dump.return_value = {'goal_tracker': {'readiness_percent': 55.0, 'priority_focus': 'proof', 'remaining_items': ['coverage'], 'domains_seen': ['general']}, 'report_path': str(unified_dir / 'generalization_proof' / 'generalization_proof_report.json')}
                proof_cls.return_value.run.return_value.report_path = str(unified_dir / 'generalization_proof' / 'generalization_proof_report.json')
                trainer_cls.return_value.train_from_cases.return_value.model_dump.return_value = {'final_average_score': 0.7, 'accepted_cases': 3, 'exact_match_cases': 2}
                service_cls.return_value.self_test.return_value.model_dump.return_value = fake_self_test
                workbench_cls.return_value.reconstruct_scene.return_value.model_dump.return_value = {'primitives': [{'id': 'p1'}], 'warnings': ['weak image'], 'export_paths': {'obj_path': 'scene.obj'}}
                summary = CapabilityAuditRunner().run(
                    workspace=str(workspace),
                    bootstrap_missing=True,
                    output_path=str(workspace / 'audit.json'),
                    unified_output_dir=str(unified_dir),
                    unified_store_path=str(store_path),
                    unified_review_queue_path=str(workspace / 'reviews.db'),
                    math_output_dir=str(math_dir),
                    math_cases_path=str(workspace / 'starter.jsonl'),
                    visual_output_dir=str(visual_dir),
                )
            self.assertTrue(summary.generated_artifacts)
            self.assertTrue(Path(summary.report_path).exists())
            self.assertTrue(any('install peft' in item or 'coverage' in item or 'train more' in item for item in summary.priority_improvements))
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

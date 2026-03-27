from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop import CpParserTrainConfig, CpParserTrainingScaffold, OperatorTrainConfig, OperatorTrainingScaffold, detect_local_hardware, detect_local_ml_stack


def _hw_tempdir(prefix: str) -> Path:
    root = Path('tests') / prefix
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


class HardwareProfileTests(unittest.TestCase):
    def test_detect_rtx_4060_profile_from_env_override(self) -> None:
        with mock.patch.dict(os.environ, {
            'SEMOP_FORCE_CUDA': '1',
            'SEMOP_FORCE_GPU_NAME': 'NVIDIA GeForce RTX 4060',
            'SEMOP_FORCE_VRAM_GB': '8.0',
        }, clear=False):
            profile = detect_local_hardware()
        self.assertEqual(profile.detected_profile, 'rtx_4060_8gb')
        self.assertTrue(profile.cuda_available)
        self.assertEqual(profile.operator_algebra_mode, 'symbolic_first_gpu_assist')
        self.assertEqual(profile.recommended_batch_size, 1)
        self.assertGreaterEqual(profile.recommended_gradient_accumulation, 16)


    def test_detect_local_ml_stack_reports_missing_low_vram_qlora_dependencies(self) -> None:
        with mock.patch('semop.hardware_profiles._package_available') as package_available, mock.patch('semop.hardware_profiles._package_version') as package_version:
            package_available.side_effect = lambda name: name in {'torch', 'transformers'}
            package_version.side_effect = lambda name: {'torch': '2.6.0', 'transformers': '4.52.0'}.get(name, '')
            with mock.patch.dict(os.environ, {
                'SEMOP_FORCE_CUDA': '1',
                'SEMOP_FORCE_GPU_NAME': 'NVIDIA GeForce RTX 4060',
                'SEMOP_FORCE_VRAM_GB': '8.0',
            }, clear=False):
                profile = detect_local_hardware()
                deps = detect_local_ml_stack(profile)
        self.assertTrue(deps.llm_ready)
        self.assertFalse(deps.training_ready)
        self.assertFalse(deps.qlora_ready)
        self.assertIn('peft', deps.missing_optional)
        self.assertIn('bitsandbytes', deps.missing_optional)
        self.assertTrue(any('4060' in note or 'low-VRAM' in note or 'QLoRA' in note for note in deps.notes))

    def test_operator_training_dry_run_auto_tunes_for_4060(self) -> None:
        workspace = _hw_tempdir('operator_hw_runtime')
        try:
            train_jsonl = workspace / 'train.jsonl'
            train_jsonl.write_text(
                json.dumps({'prompt': 'p1', 'completion': 'c1', 'task': 'hidden_premise'}, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
            with mock.patch.dict(os.environ, {
                'SEMOP_FORCE_CUDA': '1',
                'SEMOP_FORCE_GPU_NAME': 'NVIDIA GeForce RTX 4060',
                'SEMOP_FORCE_VRAM_GB': '8.0',
            }, clear=False):
                summary = OperatorTrainingScaffold().run(OperatorTrainConfig(
                    model_name_or_path='local-test-model',
                    output_dir=str(workspace / 'out'),
                    train_jsonl=str(train_jsonl),
                    dry_run=True,
                    local_files_only=True,
                    use_lora=False,
                    use_qlora=False,
                ))
            effective = summary['effective_config']
            self.assertEqual(summary['hardware_profile']['detected_profile'], 'rtx_4060_8gb')
            self.assertEqual(summary['operator_algebra_mode'], 'symbolic_first_gpu_assist')
            self.assertTrue(effective['use_lora'])
            self.assertEqual(effective['batch_size'], 1)
            self.assertGreaterEqual(effective['gradient_accumulation_steps'], 16)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_cp_training_dry_run_auto_tunes_for_4060(self) -> None:
        workspace = _hw_tempdir('cp_hw_runtime')
        try:
            dataset_path = Path(os.path.dirname(__file__)) / '..' / 'examples' / 'cp_dsl_expanded_dataset.jsonl'
            with mock.patch.dict(os.environ, {
                'SEMOP_FORCE_CUDA': '1',
                'SEMOP_FORCE_GPU_NAME': 'NVIDIA GeForce RTX 4060',
                'SEMOP_FORCE_VRAM_GB': '8.0',
            }, clear=False):
                summary = CpParserTrainingScaffold().run(CpParserTrainConfig(
                    model_name_or_path='local-test-model',
                    output_dir=str(workspace / 'out'),
                    train_jsonl=str(dataset_path),
                    dry_run=True,
                    local_files_only=True,
                    use_lora=False,
                    use_qlora=False,
                ))
            effective = summary['effective_config']
            self.assertEqual(summary['hardware_profile']['detected_profile'], 'rtx_4060_8gb')
            self.assertEqual(summary['operator_algebra_mode'], 'symbolic_first_gpu_assist')
            self.assertTrue(effective['use_lora'])
            self.assertEqual(effective['batch_size'], 1)
            self.assertGreaterEqual(effective['gradient_accumulation_steps'], 16)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop import LocalDependencyStatus, LocalHardwareProfile, build_runtime_doctor_report, launch_beginner_one_click, launch_runtime_stack


class RuntimeOpsTests(unittest.TestCase):
    def test_runtime_doctor_report_includes_beginner_launch_and_dependency_snapshot(self) -> None:
        report = build_runtime_doctor_report(workspace='.')
        self.assertTrue(report.beginner_launch_command)
        self.assertTrue(report.install_command)
        self.assertTrue(report.surfaces)
        self.assertIn('llm_ready', report.dependency_status)
        self.assertTrue(any(item.name == 'semop_studio' for item in report.surfaces))
        self.assertTrue(report.one_click_command)
        self.assertTrue(report.double_click_launcher.endswith('launch_semop_studio.bat'))


    def test_runtime_doctor_low_vram_windows_note_when_qlora_missing(self) -> None:
        fake_hw = LocalHardwareProfile(detected_profile='rtx_4060_8gb', device='cuda', cuda_available=True, gpu_name='NVIDIA GeForce RTX 4060', vram_gb=8.0, supports_fp16=True, bitsandbytes_available=False, low_vram=True, recommended_precision='fp16', recommended_use_lora=True, recommended_use_qlora=False, recommended_batch_size=1, recommended_gradient_accumulation=16, recommended_generation_tokens=384, operator_algebra_mode='symbolic_first_gpu_assist', notes=['low-vram'])
        fake_deps = LocalDependencyStatus(torch_available=True, transformers_available=True, peft_available=True, bitsandbytes_available=False, accelerate_available=False, llm_ready=True, training_ready=True, qlora_ready=False, missing_optional=['bitsandbytes', 'accelerate'], notes=['qlora missing'])
        with mock.patch('semop.runtime_ops.os.name', 'nt'), mock.patch('semop.runtime_ops.detect_local_hardware', return_value=fake_hw), mock.patch('semop.runtime_ops.detect_local_ml_stack', return_value=fake_deps):
            report = build_runtime_doctor_report(workspace='.')
        self.assertTrue(any('Windows' in item or '4060' in item or 'QLoRA' in item for item in report.recommended_actions))

    def test_launch_beginner_one_click_dry_run_writes_summary(self) -> None:
        workspace = Path('data') / 'runtime_ops_one_click_case'
        shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            for name in ('semop_studio_gui.py', 'math_world_model_gui.py', 'math_world_model_service.py', 'launch_semop_studio.bat'):
                (workspace / name).write_text('print("stub")\n', encoding='utf-8')
            summary = launch_beginner_one_click(workspace=str(workspace), dry_run=True, open_browser=False)
            self.assertTrue(os.path.exists(summary.manifest_path))
            self.assertTrue(os.path.exists(summary.summary_path))
            payload = json.loads(Path(summary.summary_path).read_text(encoding='utf-8'))
            self.assertTrue(payload['dry_run'])
            self.assertTrue(payload['preferred_url'].endswith(':8780'))
            self.assertFalse(payload['browser_opened'])
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_launch_runtime_stack_dry_run_writes_manifest(self) -> None:
        workspace = Path('data') / 'runtime_ops_test_case'
        shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            for name in ('semop_studio_gui.py', 'math_world_model_gui.py', 'math_world_model_service.py'):
                (workspace / name).write_text('print("stub")\n', encoding='utf-8')
            summary = launch_runtime_stack(workspace=str(workspace), dry_run=True)
            self.assertTrue(os.path.exists(summary.manifest_path))
            payload = json.loads(Path(summary.manifest_path).read_text(encoding='utf-8'))
            self.assertTrue(payload['dry_run'])
            self.assertGreaterEqual(len(payload['entries']), 2)
            self.assertTrue(any(item['name'] == 'semop_studio' for item in payload['entries']))
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

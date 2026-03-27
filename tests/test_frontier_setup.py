from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop import FrontierVisionInstaller


class FrontierSetupTests(unittest.TestCase):
    def test_recommended_bundle_reports_missing_required_models(self) -> None:
        with patch.object(FrontierVisionInstaller, '_is_installed', return_value=False):
            installer = FrontierVisionInstaller(target_root='models/vision/frontier-test-a')
            summary = installer.installed_bundle_status()
        self.assertEqual(summary.recommended_count, 3)
        self.assertFalse(summary.required_ready)
        self.assertIn('qwen2_5_vl', summary.missing_required)
        self.assertIn('florence2', summary.missing_required)

    def test_recommended_bundle_reports_required_ready_when_required_models_exist(self) -> None:
        calls = {'index': 0}

        def fake_is_installed(_path):
            calls['index'] += 1
            return calls['index'] <= 2

        with patch.object(FrontierVisionInstaller, '_is_installed', side_effect=fake_is_installed):
            installer = FrontierVisionInstaller(target_root='models/vision/frontier-test-b')
            summary = installer.installed_bundle_status()
        self.assertTrue(summary.required_ready)
        self.assertEqual(summary.installed_count, 2)
        self.assertIn('qwen2_5_vl', summary.installed_families)
        self.assertIn('florence2', summary.installed_families)


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations

import json
import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop.multimodal_scene_understanding import TemporalSceneReasoner
from semop.prompt_understanding import PromptUnderstandingAnalyzer


class MultimodalSceneUnderstandingTests(unittest.TestCase):
    def test_temporal_scene_reasoner_summarizes_manifest_frames(self) -> None:
        temp_dir = Path('tests') / 'multimodal_scene_case'
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            manifest_path = temp_dir / 'sequence.json'
            manifest_path.write_text(
                json.dumps(
                    {
                        'frames': [
                            {
                                'label': 'frame_1',
                                'visual_input': {
                                    'objects': [
                                        {'id': 'door', 'label': 'door', 'kind': 'opening'},
                                    ],
                                    'relations': [],
                                },
                            },
                            {
                                'label': 'frame_2',
                                'visual_input': {
                                    'objects': [
                                        {'id': 'door', 'label': 'door', 'kind': 'opening'},
                                        {'id': 'box', 'label': 'box', 'kind': 'container'},
                                    ],
                                    'relations': [
                                        {'source': 'box', 'relation': 'LEFT_OF', 'target': 'door'},
                                    ],
                                },
                            },
                            {
                                'label': 'frame_3',
                                'visual_input': {
                                    'objects': [
                                        {'id': 'door', 'label': 'door', 'kind': 'opening'},
                                    ],
                                    'relations': [],
                                },
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            summary = TemporalSceneReasoner().summarize('What changes across these frames?', str(manifest_path))
            self.assertEqual(summary.input_kind, 'json_manifest')
            self.assertEqual(summary.frame_count, 3)
            self.assertIn('door', summary.stable_entities)
            self.assertIn('box', summary.changed_entities)
            self.assertTrue(summary.situation_summary)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


    def test_temporal_scene_reasoner_exports_unified_world_with_events(self) -> None:
        temp_dir = Path('tests') / 'multimodal_scene_world_case'
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            manifest_path = temp_dir / 'sequence.json'
            manifest_path.write_text(
                json.dumps(
                    {
                        'frames': [
                            {
                                'label': 'frame_1',
                                'visual_input': {
                                    'objects': [
                                        {'id': 'door', 'label': 'door', 'kind': 'opening'},
                                    ],
                                    'states': [
                                        {'subject': 'door', 'value': 'CLOSED'},
                                    ],
                                },
                            },
                            {
                                'label': 'frame_2',
                                'visual_input': {
                                    'objects': [
                                        {'id': 'door', 'label': 'door', 'kind': 'opening'},
                                        {'id': 'box', 'label': 'box', 'kind': 'container'},
                                    ],
                                    'states': [
                                        {'subject': 'door', 'value': 'OPEN'},
                                    ],
                                },
                            },
                            {
                                'label': 'frame_3',
                                'visual_input': {
                                    'objects': [
                                        {'id': 'door', 'label': 'door', 'kind': 'opening'},
                                        {'id': 'box', 'label': 'box', 'kind': 'container'},
                                    ],
                                    'states': [
                                        {'subject': 'door', 'value': 'OPEN'},
                                    ],
                                },
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            summary = TemporalSceneReasoner().summarize('What changes across these frames?', str(manifest_path))
            self.assertTrue(summary.world)
            self.assertIn('entities', summary.world)
            self.assertIn('events', summary.world)
            event_types = {item.get('event_type') for item in summary.world['events']}
            self.assertIn('appearance', event_types)
            self.assertIn('state_change', event_types)
            self.assertTrue(any(item.get('relation') == 'INTENT_OF_AGENT' for item in summary.world.get('relations', [])))
            self.assertTrue(summary.answer_text)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_prompt_understanding_analyzer_general_domain_graph_backed(self) -> None:
        analyzer = PromptUnderstandingAnalyzer()
        summary = analyzer.analyze_base(
            'Before I open the box and take the tool, what should I check first?',
            'ops',
            visual_input='',
            domain='warehouse_exception',
            scenario='exception_response',
            source_context='',
        )
        self.assertEqual(summary.likely_domain, 'general')
        self.assertTrue(summary.hidden_context)
        self.assertTrue(summary.hidden_constraints)

    def test_temporal_scene_reasoner_backend_support_snapshot(self) -> None:
        support = TemporalSceneReasoner.backend_support()
        self.assertIn('pillow', support)
        self.assertIn('imageio', support)
        self.assertIn('opencv', support)
        self.assertIn('ffmpeg', support)

    def test_prompt_understanding_visual_route_ignores_ops_source_context(self) -> None:
        analyzer = PromptUnderstandingAnalyzer()
        summary = analyzer.analyze_base(
            '\uc774 \uc0ac\uc9c4\uc744 \uc124\uba85\ud574\ubd10',
            'vision',
            visual_input='data/scene.png',
            domain='warehouse_exception',
            scenario='exception_response',
            source_context='If the work zone is blocked or the task is not approved yet, stop first and check the alternate route and manager approval.',
        )
        self.assertEqual(summary.likely_domain, 'general')
        self.assertEqual(summary.likely_scenario, 'scene_understanding')
        self.assertIn('visible scene', summary.summary.lower())

    def test_prompt_understanding_analyzer_marks_temporal_requirement(self) -> None:
        analyzer = PromptUnderstandingAnalyzer()
        summary = analyzer.analyze_base(
            '? ???? ?? ??? ????? ????',
            'video',
            visual_input='',
            domain='general',
            scenario='qa',
        )
        self.assertEqual(summary.route, 'video')
        self.assertTrue(any('frame folder' in item or 'video file' in item for item in summary.required_inputs))
        self.assertIn('Track the scene across frames', summary.summary)


if __name__ == '__main__':
    unittest.main()

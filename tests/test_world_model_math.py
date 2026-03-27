from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from semop import MathTrainingCase, ProductionMathServiceConfig, VisualGeometry3DWorkbench, WorldModelMathProductionService, WorldModelMathTrainer
from semop.world_model_math import WorldModelMathReasoner
import math_world_model_gui


def _workspace_tempdir() -> str:
    root = os.path.join(os.path.dirname(__file__), 'tmp_world_model_math')
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"run_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}")
    os.makedirs(path, exist_ok=True)
    return path


def _write_cases(path: str) -> None:
    cases = [
        MathTrainingCase(
            case_id='odd_sum_even',
            query='Prove that the sum of two odd integers is even.',
            source_context='Use parity rewriting and explicit divisibility steps.',
            task_mode='number_theory_proof',
            expected_family='number_theory_proof',
            expected_status='accepted',
            required_terms=['odd integers', 'even'],
            exact_answer='therefore the sum of two odd integers is even',
        ),
        MathTrainingCase(
            case_id='induction_sum_formula',
            query='Prove that 1+2+...+n = n(n+1)/2 for all positive integers n.',
            source_context='Use base case, induction hypothesis, and the k+1 step explicitly.',
            task_mode='research_math',
            expected_family='research_math',
            expected_status='accepted',
            required_terms=['base case', 'induction'],
            exact_answer='n(n+1)/2',
        ),
    ]
    with open(path, 'w', encoding='utf-8') as handle:
        for case in cases:
            handle.write(json.dumps(case.model_dump(), ensure_ascii=False) + '\n')


class WorldModelMathReasonerTests(unittest.TestCase):
    def test_reasoner_solves_number_theory_proof(self) -> None:
        report = WorldModelMathReasoner().solve(
            'Prove that the sum of two odd integers is even.',
            task_mode='number_theory_proof',
        )
        self.assertEqual(report.strategy_prior.family, 'number_theory_proof')
        self.assertTrue(report.candidates)
        self.assertTrue(any('PARITY' in item for item in report.strategy_prior.recommended_operators))
        self.assertIn('Olympiad proof sketch', report.chosen_answer)
        self.assertTrue(report.solution_process)
        self.assertTrue(report.operator_trace)

    def test_reasoner_uses_diagram_world_model_for_geometry(self) -> None:
        report = WorldModelMathReasoner().solve(
            'In this geometry configuration, what proof strategy should I try first to show two angles are equal?',
            visual_input='examples/vlso/geometry_scene.json',
            task_mode='geometry_proof',
        )
        self.assertEqual(report.strategy_prior.family, 'geometry_proof')
        self.assertIsNotNone(report.visual_world_model)
        self.assertTrue(any(item.source == 'vlso_reasoner' for item in report.candidates))
        self.assertTrue(any('AUXILIARY_CONSTRUCTION' == item for item in report.strategy_prior.recommended_operators))


class WorldModelMathProductionServiceTests(unittest.TestCase):
    def test_service_accepts_strong_number_theory_request_and_logs_audit(self) -> None:
        tmp = _workspace_tempdir()
        try:
            audit_path = os.path.join(tmp, 'audit.jsonl')
            service = WorldModelMathProductionService(
                ProductionMathServiceConfig(audit_log_path=audit_path)
            )
            response = service.solve_request(
                'Prove that the sum of two odd integers is even.',
                source_context='Use parity rewriting and divisibility checks.',
                task_mode='number_theory_proof',
            )
            self.assertEqual(response.status, 'accepted')
            self.assertTrue(response.accepted)
            self.assertTrue(os.path.exists(audit_path))
            self.assertIn('Olympiad proof sketch', response.safe_answer)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_service_reviews_figure_dependent_geometry_without_visual(self) -> None:
        tmp = _workspace_tempdir()
        try:
            audit_path = os.path.join(tmp, 'audit.jsonl')
            service = WorldModelMathProductionService(
                ProductionMathServiceConfig(audit_log_path=audit_path)
            )
            response = service.solve_request(
                'In the given geometry configuration, what proof strategy should I try first to show two angles are equal?',
                task_mode='geometry_proof',
            )
            self.assertEqual(response.status, 'review')
            self.assertFalse(response.accepted)
            self.assertIn('figure_dependent_query_without_visual', response.decision.reasons)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_service_self_test_returns_structured_summary(self) -> None:
        tmp = _workspace_tempdir()
        try:
            audit_path = os.path.join(tmp, 'audit.jsonl')
            service = WorldModelMathProductionService(
                ProductionMathServiceConfig(audit_log_path=audit_path)
            )
            summary = service.self_test()
            self.assertGreaterEqual(summary.total_count, 2)
            self.assertEqual(len(summary.results), summary.total_count)
            self.assertIn(summary.status, {'ready', 'conditional'})
            self.assertIsNotNone(summary.readiness)
            self.assertIn('detected_profile', summary.readiness.hardware_profile)
            self.assertTrue(summary.readiness.operator_algebra_mode)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class WorldModelMathTrainingTests(unittest.TestCase):
    def test_trainer_emits_artifacts_and_scores_cases(self) -> None:
        tmp = _workspace_tempdir()
        try:
            cases_path = os.path.join(tmp, 'cases.jsonl')
            output_dir = os.path.join(tmp, 'out')
            _write_cases(cases_path)
            trainer = WorldModelMathTrainer()
            summary = trainer.train_from_cases(cases_path, output_dir, epochs=1)
            self.assertEqual(summary.total_cases, 2)
            self.assertGreater(summary.final_average_score, 0.6)
            self.assertIn('detected_profile', summary.hardware_profile)
            self.assertTrue(summary.operator_algebra_mode)
            self.assertTrue(os.path.exists(summary.artifact_paths['logical_weights']))
            self.assertTrue(os.path.exists(summary.artifact_paths['strategy_memory']))
            self.assertTrue(os.path.exists(summary.artifact_paths['leworldmodel_prior']))
            self.assertTrue(os.path.exists(summary.artifact_paths['leworldmodel_effect']))
            self.assertTrue(os.path.exists(summary.artifact_paths['training_summary']))
            self.assertIn('sigreg_score', summary.leworldmodel_metrics)
            self.assertIn('average_score_delta', summary.leworldmodel_effect)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_trainer_evaluates_cases(self) -> None:
        tmp = _workspace_tempdir()
        try:
            cases_path = os.path.join(tmp, 'cases.jsonl')
            output_dir = os.path.join(tmp, 'out')
            _write_cases(cases_path)
            trainer = WorldModelMathTrainer()
            summary = trainer.evaluate_cases(cases_path, output_dir)
            self.assertEqual(summary.total_cases, 2)
            self.assertGreater(summary.average_score, 0.6)
            self.assertTrue(os.path.exists(summary.output_path))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


    def test_reasoner_loads_trained_leworldmodel_prior(self) -> None:
        tmp = _workspace_tempdir()
        try:
            cases_path = os.path.join(tmp, 'cases.jsonl')
            output_dir = os.path.join(tmp, 'out')
            _write_cases(cases_path)
            trainer = WorldModelMathTrainer()
            trainer.train_from_cases(cases_path, output_dir, epochs=1)
            reasoner = WorldModelMathReasoner(leworldmodel_path=os.path.join(output_dir, 'math_leworldmodel_prior.json'))
            report = reasoner.solve('Prove that the sum of two odd integers is even.', task_mode='number_theory_proof')
            self.assertTrue(report.leworldmodel_alignment)
            self.assertTrue(report.leworldmodel_plan)
            self.assertGreaterEqual(float(report.leworldmodel_alignment.get('alignment_score', 0.0) or 0.0), 0.0)
            self.assertGreaterEqual(float(report.leworldmodel_plan.get('plan_score', 0.0) or 0.0), 0.0)
            self.assertTrue(any('lewm' in item.lower() for item in report.candidates[0].support))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class VisualGeometry3DWorkbenchTests(unittest.TestCase):
    def test_collect_starter_scenes_writes_assets(self) -> None:
        tmp = _workspace_tempdir()
        try:
            workbench = VisualGeometry3DWorkbench()
            summary = workbench.collect_starter_scenes(tmp)
            self.assertGreaterEqual(summary.scene_count, 5)
            self.assertTrue(os.path.exists(summary.eval_path))
            self.assertTrue(os.path.isdir(summary.scene_dir))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_collect_input_manifest_from_examples_folder(self) -> None:
        tmp = _workspace_tempdir()
        try:
            workbench = VisualGeometry3DWorkbench()
            source = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', 'vlso'))
            summary = workbench.collect_input_manifest(source, tmp)
            self.assertGreaterEqual(summary.input_count, 1)
            self.assertTrue(os.path.exists(summary.manifest_path))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_train_starter_bundle_writes_visual_stores(self) -> None:
        tmp = _workspace_tempdir()
        try:
            workbench = VisualGeometry3DWorkbench()
            summary = workbench.train_starter_bundle(tmp, limit_scenes=1)
            self.assertGreaterEqual(summary.image_count, 1)
            self.assertTrue(os.path.exists(summary.concept_store_path))
            self.assertTrue(os.path.exists(summary.operator_store_path))
            self.assertTrue(os.path.exists(os.path.join(tmp, 'lewm_visual_prior.json')))
            self.assertTrue(os.path.exists(os.path.join(tmp, 'lewm_visual_training_report.json')))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_train_bundle_from_custom_input_writes_visual_stores(self) -> None:
        tmp = _workspace_tempdir()
        try:
            workbench = VisualGeometry3DWorkbench()
            source = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', 'vlso', 'geometry_scene.json'))
            summary = workbench.train_bundle_from_inputs(source, tmp, limit_scenes=1)
            self.assertGreaterEqual(summary.image_count, 1)
            self.assertTrue(os.path.exists(summary.concept_store_path))
            self.assertTrue(os.path.exists(summary.operator_store_path))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_reconstruct_scene_builds_primitives_and_obj_bundle(self) -> None:
        tmp = _workspace_tempdir()
        try:
            workbench = VisualGeometry3DWorkbench()
            visual_input = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', 'vlso', 'geometry_scene.json'))
            workbench.train_bundle_from_inputs(visual_input, tmp, limit_scenes=1)
            summary = workbench.reconstruct_scene('Reconstruct the visible geometry and topology into 3D.', visual_input, tmp)
            self.assertTrue(summary.primitives)
            self.assertTrue(os.path.exists(os.path.join(tmp, 'scene_3d_reconstruction.json')))
            self.assertTrue(os.path.exists(summary.export_paths['obj_path']))
            self.assertGreater(summary.mesh_stats.get('vertex_count', 0), 0)
            self.assertIn('shape_counts', summary.topology_summary)
            self.assertTrue(summary.leworldmodel_alignment)
            self.assertTrue(summary.leworldmodel_plan)
            self.assertIn('leworldmodel_alignment_score', summary.topology_summary)
            self.assertIn('leworldmodel_plan_score', summary.topology_summary)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_reconstruct_batch_builds_multiple_obj_bundles(self) -> None:
        tmp = _workspace_tempdir()
        try:
            workbench = VisualGeometry3DWorkbench()
            source = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', 'vlso'))
            summary = workbench.reconstruct_batch('Reconstruct the visible geometry and topology into 3D.', source, tmp, limit=2)
            self.assertEqual(summary.reconstructed_count, 2)
            self.assertEqual(len(summary.obj_paths), 2)
            self.assertTrue(os.path.exists(os.path.join(tmp, 'batch_reconstruction_summary.json')))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class MathWorldModelGuiTests(unittest.TestCase):
    def test_home_page_renders_beginner_training_controls(self) -> None:
        app = math_world_model_gui.MathWorldModelGui()
        html = app.handle({})
        self.assertIn('Math and Visual World Model Studio', html)
        self.assertIn('Solve with production gate', html)
        self.assertIn('One-click starter train + eval', html)
        self.assertIn('One-click visual 3D bootcamp', html)
        self.assertIn('Reconstruct image or diagram into 3D', html)
        self.assertIn('Batch reconstruct folder or file', html)
        self.assertIn('Collect from folder or file', html)
        self.assertIn('Visual dataset source folder or file', html)
        self.assertIn('Math LeWM prior', html)
        self.assertIn('Visual LeWM prior', html)
        self.assertIn('LeWM plan', html)
        self.assertIn('QLoRA ready', html)
        self.assertIn('Missing optional packages', html)


class VisualGeometry3DCliTests(unittest.TestCase):
    def test_cli_bootcamp_writes_training_and_reconstruction_outputs(self) -> None:
        tmp = _workspace_tempdir()
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            command = [
                sys.executable,
                'run_visual_geometry_3d.py',
                'bootcamp',
                '--input-dir',
                'examples/vlso',
                '--output-dir',
                tmp,
                '--limit',
                '1',
                '--visual-input',
                'examples/vlso/geometry_scene.json',
            ]
            result = subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)
            self.assertIn('bootcamp', result.stdout)
            self.assertTrue(os.path.exists(os.path.join(tmp, 'visual_geometry_concepts.db')))
            self.assertTrue(os.path.exists(os.path.join(tmp, 'scene_3d_bundle', 'scene_3d_reconstruction.obj')))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations

import json
import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import semop_studio_gui


class StudioGuiRenderTests(unittest.TestCase):
    def test_home_page_shows_unified_chat_and_beginner_controls(self) -> None:
        app = semop_studio_gui.StudioApp()
        html = app.handle({})
        self.assertIn('Unified chat', html)
        self.assertIn('Send prompt', html)
        self.assertIn('One-click all-domain training', html)
        self.assertIn('Autopilot coach', html)
        self.assertIn('Do everything for me', html)
        self.assertIn('Capability audit', html)
        self.assertIn('Environment brain', html)
        self.assertIn('Learn this environment', html)
        self.assertIn('Self-improve this environment', html)
        self.assertIn('Recursive self-evolve', html)
        self.assertIn('Ultimate AGI', html)
        self.assertIn('Run ultimate AGI + commercialization audit', html)
        self.assertIn('RTX 4060 coach', html)
        self.assertIn('Check 4060 readiness', html)
        self.assertIn('Collect data + improve for 4060', html)
        self.assertIn('One-click collect + train stronger', html)
        self.assertIn('Collect + train + answer', html)
        self.assertIn('Improve weak areas + re-audit', html)
        self.assertIn('Manual fast path', html)
        self.assertIn('Save + approve current result', html)
        self.assertIn('Train approved reviews now', html)
        self.assertIn('ML dependencies', html)
        self.assertIn('QLoRA ready', html)

    def test_render_page_shows_active_job_banner(self) -> None:
        html = semop_studio_gui.render_page(
            semop_studio_gui.StudioState(),
            semop_studio_gui.ActionOutcome(result_kind='unified_chat', result_payload={'route': 'ops', 'status': 'completed'}),
            {
                'has_active': True,
                'running_count': 1,
                'queued_count': 0,
                'active': {
                    'label': 'Collect + train + answer',
                    'detail': 'Training the stronger local bundle.',
                    'progress': 0.42,
                    'remaining_eta': '3m 10s',
                    'job_id': 'job-1',
                    'events': [
                        {'message': 'Collect starter data'},
                        {'message': 'Train starter bundle'},
                    ],
                },
                'recent': [],
                'completed_count': 0,
                'saved_history_count': 0,
                'history_path': 'data/unified_semop_gui_run/studio_job_history.json',
            },
            {'recent': [], 'unread_count': 0, 'path': 'data/unified_semop_gui_run/studio_notifications.json'},
            {'messages': []},
        )
        self.assertIn('Now running', html)
        self.assertIn('Collect + train + answer', html)
        self.assertIn('Train starter bundle', html)

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

    def test_capability_render_mentions_readiness_delta(self) -> None:
        html = semop_studio_gui.render_result(
            'capability_improvement',
            {
                'before': {'overall_readiness_percent': 52.0},
                'after': {'overall_readiness_percent': 71.5, 'report_path': 'data/unified_semop_gui_run/capability_audit_report.json'},
                'delta_readiness_percent': 19.5,
                'weak_axes_before': ['semop_reasoning'],
                'weak_axes_after': ['generalization_proof'],
                'actions_taken': ['Seeded starter graphs.', 'Reran benchmark gate.'],
                'rounds': [{'round_index': 1, 'guided_seeded': 3, 'approved_reviews': 3, 'visual_seeded': 1}],
                'proof': {'report_path': 'data/unified_semop_gui_run/generalization_proof/generalization_proof_report.json', 'goal_tracker': {'readiness_percent': 60.0}},
                'output_path': 'data/unified_semop_gui_run/capability_improvement_report.json',
            },
        )
        self.assertIn('Before readiness', html)
        self.assertIn('After readiness', html)
        self.assertIn('19.500', html)

    def test_manual_review_render_mentions_queue_totals(self) -> None:
        html = semop_studio_gui.render_result(
            'manual_review',
            {
                'review_id': 7,
                'saved_status': 'approved',
                'query': 'Should I stop before moving?',
                'answer_text': 'Stop first and verify approval.',
                'reasons': ['grounding_review', 'approved_training_trace'],
                'source_used': 'manual_user_review',
                'review_snapshot': {'pending': 2, 'approved': 5, 'promotable': 4},
                'review_detail': {
                    'id': 7,
                    'query': 'Should I stop before moving?',
                    'status': 'approved',
                    'reasons': ['grounding_review', 'approved_training_trace'],
                    'answer_text': 'Stop first and verify approval.',
                },
            },
        )
        self.assertIn('Manual review fast-lane result', html)
        self.assertIn('Pending total', html)
        self.assertIn('Approved total', html)
        self.assertIn('manual_user_review', html)

    def test_unified_chat_render_mentions_route(self) -> None:
        html = semop_studio_gui.render_result(
            'unified_chat',
            {
                'route': 'math',
                'status': 'review',
                'prompt': '??? ABC?? ????',
                'answer_text': 'Attach a diagram for a stronger geometry answer.',
                'notes': ['Geometry questions are safer with a diagram.'],
            },
        )
        self.assertIn('Unified chat result', html)
        self.assertIn('Route', html)
        self.assertIn('math', html)

    def test_unified_chat_render_mentions_concept_fusion(self) -> None:
        html = semop_studio_gui.render_result(
            'unified_chat',
            {
                'route': 'ops',
                'status': 'completed',
                'prompt': 'Build a safer plan for a robot in a busy aisle.',
                'answer_text': 'Stop first and verify the safe path.',
                'concept_fusion': {
                    'headline': 'Context, action, and safety concepts were fused into reusable embodied priors.',
                    'hypotheses': [
                        {'label': 'Embodied action prior: safety gating + robot control', 'novelty_score': 0.74},
                    ],
                    'recommended_focus': ['Keep fused hypotheses grounded by explicit evidence or operator traces.'],
                },
            },
        )
        self.assertIn('Concept fusion', html)
        self.assertIn('Embodied action prior', html)

    def test_ultimate_agi_render_mentions_commercial_readiness(self) -> None:
        html = semop_studio_gui.render_result(
            'ultimate_agi_audit',
            {
                'overall_readiness_percent': 74.5,
                'commercial_status': 'preproduct',
                'headline': 'The stack is beyond pure research and is moving into pre-product readiness.',
                'priority_focus': 'Embodied autonomy readiness for driving and robotics',
                'completed_axes': ['Prompt understanding and logical reasoning'],
                'remaining_axes': ['Embodied autonomy readiness for driving and robotics'],
                'product_blockers': ['Grounded multimodal explanations are still weaker than compiler validity.'],
                'next_steps': ['Expand visual-control traces, safety reviews, and action-affordance verification.'],
                'axes': [
                    {
                        'label': 'Embodied autonomy readiness for driving and robotics',
                        'score': 0.61,
                        'target': 0.68,
                        'status': 'advancing',
                    }
                ],
                'concept_fusion_preview': {
                    'headline': 'Operator-level concept fusion generated reusable creative hypotheses.',
                    'hypotheses': [
                        {'label': 'Embodied action prior: safety gating + robot control', 'novelty_score': 0.74},
                    ],
                },
            },
        )
        self.assertIn('Ultimate AGI readiness result', html)
        self.assertIn('Commercial status', html)
        self.assertIn('Embodied autonomy readiness for driving and robotics', html)

    def test_unified_chat_render_mentions_prompt_understanding(self) -> None:
        html = semop_studio_gui.render_result(
            'unified_chat',
            {
                'route': 'video',
                'status': 'completed',
                'prompt': '? ???? ?? ???? ????',
                'answer_text': 'Observed 3 frames and aggregated the scene.',
                'notes': ['Frames aggregated: 3'],
                'prompt_understanding': {
                    'summary': 'Track the scene across frames and summarize the temporal situation.',
                    'hidden_context': ['The scene changes over time.'],
                    'hidden_constraints': ['The answer should stay grounded in visible evidence.'],
                    'required_inputs': ['Attach a frame folder or manifest.'],
                },
            },
        )
        self.assertIn('Prompt understanding', html)
        self.assertIn('Track the scene across frames', html)
        self.assertIn('Attach a frame folder or manifest', html)

    def test_chat_command_action_routes_data_collection_prompt(self) -> None:
        action = semop_studio_gui._chat_command_action('collect data and train stronger')
        self.assertEqual(action, 'run_data_flywheel')

    def test_chat_command_action_routes_korean_data_collection_prompt(self) -> None:
        action = semop_studio_gui._chat_command_action('\ub370\uc774\ud130 \uc218\uc9d1\ud574\uc11c \ub354 \uac15\ud558\uac8c \ud559\uc2b5\ud574\uc918')
        self.assertEqual(action, 'run_data_flywheel')

    def test_chat_command_action_routes_agi_audit_prompt(self) -> None:
        action = semop_studio_gui._chat_command_action('ultimate agi readiness')
        self.assertEqual(action, 'run_ultimate_agi_audit')

    def test_chat_command_action_routes_environment_learning_prompt(self) -> None:
        action = semop_studio_gui._chat_command_action('learn this environment')
        self.assertEqual(action, 'run_environment_brain')

    def test_chat_command_action_routes_adaptive_environment_prompt(self) -> None:
        action = semop_studio_gui._chat_command_action('self improve this environment')
        self.assertEqual(action, 'run_adaptive_environment_learning')

    def test_chat_command_action_routes_recursive_self_evolution_prompt(self) -> None:
        action = semop_studio_gui._chat_command_action('alpha evolve this environment')
        self.assertEqual(action, 'run_recursive_self_evolution')

    def test_environment_brain_render_mentions_mastery_and_routines(self) -> None:
        html = semop_studio_gui.render_result(
            'environment_brain',
            {
                'environment_name': 'warehouse_exception_exception_response',
                'stored_graph_count': 6,
                'auto_approved_review_count': 4,
                'mastery_scores': {
                    'environment_mastery': 0.78,
                    'grounding_strength': 0.74,
                    'safety_alignment': 0.83,
                },
                'stable_concepts': [
                    {'label': 'approval', 'support_count': 4},
                    {'label': 'blocked route', 'support_count': 3},
                ],
                'routine_patterns': [
                    {'label': 'verify approval before movement', 'support_count': 3},
                ],
                'hazard_patterns': ['blocked path', 'missing approval'],
                'next_probes': [
                    {'query': 'Which exact evidence must ground the next answer before action?'},
                ],
            },
        )
        self.assertIn('Environment brain result', html)
        self.assertIn('Environment mastery', html)
        self.assertIn('verify approval before movement', html)

    def test_adaptive_environment_render_mentions_local_intelligence(self) -> None:
        html = semop_studio_gui.render_result(
            'adaptive_environment_learning',
            {
                'environment_name': 'warehouse_exception_exception_response',
                'capability_scores': {
                    'local_intelligence': 0.79,
                    'self_reflection': 0.74,
                    'grounded_reasoning': 0.76,
                    'embodied_planning': 0.71,
                },
                'ready_axes': 3,
                'total_axes': 5,
                'improved_cases': 4,
                'grounding_cases_used': 6,
                'completed_skills': ['Hidden-context reasoning in this environment'],
                'remaining_gaps': ['Multimodal understanding of the same environment'],
                'next_actions': ['Attach one representative image or short clip from the same environment to add multimodal memory.'],
                'axes': [
                    {'label': 'Local autonomous intelligence', 'score': 0.79, 'target': 0.7},
                ],
                'action_rehearsals': [
                    {'label': 'Routine rehearsal 1: verify approval before movement', 'steps': [{'action': 'approval confirmed'}, {'action': 'verify approval before movement'}, {'action': 'Re-check the visible state and confirm the hazard is still controlled.'}]},
                ],
                'self_evolution': {'improved_cases': 4, 'approved_reviews': 2, 'final_grounding_score': 0.76},
                'temporal_scene': {'situation_summary': 'Door stays stable while a box appears and disappears.', 'stable_entities': ['door'], 'temporal_events': ['frame_2: appeared -> box']},
            },
        )
        self.assertIn('Adaptive environment learning result', html)
        self.assertIn('Local intelligence', html)
        self.assertIn('Self-reflection', html)
        self.assertIn('Embodied planning', html)
        self.assertIn('Routine rehearsal 1', html)

    def test_recursive_self_evolution_render_mentions_research_and_best_program(self) -> None:
        html = semop_studio_gui.render_result(
            'recursive_self_evolution',
            {
                'environment_name': 'warehouse_exception_exception_response',
                'final_best_score': 0.82,
                'score_delta': 0.11,
                'best_program': {
                    'label': 'Recombined program: evidence_guard, action_rehearsal, hidden_constraint',
                    'focus_tags': ['evidence_guard', 'action_rehearsal', 'hidden_constraint'],
                    'mutation_note': 'Recombined the best two parents to preserve strong ideas while widening focus coverage.',
                },
                'generations': [
                    {'generation_index': 1, 'best_score': 0.68, 'score_delta': 0.0},
                    {'generation_index': 2, 'best_score': 0.76, 'score_delta': 0.08},
                ],
                'research_principles': ['AlphaEvolve and FunSearch: keep a searchable population of programs and let automated evaluators decide what survives.'],
                'next_actions': ['keep evolving'],
                'deployed_summary': {'capability_scores': {'local_intelligence': 0.81, 'embodied_planning': 0.73}},
            },
        )
        self.assertIn('Recursive self-evolution result', html)
        self.assertIn('Research principles', html)
        self.assertIn('Recombined program', html)

    def test_data_flywheel_render_mentions_readiness_and_collection(self) -> None:
        html = semop_studio_gui.render_result(
            'data_flywheel',
            {
                'answer_text': 'Collected starter data and retrained the stronger local bundle.',
                'process_summary': 'Collected starter data, trained the starter bundle, ran the 4060 improvement loop, and rechecked readiness.',
                'execution_timeline': [
                    {'label': 'Collect starter data', 'status': 'done', 'detail': 'Seeded 12 starter graphs.'},
                    {'label': 'Run RTX 4060 improvement', 'status': 'done', 'detail': 'Improved readiness by 6.5 points.'},
                ],
                'used_inputs': ['prompt: collect data and train stronger', 'corpus store: data/semop_memory.db'],
                'generated_outputs': ['data/unified_semop_gui_run/rtx4060_reasoning_coach.json'],
                'notes': ['Starter graphs seeded: 12', '4060 readiness now: 88%'],
                'data_collection': {
                    'starter_graphs': 12,
                    'approved_starter_traces': 9,
                    'visual_starter_scenes': 4,
                    'next_batches': ['Grounded SOP reviews: current=12, next batch=8'],
                },
                'universal_bootcamp': {
                    'semop': {'autopilot_setup': {'total_seeded': 12, 'approved_review_count': 9}},
                    'math_training': {'final_average_score': 0.82},
                    'visual_bootcamp': {'collection': {'scene_count': 4}},
                },
                'rtx4060_improvement': {'delta_readiness_percent': 6.5, 'output_path': 'data/unified_semop_gui_run/rtx4060_reasoning_improvement.json'},
                'rtx4060_assessment': {
                    'overall_readiness_percent': 88.0,
                    'goal_readiness_percent': 84.0,
                    'overall_status': 'strong',
                    'performance_tactics': ['Keep symbolic-first mode.'],
                    'data_collection_lanes': [
                        {'label': 'Grounded SOP reviews', 'current_count': 12, 'suggested_next_batch': 8},
                    ],
                    'report_path': 'data/unified_semop_gui_run/rtx4060_reasoning_coach.json',
                },
            },
        )
        self.assertIn('One-click data collection and training result', html)
        self.assertIn('Readiness now', html)
        self.assertIn('Grounded SOP reviews', html)
        self.assertIn('6.5', html)
        self.assertIn('Step-by-step timeline', html)
        self.assertIn('Used inputs', html)

    def test_collect_train_execute_render_mentions_answer_and_training(self) -> None:
        html = semop_studio_gui.render_result(
            'collect_train_execute',
            {
                'process_summary': 'Collected starter data, trained a stronger bundle, improved the local stack, and then answered the current prompt.',
                'execution_timeline': [
                    {'label': 'Collect starter data', 'status': 'done', 'detail': 'Seeded 12 starter graphs.'},
                    {'label': 'Answer the current prompt', 'status': 'completed', 'detail': 'Answered through the ops route.'},
                ],
                'used_inputs': ['prompt: The aisle is blocked. What should I do?', 'corpus store: data/semop_memory.db'],
                'generated_outputs': ['data/unified_semop_gui_run/rtx4060_reasoning_coach.json'],
                'data_collection': {'starter_graphs': 12, 'approved_starter_traces': 9, 'visual_starter_scenes': 4},
                'rtx4060_assessment': {'overall_readiness_percent': 88.0, 'report_path': 'data/unified_semop_gui_run/rtx4060_reasoning_coach.json'},
                'rtx4060_improvement': {'output_path': 'data/unified_semop_gui_run/rtx4060_reasoning_improvement.json'},
                'answer_result': {
                    'route': 'ops',
                    'status': 'completed',
                    'answer_text': 'Stop first and verify approval before movement.',
                    'notes': ['Plan executability: 0.820'],
                    'prompt_understanding': {'summary': 'Hidden constraint and approval checks matter first.'},
                },
            },
        )
        self.assertIn('One-click collect, train, and answer result', html)
        self.assertIn('Final answer', html)
        self.assertIn('Stop first and verify approval before movement.', html)
        self.assertIn('What this button did', html)
        self.assertIn('Step-by-step timeline', html)
        self.assertIn('Used inputs', html)

    def test_chat_route_detects_video_manifest(self) -> None:
        temp_dir = Path('tests') / 'studio_gui_video_manifest_case'
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            manifest_path = temp_dir / 'frames.json'
            manifest_path.write_text(json.dumps({'frames': [{'visual_input': {'objects': [{'id': 'door', 'label': 'door', 'kind': 'opening'}]}}, {'visual_input': {'objects': [{'id': 'door', 'label': 'door', 'kind': 'opening'}, {'id': 'box', 'label': 'box', 'kind': 'container'}]}}]}, ensure_ascii=False), encoding='utf-8')
            route = semop_studio_gui._chat_route('summarize what is happening across these frames', str(manifest_path))
            self.assertEqual(route.get('kind'), 'video')
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_rtx4060_render_mentions_data_collection_plan(self) -> None:
        html = semop_studio_gui.render_result(
            'rtx4060_assessment',
            {
                'overall_readiness_percent': 83.2,
                'overall_status': 'usable',
                'detected_profile': 'rtx_4060_8gb',
                'operator_algebra_mode': 'symbolic_first_gpu_assist',
                'goal_readiness_percent': 77.4,
                'approved_review_total': 741,
                'approved_domain_slices': 8,
                'grounded_review_total': 96,
                'weak_axes': ['generalization_proof'],
                'remaining_goal_items': ['Reviewed corpus improvement proof'],
                'benchmark_blockers': ['grounded_explanation_fidelity below threshold'],
                'data_collection_lanes': [
                    {
                        'label': 'Grounded SOP reviews',
                        'current_count': 96,
                        'suggested_next_batch': 12,
                        'preferred_surface': 'SemOp Studio -> Unified chat + Manual fast path',
                    }
                ],
                'performance_tactics': ['Keep symbolic_first_gpu_assist mode.'],
                'beginner_actions': ['Use Unified chat first.'],
                'source_reports': {'capability_audit': 'data/unified_semop_gui_run/capability_audit_report.json'},
                'report_path': 'data/unified_semop_gui_run/rtx4060_reasoning_coach.json',
            },
        )
        self.assertIn('RTX 4060 reasoning coach result', html)
        self.assertIn('Grounded SOP reviews', html)
        self.assertIn('symbolic_first_gpu_assist', html)


    def test_unified_chat_handle_surfaces_prompt_understanding(self) -> None:
        app = semop_studio_gui.StudioApp()
        html = app.handle(
            {
                'action': ['run_unified_chat'],
                'chat_prompt': ['Before I open the box and take the tool, what should I check first?'],
            }
        )
        self.assertIn('Prompt understanding', html)
        self.assertIn('Likely domain', html)
        self.assertIn('general', html.lower())

    def test_unified_chat_vision_keeps_prompt_understanding_general(self) -> None:
        app = semop_studio_gui.StudioApp()
        image_path = Path(__file__).resolve()
        original = semop_studio_gui.build_vision_payload
        try:
            semop_studio_gui.build_vision_payload = lambda *args, **kwargs: {
                'kind': 'vision',
                'world': {
                    'entities': [{'id': 'shape_1', 'label': 'shape_1', 'modality': 'vision'}],
                    'relations': [{'source': 'shape_2', 'relation': 'PART_OF', 'target': 'shape_1'}],
                    'warnings': [],
                },
                'answer': {
                    'answer_text': 'The current local visual parser recovered 1 visible regions and 1 grounded relations, but it did not recover trustworthy semantic object labels for this image.',
                    'evidence': [],
                    'warnings': [],
                },
            }
            app.handle(
                {
                    'action': ['run_unified_chat'],
                    'ops_context': ['If the work zone is blocked or the task is not approved yet, stop first and check the alternate route and manager approval.'],
                    'ops_domain': ['warehouse_exception'],
                    'ops_scenario': ['exception_response'],
                    'chat_prompt': ['\uc774 \uc0ac\uc9c4\uc744 \uc124\uba85\ud574\ubd10'],
                    'chat_image': [str(image_path)],
                }
            )
            payload = app._last_outcome.result_payload
            self.assertEqual(payload.get('route'), 'vision')
            prompt_understanding = payload.get('prompt_understanding', {})
            self.assertEqual(prompt_understanding.get('likely_domain'), 'general')
            self.assertEqual(prompt_understanding.get('likely_scenario'), 'scene_understanding')
        finally:
            semop_studio_gui.build_vision_payload = original

    def test_render_vision_summary_surfaces_structural_only_reality_check(self) -> None:
        html = semop_studio_gui.render_result(
            'vision',
            {
                'world': {
                    'entities': [{'id': 'shape_1', 'label': 'shape_1', 'modality': 'vision'}],
                    'relations': [{'source': 'shape_2', 'relation': 'PART_OF', 'target': 'shape_1'}],
                    'metadata': {
                        'semantic_scene_summary': {
                            'backend': 'openclip_local',
                            'caption': 'This image is not yet being semantically understood at a human level by the current local vision stack.',
                        }
                    },
                    'warnings': ['heuristic visual fallback'],
                },
                'answer': {
                    'answer_text': 'This image is not yet being semantically understood at a human level by the current local vision stack.',
                    'answer_mode': 'structured',
                    'warnings': ['Scene semantic grounding is still structural-only for this image.'],
                    'scene_semantic_level': 'structural_only',
                },
            },
        )
        self.assertIn('Scene semantic level', html)
        self.assertIn('structural_only', html)
        self.assertIn('Reality check', html)
        self.assertIn('structural-only scene grounding', html)
        self.assertIn('Semantic backend', html)

    def test_render_unified_chat_summary_shows_semantic_caption_for_vision(self) -> None:
        html = semop_studio_gui.render_result(
            'unified_chat',
            {
                'route': 'vision',
                'status': 'completed',
                'prompt': '? ??? ?????.',
                'answer_text': 'This appears to be a first-person shooter game screenshot.',
                'prompt_understanding': {
                    'likely_domain': 'general',
                    'likely_scenario': 'scene_understanding',
                },
                'vision_payload': {
                    'answer': {
                        'scene_semantic_level': 'semantic_candidate',
                    },
                    'world': {
                        'metadata': {
                            'semantic_scene_summary': {
                                'backend': 'openclip_local',
                                'caption': 'This appears to be a first-person shooter game screenshot.',
                            }
                        }
                    },
                },
            },
        )
        self.assertIn('Semantic caption', html)
        self.assertIn('first-person shooter game screenshot', html)

    def test_render_vision_summary_shows_frontier_lane_status(self) -> None:
        html = semop_studio_gui.render_result(
            'vision',
            {
                'world': {
                    'entities': [{'id': 'shape_1', 'label': 'shape_1', 'modality': 'vision'}],
                    'relations': [],
                    'metadata': {
                        'frontier_scene_summary': {
                            'backend': 'frontier_vlm',
                            'backend_ready': True,
                            'family': 'qwen2_5_vl',
                            'answer_text': 'This appears to be a first-person shooter game screenshot.',
                        },
                        'scene_adjudication': {
                            'preferred_answer': 'This appears to be a first-person shooter game screenshot.',
                            'stack_level': 'frontier_adjudicated',
                            'confidence': 0.81,
                            'used_frontier': True,
                        },
                        'semantic_scene_summary': {
                            'backend': 'openclip_local',
                            'caption': 'This appears to be a first-person shooter game screenshot.',
                        },
                    },
                },
                'answer': {
                    'answer_text': 'This appears to be a first-person shooter game screenshot.',
                    'answer_mode': 'structured',
                    'warnings': [],
                    'scene_semantic_level': 'frontier_vlm',
                },
            },
        )
        self.assertIn('Frontier VLM', html)
        self.assertIn('Frontier backend', html)
        self.assertIn('Scene stack', html)
        self.assertIn('qwen2_5_vl', html)

    def test_render_unified_chat_summary_shows_frontier_scene_answer(self) -> None:
        html = semop_studio_gui.render_result(
            'unified_chat',
            {
                'route': 'vision',
                'status': 'completed',
                'prompt': '? ??? ?????.',
                'answer_text': 'This appears to be a game screenshot.',
                'prompt_understanding': {
                    'likely_domain': 'general',
                    'likely_scenario': 'scene_understanding',
                },
                'vision_payload': {
                    'answer': {
                        'scene_semantic_level': 'frontier_vlm',
                    },
                    'world': {
                        'metadata': {
                            'frontier_scene_summary': {
                                'backend': 'frontier_vlm',
                                'backend_ready': True,
                                'family': 'qwen2_5_vl',
                                'answer_text': 'This appears to be a first-person shooter game screenshot.',
                            },
                            'scene_adjudication': {
                                'preferred_answer': 'This appears to be a first-person shooter game screenshot.',
                                'stack_level': 'frontier_adjudicated',
                                'confidence': 0.81,
                                'used_frontier': True,
                            },
                            'semantic_scene_summary': {
                                'backend': 'openclip_local',
                                'caption': 'Fallback semantic caption.',
                            },
                        }
                    },
                },
            },
        )
        self.assertIn('Frontier scene answer', html)
        self.assertIn('Adjudicated scene answer', html)
        self.assertIn('frontier_vlm', html)

    def test_video_result_render_mentions_backend(self) -> None:
        html = semop_studio_gui.render_result(
            'video',
            {
                'input_kind': 'json_manifest',
                'frame_count': 3,
                'stable_entities': ['door'],
                'changed_entities': ['box'],
                'extraction_backend': 'manifest',
                'backend_support': {'pillow': True, 'imageio': False, 'opencv': False, 'ffmpeg': False},
                'fallback_hint': 'Frame folders and JSON manifests always work.',
                'situation_summary': 'Observed 3 frames and aggregated the scene.',
                'temporal_events': ['frame_2: appeared -> box'],
                'sampled_frames': [{'answer_text': 'door only'}, {'answer_text': 'door and box'}],
                'warnings': [],
            },
        )
        self.assertIn('Video-grounded situation result', html)
        self.assertIn('Backend', html)
        self.assertIn('Frame folders and JSON manifests always work', html)

    def test_unified_chat_action_routes_prompt_to_ops(self) -> None:
        app = semop_studio_gui.StudioApp()
        html = app.handle(
            {
                'action': ['run_unified_chat'],
                'chat_prompt': ['The aisle is blocked and approval is missing. What should I do?'],
            }
        )
        self.assertIn('Unified chat result', html)
        self.assertIn('Context reasoning', html)
        self.assertIn('ops', html)


class FrontierVisionStudioTests(unittest.TestCase):
    def test_home_page_mentions_frontier_vision_install(self) -> None:
        app = semop_studio_gui.StudioApp()
        html = app.handle({})
        self.assertIn('Frontier vision', html)
        self.assertIn('Install frontier vision bundle', html)

    def test_render_frontier_setup_result_shows_bundle_status(self) -> None:
        html = semop_studio_gui.render_result(
            'frontier_setup',
            {
                'target_root': 'models/vision/frontier',
                'detected_profile': 'rtx_4060_8gb',
                'recommended_bundle': [
                    {'family': 'qwen2_5_vl', 'installed': False, 'required': True},
                    {'family': 'florence2', 'installed': True, 'required': True},
                ],
                'installed_count': 1,
                'recommended_count': 2,
                'required_ready': False,
                'installed_families': ['florence2'],
                'missing_required': ['qwen2_5_vl'],
                'notes': ['Install the required frontier checkpoints.'],
            },
        )
        self.assertIn('Frontier vision setup result', html)
        self.assertIn('Required ready', html)
        self.assertIn('qwen2_5_vl', html)



if __name__ == '__main__':
    unittest.main()

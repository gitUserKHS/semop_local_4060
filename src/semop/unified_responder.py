from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .common_world_tasks import CommonWorldTaskEngine
from .commonsense_kb import concept_label
from .concept_fusion import ConceptFusionEngine
from .contest_programmer import CompetitiveProgrammingReasoner
from .domain_copilot import CopilotRequest, DomainCopilot
from .multimodal_scene_understanding import TemporalSceneReasoner
from .prompt_understanding import PromptUnderstandingAnalyzer
from .unified_world_model import UnifiedWorldModelEngine
from .visual_geometry_3d import VisualGeometry3DWorkbench
from .vlso.reasoner import VLSOReasoner
from .world_model_math_service import ProductionMathServiceConfig, WorldModelMathProductionService


TEMPORAL_INPUT_EXTENSIONS = {'.gif', '.mp4', '.mov', '.avi', '.webm', '.mkv'}


@dataclass
class UnifiedResponderConfig:
    ops_context: str = ''
    operating_domain: str = 'general'
    scenario_hint: str = 'qa'
    review_queue_path: str = ''
    vision_mode: str = 'deep'
    vision_answer_mode: str = 'structured'
    vision_concept_store: str = ''
    vision_operator_store: str = ''
    vision_weights: str = ''
    math_output_dir: str = 'data/math_world_model_gui_run'
    visual_output_dir: str = 'data/math_world_model_gui_run/visual_3d'

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedResponderResult:
    route: str
    status: str
    prompt: str
    answer_text: str
    notes: list[str] = field(default_factory=list)
    prompt_understanding: dict[str, Any] = field(default_factory=dict)
    concept_fusion: dict[str, Any] = field(default_factory=dict)
    detail_key: str = ''
    detail_payload: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    target: str = ''
    flash_text: str = ''
    flash_tone: str = 'success'
    domain: str = ''
    scenario: str = ''

    def model_dump(self) -> dict[str, Any]:
        payload = {
            'route': self.route,
            'status': self.status,
            'prompt': self.prompt,
            'answer_text': self.answer_text,
            'notes': list(self.notes),
            'prompt_understanding': dict(self.prompt_understanding),
            'concept_fusion': dict(self.concept_fusion),
        }
        if self.target:
            payload['target'] = self.target
        if self.domain:
            payload['domain'] = self.domain
        if self.scenario:
            payload['scenario'] = self.scenario
        if self.detail_key and self.detail_payload:
            payload[self.detail_key] = dict(self.detail_payload)
        payload.update(dict(self.extras))
        return payload


class UnifiedResponder:
    def __init__(
        self,
        config: UnifiedResponderConfig | None = None,
        *,
        prompt_analyzer: PromptUnderstandingAnalyzer | None = None,
        concept_fusion_engine: ConceptFusionEngine | None = None,
        ops_runner: Callable[[str, str, str, str], Any] | None = None,
        vision_payload_builder: Callable[[str, Any], dict[str, Any]] | None = None,
        video_payload_builder: Callable[[str, str], dict[str, Any]] | None = None,
        visual_reconstructor: Callable[[str, str], Any] | None = None,
        math_solver: Callable[[str, str, Any], Any] | None = None,
        cp_solver: Callable[[str], Any] | None = None,
        cp_problem_parser: Callable[[str], Any] | None = None,
        unified_world_model_engine: UnifiedWorldModelEngine | None = None,
        common_world_task_engine: CommonWorldTaskEngine | None = None,
    ) -> None:
        self.config = config or UnifiedResponderConfig()
        self.prompt_analyzer = prompt_analyzer or PromptUnderstandingAnalyzer()
        self.concept_fusion_engine = concept_fusion_engine or ConceptFusionEngine()
        self.ops_runner = ops_runner or self._default_ops_runner
        self.vision_payload_builder = vision_payload_builder or self._default_vision_payload_builder
        self.video_payload_builder = video_payload_builder or self._default_video_payload_builder
        self.visual_reconstructor = visual_reconstructor or self._default_visual_reconstructor
        self.math_solver = math_solver or self._default_math_solver
        self._cp_reasoner: CompetitiveProgrammingReasoner | None = None
        self.cp_solver = cp_solver or self._default_cp_solver
        self.cp_problem_parser = cp_problem_parser or self._default_cp_problem_parser
        self.unified_world_model_engine = unified_world_model_engine or UnifiedWorldModelEngine()
        self.common_world_task_engine = common_world_task_engine or CommonWorldTaskEngine()

    def respond(self, prompt: str, *, visual_input: str | dict[str, Any] | None = None, force_route: str = '') -> UnifiedResponderResult:
        route = {'kind': force_route, 'target': ''} if force_route else self.route_request(prompt, visual_input)
        kind = str(route.get('kind') or 'ops')
        target = str(route.get('target') or '')
        source_context = self.config.ops_context if kind == 'ops' else ''
        base = self.prompt_analyzer.analyze_base(
            prompt,
            kind,
            visual_input=self._visual_input_hint(visual_input),
            domain=self.config.operating_domain,
            scenario=self.config.scenario_hint,
            source_context=source_context,
        )
        if kind == 'action':
            answer_text = 'This prompt looks like a workflow or training request, so it should be handled by the product action lane.'
            prompt_understanding = base.model_dump()
            return UnifiedResponderResult(
                route='action',
                status='action_requested',
                prompt=prompt,
                answer_text=answer_text,
                notes=['Use the product action lane to queue this workflow.'],
                prompt_understanding=prompt_understanding,
                concept_fusion=self._build_concept_fusion(prompt, 'action', prompt_understanding, {'target': target}),
                target=target,
                flash_text='Unified chat detected a workflow action request.',
                flash_tone='neutral',
            )
        if kind in {'vision', 'video', 'visual_3d'} and not self._has_visual_input(visual_input):
            answer_text = (
                'Attach a valid local image, frame folder, GIF, video file, or JSON manifest first for visual reasoning.'
                if kind in {'vision', 'video'}
                else 'Attach a valid local image or diagram path first for 3D reconstruction.'
            )
            prompt_understanding = base.model_dump()
            return UnifiedResponderResult(
                route=kind,
                status='error',
                prompt=prompt,
                answer_text=answer_text,
                notes=['Example image path: data/scene.png', 'Temporal inputs: frame folder, GIF, MP4, or a JSON manifest with frames.'],
                prompt_understanding=prompt_understanding,
                concept_fusion=self._build_concept_fusion(prompt, kind, prompt_understanding, {}),
                flash_text=answer_text,
                flash_tone='error',
            )
        if kind == 'ops' and self._has_visual_input(visual_input):
            return self._respond_grounded_ops(prompt, visual_input, base)
        if kind == 'ops':
            return self._respond_ops(prompt, base)
        if kind == 'cp':
            return self._respond_cp(prompt, visual_input, base)
        if kind == 'vision':
            return self._respond_vision(prompt, visual_input, base)
        if kind == 'video':
            return self._respond_video(prompt, visual_input, base)
        if kind == 'visual_3d':
            return self._respond_visual_3d(prompt, visual_input, base)
        return self._respond_math(prompt, visual_input, base)

    def _respond_ops(self, prompt: str, base: Any) -> UnifiedResponderResult:
        domain, scenario = self._resolve_ops_domain_scenario(base)
        result = self.ops_runner(prompt, self.config.ops_context, domain, scenario)
        prompt_understanding = self.prompt_analyzer.enrich_with_ops_result(base, result).model_dump()
        notes = [
            f"Plan executability: {getattr(getattr(result, 'kpis', None), 'plan_executability', 0.0):.3f}",
            f"Relation recovery: {getattr(getattr(result, 'kpis', None), 'relation_recovery', 0.0):.3f}",
        ]
        frame = getattr(getattr(result, 'graph', None), 'context_frame', None)
        if frame is not None and getattr(frame, 'summary', ''):
            notes.append(str(frame.summary))
        integrated_world = self.unified_world_model_engine.from_graph(getattr(result, 'graph', None), source='ops')
        detail_payload = self._model_dump(result)
        answer_text, notes, extras = self._finalize_common_world_answer(
            prompt=prompt,
            route='ops',
            world=integrated_world,
            prompt_understanding=prompt_understanding,
            detail_payload=detail_payload,
            fallback_answer_text=str(getattr(result, 'answer_text', '') or ''),
            notes=notes,
        )
        return UnifiedResponderResult(
            route='ops',
            status='completed',
            prompt=prompt,
            answer_text=answer_text,
            notes=notes,
            prompt_understanding=prompt_understanding,
            concept_fusion=self._build_concept_fusion(prompt, 'ops', prompt_understanding, detail_payload),
            detail_key='ops_payload',
            detail_payload=detail_payload,
            extras=extras,
            flash_text='Unified chat routed your prompt to context reasoning.',
            flash_tone='success',
            domain=domain,
            scenario=scenario,
        )

    def _respond_grounded_ops(self, prompt: str, visual_input: Any, base: Any) -> UnifiedResponderResult:
        domain, scenario = self._resolve_ops_domain_scenario(base)
        result = self.ops_runner(prompt, self.config.ops_context, domain, scenario)
        vision_payload = self.vision_payload_builder(prompt, visual_input)
        prompt_summary = self.prompt_analyzer.enrich_with_ops_result(base, result)
        prompt_understanding = self.prompt_analyzer.enrich_with_visual_payload(prompt_summary, vision_payload, temporal=False).model_dump()
        world = vision_payload.get('world', {}) if isinstance(vision_payload.get('world'), dict) else {}
        answer = vision_payload.get('answer', {}) if isinstance(vision_payload.get('answer'), dict) else {}
        scene_semantic_level = str(answer.get('scene_semantic_level') or '').strip()
        adjudication = world.get('metadata', {}).get('scene_adjudication', {}) if isinstance(world.get('metadata'), dict) and isinstance(world.get('metadata', {}).get('scene_adjudication'), dict) else {}
        notes = [
            f"Plan executability: {getattr(getattr(result, 'kpis', None), 'plan_executability', 0.0):.3f}",
            f"Relation recovery: {getattr(getattr(result, 'kpis', None), 'relation_recovery', 0.0):.3f}",
            f"Visible entities: {len(world.get('entities') or [])}",
            f"Visible relations: {len(world.get('relations') or [])}",
            f"Scene stack: {adjudication.get('stack_level') or '-'}",
            f"Scene semantic level: {scene_semantic_level or '-'}",
        ]
        frame = getattr(getattr(result, 'graph', None), 'context_frame', None)
        if frame is not None and getattr(frame, 'summary', ''):
            notes.append(str(frame.summary))
        if scene_semantic_level == 'structural_only':
            notes.append('Visual grounding is still structural-only, so the operator graph carries more of the decision burden here.')
        integrated_world = self.unified_world_model_engine.merge(
            self.unified_world_model_engine.from_graph(getattr(result, 'graph', None), source='ops'),
            self.unified_world_model_engine.from_shared_world(world, source='vision', query=prompt),
            query=prompt,
        )
        detail_payload = self._model_dump(result)
        answer_text, notes, extras = self._finalize_common_world_answer(
            prompt=prompt,
            route='ops',
            world=integrated_world,
            prompt_understanding=prompt_understanding,
            detail_payload=detail_payload,
            fallback_answer_text=self._synthesize_grounded_ops_answer(result, vision_payload),
            notes=notes,
            extras={'vision_payload': vision_payload, 'grounding_mode': 'multimodal_ops'},
        )
        return UnifiedResponderResult(
            route='ops',
            status='completed',
            prompt=prompt,
            answer_text=answer_text,
            notes=notes,
            prompt_understanding=prompt_understanding,
            concept_fusion=self._build_concept_fusion(prompt, 'ops', prompt_understanding, {'ops': detail_payload, 'vision': vision_payload}),
            detail_key='ops_payload',
            detail_payload=detail_payload,
            extras=extras,
            flash_text='Unified chat fused visual grounding with context reasoning.',
            flash_tone='success',
            domain=domain,
            scenario=scenario,
        )

    def _respond_vision(self, prompt: str, visual_input: Any, base: Any) -> UnifiedResponderResult:
        payload = self.vision_payload_builder(prompt, visual_input)
        answer = payload.get('answer', {}) if isinstance(payload.get('answer'), dict) else {}
        world = payload.get('world', {}) if isinstance(payload.get('world'), dict) else {}
        answer_text = str(answer.get('answer_text') or 'Vision reasoning finished.')
        prompt_understanding = answer.get('prompt_understanding') if isinstance(answer.get('prompt_understanding'), dict) else {}
        if not prompt_understanding:
            prompt_understanding = self.prompt_analyzer.enrich_with_visual_payload(base, payload, temporal=False).model_dump()
            payload = dict(payload)
            payload['answer'] = dict(answer)
            payload['answer']['prompt_understanding'] = dict(prompt_understanding)
            answer = payload['answer']
        frontier_scene = world.get('metadata', {}).get('frontier_scene_summary', {}) if isinstance(world.get('metadata'), dict) and isinstance(world.get('metadata', {}).get('frontier_scene_summary'), dict) else {}
        adjudication = world.get('metadata', {}).get('scene_adjudication', {}) if isinstance(world.get('metadata'), dict) and isinstance(world.get('metadata', {}).get('scene_adjudication'), dict) else {}
        notes = [
            f"Openings or reachable candidates: {len(self._opening_candidates(world))}",
            f"Warnings: {len(world.get('warnings') or [])}",
            f"Likely scenario: {prompt_understanding.get('likely_scenario') or '-'}",
            f"Scene semantic level: {answer.get('scene_semantic_level') or '-'}",
            f"Frontier VLM: {'ready' if frontier_scene.get('backend_ready') else 'fallback'}",
            f"Scene stack: {adjudication.get('stack_level') or '-'}",
        ]
        if answer.get('scene_semantic_level') == 'structural_only':
            notes.append('This is still structural-only scene grounding, not strong human-level semantic vision.')
        integrated_world = self.unified_world_model_engine.from_shared_world(world, source='vision', query=prompt)
        answer_text, notes, extras = self._finalize_common_world_answer(
            prompt=prompt,
            route='vision',
            world=integrated_world,
            prompt_understanding=prompt_understanding,
            detail_payload=payload,
            fallback_answer_text=answer_text,
            notes=notes,
        )
        return UnifiedResponderResult(
            route='vision',
            status='completed',
            prompt=prompt,
            answer_text=answer_text,
            notes=notes,
            prompt_understanding=dict(prompt_understanding),
            concept_fusion=self._build_concept_fusion(prompt, 'vision', prompt_understanding, payload),
            detail_key='vision_payload',
            detail_payload=payload,
            extras=extras,
            flash_text='Unified chat routed your prompt to vision reasoning.',
            flash_tone='success',
        )

    def _respond_video(self, prompt: str, visual_input: Any, base: Any) -> UnifiedResponderResult:
        payload = self.video_payload_builder(prompt, str(visual_input or ''))
        prompt_understanding = self.prompt_analyzer.enrich_with_visual_payload(base, payload, temporal=True).model_dump()
        answer_text = str(payload.get('answer_text') or payload.get('situation_summary') or 'Video reasoning finished.')
        notes = [
            f"Frames aggregated: {payload.get('frame_count', 0)}",
            f"Stable entities: {len(payload.get('stable_entities') or [])}",
            f"Temporal events: {len(payload.get('temporal_events') or [])}",
            f"Video backend: {payload.get('extraction_backend') or '-'}",
        ]
        integrated_world = self.unified_world_model_engine.from_video_payload(prompt, payload)
        answer_text, notes, extras = self._finalize_common_world_answer(
            prompt=prompt,
            route='video',
            world=integrated_world,
            prompt_understanding=prompt_understanding,
            detail_payload=payload,
            fallback_answer_text=answer_text,
            notes=notes,
        )
        return UnifiedResponderResult(
            route='video',
            status='completed',
            prompt=prompt,
            answer_text=answer_text,
            notes=notes,
            prompt_understanding=prompt_understanding,
            concept_fusion=self._build_concept_fusion(prompt, 'video', prompt_understanding, payload),
            detail_key='video_payload',
            detail_payload=payload,
            extras=extras,
            flash_text='Unified chat routed your prompt to video situation reasoning.',
            flash_tone='success',
        )

    def _respond_visual_3d(self, prompt: str, visual_input: Any, base: Any) -> UnifiedResponderResult:
        reconstruction = self.visual_reconstructor(prompt, str(visual_input or ''))
        payload = self._model_dump(reconstruction)
        prompt_understanding = base.model_dump()
        notes = [
            f"Primitives: {len(payload.get('primitives') or [])}",
            f"Relations: {len(payload.get('relations') or [])}",
            f"Warnings: {len(payload.get('warnings') or [])}",
        ]
        integrated_world = self.unified_world_model_engine.from_reconstruction_payload(prompt, payload)
        answer_text, notes, extras = self._finalize_common_world_answer(
            prompt=prompt,
            route='visual_3d',
            world=integrated_world,
            prompt_understanding=prompt_understanding,
            detail_payload=payload,
            fallback_answer_text=str(payload.get('answer_text') or '3D reconstruction finished.'),
            notes=notes,
            extras={'reconstruction_path': str(Path(self.config.visual_output_dir) / 'scene_3d_reconstruction.json')},
        )
        return UnifiedResponderResult(
            route='visual_3d',
            status='completed',
            prompt=prompt,
            answer_text=answer_text,
            notes=notes,
            prompt_understanding=prompt_understanding,
            concept_fusion=self._build_concept_fusion(prompt, 'visual_3d', prompt_understanding, payload),
            detail_key='visual_3d_payload',
            detail_payload=payload,
            extras=extras,
            flash_text='Unified chat routed your prompt to 3D reconstruction.',
            flash_tone='success',
        )

    def _respond_math(self, prompt: str, visual_input: Any, base: Any) -> UnifiedResponderResult:
        response = self.math_solver(prompt, self.config.ops_context, visual_input)
        prompt_understanding = self.prompt_analyzer.enrich_with_math_response(base, response).model_dump()
        answer_text = str(getattr(response, 'safe_answer', '') or '')
        status = str(getattr(response, 'status', '') or 'completed')
        warnings = list(getattr(response, 'warnings', []) or [])
        notes = [f"Likely scenario: {prompt_understanding.get('likely_scenario') or '-'}"] + warnings[:3]
        decision = getattr(response, 'decision', None)
        if len(notes) == 1 and not warnings:
            decision_reasons = list(getattr(decision, 'reasons', []) or [])
            notes = [', '.join(str(item) for item in decision_reasons) or 'No warnings were emitted.']
        detail_payload = self._model_dump(response)
        extras = {
            'audit_log_path': str(getattr(response, 'audit_log_path', '') or ''),
            'decision': self._model_dump(decision),
        }
        math_training_path = Path(self.config.math_output_dir) / 'math_training_summary.json'
        extras['math_training_path'] = str(math_training_path) if math_training_path.exists() else ''
        integrated_world = self.unified_world_model_engine.from_math_report(getattr(response, 'report', None))
        answer_text, notes, extras = self._finalize_common_world_answer(
            prompt=prompt,
            route='math',
            world=integrated_world,
            prompt_understanding=prompt_understanding,
            detail_payload=detail_payload,
            fallback_answer_text=answer_text,
            notes=notes,
            extras=extras,
        )
        accepted = bool(getattr(response, 'accepted', False))
        return UnifiedResponderResult(
            route='math',
            status=status,
            prompt=prompt,
            answer_text=answer_text,
            notes=notes,
            prompt_understanding=prompt_understanding,
            concept_fusion=self._build_concept_fusion(prompt, 'math', prompt_understanding, detail_payload),
            detail_key='math_payload',
            detail_payload=detail_payload,
            extras=extras,
            flash_text='Unified chat routed your prompt to the math world model.',
            flash_tone='success' if accepted else 'neutral',
        )

    def _respond_cp(self, prompt: str, visual_input: Any, base: Any) -> UnifiedResponderResult:
        solution = self.cp_solver(prompt)
        structure = self.cp_problem_parser(prompt)
        prompt_understanding = self.prompt_analyzer.enrich_with_cp_result(base, solution, structure=structure).model_dump()
        detail_payload = self._model_dump(solution)
        if structure is not None and hasattr(structure, 'normalized_query'):
            detail_payload['problem_structure'] = asdict(structure)
        integrated_world = self.unified_world_model_engine.from_cp_solution(prompt, solution, structure=structure)
        notes = [
            f"Category: {getattr(solution, 'category', '-') or '-'}",
            f"Compile OK: {bool(getattr(solution, 'compile_ok', False))}",
            f"Validation OK: {bool(getattr(solution, 'validation_report', {}).get('overall_ok')) if isinstance(getattr(solution, 'validation_report', {}), dict) else False}",
        ]
        extras_seed = {'visual_input_hint': self._visual_input_hint(visual_input)} if visual_input else {}
        answer_text, notes, extras = self._finalize_common_world_answer(
            prompt=prompt,
            route='cp',
            world=integrated_world,
            prompt_understanding=prompt_understanding,
            detail_payload=detail_payload,
            fallback_answer_text='Competitive programming reasoning is structured on the shared world model, but the final summary is still weak.',
            notes=notes,
            extras=extras_seed,
        )
        return UnifiedResponderResult(
            route='cp',
            status='completed',
            prompt=prompt,
            answer_text=answer_text,
            notes=notes,
            prompt_understanding=prompt_understanding,
            concept_fusion=self._build_concept_fusion(prompt, 'cp', prompt_understanding, detail_payload),
            detail_key='cp_payload',
            detail_payload=detail_payload,
            extras=extras,
            flash_text='Unified chat routed your prompt to competitive programming reasoning.',
            flash_tone='success',
            domain='competitive_programming',
            scenario='competitive_programming',
        )

    def _resolve_ops_domain_scenario(self, base: Any) -> tuple[str, str]:
        domain = str(getattr(base, 'likely_domain', '') or self.config.operating_domain).strip() or 'general'
        scenario = str(getattr(base, 'likely_scenario', '') or self.config.scenario_hint).strip() or 'qa'
        if domain == 'warehouse_exception' and scenario in {'qa', 'generic_reasoning'}:
            scenario = 'exception_response'
        elif domain == 'warehouse_onboarding' and scenario in {'qa', 'generic_reasoning'}:
            scenario = 'onboarding'
        return domain, scenario

    def _default_ops_runner(self, prompt: str, context: str, domain: str, scenario: str) -> Any:
        request = CopilotRequest(query=prompt, context=context, domain=domain, scenario=scenario)
        return DomainCopilot(review_queue_path=self.config.review_queue_path or None).run(request)

    def _default_vision_payload_builder(self, prompt: str, visual_input: Any) -> dict[str, Any]:
        if isinstance(visual_input, str) and self._is_temporal_visual_input(visual_input):
            return self._default_video_payload_builder(prompt, visual_input)
        reasoner = VLSOReasoner(
            mode=self.config.vision_mode or 'deep',
            concept_store_path=self.config.vision_concept_store or None,
            operator_store_path=self.config.vision_operator_store or None,
            affordance_weights_path=self.config.vision_weights or None,
            answer_mode=self.config.vision_answer_mode or 'structured',
        )
        if isinstance(visual_input, dict):
            input_payload = visual_input
        else:
            image_path = str(visual_input or '').strip()
            input_payload = {'image_path': image_path, 'metadata': {'image_path': image_path}}
        world, answer = reasoner.answer(prompt, visual_input=input_payload)
        return {'kind': 'vision', 'world': world.model_dump(), 'answer': answer.model_dump()}

    def _default_video_payload_builder(self, prompt: str, visual_input: str) -> dict[str, Any]:
        summary = TemporalSceneReasoner(
            mode=self.config.vision_mode or 'deep',
            answer_mode=self.config.vision_answer_mode or 'structured',
            concept_store_path=self.config.vision_concept_store or None,
            operator_store_path=self.config.vision_operator_store or None,
            affordance_weights_path=self.config.vision_weights or None,
        ).summarize(prompt, visual_input)
        payload = summary.model_dump()
        payload['kind'] = 'video'
        return payload

    def _default_visual_reconstructor(self, prompt: str, visual_input: str) -> Any:
        workbench = VisualGeometry3DWorkbench(
            concept_store_path=self.config.vision_concept_store or None,
            operator_store_path=self.config.vision_operator_store or None,
            affordance_weights_path=self.config.vision_weights or None,
            mode=self.config.vision_mode or 'deep',
            answer_mode=self.config.vision_answer_mode or 'structured',
        )
        return workbench.reconstruct_scene(prompt, visual_input, self.config.visual_output_dir)

    def _default_math_solver(self, prompt: str, source_context: str, visual_input: Any) -> Any:
        output_dir = Path(self.config.math_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        service = WorldModelMathProductionService(
            ProductionMathServiceConfig(
                concept_store_path=self.config.vision_concept_store or None,
                operator_store_path=self.config.vision_operator_store or None,
                affordance_weights_path=self.config.vision_weights or None,
                logical_weight_path=str(output_dir / 'logical_pattern_weights.json'),
                strategy_memory_path=str(output_dir / 'math_strategy_memory.json'),
                leworldmodel_path=str(output_dir / 'math_leworldmodel_prior.json'),
                audit_log_path=str(output_dir / 'audit_log.jsonl'),
            )
        )
        return service.solve_request(
            prompt,
            source_context=source_context,
            visual_input=visual_input,
            task_mode='auto',
            metadata={'surface': 'unified_responder'},
        )

    def _ensure_cp_reasoner(self) -> CompetitiveProgrammingReasoner:
        if self._cp_reasoner is None:
            self._cp_reasoner = CompetitiveProgrammingReasoner()
        return self._cp_reasoner

    def _default_cp_solver(self, prompt: str) -> Any:
        return self._ensure_cp_reasoner().solve(prompt)

    def _default_cp_problem_parser(self, prompt: str) -> Any:
        return self._ensure_cp_reasoner().parse_problem(prompt)

    def _integrated_payload_bundle(self, world: Any) -> tuple[dict[str, Any], list[str]]:
        world_payload = self._model_dump(world)
        if not world_payload and hasattr(world, 'model_dump'):
            dumped = world.model_dump()
            world_payload = dict(dumped) if isinstance(dumped, dict) else {}
        if not world_payload:
            return {}, []
        reasoning = self.unified_world_model_engine.reason(world)
        return {
            'integrated_world': world_payload,
            'integrated_reasoning': reasoning.model_dump(),
            'integrated_reasoning_text': reasoning.summary or reasoning.to_text(),
        }, [
            f"Unified world entities: {len(world_payload.get('entities') or [])}",
            f"Unified world relations: {len(world_payload.get('relations') or [])}",
            f"Unified world operators: {len(world_payload.get('operators') or [])}",
        ]

    def _finalize_common_world_answer(
        self,
        *,
        prompt: str,
        route: str,
        world: Any,
        prompt_understanding: dict[str, Any],
        detail_payload: dict[str, Any],
        fallback_answer_text: str,
        notes: list[str],
        extras: dict[str, Any] | None = None,
    ) -> tuple[str, list[str], dict[str, Any]]:
        final_notes = list(notes)
        final_extras = dict(extras or {})
        integrated_extras, integrated_notes = self._integrated_payload_bundle(world)
        final_extras.update(integrated_extras)
        final_notes.extend(integrated_notes)
        task_result = self.common_world_task_engine.run(
            prompt,
            world,
            integrated_extras.get('integrated_reasoning', {}) if isinstance(integrated_extras, dict) else {},
            route=route,
            prompt_understanding=prompt_understanding,
            fallback_answer_text=fallback_answer_text,
            detail_payload=detail_payload,
        )
        final_extras['common_world_task'] = task_result.model_dump()
        final_notes.append(f"Common world task: {task_result.task_type}")
        answer_text = str(task_result.answer_text or fallback_answer_text or final_extras.get('integrated_reasoning_text', '') or '').strip()
        return answer_text, final_notes, final_extras

    def _build_concept_fusion(
        self,
        prompt: str,
        route: str,
        prompt_understanding: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self.concept_fusion_engine.build_summary(
            prompt=prompt,
            route=route,
            prompt_understanding=prompt_understanding if isinstance(prompt_understanding, dict) else {},
            payload=payload if isinstance(payload, dict) else {},
        ).model_dump()

    @staticmethod
    def route_request(prompt: str, visual_input: str | dict[str, Any] | None = None) -> dict[str, str]:
        action = UnifiedResponder._chat_command_action(prompt)
        if action:
            return {'kind': 'action', 'target': action}
        has_visual = UnifiedResponder._has_visual_input(visual_input)
        temporal_input = isinstance(visual_input, str) and UnifiedResponder._is_temporal_visual_input(visual_input)
        if UnifiedResponder._chat_3d_like(prompt):
            return {'kind': 'visual_3d', 'target': 'reconstruct'}
        if has_visual and (temporal_input or UnifiedResponder._chat_video_like(prompt)):
            return {'kind': 'video', 'target': 'summarize'}
        if has_visual and UnifiedResponder._chat_math_like(prompt):
            return {'kind': 'math', 'target': 'solve'}
        if has_visual and UnifiedResponder._chat_cp_like(prompt):
            return {'kind': 'cp', 'target': 'solve'}
        if has_visual and UnifiedResponder._chat_ops_like(prompt):
            return {'kind': 'ops', 'target': 'reason'}
        if has_visual:
            return {'kind': 'vision', 'target': 'answer'}
        if UnifiedResponder._chat_cp_like(prompt):
            return {'kind': 'cp', 'target': 'solve'}
        if UnifiedResponder._chat_math_like(prompt):
            return {'kind': 'math', 'target': 'solve'}
        return {'kind': 'ops', 'target': 'reason'}

    @staticmethod
    def _has_visual_input(visual_input: str | dict[str, Any] | None) -> bool:
        if isinstance(visual_input, dict):
            return bool(visual_input)
        candidate = Path(str(visual_input or '').strip())
        return bool(str(visual_input or '').strip()) and candidate.exists()

    @staticmethod
    def _visual_input_hint(visual_input: str | dict[str, Any] | None) -> str:
        if isinstance(visual_input, dict):
            metadata = visual_input.get('metadata', {}) if isinstance(visual_input.get('metadata'), dict) else {}
            image_path = str(metadata.get('image_path') or visual_input.get('image_path') or '').strip()
            return image_path or '[inline_visual_input]'
        return str(visual_input or '').strip()

    @staticmethod
    def _model_dump(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        model_dump = getattr(value, 'model_dump', None)
        if callable(model_dump):
            dumped = model_dump()
            return dict(dumped) if isinstance(dumped, dict) else {}
        return {}

    @staticmethod
    def _opening_candidates(world: dict[str, Any]) -> list[str]:
        entities = world.get('entities', []) if isinstance(world, dict) else []
        candidates: list[str] = []
        for entity in entities:
            if not isinstance(entity, dict) or entity.get('modality') != 'vision':
                continue
            labels = entity.get('attributes', {}).get('concept_labels') if isinstance(entity.get('attributes'), dict) else []
            upper = {str(item).upper() for item in labels} if isinstance(labels, list) else set()
            if {'ACCESS_OPENING_CANDIDATE', 'ACCESS_CONTROL_PART', 'ACCESS_PORT_CANDIDATE', 'EDGE_OPENING'} & upper:
                candidates.append(str(entity.get('label') or entity.get('id') or 'opening'))
        return candidates

    @staticmethod
    def _contains_any(text: str, terms: tuple[str, ...] | list[str]) -> bool:
        normalized = str(text or '').lower()
        return any(str(term).lower() in normalized for term in terms)

    @classmethod
    def _chat_command_action(cls, prompt: str) -> str:
        normalized = str(prompt or '').strip().lower()
        if not normalized:
            return ''
        if cls._contains_any(normalized, ('\ub370\uc774\ud130 \ubaa8\uc544', '\ub370\uc774\ud130 \uc218\uc9d1', '\uc218\uc9d1\ud574\uc11c \ud559\uc2b5', '\uc6d0\ud074\ub9ad \uc218\uc9d1', 'collect data', 'collect and train', 'collect and improve', 'train stronger', 'gather data and train')):
            return 'run_data_flywheel'
        if cls._contains_any(normalized, ('ultimate agi', 'agi readiness', 'commercial readiness', '\uc0c1\uc6a9\ud654 \uc810\uac80', '\uad81\uadf9 agi', '\uac1c\ubc1c \ub85c\ub4dc\ub9f5')):
            return 'run_ultimate_agi_audit'
        if cls._contains_any(normalized, ('alpha evolve this environment', 'recursive self evolve', 'recursive self-improve', 'recursive evolution')):
            return 'run_recursive_self_evolution'
        if cls._contains_any(normalized, ('self improve this environment', 'self-improve this environment', 'grow intelligence here', 'adaptive environment learning', '\uc774 \ud658\uacbd\uc5d0\uc11c \uc2a4\uc2a4\ub85c \uac1c\uc120\ud574', '\ud658\uacbd \uc790\uae30\uac1c\uc120', '\ud658\uacbd \uc790\uae30\ud559\uc2b5')):
            return 'run_adaptive_environment_learning'
        if cls._contains_any(normalized, ('learn this environment', 'environment brain', 'local environment learning', '\uc774 \ud658\uacbd\uc5d0\uc11c \uc2a4\uc2a4\ub85c \ubc30\uc6cc', '\ud658\uacbd \ud559\uc2b5', '\ud2b9\uc815 \ud658\uacbd \ud559\uc2b5')):
            return 'run_environment_brain'
        if cls._contains_any(normalized, ('\uc804\uccb4 \ud559\uc2b5', '\uc804\ubd80 \ud559\uc2b5', '\ubaa8\ub450 \ud559\uc2b5', 'full bootcamp', 'everything train', 'all-in-one train')):
            return 'run_universal_bootcamp'
        if cls._contains_any(normalized, ('\uac1c\uc120', '\ubcf4\uc644', '\ud5a5\uc0c1', 'improve', 'self-evolution', 'self evolution')):
            return 'run_capability_improvement'
        if cls._contains_any(normalized, ('capability audit', '\uc0c1\ud0dc \uc810\uac80', '\uc9c4\ub2e8', 'audit', 'readiness')):
            return 'run_capability_audit'
        if cls._contains_any(normalized, ('generalization proof', 'proof', '\ubc94\uc6a9\ud654 \uc99d\uba85', '\uc99d\uba85 \ub9ac\ud3ec\ud2b8')) and not cls._contains_any(normalized, ('\uc218\ud559', '\uc815\uc218', 'triangle', '\uc0bc\uac01\ud615', 'circle', '\uc6d0\uc758', 'prime')):
            return 'run_generalization_proof'
        if cls._contains_any(normalized, ('benchmark', 'gate', '\uac8c\uc774\ud2b8', '\ubc88\ub4e4 \ud3c9\uac00')):
            return 'run_unified_benchmark_gate'
        if cls._contains_any(normalized, ('\ud14c\uc2a4\ud2b8', 'test current', 'smoke test', '\ud604\uc7ac \ubc88\ub4e4 \ud14c\uc2a4\ud2b8')):
            return 'run_beginner_test'
        if cls._contains_any(normalized, ('\ud559\uc2b5', 'train', '\uc7ac\ud559\uc2b5', 'bootstrap', 'starter setup', 'bundle build', '\ubc88\ub4e4 \ud559\uc2b5')):
            return 'run_universal_bootcamp'
        return ''

    def _synthesize_grounded_ops_answer(self, result: Any, vision_payload: dict[str, Any]) -> str:
        graph = getattr(result, 'graph', None)
        blocked_lines: list[str] = []
        requires_lines: list[str] = []
        alternative_lines: list[str] = []
        if graph is not None:
            blocked_lines = [
                f"{self._graph_node_label(graph, edge.source)} -> {self._graph_node_label(graph, edge.target)}"
                for edge in getattr(graph, 'edges', [])
                if getattr(edge, 'relation', '') == 'BLOCKED_BY'
            ][:3]
            requires_lines = [
                f"{self._graph_node_label(graph, edge.source)} -> {self._graph_node_label(graph, edge.target)}"
                for edge in getattr(graph, 'edges', [])
                if getattr(edge, 'relation', '') == 'REQUIRES'
            ][:4]
            alternative_lines = [
                self._graph_node_label(graph, edge.target)
                for edge in getattr(graph, 'edges', [])
                if getattr(edge, 'relation', '') == 'ALTERNATIVE'
            ][:4]
        conclusion = '지금은 바로 진행하지 않는 쪽이 맞습니다.' if blocked_lines or requires_lines else '바로 실행하기보다 전제조건 확인이 먼저입니다.'
        lines = [conclusion]
        operator_reasoning: list[str] = []
        if blocked_lines:
            operator_reasoning.append('`BLOCKED_BY` 관계로 직접 진행이 막혀 있습니다: ' + ', '.join(blocked_lines))
        if requires_lines:
            operator_reasoning.append('`REQUIRES` 관계로 먼저 충족해야 하는 조건이 있습니다: ' + ', '.join(requires_lines))
        if alternative_lines:
            operator_reasoning.append('`ALTERNATIVE` 관계로 우회 가능한 흐름이 열려 있습니다: ' + ', '.join(dict.fromkeys(alternative_lines)))
        if operator_reasoning:
            lines.extend(['', '판단 이유:'])
            lines.extend(f'- {item}' for item in operator_reasoning[:3])
        plan = [step.action for step in getattr(graph, 'plan', []) if getattr(step, 'status', 'valid') == 'valid']
        if plan:
            lines.extend(['', '권장 순서:'])
            lines.extend(f'{index}. {item}' for index, item in enumerate(plan[:4], start=1))
        visual_lines = self._grounded_ops_visual_lines(vision_payload)
        if visual_lines:
            lines.extend(['', '시각 grounding:'])
            lines.extend(f'- {item}' for item in visual_lines[:3])
        cautions = list(dict.fromkeys(list(getattr(graph, 'warnings', []) or []) + list(getattr(graph, 'invalid_advice', []) or []))) if graph is not None else []
        if cautions:
            lines.extend(['', '\uc8fc\uc758:'])
            lines.extend(f'- {item}' for item in cautions[:2])
        return '\n'.join(lines).strip()

    @staticmethod
    def _graph_node_label(graph: Any, node_id: str) -> str:
        for node in getattr(graph, 'nodes', []) or []:
            if getattr(node, 'id', '') == node_id:
                return str(getattr(node, 'label', '') or node_id)
        return concept_label(node_id)

    @staticmethod
    def _grounded_ops_visual_lines(vision_payload: dict[str, Any]) -> list[str]:
        world = vision_payload.get('world', {}) if isinstance(vision_payload.get('world'), dict) else {}
        answer = vision_payload.get('answer', {}) if isinstance(vision_payload.get('answer'), dict) else {}
        metadata = world.get('metadata', {}) if isinstance(world.get('metadata'), dict) else {}
        semantic_scene = metadata.get('semantic_scene_summary', {}) if isinstance(metadata.get('semantic_scene_summary'), dict) else {}
        frontier_scene = metadata.get('frontier_scene_summary', {}) if isinstance(metadata.get('frontier_scene_summary'), dict) else {}
        adjudication = metadata.get('scene_adjudication', {}) if isinstance(metadata.get('scene_adjudication'), dict) else {}
        lines: list[str] = []
        preferred = str(adjudication.get('preferred_answer') or '').strip()
        caption = str(semantic_scene.get('caption') or '').strip()
        scene_level = str(answer.get('scene_semantic_level') or '').strip()
        if preferred and 'Best grounded answer' not in preferred and 'shape_' not in preferred:
            lines.append(preferred)
        elif caption:
            lines.append(caption)
        elif frontier_scene.get('answer_text'):
            frontier_text = str(frontier_scene.get('answer_text') or '').strip()
            if 'Best grounded answer' not in frontier_text and 'shape_' not in frontier_text:
                lines.append(frontier_text)
        if scene_level == 'structural_only':
            lines.append('현재 시각 스택은 아직 구조 신호 위주라서, 최종 판단에서는 질문의 제약과 연산자 추론 비중이 더 큽니다.')
        entity_count = len(world.get('entities') or [])
        relation_count = len(world.get('relations') or [])
        if entity_count or relation_count:
            lines.append(f'시각 월드 모델에는 엔티티 {entity_count}개, 관계 {relation_count}개가 들어왔습니다.')
        return lines[:3]

    @classmethod
    def _chat_math_like(cls, prompt: str) -> bool:
        normalized = str(prompt or '').lower()
        math_terms = (
            '\uc99d\uba85', '\uc815\uc218', '\uc218\uc5f4', '\ud568\uc218', '\ubd80\ub4f1\uc2dd', '\ub2e4\ud56d\uc2dd', '\uae30\ud558', '\ub3c4\ud615', '\uc0bc\uac01\ud615', '\uc0ac\uac01\ud615', '\uc6d0', '\uc704\uc0c1',
            'olympiad', 'geometry', 'theorem', 'prove', 'triangle', 'circle', 'prime', 'integer', 'polynomial', 'sequence',
            'integral', 'derivative', 'matrix', 'graph theory', 'combinatorics',
        )
        if cls._contains_any(normalized, math_terms):
            return True
        symbol_count = sum(normalized.count(symbol) for symbol in ('=', '+', '-', '^', '\u2220', '\u221a'))
        digit_count = sum(1 for char in normalized if char.isdigit())
        return symbol_count >= 2 or digit_count >= 3


    @classmethod
    def _chat_cp_like(cls, prompt: str) -> bool:
        normalized = str(prompt or '').lower()
        if not normalized:
            return False
        contest_terms = (
            'competitive programming', 'contest', 'algorithm', 'codeforces', 'atcoder', 'boj', 'baekjoon', 'c++', 'cpp',
            '\ubc31\uc900', '\uace0\ub824', '\uc54c\uace0\ub9ac\uc998', '\uc2dc\uac04 \uc81c\ud55c', '\uba54\ubaa8\ub9ac \uc81c\ud55c', '\uad6c\ud604', '\uc785\ub825', '\ucd9c\ub825',
        )
        structure_terms = (
            'array', 'queries', 'query', 'range', 'segment tree', 'fenwick', 'shortest path', 'grid', 'graph', 'tree', 'nodes', 'edges', 'dp', 'binary search',
            '\ubc30\uc5f4', '\ucffc\ub9ac', '\uad6c\uac04', '\uc138\uadf8\uba3c\ud2b8 \ud2b8\ub9ac', '\ud39c\uc735', '\ucd5c\ub2e8 \uacbd\ub85c', '\uadf8\ub9ac\ub4dc', '\uadf8\ub798\ud504', '\ud2b8\ub9ac', '\uc815\uc810', '\uac04\uc120', '\ub3d9\uc801 \uacc4\ud68d\ubc95', '\uc774\ubd84 \ud0d0\uc0c9',
        )
        if cls._contains_any(normalized, contest_terms) and cls._contains_any(normalized, structure_terms):
            return True
        score = sum(1 for term in structure_terms if str(term).lower() in normalized)
        return score >= 2 and cls._contains_any(normalized, ('solve', 'complexity', 'implementation', 'code', '\ud574\uacb0', '\ubcf5\uc7a1\ub3c4', '\ucf54\ub4dc'))

    @classmethod
    def _chat_ops_like(cls, prompt: str) -> bool:
        normalized = str(prompt or '').lower()
        if not normalized:
            return False
        situation_terms = (
            'blocked', 'blocking', 'approval', 'approved', 'permit', 'permission', 'safe', 'safety', 'risk', 'access', 'aisle', 'lane', 'corridor', 'route', 'warehouse',
            '통로', '막혀', '막혔', '막힌', '차단', '승인', '허가', '안전', '위험', '접근', '우회', '보고', '보류', '작업',
        )
        decision_terms = (
            'what should', 'should i', 'can i', 'how should', 'how do i', 'what do i do',
            '어떻게', '해야', '해도', '되나', '되나요', '할까', '괜찮', '가능', '어쩌',
        )
        return cls._contains_any(normalized, situation_terms) and cls._contains_any(normalized, decision_terms)

    @classmethod
    def _chat_3d_like(cls, prompt: str) -> bool:
        return cls._contains_any(str(prompt or '').lower(), ('3d', '\ucc28\uc6d0', '\uc7ac\uad6c\uc131', 'reconstruct', 'mesh', 'obj', 'topology', 'primitive'))

    @classmethod
    def _chat_video_like(cls, prompt: str) -> bool:
        return cls._contains_any(str(prompt or '').lower(), ('video', 'clip', 'frame', 'temporal', 'sequence', 'motion', '\uc601\uc0c1', '\ube44\ub514\uc624', '\ud504\ub808\uc784', '\uc7a5\uba74', '\uc6c0\uc9c1\uc784'))

    @staticmethod
    def _is_temporal_visual_input(path_value: str) -> bool:
        candidate = Path(str(path_value or '').strip())
        if not candidate.exists():
            return False
        if candidate.is_dir():
            return True
        if candidate.suffix.lower() in TEMPORAL_INPUT_EXTENSIONS:
            return True
        if candidate.suffix.lower() != '.json':
            return False
        try:
            payload = json.loads(candidate.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return False
        if isinstance(payload, dict):
            frames = payload.get('frames')
            return isinstance(frames, list) and len(frames) >= 1
        return isinstance(payload, list) and len(payload) >= 2

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .pipeline import StructuredMeaningPipeline


WAREHOUSE_EXCEPTION_TERMS = (
    'warehouse', 'aisle', 'lane', 'forklift', 'pallet', 'manager approval', 'manager', 'work zone', 'zone',
    'approval token', 'route blocker', 'blocked lane', 'tote', 'shelf', 'pick face',
    '창고', '통로', '막힌 통로', '승인', '관리자 승인', '허가', '작업 구역', '안전 구역', '지게차', '팔레트', '랙', '피킹',
)
WAREHOUSE_ONBOARDING_TERMS = (
    'onboarding', 'new worker', 'new hire', 'first day', 'walkthrough', 'training aisle',
    '온보딩', '신입', '신입 작업자', '첫날', '교육', '교육 통로',
)
VISUAL_TERMS = ('image', 'diagram', 'scene', 'photo', 'picture', 'screenshot', 'camera', 'visible')
TEMPORAL_TERMS = ('video', 'clip', 'frame', 'temporal', 'sequence', 'motion', 'timeline')
GEOMETRY_TERMS = ('geometry', 'triangle', 'circle', 'angle', 'parallel', 'perpendicular', 'quadrilateral')
PREREQUISITE_TERMS = ('before', 'first', 'prior', 'prerequisite', 'verify', 'check', 'confirm', '먼저', '사전', '미리', '확인', '점검', '승인')


@dataclass
class PromptUnderstandingSummary:
    route: str
    primary_intent: str
    hidden_context: list[str] = field(default_factory=list)
    hidden_constraints: list[str] = field(default_factory=list)
    required_inputs: list[str] = field(default_factory=list)
    likely_domain: str = 'general'
    likely_scenario: str = 'qa'
    route_reason: str = ''
    summary: str = ''

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class PromptUnderstandingAnalyzer:
    def __init__(self, mode: str = 'heuristic', model_id: str = 'Qwen/Qwen2.5-3B-Instruct') -> None:
        self.mode = mode
        self.model_id = model_id
        self._pipeline: StructuredMeaningPipeline | None = None

    def analyze_base(
        self,
        prompt: str,
        route: str,
        *,
        visual_input: str = '',
        domain: str = 'general',
        scenario: str = 'qa',
        source_context: str = '',
    ) -> PromptUnderstandingSummary:
        graph = self._graph_for(prompt, source_context=source_context, route=route)
        hidden_context = self._graph_hidden_context(graph)
        hidden_constraints = self._graph_hidden_constraints(graph)
        required_inputs = self._required_inputs(prompt, route, visual_input=visual_input)
        hidden_context.extend(self._heuristic_context_hints(prompt, route))
        hidden_constraints.extend(self._heuristic_constraint_hints(prompt, route))
        if route == 'ops' and visual_input:
            hidden_context.append('A local visual input is attached and should ground blockers, access state, and safety conditions.')
            hidden_constraints.append('The answer should fuse visible state with approvals, prerequisites, and safety constraints before suggesting an action.')
        likely_domain = self._infer_domain(prompt, source_context=source_context, route=route, configured_domain=domain, graph=graph)
        likely_scenario = self._infer_scenario(route, configured_scenario=scenario, graph=graph)
        primary_intent = self._primary_intent(route, graph=graph)
        route_reason = self._route_reason(route, visual_input=visual_input, graph=graph)
        hidden_context = self._unique(hidden_context)[:6]
        hidden_constraints = self._unique(hidden_constraints)[:6]
        required_inputs = self._unique(required_inputs)[:4]
        summary = self._build_summary(primary_intent, hidden_context, hidden_constraints, required_inputs)
        return PromptUnderstandingSummary(
            route=route,
            primary_intent=primary_intent,
            hidden_context=hidden_context,
            hidden_constraints=hidden_constraints,
            required_inputs=required_inputs,
            likely_domain=likely_domain,
            likely_scenario=likely_scenario,
            route_reason=route_reason,
            summary=summary,
        )

    def enrich_with_ops_result(self, base: PromptUnderstandingSummary, result: Any) -> PromptUnderstandingSummary:
        graph = getattr(result, 'graph', None)
        hidden_context = list(base.hidden_context)
        hidden_constraints = list(base.hidden_constraints)
        hidden_context.extend(self._graph_hidden_context(graph))
        hidden_constraints.extend(self._graph_hidden_constraints(graph))
        base.hidden_context = self._unique(hidden_context)[:6]
        base.hidden_constraints = self._unique(hidden_constraints)[:6]
        request = getattr(result, 'request', None)
        if request is not None:
            base.likely_domain = getattr(request, 'domain', base.likely_domain) or base.likely_domain
            base.likely_scenario = getattr(request, 'scenario', base.likely_scenario) or base.likely_scenario
        base.summary = self._build_summary(base.primary_intent, base.hidden_context, base.hidden_constraints, base.required_inputs)
        return base

    def enrich_with_visual_payload(
        self,
        base: PromptUnderstandingSummary,
        payload: dict[str, Any],
        *,
        temporal: bool = False,
    ) -> PromptUnderstandingSummary:
        hidden_context = list(base.hidden_context)
        hidden_constraints = list(base.hidden_constraints)
        if temporal:
            if payload.get('situation_summary'):
                hidden_context.insert(0, str(payload.get('situation_summary')))
            stable = payload.get('stable_entities') or []
            changed = payload.get('changed_entities') or []
            if stable:
                hidden_context.append('Stable scene anchors: ' + ', '.join(str(item) for item in stable[:4]))
            if changed:
                hidden_context.append('Changed entities: ' + ', '.join(str(item) for item in changed[:4]))
            temporal_events = payload.get('temporal_events') or []
            if temporal_events:
                hidden_constraints.append('Temporal evidence: ' + ' | '.join(str(item) for item in temporal_events[:3]))
            backend = str(payload.get('extraction_backend') or '').strip()
            if backend:
                hidden_context.append(f'Video extraction backend: {backend}')
            fallback_hint = str(payload.get('fallback_hint') or '').strip()
            if fallback_hint:
                hidden_constraints.append(fallback_hint)
        else:
            world = payload.get('world', {}) if isinstance(payload.get('world'), dict) else {}
            answer = payload.get('answer', {}) if isinstance(payload.get('answer'), dict) else {}
            entities = world.get('entities') or []
            relations = world.get('relations') or []
            warnings = world.get('warnings') or []
            if entities:
                hidden_context.append(f'Visible entities detected: {len(entities)}')
            if relations:
                hidden_context.append(f'Visible relations recovered: {len(relations)}')
            if answer.get('evidence'):
                hidden_context.append('Evidence anchors: ' + ', '.join(str(item) for item in answer.get('evidence', [])[:3]))
            if warnings:
                hidden_constraints.append('Visual warnings: ' + ' | '.join(str(item) for item in warnings[:3]))
        base.hidden_context = self._unique(hidden_context)[:6]
        base.hidden_constraints = self._unique(hidden_constraints)[:6]
        base.summary = self._build_summary(base.primary_intent, base.hidden_context, base.hidden_constraints, base.required_inputs)
        return base

    def enrich_with_math_response(self, base: PromptUnderstandingSummary, response: Any) -> PromptUnderstandingSummary:
        hidden_context = list(base.hidden_context)
        hidden_constraints = list(base.hidden_constraints)
        decision = getattr(response, 'decision', None)
        warnings = list(getattr(response, 'warnings', []) or [])
        if decision is not None:
            reasons = list(getattr(decision, 'reasons', []) or [])
            if reasons:
                hidden_constraints.append('Math gate reasons: ' + ', '.join(str(item) for item in reasons[:3]))
        if warnings:
            hidden_constraints.append('Math warnings: ' + ' | '.join(str(item) for item in warnings[:3]))
        if getattr(response, 'status', '') == 'review':
            hidden_context.append('The solver inferred that more grounding or human review may be needed.')
        base.hidden_context = self._unique(hidden_context)[:6]
        base.hidden_constraints = self._unique(hidden_constraints)[:6]
        base.summary = self._build_summary(base.primary_intent, base.hidden_context, base.hidden_constraints, base.required_inputs)
        return base

    def enrich_with_cp_result(
        self,
        base: PromptUnderstandingSummary,
        solution: Any,
        *,
        structure: Any | None = None,
    ) -> PromptUnderstandingSummary:
        hidden_context = list(base.hidden_context)
        hidden_constraints = list(base.hidden_constraints)
        if structure is not None:
            goals = list(getattr(structure, 'goal_types', []) or [])
            frames = list(getattr(structure, 'logical_frames', []) or [])
            if goals:
                hidden_context.append('CP goal types: ' + ', '.join(str(item) for item in goals[:4]))
            if frames:
                hidden_context.append('CP logical frames: ' + ', '.join(str(item) for item in frames[:4]))
        category = str(getattr(solution, 'category', '') or '').strip()
        if category:
            hidden_context.append('Chosen CP category: ' + category.replace('_', ' '))
        time_complexity = str(getattr(solution, 'time_complexity', '') or '').strip()
        memory_complexity = str(getattr(solution, 'memory_complexity', '') or '').strip()
        if time_complexity or memory_complexity:
            hidden_constraints.append('Complexity target: ' + ', '.join(item for item in [time_complexity, memory_complexity] if item))
        compile_ok = getattr(solution, 'compile_ok', None)
        validation = getattr(solution, 'validation_report', None)
        if compile_ok is False:
            hidden_constraints.append('The current CP candidate did not pass the compile check yet.')
        if isinstance(validation, dict) and validation.get('overall_ok') is False:
            hidden_constraints.append('The current CP candidate still has a validator failure to repair.')
        base.hidden_context = self._unique(hidden_context)[:6]
        base.hidden_constraints = self._unique(hidden_constraints)[:6]
        base.likely_domain = 'competitive_programming'
        base.likely_scenario = 'competitive_programming'
        base.summary = self._build_summary(base.primary_intent, base.hidden_context, base.hidden_constraints, base.required_inputs)
        return base

    def _graph_for(self, prompt: str, *, source_context: str, route: str) -> Any | None:
        if route in {'action', 'vision', 'video', 'visual_3d'}:
            return None
        try:
            if self._pipeline is None:
                self._pipeline = StructuredMeaningPipeline(mode=self.mode, model_id=self.model_id)
            return self._pipeline.run(prompt, source_context=source_context)
        except Exception:
            return None

    @staticmethod
    def _graph_hidden_context(graph: Any | None) -> list[str]:
        if graph is None:
            return []
        lines: list[str] = []
        frame = getattr(graph, 'context_frame', None)
        if frame is not None:
            if getattr(frame, 'summary', ''):
                lines.append(str(frame.summary))
            if getattr(frame, 'primary_goal', ''):
                lines.append('Primary goal: ' + str(frame.primary_goal).replace('_', ' '))
            if getattr(frame, 'reasoning_mode', ''):
                lines.append('Reasoning mode: ' + str(frame.reasoning_mode).replace('_', ' '))
            if getattr(frame, 'frame_type', ''):
                lines.append('Frame type: ' + str(frame.frame_type).replace('_', ' '))
        hidden_goals = list(getattr(graph, 'hidden_goals', []) or [])
        if hidden_goals:
            lines.append('Hidden goals: ' + ', '.join(str(item).replace('_', ' ') for item in hidden_goals[:3]))
        if getattr(graph, 'clarification_needed', False):
            lines.append('The prompt may be underspecified and could need clarification before execution.')
        symbolic = list(getattr(graph, 'symbolic_results', []) or [])
        if symbolic:
            top_result = symbolic[0]
            domain = getattr(top_result, 'domain', '')
            if domain:
                lines.append('Symbolic reasoning lane: ' + str(domain).replace('_', ' '))
        return lines

    @staticmethod
    def _graph_hidden_constraints(graph: Any | None) -> list[str]:
        if graph is None:
            return []
        lines: list[str] = []
        frame = getattr(graph, 'context_frame', None)
        if frame is not None:
            active_constraints = list(getattr(frame, 'active_constraints', []) or [])
            missing = list(getattr(frame, 'missing_requirements', []) or [])
            if active_constraints:
                lines.append('Active constraints: ' + ', '.join(str(item).replace('_', ' ') for item in active_constraints[:4]))
            if missing:
                lines.append('Missing requirements: ' + ', '.join(str(item).replace('_', ' ') for item in missing[:4]))
            risks = list(getattr(frame, 'risk_signals', []) or [])
            if risks:
                lines.append('Risk signals: ' + ' | '.join(str(item) for item in risks[:3]))
        warnings = list(getattr(graph, 'warnings', []) or [])
        if warnings:
            lines.append('Warnings: ' + ' | '.join(str(item) for item in warnings[:3]))
        required = list(getattr(graph, 'required_premises', []) or [])
        if required:
            lines.append('Required premises: ' + ', '.join(str(item).replace('_', ' ') for item in required[:4]))
        invalid_advice = list(getattr(graph, 'invalid_advice', []) or [])
        if invalid_advice:
            lines.append('Unsafe patterns blocked: ' + ', '.join(str(item) for item in invalid_advice[:3]))
        return lines

    def _required_inputs(self, prompt: str, route: str, *, visual_input: str) -> list[str]:
        lowered = str(prompt or '').lower()
        required: list[str] = []
        if route in {'vision', 'video'} and not visual_input:
            required.append('Attach a local image, frame folder, GIF, video file, or JSON manifest for grounded multimodal reasoning.')
        if route == 'video' and not visual_input:
            required.append('Frame folders and JSON manifests are the most stable temporal input format.')
        if route == 'math' and any(term in lowered for term in GEOMETRY_TERMS) and not visual_input:
            required.append('Attach a diagram path for stronger geometry grounding and safer proof steps.')
        if route == 'cp' and any(term in lowered for term in GEOMETRY_TERMS) and not visual_input:
            required.append('Attach a diagram only if the contest problem depends on geometry figures; otherwise keep the reasoning text-first.')
        if route == 'visual_3d' and not visual_input:
            required.append('Attach an image or diagram path so the 3D workbench has visible structure to reconstruct.')
        return required

    @staticmethod
    def _heuristic_context_hints(prompt: str, route: str) -> list[str]:
        lowered = str(prompt or '').lower()
        lines: list[str] = []
        if any(term in lowered for term in PREREQUISITE_TERMS):
            lines.append('The user is likely asking for prerequisite-aware guidance instead of a raw action only.')
        if route == 'video':
            lines.append('The user wants a scene-level explanation, not just per-frame labels.')
        if route == 'vision':
            lines.append('The request depends on visible entities and relations rather than text-only recall.')
        if route == 'math':
            lines.append('The request likely needs a proof path or verification trace, not only a final answer.')
        if route == 'cp':
            lines.append('The request likely needs constraint recovery, algorithm selection, and complexity verification together.')
        return lines

    @staticmethod
    def _heuristic_constraint_hints(prompt: str, route: str) -> list[str]:
        lowered = str(prompt or '').lower()
        lines: list[str] = []
        if route == 'ops' and any(token in lowered for token in ('blocked', 'approval', 'access', 'safe', 'risk', '\ub9c9\ud788', '\uc2b9\uc778', '\uc811\uadfc', '\uc548\uc804', '\uc704\ud5d8', '\ud1b5\ub85c', '\ud5c8\uac00')):
            lines.append('There may be hidden safety, access, or approval constraints that should be verified first.')
        if route in {'vision', 'video'}:
            lines.append('The answer should stay grounded in visible evidence and avoid hallucinating unseen entities.')
        if route == 'visual_3d':
            lines.append('The reconstruction should preserve coarse topology even when exact geometry is uncertain.')
        if route == 'action':
            lines.append('The prompt requests a workflow change such as training, testing, or evaluation.')
        if route == 'cp':
            lines.append('The answer should preserve contest constraints, algorithm fit, and validation signals before presenting code advice.')
        return lines

    def _infer_domain(self, prompt: str, *, source_context: str, route: str, configured_domain: str, graph: Any | None) -> str:
        text = str(prompt or '').lower() if route in {'math', 'cp', 'vision', 'video', 'visual_3d'} else f'{prompt} {source_context}'.lower()
        if any(term in text for term in WAREHOUSE_ONBOARDING_TERMS):
            return 'warehouse_onboarding'
        if any(term in text for term in WAREHOUSE_EXCEPTION_TERMS):
            return 'warehouse_exception'
        graph_domain = str(getattr(graph, 'domain', '') or '').strip()
        if graph_domain and graph_domain != 'general':
            return graph_domain
        if route == 'cp':
            return 'competitive_programming'
        if route in {'math', 'vision', 'video', 'visual_3d'}:
            return 'general'
        if configured_domain and configured_domain != 'general' and any(term in text for term in ('warehouse', 'aisle', 'lane', 'manager', 'zone', 'approval', '\ucc3d\uace0', '\ud1b5\ub85c', '\uc2b9\uc778', '\ud5c8\uac00', '\uc791\uc5c5 \uad6c\uc5ed')):
            return configured_domain
        return 'general'

    @staticmethod
    def _infer_scenario(route: str, *, configured_scenario: str, graph: Any | None) -> str:
        if route == 'math':
            return 'math_reasoning'
        if route == 'cp':
            return 'competitive_programming'
        if route == 'vision':
            return 'scene_understanding'
        if route == 'video':
            return 'temporal_scene_understanding'
        if route == 'visual_3d':
            return 'topology_reconstruction'
        frame = getattr(graph, 'context_frame', None)
        frame_type = str(getattr(frame, 'frame_type', '') or '').strip()
        if frame_type:
            return frame_type
        return configured_scenario or 'qa'

    @staticmethod
    def _primary_intent(route: str, graph: Any | None) -> str:
        frame = getattr(graph, 'context_frame', None)
        primary_goal = str(getattr(frame, 'primary_goal', '') or '').strip().replace('_', ' ')
        if route == 'ops' and primary_goal:
            return f'Infer the hidden context and answer logically toward the goal: {primary_goal}.'
        return {
            'ops': 'Infer hidden context and give a logically safe next-step answer.',
            'vision': 'Describe the visible scene and answer from grounded visual evidence.',
            'video': 'Track the scene across frames and summarize the temporal situation.',
            'visual_3d': 'Reconstruct the visible structure into a coarse 3D/topological view.',
            'math': 'Build a rigorous solution or proof path for the problem.',
            'cp': 'Recover constraints and choose an algorithm on one shared world model before giving code advice.',
            'action': 'Launch a training, test, or improvement workflow.',
        }.get(route, 'Interpret the request and route it to the safest reasoning path.')

    @staticmethod
    def _route_reason(route: str, *, visual_input: str, graph: Any | None) -> str:
        frame = getattr(graph, 'context_frame', None)
        frame_type = str(getattr(frame, 'frame_type', '') or '').strip().replace('_', ' ')
        if route == 'video':
            return 'The prompt or input path looks temporal, so the system aggregates multiple frames into one situation model.'
        if route == 'vision':
            return 'A visual input is attached, so grounded scene reasoning takes priority over text-only inference.'
        if route == 'ops':
            detail = f' The prompt structure suggests {frame_type}.' if frame_type else ''
            if visual_input:
                return 'The prompt asks for situational reasoning, and the attached visual input should ground blockers, permissions, and safe alternatives.' + detail
            return 'The prompt looks like a situational reasoning question, so hidden goals, constraints, and safe actions are inferred.' + detail
        if route == 'math':
            return 'The prompt looks like a mathematical problem, so the world-model math solver is used.'
        if route == 'cp':
            return 'The prompt looks like a competitive programming problem, so constraint recovery, algorithm ranking, and verification are combined.'
        if route == 'visual_3d':
            return 'The prompt asks for reconstruction, so the geometry/topology 3D workbench is used.'
        if route == 'action':
            return 'The prompt requests a workflow change such as training, testing, or evaluation.'
        if visual_input:
            return 'A local visual input is attached, so multimodal reasoning takes priority.'
        return 'The route was chosen from the prompt semantics.'

    @classmethod
    def _build_summary(
        cls,
        primary_intent: str,
        hidden_context: list[str],
        hidden_constraints: list[str],
        required_inputs: list[str],
    ) -> str:
        parts = [primary_intent]
        if hidden_context:
            parts.append('Hidden context: ' + '; '.join(hidden_context[:2]))
        if hidden_constraints:
            parts.append('Constraints: ' + '; '.join(hidden_constraints[:2]))
        if required_inputs:
            parts.append('Helpful input: ' + '; '.join(required_inputs[:1]))
        return ' '.join(part for part in parts if part).strip()

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            normalized = str(item or '').strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

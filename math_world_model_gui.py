from __future__ import annotations

import argparse
import html
import json
import os
import sys
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from semop import (
    ProductionMathServiceConfig,
    VisualGeometry3DWorkbench,
    WorldModelMathProductionService,
    WorldModelMathTrainer,
    detect_local_hardware,
    detect_local_ml_stack,
    ensure_starter_math_cases,
)


GEOMETRY_EXAMPLE = {
    'query': 'In the given geometry configuration, what proof strategy should I try first to show two angles are equal?',
    'context': 'Focus on latent geometric structure first. Recover the key objects, then search for angle chase, similar triangles, or an auxiliary construction.',
    'visual_input': 'examples/vlso/geometry_scene.json',
    'task_mode': 'geometry_proof',
}
OLYMPIAD_EXAMPLE = {
    'query': 'Prove that the sum of two odd integers is even.',
    'context': 'Prefer proof state search with parity rewriting and divisibility checks.',
    'visual_input': '',
    'task_mode': 'number_theory_proof',
}
RESEARCH_EXAMPLE = {
    'query': 'Show that every finite set of real numbers contains an element that is at most the average of the set.',
    'context': 'Treat this as a research-grade proof search task: first identify the extremal object and then test contradiction routes.',
    'visual_input': '',
    'task_mode': 'research_math',
}
VISUAL_3D_EXAMPLE = {
    'visual3d_query': 'Reconstruct the visible geometry and topology into a simple 3D scene. Preserve stable shapes and relations if they are visible.',
    'visual3d_input': 'examples/vlso/geometry_scene.json',
}
TASK_MODE_OPTIONS = (
    'auto',
    'geometry_proof',
    'computational_geometry',
    'number_theory_proof',
    'algebra_proof',
    'combinatorics_proof',
    'research_math',
    'generic_math',
)


@dataclass(frozen=True)
class MathGuiState:
    query: str = OLYMPIAD_EXAMPLE['query']
    context: str = OLYMPIAD_EXAMPLE['context']
    visual_input: str = OLYMPIAD_EXAMPLE['visual_input']
    task_mode: str = OLYMPIAD_EXAMPLE['task_mode']
    concept_store: str = 'data/vlso_visual_prototypes.db'
    operator_store: str = 'data/vlso_visual_operators.db'
    affordance_weights: str = 'data/vlso_samples/trained_affordance_weights.json'
    training_cases_path: str = 'examples/math_world_model_starter.jsonl'
    output_dir: str = 'data/math_world_model_gui_run'
    epochs: int = 2
    visual3d_query: str = VISUAL_3D_EXAMPLE['visual3d_query']
    visual3d_input: str = VISUAL_3D_EXAMPLE['visual3d_input']
    visual3d_output_dir: str = 'data/math_world_model_gui_run/visual_3d'
    visual3d_data_source: str = 'examples/vlso'
    visual3d_scene_limit: int = 3

    def apply_form(self, form: dict[str, list[str]]) -> 'MathGuiState':
        return replace(
            self,
            query=_first(form, 'query', self.query),
            context=_first(form, 'context', self.context),
            visual_input=_first(form, 'visual_input', self.visual_input),
            task_mode=_first(form, 'task_mode', self.task_mode),
            concept_store=_first(form, 'concept_store', self.concept_store),
            operator_store=_first(form, 'operator_store', self.operator_store),
            affordance_weights=_first(form, 'affordance_weights', self.affordance_weights),
            training_cases_path=_first(form, 'training_cases_path', self.training_cases_path),
            output_dir=_first(form, 'output_dir', self.output_dir),
            epochs=max(1, _first_int(form, 'epochs', self.epochs)),
            visual3d_query=_first(form, 'visual3d_query', self.visual3d_query),
            visual3d_input=_first(form, 'visual3d_input', self.visual3d_input),
            visual3d_output_dir=_first(form, 'visual3d_output_dir', self.visual3d_output_dir),
            visual3d_data_source=_first(form, 'visual3d_data_source', self.visual3d_data_source),
            visual3d_scene_limit=max(1, _first_int(form, 'visual3d_scene_limit', self.visual3d_scene_limit)),
        )


@dataclass(frozen=True)
class MathGuiOutcome:
    flash: str = ''
    tone: str = 'neutral'
    payload: dict[str, object] | None = None


def _first(form: dict[str, list[str]], key: str, default: str = '') -> str:
    values = form.get(key)
    return values[0] if values else default


def _first_int(form: dict[str, list[str]], key: str, default: int) -> int:
    raw = _first(form, key, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _render_select(name: str, current: str) -> str:
    options = []
    for option in TASK_MODE_OPTIONS:
        selected = ' selected' if option == current else ''
        options.append(f"<option value='{html.escape(option)}'{selected}>{html.escape(option)}</option>")
    return f"<select name='{html.escape(name)}'>{''.join(options)}</select>"


def _metric(label: str, value: object) -> str:
    return f"<div class='metric'><small>{html.escape(label)}</small><strong>{html.escape(str(value))}</strong></div>"


def _info(label: str, value: object) -> str:
    if isinstance(value, list):
        body = '<br>'.join(html.escape(str(item)) for item in value) if value else '-'
    elif isinstance(value, dict):
        body = '<br>'.join(f"{html.escape(str(k))}: {html.escape(str(v))}" for k, v in value.items()) if value else '-'
    else:
        body = html.escape(str(value or '-'))
    return f"<div class='info'><small>{html.escape(label)}</small><div>{body}</div></div>"


def _artifact_status(path: str) -> str:
    return 'ready' if path and os.path.exists(path) else 'missing'


def _hardware_profile_payload() -> dict[str, object]:
    return detect_local_hardware().model_dump()


def _dependency_payload() -> dict[str, object]:
    return detect_local_ml_stack().model_dump()


def _read_json_if_exists(path: str) -> dict[str, object]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_for_script(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')


def _render_scene_canvas(scene_payload: dict[str, object], canvas_id: str = 'scene3d-canvas') -> str:
    primitives = scene_payload.get('primitives', [])
    if not isinstance(primitives, list) or not primitives:
        return "<div class='empty'>No 3D primitives were reconstructed yet.</div>"
    payload_id = f'{canvas_id}-payload'
    return f"""
<div class='paint-card'>
  <div class='paint-title'>3D paint preview</div>
  <canvas id='{canvas_id}' width='760' height='420'></canvas>
  <script id='{payload_id}' type='application/json'>{_json_for_script(scene_payload)}</script>
  <script>
  (function() {{
    const source = document.getElementById('{payload_id}');
    const canvas = document.getElementById('{canvas_id}');
    if (!source || !canvas || !canvas.getContext) return;
    const scene = JSON.parse(source.textContent || '{{}}');
    const items = Array.isArray(scene.primitives) ? scene.primitives.slice() : [];
    if (!items.length) return;
    const ctx = canvas.getContext('2d');
    const ox = 390;
    const oy = 280;
    const scale = 18;
    function project(x, y, z) {{
      return {{ x: ox + (x - y) * scale * 0.9, y: oy - z * scale + (x + y) * scale * 0.45 }};
    }}
    function shade(hex, factor) {{
      const raw = String(hex || '#6c7a89').replace('#', '').padEnd(6, '0').slice(0, 6);
      const rgb = [0, 2, 4].map((i) => parseInt(raw.slice(i, i + 2), 16));
      return '#' + rgb.map((v) => Math.max(0, Math.min(255, Math.round(v * factor))).toString(16).padStart(2, '0')).join('');
    }}
    function polygon(points, fill) {{
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i += 1) ctx.lineTo(points[i].x, points[i].y);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.strokeStyle = '#26423a';
      ctx.lineWidth = 1.2;
      ctx.fill();
      ctx.stroke();
    }}
    function label(point, text) {{
      ctx.fillStyle = '#14211c';
      ctx.font = '12px Aptos, Segoe UI, sans-serif';
      ctx.fillText(String(text || ''), point.x + 6, point.y - 6);
    }}
    function drawBox(item) {{
      const x = Number(item.x || 0), y = Number(item.y || 0), z = Number(item.z || 0);
      const w = Math.max(0.7, Number(item.width || 1)), h = Math.max(0.7, Number(item.height || 1)), d = Math.max(0.7, Number(item.depth || 1));
      const color = item.color || '#5d6eb3';
      const top = [project(x - w / 2, y - h / 2, z + d), project(x + w / 2, y - h / 2, z + d), project(x + w / 2, y + h / 2, z + d), project(x - w / 2, y + h / 2, z + d)];
      const front = [project(x - w / 2, y + h / 2, z), project(x + w / 2, y + h / 2, z), project(x + w / 2, y + h / 2, z + d), project(x - w / 2, y + h / 2, z + d)];
      const side = [project(x + w / 2, y - h / 2, z), project(x + w / 2, y + h / 2, z), project(x + w / 2, y + h / 2, z + d), project(x + w / 2, y - h / 2, z + d)];
      polygon(side, shade(color, 0.82));
      polygon(front, shade(color, 0.98));
      polygon(top, shade(color, 1.15));
      label(project(x, y, z + d + 0.4), (item.label || item.shape || 'shape') + ' [' + (item.shape || 'box') + ']');
    }}
    ctx.fillStyle = '#fbf8f1';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#d8ddd2';
    for (let i = -6; i <= 6; i += 1) {{
      const a = project(-8, i, 0), b = project(8, i, 0), c = project(i, -8, 0), d = project(i, 8, 0);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(d.x, d.y); ctx.stroke();
    }}
    items.sort((a, b) => (Number(a.x || 0) + Number(a.y || 0) + Number(a.z || 0)) - (Number(b.x || 0) + Number(b.y || 0) + Number(b.z || 0))).forEach(drawBox);
  }})();
  </script>
</div>
"""

def _render_visual_collection(payload: dict[str, object]) -> str:
    files = payload.get('scene_files', []) if isinstance(payload.get('scene_files'), list) else []
    metrics = ''.join([
        _metric('Scenes', payload.get('scene_count', 0)),
        _metric('Scene dir', payload.get('scene_dir', '-')),
        _metric('Eval JSONL', _artifact_status(str(payload.get('eval_path', '')))),
        _metric('Files', len(files)),
    ])
    return ''.join([
        f"<div class='grid metrics'>{metrics}</div>",
        _info('Output dir', payload.get('output_dir', '-')),
        _info('Starter scene files', files[:10]),
        f"<details><summary>Raw payload</summary><pre>{html.escape(str(payload))}</pre></details>",
    ])


def _render_visual_training(payload: dict[str, object]) -> str:
    concept_store_path = str(payload.get('concept_store_path', '') or '')
    training_root = os.path.dirname(concept_store_path) if concept_store_path else ''
    lewm_report = _read_json_if_exists(os.path.join(training_root, 'lewm_visual_training_report.json')) if training_root else {}
    metrics = ''.join([
        _metric('Inputs', len(payload.get('inputs', [])) if isinstance(payload.get('inputs'), list) else 0),
        _metric('Image count', payload.get('image_count', 0)),
        _metric('Candidates', payload.get('candidate_images', 0)),
        _metric('Pseudo labels', payload.get('pseudo_targets', 0)),
        _metric('Concept store', _artifact_status(concept_store_path)),
        _metric('Operator store', _artifact_status(str(payload.get('operator_store_path', '')))),
        _metric('LeWM prior', _artifact_status(os.path.join(training_root, 'lewm_visual_prior.json')) if training_root else 'missing'),
    ])
    return ''.join([
        f"<div class='grid metrics'>{metrics}</div>",
        _info('Concept summary', payload.get('concept_summary', {})),
        _info('Operator summary', payload.get('operator_summary', {})),
        _info('Eval summary', payload.get('eval_summary', {})),
        _info('LeWM visual report', lewm_report),
        f"<details><summary>Raw payload</summary><pre>{html.escape(str(payload))}</pre></details>",
    ])


def _render_visual_dataset(payload: dict[str, object]) -> str:
    inputs = payload.get('input_paths', []) if isinstance(payload.get('input_paths'), list) else []
    metrics = ''.join([
        _metric('Inputs', payload.get('input_count', 0)),
        _metric('Source kind', payload.get('source_kind', '-')),
        _metric('Manifest', _artifact_status(str(payload.get('manifest_path', '')))),
        _metric('Output dir', payload.get('output_dir', '-')),
    ])
    return ''.join([
        f"<div class='grid metrics'>{metrics}</div>",
        _info('Collected inputs', inputs[:12]),
        f"<details><summary>Raw payload</summary><pre>{html.escape(str(payload))}</pre></details>",
    ])


def _render_visual_batch(payload: dict[str, object]) -> str:
    result_paths = payload.get('result_paths', []) if isinstance(payload.get('result_paths'), list) else []
    obj_paths = payload.get('obj_paths', []) if isinstance(payload.get('obj_paths'), list) else []
    metrics = ''.join([
        _metric('Inputs', payload.get('input_count', 0)),
        _metric('Reconstructed', payload.get('reconstructed_count', 0)),
        _metric('OBJ bundles', len(obj_paths)),
        _metric('Output dir', payload.get('output_dir', '-')),
    ])
    return ''.join([
        f"<div class='grid metrics'>{metrics}</div>",
        _info('Scene result files', result_paths[:12]),
        _info('OBJ bundles', obj_paths[:12]),
        _info('Warnings', payload.get('warnings', [])),
        f"<details><summary>Raw payload</summary><pre>{html.escape(str(payload))}</pre></details>",
    ])


def _render_scene_reconstruction(payload: dict[str, object]) -> str:
    primitives = payload.get('primitives', []) if isinstance(payload.get('primitives'), list) else []
    relations = payload.get('relations', []) if isinstance(payload.get('relations'), list) else []
    relation_lines = [
        f"{item.get('source', '-')} {item.get('relation', '-')} {item.get('target', '-')}"
        for item in relations[:12]
        if isinstance(item, dict)
    ]
    primitive_lines = [
        f"{item.get('label', '-')}: {item.get('shape', '-')}, xyz=({item.get('x', '-')}, {item.get('y', '-')}, {item.get('z', '-')})"
        for item in primitives[:12]
        if isinstance(item, dict)
    ]
    mesh_stats = payload.get('mesh_stats', {}) if isinstance(payload.get('mesh_stats'), dict) else {}
    export_paths = payload.get('export_paths', {}) if isinstance(payload.get('export_paths'), dict) else {}
    topology_summary = payload.get('topology_summary', {}) if isinstance(payload.get('topology_summary'), dict) else {}
    metrics = ''.join([
        _metric('Primitives', len(primitives)),
        _metric('Relations', len(relations)),
        _metric('Warnings', len(payload.get('warnings', [])) if isinstance(payload.get('warnings'), list) else 0),
        _metric('Faces', mesh_stats.get('face_count', 0)),
        _metric('Vertices', mesh_stats.get('vertex_count', 0)),
        _metric('OBJ bundle', _artifact_status(str(export_paths.get('obj_path', '')))),
    ])
    return ''.join([
        f"<div class='grid metrics'>{metrics}</div>",
        _info('Answer text', payload.get('answer_text', '-')),
        _info('3D primitives', primitive_lines),
        _info('Topology relations', relation_lines),
        _info('Topology summary', topology_summary),
        _info('Mesh stats', mesh_stats),
        _info('Export paths', export_paths),
        _info('LeWM alignment', payload.get('leworldmodel_alignment', {})),
        _info('LeWM plan', payload.get('leworldmodel_plan', {})),
        _info('Inferred steps', payload.get('inferred_steps', [])),
        _info('Constraints', payload.get('constraints', [])),
        _info('Warnings', payload.get('warnings', [])),
        _render_scene_canvas(payload),
        f"<details><summary>Raw payload</summary><pre>{html.escape(str(payload))}</pre></details>",
    ])


def _render_math_training(payload: dict[str, object]) -> str:
    results = payload.get('final_results', []) if isinstance(payload.get('final_results'), list) else []
    lewm_effect = payload.get('leworldmodel_effect', {}) if isinstance(payload.get('leworldmodel_effect'), dict) else {}
    lewm_metrics = payload.get('leworldmodel_metrics', {}) if isinstance(payload.get('leworldmodel_metrics'), dict) else {}
    lines = [
        f"{item.get('case_id', '-')}: score={float(item.get('overall_score', 0.0) or 0.0):.3f}, status={item.get('status', '-')}, exact={item.get('exact_match', False)}"
        for item in results
        if isinstance(item, dict)
    ]
    metrics = ''.join([
        _metric('Cases', payload.get('total_cases', 0)),
        _metric('Epochs', payload.get('epochs', 0)),
        _metric('Baseline', payload.get('baseline_average_score', 0.0)),
        _metric('Final', payload.get('final_average_score', 0.0)),
        _metric('Improved', payload.get('improved_case_count', 0)),
        _metric('Accepted', payload.get('accepted_cases', 0)),
        _metric('Exact', payload.get('exact_match_cases', 0)),
        _metric('LeWM delta', lewm_effect.get('average_score_delta', 0.0)),
    ])
    return ''.join([
        f"<div class='grid metrics'>{metrics}</div>",
        _info('Artifacts', payload.get('artifact_paths', {})),
        _info('LeWM metrics', lewm_metrics),
        _info('LeWM effect', lewm_effect),
        _info('Final case results', lines),
        f"<details><summary>Raw payload</summary><pre>{html.escape(str(payload))}</pre></details>",
    ])


def _render_math_eval(payload: dict[str, object]) -> str:
    results = payload.get('results', []) if isinstance(payload.get('results'), list) else []
    lines = [
        f"{item.get('case_id', '-')}: score={float(item.get('overall_score', 0.0) or 0.0):.3f}, missing={', '.join(item.get('missing_required_terms', [])) or '-'}"
        for item in results
        if isinstance(item, dict)
    ]
    metrics = ''.join([
        _metric('Cases', payload.get('total_cases', 0)),
        _metric('Passed', payload.get('passed_cases', 0)),
        _metric('Accepted', payload.get('accepted_cases', 0)),
        _metric('Solved', payload.get('solved_cases', 0)),
        _metric('Exact', payload.get('exact_match_cases', 0)),
        _metric('Average', payload.get('average_score', 0.0)),
    ])
    return ''.join([
        f"<div class='grid metrics'>{metrics}</div>",
        _info('Per-case evaluation', lines),
        f"<details><summary>Raw payload</summary><pre>{html.escape(str(payload))}</pre></details>",
    ])


def _render_self_test(payload: dict[str, object]) -> str:
    readiness = payload.get('readiness', {}) if isinstance(payload.get('readiness'), dict) else {}
    results = payload.get('results', []) if isinstance(payload.get('results'), list) else []
    lines = [
        f"{item.get('name', '-')}: {item.get('status', '-')} ({'pass' if item.get('passed') else 'fail'})"
        for item in results
        if isinstance(item, dict)
    ]
    metrics = ''.join([
        _metric('Status', payload.get('status', '-')),
        _metric('Passed', f"{payload.get('passed_count', 0)}/{payload.get('total_count', 0)}"),
        _metric('Ready', payload.get('ready', False)),
        _metric('Readiness', readiness.get('status', '-')),
    ])
    return ''.join([
        f"<div class='grid metrics'>{metrics}</div>",
        _info('Self-test results', lines),
        _info('Warnings', readiness.get('warnings', [])),
        _info('Recommended actions', readiness.get('recommended_actions', [])),
        f"<details><summary>Raw payload</summary><pre>{html.escape(str(payload))}</pre></details>",
    ])


def _render_math_response(payload: dict[str, object]) -> str:
    decision = payload.get('decision', {}) if isinstance(payload.get('decision'), dict) else {}
    report = payload.get('report', {}) if isinstance(payload.get('report'), dict) else {}
    prior = report.get('strategy_prior', {}) if isinstance(report.get('strategy_prior'), dict) else {}
    candidates = report.get('candidates', []) if isinstance(report.get('candidates'), list) else []
    lines = [
        f"{item.get('source', '-')}/{item.get('kind', '-')}: score={float(item.get('score', 0.0) or 0.0):.3f}, verified={item.get('verified', False)}"
        for item in candidates[:6]
        if isinstance(item, dict)
    ]
    metrics = ''.join([
        _metric('Gate', decision.get('status', '-')),
        _metric('Accepted', payload.get('accepted', False)),
        _metric('Family', prior.get('family', '-')),
        _metric('Difficulty', prior.get('difficulty', '-')),
        _metric('Verification', round(float(decision.get('verification_score', 0.0) or 0.0), 3)),
        _metric('Alignment', round(float(decision.get('alignment_score', 0.0) or 0.0), 3)),
        _metric('Check pass', round(float(decision.get('check_pass_rate', 0.0) or 0.0), 3)),
    ])
    return ''.join([
        f"<div class='grid metrics'>{metrics}</div>",
        _info('Safe answer', payload.get('safe_answer', '-')),
        _info('Gate reasons', decision.get('reasons', [])),
        _info('Solution process', report.get('solution_process', [])),
        _info('Operator trace', report.get('operator_trace', [])),
        _info('Matched patterns', report.get('matched_patterns', [])),
        _info('Recommended operators', prior.get('recommended_operators', [])),
        _info('LeWM alignment', report.get('leworldmodel_alignment', {})),
        _info('LeWM plan', report.get('leworldmodel_plan', {})),
        _info('Candidates', lines),
        _info('Warnings', payload.get('warnings', [])),
        f"<details><summary>Raw payload</summary><pre>{html.escape(str(payload))}</pre></details>",
    ])


def _render_visual_bootcamp(payload: dict[str, object]) -> str:
    return ''.join([
        "<div class='stack'>",
        "<div><small>Starter collection</small>",
        _render_visual_collection(payload.get('collection', {}) if isinstance(payload.get('collection'), dict) else {}),
        "</div>",
        "<div><small>Visual training bundle</small>",
        _render_visual_training(payload.get('training', {}) if isinstance(payload.get('training'), dict) else {}),
        "</div>",
        "<div><small>3D reconstruction</small>",
        _render_scene_reconstruction(payload.get('reconstruction', {}) if isinstance(payload.get('reconstruction'), dict) else {}),
        "</div>",
        "</div>",
    ])


def _render_full_bootcamp(payload: dict[str, object]) -> str:
    return ''.join([
        "<div class='stack'>",
        "<div><small>Math training bundle</small>",
        _render_math_training(payload.get('math_training', {}) if isinstance(payload.get('math_training'), dict) else {}),
        "</div>",
        "<div><small>Visual geometry 3D bootcamp</small>",
        _render_visual_bootcamp(payload.get('visual_bootcamp', {}) if isinstance(payload.get('visual_bootcamp'), dict) else {}),
        "</div>",
        "<div><small>Production readiness</small>",
        _render_self_test(payload.get('self_test', {}) if isinstance(payload.get('self_test'), dict) else {}),
        "</div>",
        "</div>",
    ])


def render_result(payload: dict[str, object] | None) -> str:
    if payload is None:
        return "<div class='empty'>Run the starter bootcamp once for math, or run the visual 3D bootcamp once for image-to-3D. After that, reuse the same local bundle for your own problems and diagrams.</div>"
    if payload.get('full_bootcamp'):
        return _render_full_bootcamp(payload)
    if payload.get('visual_bootcamp'):
        return _render_visual_bootcamp(payload)
    if 'result_paths' in payload and 'obj_paths' in payload and 'reconstructed_count' in payload:
        return _render_visual_batch(payload)
    if 'manifest_path' in payload and 'input_count' in payload and 'input_paths' in payload:
        return _render_visual_dataset(payload)
    if 'scene_dir' in payload and 'scene_count' in payload:
        return _render_visual_collection(payload)
    if 'candidate_images' in payload and 'concept_store_path' in payload and 'operator_store_path' in payload:
        return _render_visual_training(payload)
    if 'primitives' in payload and 'relations' in payload and 'raw_world_model' in payload:
        return _render_scene_reconstruction(payload)
    if 'baseline_average_score' in payload and 'final_average_score' in payload:
        return _render_math_training(payload)
    if 'average_score' in payload and 'results' in payload:
        return _render_math_eval(payload)
    if 'results' in payload and 'passed_count' in payload:
        return _render_self_test(payload)
    return _render_math_response(payload)

class MathWorldModelGui:
    def __init__(self) -> None:
        self._state = MathGuiState()
        self._outcome = MathGuiOutcome()

    def _service(self, state: MathGuiState) -> WorldModelMathProductionService:
        output_dir = state.output_dir or 'data/math_world_model_gui_run'
        os.makedirs(output_dir, exist_ok=True)
        return WorldModelMathProductionService(
            ProductionMathServiceConfig(
                concept_store_path=state.concept_store or None,
                operator_store_path=state.operator_store or None,
                affordance_weights_path=state.affordance_weights or None,
                logical_weight_path=os.path.join(output_dir, 'logical_pattern_weights.json'),
                strategy_memory_path=os.path.join(output_dir, 'math_strategy_memory.json'),
                leworldmodel_path=os.path.join(output_dir, 'math_leworldmodel_prior.json'),
                audit_log_path=os.path.join(output_dir, 'audit_log.jsonl'),
            )
        )

    def _trainer(self, state: MathGuiState) -> WorldModelMathTrainer:
        return WorldModelMathTrainer(
            concept_store_path=state.concept_store or None,
            operator_store_path=state.operator_store or None,
            affordance_weights_path=state.affordance_weights or None,
        )

    def _visual_workbench(self, state: MathGuiState) -> VisualGeometry3DWorkbench:
        return VisualGeometry3DWorkbench(
            concept_store_path=state.concept_store or None,
            operator_store_path=state.operator_store or None,
            affordance_weights_path=state.affordance_weights or None,
            mode='deep',
            answer_mode='structured',
        )

    @staticmethod
    def _pick_visual_input(state: MathGuiState, visual_output_dir: str, preferred_input: str = '') -> str:
        candidates = [
            preferred_input,
            state.visual3d_input,
            state.visual3d_data_source,
            'examples/vlso/geometry_scene.json',
            os.path.join(visual_output_dir, 'starter_scenes', 'parallel_perpendicular_scene.json'),
            os.path.join(visual_output_dir, 'starter_scenes', 'square_scene.json'),
        ]
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        source_path = state.visual3d_data_source
        if source_path and os.path.isdir(source_path):
            for name in sorted(os.listdir(source_path)):
                if name.lower().endswith(('.json', '.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    return os.path.join(source_path, name)
        starter_dir = os.path.join(visual_output_dir, 'starter_scenes')
        if os.path.isdir(starter_dir):
            for name in sorted(os.listdir(starter_dir)):
                if name.lower().endswith('.json'):
                    return os.path.join(starter_dir, name)
        return preferred_input or state.visual3d_input or 'examples/vlso/geometry_scene.json'

    @staticmethod
    def _has_custom_visual_source(state: MathGuiState) -> bool:
        source = (state.visual3d_data_source or '').strip()
        return bool(source and os.path.exists(source))

    def handle(self, form: dict[str, list[str]]) -> str:
        state = self._state.apply_form(form) if form else self._state
        outcome = self._outcome
        if form:
            action = _first(form, 'action', '')
            if action == 'load_geometry_example':
                state = replace(state, **GEOMETRY_EXAMPLE)
                outcome = MathGuiOutcome(flash='Loaded the geometry example.', tone='success', payload=self._outcome.payload)
            elif action == 'load_olympiad_example':
                state = replace(state, **OLYMPIAD_EXAMPLE)
                outcome = MathGuiOutcome(flash='Loaded the olympiad example.', tone='success', payload=self._outcome.payload)
            elif action == 'load_research_example':
                state = replace(state, **RESEARCH_EXAMPLE)
                outcome = MathGuiOutcome(flash='Loaded the research-style example.', tone='success', payload=self._outcome.payload)
            elif action == 'load_starter_curriculum':
                state = replace(state, training_cases_path=ensure_starter_math_cases(state.training_cases_path or 'examples/math_world_model_starter.jsonl'))
                outcome = MathGuiOutcome(flash='Starter math curriculum is ready.', tone='success', payload=self._outcome.payload)
            elif action == 'load_visual_3d_example':
                state = replace(state, **VISUAL_3D_EXAMPLE)
                outcome = MathGuiOutcome(flash='Loaded the visual 3D example.', tone='success', payload=self._outcome.payload)
            elif action == 'self_check':
                try:
                    summary = self._service(state).self_test()
                    outcome = MathGuiOutcome(flash=f"Production self-check finished with status `{summary.status}`.", tone='success' if summary.ready else 'warning', payload=summary.model_dump())
                except Exception as exc:
                    outcome = MathGuiOutcome(flash=f'Self-check failed: {exc}', tone='error', payload=self._outcome.payload)
            elif action == 'solve':
                try:
                    response = self._service(state).solve_request(
                        state.query,
                        source_context=state.context,
                        visual_input=state.visual_input or None,
                        task_mode=state.task_mode,
                        metadata={'surface': 'math_world_model_gui'},
                    )
                    outcome = MathGuiOutcome(flash=f"Production solve finished with gate `{response.status}`.", tone='success' if response.accepted else 'warning', payload=response.model_dump())
                except Exception as exc:
                    outcome = MathGuiOutcome(flash=f'Run failed: {exc}', tone='error', payload=self._outcome.payload)
            elif action == 'train':
                try:
                    cases_path = ensure_starter_math_cases(state.training_cases_path)
                    state = replace(state, training_cases_path=cases_path)
                    summary = self._trainer(state).train_from_cases(cases_path, state.output_dir, epochs=state.epochs)
                    outcome = MathGuiOutcome(flash='Math world-model training finished.', tone='success', payload=summary.model_dump())
                except Exception as exc:
                    outcome = MathGuiOutcome(flash=f'Training failed: {exc}', tone='error', payload=self._outcome.payload)
            elif action == 'evaluate':
                try:
                    cases_path = ensure_starter_math_cases(state.training_cases_path)
                    state = replace(state, training_cases_path=cases_path)
                    summary = self._trainer(state).evaluate_cases(cases_path, state.output_dir)
                    outcome = MathGuiOutcome(flash='Math evaluation finished.', tone='success', payload=summary.model_dump())
                except Exception as exc:
                    outcome = MathGuiOutcome(flash=f'Evaluation failed: {exc}', tone='error', payload=self._outcome.payload)
            elif action == 'bootcamp':
                try:
                    cases_path = ensure_starter_math_cases(state.training_cases_path)
                    state = replace(state, training_cases_path=cases_path)
                    summary = self._trainer(state).train_from_cases(cases_path, state.output_dir, epochs=state.epochs, bootstrap_starter=False)
                    outcome = MathGuiOutcome(flash='Starter math bootcamp finished.', tone='success', payload=summary.model_dump())
                except Exception as exc:
                    outcome = MathGuiOutcome(flash=f'Bootcamp failed: {exc}', tone='error', payload=self._outcome.payload)
            elif action == 'collect_visual_scenes':
                try:
                    summary = self._visual_workbench(state).collect_starter_scenes(state.visual3d_output_dir)
                    outcome = MathGuiOutcome(flash='Starter geometry scenes were collected.', tone='success', payload=summary.model_dump())
                except Exception as exc:
                    outcome = MathGuiOutcome(flash=f'Visual scene collection failed: {exc}', tone='error', payload=self._outcome.payload)
            elif action == 'collect_visual_dataset':
                try:
                    summary = self._visual_workbench(state).collect_input_manifest(state.visual3d_data_source, state.visual3d_output_dir)
                    outcome = MathGuiOutcome(flash='Visual dataset manifest was collected from the current folder or file.', tone='success', payload=summary.model_dump())
                except Exception as exc:
                    outcome = MathGuiOutcome(flash=f'Visual dataset collection failed: {exc}', tone='error', payload=self._outcome.payload)
            elif action == 'train_visual_3d':
                try:
                    workbench = self._visual_workbench(state)
                    if self._has_custom_visual_source(state):
                        summary = workbench.train_bundle_from_inputs(state.visual3d_data_source, state.visual3d_output_dir, limit_scenes=state.visual3d_scene_limit)
                    else:
                        summary = workbench.train_starter_bundle(state.visual3d_output_dir, limit_scenes=state.visual3d_scene_limit)
                    outcome = MathGuiOutcome(flash='Visual 3D training finished.', tone='success', payload=summary.model_dump())
                except Exception as exc:
                    outcome = MathGuiOutcome(flash=f'Visual 3D training failed: {exc}', tone='error', payload=self._outcome.payload)
            elif action == 'reconstruct_visual_3d':
                try:
                    visual_input = self._pick_visual_input(state, state.visual3d_output_dir)
                    state = replace(state, visual3d_input=visual_input)
                    summary = self._visual_workbench(state).reconstruct_scene(state.visual3d_query, visual_input, state.visual3d_output_dir)
                    outcome = MathGuiOutcome(flash='The current image or diagram was reconstructed into a simple 3D scene.', tone='success', payload=summary.model_dump())
                except Exception as exc:
                    outcome = MathGuiOutcome(flash=f'3D reconstruction failed: {exc}', tone='error', payload=self._outcome.payload)
            elif action == 'reconstruct_visual_3d_batch':
                try:
                    inputs = state.visual3d_data_source if self._has_custom_visual_source(state) else state.visual3d_output_dir
                    summary = self._visual_workbench(state).reconstruct_batch(state.visual3d_query, inputs, state.visual3d_output_dir, limit=state.visual3d_scene_limit)
                    outcome = MathGuiOutcome(flash='Batch 3D reconstruction finished for the current folder or file set.', tone='success', payload=summary.model_dump())
                except Exception as exc:
                    outcome = MathGuiOutcome(flash=f'Batch 3D reconstruction failed: {exc}', tone='error', payload=self._outcome.payload)
            elif action == 'visual3d_bootcamp':
                try:
                    workbench = self._visual_workbench(state)
                    if self._has_custom_visual_source(state):
                        collection = workbench.collect_input_manifest(state.visual3d_data_source, state.visual3d_output_dir)
                        training = workbench.train_bundle_from_inputs(state.visual3d_data_source, state.visual3d_output_dir, limit_scenes=state.visual3d_scene_limit)
                    else:
                        collection = workbench.collect_starter_scenes(state.visual3d_output_dir)
                        training = workbench.train_starter_bundle(state.visual3d_output_dir, limit_scenes=state.visual3d_scene_limit)
                    visual_input = self._pick_visual_input(state, state.visual3d_output_dir, preferred_input=state.visual3d_input)
                    state = replace(state, visual3d_input=visual_input)
                    reconstruction = workbench.reconstruct_scene(state.visual3d_query, visual_input, state.visual3d_output_dir)
                    outcome = MathGuiOutcome(flash='Visual 3D bootcamp finished.', tone='success', payload={'visual_bootcamp': True, 'collection': collection.model_dump(), 'training': training.model_dump(), 'reconstruction': reconstruction.model_dump()})
                except Exception as exc:
                    outcome = MathGuiOutcome(flash=f'Visual 3D bootcamp failed: {exc}', tone='error', payload=self._outcome.payload)
            elif action == 'full_bootcamp':
                try:
                    cases_path = ensure_starter_math_cases(state.training_cases_path)
                    state = replace(state, training_cases_path=cases_path)
                    math_training = self._trainer(state).train_from_cases(cases_path, state.output_dir, epochs=state.epochs, bootstrap_starter=False)
                    workbench = self._visual_workbench(state)
                    if self._has_custom_visual_source(state):
                        collection = workbench.collect_input_manifest(state.visual3d_data_source, state.visual3d_output_dir)
                        training = workbench.train_bundle_from_inputs(state.visual3d_data_source, state.visual3d_output_dir, limit_scenes=state.visual3d_scene_limit)
                    else:
                        collection = workbench.collect_starter_scenes(state.visual3d_output_dir)
                        training = workbench.train_starter_bundle(state.visual3d_output_dir, limit_scenes=state.visual3d_scene_limit)
                    visual_input = self._pick_visual_input(state, state.visual3d_output_dir, preferred_input=state.visual3d_input)
                    state = replace(state, visual3d_input=visual_input)
                    reconstruction = workbench.reconstruct_scene(state.visual3d_query, visual_input, state.visual3d_output_dir)
                    self_test = self._service(state).self_test()
                    outcome = MathGuiOutcome(flash='Full beginner bootcamp finished: math lane, visual lane, and production self-check are ready.', tone='success', payload={'full_bootcamp': True, 'math_training': math_training.model_dump(), 'visual_bootcamp': {'visual_bootcamp': True, 'collection': collection.model_dump(), 'training': training.model_dump(), 'reconstruction': reconstruction.model_dump()}, 'self_test': self_test.model_dump()})
                except Exception as exc:
                    outcome = MathGuiOutcome(flash=f'Full beginner bootcamp failed: {exc}', tone='error', payload=self._outcome.payload)
        self._state = state
        self._outcome = outcome
        return self.render_page(state, outcome)

    def render_page(self, state: MathGuiState, outcome: MathGuiOutcome) -> str:
        readiness = self._service(state).readiness().model_dump()
        output_dir = state.output_dir or 'data/math_world_model_gui_run'
        visual_output_dir = state.visual3d_output_dir or 'data/math_world_model_gui_run/visual_3d'
        flash = f"<div class='flash {html.escape(outcome.tone)}'>{html.escape(outcome.flash)}</div>" if outcome.flash else ''
        hardware = _hardware_profile_payload()
        dependencies = _dependency_payload()
        workspace_metrics = ''.join([
            _metric('Readiness', readiness.get('status', '-')),
            _metric('Hardware', hardware.get('detected_profile', '-')),
            _metric('VRAM', hardware.get('vram_gb', 0.0)),
            _metric('Operator algebra', hardware.get('operator_algebra_mode', '-')),
            _metric('LLM ready', dependencies.get('llm_ready', False)),
            _metric('Training ready', dependencies.get('training_ready', False)),
            _metric('QLoRA ready', dependencies.get('qlora_ready', False)),
            _metric('Starter cases', _artifact_status(state.training_cases_path)),
            _metric('Logical weights', _artifact_status(os.path.join(output_dir, 'logical_pattern_weights.json'))),
            _metric('Strategy memory', _artifact_status(os.path.join(output_dir, 'math_strategy_memory.json'))),
            _metric('Math LeWM prior', _artifact_status(os.path.join(output_dir, 'math_leworldmodel_prior.json'))),
            _metric('Final eval', _artifact_status(os.path.join(output_dir, 'math_eval_final.json'))),
            _metric('Starter scenes', _artifact_status(os.path.join(visual_output_dir, 'starter_scenes'))),
            _metric('Visual input manifest', _artifact_status(os.path.join(visual_output_dir, 'visual_geometry_input_manifest.json'))),
            _metric('Visual concept store', _artifact_status(os.path.join(visual_output_dir, 'visual_geometry_concepts.db'))),
            _metric('Visual operator store', _artifact_status(os.path.join(visual_output_dir, 'visual_geometry_operators.db'))),
            _metric('Visual LeWM prior', _artifact_status(os.path.join(visual_output_dir, 'lewm_visual_prior.json'))),
            _metric('3D reconstruction', _artifact_status(os.path.join(visual_output_dir, 'scene_3d_reconstruction.json'))),
            _metric('OBJ bundle', _artifact_status(os.path.join(visual_output_dir, 'scene_3d_bundle', 'scene_3d_reconstruction.obj'))),
        ])
        return f"""
<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Math and Visual World Model Studio</title>
<style>
:root {{ --card:#fffdf8; --line:#d6d7cf; --ink:#16211d; --muted:#58655c; --accent:#1e5a46; --soft:#edf5ef; --ok:#256847; --warn:#a5542d; --err:#8d2b2b; }}
body {{ margin:0; background:linear-gradient(180deg,#f7f1e8 0%,#eef5ef 100%); color:var(--ink); font-family:"Aptos","Segoe UI Variable","Malgun Gothic",sans-serif; }}
main {{ max-width:1280px; margin:0 auto; padding:24px 18px 42px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:24px; padding:20px; box-shadow:0 14px 34px rgba(22,33,29,.08); }}
.grid {{ display:grid; gap:18px; }}
.stack {{ display:grid; gap:20px; }}
.hero {{ grid-template-columns:1.1fr .9fr; }}
.twocol {{ grid-template-columns:1fr 1fr; margin-top:20px; }}
.metrics {{ grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); }}
.metric, .info {{ background:#f7faf5; border:1px solid var(--line); border-radius:16px; padding:12px; }}
.metric small, .info small {{ display:block; color:var(--muted); margin-bottom:6px; }}
.metric strong {{ font-size:16px; }}
textarea, input, select {{ width:100%; padding:12px 14px; border-radius:14px; border:1px solid var(--line); font:inherit; background:#fff; box-sizing:border-box; }}
textarea {{ min-height:120px; resize:vertical; }}
button {{ border:0; border-radius:999px; padding:11px 14px; font:inherit; cursor:pointer; }}
button.primary {{ background:var(--accent); color:#fff; font-weight:700; }}
button.secondary {{ background:var(--soft); color:var(--ink); font-weight:700; }}
.actions {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }}
.flash {{ margin:16px 0; padding:12px 14px; border-radius:14px; }}
.flash.success {{ background:#ebf8ef; color:var(--ok); }}
.flash.warning {{ background:#fff2eb; color:var(--warn); }}
.flash.error {{ background:#fff0f0; color:var(--err); }}
.empty {{ border:1px dashed var(--line); border-radius:16px; padding:20px; color:var(--muted); }}
.paint-card {{ margin-top:16px; border:1px solid var(--line); border-radius:18px; padding:14px; background:linear-gradient(180deg,#fffdf7 0%,#f0f5ef 100%); }}
.paint-title {{ font-weight:700; margin-bottom:8px; }}
canvas {{ width:100%; max-width:760px; border-radius:16px; border:1px solid var(--line); background:#fbf8f1; }}
pre {{ white-space:pre-wrap; overflow:auto; background:#14211c; color:#eef7ef; border-radius:16px; padding:14px; }}
label {{ display:block; font-size:14px; margin:12px 0 6px; }}
@media (max-width:980px) {{ .hero, .twocol {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<main>
  <section class='grid hero'>
    <div class='card'>
      <small>Math and Visual World Model Studio</small>
      <h1>Solve math problems, learn patterns, analyze images into geometry and topology, and preview the result in a simple 3D paint surface.</h1>
      <p>This lab now covers both lanes: proof-style math solving with a production gate, and image or diagram analysis that is turned into geometric and topological 3D primitives. Beginners can stay on the one-click paths.</p>
      <p>When a LeWM prior exists, the studio also shows a latent <strong>LeWM plan</strong> so you can see the proposed trajectory before trusting the final answer or reconstruction.</p>
      <div class='actions'>
        <form method='post'><button class='secondary' name='action' value='load_geometry_example'>Load geometry example</button></form>
        <form method='post'><button class='secondary' name='action' value='load_olympiad_example'>Load olympiad example</button></form>
        <form method='post'><button class='secondary' name='action' value='load_research_example'>Load research example</button></form>
        <form method='post'><button class='secondary' name='action' value='load_visual_3d_example'>Load visual 3D example</button></form>
        <form method='post'><button class='primary' name='action' value='full_bootcamp'>One-click math + visual bootcamp</button></form>
      </div>
    </div>
    <div class='card'>
      <small>Workspace snapshot</small>
      <div class='grid metrics'>{workspace_metrics}</div>
      {_info('Warnings', readiness.get('warnings', []))}
      {_info('Recommended actions', readiness.get('recommended_actions', []))}
      {_info('Hardware notes', hardware.get('notes', []))}
      {_info('Dependency notes', dependencies.get('notes', []))}
      {_info('Missing core packages', dependencies.get('missing_core', []))}
      {_info('Missing optional packages', dependencies.get('missing_optional', []))}
      {_info('Math output dir', output_dir)}
      {_info('Visual 3D output dir', visual_output_dir)}
    </div>
  </section>
  {flash}
  <section class='grid twocol'>
    <form method='post' class='card'>
      <small>Math problem solving</small>
      <label>Problem</label>
      <textarea name='query'>{html.escape(state.query)}</textarea>
      <label>Given context or lemmas</label>
      <textarea name='context'>{html.escape(state.context)}</textarea>
      <label>Diagram JSON or image path (optional)</label>
      <input name='visual_input' value='{html.escape(state.visual_input)}'>
      <div class='grid twocol'>
        <div><label>Task mode</label>{_render_select('task_mode', state.task_mode)}</div>
        <div><label>Math output dir</label><input name='output_dir' value='{html.escape(state.output_dir)}'></div>
        <div><label>Concept store</label><input name='concept_store' value='{html.escape(state.concept_store)}'></div>
        <div><label>Operator store</label><input name='operator_store' value='{html.escape(state.operator_store)}'></div>
        <div><label>Affordance weights</label><input name='affordance_weights' value='{html.escape(state.affordance_weights)}'></div>
      </div>
      <div class='actions'>
        <button class='primary' name='action' value='solve'>Solve with production gate</button>
        <button class='secondary' name='action' value='self_check'>Run production self-check</button>
      </div>
    </form>
    <form method='post' class='card'>
      <small>Math training and evaluation</small>
      <label>Training cases JSONL</label>
      <input name='training_cases_path' value='{html.escape(state.training_cases_path)}'>
      <label>Training epochs</label>
      <input name='epochs' value='{html.escape(str(state.epochs))}'>
      <div class='actions'>
        <button class='secondary' name='action' value='load_starter_curriculum'>Load starter curriculum</button>
        <button class='primary' name='action' value='bootcamp'>One-click starter train + eval</button>
        <button class='secondary' name='action' value='train'>Run math training</button>
        <button class='secondary' name='action' value='evaluate'>Run math evaluation</button>
      </div>
      {_info('What this does', ['starter curriculum -> local pattern learning -> strategy memory -> evaluation summary', 'the same math output dir is reused when you solve your own next problem'])}
    </form>
  </section>
  <section class='card' style='margin-top:20px;'>
    <small>Visual geometry to 3D</small>
    <p>Collect starter geometry scenes, train the visual concept and operator stores, then reconstruct an image or diagram into a simple 3D paint view.</p>
    <form method='post'>
      <div class='grid twocol'>
        <div>
          <label>3D reconstruction query</label>
          <textarea name='visual3d_query'>{html.escape(state.visual3d_query)}</textarea>
        </div>
        <div>
          <label>Image or diagram JSON path</label>
          <input name='visual3d_input' value='{html.escape(state.visual3d_input)}'>
          <label>Visual 3D output dir</label>
          <input name='visual3d_output_dir' value='{html.escape(state.visual3d_output_dir)}'>
          <label>Visual dataset source folder or file</label>
          <input name='visual3d_data_source' value='{html.escape(state.visual3d_data_source)}'>
          <label>Starter scene limit for training</label>
          <input name='visual3d_scene_limit' value='{html.escape(str(state.visual3d_scene_limit))}'>
        </div>
      </div>
      <div class='actions'>
        <button class='secondary' name='action' value='load_visual_3d_example'>Load visual 3D example</button>
        <button class='secondary' name='action' value='collect_visual_scenes'>Collect starter geometry scenes</button>
        <button class='secondary' name='action' value='collect_visual_dataset'>Collect from folder or file</button>
        <button class='secondary' name='action' value='train_visual_3d'>Run visual 3D training</button>
        <button class='primary' name='action' value='reconstruct_visual_3d'>Reconstruct image or diagram into 3D</button>
        <button class='secondary' name='action' value='reconstruct_visual_3d_batch'>Batch reconstruct folder or file</button>
        <button class='primary' name='action' value='visual3d_bootcamp'>One-click visual 3D bootcamp</button>
      </div>
      {_info('Beginner shortcut', ['If you already have your own folder or JSON file, set Visual dataset source folder or file and press Collect from folder or file.', 'Run One-click visual 3D bootcamp once to collect or index data, train the visual stores, and preview a 3D reconstruction.', 'Use Batch reconstruct folder or file when you want OBJ bundles for several files at once.'])}
    </form>
  </section>
  <section class='card' style='margin-top:20px;'>
    <small>Result</small>
    {render_result(outcome.payload)}
  </section>
</main>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    app: MathWorldModelGui

    def do_GET(self) -> None:
        self._send(self.app.handle({}))

    def do_POST(self) -> None:
        length = int(self.headers.get('Content-Length', '0') or 0)
        body = self.rfile.read(length).decode('utf-8', errors='replace')
        self._send(self.app.handle(parse_qs(body, keep_blank_values=True)))

    def _send(self, body: str) -> None:
        encoded = body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description='Math and visual world model studio')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8791)
    args = parser.parse_args()
    app = MathWorldModelGui()
    handler = type('MathWorldModelGuiHandler', (_Handler,), {'app': app})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f'Math World Model Studio running at http://{args.host}:{args.port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

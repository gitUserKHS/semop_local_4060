from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from semop import (
    CompetitiveProgrammingReasoner,
    CpLearnedParser,
    CpParserEvaluator,
    CopilotRequest,
    CpGeometryTemplateGenerator,
    CpLabeledDatasetDownloader,
    CpLoraExperimentConfig,
    CpLoraExperimentRunner,
    CpParserTrainConfig,
    CpParserTrainingScaffold,
    DomainCopilot,
    HiddenPremiseEvalCase,
    OperatorCurriculumBuilder,
    OperatorTrainConfig,
    OperatorTrainingScaffold,
    PseudoLabelAcceptanceConfig,
    SemOpUnderstandingEvaluator,
    SyntheticGeometrySceneBuilder,
    VLSOReasoner,
    VlsoGroundedEvaluator,
    VlsoReviewImpactEvaluator,
    VisualApprovedReviewRetrainer,
    VisualClusterReviewDecision,
    VisualClusterReviewStore,
    VisualConceptPrototypeTrainer,
    VisualDataCollector,
    VisualFamilyBatchSummary,
    VisualGeometryBootstrapPipeline,
    VisualOperatorPrototypeTrainer,
    TeacherTraceExporter,
    build_object_family_manifest,
)


OPS_EXAMPLE = {
    "query": "The aisle is blocked and approval is still missing. What should I do?",
    "context": "If the work zone is blocked or the task is not approved yet, stop first and check the alternate route and manager approval.",
    "domain": "warehouse_exception",
    "scenario": "exception_response",
}
CP_EXAMPLE = {
    "statement": "Given an array, answer many range sum queries with no updates.",
}
VISION_EXAMPLE = {
    "query": "What objects or openings are visible here?",
    "image_path": "data/scene.png",
}
CP_DOWNLOAD_EXAMPLE = {
    "manifest": "examples/cp_labeled_manifest.json",
    "download_root": "data/cp_labeled_downloads",
    "output": "data/cp_labeled_normalized.jsonl",
}
VLSO_COLLECTION_EXAMPLE = {
    "manifest": "examples/vlso_collection_manifest.json",
    "output": "data/vlso_collection_plan.jsonl",
}
VLSO_FAMILY_COLLECTION_EXAMPLE = {
    "manifest": "data/vlso_family_collection_manifest.json",
    "families": "bag box drawer door bottle tool cabinet suitcase jar bin pouch",
    "limit": "10",
}
VLSO_FAMILY_BATCH_EXAMPLE = {
    "workspace": "data/vlso_family_batch",
    "targets": "bag:20 box:20 drawer:15 door:15 bottle:15 tool:15 cabinet:10 suitcase:10 jar:10 bin:10 pouch:10",
}
VLSO_DOWNLOAD_EXAMPLE = {
    "records": "data/vlso_collection_records.jsonl",
    "approved": "data/vlso_collection_approved.jsonl",
    "manifest": "data/vlso_download_manifest.jsonl",
    "download_root": "data/vlso_downloads",
    "providers": "wikimedia_commons openverse",
    "licenses": "cc0 by by-sa",
}
VLSO_COMPARE_EXAMPLE = {
    "primary_store": "data/vlso_visual_prototypes.db",
    "compare_store": "data/vlso_visual_pseudo.db",
    "summary": "data/vlso_visual_pseudo_summary.json",
    "reviews": "data/vlso_cluster_reviews.json",
    "operator_store": "data/vlso_visual_operators.db",
    "compare_operator_store": "data/vlso_visual_pseudo_operators.db",
    "eval_input": "examples/vlso_eval.jsonl",
}
GEOMETRY_GENERATOR_EXAMPLE = {
    "output_dir": "examples/vlso/generated_geometry",
    "eval_output": "examples/vlso_geometry_eval.jsonl",
}
CP_GEOMETRY_EXAMPLE = {
    "eval_output": "examples/cp_geometry_parser_eval.jsonl",
    "train_output": "examples/cp_geometry_train.jsonl",
    "val_output": "examples/cp_geometry_val.jsonl",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "output_dir": "data/cp_geometry_parser_dry_run",
    "compare_input": "examples/cp_parser_eval.jsonl",
}
CP_LORA_EXAMPLE = {
    "workspace": "data/cp_lora_gui_run",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "train_inputs": "examples/cp_parser_eval.jsonl examples/cp_hidden_constraint_eval.jsonl",
    "eval_inputs": "examples/cp_geometry_parser_eval.jsonl examples/cp_hidden_constraint_eval.jsonl",
    "max_steps": "100",
    "save_steps": "25",
    "save_total_limit": "2",
}
UNDERSTANDING_EVAL_EXAMPLE = {
    "hidden": "examples/hidden_premise_eval.jsonl",
    "cp": "examples/cp_parser_eval.jsonl",
    "cp_hidden": "examples/cp_hidden_constraint_eval.jsonl",
    "vlso": "examples/vlso_eval.jsonl",
    "vlso_real": "examples/vlso_real_image_eval_gold.jsonl",
}
OPERATOR_TRAIN_EXAMPLE = {
    "workspace": "data/operator_learning_gui_run",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "teacher_output": "data/operator_learning_gui_run/teacher_traces.jsonl",
    "teacher_sft_output": "data/operator_learning_gui_run/teacher_traces_sft.jsonl",
    "max_steps": "100",
    "save_steps": "25",
    "save_total_limit": "2",
}
GEOMETRY_PIPELINE_EXAMPLE = {
    "input_root": "examples/vlso/generated_geometry",
    "workspace": "data/vlso_geometry_pipeline_gui",
    "cluster_threshold": "0.90",
    "pseudo_threshold": "0.72",
    "consensus_threshold": "0.55",
    "concept_match_threshold": "0.86",
    "min_cluster_size": "2",
}
DOWNLOADED_IMAGE_PIPELINE_EXAMPLE = {
    "workspace": "data/vlso_download_pipeline_gui",
    "labels": "data/vlso_download_pipeline_gui/manual_labels.jsonl",
    "concept_store": "data/vlso_download_pipeline_gui/manual_concepts.db",
    "operator_store": "data/vlso_download_pipeline_gui/manual_operators.db",
    "review": "data/vlso_download_pipeline_gui/manual_label_reviews.json",
}


VLSO_LABEL_SUGGESTIONS = [
    "BAG_LIKE_CONTAINER",
    "HAS_INTERIOR",
    "ACCESS_OPENING_CANDIDATE",
    "ZIPPER_LIKE_PART",
    "STRAP_LIKE_PART",
    "GRASPABLE_PART",
    "HANDLE_LIKE_PART",
    "BOX_LIKE_CONTAINER",
    "DRAWER_LIKE_CONTAINER",
    "CABINET_LIKE_CONTAINER",
    "DOOR_PANEL",
    "HINGE_LIKE_PART",
    "LID_LIKE_PART",
    "KNOB_LIKE_PART",
    "CAPPED_OPENING",
    "BOTTLE_LIKE_CONTAINER",
    "TOOL_GRIP_PART",
    "DRAWER_LIKE_CONTAINER",
    "DOOR_PANEL",
    "HINGE_LIKE_PART",
    "CAPPED_OPENING",
    "BOTTLE_LIKE_CONTAINER",
    "TOOL_GRIP_PART",
    "RECTANGLE_LIKE_OBJECT",
]


def _first(form: dict[str, list[str]], key: str, default: str = "") -> str:
    values = form.get(key)
    return values[0] if values else default


def _checked_form(form: dict[str, list[str]], key: str, default: bool = False) -> bool:
    values = form.get(key)
    if not values:
        return default
    return values[-1] == "1"


def _checked(value: bool) -> str:
    return " checked" if value else ""


def _safe_json(payload: object) -> str:
    return html.escape(json.dumps(payload, ensure_ascii=False, indent=2))


def _gui_vlso_progress(event: str, payload: dict[str, object]) -> None:
    idx = payload.get('index', '?')
    total = payload.get('total', '?')
    provider = payload.get('provider', '')
    query = payload.get('query', '')
    title = payload.get('title', '')
    if event == 'plan':
        print(f'[GUI/VLSO plan {idx}/{total}] {provider} :: {query}', file=sys.stderr, flush=True)
    elif event == 'request_succeeded':
        print(f'[GUI/VLSO ok {idx}/{total}] {provider} :: {query} -> {payload.get("records", 0)} records', file=sys.stderr, flush=True)
    elif event == 'request_failed':
        print(f'[GUI/VLSO fail {idx}/{total}] {provider} :: {query} -> {payload.get("error", "error")}', file=sys.stderr, flush=True)
    elif event == 'download_start':
        print(f'[GUI/VLSO download {idx}/{total}] {provider} :: {title}', file=sys.stderr, flush=True)
    elif event == 'download_succeeded':
        print(f'[GUI/VLSO saved {idx}/{total}] {provider} :: {title}', file=sys.stderr, flush=True)
    elif event == 'download_failed':
        print(f'[GUI/VLSO download-fail {idx}/{total}] {provider} :: {title} -> {payload.get("error", "error")}', file=sys.stderr, flush=True)


def _tokens(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace(',', ' ').split() if item.strip()]


def _parse_family_targets(raw: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for token in raw.replace(',', ' ').split():
        if ':' in token:
            name, value = token.split(':', 1)
        else:
            name, value = token, '10'
        name = name.strip().lower()
        if not name:
            continue
        try:
            output[name] = max(1, int(value.strip()))
        except ValueError:
            output[name] = 10
    return output


def _latest_checkpoint_path(root: str) -> str:
    base = Path(root)
    if not base.exists():
        return ''
    candidates: list[tuple[int, Path]] = []
    for path_value in base.rglob('checkpoint-*'):
        if not path_value.is_dir():
            continue
        try:
            step = int(path_value.name.split('-', 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((step, path_value))
    if not candidates:
        return ''
    candidates.sort(key=lambda item: item[0])
    return str(candidates[-1][1])


def _render_value(value: object) -> str:
    if value is None or value == '':
        return '-'
    if isinstance(value, float):
        return f'{value:.3f}'
    if isinstance(value, (list, tuple)):
        return ', '.join(str(item) for item in value[:8]) or '-'
    return str(value)


def _metric_card(label: str, value: object) -> str:
    return f"<div class='metric-card'><small>{html.escape(label)}</small><strong>{html.escape(_render_value(value))}</strong></div>"


def _info_block(label: str, value: object) -> str:
    rendered = html.escape(_render_value(value))
    return f"<div class='info-block'><small>{html.escape(label)}</small><div>{rendered}</div></div>"


def _raw_details(payload: dict[str, object]) -> str:
    return f"<details class='raw-json'><summary>Raw JSON</summary><pre>{_safe_json(payload)}</pre></details>"


def _load_hidden_premise_cases(path_value: str) -> list[HiddenPremiseEvalCase]:
    path = Path(path_value)
    cases: list[HiddenPremiseEvalCase] = []
    with path.open('r', encoding='utf-8-sig') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            cases.append(HiddenPremiseEvalCase(**payload))
    return cases


def _record_id(record: dict[str, object]) -> str:
    return str(record.get('source_id') or record.get('media_url') or record.get('title') or '')


def _field_key(value: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in value)


def _local_image_data_uri(path_value: str) -> str:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return ''
    mime_type = mimetypes.guess_type(str(path))[0] or 'image/jpeg'
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'data:{mime_type};base64,{encoded}'


def _collect_local_downloaded_images(root: str, limit: int = 16) -> list[dict[str, object]]:
    base = Path(root)
    if not base.exists():
        return []
    rows: list[dict[str, object]] = []
    for child in sorted(base.rglob('*')):
        if not child.is_file() or child.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}:
            continue
        rows.append({
            'source_id': str(child.relative_to(base)).replace('\\', '/'),
            'title': child.stem,
            'provider': child.parent.name,
            'license': 'downloaded',
            'local_path': str(child),
        })
        if len(rows) >= limit:
            break
    return rows


def _load_download_label_reviews(path_value: str) -> dict[str, dict[str, object]]:
    path = Path(path_value)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        return {}
    rows = payload.get('reviews', payload)
    if not isinstance(rows, dict):
        return {}
    cleaned: dict[str, dict[str, object]] = {}
    for key, value in rows.items():
        if isinstance(key, str) and isinstance(value, dict):
            cleaned[key] = value
    return cleaned


def _save_download_label_reviews(path_value: str, rows: dict[str, dict[str, object]]) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'reviews': rows}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def render_downloaded_label_editor(download_root: str, payload: dict[str, object] | None, labels_output: str, concept_store: str, operator_store: str, review_path: str, review_rows: dict[str, dict[str, object]] | None = None, review_filter: str = "all") -> str:
    if not download_root:
        return "<div class='empty'>Set a download root first.</div>"
    if payload is None:
        return "<div class='empty'>No downloaded-image preview yet. Click Load downloaded image cards.</div>"
    records = payload.get('records', []) if isinstance(payload, dict) else []
    if not isinstance(records, list) or not records:
        return "<div class='empty'>No downloaded local images found.</div>"
    reviews = review_rows or {}
    cards: list[str] = []
    approved_count = 0
    visible_count = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = _record_id(record)
        field_key = _field_key(record_id)
        review = reviews.get(record_id, {}) if isinstance(reviews.get(record_id, {}), dict) else {}
        labels_value = ', '.join(str(item) for item in review.get('positive_labels', []))
        note_value = str(review.get('notes', ''))
        status_value = str(review.get('status', 'pending'))
        if status_value == 'approved':
            approved_count += 1
        if review_filter in {'approved', 'pending', 'rejected'} and status_value != review_filter:
            continue
        visible_count += 1
        image_src = _local_image_data_uri(str(record.get('local_path', '')))
        title = html.escape(str(record.get('title', '') or record_id))
        provider = html.escape(str(record.get('provider', '') or 'local'))
        local_path = html.escape(str(record.get('local_path', '') or ''))
        thumb_html = f"<img class='record-thumb' src='{image_src}' alt='{title}'>" if image_src else "<div class='empty'>no preview</div>"
        cards.append(f"""
        <div class='record-card'>
          <div class='record-thumb-wrap'>{thumb_html}</div>
          <div class='record-meta'>
            <strong>{title}</strong>
            <small>{provider}</small>
            <small>{local_path}</small>
            <small>status: {html.escape(status_value)}</small>
          </div>
          <label>Positive labels</label>
          <input name='labels_{field_key}' value='{html.escape(labels_value)}' list='vlso-label-suggestions'>
          <label>Notes</label>
          <input name='notes_{field_key}' value='{html.escape(note_value)}'>
          <label>Review status</label>
          <select name='status_{field_key}'>
            <option value='pending'{' selected' if status_value == 'pending' else ''}>pending</option>
            <option value='approved'{' selected' if status_value == 'approved' else ''}>approved</option>
            <option value='rejected'{' selected' if status_value == 'rejected' else ''}>rejected</option>
          </select>
        </div>
        """)
    stats = ''.join([
        _metric_card('Downloaded images', len(records)),
        _metric_card('Visible cards', visible_count),
        _metric_card('Approved labels', approved_count),
        _metric_card('Labels output', labels_output),
        _metric_card('Review file', review_path),
    ])
    hidden = ''.join([
        f"<input type='hidden' name='downloaded_record_id' value='{html.escape(_record_id(record))}'>"
        f"<input type='hidden' name='downloaded_local_path_{_field_key(_record_id(record))}' value='{html.escape(str(record.get('local_path', '')))}'>"
        for record in records if isinstance(record, dict)
    ])
    datalist = ''.join(f"<option value='{html.escape(label)}'></option>" for label in VLSO_LABEL_SUGGESTIONS)
    return f"""
    <div class='summary-grid'>{stats}</div>
    <form method='post' class='cluster-form'>
      <input type='hidden' name='vlso_download_root' value='{html.escape(download_root)}'>
      <input type='hidden' name='vlso_download_labels_output' value='{html.escape(labels_output)}'>
      <input type='hidden' name='vlso_download_concept_store_output' value='{html.escape(concept_store)}'>
      <input type='hidden' name='vlso_download_operator_store_output' value='{html.escape(operator_store)}'>
      <input type='hidden' name='vlso_download_review_path' value='{html.escape(review_path)}'>
      <label>Review filter</label>
      <select name='vlso_download_review_filter'>
        <option value='all'{' selected' if review_filter == 'all' else ''}>all</option>
        <option value='approved'{' selected' if review_filter == 'approved' else ''}>approved</option>
        <option value='pending'{' selected' if review_filter == 'pending' else ''}>pending</option>
        <option value='rejected'{' selected' if review_filter == 'rejected' else ''}>rejected</option>
      </select>
      <div class='actions'>
        <button class='secondary' name='action' value='load_vlso_downloaded_preview'>Apply filter</button>
      </div>
    </form>
    <form method='post'>
      <input type='hidden' name='vlso_download_root' value='{html.escape(download_root)}'>
      <input type='hidden' name='vlso_download_labels_output' value='{html.escape(labels_output)}'>
      <input type='hidden' name='vlso_download_concept_store_output' value='{html.escape(concept_store)}'>
      <input type='hidden' name='vlso_download_operator_store_output' value='{html.escape(operator_store)}'>
      <input type='hidden' name='vlso_download_review_path' value='{html.escape(review_path)}'>
      <input type='hidden' name='vlso_download_review_filter' value='{html.escape(review_filter)}'>
      <datalist id='vlso-label-suggestions'>{datalist}</datalist>
      {hidden}
      <div class='record-grid'>{''.join(cards)}</div>
      <div class='actions'>
        <button class='secondary' name='action' value='load_vlso_downloaded_preview'>Refresh downloaded image cards</button>
        <button class='secondary' name='action' value='save_vlso_download_labels'>Save review queue</button>
        <button class='primary' name='action' value='retrain_vlso_download_labels'>Retrain approved labels only</button>
      </div>
    </form>
    """


def render_visual_record_preview(records_path: str, payload: dict[str, object] | None, selected_ids: list[str], allow_providers: str, allow_licenses: str, approved_output: str, manifest_output: str, download_root: str, accept_all: bool, execute_downloads: bool) -> str:
    if not records_path:
        return "<div class='empty'>Set a records JSONL path first.</div>"
    if payload is None:
        return "<div class='empty'>No preview loaded yet. Click Load image preview cards.</div>"
    records = payload.get('records', []) if isinstance(payload, dict) else []
    if not isinstance(records, list) or not records:
        return "<div class='empty'>No image records found to preview.</div>"
    selected = set(selected_ids)
    cards: list[str] = []
    for record in records[:24]:
        if not isinstance(record, dict):
            continue
        record_id = _record_id(record)
        media_url = html.escape(str(record.get('media_url', '')))
        title = html.escape(str(record.get('title', '') or '(untitled)'))
        provider = html.escape(str(record.get('provider', '')))
        license_name = html.escape(str(record.get('license', '') or '-'))
        page_url = html.escape(str(record.get('page_url', '') or ''))
        checked = ' checked' if record_id in selected else ''
        page_link = f"<a href='{page_url}' target='_blank' rel='noopener'>source</a>" if page_url else ''
        cards.append(f"""
        <label class='record-card'>
          <input type='checkbox' name='selected_record' value='{html.escape(record_id)}'{checked}>
          <div class='record-thumb-wrap'>
            <img class='record-thumb' src='{media_url}' alt='{title}'>
          </div>
          <div class='record-meta'>
            <strong>{title}</strong>
            <small>{provider} / {license_name}</small>
            <small>{page_link}</small>
          </div>
        </label>
        """)
    cards_html = ''.join(cards)
    summary = payload.get('summary', {}) if isinstance(payload.get('summary'), dict) else {}
    stats = ''.join([
        _metric_card('Loaded records', summary.get('loaded_records', len(records))),
        _metric_card('Previewed', min(len(records), 24)),
        _metric_card('Selected', len(selected_ids)),
    ])
    accept_value = '1' if accept_all else ''
    execute_value = '1' if execute_downloads else ''
    return f"""
    <div class='summary-grid'>{stats}</div>
    <form method='post'>
      <input type='hidden' name='vlso_records_input' value='{html.escape(records_path)}'>
      <input type='hidden' name='vlso_approved_output' value='{html.escape(approved_output)}'>
      <input type='hidden' name='vlso_download_manifest_output' value='{html.escape(manifest_output)}'>
      <input type='hidden' name='vlso_download_root' value='{html.escape(download_root)}'>
      <input type='hidden' name='vlso_allow_providers' value='{html.escape(allow_providers)}'>
      <input type='hidden' name='vlso_allow_licenses' value='{html.escape(allow_licenses)}'>
      <input type='hidden' name='vlso_accept_all' value='0'>
      <input type='hidden' name='vlso_execute_downloads' value='{execute_value}'>
      <label class='inline'><input type='checkbox' name='vlso_accept_all' value='1'{_checked(bool(accept_all))}> Select all previewed records</label>
      <div class='record-grid'>{cards_html}</div>
      <div class='actions'>
        <button class='secondary' name='action' value='load_vlso_records_preview'>Refresh preview</button>
        <button class='primary' name='action' value='prepare_vlso_downloads'>Prepare selected downloads</button>
      </div>
    </form>
    """


def _render_generic_summary(kind: str, payload: dict[str, object]) -> str:
    cards = []
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)):
            cards.append(_metric_card(key.replace('_', ' '), value))
    body = ''.join(cards) if cards else "<div class='empty'>No quick summary available.</div>"
    return f"<div class='summary-grid'>{body}</div>{_raw_details(payload)}"


def _render_ops_summary(payload: dict[str, object]) -> str:
    answer_text = payload.get('response_text') or payload.get('answer_text') or payload.get('summary') or ''
    metrics = []
    for key in ('domain', 'scenario', 'plan_executability', 'relation_recovery', 'human_audit_usefulness'):
        if key in payload:
            metrics.append(_metric_card(key.replace('_', ' '), payload.get(key)))
    details = [item for item in (
        _info_block('Answer', answer_text) if answer_text else '',
        _info_block('Warnings', payload.get('warnings')) if payload.get('warnings') else '',
        _info_block('Next steps', payload.get('recommended_actions')) if payload.get('recommended_actions') else '',
    ) if item]
    return f"<div class='summary-grid'>{''.join(metrics) or _metric_card('Status', 'completed')}</div>{''.join(details)}{_raw_details(payload)}"


def _render_cp_summary(payload: dict[str, object]) -> str:
    validation = payload.get('validation_report', {}) if isinstance(payload.get('validation_report'), dict) else {}
    metrics = ''.join([
        _metric_card('Algorithm', payload.get('category') or payload.get('algorithm_family') or '-'),
        _metric_card('Complexity', payload.get('time_complexity') or payload.get('complexity') or '-'),
        _metric_card('Selection', payload.get('selection_strategy') or '-'),
        _metric_card('Validation', validation.get('overall_ok') if validation else 'n/a'),
    ])
    trace = payload.get('search_trace') or []
    top_trace = trace[0] if isinstance(trace, list) and trace else {}
    details = ''.join([
        _info_block('Sketch', payload.get('reasoning_outline') or payload.get('analysis') or '-'),
        _info_block('Top candidate', top_trace.get('category') if isinstance(top_trace, dict) else '-'),
        _info_block('Validator notes', validation.get('notes') or validation.get('failure_type') or '-'),
    ])
    return f"<div class='summary-grid'>{metrics}</div>{details}{_raw_details(payload)}"


def _render_vision_summary(payload: dict[str, object]) -> str:
    answer = payload.get('answer', {}) if isinstance(payload.get('answer'), dict) else {}
    world = payload.get('world', {}) if isinstance(payload.get('world'), dict) else {}
    entities = world.get('entities', []) if isinstance(world.get('entities'), list) else []
    visual_entities = [entity for entity in entities if isinstance(entity, dict) and entity.get('modality') == 'vision']
    objects = []
    for entity in visual_entities[:8]:
        attrs = entity.get('attributes', {}) if isinstance(entity.get('attributes'), dict) else {}
        labels = attrs.get('concept_labels') or []
        if labels:
            objects.append(f"{entity.get('id')}: {', '.join(str(item) for item in labels[:3])}")
    metrics = ''.join([
        _metric_card('Vision entities', len(visual_entities)),
        _metric_card('Relations', len(world.get('relations', []) if isinstance(world.get('relations'), list) else [])),
        _metric_card('Opening candidates', len(_opening_candidates(world))),
        _metric_card('Backend', ((world.get('metadata') or {}).get('vision_backend') or {}).get('active_backend', '-')),
    ])
    details = ''.join([
        _info_block('Answer', answer.get('answer_text') or '-'),
        _info_block('Likely objects', objects or '-'),
        _info_block('Likely openings', _opening_candidates(world) or '-'),
        _info_block('Warnings', answer.get('warnings') or world.get('warnings') or '-'),
    ])
    return f"<div class='summary-grid'>{metrics}</div>{details}{_raw_details(payload)}"


def _render_vision_compare_summary(payload: dict[str, object]) -> str:
    summary = payload.get('summary', {}) if isinstance(payload.get('summary'), dict) else {}
    metrics = ''.join([
        _metric_card('Primary store', summary.get('primary_store') or '-'),
        _metric_card('Compare store', summary.get('compare_store') or '-'),
    ])
    details = ''.join([
        _info_block('Primary answer', summary.get('primary_answer') or '-'),
        _info_block('Compare answer', summary.get('compare_answer') or '-'),
        _info_block('Primary openings', summary.get('primary_opening_candidates') or '-'),
        _info_block('Compare openings', summary.get('compare_opening_candidates') or '-'),
    ])
    return f"<div class='summary-grid'>{metrics}</div>{details}{_raw_details(payload)}"


def _render_review_impact_summary(payload: dict[str, object]) -> str:
    primary = payload.get('primary', {}) if isinstance(payload.get('primary'), dict) else {}
    compare = payload.get('compare', {}) if isinstance(payload.get('compare'), dict) else {}
    delta = payload.get('delta', {}) if isinstance(payload.get('delta'), dict) else {}
    metrics = ''.join([
        _metric_card('Primary answer acc', primary.get('grounded_answer_accuracy')),
        _metric_card('Compare answer acc', compare.get('grounded_answer_accuracy')),
        _metric_card('Answer delta', delta.get('grounded_answer_accuracy')),
        _metric_card('Relation delta', delta.get('relation_recall')),
    ])
    return f"<div class='summary-grid'>{metrics}</div>{_raw_details(payload)}"


def _render_cp_parser_compare_summary(payload: dict[str, object]) -> str:
    heuristic = payload.get('heuristic', {}) if isinstance(payload.get('heuristic'), dict) else {}
    model = payload.get('model', {}) if isinstance(payload.get('model'), dict) else {}
    delta = payload.get('delta', {}) if isinstance(payload.get('delta'), dict) else {}
    metrics = ''.join([
        _metric_card('Heuristic algorithm EM', heuristic.get('algorithm_exact_match')),
        _metric_card('Model algorithm EM', model.get('algorithm_exact_match')),
        _metric_card('Algorithm delta', delta.get('algorithm_exact_match')),
        _metric_card('Frame delta', delta.get('frame_jaccard')),
    ])
    return f"<div class='summary-grid'>{metrics}</div>{_raw_details(payload)}"


def _render_cp_lora_experiment_summary(payload: dict[str, object]) -> str:
    heuristic = payload.get('heuristic_eval', {}) if isinstance(payload.get('heuristic_eval'), dict) else {}
    training = payload.get('training_summary', {}) if isinstance(payload.get('training_summary'), dict) else {}
    compare_eval = payload.get('compare_eval', {}) if isinstance(payload.get('compare_eval'), dict) else {}
    metrics = ''.join([
        _metric_card('Heuristic algorithm EM', heuristic.get('algorithm_exact_match')),
        _metric_card('Heuristic frame jaccard', heuristic.get('frame_jaccard')),
        _metric_card('Training mode', training.get('mode') or '-'),
        _metric_card('Latest checkpoint', training.get('latest_checkpoint') or '-'),
    ])
    details = ''.join([
        _info_block('Workspace train JSONL', payload.get('train_jsonl') or '-'),
        _info_block('Workspace val JSONL', payload.get('val_jsonl') or '-'),
        _info_block('Final model dir', training.get('final_model_dir') or '-'),
        _info_block('Compare eval', compare_eval or '-'),
    ])
    return f"<div class='summary-grid'>{metrics}</div>{details}{_raw_details(payload)}"


def _render_understanding_summary(payload: dict[str, object]) -> str:
    progress = payload.get('progress', {}) if isinstance(payload.get('progress'), dict) else {}
    interpretation = payload.get('interpretation', {}) if isinstance(payload.get('interpretation'), dict) else {}
    operator_arch = progress.get('operator_architecture', {}) if isinstance(progress.get('operator_architecture'), dict) else {}
    premise_reasoning = progress.get('premise_reasoning', {}) if isinstance(progress.get('premise_reasoning'), dict) else {}
    shared_world = progress.get('shared_world_model', {}) if isinstance(progress.get('shared_world_model'), dict) else {}
    raw_visual = progress.get('raw_visual_reasoning', {}) if isinstance(progress.get('raw_visual_reasoning'), dict) else {}
    metrics = ''.join([
        _metric_card('Research architecture', progress.get('research_architecture_overall')),
        _metric_card('Robust understanding', progress.get('robust_general_intelligence_overall')),
        _metric_card('Premise reasoning', premise_reasoning.get('score')),
        _metric_card('Raw visual reasoning', raw_visual.get('score')),
        _metric_card('Operator architecture', operator_arch.get('score')),
        _metric_card('Shared world model', shared_world.get('score')),
    ])
    strengths = ''.join(f"<li>{html.escape(str(item))}</li>" for item in interpretation.get('strengths', []))
    risks = ''.join(f"<li>{html.escape(str(item))}</li>" for item in interpretation.get('risks', []))
    next_steps = ''.join(f"<li>{html.escape(str(item))}</li>" for item in interpretation.get('next_steps', []))
    overview = f"<div class='info-block'><small>Headline</small><div>{html.escape(str(interpretation.get('headline', '-')))}</div></div>"
    lists = (
        f"<div class='mini-grid'><div class='info-block'><small>Strengths</small><ul>{strengths or '<li>-</li>'}</ul></div>"
        f"<div class='info-block'><small>Risks</small><ul>{risks or '<li>-</li>'}</ul></div></div>"
        f"<div class='info-block'><small>Next steps</small><ul>{next_steps or '<li>-</li>'}</ul></div>"
    )
    return f"<div class='summary-grid'>{metrics}</div>{overview}{lists}{_raw_details(payload)}"


def _render_operator_training_summary(payload: dict[str, object]) -> str:
    export_summary = payload.get('teacher_trace_summary', {}) if isinstance(payload.get('teacher_trace_summary'), dict) else {}
    bundle_summary = payload.get('bundle_summary', {}) if isinstance(payload.get('bundle_summary'), dict) else {}
    training = payload.get('training_summary', {}) if isinstance(payload.get('training_summary'), dict) else {}
    metrics = ''.join([
        _metric_card('Teacher traces', export_summary.get('num_traces')),
        _metric_card('Train rows', bundle_summary.get('num_train')),
        _metric_card('Val rows', bundle_summary.get('num_val')),
        _metric_card('Training mode', training.get('mode') or '-'),
    ])
    details = ''.join([
        _info_block('Teacher trace JSONL', payload.get('teacher_trace_path') or '-'),
        _info_block('Bundle workspace', payload.get('workspace') or '-'),
        _info_block('Latest checkpoint', training.get('latest_checkpoint') or '-'),
        _info_block('Final model dir', training.get('final_model_dir') or '-'),
    ])
    return f"<div class='summary-grid'>{metrics}</div>{details}{_raw_details(payload)}"


def _opening_candidates(world_payload: dict[str, object]) -> list[str]:
    entities = world_payload.get("entities", []) if isinstance(world_payload, dict) else []
    ranked: list[tuple[int, str]] = []
    for entity in entities:
        if not isinstance(entity, dict) or entity.get("modality") != "vision":
            continue
        attrs = entity.get("attributes", {}) if isinstance(entity.get("attributes"), dict) else {}
        labels = attrs.get("concept_labels") or []
        if not isinstance(labels, list):
            continue
        upper = {str(item).upper() for item in labels}
        if not ({"ACCESS_OPENING_CANDIDATE", "ZIPPER_LIKE_PART", "EDGE_OPENING"} & upper):
            continue
        score = 0
        if "ACCESS_OPENING_CANDIDATE" in upper:
            score += 3
        if "ZIPPER_LIKE_PART" in upper:
            score += 2
        if "EDGE_OPENING" in upper:
            score += 2
        if "STRAP_LIKE_PART" in upper and "ACCESS_OPENING_CANDIDATE" not in upper:
            score -= 2
        bbox = attrs.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            area = max(1, (x2 - x1) * (y2 - y1))
            if (x1 <= 2 or y1 <= 2) and area <= 600:
                continue
            if x1 <= 2 or y1 <= 2:
                score -= 4
            if area <= 600:
                score -= 2
        entity_id = str(entity.get("id", ""))
        if entity_id:
            ranked.append((score, entity_id))
    ranked.sort(key=lambda item: item[0], reverse=True)
    output: list[str] = []
    seen: set[str] = set()
    for score, entity_id in ranked:
        if score <= 0 or entity_id in seen:
            continue
        output.append(entity_id)
        seen.add(entity_id)
    return output


def build_vision_payload(query: str, image_path: str, mode: str, answer_mode: str, concept_store: str, operator_store: str, weights: str) -> dict[str, object]:
    reasoner = VLSOReasoner(
        mode=mode,
        concept_store_path=concept_store or None,
        operator_store_path=operator_store or None,
        affordance_weights_path=weights or None,
        answer_mode=answer_mode,
    )
    world, answer = reasoner.answer(
        query,
        visual_input={"image_path": image_path, "metadata": {"image_path": image_path}},
    )
    return {"world": world.model_dump(), "answer": answer.model_dump()}


def build_vision_comparison(query: str, image_path: str, mode: str, answer_mode: str, primary_store: str, primary_operator_store: str, compare_store: str, compare_operator_store: str, weights: str) -> dict[str, object]:
    primary = build_vision_payload(query, image_path, mode, answer_mode, primary_store, primary_operator_store, weights)
    comparison = build_vision_payload(query, image_path, mode, answer_mode, compare_store, compare_operator_store, weights)
    primary_world = primary.get("world", {}) if isinstance(primary.get("world"), dict) else {}
    comparison_world = comparison.get("world", {}) if isinstance(comparison.get("world"), dict) else {}
    return {
        "primary": primary,
        "comparison": comparison,
        "summary": {
            "primary_store": primary_store,
            "primary_operator_store": primary_operator_store,
            "compare_store": compare_store,
            "compare_operator_store": compare_operator_store,
            "primary_answer": primary.get("answer", {}).get("answer_text", "") if isinstance(primary.get("answer"), dict) else "",
            "compare_answer": comparison.get("answer", {}).get("answer_text", "") if isinstance(comparison.get("answer"), dict) else "",
            "primary_opening_candidates": _opening_candidates(primary_world),
            "compare_opening_candidates": _opening_candidates(comparison_world),
        },
    }


def render_result(kind: str, payload: dict[str, object] | None) -> str:
    if not kind or payload is None:
        return "<div class='empty'>No result yet. Load an example above and run it.</div>"
    titles = {
        "ops": "Ops result",
        "cp": "CP result",
        "vision": "Vision result",
        "vision_compare": "Vision concept-store comparison",
        "cp_download": "CP labeled dataset result",
        "vlso_collection": "VLSO collection plan result",
        "vlso_download_prepare": "VLSO download preparation result",
        "vlso_geometry_generate": "Synthetic geometry dataset result",
        "cp_geometry_generate": "CP geometry eval result",
        "cp_geometry_train_dry_run": "CP geometry training dry-run result",
        "vlso_geometry_pipeline": "VLSO geometry self-training result",
        "vlso_cluster_review": "VLSO cluster review result",
        "vlso_review_retrain": "Approved-cluster retrain result",
        "vlso_review_impact": "VLSO approved-store impact",
        "cp_parser_compare": "CP parser comparison",
        "cp_lora_experiment": "CP LoRA experiment",
        "operator_training": "Generic operator student training",
    }
    if kind == "ops":
        body = _render_ops_summary(payload)
    elif kind == "cp":
        body = _render_cp_summary(payload)
    elif kind == "vision":
        body = _render_vision_summary(payload)
    elif kind == "vision_compare":
        body = _render_vision_compare_summary(payload)
    elif kind == "vlso_review_impact":
        body = _render_review_impact_summary(payload)
    elif kind == "cp_parser_compare":
        body = _render_cp_parser_compare_summary(payload)
    elif kind == "cp_lora_experiment":
        body = _render_cp_lora_experiment_summary(payload)
    elif kind == "operator_training":
        body = _render_operator_training_summary(payload)
    elif kind == "understanding_eval":
        body = _render_understanding_summary(payload)
    else:
        body = _render_generic_summary(kind, payload)
    return f"<h3>{html.escape(titles.get(kind, kind))}</h3>{body}"


def render_cluster_review(summary_path: str, review_path: str, payload: dict[str, object] | None) -> str:
    if not summary_path:
        return "<div class='empty'>Set a self-training summary path to inspect clusters.</div>"
    if payload is None:
        return "<div class='empty'>No cluster data loaded yet. Click Load cluster summary.</div>"
    stats = payload.get("stats", {}) if isinstance(payload, dict) else {}
    clusters = payload.get("clusters", []) if isinstance(payload, dict) else []
    cards: list[str] = []
    for cluster in clusters[:12] if isinstance(clusters, list) else []:
        if not isinstance(cluster, dict):
            continue
        cluster_id = html.escape(str(cluster.get("cluster_id", "")))
        support = html.escape(str(cluster.get("support", "")))
        accepted = ", ".join(str(item.get("label", "")) for item in cluster.get("accepted_labels", []) if isinstance(item, dict))
        suggested = ", ".join(str(item.get("label", "")) for item in cluster.get("suggested_labels", []) if isinstance(item, dict))
        review = cluster.get("review", {}) if isinstance(cluster.get("review"), dict) else {}
        current_status = html.escape(str(review.get("status", "pending")))
        current_note = html.escape(str(review.get("note", "")))
        cards.append(f"""
        <div class='cluster-card'>
          <div class='cluster-top'>
            <strong>{cluster_id}</strong>
            <span>support {support}</span>
            <span>review {current_status}</span>
            <span>priority {html.escape(str(cluster.get('review_priority', 0.0)))}</span>
          </div>
          <div><small>accepted labels</small><div>{html.escape(accepted or 'none')}</div></div>
          <div><small>suggested labels</small><div>{html.escape(suggested or 'none')}</div></div>
          <div><small>priority reason</small><div>{html.escape(str(cluster.get('review_reason', 'stable cluster')))}</div></div>
          <div><small>note</small><div>{current_note or 'none'}</div></div>
          <form method='post' class='cluster-form'>
            <input type='hidden' name='cluster_summary_path' value='{html.escape(summary_path)}'>
            <input type='hidden' name='cluster_review_path' value='{html.escape(review_path)}'>
            <input type='hidden' name='cluster_id' value='{cluster_id}'>
            <label>Approved labels (comma separated)</label>
            <input name='approved_labels' value='{html.escape(accepted)}'>
            <label>Review note</label>
            <input name='cluster_note' value=''>
            <div class='actions'>
              <button class='secondary' name='cluster_status' value='approved'>Approve</button>
              <button class='secondary' name='cluster_status' value='rejected'>Reject</button>
              <button class='secondary' name='cluster_status' value='pending'>Reset</button>
              <button class='primary' name='action' value='review_vlso_cluster'>Save review</button>
            </div>
          </form>
        </div>
        """)
    stats_html = f"total {html.escape(str(stats.get('total', 0)))} / approved {html.escape(str(stats.get('approved', 0)))} / rejected {html.escape(str(stats.get('rejected', 0)))} / pending {html.escape(str(stats.get('pending', 0)))}"
    body = ''.join(cards) if cards else "<div class='empty'>No clusters found.</div>"
    return f"<div class='cluster-stats'>{stats_html}</div><div class='cluster-grid'>{body}</div>"


class StarterApp:
    def __init__(self) -> None:
        self.ops = DomainCopilot(mode="heuristic", review_queue_path="data/ops_review_queue.db")
        self.cp = CompetitiveProgrammingReasoner(episode_store_path="data/cp_episodes.db")

    def handle(self, form: dict[str, list[str]]) -> str:
        action = _first(form, "action", "")
        flash = ""
        flash_tone = "neutral"
        result_kind = ""
        result_payload: dict[str, object] | None = None
        cluster_payload: dict[str, object] | None = None
        visual_record_payload: dict[str, object] | None = None
        downloaded_record_payload: dict[str, object] | None = None
        downloaded_review_payload: dict[str, dict[str, object]] | None = None

        ops_query = _first(form, "ops_query", OPS_EXAMPLE["query"])
        ops_context = _first(form, "ops_context", OPS_EXAMPLE["context"])
        ops_domain = _first(form, "ops_domain", OPS_EXAMPLE["domain"])
        ops_scenario = _first(form, "ops_scenario", OPS_EXAMPLE["scenario"])
        cp_statement = _first(form, "cp_statement", CP_EXAMPLE["statement"])
        vision_query = _first(form, "vision_query", VISION_EXAMPLE["query"])
        vision_image = _first(form, "vision_image", VISION_EXAMPLE["image_path"])
        vision_mode = _first(form, "vision_mode", "deep")
        vision_answer_mode = _first(form, "vision_answer_mode", "structured")
        vision_concept_store = _first(form, "vision_concept_store", VLSO_COMPARE_EXAMPLE["primary_store"])
        vision_compare_store = _first(form, "vision_compare_store", VLSO_COMPARE_EXAMPLE["compare_store"])
        vision_operator_store = _first(form, "vision_operator_store", VLSO_COMPARE_EXAMPLE["operator_store"])
        vision_compare_operator_store = _first(form, "vision_compare_operator_store", VLSO_COMPARE_EXAMPLE["compare_operator_store"])
        vision_weights = _first(form, "vision_weights", "data/vlso_samples/trained_affordance_weights.json")
        cp_manifest = _first(form, "cp_manifest", CP_DOWNLOAD_EXAMPLE["manifest"])
        cp_download_root = _first(form, "cp_download_root", CP_DOWNLOAD_EXAMPLE["download_root"])
        cp_download_output = _first(form, "cp_download_output", CP_DOWNLOAD_EXAMPLE["output"])
        vlso_manifest = _first(form, "vlso_manifest", VLSO_COLLECTION_EXAMPLE["manifest"])
        vlso_collection_output = _first(form, "vlso_collection_output", VLSO_COLLECTION_EXAMPLE["output"])
        vlso_family_manifest = _first(form, "vlso_family_manifest", VLSO_FAMILY_COLLECTION_EXAMPLE["manifest"])
        vlso_family_names = _first(form, "vlso_family_names", VLSO_FAMILY_COLLECTION_EXAMPLE["families"])
        vlso_family_limit = _first(form, "vlso_family_limit", VLSO_FAMILY_COLLECTION_EXAMPLE["limit"])
        vlso_family_batch_workspace = _first(form, "vlso_family_batch_workspace", VLSO_FAMILY_BATCH_EXAMPLE["workspace"])
        vlso_family_targets = _first(form, "vlso_family_targets", VLSO_FAMILY_BATCH_EXAMPLE["targets"])
        vlso_dry_run = _checked_form(form, "vlso_dry_run", default=True)
        vlso_records_input = _first(form, "vlso_records_input", VLSO_DOWNLOAD_EXAMPLE["records"])
        vlso_approved_output = _first(form, "vlso_approved_output", VLSO_DOWNLOAD_EXAMPLE["approved"])
        vlso_download_manifest_output = _first(form, "vlso_download_manifest_output", VLSO_DOWNLOAD_EXAMPLE["manifest"])
        vlso_download_root = _first(form, "vlso_download_root", VLSO_DOWNLOAD_EXAMPLE["download_root"])
        vlso_allow_providers = _first(form, "vlso_allow_providers", VLSO_DOWNLOAD_EXAMPLE["providers"])
        vlso_allow_licenses = _first(form, "vlso_allow_licenses", VLSO_DOWNLOAD_EXAMPLE["licenses"])
        vlso_accept_all = _checked_form(form, "vlso_accept_all", default=False)
        vlso_execute_downloads = _checked_form(form, "vlso_execute_downloads", default=False)
        selected_record_ids = form.get("selected_record", [])
        cluster_summary_path = _first(form, "cluster_summary_path", VLSO_COMPARE_EXAMPLE["summary"])
        cluster_review_path = _first(form, "cluster_review_path", VLSO_COMPARE_EXAMPLE["reviews"])
        geometry_output_dir = _first(form, "geometry_output_dir", GEOMETRY_GENERATOR_EXAMPLE["output_dir"])
        geometry_eval_output = _first(form, "geometry_eval_output", GEOMETRY_GENERATOR_EXAMPLE["eval_output"])
        cp_geometry_eval_output = _first(form, "cp_geometry_eval_output", CP_GEOMETRY_EXAMPLE["eval_output"])
        cp_geometry_train_output = _first(form, "cp_geometry_train_output", CP_GEOMETRY_EXAMPLE["train_output"])
        cp_geometry_val_output = _first(form, "cp_geometry_val_output", CP_GEOMETRY_EXAMPLE["val_output"])
        cp_geometry_model = _first(form, "cp_geometry_model", CP_GEOMETRY_EXAMPLE["model"])
        cp_geometry_model_output = _first(form, "cp_geometry_model_output", CP_GEOMETRY_EXAMPLE["output_dir"])
        cp_compare_input = _first(form, "cp_compare_input", CP_GEOMETRY_EXAMPLE["compare_input"])
        cp_lora_workspace = _first(form, "cp_lora_workspace", CP_LORA_EXAMPLE["workspace"])
        cp_lora_model = _first(form, "cp_lora_model", CP_LORA_EXAMPLE["model"])
        cp_lora_train_inputs = _first(form, "cp_lora_train_inputs", CP_LORA_EXAMPLE["train_inputs"])
        cp_lora_eval_inputs = _first(form, "cp_lora_eval_inputs", CP_LORA_EXAMPLE["eval_inputs"])
        cp_lora_max_steps = _first(form, "cp_lora_max_steps", CP_LORA_EXAMPLE["max_steps"])
        cp_lora_save_steps = _first(form, "cp_lora_save_steps", CP_LORA_EXAMPLE["save_steps"])
        cp_lora_save_total_limit = _first(form, "cp_lora_save_total_limit", CP_LORA_EXAMPLE["save_total_limit"])
        cp_lora_resume_latest = _checked_form(form, "cp_lora_resume_latest", default=True)
        cp_lora_dry_run = _checked_form(form, "cp_lora_dry_run", default=True)
        operator_train_workspace = _first(form, "operator_train_workspace", OPERATOR_TRAIN_EXAMPLE["workspace"])
        operator_train_model = _first(form, "operator_train_model", OPERATOR_TRAIN_EXAMPLE["model"])
        operator_train_teacher_output = _first(form, "operator_train_teacher_output", OPERATOR_TRAIN_EXAMPLE["teacher_output"])
        operator_train_teacher_sft_output = _first(form, "operator_train_teacher_sft_output", OPERATOR_TRAIN_EXAMPLE["teacher_sft_output"])
        operator_train_max_steps = _first(form, "operator_train_max_steps", OPERATOR_TRAIN_EXAMPLE["max_steps"])
        operator_train_save_steps = _first(form, "operator_train_save_steps", OPERATOR_TRAIN_EXAMPLE["save_steps"])
        operator_train_save_total_limit = _first(form, "operator_train_save_total_limit", OPERATOR_TRAIN_EXAMPLE["save_total_limit"])
        operator_train_resume_latest = _checked_form(form, "operator_train_resume_latest", default=True)
        operator_train_dry_run = _checked_form(form, "operator_train_dry_run", default=True)
        operator_train_local_files_only = _checked_form(form, "operator_train_local_files_only", default=True)
        operator_train_use_lora = _checked_form(form, "operator_train_use_lora", default=True)
        operator_train_use_qlora = _checked_form(form, "operator_train_use_qlora", default=False)
        vision_eval_input = _first(form, "vision_eval_input", VLSO_COMPARE_EXAMPLE["eval_input"])
        understanding_hidden_input = _first(form, "understanding_hidden_input", UNDERSTANDING_EVAL_EXAMPLE["hidden"])
        understanding_cp_input = _first(form, "understanding_cp_input", UNDERSTANDING_EVAL_EXAMPLE["cp"])
        understanding_cp_hidden_input = _first(form, "understanding_cp_hidden_input", UNDERSTANDING_EVAL_EXAMPLE["cp_hidden"])
        understanding_vlso_input = _first(form, "understanding_vlso_input", UNDERSTANDING_EVAL_EXAMPLE["vlso"])
        understanding_vlso_real_input = _first(form, "understanding_vlso_real_input", UNDERSTANDING_EVAL_EXAMPLE["vlso_real"])
        geometry_pipeline_input_root = _first(form, "geometry_pipeline_input_root", GEOMETRY_PIPELINE_EXAMPLE["input_root"])
        geometry_pipeline_workspace = _first(form, "geometry_pipeline_workspace", GEOMETRY_PIPELINE_EXAMPLE["workspace"])
        geometry_cluster_threshold = _first(form, "geometry_cluster_threshold", GEOMETRY_PIPELINE_EXAMPLE["cluster_threshold"])
        geometry_pseudo_threshold = _first(form, "geometry_pseudo_threshold", GEOMETRY_PIPELINE_EXAMPLE["pseudo_threshold"])
        geometry_consensus_threshold = _first(form, "geometry_consensus_threshold", GEOMETRY_PIPELINE_EXAMPLE["consensus_threshold"])
        geometry_concept_match_threshold = _first(form, "geometry_concept_match_threshold", GEOMETRY_PIPELINE_EXAMPLE["concept_match_threshold"])
        geometry_min_cluster_size = _first(form, "geometry_min_cluster_size", GEOMETRY_PIPELINE_EXAMPLE["min_cluster_size"])
        vlso_download_pipeline_workspace = _first(form, "vlso_download_pipeline_workspace", DOWNLOADED_IMAGE_PIPELINE_EXAMPLE["workspace"])
        vlso_download_labels_output = _first(form, "vlso_download_labels_output", DOWNLOADED_IMAGE_PIPELINE_EXAMPLE["labels"])
        vlso_download_concept_store_output = _first(form, "vlso_download_concept_store_output", DOWNLOADED_IMAGE_PIPELINE_EXAMPLE["concept_store"])
        vlso_download_operator_store_output = _first(form, "vlso_download_operator_store_output", DOWNLOADED_IMAGE_PIPELINE_EXAMPLE["operator_store"])
        vlso_download_review_path = _first(form, "vlso_download_review_path", DOWNLOADED_IMAGE_PIPELINE_EXAMPLE["review"])
        vlso_download_review_filter = _first(form, "vlso_download_review_filter", "all")

        try:
            if action == "load_ops_example":
                flash = "Loaded the ops example. You can run it directly."
            elif action == "load_cp_example":
                flash = "Loaded the CP example."
            elif action == "load_vision_example":
                flash = "Loaded the vision example."
            elif action == "run_ops":
                request = CopilotRequest(query=ops_query, context=ops_context, domain=ops_domain, scenario=ops_scenario)
                result = self.ops.run(request)
                result_kind = "ops"
                result_payload = result.model_dump()
                flash = "Ops reasoning finished."
                flash_tone = "success"
            elif action == "run_cp":
                solution = self.cp.solve(cp_statement)
                result_kind = "cp"
                result_payload = solution.model_dump() if solution is not None else {"message": "No statement provided."}
                flash = "CP analysis finished."
                flash_tone = "success"
            elif action == "run_vision":
                result_kind = "vision"
                result_payload = build_vision_payload(vision_query, vision_image, vision_mode, vision_answer_mode, vision_concept_store, vision_operator_store, vision_weights)
                flash = "Vision reasoning finished."
                flash_tone = "success"
            elif action == "run_vision_compare":
                result_kind = "vision_compare"
                result_payload = build_vision_comparison(vision_query, vision_image, vision_mode, vision_answer_mode, vision_concept_store, vision_operator_store, vision_compare_store, vision_compare_operator_store, vision_weights)
                flash = "Concept-store comparison finished."
                flash_tone = "success"
            elif action == "evaluate_vlso_review_impact":
                summary = VlsoReviewImpactEvaluator(
                    mode=vision_mode,
                    answer_mode='structured',
                    affordance_weights_path=vision_weights or None,
                ).compare_stores(
                    input_path=vision_eval_input,
                    primary_concept_store=vision_concept_store or None,
                    primary_operator_store=vision_operator_store or None,
                    compare_concept_store=vision_compare_store or None,
                    compare_operator_store=vision_compare_operator_store or None,
                )
                result_kind = "vlso_review_impact"
                result_payload = summary.model_dump()
                flash = "VLSO approved-store impact evaluation finished."
                flash_tone = "success"
            elif action == "compare_cp_parsers":
                examples = CpParserEvaluator.load_examples(cp_compare_input)
                model = CpLearnedParser(cp_geometry_model, local_files_only=False)
                summary = CpParserEvaluator().compare_examples(examples, model=model)
                result_kind = "cp_parser_compare"
                result_payload = summary.model_dump()
                flash = "CP parser comparison finished."
                flash_tone = "success"
            elif action == "run_understanding_eval":
                hidden_cases = _load_hidden_premise_cases(understanding_hidden_input) if understanding_hidden_input else None
                cp_examples = CpParserEvaluator.load_examples(Path(understanding_cp_input)) if understanding_cp_input else None
                cp_hidden_examples = CpParserEvaluator.load_examples(Path(understanding_cp_hidden_input)) if understanding_cp_hidden_input else None
                vlso_cases = VlsoGroundedEvaluator.load_cases(Path(understanding_vlso_input)) if understanding_vlso_input else None
                vlso_real_cases = VlsoGroundedEvaluator.load_cases(Path(understanding_vlso_real_input)) if understanding_vlso_real_input else None
                summary = SemOpUnderstandingEvaluator().evaluate(
                    hidden_premise_cases=hidden_cases,
                    cp_examples=cp_examples,
                    cp_hidden_examples=cp_hidden_examples,
                    vlso_cases=vlso_cases,
                    vlso_real_image_cases=vlso_real_cases,
                    cp_mode='heuristic',
                )
                result_kind = "understanding_eval"
                result_payload = summary.model_dump()
                flash = "Overall understanding benchmark finished."
                flash_tone = "success"
            elif action == "download_cp_labels":
                summary = CpLabeledDatasetDownloader().download_and_normalize(cp_manifest, cp_download_root, cp_download_output)
                result_kind = "cp_download"
                result_payload = summary.model_dump()
                flash = "CP labeled dataset normalization finished."
                flash_tone = "success"
            elif action == "build_vlso_family_manifest":
                families = [item.strip() for item in vlso_family_names.replace(',', ' ').split() if item.strip()]
                manifest_payload = build_object_family_manifest(families, limit_per_source=int(vlso_family_limit or '10'))
                manifest_path = Path(vlso_family_manifest)
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding='utf-8')
                vlso_manifest = vlso_family_manifest
                result_kind = "vlso_collection"
                result_payload = {"manifest": vlso_family_manifest, "families": families, "sources": len(manifest_payload.get('sources', []))}
                flash = "Built an object-family VLSO collection manifest."
                flash_tone = "success"
            elif action == "run_vlso_family_batch":
                family_targets = _parse_family_targets(vlso_family_targets)
                batch_summary = VisualDataCollector().run_family_batch(
                    family_targets=family_targets,
                    workspace=vlso_family_batch_workspace,
                    execute_collect=not vlso_dry_run,
                    execute_downloads=vlso_execute_downloads and not vlso_dry_run,
                    allow_providers=_tokens(vlso_allow_providers),
                    allow_licenses=_tokens(vlso_allow_licenses),
                    accept_all=True,
                    progress_callback=_gui_vlso_progress,
                )
                vlso_manifest = batch_summary.manifest_path
                vlso_records_input = batch_summary.records_path
                vlso_approved_output = batch_summary.approved_output
                vlso_download_manifest_output = batch_summary.download_manifest_output
                vlso_download_root = batch_summary.download_root
                result_kind = "vlso_collection"
                result_payload = batch_summary.model_dump()
                if not batch_summary.dry_run and Path(batch_summary.records_path).exists():
                    preview_records = [record.model_dump() for record in VisualDataCollector().load_records(batch_summary.records_path)[:24]]
                    visual_record_payload = {
                        "summary": {"loaded_records": len(preview_records), "records_path": batch_summary.records_path},
                        "records": preview_records,
                    }
                flash = "Ran the VLSO family batch collector."
                flash_tone = "success"
            elif action == "plan_vlso_collection":
                summary = VisualDataCollector().run_manifest(vlso_manifest, vlso_collection_output, dry_run=vlso_dry_run, progress_callback=_gui_vlso_progress)
                result_kind = "vlso_collection"
                result_payload = summary.model_dump()
                flash = "VLSO collection plan created."
                flash_tone = "success"
            elif action == "load_vlso_records_preview":
                records = [record.model_dump() for record in VisualDataCollector().load_records(vlso_records_input)[:24]]
                visual_record_payload = {
                    "summary": {"loaded_records": len(records), "records_path": vlso_records_input},
                    "records": records,
                }
                result_kind = "vlso_collection"
                result_payload = {"preview_loaded": True, "records_path": vlso_records_input, "preview_count": len(records)}
                flash = "Loaded VLSO image preview cards."
                flash_tone = "success"
            elif action == "prepare_vlso_downloads":
                summary = VisualDataCollector().prepare_downloads(
                    records_path=vlso_records_input,
                    approved_output=vlso_approved_output,
                    manifest_output=vlso_download_manifest_output,
                    download_root=vlso_download_root,
                    allow_providers=_tokens(vlso_allow_providers),
                    allow_licenses=_tokens(vlso_allow_licenses),
                    approved_ids=selected_record_ids or None,
                    accept_all=vlso_accept_all,
                    execute=vlso_execute_downloads,
                    progress_callback=_gui_vlso_progress,
                )
                visual_record_payload = {
                    "summary": {"loaded_records": len(VisualDataCollector().load_records(vlso_records_input)[:24]), "records_path": vlso_records_input},
                    "records": [record.model_dump() for record in VisualDataCollector().load_records(vlso_records_input)[:24]],
                }
                result_kind = "vlso_download_prepare"
                result_payload = summary.model_dump()
                flash = "VLSO download manifest prepared." if not vlso_execute_downloads else "VLSO downloads executed."
                flash_tone = "success"
            elif action == "load_vlso_downloaded_preview":
                local_records = _collect_local_downloaded_images(vlso_download_root)
                downloaded_record_payload = {
                    "summary": {"loaded_records": len(local_records), "download_root": vlso_download_root},
                    "records": local_records,
                }
                downloaded_review_payload = _load_download_label_reviews(vlso_download_review_path)
                result_kind = "vlso_download_prepare"
                result_payload = {"download_root": vlso_download_root, "preview_count": len(local_records), "local_preview": True}
                flash = "Loaded downloaded local image cards."
                flash_tone = "success"
            elif action == "save_vlso_download_labels":
                records = _collect_local_downloaded_images(vlso_download_root)
                review_rows: dict[str, dict[str, object]] = {}
                label_rows = []
                for record in records:
                    record_id = _record_id(record)
                    key = _field_key(record_id)
                    labels = [item.strip() for item in _first(form, f"labels_{key}", "").split(',') if item.strip()]
                    notes = _first(form, f"notes_{key}", "")
                    status = _first(form, f"status_{key}", "pending")
                    review_rows[record_id] = {
                        "positive_labels": labels,
                        "notes": notes,
                        "status": status,
                        "image_path": str(record.get("local_path", "")),
                    }
                    if labels:
                        label_rows.append({
                            "image_path": str(record.get("local_path", "")),
                            "targets": [{"subject_id": "", "positive_labels": labels, "negative_labels": [], "notes": notes}],
                        })
                _save_download_label_reviews(vlso_download_review_path, review_rows)
                output = Path(vlso_download_labels_output)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open('w', encoding='utf-8') as handle:
                    for row in label_rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                downloaded_record_payload = {
                    "summary": {"loaded_records": len(records), "download_root": vlso_download_root},
                    "records": records,
                }
                downloaded_review_payload = review_rows
                result_kind = "vlso_download_prepare"
                result_payload = {"labels_output": vlso_download_labels_output, "saved_examples": len(label_rows), "review_path": vlso_download_review_path}
                flash = "Saved the downloaded-image review queue."
                flash_tone = "success"
            elif action == "retrain_vlso_download_labels":
                records = _collect_local_downloaded_images(vlso_download_root)
                review_rows: dict[str, dict[str, object]] = {}
                approved_rows = []
                for record in records:
                    record_id = _record_id(record)
                    key = _field_key(record_id)
                    labels = [item.strip() for item in _first(form, f"labels_{key}", "").split(',') if item.strip()]
                    notes = _first(form, f"notes_{key}", "")
                    status = _first(form, f"status_{key}", "pending")
                    review_rows[record_id] = {
                        "positive_labels": labels,
                        "notes": notes,
                        "status": status,
                        "image_path": str(record.get("local_path", "")),
                    }
                    if status != "approved" or not labels:
                        continue
                    approved_rows.append({
                        "image_path": str(record.get("local_path", "")),
                        "targets": [{"subject_id": "", "positive_labels": labels, "negative_labels": [], "notes": notes}],
                    })
                _save_download_label_reviews(vlso_download_review_path, review_rows)
                if not approved_rows:
                    raise FileNotFoundError("No approved labels found for retraining. Mark at least one image as approved.")
                labels_path = Path(vlso_download_labels_output)
                labels_path.parent.mkdir(parents=True, exist_ok=True)
                with labels_path.open('w', encoding='utf-8') as handle:
                    for row in approved_rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                concept_summary = VisualConceptPrototypeTrainer().train_jsonl(labels_path, vlso_download_concept_store_output).model_dump()
                operator_summary = VisualOperatorPrototypeTrainer().train_jsonl(labels_path, vlso_download_operator_store_output).model_dump()
                vision_concept_store = vlso_download_concept_store_output
                vision_operator_store = vlso_download_operator_store_output
                downloaded_record_payload = {
                    "summary": {"loaded_records": len(records), "download_root": vlso_download_root},
                    "records": records,
                }
                downloaded_review_payload = review_rows
                result_kind = "vlso_download_prepare"
                result_payload = {
                    "labels_output": vlso_download_labels_output,
                    "review_path": vlso_download_review_path,
                    "approved_examples": len(approved_rows),
                    "concept_summary": concept_summary,
                    "operator_summary": operator_summary,
                }
                flash = "Retrained concept and operator stores from approved labels only."
                flash_tone = "success"
            elif action == "run_vlso_download_pipeline":
                workspace = Path(vlso_download_pipeline_workspace)
                workspace.mkdir(parents=True, exist_ok=True)
                summary = VisualGeometryBootstrapPipeline(weights_path=vision_weights or None).run(
                    inputs=[vlso_download_root],
                    candidates_path=workspace / 'download_candidates.jsonl',
                    pseudo_labels_path=workspace / 'download_pseudo_labels.jsonl',
                    concept_store_path=workspace / 'download_concepts.db',
                    operator_store_path=workspace / 'download_operators.db',
                    eval_input=vision_eval_input if Path(vision_eval_input).exists() else None,
                    eval_mode=vision_mode,
                    answer_mode=vision_answer_mode,
                    config=PseudoLabelAcceptanceConfig(
                        cluster_similarity_threshold=float(geometry_cluster_threshold),
                        pseudo_confidence_threshold=float(geometry_pseudo_threshold),
                        cluster_consensus_threshold=float(geometry_consensus_threshold),
                        concept_match_threshold=float(geometry_concept_match_threshold),
                        min_cluster_size=int(geometry_min_cluster_size),
                    ),
                    concept_summary_output=workspace / 'download_cluster_summary.json',
                )
                result_kind = "vlso_geometry_pipeline"
                result_payload = summary.model_dump()
                vision_concept_store = summary.concept_store_path
                vision_operator_store = summary.operator_store_path
                if summary.cluster_summary_path:
                    cluster_summary_path = summary.cluster_summary_path
                    cluster_review_path = str(workspace / 'download_cluster_reviews.json')
                    review_store = VisualClusterReviewStore(cluster_review_path)
                    cluster_payload = {
                        "stats": review_store.stats(cluster_summary_path),
                        "clusters": review_store.list_clusters(cluster_summary_path),
                    }
                flash = "Downloaded-image self-training finished."
                flash_tone = "success"
            elif action == "generate_vlso_geometry_dataset":
                builder = SyntheticGeometrySceneBuilder()
                scenes = builder.build(geometry_output_dir)
                builder.write_eval_jsonl(scenes, geometry_eval_output)
                result_kind = "vlso_geometry_generate"
                result_payload = {
                    "output_dir": geometry_output_dir,
                    "eval_output": geometry_eval_output,
                    "num_scenes": len(scenes),
                    "scene_ids": [scene.scene_id for scene in scenes],
                }
                flash = "Synthetic geometry dataset generated."
                flash_tone = "success"
            elif action == "generate_cp_geometry_eval":
                summary = CpGeometryTemplateGenerator().build_eval_set(cp_geometry_eval_output)
                result_kind = "cp_geometry_generate"
                result_payload = summary.model_dump()
                flash = "CP geometry eval set generated."
                flash_tone = "success"
            elif action == "generate_cp_geometry_train_bundle":
                summary = CpGeometryTemplateGenerator().build_train_val_split(cp_geometry_train_output, cp_geometry_val_output, train_ratio=0.8)
                result_kind = "cp_geometry_generate"
                result_payload = summary.model_dump()
                flash = "CP geometry train/val bundle generated."
                flash_tone = "success"
            elif action == "run_cp_lora_experiment":
                train_inputs = _tokens(cp_lora_train_inputs)
                eval_inputs = _tokens(cp_lora_eval_inputs)
                training_run_dir = Path(cp_lora_workspace) / 'training_run'
                resume_path = _latest_checkpoint_path(str(training_run_dir)) if cp_lora_resume_latest else ''
                print(f"[GUI/CP] starting LoRA experiment workspace={cp_lora_workspace} dry_run={cp_lora_dry_run} resume={resume_path or '-'}", file=sys.stderr, flush=True)
                summary = CpLoraExperimentRunner().run(CpLoraExperimentConfig(
                    workspace=cp_lora_workspace,
                    model_name_or_path=cp_lora_model,
                    dataset_paths=train_inputs,
                    eval_dataset_paths=eval_inputs or None,
                    execute_train=True,
                    dry_run_train=cp_lora_dry_run,
                    local_files_only=False,
                    use_lora=True,
                    max_steps=int(cp_lora_max_steps or '100'),
                    save_steps=int(cp_lora_save_steps or '25'),
                    save_total_limit=int(cp_lora_save_total_limit or '2'),
                    resume_from_checkpoint=resume_path or None,
                ))
                result_kind = "cp_lora_experiment"
                result_payload = summary.model_dump()
                print(f"[GUI/CP] finished LoRA experiment workspace={cp_lora_workspace} mode={summary.training_summary.get('mode') if summary.training_summary else '-'}", file=sys.stderr, flush=True)
                flash = "CP LoRA experiment finished." if not cp_lora_dry_run else "CP LoRA dry-run finished."
                flash_tone = "success"
            elif action == "run_operator_training":
                workspace_path = Path(operator_train_workspace)
                workspace_path.mkdir(parents=True, exist_ok=True)
                print(f"[GUI/Operator] exporting teacher traces workspace={operator_train_workspace}", file=sys.stderr, flush=True)
                exporter = TeacherTraceExporter()
                traces = []
                traces.extend(exporter.export_hidden_premise_eval('examples/hidden_premise_eval.jsonl'))
                traces.extend(exporter.export_cp_parser_eval('examples/cp_parser_eval.jsonl'))
                traces.extend(exporter.export_cp_parser_eval('examples/cp_hidden_constraint_eval.jsonl'))
                traces.extend(exporter.export_vlso_eval('examples/vlso_eval.jsonl', base_dir='.'))
                real_image_eval = Path('examples/vlso_real_image_eval_gold.jsonl')
                if real_image_eval.exists():
                    traces.extend(exporter.export_vlso_eval(real_image_eval, base_dir='.'))
                transfer_eval = Path('examples/operator_transfer_eval.jsonl')
                if transfer_eval.exists():
                    traces.extend(exporter.export_operator_transfer_eval(transfer_eval))
                TeacherTraceExporter.save_jsonl(operator_train_teacher_output, traces)
                TeacherTraceExporter.save_jsonl(operator_train_teacher_sft_output, TeacherTraceExporter.to_sft_records(traces))
                export_summary = {
                    'num_traces': len(traces),
                    'teacher_trace_path': operator_train_teacher_output,
                    'teacher_sft_output': operator_train_teacher_sft_output,
                }
                print(f"[GUI/Operator] building curriculum workspace={operator_train_workspace}", file=sys.stderr, flush=True)
                bundle_summary = OperatorCurriculumBuilder().build_bundle(
                    teacher_trace_path=operator_train_teacher_output,
                    workspace=operator_train_workspace,
                    val_ratio=0.15,
                )
                training_run_dir = workspace_path / 'training_run'
                resume_path = _latest_checkpoint_path(str(training_run_dir)) if operator_train_resume_latest else ''
                print(f"[GUI/Operator] starting training workspace={operator_train_workspace} dry_run={operator_train_dry_run} resume={resume_path or '-'}", file=sys.stderr, flush=True)
                training_summary = OperatorTrainingScaffold().run(OperatorTrainConfig(
                    model_name_or_path=operator_train_model,
                    output_dir=str(training_run_dir),
                    train_jsonl=bundle_summary.train_sft_jsonl,
                    max_steps=int(operator_train_max_steps or '100'),
                    dry_run=operator_train_dry_run,
                    local_files_only=operator_train_local_files_only,
                    use_lora=operator_train_use_lora,
                    use_qlora=operator_train_use_qlora,
                    resume_from_checkpoint=resume_path or None,
                    save_steps=int(operator_train_save_steps or '25'),
                    save_total_limit=int(operator_train_save_total_limit or '2'),
                ))
                result_kind = 'operator_training'
                result_payload = {
                    'workspace': operator_train_workspace,
                    'teacher_trace_path': operator_train_teacher_output,
                    'teacher_trace_summary': export_summary,
                    'bundle_summary': bundle_summary.model_dump(),
                    'training_summary': training_summary,
                }
                print(f"[GUI/Operator] finished training workspace={operator_train_workspace} mode={training_summary.get('mode', '-')}", file=sys.stderr, flush=True)
                flash = 'Generic operator student training finished.' if not operator_train_dry_run else 'Generic operator student dry-run finished.'
                flash_tone = 'success'
            elif action == "dry_run_cp_geometry_train":
                summary = CpParserTrainingScaffold().run(CpParserTrainConfig(
                    model_name_or_path=cp_geometry_model,
                    output_dir=cp_geometry_model_output,
                    train_jsonl=cp_geometry_train_output,
                    dry_run=True,
                    use_lora=True,
                    local_files_only=False,
                ))
                result_kind = "cp_geometry_train_dry_run"
                result_payload = summary
                flash = "CP geometry training dry-run finished."
                flash_tone = "success"
            elif action == "run_vlso_geometry_pipeline":
                workspace = Path(geometry_pipeline_workspace)
                workspace.mkdir(parents=True, exist_ok=True)
                summary = VisualGeometryBootstrapPipeline(weights_path=vision_weights or None).run(
                    inputs=[geometry_pipeline_input_root],
                    candidates_path=workspace / 'geometry_candidates.jsonl',
                    pseudo_labels_path=workspace / 'geometry_pseudo_labels.jsonl',
                    concept_store_path=workspace / 'geometry_concepts.db',
                    operator_store_path=workspace / 'geometry_operators.db',
                    eval_input=geometry_eval_output if Path(geometry_eval_output).exists() else None,
                    eval_mode=vision_mode,
                    answer_mode=vision_answer_mode,
                    config=PseudoLabelAcceptanceConfig(
                        cluster_similarity_threshold=float(geometry_cluster_threshold),
                        pseudo_confidence_threshold=float(geometry_pseudo_threshold),
                        cluster_consensus_threshold=float(geometry_consensus_threshold),
                        concept_match_threshold=float(geometry_concept_match_threshold),
                        min_cluster_size=int(geometry_min_cluster_size),
                    ),
                    concept_summary_output=workspace / 'geometry_cluster_summary.json',
                )
                result_kind = "vlso_geometry_pipeline"
                result_payload = summary.model_dump()
                if summary.cluster_summary_path:
                    cluster_summary_path = summary.cluster_summary_path
                    cluster_review_path = str(workspace / 'geometry_cluster_reviews.json')
                    review_store = VisualClusterReviewStore(cluster_review_path)
                    cluster_payload = {
                        "stats": review_store.stats(cluster_summary_path),
                        "clusters": review_store.list_clusters(cluster_summary_path),
                    }
                flash = "Geometry self-training and operator learning finished."
                flash_tone = "success"
            elif action == "view_vlso_clusters":
                review_store = VisualClusterReviewStore(cluster_review_path)
                cluster_payload = {
                    "stats": review_store.stats(cluster_summary_path),
                    "clusters": review_store.list_clusters(cluster_summary_path),
                }
                result_kind = "vlso_cluster_review"
                result_payload = cluster_payload
                flash = "Loaded cluster summary."
                flash_tone = "success"
            elif action == "review_vlso_cluster":
                review_store = VisualClusterReviewStore(cluster_review_path)
                approved_labels = [item.strip() for item in _first(form, "approved_labels", "").split(",") if item.strip()]
                decision = VisualClusterReviewDecision(
                    cluster_id=_first(form, "cluster_id", ""),
                    status=_first(form, "cluster_status", "pending"),
                    note=_first(form, "cluster_note", ""),
                    approved_labels=approved_labels,
                )
                review_store.save_decision(decision)
                cluster_payload = {
                    "stats": review_store.stats(cluster_summary_path),
                    "clusters": review_store.list_clusters(cluster_summary_path),
                }
                result_kind = "vlso_cluster_review"
                result_payload = cluster_payload
                flash = f"Saved review for {decision.cluster_id}."
                flash_tone = "success"
            elif action == "retrain_vlso_from_reviews":
                workspace = Path(cluster_summary_path).resolve().parent
                summary = VisualApprovedReviewRetrainer().export_and_retrain(
                    summary_path=cluster_summary_path,
                    review_path=cluster_review_path,
                    labels_path=workspace / "approved_review_labels.jsonl",
                    concept_store_path=workspace / "approved_review_concepts.db",
                    operator_store_path=workspace / "approved_review_operators.db",
                    operator_summary_output=workspace / "approved_review_operator_summary.json",
                )
                vision_concept_store = summary.concept_store_path
                vision_operator_store = summary.operator_store_path
                result_kind = "vlso_review_retrain"
                result_payload = summary.model_dump()
                review_store = VisualClusterReviewStore(cluster_review_path)
                cluster_payload = {
                    "stats": review_store.stats(cluster_summary_path),
                    "clusters": review_store.list_clusters(cluster_summary_path),
                }
                flash = "Approved clusters were exported and retrained into concept/operator stores."
                flash_tone = "success"
        except Exception as exc:
            flash = f"Action failed: {exc}"
            flash_tone = "error"

        if cluster_payload is None:
            try:
                if Path(cluster_summary_path).exists():
                    review_store = VisualClusterReviewStore(cluster_review_path)
                    cluster_payload = {
                        "stats": review_store.stats(cluster_summary_path),
                        "clusters": review_store.list_clusters(cluster_summary_path),
                    }
            except Exception:
                cluster_payload = None

        if visual_record_payload is None:
            try:
                if Path(vlso_records_input).exists():
                    preview_records = [record.model_dump() for record in VisualDataCollector().load_records(vlso_records_input)[:24]]
                    visual_record_payload = {
                        "summary": {"loaded_records": len(preview_records), "records_path": vlso_records_input},
                        "records": preview_records,
                    }
            except Exception:
                visual_record_payload = None

        if downloaded_record_payload is None:
            try:
                local_records = _collect_local_downloaded_images(vlso_download_root)
                if local_records:
                    downloaded_record_payload = {
                        "summary": {"loaded_records": len(local_records), "download_root": vlso_download_root},
                        "records": local_records,
                    }
            except Exception:
                downloaded_record_payload = None

        if downloaded_review_payload is None:
            try:
                downloaded_review_payload = _load_download_label_reviews(vlso_download_review_path)
            except Exception:
                downloaded_review_payload = {}

        return render_page(
            ops_query=ops_query,
            ops_context=ops_context,
            ops_domain=ops_domain,
            ops_scenario=ops_scenario,
            cp_statement=cp_statement,
            vision_query=vision_query,
            vision_image=vision_image,
            vision_mode=vision_mode,
            vision_answer_mode=vision_answer_mode,
            vision_concept_store=vision_concept_store,
            vision_compare_store=vision_compare_store,
            vision_operator_store=vision_operator_store,
            vision_compare_operator_store=vision_compare_operator_store,
            vision_weights=vision_weights,
            cp_manifest=cp_manifest,
            cp_download_root=cp_download_root,
            cp_download_output=cp_download_output,
            vlso_manifest=vlso_manifest,
            vlso_collection_output=vlso_collection_output,
            vlso_family_manifest=vlso_family_manifest,
            vlso_family_names=vlso_family_names,
            vlso_family_limit=vlso_family_limit,
            vlso_family_batch_workspace=vlso_family_batch_workspace,
            vlso_family_targets=vlso_family_targets,
            vlso_dry_run=vlso_dry_run,
            vlso_records_input=vlso_records_input,
            vlso_approved_output=vlso_approved_output,
            vlso_download_manifest_output=vlso_download_manifest_output,
            vlso_download_root=vlso_download_root,
            vlso_allow_providers=vlso_allow_providers,
            vlso_allow_licenses=vlso_allow_licenses,
            vlso_accept_all=vlso_accept_all,
            vlso_execute_downloads=vlso_execute_downloads,
            visual_record_payload=visual_record_payload,
            downloaded_record_payload=downloaded_record_payload,
            downloaded_review_payload=downloaded_review_payload,
            selected_record_ids=selected_record_ids,
            cluster_summary_path=cluster_summary_path,
            cluster_review_path=cluster_review_path,
            geometry_output_dir=geometry_output_dir,
            geometry_eval_output=geometry_eval_output,
            cp_geometry_eval_output=cp_geometry_eval_output,
            cp_geometry_train_output=cp_geometry_train_output,
            cp_geometry_val_output=cp_geometry_val_output,
            cp_geometry_model=cp_geometry_model,
            cp_geometry_model_output=cp_geometry_model_output,
            cp_compare_input=cp_compare_input,
            cp_lora_workspace=cp_lora_workspace,
            cp_lora_model=cp_lora_model,
            cp_lora_train_inputs=cp_lora_train_inputs,
            cp_lora_eval_inputs=cp_lora_eval_inputs,
            cp_lora_max_steps=cp_lora_max_steps,
            cp_lora_save_steps=cp_lora_save_steps,
            cp_lora_save_total_limit=cp_lora_save_total_limit,
            cp_lora_resume_latest=cp_lora_resume_latest,
            cp_lora_dry_run=cp_lora_dry_run,
            operator_train_workspace=operator_train_workspace,
            operator_train_model=operator_train_model,
            operator_train_teacher_output=operator_train_teacher_output,
            operator_train_teacher_sft_output=operator_train_teacher_sft_output,
            operator_train_max_steps=operator_train_max_steps,
            operator_train_save_steps=operator_train_save_steps,
            operator_train_save_total_limit=operator_train_save_total_limit,
            operator_train_resume_latest=operator_train_resume_latest,
            operator_train_dry_run=operator_train_dry_run,
            operator_train_local_files_only=operator_train_local_files_only,
            operator_train_use_lora=operator_train_use_lora,
            operator_train_use_qlora=operator_train_use_qlora,
            vision_eval_input=vision_eval_input,
            understanding_hidden_input=understanding_hidden_input,
            understanding_cp_input=understanding_cp_input,
            understanding_cp_hidden_input=understanding_cp_hidden_input,
            understanding_vlso_input=understanding_vlso_input,
            understanding_vlso_real_input=understanding_vlso_real_input,
            geometry_pipeline_input_root=geometry_pipeline_input_root,
            geometry_pipeline_workspace=geometry_pipeline_workspace,
            geometry_cluster_threshold=geometry_cluster_threshold,
            geometry_pseudo_threshold=geometry_pseudo_threshold,
            geometry_consensus_threshold=geometry_consensus_threshold,
            geometry_concept_match_threshold=geometry_concept_match_threshold,
            geometry_min_cluster_size=geometry_min_cluster_size,
            vlso_download_pipeline_workspace=vlso_download_pipeline_workspace,
            vlso_download_labels_output=vlso_download_labels_output,
            vlso_download_concept_store_output=vlso_download_concept_store_output,
            vlso_download_operator_store_output=vlso_download_operator_store_output,
            vlso_download_review_path=vlso_download_review_path,
            vlso_download_review_filter=vlso_download_review_filter,
            flash=flash,
            flash_tone=flash_tone,
            result_kind=result_kind,
            result_payload=result_payload,
            cluster_payload=cluster_payload,
        )


def render_page(**ctx: object) -> str:
    flash = ctx["flash"]
    flash_tone = ctx["flash_tone"]
    flash_html = f"<div class='flash {html.escape(str(flash_tone))}'>{html.escape(str(flash))}</div>" if flash else ""
    result_html = render_result(str(ctx["result_kind"]), ctx["result_payload"] if isinstance(ctx["result_payload"], dict) else None)
    cluster_review_html = render_cluster_review(
        str(ctx["cluster_summary_path"]),
        str(ctx["cluster_review_path"]),
        ctx["cluster_payload"] if isinstance(ctx["cluster_payload"], dict) else None,
    )
    return f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SemOp Easy GUI</title>
<style>
:root {{ --bg:#f5f1e8; --card:#fffdf9; --line:#d7d8de; --ink:#17212b; --muted:#5f6975; --accent:#175c87; --ok:#206343; --warn:#8b3d1f; --soft:#edf4f8; --shadow:0 16px 40px rgba(20,32,44,.08); }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:linear-gradient(180deg,#f7f2e9 0%,#eef4f8 100%); color:var(--ink); font-family:"Segoe UI","Malgun Gothic",sans-serif; }}
main {{ max-width:1380px; margin:0 auto; padding:24px 18px 48px; }}
h1 {{ margin:0 0 8px; font-size:34px; letter-spacing:-.03em; }}
h2 {{ margin:0 0 10px; font-size:20px; }}
h3 {{ margin:0 0 10px; font-size:16px; }}
p,li,label,small {{ color:var(--muted); line-height:1.6; }}
.hero,.grid,.mini-grid,.cluster-grid {{ display:grid; gap:18px; }}
.hero {{ grid-template-columns:1.3fr .9fr; margin-bottom:18px; }}
.grid {{ grid-template-columns:1fr 1fr; }}
.mini-grid {{ grid-template-columns:1fr 1fr; }}
.cluster-grid {{ grid-template-columns:1fr 1fr; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:24px; padding:20px; box-shadow:var(--shadow); }}
.cluster-card {{ background:#fbfbff; border:1px solid var(--line); border-radius:18px; padding:14px; }}
.cluster-top {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px; color:var(--muted); }}
.cluster-form {{ margin-top:10px; }}
.cluster-stats {{ margin-bottom:14px; color:var(--muted); }}
textarea,input,select {{ width:100%; padding:12px 14px; border:1px solid var(--line); border-radius:16px; font:inherit; background:#fff; }}
textarea {{ min-height:110px; resize:vertical; }}
button {{ border:0; border-radius:16px; padding:12px 14px; font:inherit; cursor:pointer; }}
button.primary {{ background:var(--accent); color:#fff; font-weight:700; }}
button.secondary {{ background:var(--soft); color:var(--ink); font-weight:600; }}
.actions {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }}
.flash {{ margin-bottom:14px; padding:12px 14px; border-radius:16px; background:#eef5f8; color:var(--accent); }}
.flash.success {{ background:#ecf8f0; color:var(--ok); }}
.flash.error {{ background:#fff1ec; color:var(--warn); }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; }}
.chip {{ background:#edf4f8; color:var(--accent); border-radius:999px; padding:6px 10px; font-size:12px; }}
.summary-grid {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); margin-bottom:14px; }}
.metric-card {{ background:#f6f8fb; border:1px solid var(--line); border-radius:16px; padding:12px; }}
.metric-card small {{ display:block; margin-bottom:6px; }}
.metric-card strong {{ display:block; font-size:16px; color:var(--ink); line-height:1.4; word-break:break-word; }}
.info-block {{ background:#fbfcfe; border:1px solid var(--line); border-radius:16px; padding:12px 14px; margin-bottom:12px; }}
.info-block small {{ display:block; margin-bottom:6px; }}
.raw-json summary {{ cursor:pointer; color:var(--accent); font-weight:600; margin-bottom:10px; }}
.record-grid {{ display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); margin-top:14px; }}
.record-card {{ display:block; background:#fbfcfe; border:1px solid var(--line); border-radius:18px; padding:10px; color:var(--ink); }}
.record-card input {{ width:auto; margin-bottom:8px; }}
.record-thumb-wrap {{ aspect-ratio:1 / 1; overflow:hidden; border-radius:14px; background:#eef2f6; display:flex; align-items:center; justify-content:center; }}
.record-thumb {{ width:100%; height:100%; object-fit:cover; display:block; }}
.record-meta {{ margin-top:10px; display:grid; gap:4px; }}
.record-meta strong {{ font-size:14px; line-height:1.4; }}
pre {{ background:#14202a; color:#eef6fb; border-radius:18px; padding:16px; overflow:auto; white-space:pre-wrap; }}
.empty {{ border:1px dashed var(--line); border-radius:16px; padding:20px; color:var(--muted); }}
.inline {{ display:flex; align-items:center; gap:8px; }}
@media (max-width:1100px) {{ .hero,.grid,.mini-grid,.cluster-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<main>
  <section class="hero">
    <div class="card">
      <h1>SemOp Easy GUI</h1>
      <p>This is a one-page starter GUI. You can test ops reasoning, competitive programming analysis, image-grounded QA, concept-store comparison, cluster review, and labeled dataset download without memorizing commands.</p>
      <div class="chips">
        <div class="chip">ops reasoning</div>
        <div class="chip">CP analysis</div>
        <div class="chip">vision QA</div>
        <div class="chip">concept store compare</div>
        <div class="chip">cluster review</div>
        <div class="chip">dataset download</div>
      </div>
    </div>
    <div class="card">
      <h2>Fastest path</h2>
      <ol>
        <li>Click a load-example button.</li>
        <li>Click run.</li>
        <li>Check the JSON result at the bottom.</li>
        <li>For self-training, load cluster summary and approve or reject clusters.</li>
      </ol>
      <small>Advanced GUIs still exist: <code>ops_copilot_gui.py</code>, <code>cp_copilot_gui.py</code>.</small>
    </div>
  </section>
  {flash_html}
  <section class="grid">
    <form method="post" class="card">
      <h2>1. Ops / SOP question</h2>
      <label>Question</label>
      <textarea name="ops_query">{html.escape(str(ctx['ops_query']))}</textarea>
      <label>Context / SOP</label>
      <textarea name="ops_context">{html.escape(str(ctx['ops_context']))}</textarea>
      <div class="mini-grid">
        <div><label>Domain</label><input name="ops_domain" value="{html.escape(str(ctx['ops_domain']))}"></div>
        <div><label>Scenario</label><input name="ops_scenario" value="{html.escape(str(ctx['ops_scenario']))}"></div>
      </div>
      <div class="actions">
        <button class="secondary" name="action" value="load_ops_example">Load example</button>
        <button class="primary" name="action" value="run_ops">Run</button>
      </div>
    </form>

    <form method="post" class="card">
      <h2>2. Competitive programming</h2>
      <label>Problem statement</label>
      <textarea name="cp_statement">{html.escape(str(ctx['cp_statement']))}</textarea>
      <div class="actions">
        <button class="secondary" name="action" value="load_cp_example">Load example</button>
        <button class="primary" name="action" value="run_cp">Analyze</button>
      </div>
      <small>You get algorithm family, DSL, complexity, C++17 code, and validator output.</small>
    </form>

    <form method="post" class="card">
      <h2>3. Image question</h2>
      <label>Question</label>
      <input name="vision_query" value="{html.escape(str(ctx['vision_query']))}">
      <label>Image path</label>
      <input name="vision_image" value="{html.escape(str(ctx['vision_image']))}">
      <div class="mini-grid">
        <div>
          <label>Reasoning mode</label>
          <select name="vision_mode">
            <option value="deep"{' selected' if str(ctx['vision_mode']) == 'deep' else ''}>deep</option>
            <option value="hybrid"{' selected' if str(ctx['vision_mode']) == 'hybrid' else ''}>hybrid</option>
            <option value="heuristic"{' selected' if str(ctx['vision_mode']) == 'heuristic' else ''}>heuristic</option>
          </select>
        </div>
        <div>
          <label>Answer mode</label>
          <select name="vision_answer_mode">
            <option value="structured"{' selected' if str(ctx['vision_answer_mode']) == 'structured' else ''}>structured</option>
            <option value="llm"{' selected' if str(ctx['vision_answer_mode']) == 'llm' else ''}>llm</option>
          </select>
        </div>
      </div>
      <label>Primary concept DB</label>
      <input name="vision_concept_store" value="{html.escape(str(ctx['vision_concept_store']))}">
      <label>Compare concept DB</label>
      <input name="vision_compare_store" value="{html.escape(str(ctx['vision_compare_store']))}">
      <label>Primary operator DB</label>
      <input name="vision_operator_store" value="{html.escape(str(ctx['vision_operator_store']))}">
      <label>Compare operator DB</label>
      <input name="vision_compare_operator_store" value="{html.escape(str(ctx['vision_compare_operator_store']))}">
      <label>Affordance weights JSON</label>
      <input name="vision_weights" value="{html.escape(str(ctx['vision_weights']))}">
      <div class="actions">
        <button class="secondary" name="action" value="load_vision_example">Load example</button>
        <button class="primary" name="action" value="run_vision">Run image QA</button>
        <button class="secondary" name="action" value="run_vision_compare">Compare stores</button>
      </div>
    </form>

    <div class="card">
      <h2>4. Labeled data tools</h2>
      <form method="post">
        <h3>CP labeled dataset</h3>
        <label>Manifest</label>
        <input name="cp_manifest" value="{html.escape(str(ctx['cp_manifest']))}">
        <label>Download root</label>
        <input name="cp_download_root" value="{html.escape(str(ctx['cp_download_root']))}">
        <label>Normalized output</label>
        <input name="cp_download_output" value="{html.escape(str(ctx['cp_download_output']))}">
        <div class="actions"><button class="primary" name="action" value="download_cp_labels">Download labeled CP data</button></div>
      </form>
      <hr style="border:none;border-top:1px solid var(--line);margin:18px 0;">
      <form method="post">
        <h3>VLSO public-image collection</h3>
        <label>Object family preset</label>
        <input name="vlso_family_names" value="{html.escape(str(ctx['vlso_family_names']))}">
        <div class="mini-grid">
          <div><label>Family manifest path</label><input name="vlso_family_manifest" value="{html.escape(str(ctx['vlso_family_manifest']))}"></div>
          <div><label>Limit per source</label><input name="vlso_family_limit" value="{html.escape(str(ctx['vlso_family_limit']))}"></div>
        </div>
        <div class="actions"><button class="secondary" name="action" value="build_vlso_family_manifest">Build family manifest</button></div>
        <label>Family targets</label>
        <input name="vlso_family_targets" value="{html.escape(str(ctx['vlso_family_targets']))}">
        <label>Batch workspace</label>
        <input name="vlso_family_batch_workspace" value="{html.escape(str(ctx['vlso_family_batch_workspace']))}">
        <div class="actions"><button class="primary" name="action" value="run_vlso_family_batch">Run VLSO family batch</button></div>
        <label>Manifest</label>
        <input name="vlso_manifest" value="{html.escape(str(ctx['vlso_manifest']))}">
        <label>Plan or records output JSONL</label>
        <input name="vlso_collection_output" value="{html.escape(str(ctx['vlso_collection_output']))}">
        <input type="hidden" name="vlso_dry_run" value="0">
        <label class="inline"><input type="checkbox" name="vlso_dry_run" value="1"{_checked(bool(ctx['vlso_dry_run']))}> dry run only (plan only, no API fetch)</label>
        <div class="actions"><button class="primary" name="action" value="plan_vlso_collection">Build VLSO collection plan</button></div>
        <small>Recommended families: bag, box, drawer, door, bottle, tool, cabinet, suitcase, jar, bin, pouch. The batch collector auto-builds a manifest, records file, approved file, and download manifest inside the workspace.</small>
      </form>
      <hr style="border:none;border-top:1px solid var(--line);margin:18px 0;">
      <form method="post">
        <h3>VLSO download preparation</h3>
        <label>Collected records JSONL</label>
        <input name="vlso_records_input" value="{html.escape(str(ctx['vlso_records_input']))}">
        <label>Approved records JSONL</label>
        <input name="vlso_approved_output" value="{html.escape(str(ctx['vlso_approved_output']))}">
        <label>Download manifest JSONL</label>
        <input name="vlso_download_manifest_output" value="{html.escape(str(ctx['vlso_download_manifest_output']))}">
        <label>Download root folder</label>
        <input name="vlso_download_root" value="{html.escape(str(ctx['vlso_download_root']))}">
        <label>Allowed providers</label>
        <input name="vlso_allow_providers" value="{html.escape(str(ctx['vlso_allow_providers']))}">
        <label>Allowed licenses</label>
        <input name="vlso_allow_licenses" value="{html.escape(str(ctx['vlso_allow_licenses']))}">
        <input type="hidden" name="vlso_accept_all" value="0">
        <label class="inline"><input type="checkbox" name="vlso_accept_all" value="1"{_checked(bool(ctx['vlso_accept_all']))}> accept all records with media URLs</label>
        <input type="hidden" name="vlso_execute_downloads" value="0">
        <label class="inline"><input type="checkbox" name="vlso_execute_downloads" value="1"{_checked(bool(ctx['vlso_execute_downloads']))}> execute downloads now</label>
        <div class="actions"><button class="secondary" name="action" value="load_vlso_records_preview">Load image preview cards</button><button class="primary" name="action" value="prepare_vlso_downloads">Prepare VLSO downloads</button></div>
      </form>
      <hr style="border:none;border-top:1px solid var(--line);margin:18px 0;">
      <form method="post">
        <h3>Downloaded-image self-training</h3>
        <label>Learning workspace</label>
        <input name="vlso_download_pipeline_workspace" value="{html.escape(str(ctx['vlso_download_pipeline_workspace']))}">
        <div class="actions"><button class="primary" name="action" value="run_vlso_download_pipeline">Run learning on downloaded images</button></div>
        <small>This uses the current download root as input and builds candidates, pseudo labels, concept DB, operator DB, and cluster review files.</small>
      </form>
      <hr style="border:none;border-top:1px solid var(--line);margin:18px 0;">
      <form method="post">
        <h3>Manual labels for downloaded images</h3>
        <label>Labels JSONL output</label>
        <input name="vlso_download_labels_output" value="{html.escape(str(ctx['vlso_download_labels_output']))}">
        <label>Concept DB output</label>
        <input name="vlso_download_concept_store_output" value="{html.escape(str(ctx['vlso_download_concept_store_output']))}">
        <label>Operator DB output</label>
        <input name="vlso_download_operator_store_output" value="{html.escape(str(ctx['vlso_download_operator_store_output']))}">
        <label>Review JSON output</label>
        <input name="vlso_download_review_path" value="{html.escape(str(ctx['vlso_download_review_path']))}">
        <label>Default review filter</label>
        <select name="vlso_download_review_filter">
          <option value="all"{' selected' if str(ctx['vlso_download_review_filter']) == 'all' else ''}>all</option>
          <option value="approved"{' selected' if str(ctx['vlso_download_review_filter']) == 'approved' else ''}>approved</option>
          <option value="pending"{' selected' if str(ctx['vlso_download_review_filter']) == 'pending' else ''}>pending</option>
          <option value="rejected"{' selected' if str(ctx['vlso_download_review_filter']) == 'rejected' else ''}>rejected</option>
        </select>
        <div class="actions"><button class="secondary" name="action" value="load_vlso_downloaded_preview">Load downloaded image cards</button></div>
      </form>
    </div>

    <div class="card">
      <h2>5. Geometry starter tools</h2>
      <form method="post">
        <h3>Synthetic VLSO geometry data</h3>
        <label>Output directory</label>
        <input name="geometry_output_dir" value="{html.escape(str(ctx['geometry_output_dir']))}">
        <label>Eval JSONL</label>
        <input name="geometry_eval_output" value="{html.escape(str(ctx['geometry_eval_output']))}">
        <div class="actions"><button class="primary" name="action" value="generate_vlso_geometry_dataset">Generate geometry scenes</button></div>
      </form>
      <hr style="border:none;border-top:1px solid var(--line);margin:18px 0;">
      <form method="post">
        <h3>CP geometry eval assets</h3>
        <label>Eval JSONL</label>
        <input name="cp_geometry_eval_output" value="{html.escape(str(ctx['cp_geometry_eval_output']))}">
        <div class="actions"><button class="secondary" name="action" value="generate_cp_geometry_eval">Generate geometry eval set</button></div>
        <label>Train JSONL</label>
        <input name="cp_geometry_train_output" value="{html.escape(str(ctx['cp_geometry_train_output']))}">
        <label>Validation JSONL</label>
        <input name="cp_geometry_val_output" value="{html.escape(str(ctx['cp_geometry_val_output']))}">
        <div class="actions"><button class="secondary" name="action" value="generate_cp_geometry_train_bundle">Generate train/val bundle</button></div>
      </form>
      <hr style="border:none;border-top:1px solid var(--line);margin:18px 0;">
      <form method="post">
        <h3>CP geometry training dry-run</h3>
        <label>Model id or path</label>
        <input name="cp_geometry_model" value="{html.escape(str(ctx['cp_geometry_model']))}">
        <label>Dry-run output directory</label>
        <input name="cp_geometry_model_output" value="{html.escape(str(ctx['cp_geometry_model_output']))}">
        <div class="actions"><button class="primary" name="action" value="dry_run_cp_geometry_train">Build training plan</button></div>
      </form>
      <hr style="border:none;border-top:1px solid var(--line);margin:18px 0;">
      <form method="post">
        <h3>CP LoRA train/resume</h3>
        <label>Model id or path</label>
        <input name="cp_lora_model" value="{html.escape(str(ctx['cp_lora_model']))}">
        <label>Workspace</label>
        <input name="cp_lora_workspace" value="{html.escape(str(ctx['cp_lora_workspace']))}">
        <label>Train inputs</label>
        <input name="cp_lora_train_inputs" value="{html.escape(str(ctx['cp_lora_train_inputs']))}">
        <label>Eval inputs</label>
        <input name="cp_lora_eval_inputs" value="{html.escape(str(ctx['cp_lora_eval_inputs']))}">
        <div class="mini-grid">
          <div><label>Max steps</label><input name="cp_lora_max_steps" value="{html.escape(str(ctx['cp_lora_max_steps']))}"></div>
          <div><label>Save steps</label><input name="cp_lora_save_steps" value="{html.escape(str(ctx['cp_lora_save_steps']))}"></div>
          <div><label>Save total limit</label><input name="cp_lora_save_total_limit" value="{html.escape(str(ctx['cp_lora_save_total_limit']))}"></div>
        </div>
        <input type="hidden" name="cp_lora_resume_latest" value="0">
        <label class="inline"><input type="checkbox" name="cp_lora_resume_latest" value="1"{_checked(bool(ctx['cp_lora_resume_latest']))}> resume latest checkpoint if present</label>
        <input type="hidden" name="cp_lora_dry_run" value="0">
        <label class="inline"><input type="checkbox" name="cp_lora_dry_run" value="1"{_checked(bool(ctx['cp_lora_dry_run']))}> dry-run only</label>
        <div class="actions"><button class="primary" name="action" value="run_cp_lora_experiment">Run CP LoRA experiment</button></div>
      </form>
      <hr style="border:none;border-top:1px solid var(--line);margin:18px 0;">
      <form method="post">
        <h3>Generic operator student train/resume</h3>
        <label>Model id or path</label>
        <input name="operator_train_model" value="{html.escape(str(ctx['operator_train_model']))}">
        <label>Workspace</label>
        <input name="operator_train_workspace" value="{html.escape(str(ctx['operator_train_workspace']))}">
        <label>Teacher traces JSONL</label>
        <input name="operator_train_teacher_output" value="{html.escape(str(ctx['operator_train_teacher_output']))}">
        <label>Teacher SFT JSONL</label>
        <input name="operator_train_teacher_sft_output" value="{html.escape(str(ctx['operator_train_teacher_sft_output']))}">
        <div class="mini-grid">
          <div><label>Max steps</label><input name="operator_train_max_steps" value="{html.escape(str(ctx['operator_train_max_steps']))}"></div>
          <div><label>Save steps</label><input name="operator_train_save_steps" value="{html.escape(str(ctx['operator_train_save_steps']))}"></div>
          <div><label>Save total limit</label><input name="operator_train_save_total_limit" value="{html.escape(str(ctx['operator_train_save_total_limit']))}"></div>
        </div>
        <input type="hidden" name="operator_train_resume_latest" value="0">
        <label class="inline"><input type="checkbox" name="operator_train_resume_latest" value="1"{_checked(bool(ctx['operator_train_resume_latest']))}> resume latest checkpoint if present</label>
        <input type="hidden" name="operator_train_dry_run" value="0">
        <label class="inline"><input type="checkbox" name="operator_train_dry_run" value="1"{_checked(bool(ctx['operator_train_dry_run']))}> dry-run only</label>
        <input type="hidden" name="operator_train_local_files_only" value="0">
        <label class="inline"><input type="checkbox" name="operator_train_local_files_only" value="1"{_checked(bool(ctx['operator_train_local_files_only']))}> local-files-only</label>
        <input type="hidden" name="operator_train_use_lora" value="0">
        <label class="inline"><input type="checkbox" name="operator_train_use_lora" value="1"{_checked(bool(ctx['operator_train_use_lora']))}> use LoRA</label>
        <input type="hidden" name="operator_train_use_qlora" value="0">
        <label class="inline"><input type="checkbox" name="operator_train_use_qlora" value="1"{_checked(bool(ctx['operator_train_use_qlora']))}> use QLoRA (requires bitsandbytes)</label>
        <div class="actions"><button class="primary" name="action" value="run_operator_training">Run generic operator training</button></div>
      </form>
      <hr style="border:none;border-top:1px solid var(--line);margin:18px 0;">
      <form method="post">
        <h3>Evaluation shortcuts</h3>
        <label>VLSO eval JSONL</label>
        <input name="vision_eval_input" value="{html.escape(str(ctx['vision_eval_input']))}">
        <div class="actions"><button class="secondary" name="action" value="evaluate_vlso_review_impact">Compare approved VLSO stores</button></div>
        <label>CP parser eval JSONL</label>
        <input name="cp_compare_input" value="{html.escape(str(ctx['cp_compare_input']))}">
        <div class="actions"><button class="secondary" name="action" value="compare_cp_parsers">Compare CP heuristic vs model</button></div>
        <label>Hidden premise eval JSONL</label>
        <input name="understanding_hidden_input" value="{html.escape(str(ctx['understanding_hidden_input']))}">
        <label>CP hidden-constraint eval JSONL</label>
        <input name="understanding_cp_hidden_input" value="{html.escape(str(ctx['understanding_cp_hidden_input']))}">
        <label>VLSO real-image eval JSONL</label>
        <input name="understanding_vlso_real_input" value="{html.escape(str(ctx['understanding_vlso_real_input']))}">
        <div class="mini-grid">
          <div><label>CP understanding eval JSONL</label><input name="understanding_cp_input" value="{html.escape(str(ctx['understanding_cp_input']))}"></div>
          <div><label>VLSO understanding eval JSONL</label><input name="understanding_vlso_input" value="{html.escape(str(ctx['understanding_vlso_input']))}"></div>
        </div>
        <div class="actions"><button class="primary" name="action" value="run_understanding_eval">Run overall understanding benchmark</button></div>
      </form>
      <hr style="border:none;border-top:1px solid var(--line);margin:18px 0;">
      <form method="post">
        <h3>One-click VLSO geometry self-training</h3>
        <label>Geometry input folder or file</label>
        <input name="geometry_pipeline_input_root" value="{html.escape(str(ctx['geometry_pipeline_input_root']))}">
        <label>Workspace</label>
        <input name="geometry_pipeline_workspace" value="{html.escape(str(ctx['geometry_pipeline_workspace']))}">
        <div class="mini-grid">
          <div><label>Cluster similarity</label><input name="geometry_cluster_threshold" value="{html.escape(str(ctx['geometry_cluster_threshold']))}"></div>
          <div><label>Pseudo threshold</label><input name="geometry_pseudo_threshold" value="{html.escape(str(ctx['geometry_pseudo_threshold']))}"></div>
          <div><label>Consensus threshold</label><input name="geometry_consensus_threshold" value="{html.escape(str(ctx['geometry_consensus_threshold']))}"></div>
          <div><label>Concept match threshold</label><input name="geometry_concept_match_threshold" value="{html.escape(str(ctx['geometry_concept_match_threshold']))}"></div>
        </div>
        <label>Min cluster size</label>
        <input name="geometry_min_cluster_size" value="{html.escape(str(ctx['geometry_min_cluster_size']))}">
        <div class="actions"><button class="primary" name="action" value="run_vlso_geometry_pipeline">Run geometry self-training</button></div>
        <small>This runs candidate extraction, pseudo-labeling, concept self-training, operator learning, and grounded eval in one step.</small>
      </form>
    </div>
  </section>
  <section class="card" style="margin-top:18px;">
    <h2>5. VLSO image preview and approval</h2>
    <div style="margin-top:8px;">{render_visual_record_preview(str(ctx['vlso_records_input']), ctx['visual_record_payload'] if isinstance(ctx['visual_record_payload'], dict) else None, list(ctx['selected_record_ids']) if isinstance(ctx['selected_record_ids'], list) else [], str(ctx['vlso_allow_providers']), str(ctx['vlso_allow_licenses']), str(ctx['vlso_approved_output']), str(ctx['vlso_download_manifest_output']), str(ctx['vlso_download_root']), bool(ctx['vlso_accept_all']), bool(ctx['vlso_execute_downloads']))}</div>
  </section>
  <section class="card" style="margin-top:18px;">
    <h2>6. Downloaded image labeling</h2>
    <div style="margin-top:8px;">{render_downloaded_label_editor(
      str(ctx['vlso_download_root']),
      ctx['downloaded_record_payload'] if isinstance(ctx['downloaded_record_payload'], dict) else None,
      str(ctx['vlso_download_labels_output']),
      str(ctx['vlso_download_concept_store_output']),
      str(ctx['vlso_download_operator_store_output']),
      str(ctx['vlso_download_review_path']),
      ctx['downloaded_review_payload'] if isinstance(ctx['downloaded_review_payload'], dict) else None,
      str(ctx['vlso_download_review_filter']),
    )}</div>
  </section>
  <section class="card" style="margin-top:18px;">
    <h2>7. Self-training cluster review</h2>
    <form method="post">
      <div class="mini-grid">
        <div>
          <label>Cluster summary JSON</label>
          <input name="cluster_summary_path" value="{html.escape(str(ctx['cluster_summary_path']))}">
        </div>
        <div>
          <label>Review decisions JSON</label>
          <input name="cluster_review_path" value="{html.escape(str(ctx['cluster_review_path']))}">
        </div>
      </div>
      <div class="actions"><button class="primary" name="action" value="view_vlso_clusters">Load cluster summary</button><button class="secondary" name="action" value="retrain_vlso_from_reviews">Retrain from approved clusters</button></div>
    </form>
    <div style="margin-top:16px;">{cluster_review_html}</div>
  </section>
  <section class="card" style="margin-top:18px;">
    <h2>Result</h2>
    {result_html}
  </section>
</main>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    app: StarterApp

    def do_GET(self) -> None:
        self._send(self.app.handle({}))

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(body, keep_blank_values=True)
        self._send(self.app.handle(form))

    def _send(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Easy starter GUI for SemOp")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()

    app = StarterApp()
    handler = type("SemOpEasyHandler", (_Handler,), {"app": app})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"SemOp Easy GUI running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

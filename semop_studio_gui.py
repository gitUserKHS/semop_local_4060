from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from semop import (
    AnalogyEvalCase,
    BenchmarkGatedContinuousTrainer,
    CompilerRepairEvalCase,
    CorpusMemoryStore,
    CopilotRequest,
    DomainCopilot,
    GeneralizationProofHarness,
    GroundedExplanationEvalCase,
    HiddenPremiseEvalCase,
    OperatorTransferEvalCase,
    ReviewQueueStore,
    SemOpUnderstandingEvaluator,
    StructuredMeaningPipeline,
    UnifiedSemOpTrainer,
    VLSOReasoner,
    VlsoGroundedEvaluator,
    collect_review_promotion_decisions,
    infer_review_severity,
    review_reasons_from_graph_and_kpis,
)
from semop.structures import StructuredMeaningGraph


OPS_EXAMPLE = {
    'query': 'The aisle is blocked and approval is still missing. What should I do?',
    'context': 'If the work zone is blocked or the task is not approved yet, stop first and check the alternate route and manager approval.',
    'domain': 'warehouse_exception',
    'scenario': 'exception_response',
}
VISION_EXAMPLE = {
    'query': 'What objects or openings are visible here?',
    'image_path': 'data/scene.png',
}
UNIFIED_EXAMPLE = {
    'store': 'data/semop_memory.db',
    'output_dir': 'data/unified_semop_gui_run',
    'review_queue': 'data/ops_review_queue.db',
    'source': '',
    'split': 'train',
    'operating_domain': OPS_EXAMPLE['domain'],
    'baseline_summary': '',
    'benchmark_corpus': 'data/unified_semop_gui_run/persistent_benchmark_corpus.json',
    'transfer_input': 'examples/operator_transfer_eval.jsonl',
}
UNDERSTANDING_EXAMPLE = {
    'hidden': 'examples/hidden_premise_eval.jsonl',
    'vlso': 'examples/vlso_eval.jsonl',
    'vlso_real': 'examples/vlso_real_image_eval_gold.jsonl',
}
VISION_STORE_EXAMPLE = {
    'concept_store': 'data/vlso_visual_prototypes.db',
    'operator_store': 'data/vlso_visual_operators.db',
    'weights': 'data/vlso_samples/trained_affordance_weights.json',
    'review_path': 'data/vlso_download_pipeline_gui/manual_label_reviews.json',
}
GUIDED_BOOTSTRAP_SOURCE = 'studio_bootstrap'
BEGINNER_AUTOPILOT_SOURCE = 'studio_beginner_autopilot'
AUTOPILOT_PROOF_ROUNDS = 3
PROOF_CURRICULUM_ROUNDS = (
    {
        'domain': 'warehouse_exception',
        'scenario': 'exception_response',
        'context': 'If the lane is blocked, access is uncertain, or approval is missing, stop first and verify the safe alternative path and approval before moving.',
        'queries': (
            'The lane is partly blocked and the approval token is missing. What do I verify before continuing?',
            'An access door is closed and the route is narrow. Do I force it open or confirm the safe path first?',
            'The item is urgent but the zone is blocked. What checks come before movement?',
        ),
        'visual_queries': (
            'What objects or openings are visible here?',
            'Which openings or reachable objects should be checked first here?',
        ),
    },
    {
        'domain': 'warehouse_onboarding',
        'scenario': 'guided_walkthrough',
        'context': 'For onboarding, confirm the item identity, access path, and allowed opening before touching storage or moving through a boundary.',
        'queries': (
            'I am new to this aisle. What should I confirm before opening the cabinet and reaching for the item?',
            'Before I take the tote from the shelf, what access and identity checks come first?',
            'The drawer looks available, but I am not sure the item is correct. What do I verify first?',
        ),
        'visual_queries': (
            'What access points or containers are visible in this scene?',
            'Which container or opening in this scene looks reachable?',
        ),
    },
    {
        'domain': 'general',
        'scenario': 'access_reasoning',
        'context': 'When access is uncertain, identify the target, opening state, and safety constraints before forcing movement or reaching into a container.',
        'queries': (
            'The pouch might contain the tool, but I cannot tell if it is safely accessible. What should I check first?',
            'Before I open the box and pull something out, what hidden requirements should I verify?',
            'I can see a door and a container. Which access constraints matter before acting?',
        ),
        'visual_queries': (
            'What openings, containers, or reachable objects are visible here?',
            'Which visible object here looks like an access path or container?',
        ),
    },
)
GUIDED_BOOTSTRAP_VARIANTS = (
    'The route is blocked and approval has not arrived yet. What should I verify before moving?',
    'Manager approval is still pending. Do I continue the task or stop first?',
    'The work zone is blocked. Should I reroute now or confirm approval before continuing?',
)
SPLIT_OPTIONS = ('train', 'val', 'test')
VISION_MODE_OPTIONS = ('deep', 'hybrid', 'heuristic')
VISION_ANSWER_MODE_OPTIONS = ('structured', 'llm')
ACTION_TIME_HINTS = {
    'run_unified_training': 'about 10-30 seconds',
    'run_unified_benchmark_gate': 'about 20-60 seconds',
    'run_guided_learning': 'about 20-60 seconds',
    'run_beginner_autopilot': 'about 30-90 seconds',
    'run_beginner_test': 'about 15-45 seconds',
    'run_generalization_proof': 'about 45-120 seconds',
    'run_autopilot_coach': 'about 1-3 minutes',
    'run_understanding_eval': 'about 10-25 seconds',
}
ACTION_TIME_SECONDS = {
    'run_unified_training': 25,
    'run_unified_benchmark_gate': 45,
    'run_guided_learning': 50,
    'run_beginner_autopilot': 75,
    'run_beginner_test': 35,
    'run_generalization_proof': 90,
    'run_autopilot_coach': 150,
    'run_understanding_eval': 20,
}


@dataclass(frozen=True)
class StudioState:
    ops_query: str = OPS_EXAMPLE['query']
    ops_context: str = OPS_EXAMPLE['context']
    ops_domain: str = OPS_EXAMPLE['domain']
    ops_scenario: str = OPS_EXAMPLE['scenario']
    vision_query: str = VISION_EXAMPLE['query']
    vision_image: str = VISION_EXAMPLE['image_path']
    vision_mode: str = 'deep'
    vision_answer_mode: str = 'structured'
    vision_concept_store: str = VISION_STORE_EXAMPLE['concept_store']
    vision_operator_store: str = VISION_STORE_EXAMPLE['operator_store']
    vision_weights: str = VISION_STORE_EXAMPLE['weights']
    vision_review_path: str = VISION_STORE_EXAMPLE['review_path']
    unified_store_path: str = UNIFIED_EXAMPLE['store']
    unified_output_dir: str = UNIFIED_EXAMPLE['output_dir']
    unified_review_queue_path: str = UNIFIED_EXAMPLE['review_queue']
    unified_source: str = UNIFIED_EXAMPLE['source']
    unified_split: str = UNIFIED_EXAMPLE['split']
    unified_operating_domain: str = UNIFIED_EXAMPLE['operating_domain']
    unified_baseline_summary_path: str = UNIFIED_EXAMPLE['baseline_summary']
    unified_benchmark_corpus_path: str = UNIFIED_EXAMPLE['benchmark_corpus']
    unified_transfer_input: str = UNIFIED_EXAMPLE['transfer_input']
    unified_approved_queries_only: bool = False
    understanding_hidden_input: str = UNDERSTANDING_EXAMPLE['hidden']
    understanding_vlso_input: str = UNDERSTANDING_EXAMPLE['vlso']
    understanding_vlso_real_input: str = UNDERSTANDING_EXAMPLE['vlso_real']
    compare_left_path: str = ''
    compare_right_path: str = ''

    @classmethod
    def from_form(cls, form: dict[str, list[str]]) -> 'StudioState':
        return cls(
            ops_query=_first(form, 'ops_query', cls.ops_query),
            ops_context=_first(form, 'ops_context', cls.ops_context),
            ops_domain=_first(form, 'ops_domain', cls.ops_domain),
            ops_scenario=_first(form, 'ops_scenario', cls.ops_scenario),
            vision_query=_first(form, 'vision_query', cls.vision_query),
            vision_image=_first(form, 'vision_image', cls.vision_image),
            vision_mode=_first(form, 'vision_mode', cls.vision_mode),
            vision_answer_mode=_first(form, 'vision_answer_mode', cls.vision_answer_mode),
            vision_concept_store=_first(form, 'vision_concept_store', cls.vision_concept_store),
            vision_operator_store=_first(form, 'vision_operator_store', cls.vision_operator_store),
            vision_weights=_first(form, 'vision_weights', cls.vision_weights),
            vision_review_path=_first(form, 'vision_review_path', cls.vision_review_path),
            unified_store_path=_first(form, 'unified_store_path', cls.unified_store_path),
            unified_output_dir=_first(form, 'unified_output_dir', cls.unified_output_dir),
            unified_review_queue_path=_first(form, 'unified_review_queue_path', cls.unified_review_queue_path),
            unified_source=_first(form, 'unified_source', cls.unified_source),
            unified_split=_first(form, 'unified_split', cls.unified_split),
            unified_operating_domain=_first(form, 'unified_operating_domain', cls.unified_operating_domain),
            unified_baseline_summary_path=_first(form, 'unified_baseline_summary_path', cls.unified_baseline_summary_path),
            unified_benchmark_corpus_path=_first(form, 'unified_benchmark_corpus_path', cls.unified_benchmark_corpus_path),
            unified_transfer_input=_first(form, 'unified_transfer_input', cls.unified_transfer_input),
            unified_approved_queries_only=_checked_form(form, 'unified_approved_queries_only', default=False),
            understanding_hidden_input=_first(form, 'understanding_hidden_input', cls.understanding_hidden_input),
            understanding_vlso_input=_first(form, 'understanding_vlso_input', cls.understanding_vlso_input),
            understanding_vlso_real_input=_first(form, 'understanding_vlso_real_input', cls.understanding_vlso_real_input),
            compare_left_path=_first(form, 'compare_left_path', cls.compare_left_path),
            compare_right_path=_first(form, 'compare_right_path', cls.compare_right_path),
        )


@dataclass(frozen=True)
class ActionOutcome:
    flash: str = ''
    flash_tone: str = 'neutral'
    result_kind: str = ''
    result_payload: dict[str, object] | None = None


@dataclass
class BackgroundJob:
    job_id: str
    action: str
    label: str
    state: StudioState
    status: str = 'queued'
    detail: str = 'Waiting to start.'
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result_kind: str = ''
    result_payload: dict[str, object] | None = None
    flash: str = ''
    flash_tone: str = 'neutral'
    error_text: str = ''
    cancel_requested: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)


JOB_HISTORY_FILENAME = 'studio_job_history.json'
NOTIFICATION_FILENAME = 'studio_notifications.json'
MAX_JOB_HISTORY = 24
MAX_NOTIFICATIONS = 32


class JobCancelledError(RuntimeError):
    pass


def _first(form: dict[str, list[str]], key: str, default: str = '') -> str:
    values = form.get(key)
    return values[0] if values else default


def _checked_form(form: dict[str, list[str]], key: str, default: bool = False) -> bool:
    values = form.get(key)
    if not values:
        return default
    return values[-1] == '1'


def _checked(value: bool) -> str:
    return ' checked' if value else ''


def _safe_json(payload: object) -> str:
    return html.escape(json.dumps(payload, ensure_ascii=False, indent=2))


def _render_value(value: object) -> str:
    if value is None or value == '':
        return '-'
    if isinstance(value, float):
        return f'{value:.3f}'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (list, tuple)):
        return ', '.join(str(item) for item in value[:8]) or '-'
    return str(value)


def _metric_card(label: str, value: object) -> str:
    return f"<div class='metric-card'><small>{html.escape(label)}</small><strong>{html.escape(_render_value(value))}</strong></div>"


def _info_block(label: str, value: object) -> str:
    return f"<div class='info-block'><small>{html.escape(label)}</small><div>{html.escape(_render_value(value))}</div></div>"


def _action_time_hint(action: str) -> str:
    return ACTION_TIME_HINTS.get(str(action).strip(), 'varies by local data size')


def _action_estimated_seconds(action: str) -> int | None:
    value = ACTION_TIME_SECONDS.get(str(action).strip())
    return int(value) if value is not None else None


def _remaining_eta_label(action: str, progress: float) -> str:
    total = _action_estimated_seconds(action)
    if total is None:
        return _action_time_hint(action)
    remaining = max(0, int(total * (1.0 - max(0.0, min(1.0, float(progress))))))
    return _format_elapsed(float(remaining))


def _file_size_label(path_value: str) -> str:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return '-'
    size = path.stat().st_size
    units = ['B', 'KB', 'MB', 'GB']
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f'{value:.1f} {unit}' if unit != 'B' else f'{int(value)} {unit}'
        value /= 1024.0
    return '-'


def _raw_details(payload: dict[str, object]) -> str:
    return f"<details class='raw-json'><summary>Raw JSON</summary><pre>{_safe_json(payload)}</pre></details>"


def _local_image_data_uri(path_value: str) -> str:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return ''
    mime_type = mimetypes.guess_type(str(path))[0] or 'image/jpeg'
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'data:{mime_type};base64,{encoded}'


def _read_json_dict(path_value: str) -> dict[str, object]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _status_text(path_value: str, ready: str = 'ready', missing: str = 'missing') -> str:
    return ready if path_value and Path(path_value).exists() else missing


def _snippet(text: str, max_words: int = 6, max_chars: int = 72) -> str:
    cleaned = ' '.join(str(text or '').split())
    if not cleaned:
        return ''
    words = cleaned.split()
    short = ' '.join(words[:max_words])
    if len(short) > max_chars:
        return short[: max_chars - 3].rstrip() + '...'
    return short


def _load_hidden_premise_cases(path_value: str) -> list[HiddenPremiseEvalCase]:
    path = Path(path_value)
    if not path.exists():
        return []
    rows: list[HiddenPremiseEvalCase] = []
    with path.open('r', encoding='utf-8-sig') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(HiddenPremiseEvalCase(**json.loads(line)))
    return rows


def _load_operator_transfer_cases(path_value: str) -> list[OperatorTransferEvalCase]:
    path = Path(path_value)
    if not path.exists():
        return []
    rows: list[OperatorTransferEvalCase] = []
    with path.open('r', encoding='utf-8-sig') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(OperatorTransferEvalCase(**json.loads(line)))
    return rows


def _load_vlso_cases(path_value: str) -> list:
    path = Path(path_value)
    if not path.exists():
        return []
    return VlsoGroundedEvaluator.load_cases(path)


def _clone_graph(graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
    return StructuredMeaningGraph.from_dict(graph.model_dump())


def _graph_domain(graph: StructuredMeaningGraph) -> str:
    return str(getattr(graph, 'domain', '') or 'general')


def _graph_scenario(graph: StructuredMeaningGraph) -> str:
    return str(getattr(graph, 'scenario', '') or 'qa')


def _load_store_graphs(store_path: str, source: str, split: str, limit: int = 120) -> list[StructuredMeaningGraph]:
    if not store_path or not Path(store_path).exists():
        return []
    store = CorpusMemoryStore(store_path)
    rows = list(store.fetch_graphs(split=split or None, source=source or None))
    return rows[:limit]


def _derive_starter_analogy_cases(graphs: list[StructuredMeaningGraph], limit: int = 10) -> list[AnalogyEvalCase]:
    cases: list[AnalogyEvalCase] = []
    for graph in graphs:
        if not graph.query.strip():
            continue
        anchor_requirements = [item for item in graph.required_premises if item]
        anchor_goals = [item for item in graph.hidden_goals if item]
        if not anchor_requirements and not anchor_goals:
            continue
        anchor = set(anchor_requirements + anchor_goals)
        similar_graphs: list[StructuredMeaningGraph] = []
        for other in graphs:
            if other.query == graph.query:
                continue
            overlap = anchor & set(other.required_premises + other.hidden_goals)
            if not overlap:
                continue
            similar_graphs.append(_clone_graph(other))
            if len(similar_graphs) >= 2:
                break
        if not similar_graphs:
            continue
        cases.append(AnalogyEvalCase(query=graph.query, similar_graphs=similar_graphs, expected_requirement=anchor_requirements[0] if anchor_requirements else ''))
        if len(cases) >= limit:
            break
    return cases

def _derive_starter_grounding_cases(graphs: list[StructuredMeaningGraph], limit: int = 10) -> list[GroundedExplanationEvalCase]:
    cases: list[GroundedExplanationEvalCase] = []
    for graph in graphs:
        if not graph.query.strip() or not graph.source_context.strip():
            continue
        grounded_claims: list[str] = []
        unsupported_claims: list[str] = []
        if graph.operator_execution is not None:
            for item in graph.operator_execution.claim_groundings:
                claim = _snippet(item.claim)
                if not claim:
                    continue
                if item.grounded:
                    grounded_claims.append(claim)
                else:
                    unsupported_claims.append(claim)
        evidence_terms: list[str] = []
        for result in graph.symbolic_results:
            if result.domain != 'document_grounding':
                continue
            for evidence in result.evidence:
                snippet = _snippet(evidence, max_words=5)
                if snippet and snippet not in evidence_terms:
                    evidence_terms.append(snippet)
            if not grounded_claims and result.answer.strip():
                for chunk in re.split(r"\band\b|[.;]\s*", result.answer):
                    snippet = _snippet(chunk)
                    if snippet and snippet not in grounded_claims:
                        grounded_claims.append(snippet)
        if not evidence_terms:
            for node in graph.nodes:
                if node.kind != 'evidence':
                    continue
                snippet = _snippet(str(node.attributes.get('text', node.label)))
                if snippet and snippet not in evidence_terms:
                    evidence_terms.append(snippet)
        if not grounded_claims and not evidence_terms:
            continue
        cases.append(
            GroundedExplanationEvalCase(
                query=graph.query,
                source_context=graph.source_context,
                expected_evidence_terms=evidence_terms[:3],
                expected_claim_terms=grounded_claims[:2],
                forbidden_unsupported_claim_terms=unsupported_claims[:2],
                domain=_graph_domain(graph),
                scenario=_graph_scenario(graph),
            )
        )
        if len(cases) >= limit:
            break
    return cases


def _derive_starter_compiler_cases(graphs: list[StructuredMeaningGraph], limit: int = 10) -> list[CompilerRepairEvalCase]:
    cases: list[CompilerRepairEvalCase] = []
    for graph in graphs:
        variants: list[StructuredMeaningGraph] = []
        if graph.hidden_goals and (graph.required_premises or graph.satisfied_premises or graph.missing_premises):
            broken = _clone_graph(graph)
            broken.operator_decompositions = []
            variants.append(broken)
        if graph.source_context.strip():
            broken = _clone_graph(graph)
            broken.nodes = [node for node in broken.nodes if node.id not in {'source_document'} and node.kind != 'evidence']
            broken.edges = [edge for edge in broken.edges if edge.relation not in {'USES_CONTEXT', 'HAS_EVIDENCE', 'GROUNDED_BY'}]
            variants.append(broken)
        if any(any(tag.startswith('multimodal:vision') for tag in node.provenance) for node in graph.nodes):
            broken = _clone_graph(graph)
            broken.nodes = [node for node in broken.nodes if node.id != 'visual_scene']
            broken.edges = [edge for edge in broken.edges if not (edge.source == 'question' and edge.relation == 'CONDITIONS_ON' and edge.target == 'visual_scene')]
            variants.append(broken)
        if graph.functor_hypotheses:
            broken = _clone_graph(graph)
            for functor in broken.functor_hypotheses:
                functor.object_map = {}
            variants.append(broken)
        if graph.symbolic_results:
            broken = _clone_graph(graph)
            mutated = False
            for result in broken.symbolic_results:
                if result.domain == 'document_grounding' and result.answer.strip():
                    result.answer = 'Inspect the hidden sensor.'
                    mutated = True
            if mutated:
                variants.append(broken)
        for broken in variants:
            cases.append(CompilerRepairEvalCase(graph=broken, expected_repair_terms=[], domain=_graph_domain(graph), scenario=_graph_scenario(graph)))
            if len(cases) >= limit:
                return cases
    return cases


def _opening_candidates(world_payload: dict[str, object]) -> list[str]:
    entities = world_payload.get('entities', []) if isinstance(world_payload, dict) else []
    ranked: list[tuple[int, str]] = []
    for entity in entities:
        if not isinstance(entity, dict) or entity.get('modality') != 'vision':
            continue
        attrs = entity.get('attributes', {}) if isinstance(entity.get('attributes'), dict) else {}
        labels = attrs.get('concept_labels') or []
        if not isinstance(labels, list):
            continue
        upper = {str(item).upper() for item in labels}
        if not ({'ACCESS_OPENING_CANDIDATE', 'ZIPPER_LIKE_PART', 'EDGE_OPENING'} & upper):
            continue
        score = 0
        if 'ACCESS_OPENING_CANDIDATE' in upper:
            score += 3
        if 'ZIPPER_LIKE_PART' in upper:
            score += 2
        if 'EDGE_OPENING' in upper:
            score += 2
        entity_id = str(entity.get('id', ''))
        if entity_id:
            ranked.append((score, entity_id))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [entity_id for score, entity_id in ranked if score > 0][:6]


def _download_review_counts(path_value: str) -> dict[str, int]:
    payload = _read_json_dict(path_value)
    rows = payload.get('reviews', payload) if isinstance(payload, dict) else {}
    counts = {'approved': 0, 'pending': 0, 'rejected': 0}
    if not isinstance(rows, dict):
        return counts
    for value in rows.values():
        if not isinstance(value, dict):
            continue
        status = str(value.get('status', 'pending')).strip().lower() or 'pending'
        if status in counts:
            counts[status] += 1
        else:
            counts['pending'] += 1
    return counts


def build_vision_payload(query: str, image_path: str, mode: str, answer_mode: str, concept_store: str, operator_store: str, weights: str) -> dict[str, object]:
    reasoner = VLSOReasoner(
        mode=mode,
        concept_store_path=concept_store or None,
        operator_store_path=operator_store or None,
        affordance_weights_path=weights or None,
        answer_mode=answer_mode,
    )
    world, answer = reasoner.answer(query, visual_input={'image_path': image_path, 'metadata': {'image_path': image_path}})
    return {'world': world.model_dump(), 'answer': answer.model_dump()}


def _store_graph_count(store_path: str, source: str, split: str) -> int:
    if not store_path or not Path(store_path).exists():
        return 0
    return CorpusMemoryStore(store_path).count_examples(split=split or None, source=source or None)


def _review_queue_snapshot(review_queue_path: str, domain: str) -> dict[str, int]:
    snapshot = {'pending': 0, 'approved': 0, 'rejected': 0, 'needs_followup': 0, 'total': 0, 'promotable': 0}
    if not review_queue_path or not Path(review_queue_path).exists():
        return snapshot
    store = ReviewQueueStore(review_queue_path)
    snapshot.update(store.fetch_stats())
    decisions = collect_review_promotion_decisions(review_queue_path, domain=domain or None)
    snapshot['promotable'] = sum(1 for _detail, decision in decisions if decision.promotable)
    return snapshot


def _unique_texts(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _resolve_operating_domain(state: StudioState) -> str:
    configured = str(state.unified_operating_domain or '').strip()
    ops_domain = str(state.ops_domain or '').strip()
    if configured and configured != 'general':
        return configured
    if ops_domain and ops_domain != 'general':
        return ops_domain
    return configured or 'general'


def _guided_bootstrap_requests(state: StudioState) -> list[CopilotRequest]:
    context = state.ops_context.strip() or OPS_EXAMPLE['context']
    domain = state.ops_domain.strip() or OPS_EXAMPLE['domain']
    scenario = state.ops_scenario.strip() or OPS_EXAMPLE['scenario']
    queries = _unique_texts([state.ops_query.strip() or OPS_EXAMPLE['query'], *GUIDED_BOOTSTRAP_VARIANTS])
    return [CopilotRequest(query=query, context=context, domain=domain, scenario=scenario) for query in queries]


def _proof_round_spec(state: StudioState, round_index: int) -> dict[str, Any]:
    template = PROOF_CURRICULUM_ROUNDS[min(max(0, round_index), len(PROOF_CURRICULUM_ROUNDS) - 1)]
    if round_index == 0:
        return {
            'domain': str(state.ops_domain or template['domain']).strip() or template['domain'],
            'scenario': str(state.ops_scenario or template['scenario']).strip() or template['scenario'],
            'context': str(state.ops_context or template['context']).strip() or template['context'],
            'queries': _unique_texts([state.ops_query.strip() or OPS_EXAMPLE['query'], *GUIDED_BOOTSTRAP_VARIANTS, *template['queries']]),
            'visual_queries': _unique_texts([state.vision_query.strip() or VISION_EXAMPLE['query'], *template['visual_queries']]),
        }
    return {
        'domain': template['domain'],
        'scenario': template['scenario'],
        'context': template['context'],
        'queries': _unique_texts(list(template['queries'])),
        'visual_queries': _unique_texts(list(template['visual_queries'])),
    }


def _proof_round_requests(state: StudioState, round_index: int) -> list[CopilotRequest]:
    spec = _proof_round_spec(state, round_index)
    return [
        CopilotRequest(
            query=query,
            context=spec['context'],
            domain=spec['domain'],
            scenario=spec['scenario'],
        )
        for query in spec['queries']
    ]


def _seed_visual_curriculum(state: StudioState, source_used: str, queries: list[str] | tuple[str, ...], split: str = 'train') -> dict[str, Any]:
    image_path = Path(state.vision_image)
    summary = {
        'visual_seeded': 0,
        'visual_skipped': 0,
        'visual_queries': _unique_texts(list(queries)),
        'errors': [],
    }
    if not image_path.exists():
        summary['errors'].append(f'missing image: {image_path}')
        return summary
    store = CorpusMemoryStore(state.unified_store_path)
    pipeline = StructuredMeaningPipeline(mode='heuristic')
    existing_queries = {str(query).strip() for query in store.fetch_queries(split=split, source=source_used)}
    for query in summary['visual_queries']:
        if query in existing_queries:
            summary['visual_skipped'] += 1
            continue
        try:
            graph = pipeline.run(
                query,
                visual_input={'image_path': str(image_path), 'metadata': {'image_path': str(image_path)}},
            )
            graph.domain = 'vlso'
            _persist_graph(store, graph, source_used, split)
            existing_queries.add(query)
            summary['visual_seeded'] += 1
        except Exception as exc:
            summary['errors'].append(f'{query[:60]} ({exc})')
    return summary

def _bootstrap_review_reasons(graph: StructuredMeaningGraph, kpis: dict[str, Any]) -> list[str]:
    reasons = review_reasons_from_graph_and_kpis(graph, kpis)
    if graph.source_context.strip() and 'grounding_review' not in reasons:
        reasons.append('grounding_review')
    reasons.append('approved_training_trace')
    return _unique_texts(reasons)


def _grounded_review_answer(context: str, fallback: str) -> str:
    normalized_context = re.sub(r'\s+', ' ', str(context).strip())
    if normalized_context:
        for chunk in re.split(r'(?<=[.!?])\s+|\n+', normalized_context):
            line = str(chunk).strip(' -')
            if len(line) >= 12:
                return line
        return normalized_context
    return str(fallback).strip()


def _persist_graph(store: CorpusMemoryStore, graph: StructuredMeaningGraph, source: str, split: str = 'train') -> None:
    store.upsert_graph(graph, source=source, split=split)
    store.upsert_premise_operator_memory(graph, source=source, split=split)


def _seed_builtin_corpus(state: StudioState, source_used: str, split: str = 'train') -> dict[str, Any]:
    store = CorpusMemoryStore(state.unified_store_path)
    pipeline = StructuredMeaningPipeline(mode='heuristic')
    existing_queries = {str(query).strip() for query in store.fetch_queries(split=split, source=source_used)}
    summary: dict[str, Any] = {
        'hidden_cases_loaded': 0,
        'hidden_seeded': 0,
        'transfer_cases_loaded': 0,
        'transfer_seeded': 0,
        'visual_seeded': 0,
        'skipped_existing': 0,
        'errors': [],
    }

    hidden_cases = _load_hidden_premise_cases(state.understanding_hidden_input)
    summary['hidden_cases_loaded'] = len(hidden_cases)
    for case in hidden_cases:
        query = str(case.query).strip()
        if not query:
            continue
        if query in existing_queries:
            summary['skipped_existing'] += 1
            continue
        try:
            graph = pipeline.run(query)
            graph.domain = str(case.domain or 'general').strip() or 'general'
            _persist_graph(store, graph, source_used, split)
            existing_queries.add(query)
            summary['hidden_seeded'] += 1
        except Exception as exc:
            summary['errors'].append(f'hidden case failed: {query[:60]} ({exc})')

    transfer_cases = _load_operator_transfer_cases(state.unified_transfer_input)
    summary['transfer_cases_loaded'] = len(transfer_cases)
    for case in transfer_cases:
        query = str(case.query).strip()
        if not query:
            continue
        if query in existing_queries:
            summary['skipped_existing'] += 1
            continue
        try:
            graph = pipeline.run(query)
            graph.domain = str(getattr(case, 'domain', 'general') or 'general').strip() or 'general'
            _persist_graph(store, graph, source_used, split)
            existing_queries.add(query)
            summary['transfer_seeded'] += 1
        except Exception as exc:
            summary['errors'].append(f'transfer case failed: {query[:60]} ({exc})')

    image_path = Path(state.vision_image)
    visual_query = str(state.vision_query).strip() or VISION_EXAMPLE['query']
    if image_path.exists() and visual_query not in existing_queries:
        try:
            graph = pipeline.run(
                visual_query,
                visual_input={'image_path': str(image_path), 'metadata': {'image_path': str(image_path)}},
            )
            graph.domain = 'vlso'
            _persist_graph(store, graph, source_used, split)
            existing_queries.add(visual_query)
            summary['visual_seeded'] += 1
        except Exception as exc:
            summary['errors'].append(f'visual seed failed: {exc}')

    summary['total_seeded'] = int(summary['hidden_seeded']) + int(summary['transfer_seeded']) + int(summary['visual_seeded'])
    return summary


def _load_gate_payload(output_dir: str) -> dict[str, Any]:
    gate_path = Path(output_dir) / 'benchmark_gate.json'
    if not gate_path.exists():
        return {}
    try:
        return json.loads(gate_path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _state_to_payload(state: StudioState) -> dict[str, Any]:
    return {name: getattr(state, name) for name in StudioState.__dataclass_fields__}


def _state_from_payload(payload: dict[str, Any] | None, fallback: StudioState | None = None) -> StudioState:
    if not isinstance(payload, dict):
        return fallback or StudioState()
    values = {name: payload.get(name, getattr(StudioState, name)) for name in StudioState.__dataclass_fields__}
    try:
        return StudioState(**values)
    except Exception:
        return fallback or StudioState()


def _job_history_path(output_dir: str) -> Path:
    base = Path(output_dir or UNIFIED_EXAMPLE['output_dir'])
    return base / JOB_HISTORY_FILENAME


def _read_job_history(output_dir: str) -> list[dict[str, Any]]:
    path = _job_history_path(output_dir)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def _write_job_history(output_dir: str, entries: list[dict[str, Any]]) -> None:
    path = _job_history_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries[:MAX_JOB_HISTORY], ensure_ascii=False, indent=2), encoding='utf-8')


def _notification_history_path(output_dir: str) -> Path:
    base = Path(output_dir or UNIFIED_EXAMPLE['output_dir'])
    return base / NOTIFICATION_FILENAME


def _read_notifications(output_dir: str) -> list[dict[str, Any]]:
    path = _notification_history_path(output_dir)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def _write_notifications(output_dir: str, entries: list[dict[str, Any]]) -> None:
    path = _notification_history_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries[:MAX_NOTIFICATIONS], ensure_ascii=False, indent=2), encoding='utf-8')


def _notification_snapshot(output_dir: str) -> dict[str, Any]:
    entries = _read_notifications(output_dir)
    unread = [item for item in entries if isinstance(item, dict) and not bool(item.get('read'))]
    return {
        'unread_count': len(unread),
        'recent': entries[:8],
        'path': str(_notification_history_path(output_dir)),
    }


def _build_notification(job: BackgroundJob) -> dict[str, Any]:
    level = 'success'
    title = f"{job.label} finished"
    if job.status == 'failed':
        level = 'error'
        title = f"{job.label} failed"
    elif job.status == 'cancelled':
        level = 'neutral'
        title = f"{job.label} cancelled"
    return {
        'id': f"note-{job.job_id}",
        'job_id': job.job_id,
        'created_at': time.time(),
        'level': level,
        'title': title,
        'body': job.detail,
        'read': False,
        'result_kind': job.result_kind,
        'action': job.action,
    }


def _upsert_notification(output_dir: str, entry: dict[str, Any]) -> None:
    notifications = [item for item in _read_notifications(output_dir) if str(item.get('id', '')) != str(entry.get('id', ''))]
    notifications.insert(0, entry)
    _write_notifications(output_dir, notifications)


def _mark_notification(output_dir: str, notification_id: str, *, read: bool = True) -> None:
    notifications = _read_notifications(output_dir)
    changed = False
    for item in notifications:
        if str(item.get('id', '')) != notification_id:
            continue
        item['read'] = read
        changed = True
        break
    if changed:
        _write_notifications(output_dir, notifications)


def _mark_all_notifications(output_dir: str, *, read: bool = True) -> None:
    notifications = _read_notifications(output_dir)
    changed = False
    for item in notifications:
        if not isinstance(item, dict):
            continue
        if bool(item.get('read')) == read:
            continue
        item['read'] = read
        changed = True
    if changed:
        _write_notifications(output_dir, notifications)


def _history_to_snapshot(entry: dict[str, Any]) -> dict[str, Any]:
    created_at = float(entry.get('created_at') or 0.0)
    started_at = float(entry.get('started_at') or 0.0)
    finished_at = float(entry.get('finished_at') or time.time())
    baseline = started_at or created_at or finished_at
    status = str(entry.get('status', 'completed'))
    return {
        'job_id': str(entry.get('job_id', '')),
        'label': str(entry.get('label', '-')),
        'status': status,
        'detail': str(entry.get('detail', '-')),
        'progress': float(entry.get('progress', 0.0) or 0.0),
        'elapsed_seconds': max(0.0, finished_at - baseline),
        'result_kind': str(entry.get('result_kind', '')),
        'action': str(entry.get('action', '')),
        'can_cancel': False,
        'can_retry': bool(entry.get('action')),
        'can_load': bool(entry.get('result_kind')),
        'persisted': True,
        'output_dir': str(entry.get('output_dir', '')),
        'events': entry.get('events', []) if isinstance(entry.get('events'), list) else [],
        'gate_accepted': entry.get('gate_accepted'),
        'time_hint': _action_time_hint(str(entry.get('action', ''))),
        'remaining_eta': '-',
    }


def _output_file_action_form(path_value: str, label: str) -> str:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return ''
    return ''.join([
        "<form method='post' class='mini-form'>",
        "<input type='hidden' name='action' value='load_output_file'>",
        f"<input type='hidden' name='file_path' value='{html.escape(str(path))}'>",
        f"<button class='secondary compact' type='submit'>{html.escape(label)}</button>",
        "</form>",
    ])


def _render_output_quick_actions(candidates: list[tuple[str, str]]) -> str:
    controls = [_output_file_action_form(path_value, label) for label, path_value in candidates if path_value]
    controls = [control for control in controls if control]
    if not controls:
        return ''
    return "<div class='job-recovery'><small>Quick file preview</small><div class='job-actions'>" + ''.join(controls) + "</div></div>"


def _preview_output_file(file_path: str) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f'Output file not found: {file_path}')
    suffix = path.suffix.lower()
    if suffix in {'.json', '.jsonl'}:
        try:
            content = path.read_text(encoding='utf-8')
            payload = json.loads(content) if suffix == '.json' else [json.loads(line) for line in content.splitlines() if line.strip()][:20]
            preview_text = json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            preview_text = path.read_text(encoding='utf-8', errors='replace')[:12000]
    else:
        preview_text = path.read_text(encoding='utf-8', errors='replace')[:12000]
    return {
        'path': str(path),
        'filename': path.name,
        'size_label': _file_size_label(str(path)),
        'modified_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(path.stat().st_mtime)),
        'preview_text': preview_text,
    }


def _default_compare_paths(state: StudioState) -> tuple[str, str]:
    left = str(state.compare_left_path or '').strip()
    right = str(state.compare_right_path or '').strip()
    if not left:
        left = str(Path(state.unified_output_dir) / 'benchmark_gate.json')
    if not right:
        baseline = str(state.unified_baseline_summary_path or '').strip()
        accepted = str(Path(state.unified_output_dir) / 'accepted_benchmark_summary.json')
        parser = str(Path(state.unified_output_dir) / 'unified_parser.json')
        if baseline:
            right = baseline
        elif Path(accepted).exists():
            right = accepted
        else:
            right = parser
    return left, right


def _load_text_for_compare(path: Path) -> str:
    if path.is_dir():
        rows: list[str] = []
        for child in sorted(p for p in path.rglob('*') if p.is_file()):
            rel = child.relative_to(path)
            rows.append(f"{rel} | {_file_size_label(str(child))}")
        return '\n'.join(rows)
    return path.read_text(encoding='utf-8', errors='replace')


def _flatten_compare_value(value: Any, prefix: str = '$') -> dict[str, str]:
    flattened: dict[str, str] = {}
    if isinstance(value, dict):
        if not value:
            flattened[prefix] = '{}'
        for key in sorted(value):
            flattened.update(_flatten_compare_value(value[key], f'{prefix}.{key}'))
        return flattened
    if isinstance(value, list):
        if not value:
            flattened[prefix] = '[]'
        for index, item in enumerate(value):
            flattened.update(_flatten_compare_value(item, f'{prefix}[{index}]'))
        return flattened
    flattened[prefix] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return flattened


def _compare_paths(left_path_value: str, right_path_value: str) -> dict[str, Any]:
    left = Path(left_path_value)
    right = Path(right_path_value)
    if not left.exists():
        raise FileNotFoundError(f'Left compare path not found: {left}')
    if not right.exists():
        raise FileNotFoundError(f'Right compare path not found: {right}')
    if left.is_dir() and right.is_dir():
        left_files = {str(path.relative_to(left)): path for path in left.rglob('*') if path.is_file()}
        right_files = {str(path.relative_to(right)): path for path in right.rglob('*') if path.is_file()}
        left_only = sorted(set(left_files) - set(right_files))
        right_only = sorted(set(right_files) - set(left_files))
        changed: list[str] = []
        same = 0
        diff_preview = ''
        for rel in sorted(set(left_files) & set(right_files)):
            left_text = _load_text_for_compare(left_files[rel])
            right_text = _load_text_for_compare(right_files[rel])
            if left_text == right_text:
                same += 1
                continue
            changed.append(rel)
            if not diff_preview:
                diff_preview = '\n'.join(list(__import__('difflib').unified_diff(
                    left_text.splitlines(),
                    right_text.splitlines(),
                    fromfile=f'left/{rel}',
                    tofile=f'right/{rel}',
                    lineterm='',
                ))[:200])
        return {
            'compare_mode': 'directory',
            'left_path': str(left),
            'right_path': str(right),
            'left_size_label': f"{len(left_files)} files",
            'right_size_label': f"{len(right_files)} files",
            'changed_count': len(changed),
            'left_only_count': len(left_only),
            'right_only_count': len(right_only),
            'same_count': same,
            'changed_items': changed[:40],
            'left_only_items': left_only[:40],
            'right_only_items': right_only[:40],
            'diff_preview': diff_preview or 'No textual diff preview was needed. Files are identical where both sides overlap.',
            'identical': not changed and not left_only and not right_only,
        }
    if left.is_dir() != right.is_dir():
        raise ValueError('Compare both paths as files or both as directories.')
    left_text = _load_text_for_compare(left)
    right_text = _load_text_for_compare(right)
    changed_paths: list[str] = []
    left_only_paths: list[str] = []
    right_only_paths: list[str] = []
    same_count = 0
    if left.suffix.lower() == '.json' and right.suffix.lower() == '.json':
        left_payload = json.loads(left_text)
        right_payload = json.loads(right_text)
        left_flat = _flatten_compare_value(left_payload)
        right_flat = _flatten_compare_value(right_payload)
        for key in sorted(set(left_flat) | set(right_flat)):
            if key not in right_flat:
                left_only_paths.append(key)
            elif key not in left_flat:
                right_only_paths.append(key)
            elif left_flat[key] != right_flat[key]:
                changed_paths.append(key)
            else:
                same_count += 1
    diff_preview = '\n'.join(list(__import__('difflib').unified_diff(
        left_text.splitlines(),
        right_text.splitlines(),
        fromfile=left.name,
        tofile=right.name,
        lineterm='',
    ))[:220])
    return {
        'compare_mode': 'file',
        'left_path': str(left),
        'right_path': str(right),
        'left_size_label': _file_size_label(str(left)),
        'right_size_label': _file_size_label(str(right)),
        'left_format': left.suffix.lower() or 'text',
        'right_format': right.suffix.lower() or 'text',
        'changed_count': len(changed_paths),
        'left_only_count': len(left_only_paths),
        'right_only_count': len(right_only_paths),
        'same_count': same_count,
        'changed_items': changed_paths[:40],
        'left_only_items': left_only_paths[:40],
        'right_only_items': right_only_paths[:40],
        'diff_preview': diff_preview or 'No line-level difference detected.',
        'identical': left_text == right_text,
    }
def _job_action_form(action: str, job_id: str, label: str, tone: str = 'secondary') -> str:
    if not job_id:
        return ''
    return ''.join([
        "<form method='post' class='mini-form'>",
        f"<input type='hidden' name='action' value='{html.escape(action)}'>",
        f"<input type='hidden' name='job_id' value='{html.escape(job_id)}'>",
        f"<button class='{html.escape(tone)} compact' type='submit'>{html.escape(label)}</button>",
        "</form>",
    ])


def _render_job_action_bar(item: dict[str, Any]) -> str:
    controls: list[str] = []
    if item.get('can_cancel'):
        controls.append(_job_action_form('cancel_job', str(item.get('job_id', '')), 'Cancel'))
    if item.get('can_retry'):
        controls.append(_job_action_form('retry_job', str(item.get('job_id', '')), 'Retry'))
    if item.get('can_load'):
        controls.append(_job_action_form('load_job_result', str(item.get('job_id', '')), 'Load result'))
    if not controls:
        return ''
    return f"<div class='job-actions'>{''.join(controls)}</div>"


def _job_recovery_form(job_id: str, recovery_action: str, label: str) -> str:
    if not job_id or not recovery_action:
        return ''
    return ''.join([
        "<form method='post' class='mini-form'>",
        "<input type='hidden' name='action' value='run_recovery_action'>",
        f"<input type='hidden' name='job_id' value='{html.escape(job_id)}'>",
        f"<input type='hidden' name='recovery_action' value='{html.escape(recovery_action)}'>",
        f"<button class='secondary compact' type='submit'>{html.escape(label)}</button>",
        "</form>",
    ])


def _job_recovery_specs(item: dict[str, Any]) -> list[tuple[str, str]]:
    status = str(item.get('status', '')).strip().lower()
    action = str(item.get('action', '')).strip()
    gate_accepted = item.get('gate_accepted')
    detail = str(item.get('detail', '')).lower()
    specs: list[tuple[str, str]] = []
    if status in {'failed', 'cancelled'}:
        if action in {'run_unified_benchmark_gate', 'run_unified_training', 'run_generalization_proof'} or 'no graphs were found' in detail:
            specs.append(('run_autopilot_coach', 'Do everything for me'))
            specs.append(('run_beginner_autopilot', 'Beginner setup'))
            specs.append(('run_guided_learning', 'Guided starter loop'))
        elif action in {'run_beginner_test', 'run_beginner_autopilot'}:
            specs.append(('run_beginner_test', 'One-click test'))
            specs.append(('run_beginner_autopilot', 'One-click setup'))
        else:
            specs.append(('run_understanding_eval', 'Understanding benchmark'))
    if gate_accepted is False:
        specs.append(('run_autopilot_coach', 'Do everything for me'))
        specs.append(('run_guided_learning', 'Bootstrap gate'))
        specs.append(('run_beginner_autopilot', 'One-click recovery'))
    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for recovery_action, label in specs:
        if recovery_action in seen:
            continue
        seen.add(recovery_action)
        ordered.append((recovery_action, label))
    return ordered[:3]


def _render_job_recovery_bar(item: dict[str, Any]) -> str:
    job_id = str(item.get('job_id', ''))
    controls = [_job_recovery_form(job_id, action, label) for action, label in _job_recovery_specs(item)]
    controls = [control for control in controls if control]
    if not controls:
        return ''
    return "<div class='job-recovery'><small>Suggested recovery</small><div class='job-actions'>" + ''.join(controls) + "</div></div>"


def _render_job_log(item: dict[str, Any]) -> str:
    events = item.get('events', []) if isinstance(item.get('events'), list) else []
    if not events:
        return ''
    lines: list[str] = []
    for event in events[-8:]:
        if not isinstance(event, dict):
            continue
        stamp = time.strftime('%H:%M:%S', time.localtime(float(event.get('at', time.time()) or time.time())))
        message = str(event.get('message', '-'))
        lines.append(f"<div class='job-log-line'><span>{html.escape(stamp)}</span><code>{html.escape(message)}</code></div>")
    if not lines:
        return ''
    return "<details class='job-log'><summary>Step log</summary>" + ''.join(lines) + "</details>"


def _gate_diagnosis(payload: dict[str, object] | None) -> dict[str, list[str]]:
    if not isinstance(payload, dict):
        return {'observations': [], 'actions': []}
    benchmark = payload.get('benchmark', {}) if isinstance(payload.get('benchmark'), dict) else {}
    gate = payload.get('gate', {}) if isinstance(payload.get('gate'), dict) else {}
    promoted = payload.get('promoted_review_benchmarks', {}) if isinstance(payload.get('promoted_review_benchmarks'), dict) else {}
    corpus = payload.get('benchmark_corpus', {}) if isinstance(payload.get('benchmark_corpus'), dict) else {}
    training = payload.get('training', {}) if isinstance(payload.get('training'), dict) else {}

    observations: list[str] = []
    actions: list[str] = []

    promoted_count = int(promoted.get('promoted_review_count', 0) or 0)
    grounding_cases = int(promoted.get('grounding_case_count', 0) or 0) + int(corpus.get('grounding_case_count', 0) or 0)
    compiler_cases = int(promoted.get('compiler_case_count', 0) or 0) + int(corpus.get('compiler_case_count', 0) or 0)
    seed_graphs = int(training.get('trained_on_graphs', 0) or 0)

    if promoted_count == 0:
        observations.append('No approved review traces were promoted into the learning loop yet.')
        actions.append('Run Guided starter loop once to seed approved training traces automatically.')
    if seed_graphs < 3:
        observations.append(f'The current seed graph count is only {seed_graphs}, so analogy learning is still thin.')
        actions.append('Add at least 3-4 closely related starter queries so the analogy policy can compare similar cases.')
    if float(benchmark.get('analogy_usefulness', 0.0) or 0.0) <= 0.0:
        observations.append('Analogy usefulness is zero, which usually means the seed graphs do not overlap enough yet.')
    if grounding_cases == 0:
        observations.append('No grounded explanation benchmark cases were available for the gate.')
        actions.append('Approve at least one grounded trace with SOP context so grounding cases can be derived.')
    elif float(benchmark.get('grounded_explanation_fidelity', 0.0) or 0.0) <= 0.0:
        observations.append('Grounding cases exist, but the current bundle did not support their claims or evidence yet.')
        actions.append('Inspect grounded answers and keep approved document-grounded examples in the review queue.')
    if compiler_cases == 0:
        observations.append('No compiler or repair benchmark cases were available for the gate.')
        actions.append('Keep at least one approved grounding or compiler review so repair cases can be derived automatically.')
    elif float(benchmark.get('repair_success_rate', 0.0) or 0.0) <= 0.0:
        observations.append('Repair cases exist, but the learned repair program did not fire on them yet.')
        actions.append('Seed approved grounded traces first, then rerun the gate so document and claim repair cases are harvested.')
    if gate.get('blocking_reasons') and not actions:
        actions.append('Open the blocking reasons below and fix the lowest-scoring axis first.')
    return {'observations': _unique_texts(observations), 'actions': _unique_texts(actions)}


def _render_diagnosis_blocks(payload: dict[str, object] | None) -> str:
    diagnosis = _gate_diagnosis(payload)
    observations = diagnosis.get('observations', [])
    actions = diagnosis.get('actions', [])
    if not observations and not actions:
        return ''
    return ''.join([
        _info_block('What this result means', observations or '-'),
        _info_block('Beginner next step', actions or '-'),
    ])


def _render_ops_summary(payload: dict[str, object]) -> str:
    metrics = ''.join([
        _metric_card('Domain', payload.get('domain')),
        _metric_card('Scenario', payload.get('scenario')),
        _metric_card('Plan executability', payload.get('plan_executability')),
        _metric_card('Relation recovery', payload.get('relation_recovery')),
        _metric_card('Audit usefulness', payload.get('human_audit_usefulness')),
    ])
    details = ''.join([
        _info_block('Answer', payload.get('response_text') or payload.get('answer_text') or payload.get('summary') or '-'),
        _info_block('Warnings', payload.get('warnings') or '-'),
        _info_block('Recommended actions', payload.get('recommended_actions') or '-'),
    ])
    return f"<div class='summary-grid'>{metrics}</div>{details}{_raw_details(payload)}"


def _render_vision_summary(payload: dict[str, object]) -> str:
    answer = payload.get('answer', {}) if isinstance(payload.get('answer'), dict) else {}
    world = payload.get('world', {}) if isinstance(payload.get('world'), dict) else {}
    entities = world.get('entities', []) if isinstance(world.get('entities'), list) else []
    visual_entities = [entity for entity in entities if isinstance(entity, dict) and entity.get('modality') == 'vision']
    metrics = ''.join([
        _metric_card('Vision entities', len(visual_entities)),
        _metric_card('Relations', len(world.get('relations', []) if isinstance(world.get('relations'), list) else [])),
        _metric_card('Opening candidates', len(_opening_candidates(world))),
        _metric_card('Answer mode', answer.get('answer_mode') or '-'),
    ])
    details = ''.join([
        _info_block('Answer', answer.get('answer_text') or '-'),
        _info_block('Likely openings', _opening_candidates(world) or '-'),
        _info_block('Warnings', answer.get('warnings') or world.get('warnings') or '-'),
    ])
    return f"<div class='summary-grid'>{metrics}</div>{details}{_raw_details(payload)}"


def _render_unified_training_summary(payload: dict[str, object]) -> str:
    artifacts = payload.get('artifacts', {}) if isinstance(payload.get('artifacts'), dict) else {}
    artifact_ready = sum(1 for value in artifacts.values() if isinstance(value, str) and value and Path(value).exists())
    metrics = ''.join([
        _metric_card('Seed graphs', payload.get('trained_on_graphs')),
        _metric_card('Promoted review graphs', payload.get('promoted_review_graph_count')),
        _metric_card('Repair trace graphs', payload.get('repair_trace_graph_count')),
        _metric_card('Augmented graphs', payload.get('augmented_graph_count')),
        _metric_card('Artifacts ready', f"{artifact_ready}/{len(artifacts) or 1}"),
    ])
    details = ''.join([
        _info_block('Analogy policy', artifacts.get('analogy_policy_path') or '-'),
        _info_block('Unified parser', artifacts.get('unified_parser_path') or '-'),
        _info_block('Repair utility', artifacts.get('repair_utility_path') or '-'),
        _info_block('Continuous learning bundle', artifacts.get('continuous_learning_bundle_dir') or '-'),
    ])
    quick_actions = _render_output_quick_actions([
        ('Analogy policy', str(artifacts.get('analogy_policy_path') or '')),
        ('Unified parser', str(artifacts.get('unified_parser_path') or '')),
        ('Repair utility', str(artifacts.get('repair_utility_path') or '')),
        ('Retained algebra', str(artifacts.get('retained_operator_algebra_path') or '')),
    ])
    return f"<div class='summary-grid'>{metrics}</div>{details}{quick_actions}{_raw_details(payload)}"


def _render_unified_gate_summary(payload: dict[str, object]) -> str:
    benchmark = payload.get('benchmark', {}) if isinstance(payload.get('benchmark'), dict) else {}
    gate = payload.get('gate', {}) if isinstance(payload.get('gate'), dict) else {}
    promoted = payload.get('promoted_review_benchmarks', {}) if isinstance(payload.get('promoted_review_benchmarks'), dict) else {}
    corpus = payload.get('benchmark_corpus', {}) if isinstance(payload.get('benchmark_corpus'), dict) else {}
    guided = payload.get('guided_bootstrap', {}) if isinstance(payload.get('guided_bootstrap'), dict) else {}
    metrics = ''.join([
        _metric_card('Accepted', gate.get('accepted')),
        _metric_card('Unseen transfer', benchmark.get('unseen_transfer')),
        _metric_card('Analogy usefulness', benchmark.get('analogy_usefulness')),
        _metric_card('Compiler validity', benchmark.get('compiler_validity')),
        _metric_card('Grounded fidelity', benchmark.get('grounded_explanation_fidelity')),
        _metric_card('Repair success', benchmark.get('repair_success_rate')),
    ])
    details = ''.join([
        _info_block('Improved axes', gate.get('improved_axes') or '-'),
        _info_block('Blocking reasons', gate.get('blocking_reasons') or gate.get('slice_blocking_reasons') or '-'),
        _info_block('Decision path', gate.get('decision_path') or '-'),
        _info_block('Promoted review cases', promoted.get('promoted_review_count') or '-'),
        _info_block('Grounding cases', promoted.get('grounding_case_count') or corpus.get('grounding_case_count') or '-'),
        _info_block('Repair cases', promoted.get('compiler_case_count') or corpus.get('compiler_case_count') or '-'),
        _info_block('Guided source', guided.get('source_used') or '-'),
        _info_block('Persistent corpus added', corpus.get('added_case_count') or '-'),
    ])
    quick_actions = _render_output_quick_actions([
        ('Gate decision', str(gate.get('decision_path') or '')),
        ('Accepted summary', str(gate.get('accepted_summary_path') or '')),
        ('Benchmark corpus', str(corpus.get('benchmark_corpus_path') or payload.get('benchmark_corpus_path') or '')),
    ])
    diagnosis = _render_diagnosis_blocks(payload)
    return f"<div class='summary-grid'>{metrics}</div>{details}{quick_actions}{diagnosis}{_raw_details(payload)}"


def _render_understanding_summary(payload: dict[str, object]) -> str:
    progress = payload.get('progress', {}) if isinstance(payload.get('progress'), dict) else {}
    interpretation = payload.get('interpretation', {}) if isinstance(payload.get('interpretation'), dict) else {}
    metrics = ''.join([
        _metric_card('Research architecture', progress.get('research_architecture_overall')),
        _metric_card('Robust understanding', progress.get('robust_general_intelligence_overall')),
        _metric_card('Premise reasoning', (progress.get('premise_reasoning') or {}).get('score') if isinstance(progress.get('premise_reasoning'), dict) else '-'),
        _metric_card('Shared world model', (progress.get('shared_world_model') or {}).get('score') if isinstance(progress.get('shared_world_model'), dict) else '-'),
        _metric_card('Operator architecture', (progress.get('operator_architecture') or {}).get('score') if isinstance(progress.get('operator_architecture'), dict) else '-'),
    ])
    details = ''.join([
        _info_block('Headline', interpretation.get('headline') or '-'),
        _info_block('Strengths', interpretation.get('strengths') or '-'),
        _info_block('Risks', interpretation.get('risks') or '-'),
        _info_block('Next steps', interpretation.get('next_steps') or '-'),
    ])
    return f"<div class='summary-grid'>{metrics}</div>{details}{_raw_details(payload)}"


def _render_beginner_suite_summary(payload: dict[str, object]) -> str:
    setup = payload.get('autopilot_setup', {}) if isinstance(payload.get('autopilot_setup'), dict) else {}
    benchmark = payload.get('benchmark', {}) if isinstance(payload.get('benchmark'), dict) else {}
    gate = payload.get('gate', {}) if isinstance(payload.get('gate'), dict) else {}
    understanding = payload.get('understanding', {}) if isinstance(payload.get('understanding'), dict) else {}
    ops = payload.get('ops', {}) if isinstance(payload.get('ops'), dict) else {}
    vision = payload.get('vision', {}) if isinstance(payload.get('vision'), dict) else {}
    progress = understanding.get('progress', {}) if isinstance(understanding.get('progress'), dict) else {}
    metrics = ''.join([
        _metric_card('Seeded graphs', setup.get('total_seeded') or 0),
        _metric_card('Approved traces', setup.get('approved_review_count') or 0),
        _metric_card('Gate accepted', gate.get('accepted')),
        _metric_card('Analogy usefulness', benchmark.get('analogy_usefulness')),
        _metric_card('Grounded fidelity', benchmark.get('grounded_explanation_fidelity')),
        _metric_card('Understanding', progress.get('robust_general_intelligence_overall') or '-'),
    ])
    details = ''.join([
        _info_block('Setup summary', [
            f"hidden seeded: {setup.get('hidden_seeded', 0)}",
            f"transfer seeded: {setup.get('transfer_seeded', 0)}",
            f"visual seeded: {setup.get('visual_seeded', 0)}",
            f"promotable reviews: {setup.get('promotable_review_count', 0)}",
        ]),
        _info_block('Context smoke test', ops.get('answer_text') or ops.get('summary') or '-'),
        _info_block('Vision smoke test', (vision.get('answer', {}) if isinstance(vision.get('answer'), dict) else {}).get('answer_text') or '-'),
        _info_block('Gate diagnosis', gate.get('blocking_reasons') or gate.get('slice_blocking_reasons') or 'ready'),
    ])
    quick_actions = _render_output_quick_actions([
        ('Gate decision', str(gate.get('decision_path') or '')),
        ('Accepted summary', str(gate.get('accepted_summary_path') or '')),
        ('Understanding report', str(payload.get('understanding_report_path') or '')),
    ])
    diagnosis = _render_diagnosis_blocks(payload)
    return f"<div class='summary-grid'>{metrics}</div>{details}{quick_actions}{diagnosis}{_raw_details(payload)}"

def _render_generalization_proof_summary(payload: dict[str, object]) -> str:
    proof = payload.get('proof', {}) if isinstance(payload.get('proof'), dict) else payload
    evidence = proof.get('evidence', {}) if isinstance(proof.get('evidence'), dict) else {}
    goal_tracker = proof.get('goal_tracker', {}) if isinstance(proof.get('goal_tracker'), dict) else {}
    rounds = proof.get('rounds', []) if isinstance(proof.get('rounds'), list) else []
    last_round = rounds[-1] if rounds else {}
    last_benchmark = last_round.get('benchmark', {}) if isinstance(last_round, dict) and isinstance(last_round.get('benchmark'), dict) else {}
    last_understanding = last_round.get('understanding', {}) if isinstance(last_round, dict) and isinstance(last_round.get('understanding'), dict) else {}
    progress = last_understanding.get('progress', {}) if isinstance(last_understanding.get('progress'), dict) else {}
    curriculum_rounds = payload.get('curriculum_rounds', []) if isinstance(payload.get('curriculum_rounds'), list) else []
    tracker_axes = goal_tracker.get('axes', []) if isinstance(goal_tracker.get('axes'), list) else []
    axis_lines = [
        f"{item.get('label', item.get('key', 'axis'))}: {float(item.get('score', 0.0) or 0.0):.3f}/{float(item.get('target', 0.0) or 0.0):.3f} ({'ready' if item.get('verified') else 'pending'})"
        for item in tracker_axes
        if isinstance(item, dict)
    ]
    metrics = ''.join([
        _metric_card('Rounds', len(rounds)),
        _metric_card('Accepted rounds', proof.get('accepted_rounds', 0)),
        _metric_card('Learned generalization', evidence.get('learned_generalization_score')),
        _metric_card('Corpus growth', evidence.get('reviewed_corpus_growth_score')),
        _metric_card('Multimodal transfer', evidence.get('multimodal_transfer_score')),
        _metric_card('Domain coverage', evidence.get('domain_coverage_score')),
        _metric_card('Strong model score', evidence.get('strong_model_score')),
        _metric_card('Goal readiness', f"{goal_tracker.get('readiness_percent', 0)}%"),
        _metric_card('Robust understanding', progress.get('robust_general_intelligence_overall') or '-'),
    ])
    curriculum_summary = [
        f"round {item.get('round_index', '?')}: {item.get('approved_review_count', 0)} approved reviews, {item.get('visual_seeded', 0)} visual graphs"
        for item in curriculum_rounds
        if isinstance(item, dict)
    ]
    details = ''.join([
        _info_block('Headline', evidence.get('headline') or '-'),
        _info_block('Ultimate goal tracker', [
            f"ready axes: {goal_tracker.get('ready_axes', 0)}/{goal_tracker.get('total_axes', 0)}",
            f"priority focus: {goal_tracker.get('priority_focus') or '-'}",
        ]),
        _info_block('Completed axes', goal_tracker.get('completed_items') or '-'),
        _info_block('Remaining axes', goal_tracker.get('remaining_items') or '-'),
        _info_block('Coverage', [
            f"domains: {', '.join(goal_tracker.get('domains_seen', []) or []) or '-'}",
            f"scenarios: {', '.join(goal_tracker.get('scenarios_seen', []) or []) or '-'}",
        ]),
        _info_block('Axis detail', axis_lines or '-'),
        _info_block('Strengths', evidence.get('strengths') or '-'),
        _info_block('Risks', evidence.get('risks') or '-'),
        _info_block('Next steps', evidence.get('next_steps') or '-'),
        _info_block('Latest benchmark', [
            f"unseen transfer: {last_benchmark.get('unseen_transfer', 0.0)}",
            f"analogy usefulness: {last_benchmark.get('analogy_usefulness', 0.0)}",
            f"compiler validity: {last_benchmark.get('compiler_validity', 0.0)}",
            f"grounded fidelity: {last_benchmark.get('grounded_explanation_fidelity', 0.0)}",
            f"repair success: {last_benchmark.get('repair_success_rate', 0.0)}",
        ]),
        _info_block('Curriculum rounds', curriculum_summary or '-'),
    ])
    report_path = str(proof.get('report_path') or payload.get('proof_report_path') or '')
    round_output_dir = str(last_round.get('output_dir') or '') if isinstance(last_round, dict) else ''
    quick_actions = _render_output_quick_actions([
        ('Proof report', report_path),
        ('Last gate decision', str(Path(round_output_dir) / 'benchmark_gate.json') if round_output_dir else ''),
        ('Last accepted summary', str(Path(round_output_dir) / 'accepted_benchmark_summary.json') if round_output_dir else ''),
    ])
    return f"<div class='summary-grid'>{metrics}</div>{details}{quick_actions}{_raw_details(payload)}"

def _render_file_preview_summary(payload: dict[str, object]) -> str:
    metrics = ''.join([
        _metric_card('Filename', payload.get('filename')),
        _metric_card('Size', payload.get('size_label')),
        _metric_card('Modified', payload.get('modified_at')),
    ])
    details = ''.join([
        _info_block('Path', payload.get('path') or '-'),
    ])
    preview = html.escape(str(payload.get('preview_text', '')))
    return f"<div class='summary-grid'>{metrics}</div>{details}<details class='raw-json' open><summary>File preview</summary><pre>{preview}</pre></details>"


def _render_artifact_compare_summary(payload: dict[str, object]) -> str:
    metrics = ''.join([
        _metric_card('Mode', payload.get('compare_mode')),
        _metric_card('Changed', payload.get('changed_count')),
        _metric_card('Left only', payload.get('left_only_count')),
        _metric_card('Right only', payload.get('right_only_count')),
        _metric_card('Same', payload.get('same_count')),
        _metric_card('Identical', payload.get('identical')),
    ])
    details = ''.join([
        _info_block('Left', payload.get('left_path') or '-'),
        _info_block('Right', payload.get('right_path') or '-'),
        _info_block('Left size', payload.get('left_size_label') or '-'),
        _info_block('Right size', payload.get('right_size_label') or '-'),
        _info_block('Changed items', payload.get('changed_items') or '-'),
        _info_block('Left-only items', payload.get('left_only_items') or '-'),
        _info_block('Right-only items', payload.get('right_only_items') or '-'),
    ])
    diff_preview = html.escape(str(payload.get('diff_preview', '')))
    return f"<div class='summary-grid'>{metrics}</div>{details}<details class='raw-json' open><summary>Diff preview</summary><pre>{diff_preview}</pre></details>{_raw_details(payload)}"


def _render_notification_panel(snapshot: dict[str, object]) -> str:
    recent = snapshot.get('recent', []) if isinstance(snapshot.get('recent'), list) else []
    unread_count = int(snapshot.get('unread_count', 0) or 0)
    metrics = ''.join([
        _metric_card('Unread alerts', unread_count),
        _metric_card('Saved alerts', len(recent)),
        _metric_card('Notification file', Path(str(snapshot.get('path', '-'))).name if snapshot.get('path') else '-'),
    ])
    controls = ''.join([
        "<form method='post' class='mini-form'><input type='hidden' name='action' value='mark_all_notifications_read'><button class='secondary compact' type='submit'>Mark all read</button></form>",
    ])
    rows: list[str] = []
    for item in recent[:6]:
        if not isinstance(item, dict):
            continue
        note_id = str(item.get('id', ''))
        level = str(item.get('level', 'neutral'))
        read = bool(item.get('read'))
        tone = 'unread' if not read else 'read'
        dismiss = ''
        if note_id and not read:
            dismiss = ''.join([
                "<form method='post' class='mini-form'>",
                "<input type='hidden' name='action' value='dismiss_notification'>",
                f"<input type='hidden' name='notification_id' value='{html.escape(note_id)}'>",
                "<button class='secondary compact' type='submit'>Dismiss</button>",
                "</form>",
            ])
        rows.append(
            "".join([
                f"<div class='notification-row {tone} {html.escape(level)}'>",
                f"<div class='job-row-head'><strong>{html.escape(str(item.get('title', '-')))}</strong><span class='status-chip'>{html.escape(level)}</span></div>",
                f"<div>{html.escape(str(item.get('body', '-')))}</div>",
                f"<div class='job-meta'>Time: {html.escape(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(item.get('created_at', time.time()) or time.time()))))}</div>",
                dismiss,
                "</div>",
            ])
        )
    rows_html = ''.join(rows) or "<div class='empty'>No saved notifications yet.</div>"
    return (
        "<section class='card' id='notification-panel' style='margin-bottom:20px;'>"
        "<div class='section-head'><div><small class='eyebrow'>Notifications</small><h2>Completion alerts</h2><p>Finished, failed, and cancelled background jobs show up here so a beginner can tell what changed without scanning the full log.</p></div>"
        f"<div class='job-actions'>{controls}</div></div>"
        f"<div class='summary-grid'>{metrics}</div>"
        f"<div class='job-list'>{rows_html}</div>"
        "</section>"
    )


def render_result(kind: str, payload: dict[str, object] | None) -> str:
    if payload is None:
        return "<div class='empty'>Run one of the actions above. The result console will summarize the latest answer or training run here.</div>"
    titles = {
        'ops': 'Context reasoning result',
        'vision': 'Vision-grounded reasoning result',
        'unified_training': 'Unified SemOp training result',
        'unified_benchmark_gate': 'Unified benchmark gate result',
        'guided_learning': 'Guided starter learning result',
        'beginner_autopilot': 'One-click beginner setup result',
        'beginner_test': 'One-click beginner test result',
        'generalization_proof': 'Generalization proof result',
        'autopilot_coach': 'Autopilot coach result',
        'understanding_eval': 'Overall understanding benchmark',
        'file_preview': 'Output file preview',
        'artifact_compare': 'Artifact comparison',
    }
    if kind == 'ops':
        body = _render_ops_summary(payload)
    elif kind == 'vision':
        body = _render_vision_summary(payload)
    elif kind == 'unified_training':
        body = _render_unified_training_summary(payload)
    elif kind in {'unified_benchmark_gate', 'guided_learning'}:
        body = _render_unified_gate_summary(payload)
    elif kind in {'beginner_autopilot', 'beginner_test'}:
        body = _render_beginner_suite_summary(payload)
    elif kind in {'generalization_proof', 'autopilot_coach'}:
        body = _render_generalization_proof_summary(payload)
    elif kind == 'understanding_eval':
        body = _render_understanding_summary(payload)
    elif kind == 'file_preview':
        body = _render_file_preview_summary(payload)
    elif kind == 'artifact_compare':
        body = _render_artifact_compare_summary(payload)
    else:
        body = f"<div class='summary-grid'>{_metric_card('Status', 'completed')}</div>{_raw_details(payload)}"
    return f"<h3>{html.escape(titles.get(kind, kind or 'Result'))}</h3>{body}"


def render_workspace_snapshot(state: StudioState, result_kind: str) -> str:
    unified_output_dir = Path(state.unified_output_dir)
    unified_artifacts = [
        unified_output_dir / 'analogy_policy.json',
        unified_output_dir / 'unified_parser.json',
        unified_output_dir / 'graph_supervision.jsonl',
        unified_output_dir / 'retained_operator_algebra.json',
        unified_output_dir / 'operator_repair_policy.json',
        unified_output_dir / 'retained_repair_programs.json',
        unified_output_dir / 'repair_utility.json',
        unified_output_dir / 'multimodal_alignment.json',
    ]
    artifact_ready = sum(1 for path in unified_artifacts if path.exists())
    download_reviews = _download_review_counts(state.vision_review_path)
    benchmark_inputs = [
        state.understanding_hidden_input,
        state.unified_transfer_input,
        state.understanding_vlso_input,
        state.understanding_vlso_real_input,
    ]
    benchmark_ready = sum(1 for path in benchmark_inputs if Path(path).exists())
    store_graphs = _store_graph_count(state.unified_store_path, state.unified_source, state.unified_split)
    review_snapshot = _review_queue_snapshot(state.unified_review_queue_path, _resolve_operating_domain(state))
    proof_report_path = unified_output_dir / 'generalization_proof' / 'generalization_proof_report.json'
    proof_snapshot = _read_json_dict(str(proof_report_path))
    proof_goal_tracker = proof_snapshot.get('goal_tracker', {}) if isinstance(proof_snapshot.get('goal_tracker'), dict) else {}
    cards = [
        (
            'Unified trainer lane',
            'Main artifacts for parser, analogy, repair, and continuous learning.',
            [
                _metric_card('Corpus store', _status_text(state.unified_store_path, 'connected', 'missing')),
                _metric_card('Seed graphs', store_graphs),
                _metric_card('Approved reviews', review_snapshot.get('approved', 0)),
                _metric_card('Promotable reviews', review_snapshot.get('promotable', 0)),
                _metric_card('Artifacts ready', f'{artifact_ready}/{len(unified_artifacts)}'),
                _metric_card('Gate summary', _status_text(str(unified_output_dir / 'benchmark_gate.json'), 'ready', 'not run')),
                _metric_card('Proof report', _status_text(str(proof_report_path), 'ready', 'not run')),
                _metric_card('Proof readiness', f"{proof_goal_tracker.get('readiness_percent', 0)}%" if proof_goal_tracker else 'not run'),
            ],
            [
                _info_block('Output dir', state.unified_output_dir),
                _info_block('Persistent benchmark corpus', state.unified_benchmark_corpus_path),
                _info_block('Proof report', str(proof_report_path)),
                _info_block('Proof priority focus', proof_goal_tracker.get('priority_focus') or '-'),
            ],
        ),
        (
            'Vision assets',
            'Starter image QA is wired to the same visual stores used in training workflows.',
            [
                _metric_card('Scene image', _status_text(state.vision_image, 'ready', 'missing')),
                _metric_card('Concept store', _status_text(state.vision_concept_store, 'ready', 'missing')),
                _metric_card('Operator store', _status_text(state.vision_operator_store, 'ready', 'missing')),
                _metric_card('Approved label reviews', download_reviews.get('approved', 0)),
            ],
            [
                _info_block('Review file', state.vision_review_path),
                _info_block('Affordance weights', state.vision_weights),
            ],
        ),
        (
            'Benchmark readiness',
            'The gate is only meaningful when starter benchmark inputs are available.',
            [
                _metric_card('Benchmark files ready', f'{benchmark_ready}/{len(benchmark_inputs)}'),
                _metric_card('Transfer input', _status_text(state.unified_transfer_input, 'ready', 'missing')),
                _metric_card('Pending reviews', review_snapshot.get('pending', 0)),
                _metric_card('Last result', result_kind or 'none'),
                _metric_card('Approved-only mode', state.unified_approved_queries_only),
            ],
            [
                _info_block('Hidden eval', state.understanding_hidden_input),
                _info_block('VLSO eval', state.understanding_vlso_input),
            ],
        ),
        (
            'Advanced labs',
            'Keep the friendly studio for the main loop and open the older tools only when you need more knobs.',
            [
                _metric_card('Easy lab', 'semop_easy_gui.py'),
                _metric_card('Ops lab', 'ops_copilot_gui.py'),
                _metric_card('CP lab', 'cp_copilot_gui.py'),
                _metric_card('Accepted summary', _status_text(str(unified_output_dir / 'accepted_benchmark_summary.json'), 'ready', 'none')),
            ],
            [
                _info_block('Use when', 'Power-user data collection, cluster review, or CP-only workflows'),
                _info_block('Main recommendation', 'Stay in this studio for reasoning, unified training, and benchmark gate runs'),
            ],
        ),
    ]
    rendered = []
    for title, subtitle, metrics, details in cards:
        rendered.append(
            f"<article class='lane-card'><div class='lane-top'><h3>{html.escape(title)}</h3><p>{html.escape(subtitle)}</p></div><div class='summary-grid'>{''.join(metrics)}</div>{''.join(details)}</article>"
        )
    return ''.join(rendered)


def _format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return '-'
    total = max(0, int(seconds))
    if total < 60:
        return f'{total}s'
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f'{minutes}m {secs:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes:02d}m'


def _render_job_queue_panel(snapshot: dict[str, object]) -> str:
    active = snapshot.get('active', {}) if isinstance(snapshot.get('active'), dict) else {}
    recent = snapshot.get('recent', []) if isinstance(snapshot.get('recent'), list) else []
    active_eta = active.get('remaining_eta') if isinstance(active, dict) else ''
    metrics = ''.join([
        _metric_card('Running jobs', snapshot.get('running_count', 0)),
        _metric_card('Queued jobs', snapshot.get('queued_count', 0)),
        _metric_card('Completed jobs', snapshot.get('completed_count', 0)),
        _metric_card('Saved history', snapshot.get('saved_history_count', 0)),
        _metric_card('Current ETA', active_eta or '-'),
        _metric_card('Auto refresh', 'on' if snapshot.get('has_active') else 'idle'),
    ])
    if active:
        progress = max(0, min(100, int((float(active.get('progress', 0.0) or 0.0)) * 100)))
        progress_label = f'{progress}%'
        active_html = ''.join([
            "<div class='info-block'><small>Active job</small>",
            f"<div class='job-row-head'><strong>{html.escape(str(active.get('label', '-')))}</strong><span class='status-chip'>{html.escape(str(active.get('status', '-')))}</span></div>",
            f"<div>{html.escape(str(active.get('detail', '-')))}</div>",
            f"<div class='job-progress'><span style='width:{progress}%;'></span></div>",
            f"<div class='job-meta'>Elapsed: {html.escape(_format_elapsed(active.get('elapsed_seconds')))} | Progress: {html.escape(progress_label)} | ETA: {html.escape(str(active.get('remaining_eta', '-')))} | Job id: {html.escape(str(active.get('job_id', '-')))}</div>",
            _render_job_action_bar(active),
            _render_job_recovery_bar(active),
            _render_job_log(active),
            "</div>",
        ])
    else:
        active_html = "<div class='empty'>No background job is running right now. Queue one of the training or one-click actions and this panel will start tracking it automatically.</div>"
    recent_rows: list[str] = []
    for item in recent[:8]:
        if not isinstance(item, dict):
            continue
        recent_rows.append(
            "".join([
                "<div class='job-row'>",
                f"<div class='job-row-head'><strong>{html.escape(str(item.get('label', '-')))}</strong><span class='status-chip'>{html.escape(str(item.get('status', '-')))}</span></div>",
                f"<div>{html.escape(str(item.get('detail', '-')))}</div>",
                f"<div class='job-meta'>Elapsed: {html.escape(_format_elapsed(item.get('elapsed_seconds')))} | Progress: {html.escape(str(max(0, min(100, int((float(item.get('progress', 0.0) or 0.0)) * 100)))) + '%')} | ETA: {html.escape(str(item.get('remaining_eta', '-')))}</div>",
                _render_job_action_bar(item),
                _render_job_recovery_bar(item),
                _render_job_log(item),
                "</div>",
            ])
        )
    recent_html = ''.join(recent_rows) or "<div class='empty'>No background job history yet.</div>"
    history_path = snapshot.get('history_path') or '-'
    return (
        "<section class='card' id='job-panel' style='margin-bottom:20px;'>"
        "<div class='section-head'><div><small class='eyebrow'>Live jobs</small><h2>Background queue and progress</h2><p>Long training and beginner flows now run in the background. Keep this page open; it refreshes automatically while a job is active.</p></div></div>"
        f"<div class='summary-grid'>{metrics}</div>"
        f"{active_html}"
        f"<div class='info-block'><small>Saved history file</small><div>{html.escape(str(history_path))}</div></div>"
        f"<div class='info-block'><small>Recent jobs</small><div class='job-list'>{recent_html}</div></div>"
        "</section>"
    )


def render_result_spotlight(kind: str, payload: dict[str, object] | None) -> str:
    if payload is None:
        tips = ''.join(f'<li>{html.escape(item)}</li>' for item in [
            'Run the ops or vision example first to make sure the local stack is healthy.',
            'Then launch Unified trainer to export parser, repair, and continuous-learning artifacts.',
            'Finish with Train + benchmark gate to see whether the new bundle clears the operating policy.',
        ])
        return (
            "<div class='spotlight-card'><small class='eyebrow'>No latest run yet</small>"
            "<h2>Start from a single example, then move into the training loop.</h2>"
            f"<ul class='spotlight-list'>{tips}</ul></div>"
        )
    if kind in {'unified_benchmark_gate', 'guided_learning'}:
        gate = payload.get('gate', {}) if isinstance(payload.get('gate'), dict) else {}
        benchmark = payload.get('benchmark', {}) if isinstance(payload.get('benchmark'), dict) else {}
        title = 'The benchmark gate is open.' if gate.get('accepted') else 'The benchmark gate is still blocked.'
        notes = [
            f"Compiler validity: {_render_value(benchmark.get('compiler_validity'))}",
            f"Grounded fidelity: {_render_value(benchmark.get('grounded_explanation_fidelity'))}",
            f"Repair success: {_render_value(benchmark.get('repair_success_rate'))}",
        ]
        diagnosis = _gate_diagnosis(payload)
        notes.extend(diagnosis.get('observations', [])[:2])
        notes.extend(diagnosis.get('actions', [])[:2])
        if gate.get('blocking_reasons'):
            notes.extend(str(item) for item in gate.get('blocking_reasons', [])[:2])
    elif kind in {'beginner_autopilot', 'beginner_test'}:
        gate = payload.get('gate', {}) if isinstance(payload.get('gate'), dict) else {}
        benchmark = payload.get('benchmark', {}) if isinstance(payload.get('benchmark'), dict) else {}
        understanding = payload.get('understanding', {}) if isinstance(payload.get('understanding'), dict) else {}
        progress = understanding.get('progress', {}) if isinstance(understanding.get('progress'), dict) else {}
        setup = payload.get('autopilot_setup', {}) if isinstance(payload.get('autopilot_setup'), dict) else {}
        title = 'One-click beginner flow finished.' if kind == 'beginner_autopilot' else 'One-click beginner test finished.'
        notes = [
            f"Seeded graphs: {_render_value(setup.get('total_seeded'))}",
            f"Approved traces: {_render_value(setup.get('approved_review_count'))}",
            f"Gate accepted: {_render_value(gate.get('accepted'))}",
            f"Understanding: {_render_value(progress.get('robust_general_intelligence_overall'))}",
        ]
        diagnosis = _gate_diagnosis(payload)
        notes.extend(diagnosis.get('actions', [])[:2])
        if not gate.get('accepted') and gate.get('blocking_reasons'):
            notes.extend(str(item) for item in gate.get('blocking_reasons', [])[:2])
    elif kind == 'unified_training':
        title = 'Unified artifacts were exported successfully.'
        notes = [
            f"Seed graphs: {_render_value(payload.get('trained_on_graphs'))}",
            f"Augmented graphs: {_render_value(payload.get('augmented_graph_count'))}",
            'Next: run Train + benchmark gate to score the new artifact bundle.',
        ]
    elif kind == 'ops':
        title = 'Context reasoning finished.'
        notes = [
            str(payload.get('response_text') or payload.get('answer_text') or payload.get('summary') or '-'),
            f"Plan executability: {_render_value(payload.get('plan_executability'))}",
            f"Relation recovery: {_render_value(payload.get('relation_recovery'))}",
        ]
    elif kind == 'vision':
        answer = payload.get('answer', {}) if isinstance(payload.get('answer'), dict) else {}
        world = payload.get('world', {}) if isinstance(payload.get('world'), dict) else {}
        title = 'Vision-grounded reasoning finished.'
        notes = [
            str(answer.get('answer_text') or '-'),
            f"Opening candidates: {_render_value(_opening_candidates(world))}",
            'If this looks right, keep the same stores for the unified benchmark gate.',
        ]
    elif kind == 'artifact_compare':
        title = 'Artifact comparison finished.'
        notes = [
            f"Mode: {_render_value(payload.get('compare_mode'))}",
            f"Changed: {_render_value(payload.get('changed_count'))}",
            f"Left only: {_render_value(payload.get('left_only_count'))}",
            f"Right only: {_render_value(payload.get('right_only_count'))}",
        ]
    elif kind in {'generalization_proof', 'autopilot_coach'}:
        proof = payload.get('proof', {}) if isinstance(payload.get('proof'), dict) else payload
        evidence = proof.get('evidence', {}) if isinstance(proof.get('evidence'), dict) else {}
        goal_tracker = proof.get('goal_tracker', {}) if isinstance(proof.get('goal_tracker'), dict) else {}
        title = 'The stronger proof loop cleared the current bar.' if evidence.get('strong_model_ready') else 'The stronger proof loop still shows remaining gaps.'
        notes = [
            f"Goal readiness: {_render_value(goal_tracker.get('readiness_percent'))}%",
            f"Ready axes: {_render_value(goal_tracker.get('ready_axes'))}/{_render_value(goal_tracker.get('total_axes'))}",
            f"Priority focus: {goal_tracker.get('priority_focus') or '-'}",
            f"Strong model score: {_render_value(evidence.get('strong_model_score'))}",
        ]
        notes.extend(str(item) for item in (goal_tracker.get('remaining_items') or [])[:2])
    elif kind == 'file_preview':
        title = 'Output file preview loaded.'
        notes = [
            str(payload.get('filename') or '-'),
            f"Size: {_render_value(payload.get('size_label'))}",
            'Use Compare artifacts to inspect changes across two runs or two files.',
        ]
    else:
        title = 'The last run finished.'
        notes = ['Open the result console below for the full structured payload.']
    items = ''.join(f'<li>{html.escape(item)}</li>' for item in notes if item)
    return f"<div class='spotlight-card'><small class='eyebrow'>Latest spotlight</small><h2>{html.escape(title)}</h2><ul class='spotlight-list'>{items}</ul></div>"


def _render_select(name: str, current: str, options: tuple[str, ...]) -> str:
    rendered = []
    for option in options:
        selected = ' selected' if current == option else ''
        rendered.append(f"<option value='{html.escape(option)}'{selected}>{html.escape(option)}</option>")
    return f"<select name='{html.escape(name)}'>{''.join(rendered)}</select>"


def _render_checkbox(name: str, checked: bool, label: str) -> str:
    return f"<input type='hidden' name='{html.escape(name)}' value='0'><label class='inline'><input type='checkbox' name='{html.escape(name)}' value='1'{_checked(checked)}> {html.escape(label)}</label>"


def _render_hidden_state_inputs(state: StudioState) -> str:
    controls: list[str] = []
    for name in StudioState.__dataclass_fields__:
        value = getattr(state, name)
        if isinstance(value, bool):
            encoded = '1' if value else '0'
        else:
            encoded = str(value)
        controls.append(f"<input type='hidden' name='{html.escape(name)}' value='{html.escape(encoded)}'>")
    return ''.join(controls)

def _render_unified_scope_fields(state: StudioState) -> str:
    return "".join([
        "<div class='mini-grid'>",
        f"<div><label>Corpus store path</label><input name='unified_store_path' value='{html.escape(state.unified_store_path)}'></div>",
        f"<div><label>Output dir</label><input name='unified_output_dir' value='{html.escape(state.unified_output_dir)}'></div>",
        f"<div><label>Review queue path</label><input name='unified_review_queue_path' value='{html.escape(state.unified_review_queue_path)}'></div>",
        f"<div><label>Operating domain</label><input name='unified_operating_domain' value='{html.escape(state.unified_operating_domain)}'></div>",
        f"<div><label>Source filter</label><input name='unified_source' value='{html.escape(state.unified_source)}'></div>",
        f"<div><label>Split</label>{_render_select('unified_split', state.unified_split, SPLIT_OPTIONS)}</div>",
        "</div>",
        _render_checkbox('unified_approved_queries_only', state.unified_approved_queries_only, 'use approved review queries only'),
    ])


def _render_reasoning_section(state: StudioState) -> str:
    return f"""
  <section class="grid" id="reasoning-lab">
    <form method="post" class="card">
      <div class="section-head"><div><small class="eyebrow">Reasoning</small><h2>Context reasoning lab</h2><p>Use this to test hidden goals, constraints, and action guidance before you train.</p></div></div>
      <label>Question</label>
      <textarea name="ops_query">{html.escape(state.ops_query)}</textarea>
      <label>Context or SOP</label>
      <textarea name="ops_context">{html.escape(state.ops_context)}</textarea>
      <div class="mini-grid">
        <div><label>Domain</label><input name="ops_domain" value="{html.escape(state.ops_domain)}"></div>
        <div><label>Scenario</label><input name="ops_scenario" value="{html.escape(state.ops_scenario)}"></div>
      </div>
      <div class="actions"><button class="secondary" name="action" value="load_ops_example">Load example</button><button class="primary" name="action" value="run_ops">Run context reasoning</button></div>
    </form>
    <form method="post" class="card">
      <div class="section-head"><div><small class="eyebrow">Vision</small><h2>Vision-grounded reasoning lab</h2><p>Ask an image question with the same concept and operator stores used by your training loops.</p></div></div>
      <label>Question</label>
      <input name="vision_query" value="{html.escape(state.vision_query)}">
      <label>Image path</label>
      <input name="vision_image" value="{html.escape(state.vision_image)}">
      <div class="mini-grid">
        <div><label>Reasoning mode</label>{_render_select('vision_mode', state.vision_mode, VISION_MODE_OPTIONS)}</div>
        <div><label>Answer mode</label>{_render_select('vision_answer_mode', state.vision_answer_mode, VISION_ANSWER_MODE_OPTIONS)}</div>
      </div>
      <label>Concept store</label><input name="vision_concept_store" value="{html.escape(state.vision_concept_store)}">
      <label>Operator store</label><input name="vision_operator_store" value="{html.escape(state.vision_operator_store)}">
      <label>Affordance weights</label><input name="vision_weights" value="{html.escape(state.vision_weights)}">
      <label>Downloaded-label review file</label><input name="vision_review_path" value="{html.escape(state.vision_review_path)}">
      <div class="actions"><button class="secondary" name="action" value="load_vision_example">Load example</button><button class="primary" name="action" value="run_vision">Run vision reasoning</button></div>
    </form>
  </section>
"""

def _render_autopilot_coach_section(state: StudioState) -> str:
    return f"""
  <section class="card" id="autopilot-lab" style="margin-top:20px;">
    <div class="section-head"><div><small class="eyebrow">Zero-brain mode</small><h2>Autopilot coach</h2><p>Use one button to seed starter data, grow a small reviewed curriculum, rerun benchmark-gated learning across multiple rounds, and write a plain-language proof report.</p></div></div>
    <form method="post">
      {_render_hidden_state_inputs(state)}
      <div class="info-block"><small>What happens</small>1. Beginner setup seeds starter graphs and approved traces. 2. Three curriculum rounds add nearby text and visual cases. 3. The benchmark gate reruns across repeated rounds. 4. A proof report explains whether generalization, multimodal transfer, and reviewed-corpus growth are actually improving.</div>
      <div class="info-block"><small>Expected time</small>{html.escape(_action_time_hint('run_autopilot_coach'))} for the full coach, or {html.escape(_action_time_hint('run_generalization_proof'))} to re-check the proof on the current store.</div>
      <div class="actions"><button class="primary" name="action" value="run_autopilot_coach">Do everything for me</button><button class="secondary" name="action" value="run_generalization_proof">Re-check proof on current data</button></div>
    </form>
  </section>
"""

def _render_beginner_section(state: StudioState) -> str:
    return f"""
  <section class="card" id="beginner-lab" style="margin-top:20px;">
    <div class="section-head"><div><small class="eyebrow">Beginner Mode</small><h2>One-click setup, training, and testing</h2><p>Use the built-in example corpora to seed memory, create approved traces, train the core SemOp artifacts, and run a beginner-friendly smoke test without hand-assembling data first.</p></div></div>
    <form method="post">
      {_render_unified_scope_fields(state)}
      <div class="mini-grid">
        <div><label>Transfer eval JSONL</label><input name="unified_transfer_input" value="{html.escape(state.unified_transfer_input)}"></div>
        <div><label>Hidden premise eval JSONL</label><input name="understanding_hidden_input" value="{html.escape(state.understanding_hidden_input)}"></div>
        <div><label>VLSO eval JSONL</label><input name="understanding_vlso_input" value="{html.escape(state.understanding_vlso_input)}"></div>
        <div><label>VLSO real-image eval JSONL</label><input name="understanding_vlso_real_input" value="{html.escape(state.understanding_vlso_real_input)}"></div>
      </div>
      <div class="info-block"><small>What happens</small>1. Built-in hidden-premise, transfer, and starter vision examples are stored into your corpus DB. 2. Grounded starter reviews are approved automatically. 3. The unified artifact bundle is trained. 4. The benchmark gate and beginner smoke tests are run.</div>
      <div class="info-block"><small>Expected time</small>{html.escape(_action_time_hint('run_beginner_autopilot'))} for full setup, or {html.escape(_action_time_hint('run_beginner_test'))} to re-check the current bundle.</div>
      <div class="actions"><button class="primary" name="action" value="run_beginner_autopilot">One-click setup + train + test</button><button class="secondary" name="action" value="run_beginner_test">One-click test current bundle</button></div>
    </form>
  </section>
"""


def _render_training_section(state: StudioState) -> str:
    return f"""
  <section class="card" id="training-lab" style="margin-top:20px;">
    <div class="section-head"><div><small class="eyebrow">Training</small><h2>Unified artifact trainer</h2><p>This exports parser, analogy, retained algebra, repair policy, repair utility, multimodal alignment, and continuous-learning artifacts from one corpus store.</p></div></div>
    <form method="post">
      {_render_unified_scope_fields(state)}
      <div class="info-block"><small>Beginner shortcut</small>If the benchmark gate is stuck at 0.0 for analogy, grounding, or repair, use the guided starter loop once. It seeds approved traces, stores starter graphs, then runs training and the benchmark gate in one pass.</div>
      <div class="info-block"><small>Expected time</small>{html.escape(_action_time_hint('run_unified_training'))} for artifacts only, or {html.escape(_action_time_hint('run_guided_learning'))} if you include starter seeding and a gate rerun.</div>
      <div class="actions"><button class="primary" name="action" value="run_unified_training">Run unified trainer</button><button class="secondary" name="action" value="run_guided_learning">Guided starter loop</button></div>
    </form>
  </section>
"""


def _render_benchmark_section(state: StudioState) -> str:
    return f"""
  <section class="card" id="benchmark-lab" style="margin-top:20px;">
    <div class="section-head"><div><small class="eyebrow">Benchmark Gate</small><h2>Train plus benchmark gate</h2><p>This re-trains the unified artifact bundle, derives starter analogy and repair cases from the store, reuses your benchmark files, and writes a gate decision JSON.</p></div></div>
    <form method="post">
      {_render_unified_scope_fields(state)}
      <div class="mini-grid">
        <div><label>Transfer eval JSONL</label><input name="unified_transfer_input" value="{html.escape(state.unified_transfer_input)}"></div>
        <div><label>Baseline summary path</label><input name="unified_baseline_summary_path" value="{html.escape(state.unified_baseline_summary_path)}"></div>
        <div><label>Persistent benchmark corpus</label><input name="unified_benchmark_corpus_path" value="{html.escape(state.unified_benchmark_corpus_path)}"></div>
        <div><label>Hidden premise eval JSONL</label><input name="understanding_hidden_input" value="{html.escape(state.understanding_hidden_input)}"></div>
        <div><label>VLSO eval JSONL</label><input name="understanding_vlso_input" value="{html.escape(state.understanding_vlso_input)}"></div>
        <div><label>VLSO real-image eval JSONL</label><input name="understanding_vlso_real_input" value="{html.escape(state.understanding_vlso_real_input)}"></div>
      </div>
      <div class="actions"><button class="primary" name="action" value="run_unified_benchmark_gate">Train + benchmark gate</button><button class="secondary" name="action" value="run_guided_learning">Bootstrap + train + gate</button><button class="secondary" name="action" value="run_understanding_eval">Run overall understanding benchmark</button></div>
      <div class="info-block"><small>Expected time</small>{html.escape(_action_time_hint('run_unified_benchmark_gate'))} for the gate, or {html.escape(_action_time_hint('run_understanding_eval'))} for the understanding benchmark alone.</div>
      <small>The benchmark gate will also derive starter analogy, grounding, and repair cases from the selected corpus store, so you do not have to hand-author every case first.</small>
    </form>
  </section>
"""


def _render_notes_section() -> str:
    return """
  <section class="card" style="margin-top:20px;">
    <div class="section-head"><div><small class="eyebrow">Notes</small><h2>Friendly defaults, advanced escape hatch</h2><p>This studio keeps the main loop small. For bulk public-image collection, cluster review, or CP-only tooling, open the advanced GUIs that already ship in this repo.</p></div><div class="nav-row"><span class="nav-pill">semop_easy_gui.py</span><span class="nav-pill">ops_copilot_gui.py</span><span class="nav-pill">cp_copilot_gui.py</span></div></div>
  </section>
"""


def _render_compare_section(state: StudioState) -> str:
    left, right = _default_compare_paths(state)
    return f"""
  <section class="card" id="compare-lab" style="margin-top:20px;">
    <div class="section-head"><div><small class="eyebrow">Compare</small><h2>Artifact compare</h2><p>Compare two files or two output directories to see what changed across runs. This is useful for `benchmark_gate.json`, `unified_parser.json`, or two whole output bundles.</p></div></div>
    <form method="post">
      <div class="mini-grid">
        <div><label>Left file or dir</label><input name="compare_left_path" value="{html.escape(left)}"></div>
        <div><label>Right file or dir</label><input name="compare_right_path" value="{html.escape(right)}"></div>
      </div>
      <div class="info-block"><small>Starter suggestion</small>Use the current output dir's `benchmark_gate.json` on the left and either `accepted_benchmark_summary.json`, another run folder, or a second artifact file on the right.</div>
      <div class="actions"><button class="primary" name="action" value="run_artifact_compare">Compare artifacts</button><button class="secondary" name="action" value="load_compare_example">Use output defaults</button></div>
    </form>
  </section>
"""


class StudioApp:
    LONG_ACTIONS = {
        'run_unified_training',
        'run_unified_benchmark_gate',
        'run_guided_learning',
        'run_beginner_autopilot',
        'run_beginner_test',
        'run_generalization_proof',
        'run_autopilot_coach',
        'run_understanding_eval',
    }
    ACTION_LABELS = {
        'run_unified_training': 'Unified trainer',
        'run_unified_benchmark_gate': 'Benchmark gate',
        'run_guided_learning': 'Guided starter loop',
        'run_beginner_autopilot': 'One-click setup + train + test',
        'run_beginner_test': 'One-click test current bundle',
        'run_generalization_proof': 'Generalization proof',
        'run_autopilot_coach': 'Autopilot coach',
        'run_understanding_eval': 'Overall understanding benchmark',
    }

    def __init__(self) -> None:
        self._last_state = StudioState()
        self._last_outcome = ActionOutcome()
        self._job_lock = threading.Lock()
        self._job_queue: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, BackgroundJob] = {}
        self._job_order: list[str] = []
        self._job_counter = 0
        self._worker = threading.Thread(target=self._job_worker, name='semop-studio-worker', daemon=True)
        self._worker.start()

    @staticmethod
    def _copilot(review_queue_path: str | None = None) -> DomainCopilot:
        return DomainCopilot(mode='heuristic', review_queue_path=review_queue_path or None)

    def handle(self, form: dict[str, list[str]]) -> str:
        if form:
            action = _first(form, 'action', '')
            job_id = _first(form, 'job_id', '')
            notification_id = _first(form, 'notification_id', '')
            recovery_action = _first(form, 'recovery_action', '')
            file_path = _first(form, 'file_path', '')
            state = StudioState.from_form(form)
            try:
                state, outcome = self._run_action(
                    action,
                    state,
                    job_id=job_id,
                    notification_id=notification_id,
                    recovery_action=recovery_action,
                    file_path=file_path,
                )
            except Exception as exc:
                outcome = ActionOutcome(
                    flash=f'Action failed: {exc}',
                    flash_tone='error',
                    result_kind=self._last_outcome.result_kind,
                    result_payload=self._last_outcome.result_payload,
                )
            self._last_state = state
            self._last_outcome = outcome
        else:
            state = self._last_state
            outcome = self._last_outcome
        return render_page(
            state,
            outcome,
            self._job_snapshot(state.unified_output_dir),
            _notification_snapshot(state.unified_output_dir),
        )

    def _run_action(
        self,
        action: str,
        state: StudioState,
        *,
        job_id: str = '',
        notification_id: str = '',
        recovery_action: str = '',
        file_path: str = '',
    ) -> tuple[StudioState, ActionOutcome]:
        if action == 'load_ops_example':
            state = replace(
                state,
                ops_query=OPS_EXAMPLE['query'],
                ops_context=OPS_EXAMPLE['context'],
                ops_domain=OPS_EXAMPLE['domain'],
                ops_scenario=OPS_EXAMPLE['scenario'],
                unified_operating_domain=OPS_EXAMPLE['domain'],
            )
            return state, ActionOutcome(flash='Loaded the default context-reasoning example.', result_kind=self._last_outcome.result_kind, result_payload=self._last_outcome.result_payload)
        if action == 'load_vision_example':
            state = replace(
                state,
                vision_query=VISION_EXAMPLE['query'],
                vision_image=VISION_EXAMPLE['image_path'],
            )
            return state, ActionOutcome(flash='Loaded the default vision example.', result_kind=self._last_outcome.result_kind, result_payload=self._last_outcome.result_payload)
        if action == 'load_compare_example':
            left, right = _default_compare_paths(state)
            state = replace(state, compare_left_path=left, compare_right_path=right)
            return state, ActionOutcome(
                flash='Loaded compare defaults from the current output directory.',
                flash_tone='success',
                result_kind=self._last_outcome.result_kind,
                result_payload=self._last_outcome.result_payload,
            )
        if action == 'dismiss_notification':
            return self._dismiss_notification(notification_id, state)
        if action == 'mark_all_notifications_read':
            return self._mark_all_notifications_read(state)
        if action == 'cancel_job':
            return self._cancel_background_job(job_id, state)
        if action == 'retry_job':
            return self._retry_background_job(job_id, state)
        if action == 'load_job_result':
            return self._load_job_result(job_id, state)
        if action == 'load_output_file':
            return self._load_output_file(file_path, state)
        if action == 'run_recovery_action':
            return self._run_recovery_action(job_id, recovery_action, state)
        if action == 'run_artifact_compare':
            left, right = _default_compare_paths(state)
            state = replace(state, compare_left_path=left, compare_right_path=right)
            return state, self._run_artifact_compare(state)
        if action == 'run_ops':
            return state, self._run_ops(state)
        if action == 'run_vision':
            return state, self._run_vision(state)
        if action in self.LONG_ACTIONS:
            return self._enqueue_background_action(action, state)
        return state, ActionOutcome(result_kind=self._last_outcome.result_kind, result_payload=self._last_outcome.result_payload)

    def _enqueue_background_action(self, action: str, state: StudioState) -> tuple[StudioState, ActionOutcome]:
        label = self.ACTION_LABELS.get(action, action)
        with self._job_lock:
            self._job_counter += 1
            job_id = f"job-{time.strftime('%Y%m%d%H%M%S')}-{self._job_counter:04d}"
            job = BackgroundJob(job_id=job_id, action=action, label=label, state=state)
            self._append_job_event(job, f'{label} queued.')
            self._jobs[job_id] = job
            self._job_order.append(job_id)
            ahead = sum(1 for item in self._jobs.values() if item.status in {'queued', 'running', 'cancelling'}) - 1
        self._job_queue.put(job_id)
        queue_note = 'Starting now.' if ahead <= 0 else f'{ahead} job(s) ahead in the queue.'
        return state, ActionOutcome(
            flash=f'{label} queued in the background. {queue_note} This page refreshes automatically while it runs.',
            flash_tone='success',
            result_kind=self._last_outcome.result_kind,
            result_payload=self._last_outcome.result_payload,
        )

    def _job_worker(self) -> None:
        while True:
            job_id = self._job_queue.get()
            try:
                self._execute_job(job_id)
            finally:
                self._job_queue.task_done()

    def _execute_job(self, job_id: str) -> None:
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.status == 'cancelled':
                return
            if job.cancel_requested:
                job.status = 'cancelled'
                job.detail = 'Cancelled before the job started.'
                job.finished_at = time.time()
                job.progress = 0.0
                job.flash = job.detail
                job.flash_tone = 'neutral'
                self._append_job_event(job, job.detail)
                cancelled_job = job
            else:
                job.status = 'running'
                job.started_at = time.time()
                job.progress = 0.05
                job.detail = 'Starting background work.'
                self._append_job_event(job, job.detail)
                state = job.state
                action = job.action
                cancelled_job = None
        if cancelled_job is not None:
            self._persist_job_history(cancelled_job)
            self._emit_job_notification(cancelled_job)
            return
        try:
            if action == 'run_unified_training':
                outcome = self._run_unified_training(state, progress_callback=lambda p, d: self._update_job(job_id, progress=p, detail=d))
                final_state = state
            elif action == 'run_unified_benchmark_gate':
                outcome = self._run_unified_benchmark_gate(state, progress_callback=lambda p, d: self._update_job(job_id, progress=p, detail=d))
                final_state = state
            elif action == 'run_guided_learning':
                final_state, outcome = self._run_guided_learning(state, progress_callback=lambda p, d: self._update_job(job_id, progress=p, detail=d))
            elif action == 'run_beginner_autopilot':
                final_state, outcome = self._run_beginner_autopilot(state, progress_callback=lambda p, d: self._update_job(job_id, progress=p, detail=d))
            elif action == 'run_beginner_test':
                final_state, outcome = self._run_beginner_test(state, progress_callback=lambda p, d: self._update_job(job_id, progress=p, detail=d))
            elif action == 'run_generalization_proof':
                final_state, outcome = self._run_generalization_proof(state, progress_callback=lambda p, d: self._update_job(job_id, progress=p, detail=d))
            elif action == 'run_autopilot_coach':
                final_state, outcome = self._run_autopilot_coach(state, progress_callback=lambda p, d: self._update_job(job_id, progress=p, detail=d))
            elif action == 'run_understanding_eval':
                outcome = self._run_understanding_eval(state, progress_callback=lambda p, d: self._update_job(job_id, progress=p, detail=d))
                final_state = state
            else:
                raise ValueError(f'Unknown background action: {action}')
            self._complete_job(job_id, final_state, outcome)
        except JobCancelledError:
            self._mark_job_cancelled(job_id, 'Background job cancelled at the next safe checkpoint.')
        except Exception as exc:
            self._fail_job(job_id, exc)

    def _append_job_event(self, job: BackgroundJob, message: str) -> None:
        normalized = str(message).strip()
        if not normalized:
            return
        if job.events and job.events[-1].get('message') == normalized:
            return
        job.events.append({'at': time.time(), 'message': normalized})
        if len(job.events) > 40:
            del job.events[:-40]

    def _update_job(self, job_id: str, *, progress: float | None = None, detail: str | None = None) -> None:
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if progress is not None:
                job.progress = max(0.0, min(1.0, float(progress)))
            if detail is not None:
                job.detail = str(detail)
                self._append_job_event(job, job.detail)
            cancel_requested = job.cancel_requested
        if cancel_requested:
            raise JobCancelledError('Cancellation requested.')

    def _run_recovery_action(self, job_id: str, recovery_action: str, state: StudioState) -> tuple[StudioState, ActionOutcome]:
        record = self._find_job_record(job_id, state.unified_output_dir)
        if not record:
            return state, ActionOutcome(
                flash='No saved job record was found for that recovery action.',
                flash_tone='error',
                result_kind=self._last_outcome.result_kind,
                result_payload=self._last_outcome.result_payload,
            )
        job_state = _state_from_payload(record.get('state'), fallback=state)
        if recovery_action not in self.LONG_ACTIONS:
            return job_state, ActionOutcome(
                flash='That recovery action is not available.',
                flash_tone='error',
                result_kind=self._last_outcome.result_kind,
                result_payload=self._last_outcome.result_payload,
            )
        return self._enqueue_background_action(recovery_action, job_state)

    def _complete_job(self, job_id: str, state: StudioState, outcome: ActionOutcome) -> None:
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.cancel_requested:
                job.status = 'cancelled'
                job.progress = min(job.progress, 0.99)
                job.detail = 'Cancellation was requested. Any partially written artifacts were left on disk.'
                job.finished_at = time.time()
                job.flash = job.detail
                job.flash_tone = 'neutral'
                self._append_job_event(job, job.detail)
                cancelled_job = job
                completed_job = None
            else:
                job.status = 'completed'
                job.progress = 1.0
                job.detail = outcome.flash or 'Completed.'
                job.finished_at = time.time()
                job.result_kind = outcome.result_kind
                job.result_payload = outcome.result_payload
                job.flash = outcome.flash
                job.flash_tone = outcome.flash_tone
                self._append_job_event(job, job.detail)
                completed_job = job
                cancelled_job = None
        if cancelled_job is not None:
            self._persist_job_history(cancelled_job)
            self._emit_job_notification(cancelled_job)
            self._last_outcome = ActionOutcome(
                flash=cancelled_job.detail,
                flash_tone='neutral',
                result_kind=self._last_outcome.result_kind,
                result_payload=self._last_outcome.result_payload,
            )
            return
        if completed_job is not None:
            self._persist_job_history(completed_job)
            self._emit_job_notification(completed_job)
            self._last_state = state
            self._last_outcome = outcome

    def _mark_job_cancelled(self, job_id: str, detail: str) -> None:
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = 'cancelled'
            job.finished_at = time.time()
            job.detail = detail
            job.flash = detail
            job.flash_tone = 'neutral'
            self._append_job_event(job, detail)
        self._persist_job_history(job)
        self._emit_job_notification(job)
        self._last_outcome = ActionOutcome(
            flash=detail,
            flash_tone='neutral',
            result_kind=self._last_outcome.result_kind,
            result_payload=self._last_outcome.result_payload,
        )

    def _fail_job(self, job_id: str, exc: Exception) -> None:
        message = f'Background job failed: {exc}'
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = 'failed'
            job.finished_at = time.time()
            job.detail = message
            job.error_text = str(exc)
            job.flash = message
            job.flash_tone = 'error'
            self._append_job_event(job, message)
        self._persist_job_history(job)
        self._emit_job_notification(job)
        self._last_outcome = ActionOutcome(
            flash=message,
            flash_tone='error',
            result_kind=self._last_outcome.result_kind,
            result_payload=self._last_outcome.result_payload,
        )

    def _cancel_background_job(self, job_id: str, state: StudioState) -> tuple[StudioState, ActionOutcome]:
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return state, ActionOutcome(
                    flash='That job could not be found in the current queue or history.',
                    flash_tone='error',
                    result_kind=self._last_outcome.result_kind,
                    result_payload=self._last_outcome.result_payload,
                )
            job_state = job.state
            if job.status == 'queued':
                job.cancel_requested = True
                job.status = 'cancelled'
                job.detail = 'Cancelled before the job started.'
                job.finished_at = time.time()
                job.flash = job.detail
                job.flash_tone = 'neutral'
                self._append_job_event(job, job.detail)
                cancelled_now = True
                message = job.detail
            elif job.status in {'running', 'cancelling'}:
                job.cancel_requested = True
                job.status = 'cancelling'
                job.detail = 'Cancel requested. The current phase will stop at the next safe checkpoint.'
                self._append_job_event(job, job.detail)
                cancelled_now = False
                message = job.detail
            else:
                cancelled_now = False
                message = f"This job is already {job.status}."
        if cancelled_now:
            self._persist_job_history(job)
            self._emit_job_notification(job)
        return job_state, ActionOutcome(
            flash=message,
            flash_tone='success' if 'Cancel requested' in message or 'Cancelled' in message else 'neutral',
            result_kind=self._last_outcome.result_kind,
            result_payload=self._last_outcome.result_payload,
        )

    def _retry_background_job(self, job_id: str, state: StudioState) -> tuple[StudioState, ActionOutcome]:
        record = self._find_job_record(job_id, state.unified_output_dir)
        if not record:
            return state, ActionOutcome(
                flash='No saved job record was found for retry.',
                flash_tone='error',
                result_kind=self._last_outcome.result_kind,
                result_payload=self._last_outcome.result_payload,
            )
        action = str(record.get('action', '')).strip()
        job_state = _state_from_payload(record.get('state'), fallback=state)
        if action not in self.LONG_ACTIONS:
            return job_state, ActionOutcome(
                flash='This saved job cannot be retried automatically.',
                flash_tone='error',
                result_kind=self._last_outcome.result_kind,
                result_payload=self._last_outcome.result_payload,
            )
        return self._enqueue_background_action(action, job_state)

    def _load_job_result(self, job_id: str, state: StudioState) -> tuple[StudioState, ActionOutcome]:
        record = self._find_job_record(job_id, state.unified_output_dir)
        if not record:
            return state, ActionOutcome(
                flash='No saved job result was found.',
                flash_tone='error',
                result_kind=self._last_outcome.result_kind,
                result_payload=self._last_outcome.result_payload,
            )
        result_kind = str(record.get('result_kind', '')).strip()
        result_payload = record.get('result_payload')
        job_state = _state_from_payload(record.get('state'), fallback=state)
        if not result_kind or not isinstance(result_payload, dict):
            return job_state, ActionOutcome(
                flash='This job does not have a saved structured result yet.',
                flash_tone='error',
                result_kind=self._last_outcome.result_kind,
                result_payload=self._last_outcome.result_payload,
            )
        return job_state, ActionOutcome(
            flash=f"Loaded saved result for {record.get('label', 'background job')}.",
            flash_tone='success',
            result_kind=result_kind,
            result_payload=result_payload,
        )

    def _load_output_file(self, file_path: str, state: StudioState) -> tuple[StudioState, ActionOutcome]:
        preview = _preview_output_file(file_path)
        return self._last_state, ActionOutcome(
            flash=f"Loaded file preview for {preview.get('filename', 'output file')}.",
            flash_tone='success',
            result_kind='file_preview',
            result_payload=preview,
        )

    def _dismiss_notification(self, notification_id: str, state: StudioState) -> tuple[StudioState, ActionOutcome]:
        if notification_id:
            _mark_notification(state.unified_output_dir, notification_id, read=True)
        return state, ActionOutcome(
            flash='Notification marked as read.',
            flash_tone='success',
            result_kind=self._last_outcome.result_kind,
            result_payload=self._last_outcome.result_payload,
        )

    def _mark_all_notifications_read(self, state: StudioState) -> tuple[StudioState, ActionOutcome]:
        _mark_all_notifications(state.unified_output_dir, read=True)
        return state, ActionOutcome(
            flash='All notifications were marked as read.',
            flash_tone='success',
            result_kind=self._last_outcome.result_kind,
            result_payload=self._last_outcome.result_payload,
        )

    def _run_artifact_compare(self, state: StudioState) -> ActionOutcome:
        left = str(state.compare_left_path or '').strip()
        right = str(state.compare_right_path or '').strip()
        summary = _compare_paths(left, right)
        return ActionOutcome(
            flash='Artifact comparison finished.',
            flash_tone='success',
            result_kind='artifact_compare',
            result_payload=summary,
        )

    def _emit_job_notification(self, job: BackgroundJob) -> None:
        _upsert_notification(job.state.unified_output_dir, _build_notification(job))

    def _find_job_record(self, job_id: str, output_dir: str) -> dict[str, Any] | None:
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return self._history_record(job)
        for entry in _read_job_history(output_dir):
            if str(entry.get('job_id', '')) == job_id:
                return entry
        return None

    def _history_record(self, job: BackgroundJob) -> dict[str, Any]:
        gate_payload = job.result_payload.get('gate') if isinstance(job.result_payload, dict) else {}
        gate_accepted = gate_payload.get('accepted') if isinstance(gate_payload, dict) else None
        return {
            'job_id': job.job_id,
            'label': job.label,
            'action': job.action,
            'status': job.status,
            'detail': job.detail,
            'progress': job.progress,
            'created_at': job.created_at,
            'started_at': job.started_at,
            'finished_at': job.finished_at,
            'result_kind': job.result_kind,
            'result_payload': job.result_payload,
            'flash': job.flash,
            'flash_tone': job.flash_tone,
            'error_text': job.error_text,
            'events': job.events,
            'gate_accepted': gate_accepted,
            'state': _state_to_payload(job.state),
            'output_dir': job.state.unified_output_dir,
        }

    def _persist_job_history(self, job: BackgroundJob) -> None:
        output_dir = job.state.unified_output_dir
        history = _read_job_history(output_dir)
        record = self._history_record(job)
        history = [item for item in history if str(item.get('job_id', '')) != job.job_id]
        history.insert(0, record)
        _write_job_history(output_dir, history)

    @staticmethod
    def _serialize_job(job: BackgroundJob) -> dict[str, Any]:
        end_time = job.finished_at or time.time()
        baseline = job.started_at or job.created_at
        status = job.status
        gate_payload = job.result_payload.get('gate') if isinstance(job.result_payload, dict) else {}
        gate_accepted = gate_payload.get('accepted') if isinstance(gate_payload, dict) else None
        return {
            'job_id': job.job_id,
            'label': job.label,
            'status': status,
            'detail': job.detail,
            'progress': job.progress,
            'elapsed_seconds': max(0.0, end_time - baseline),
            'result_kind': job.result_kind,
            'action': job.action,
            'time_hint': _action_time_hint(job.action),
            'remaining_eta': _remaining_eta_label(job.action, job.progress),
            'can_cancel': status in {'queued', 'running', 'cancelling'},
            'can_retry': status in {'completed', 'failed', 'cancelled'} and bool(job.action),
            'can_load': bool(job.result_kind and isinstance(job.result_payload, dict)),
            'persisted': False,
            'output_dir': job.state.unified_output_dir,
            'events': job.events[-8:],
            'gate_accepted': gate_accepted,
        }

    def _job_snapshot(self, output_dir: str) -> dict[str, Any]:
        with self._job_lock:
            jobs = [self._jobs[job_id] for job_id in self._job_order]
        running = [job for job in jobs if job.status in {'running', 'cancelling'}]
        queued = [job for job in jobs if job.status == 'queued']
        completed = [job for job in jobs if job.status == 'completed']
        failed = [job for job in jobs if job.status == 'failed']
        active = running[0] if running else (queued[0] if queued else None)
        recent: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for job in reversed(jobs[-8:]):
            snapshot = self._serialize_job(job)
            recent.append(snapshot)
            seen_ids.add(snapshot['job_id'])
        history_entries = _read_job_history(output_dir)
        for entry in history_entries:
            snapshot = _history_to_snapshot(entry)
            job_id = str(snapshot.get('job_id', ''))
            if not job_id or job_id in seen_ids:
                continue
            recent.append(snapshot)
            seen_ids.add(job_id)
            if len(recent) >= 8:
                break
        return {
            'has_active': active is not None,
            'running_count': len(running),
            'queued_count': len(queued),
            'completed_count': len(completed),
            'failed_count': len(failed),
            'saved_history_count': len(history_entries),
            'history_path': str(_job_history_path(output_dir)),
            'active': self._serialize_job(active) if active is not None else {},
            'recent': recent,
        }

    def _run_ops(self, state: StudioState) -> ActionOutcome:
        request = CopilotRequest(
            query=state.ops_query,
            context=state.ops_context,
            domain=state.ops_domain,
            scenario=state.ops_scenario,
        )
        result = self._copilot(state.unified_review_queue_path).run(request)
        return ActionOutcome(
            flash='Context reasoning finished.',
            flash_tone='success',
            result_kind='ops',
            result_payload=result.model_dump(),
        )

    @staticmethod
    def _run_vision(state: StudioState) -> ActionOutcome:
        payload = build_vision_payload(
            state.vision_query,
            state.vision_image,
            state.vision_mode,
            state.vision_answer_mode,
            state.vision_concept_store,
            state.vision_operator_store,
            state.vision_weights,
        )
        return ActionOutcome(
            flash='Vision-grounded reasoning finished.',
            flash_tone='success',
            result_kind='vision',
            result_payload=payload,
        )

    @staticmethod
    def _run_unified_training(state: StudioState, progress_callback: Any = None) -> ActionOutcome:
        if progress_callback:
            progress_callback(0.15, 'Scanning the corpus store and approved review traces.')
        summary = UnifiedSemOpTrainer().train_from_store(
            state.unified_store_path,
            state.unified_output_dir,
            source=state.unified_source or None,
            split=state.unified_split,
            review_store_path=state.unified_review_queue_path or None,
            approved_queries_only=state.unified_approved_queries_only,
            operating_domain=_resolve_operating_domain(state),
        )
        if progress_callback:
            progress_callback(0.92, 'Artifact bundle exported. Finalizing the result card.')
        return ActionOutcome(
            flash='Unified trainer finished and exported a fresh artifact bundle.',
            flash_tone='success',
            result_kind='unified_training',
            result_payload=summary.model_dump(),
        )

    @staticmethod
    def _run_unified_benchmark_gate(state: StudioState, progress_callback: Any = None) -> ActionOutcome:
        if progress_callback:
            progress_callback(0.12, 'Loading seed graphs from the selected corpus store.')
        graphs = _load_store_graphs(state.unified_store_path, state.unified_source, state.unified_split)
        if not graphs:
            raise FileNotFoundError('No graphs were found in the selected store and split.')
        if progress_callback:
            progress_callback(0.3, 'Deriving starter analogy, grounding, and repair benchmark cases.')
        summary = BenchmarkGatedContinuousTrainer().train_evaluate_and_gate(
            state.unified_store_path,
            state.unified_output_dir,
            source=state.unified_source or None,
            split=state.unified_split,
            review_store_path=state.unified_review_queue_path or None,
            approved_queries_only=state.unified_approved_queries_only,
            hidden_premise_cases=_load_hidden_premise_cases(state.understanding_hidden_input),
            transfer_cases=_load_operator_transfer_cases(state.unified_transfer_input),
            analogy_cases=_derive_starter_analogy_cases(graphs),
            grounding_cases=_derive_starter_grounding_cases(graphs),
            compiler_cases=_derive_starter_compiler_cases(graphs),
            vlso_cases=_load_vlso_cases(state.understanding_vlso_input),
            baseline_summary_path=state.unified_baseline_summary_path or None,
            operating_domain=_resolve_operating_domain(state),
            benchmark_corpus_path=state.unified_benchmark_corpus_path or None,
        )
        if progress_callback:
            progress_callback(0.94, 'Benchmark gate summary written. Preparing the result card.')
        return ActionOutcome(
            flash=f'Unified trainer and benchmark gate finished on {len(graphs)} seed graphs.',
            flash_tone='success',
            result_kind='unified_benchmark_gate',
            result_payload=summary.model_dump(),
        )

    def _run_guided_learning(self, state: StudioState, progress_callback: Any = None) -> tuple[StudioState, ActionOutcome]:
        source_used = (state.unified_source or GUIDED_BOOTSTRAP_SOURCE).strip() or GUIDED_BOOTSTRAP_SOURCE
        seeded_state = replace(
            state,
            unified_source=source_used,
            unified_split='train',
            unified_approved_queries_only=True,
            unified_operating_domain=_resolve_operating_domain(state),
        )
        if progress_callback:
            progress_callback(0.15, 'Generating starter reasoning traces and approving grounded reviews.')
        guided_summary = self._seed_guided_reviews(
            seeded_state,
            source_used,
            resolution_note='SemOp Studio guided starter trace.',
        )
        if progress_callback:
            progress_callback(0.4, 'Loading starter graphs and building gate inputs.')
        graphs = _load_store_graphs(seeded_state.unified_store_path, seeded_state.unified_source, seeded_state.unified_split)
        if progress_callback:
            progress_callback(0.62, 'Training the artifact bundle and rerunning the benchmark gate.')
        summary = BenchmarkGatedContinuousTrainer().train_evaluate_and_gate(
            seeded_state.unified_store_path,
            seeded_state.unified_output_dir,
            source=seeded_state.unified_source or None,
            split=seeded_state.unified_split,
            review_store_path=seeded_state.unified_review_queue_path or None,
            approved_queries_only=seeded_state.unified_approved_queries_only,
            hidden_premise_cases=_load_hidden_premise_cases(seeded_state.understanding_hidden_input),
            transfer_cases=_load_operator_transfer_cases(seeded_state.unified_transfer_input),
            analogy_cases=_derive_starter_analogy_cases(graphs),
            grounding_cases=_derive_starter_grounding_cases(graphs),
            compiler_cases=_derive_starter_compiler_cases(graphs),
            vlso_cases=_load_vlso_cases(seeded_state.understanding_vlso_input),
            baseline_summary_path=seeded_state.unified_baseline_summary_path or None,
            operating_domain=_resolve_operating_domain(seeded_state),
            benchmark_corpus_path=seeded_state.unified_benchmark_corpus_path or None,
        )
        if progress_callback:
            progress_callback(0.94, 'Guided starter loop finished. Preparing the report.')
        payload = summary.model_dump()
        payload['guided_bootstrap'] = {
            'source_used': source_used,
            'split_used': seeded_state.unified_split,
            **guided_summary,
        }
        return seeded_state, ActionOutcome(
            flash=(
                f"Guided starter loop seeded {guided_summary.get('guided_query_count', 0)} starter traces "
                f"and approved {guided_summary.get('approved_review_count', 0)} reviews before rerunning the gate."
            ),
            flash_tone='success',
            result_kind='guided_learning',
            result_payload=payload,
        )

    def _seed_guided_reviews(
        self,
        state: StudioState,
        source_used: str,
        *,
        allow_duplicate_reviews: bool = False,
        resolution_note: str,
        requests: list[CopilotRequest] | None = None,
    ) -> dict[str, Any]:
        store = CorpusMemoryStore(state.unified_store_path)
        review_store = ReviewQueueStore(state.unified_review_queue_path)
        existing_approved = {
            (item.domain, item.scenario, item.query)
            for item in review_store.fetch_items(status='approved', limit=500)
        }
        seeded_queries: list[str] = []
        approved_items = 0
        request_list = requests or _guided_bootstrap_requests(state)
        for request in request_list:
            result = self._copilot(None).run(request)
            _persist_graph(store, result.graph, source_used, 'train')
            seeded_queries.append(request.query)
            reasons = _bootstrap_review_reasons(result.graph, result.kpis.model_dump())
            severity = infer_review_severity(request.domain, request.scenario, reasons, result.kpis.model_dump())
            corrected_answer = _grounded_review_answer(request.context, result.answer_text)
            key = (request.domain, request.scenario, request.query)
            if key in existing_approved and not allow_duplicate_reviews:
                continue
            item_id = review_store.enqueue(
                domain=request.domain,
                scenario=request.scenario,
                query=request.query,
                reasons=reasons,
                answer_text=corrected_answer,
                kpis=result.kpis.model_dump(),
                audit_items=[{'stage': item.stage, 'detail': item.detail} for item in result.audit_items],
                context_text=request.context,
                graph_payload=result.graph.model_dump(),
                severity=severity,
            )
            review_store.update_status(item_id, 'approved', resolution_note)
            existing_approved.add(key)
            approved_items += 1
        snapshot = _review_queue_snapshot(state.unified_review_queue_path, _resolve_operating_domain(state))
        return {
            'guided_queries': _unique_texts(seeded_queries),
            'guided_query_count': len(_unique_texts(seeded_queries)),
            'approved_review_count': approved_items,
            'promotable_review_count': snapshot.get('promotable', 0),
            'approved_total': snapshot.get('approved', 0),
        }

    def _run_beginner_autopilot(self, state: StudioState, progress_callback: Any = None) -> tuple[StudioState, ActionOutcome]:
        source_used = (state.unified_source or BEGINNER_AUTOPILOT_SOURCE).strip() or BEGINNER_AUTOPILOT_SOURCE
        seeded_state = replace(
            state,
            unified_source=source_used,
            unified_split='train',
            unified_approved_queries_only=True,
            unified_operating_domain=_resolve_operating_domain(state),
        )
        if progress_callback:
            progress_callback(0.08, 'Seeding built-in hidden-premise, transfer, and starter vision examples.')
        setup_summary = _seed_builtin_corpus(seeded_state, source_used, 'train')
        if progress_callback:
            progress_callback(0.24, 'Creating grounded starter reviews and approving them automatically.')
        guided_summary = self._seed_guided_reviews(
            seeded_state,
            source_used,
            allow_duplicate_reviews=True,
            resolution_note='SemOp Studio one-click beginner trace.',
        )
        if progress_callback:
            progress_callback(0.46, 'Training the unified artifact bundle and running the benchmark gate.')
        gate_outcome = self._run_unified_benchmark_gate(
            seeded_state,
            progress_callback=(lambda p, d: progress_callback(min(0.78, 0.46 + (float(p) * 0.32)), d)) if progress_callback else None,
        )
        payload = dict(gate_outcome.result_payload or {})
        payload['autopilot_setup'] = {
            **setup_summary,
            **guided_summary,
            'source_used': source_used,
            'split_used': seeded_state.unified_split,
        }
        if progress_callback:
            progress_callback(0.82, 'Running the overall understanding benchmark for the fresh bundle.')
        payload['understanding'] = self._run_understanding_eval(
            seeded_state,
            progress_callback=(lambda p, d: progress_callback(min(0.9, 0.82 + (float(p) * 0.08)), d)) if progress_callback else None,
        ).result_payload
        if progress_callback:
            progress_callback(0.92, 'Running context and vision smoke tests for the beginner report.')
        payload['ops'] = self._run_ops(seeded_state).result_payload
        try:
            payload['vision'] = self._run_vision(seeded_state).result_payload
        except Exception as exc:
            payload['vision'] = {'error': str(exc)}
        return seeded_state, ActionOutcome(
            flash=(
                f"One-click beginner flow seeded {payload['autopilot_setup'].get('total_seeded', 0)} graphs, "
                f"approved {payload['autopilot_setup'].get('approved_review_count', 0)} traces, and ran training plus tests."
            ),
            flash_tone='success',
            result_kind='beginner_autopilot',
            result_payload=payload,
        )

    def _run_beginner_test(self, state: StudioState, progress_callback: Any = None) -> tuple[StudioState, ActionOutcome]:
        test_state = replace(
            state,
            unified_source=(state.unified_source or BEGINNER_AUTOPILOT_SOURCE).strip() or BEGINNER_AUTOPILOT_SOURCE,
            unified_split='train',
            unified_approved_queries_only=True,
            unified_operating_domain=_resolve_operating_domain(state),
        )
        if progress_callback:
            progress_callback(0.12, 'Checking whether a beginner-ready bundle already exists for this source.')
        autopilot_ran = False
        source_graphs = _load_store_graphs(test_state.unified_store_path, test_state.unified_source, test_state.unified_split)
        if not source_graphs:
            if progress_callback:
                progress_callback(0.2, 'No beginner bundle exists yet. Running one-click setup first.')
            test_state, outcome = self._run_beginner_autopilot(
                test_state,
                progress_callback=(lambda p, d: progress_callback(min(0.72, 0.2 + (float(p) * 0.52)), d)) if progress_callback else None,
            )
            payload = dict(outcome.result_payload or {})
            autopilot_ran = True
        else:
            payload = _load_gate_payload(test_state.unified_output_dir)
            if not payload or not isinstance(payload.get('benchmark'), dict) or not isinstance(payload.get('gate'), dict):
                if progress_callback:
                    progress_callback(0.32, 'Refreshing the benchmark gate because no cached decision was found.')
                gate_outcome = self._run_unified_benchmark_gate(
                    test_state,
                    progress_callback=(lambda p, d: progress_callback(min(0.68, 0.32 + (float(p) * 0.36)), d)) if progress_callback else None,
                )
                payload = dict(gate_outcome.result_payload or {})
            payload.setdefault(
                'autopilot_setup',
                {
                    'source_used': test_state.unified_source,
                    'split_used': test_state.unified_split,
                    'total_seeded': len(source_graphs),
                    'approved_review_count': _review_queue_snapshot(
                        test_state.unified_review_queue_path,
                        _resolve_operating_domain(test_state),
                    ).get('approved', 0),
                },
            )
        if progress_callback:
            progress_callback(0.76, 'Running context, understanding, and vision smoke tests.')
        payload['ops'] = self._run_ops(test_state).result_payload
        payload['understanding'] = self._run_understanding_eval(
            test_state,
            progress_callback=(lambda p, d: progress_callback(min(0.9, 0.76 + (float(p) * 0.14)), d)) if progress_callback else None,
        ).result_payload
        try:
            payload['vision'] = self._run_vision(test_state).result_payload
        except Exception as exc:
            payload['vision'] = {'error': str(exc)}
        payload['autopilot_ran'] = autopilot_ran
        return test_state, ActionOutcome(
            flash='One-click beginner test ran the current bundle and refreshed the starter diagnostics.',
            flash_tone='success',
            result_kind='beginner_test',
            result_payload=payload,
        )

    def _run_generalization_proof_loop(
        self,
        state: StudioState,
        *,
        source_prefix: str,
        progress_callback: Any = None,
    ) -> tuple[StudioState, ActionOutcome]:
        proof_state = replace(
            state,
            unified_approved_queries_only=True,
            unified_operating_domain=_resolve_operating_domain(state),
        )
        curriculum_rounds: list[dict[str, Any]] = []
        source_schedule: list[str] = []
        all_graphs: list[StructuredMeaningGraph] = []
        for round_index in range(AUTOPILOT_PROOF_ROUNDS):
            if progress_callback:
                progress_callback(0.08 + (round_index * 0.12), f'Seeding proof curriculum round {round_index + 1}/{AUTOPILOT_PROOF_ROUNDS}.')
            source_used = f'{source_prefix}_round_{round_index + 1:02d}'
            spec = _proof_round_spec(proof_state, round_index)
            guided_summary = self._seed_guided_reviews(
                proof_state,
                source_used,
                allow_duplicate_reviews=True,
                resolution_note=f'SemOp Studio proof round {round_index + 1}.',
                requests=_proof_round_requests(proof_state, round_index),
            )
            visual_summary = _seed_visual_curriculum(
                proof_state,
                source_used,
                spec['visual_queries'],
                split=proof_state.unified_split,
            )
            curriculum_rounds.append({
                'round_index': round_index + 1,
                'source_used': source_used,
                'domain': spec['domain'],
                'scenario': spec['scenario'],
                'approved_review_count': guided_summary.get('approved_review_count', 0),
                'promotable_review_count': guided_summary.get('promotable_review_count', 0),
                'visual_seeded': visual_summary.get('visual_seeded', 0),
                'visual_skipped': visual_summary.get('visual_skipped', 0),
                'visual_queries': visual_summary.get('visual_queries', []),
                'errors': visual_summary.get('errors', []),
            })
            source_schedule.append(source_used)
            all_graphs.extend(_load_store_graphs(proof_state.unified_store_path, source_used, proof_state.unified_split))
        if progress_callback:
            progress_callback(0.5, 'Running repeated benchmark-gated proof rounds.')
        proof = GeneralizationProofHarness().run(
            proof_state.unified_store_path,
            proof_state.unified_output_dir,
            source_schedule=source_schedule,
            split=proof_state.unified_split,
            rounds=len(source_schedule),
            review_store_path=proof_state.unified_review_queue_path or None,
            approved_queries_only=True,
            hidden_premise_cases=_load_hidden_premise_cases(proof_state.understanding_hidden_input),
            transfer_cases=_load_operator_transfer_cases(proof_state.unified_transfer_input),
            analogy_cases=_derive_starter_analogy_cases(all_graphs),
            grounding_cases=_derive_starter_grounding_cases(all_graphs),
            compiler_cases=_derive_starter_compiler_cases(all_graphs),
            vlso_cases=_load_vlso_cases(proof_state.understanding_vlso_input),
            vlso_real_image_cases=_load_vlso_cases(proof_state.understanding_vlso_real_input),
            operating_domain=_resolve_operating_domain(proof_state),
            benchmark_corpus_path=proof_state.unified_benchmark_corpus_path or None,
            progress_callback=(lambda p, d: progress_callback(min(0.95, 0.5 + (float(p) * 0.42)), d)) if progress_callback else None,
        )
        proof_payload = proof.model_dump()
        last_round = proof_payload.get('rounds', [])[-1] if proof_payload.get('rounds') else {}
        payload = {
            'proof': proof_payload,
            'curriculum_rounds': curriculum_rounds,
            'proof_report_path': proof_payload.get('report_path', ''),
            'gate': last_round.get('gate', {}) if isinstance(last_round, dict) else {},
            'benchmark': last_round.get('benchmark', {}) if isinstance(last_round, dict) else {},
            'understanding': last_round.get('understanding', {}) if isinstance(last_round, dict) else {},
            'autopilot_setup': {
                'source_prefix': source_prefix,
                'approved_review_count': sum(int(item.get('approved_review_count', 0) or 0) for item in curriculum_rounds),
                'visual_seeded': sum(int(item.get('visual_seeded', 0) or 0) for item in curriculum_rounds),
                'total_seeded': sum(int(item.get('approved_review_count', 0) or 0) + int(item.get('visual_seeded', 0) or 0) for item in curriculum_rounds),
            },
        }
        evidence = proof_payload.get('evidence', {}) if isinstance(proof_payload.get('evidence'), dict) else {}
        flash = (
            f"Generalization proof finished. Strong model score: {evidence.get('strong_model_score', 0.0)}"
            if proof_payload.get('rounds')
            else 'Generalization proof finished.'
        )
        return proof_state, ActionOutcome(
            flash=flash,
            flash_tone='success',
            result_kind='generalization_proof',
            result_payload=payload,
        )

    def _run_generalization_proof(self, state: StudioState, progress_callback: Any = None) -> tuple[StudioState, ActionOutcome]:
        if not _load_store_graphs(state.unified_store_path, state.unified_source, state.unified_split):
            return self._run_autopilot_coach(state, progress_callback=progress_callback)
        source_prefix = (state.unified_source or 'autopilot_proof').strip() or 'autopilot_proof'
        return self._run_generalization_proof_loop(state, source_prefix=source_prefix, progress_callback=progress_callback)

    def _run_autopilot_coach(self, state: StudioState, progress_callback: Any = None) -> tuple[StudioState, ActionOutcome]:
        coach_state = replace(
            state,
            unified_approved_queries_only=True,
            unified_operating_domain=_resolve_operating_domain(state),
        )
        if progress_callback:
            progress_callback(0.04, 'Running beginner setup before the proof coach.')
        coach_state, beginner_outcome = self._run_beginner_autopilot(
            coach_state,
            progress_callback=(lambda p, d: progress_callback(min(0.48, 0.04 + (float(p) * 0.44)), d)) if progress_callback else None,
        )
        if progress_callback:
            progress_callback(0.5, 'Beginner setup finished. Starting the proof coach rounds.')
        proof_state, proof_outcome = self._run_generalization_proof_loop(
            coach_state,
            source_prefix='autopilot_coach',
            progress_callback=(lambda p, d: progress_callback(min(0.97, 0.5 + (float(p) * 0.47)), d)) if progress_callback else None,
        )
        payload = dict(proof_outcome.result_payload or {})
        payload['beginner_autopilot'] = beginner_outcome.result_payload
        payload['autopilot_setup'] = {
            **(beginner_outcome.result_payload.get('autopilot_setup', {}) if isinstance(beginner_outcome.result_payload, dict) else {}),
            **(payload.get('autopilot_setup', {}) if isinstance(payload.get('autopilot_setup'), dict) else {}),
        }
        evidence = (payload.get('proof', {}) if isinstance(payload.get('proof'), dict) else {}).get('evidence', {})
        flash = (
            f"Autopilot coach finished. Strong model score: {evidence.get('strong_model_score', 0.0)}"
            if isinstance(evidence, dict)
            else 'Autopilot coach finished.'
        )
        return proof_state, ActionOutcome(
            flash=flash,
            flash_tone='success',
            result_kind='autopilot_coach',
            result_payload=payload,
        )
    @staticmethod
    def _run_understanding_eval(state: StudioState, progress_callback: Any = None) -> ActionOutcome:
        if progress_callback:
            progress_callback(0.18, 'Loading hidden-premise and VLSO evaluation files.')
        summary = SemOpUnderstandingEvaluator().evaluate(
            hidden_premise_cases=_load_hidden_premise_cases(state.understanding_hidden_input),
            cp_examples=None,
            cp_hidden_examples=None,
            vlso_cases=_load_vlso_cases(state.understanding_vlso_input) or None,
            vlso_real_image_cases=_load_vlso_cases(state.understanding_vlso_real_input) or None,
            cp_mode='heuristic',
        )
        if progress_callback:
            progress_callback(0.94, 'Overall understanding benchmark finished. Packaging the score summary.')
        return ActionOutcome(
            flash='Overall understanding benchmark finished.',
            flash_tone='success',
            result_kind='understanding_eval',
            result_payload=summary.model_dump(),
        )

def render_page(
    state: StudioState,
    outcome: ActionOutcome,
    job_snapshot: dict[str, object],
    notification_snapshot: dict[str, object],
) -> str:
    flash_html = ''
    if outcome.flash:
        flash_html = f"<div class='flash {html.escape(outcome.flash_tone)}'>{html.escape(outcome.flash)}</div>"
    result_html = render_result(outcome.result_kind, outcome.result_payload)
    snapshot_html = render_workspace_snapshot(state, outcome.result_kind)
    spotlight_html = render_result_spotlight(outcome.result_kind, outcome.result_payload)
    job_panel_html = _render_job_queue_panel(job_snapshot)
    notification_panel_html = _render_notification_panel(notification_snapshot)
    refresh_html = "<meta http-equiv='refresh' content='2'>" if job_snapshot.get('has_active') else ''
    unread_count = int(notification_snapshot.get('unread_count', 0) or 0)
    title_prefix = f'({unread_count}) ' if unread_count else ''
    vision_preview = _local_image_data_uri(state.vision_image)
    vision_preview_html = "<div class='empty'>Set a local image path to preview it here.</div>"
    if vision_preview:
        vision_preview_html = f"<img class='hero-preview' src='{vision_preview}' alt='vision preview'>"
    return f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh_html}
<title>{html.escape(title_prefix)}SemOp Studio</title>
<style>
:root {{ --bg:#f3eee4; --card:rgba(255,252,246,.92); --line:#d4d8cf; --ink:#17241d; --muted:#58685f; --accent:#245842; --soft:#eef5ef; --ok:#266747; --warn:#a5542d; --shadow:0 18px 44px rgba(23,36,29,.09); }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; color:var(--ink); background:radial-gradient(circle at top left, rgba(212,134,46,.18), transparent 24%), radial-gradient(circle at top right, rgba(36,88,66,.14), transparent 28%), linear-gradient(180deg,#f8f1e6 0%,#edf4ef 100%); font-family:"Aptos","Segoe UI Variable","Segoe UI","Malgun Gothic",sans-serif; }}
main {{ max-width:1360px; margin:0 auto; padding:24px 18px 48px; }}
a {{ color:inherit; text-decoration:none; }}
h1 {{ margin:0 0 10px; font-size:38px; line-height:1.08; letter-spacing:-.04em; }}
h2 {{ margin:0; font-size:22px; letter-spacing:-.02em; }}
h3 {{ margin:0 0 10px; font-size:17px; }}
p,li,label,small {{ color:var(--muted); line-height:1.6; }}
.hero, .grid, .dashboard-grid, .mini-grid {{ display:grid; gap:18px; }}
.hero {{ grid-template-columns:1.25fr .95fr; margin-bottom:20px; }}
.dashboard-grid {{ grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); margin-bottom:20px; }}
.grid {{ grid-template-columns:1fr 1fr; }}
.mini-grid {{ grid-template-columns:1fr 1fr; }}
.card, .lane-card, .spotlight-card {{ background:var(--card); border:1px solid rgba(113,138,123,.24); border-radius:24px; box-shadow:var(--shadow); }}
.card {{ padding:22px; }}
.lane-card {{ padding:18px; }}
.spotlight-card {{ padding:22px; background:linear-gradient(145deg, rgba(255,250,243,.98), rgba(239,247,240,.95)); }}
.eyebrow {{ display:inline-block; margin-bottom:10px; letter-spacing:.12em; text-transform:uppercase; color:var(--accent); font-weight:700; font-size:11px; }}
.hero p {{ margin:0 0 16px; font-size:16px; }}
.hero-actions, .actions, .nav-row {{ display:flex; gap:10px; flex-wrap:wrap; }}
.nav-pill, button {{ border:0; border-radius:999px; padding:11px 14px; font:inherit; cursor:pointer; }}
.nav-pill {{ background:var(--soft); color:var(--accent); font-weight:700; }}
button.primary {{ background:var(--accent); color:#fff; font-weight:700; }}
button.secondary {{ background:var(--soft); color:var(--ink); font-weight:700; }}
textarea, input, select {{ width:100%; padding:12px 14px; border:1px solid var(--line); border-radius:16px; font:inherit; background:#fff; }}
textarea {{ min-height:120px; resize:vertical; }}
.flash {{ margin-bottom:16px; padding:13px 15px; border-radius:16px; background:#edf4f8; color:var(--accent); }}
.flash.success {{ background:#ebf8ef; color:var(--ok); }}
.flash.error {{ background:#fff1eb; color:var(--warn); }}
.summary-grid {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); margin:14px 0; }}
.metric-card {{ background:#f6f8f4; border:1px solid var(--line); border-radius:16px; padding:12px; }}
.metric-card small {{ display:block; margin-bottom:6px; }}
.metric-card strong {{ display:block; font-size:16px; line-height:1.35; word-break:break-word; }}
.info-block {{ background:#fbfcfa; border:1px solid var(--line); border-radius:16px; padding:12px 14px; margin-top:10px; }}
.info-block small {{ display:block; margin-bottom:6px; }}
.section-head {{ display:flex; justify-content:space-between; gap:18px; align-items:end; margin-bottom:14px; }}
.section-head p {{ margin:6px 0 0; }}
.lane-top p {{ margin:6px 0 0; }}
.hero-preview-wrap {{ aspect-ratio:4 / 3; border-radius:20px; overflow:hidden; background:#e8eee8; border:1px solid var(--line); }}
.hero-preview {{ width:100%; height:100%; object-fit:cover; display:block; }}
.spotlight-list {{ margin:12px 0 0; padding-left:18px; }}
.spotlight-list li {{ margin-bottom:6px; }}
.raw-json summary {{ cursor:pointer; color:var(--accent); font-weight:700; margin-bottom:10px; }}
pre {{ background:#14211c; color:#eff7f0; border-radius:18px; padding:16px; overflow:auto; white-space:pre-wrap; }}
.empty {{ border:1px dashed var(--line); border-radius:16px; padding:20px; color:var(--muted); }}
.inline {{ display:flex; align-items:center; gap:8px; margin-top:12px; }}
.inline input {{ width:auto; }}
.status-chip {{ display:inline-flex; align-items:center; justify-content:center; padding:4px 10px; border-radius:999px; background:var(--soft); color:var(--accent); font-size:12px; font-weight:700; text-transform:capitalize; }}
.job-progress {{ margin-top:10px; height:10px; border-radius:999px; background:#dfe8df; overflow:hidden; }}
.job-progress span {{ display:block; height:100%; background:linear-gradient(90deg, #245842, #d4862e); border-radius:999px; }}
.job-row {{ border:1px solid var(--line); border-radius:14px; padding:10px 12px; background:#f7faf6; margin-top:10px; }}
.job-row-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; }}
.job-meta {{ margin-top:8px; color:var(--muted); font-size:13px; }}
.job-list {{ margin-top:8px; }}
.job-actions {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }}
.job-recovery {{ margin-top:10px; padding-top:10px; border-top:1px dashed var(--line); }}
.job-recovery small {{ display:block; margin-bottom:6px; }}
.job-log {{ margin-top:10px; }}
.job-log summary {{ cursor:pointer; color:var(--accent); font-weight:700; }}
.job-log-line {{ display:flex; gap:10px; align-items:flex-start; margin-top:8px; font-size:13px; }}
.job-log-line span {{ color:var(--muted); min-width:56px; }}
.job-log-line code {{ white-space:pre-wrap; background:#f2f6f1; border-radius:10px; padding:4px 8px; color:var(--ink); flex:1; }}
.notification-row {{ border:1px solid var(--line); border-radius:14px; padding:10px 12px; background:#f7faf6; margin-top:10px; }}
.notification-row.unread {{ border-color:#d4862e; background:#fff8ef; }}
.notification-row.error {{ background:#fff3ec; }}
.notification-row.success {{ background:#eef8f0; }}
.notification-row.neutral {{ background:#f5f7f5; }}
.mini-form {{ margin:0; }}
button.compact {{ padding:8px 12px; font-size:13px; }}
@media (max-width:1080px) {{ .hero, .grid, .mini-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<main>
  <section class="hero">
    <div class="card">
      <small class="eyebrow">SemOp Studio</small>
      <h1>Train, inspect, and gate the operator-algebra stack from one friendly local UI.</h1>
      <p>This studio is the product-facing entry point for context reasoning, visual grounding, unified artifact training, and benchmark-gated learning. Keep the older GUIs for power-user data operations; stay here for the main loop.</p>
      <div class="hero-actions">
        <a class="nav-pill" href="#reasoning-lab">Try reasoning</a>
        <a class="nav-pill" href="#autopilot-lab">Do everything</a>
        <a class="nav-pill" href="#beginner-lab">One-click mode</a>
        <a class="nav-pill" href="#training-lab">Train artifacts</a>
        <a class="nav-pill" href="#benchmark-lab">Benchmark gate</a>
        <a class="nav-pill" href="#compare-lab">Compare outputs</a>
        <a class="nav-pill" href="#notification-panel">Alerts</a>
        <a class="nav-pill" href="#job-panel">Live jobs</a>
        <a class="nav-pill" href="#result-panel">See results</a>
      </div>
    </div>
    <div class="card">
      <small class="eyebrow">Starter flow</small>
      <h2>Use the same order every time.</h2>
      <ol>
        <li>For the easiest path, press Do everything for me once.</li>
        <li>Watch the Live jobs card while long tasks run in the background.</li>
        <li>If the gate is blocked at 0.0, use Guided starter loop once.</li>
        <li>Inspect the diagnosis and keep only accepted bundles.</li>
      </ol>
      <div class="hero-preview-wrap">{vision_preview_html}</div>
    </div>
  </section>
  {flash_html}
  <section class="dashboard-grid">{snapshot_html}</section>
  <section style="margin-bottom:20px;">{spotlight_html}</section>
  {notification_panel_html}
  {job_panel_html}
  {_render_autopilot_coach_section(state)}
  {_render_beginner_section(state)}
  {_render_reasoning_section(state)}
  {_render_training_section(state)}
  {_render_benchmark_section(state)}
  {_render_compare_section(state)}
  {_render_notes_section()}
  <section class="card" id="result-panel" style="margin-top:20px;"><div class="section-head"><div><small class="eyebrow">Result console</small><h2>Latest run</h2><p>Every action returns a compact summary first and the full structured payload below it.</p></div></div>{result_html}</section>
</main>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    app: StudioApp

    def do_GET(self) -> None:
        self._send(self.app.handle({}))

    def do_POST(self) -> None:
        length = int(self.headers.get('Content-Length', '0') or 0)
        body = self.rfile.read(length).decode('utf-8', errors='replace')
        form = parse_qs(body, keep_blank_values=True)
        self._send(self.app.handle(form))

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
    parser = argparse.ArgumentParser(description='Friendly SemOp Studio GUI')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8780)
    args = parser.parse_args()

    app = StudioApp()
    handler = type('SemOpStudioHandler', (_Handler,), {'app': app})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f'SemOp Studio running at http://{args.host}:{args.port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()















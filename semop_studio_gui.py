from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import os
import re
import sys
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from semop import (
    AnalogyEvalCase,
    BenchmarkGatedContinuousTrainer,
    CompilerRepairEvalCase,
    CorpusMemoryStore,
    CopilotRequest,
    DomainCopilot,
    GroundedExplanationEvalCase,
    HiddenPremiseEvalCase,
    OperatorTransferEvalCase,
    SemOpUnderstandingEvaluator,
    UnifiedSemOpTrainer,
    VLSOReasoner,
    VlsoGroundedEvaluator,
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
    'operating_domain': 'general',
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
SPLIT_OPTIONS = ('train', 'val', 'test')
VISION_MODE_OPTIONS = ('deep', 'hybrid', 'heuristic')
VISION_ANSWER_MODE_OPTIONS = ('structured', 'llm')


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
        )


@dataclass(frozen=True)
class ActionOutcome:
    flash: str = ''
    flash_tone: str = 'neutral'
    result_kind: str = ''
    result_payload: dict[str, object] | None = None


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
    return f"<div class='summary-grid'>{metrics}</div>{details}{_raw_details(payload)}"


def _render_unified_gate_summary(payload: dict[str, object]) -> str:
    benchmark = payload.get('benchmark', {}) if isinstance(payload.get('benchmark'), dict) else {}
    gate = payload.get('gate', {}) if isinstance(payload.get('gate'), dict) else {}
    promoted = payload.get('promoted_review_benchmarks', {}) if isinstance(payload.get('promoted_review_benchmarks'), dict) else {}
    corpus = payload.get('benchmark_corpus', {}) if isinstance(payload.get('benchmark_corpus'), dict) else {}
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
        _info_block('Persistent corpus added', corpus.get('added_case_count') or '-'),
    ])
    return f"<div class='summary-grid'>{metrics}</div>{details}{_raw_details(payload)}"


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

def render_result(kind: str, payload: dict[str, object] | None) -> str:
    if payload is None:
        return "<div class='empty'>Run one of the actions above. The result console will summarize the latest answer or training run here.</div>"
    titles = {
        'ops': 'Context reasoning result',
        'vision': 'Vision-grounded reasoning result',
        'unified_training': 'Unified SemOp training result',
        'unified_benchmark_gate': 'Unified benchmark gate result',
        'understanding_eval': 'Overall understanding benchmark',
    }
    if kind == 'ops':
        body = _render_ops_summary(payload)
    elif kind == 'vision':
        body = _render_vision_summary(payload)
    elif kind == 'unified_training':
        body = _render_unified_training_summary(payload)
    elif kind == 'unified_benchmark_gate':
        body = _render_unified_gate_summary(payload)
    elif kind == 'understanding_eval':
        body = _render_understanding_summary(payload)
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
    cards = [
        (
            'Unified trainer lane',
            'Main artifacts for parser, analogy, repair, and continuous learning.',
            [
                _metric_card('Corpus store', _status_text(state.unified_store_path, 'connected', 'missing')),
                _metric_card('Review queue', _status_text(state.unified_review_queue_path, 'connected', 'missing')),
                _metric_card('Artifacts ready', f'{artifact_ready}/{len(unified_artifacts)}'),
                _metric_card('Gate summary', _status_text(str(unified_output_dir / 'benchmark_gate.json'), 'ready', 'not run')),
            ],
            [
                _info_block('Output dir', state.unified_output_dir),
                _info_block('Persistent benchmark corpus', state.unified_benchmark_corpus_path),
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
    if kind == 'unified_benchmark_gate':
        gate = payload.get('gate', {}) if isinstance(payload.get('gate'), dict) else {}
        benchmark = payload.get('benchmark', {}) if isinstance(payload.get('benchmark'), dict) else {}
        title = 'The benchmark gate is open.' if gate.get('accepted') else 'The benchmark gate is still blocked.'
        notes = [
            f"Compiler validity: {_render_value(benchmark.get('compiler_validity'))}",
            f"Grounded fidelity: {_render_value(benchmark.get('grounded_explanation_fidelity'))}",
            f"Repair success: {_render_value(benchmark.get('repair_success_rate'))}",
        ]
        if gate.get('blocking_reasons'):
            notes.extend(str(item) for item in gate.get('blocking_reasons', [])[:3])
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

def _render_training_section(state: StudioState) -> str:
    return f"""
  <section class="card" id="training-lab" style="margin-top:20px;">
    <div class="section-head"><div><small class="eyebrow">Training</small><h2>Unified artifact trainer</h2><p>This exports parser, analogy, retained algebra, repair policy, repair utility, multimodal alignment, and continuous-learning artifacts from one corpus store.</p></div></div>
    <form method="post">
      {_render_unified_scope_fields(state)}
      <div class="actions"><button class="primary" name="action" value="run_unified_training">Run unified trainer</button></div>
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
      <div class="actions"><button class="primary" name="action" value="run_unified_benchmark_gate">Train + benchmark gate</button><button class="secondary" name="action" value="run_understanding_eval">Run overall understanding benchmark</button></div>
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


class StudioApp:
    def __init__(self) -> None:
        self.ops = DomainCopilot(mode='heuristic', review_queue_path=UNIFIED_EXAMPLE['review_queue'])

    def handle(self, form: dict[str, list[str]]) -> str:
        action = _first(form, 'action', '')
        state = StudioState.from_form(form)
        try:
            state, outcome = self._run_action(action, state)
        except Exception as exc:
            outcome = ActionOutcome(flash=f'Action failed: {exc}', flash_tone='error')
        return render_page(state, outcome)

    def _run_action(self, action: str, state: StudioState) -> tuple[StudioState, ActionOutcome]:
        if action == 'load_ops_example':
            state = replace(
                state,
                ops_query=OPS_EXAMPLE['query'],
                ops_context=OPS_EXAMPLE['context'],
                ops_domain=OPS_EXAMPLE['domain'],
                ops_scenario=OPS_EXAMPLE['scenario'],
            )
            return state, ActionOutcome(flash='Loaded the default context-reasoning example.')
        if action == 'load_vision_example':
            state = replace(
                state,
                vision_query=VISION_EXAMPLE['query'],
                vision_image=VISION_EXAMPLE['image_path'],
            )
            return state, ActionOutcome(flash='Loaded the default vision example.')
        if action == 'run_ops':
            return state, self._run_ops(state)
        if action == 'run_vision':
            return state, self._run_vision(state)
        if action == 'run_unified_training':
            return state, self._run_unified_training(state)
        if action == 'run_unified_benchmark_gate':
            return state, self._run_unified_benchmark_gate(state)
        if action == 'run_understanding_eval':
            return state, self._run_understanding_eval(state)
        return state, ActionOutcome()

    def _run_ops(self, state: StudioState) -> ActionOutcome:
        request = CopilotRequest(
            query=state.ops_query,
            context=state.ops_context,
            domain=state.ops_domain,
            scenario=state.ops_scenario,
        )
        result = self.ops.run(request)
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
    def _run_unified_training(state: StudioState) -> ActionOutcome:
        summary = UnifiedSemOpTrainer().train_from_store(
            state.unified_store_path,
            state.unified_output_dir,
            source=state.unified_source or None,
            split=state.unified_split,
            review_store_path=state.unified_review_queue_path or None,
            approved_queries_only=state.unified_approved_queries_only,
            operating_domain=state.unified_operating_domain or None,
        )
        return ActionOutcome(
            flash='Unified trainer finished and exported a fresh artifact bundle.',
            flash_tone='success',
            result_kind='unified_training',
            result_payload=summary.model_dump(),
        )

    @staticmethod
    def _run_unified_benchmark_gate(state: StudioState) -> ActionOutcome:
        graphs = _load_store_graphs(state.unified_store_path, state.unified_source, state.unified_split)
        if not graphs:
            raise FileNotFoundError('No graphs were found in the selected store and split.')
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
            operating_domain=state.unified_operating_domain or None,
            benchmark_corpus_path=state.unified_benchmark_corpus_path or None,
        )
        return ActionOutcome(
            flash=f'Unified trainer and benchmark gate finished on {len(graphs)} seed graphs.',
            flash_tone='success',
            result_kind='unified_benchmark_gate',
            result_payload=summary.model_dump(),
        )

    @staticmethod
    def _run_understanding_eval(state: StudioState) -> ActionOutcome:
        summary = SemOpUnderstandingEvaluator().evaluate(
            hidden_premise_cases=_load_hidden_premise_cases(state.understanding_hidden_input),
            cp_examples=None,
            cp_hidden_examples=None,
            vlso_cases=_load_vlso_cases(state.understanding_vlso_input) or None,
            vlso_real_image_cases=_load_vlso_cases(state.understanding_vlso_real_input) or None,
            cp_mode='heuristic',
        )
        return ActionOutcome(
            flash='Overall understanding benchmark finished.',
            flash_tone='success',
            result_kind='understanding_eval',
            result_payload=summary.model_dump(),
        )

def render_page(state: StudioState, outcome: ActionOutcome) -> str:
    flash_html = ''
    if outcome.flash:
        flash_html = f"<div class='flash {html.escape(outcome.flash_tone)}'>{html.escape(outcome.flash)}</div>"
    result_html = render_result(outcome.result_kind, outcome.result_payload)
    snapshot_html = render_workspace_snapshot(state, outcome.result_kind)
    spotlight_html = render_result_spotlight(outcome.result_kind, outcome.result_payload)
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
<title>SemOp Studio</title>
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
        <a class="nav-pill" href="#training-lab">Train artifacts</a>
        <a class="nav-pill" href="#benchmark-lab">Benchmark gate</a>
        <a class="nav-pill" href="#result-panel">See results</a>
      </div>
    </div>
    <div class="card">
      <small class="eyebrow">Starter flow</small>
      <h2>Use the same order every time.</h2>
      <ol>
        <li>Run the ops or vision example to confirm local reasoning works.</li>
        <li>Launch Unified trainer on your corpus store.</li>
        <li>Run Train + benchmark gate to score the new artifact bundle.</li>
        <li>Inspect the result console and keep only accepted bundles.</li>
      </ol>
      <div class="hero-preview-wrap">{vision_preview_html}</div>
    </div>
  </section>
  {flash_html}
  <section class="dashboard-grid">{snapshot_html}</section>
  <section style="margin-bottom:20px;">{spotlight_html}</section>
  {_render_reasoning_section(state)}
  {_render_training_section(state)}
  {_render_benchmark_section(state)}
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

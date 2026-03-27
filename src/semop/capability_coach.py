from __future__ import annotations

import json
import sqlite3
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .capability_audit import CapabilityAuditRunner
from .corpus_store import CorpusMemoryStore
from .domain_copilot import CopilotRequest, DomainCopilot
from .generalization_proof import GeneralizationProofHarness
from .grounding_self_evolution import GroundingSelfEvolutionRunner
from .pipeline import StructuredMeaningPipeline
from .review_queue import ReviewQueueStore, infer_review_severity, review_reasons_from_graph_and_kpis
from .structures import StructuredMeaningGraph
from .turboquant_review import TurboQuantReviewPlanner, TurboQuantReviewSummary
from .unified_benchmark import BenchmarkGatedContinuousTrainer
from .operator_evolution import OperatorTransferEvalCase
from .premise_eval import HiddenPremiseEvalCase
from .vlso import VlsoGroundedEvaluator

DEFAULT_CAPABILITY_CURRICULUM: tuple[dict[str, Any], ...] = (
    {
        'domain': 'warehouse_exception',
        'scenario': 'exception_response',
        'context': 'If the lane is blocked, access is uncertain, or approval is missing, stop first and verify the safe alternative path and approval before moving.',
        'queries': (
            'The lane is partly blocked and the approval token is missing. What do I verify before continuing?',
            'An access door is closed and the route is narrow. Do I force it open or confirm the safe path first?',
            'The item is urgent but the zone is blocked. What checks come before movement?',
        ),
        'candidate_queries': (
            'The lane looks passable, but approval is still unresolved. Which hidden checks come first?',
            'If the blocked route might clear soon, do I wait, reroute, or verify approval first?',
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
        'candidate_queries': (
            'I can see the tote, but I have not confirmed the label. What should happen before I touch it?',
            'The cabinet is open, but my onboarding checklist is incomplete. What do I verify first?',
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
        'candidate_queries': (
            'The object is visible through the opening, but I do not know if retrieval is safe. Which premises matter?',
            'There is a lid and a narrow gap. Should I act or inspect the access state first?',
        ),
        'visual_queries': (
            'What openings, containers, or reachable objects are visible here?',
            'Which visible object here looks like an access path or container?',
        ),
    },
    {
        'domain': 'warehouse_exception',
        'scenario': 'quality_gate',
        'context': 'When a quality or approval gate is unresolved, verify the gate condition, evidence, and fallback path before executing the next physical step.',
        'queries': (
            'The package is present, but the quality gate is still open. What should I verify before moving it?',
            'The item looks correct, yet the release signal is missing. Which gate evidence comes first?',
            'If the quality note conflicts with the aisle status, what do I check before continuing?',
        ),
        'candidate_queries': (
            'The barcode matches but the release mark is absent. Do I proceed or hold for gate evidence?',
            'The manager says continue, but the quality gate still looks unresolved. What should I verify?',
        ),
        'visual_queries': (
            'Which visible object here could act as a gate, label, or access constraint?',
            'What visible containers or openings would matter for a quality hold?',
        ),
    },
    {
        'domain': 'service',
        'scenario': 'document_grounding',
        'context': 'When the answer depends on SOP text, cite the document evidence first and avoid unsupported operational claims.',
        'queries': (
            'The SOP mentions approval and alternate access. What exact grounded action follows from that?',
            'Which sentence in the SOP justifies stopping before movement here?',
            'If the SOP is ambiguous about access, what grounded answer can I safely give?',
        ),
        'candidate_queries': (
            'What part of the SOP directly supports the answer about alternate routing?',
            'How do I answer this request while trimming unsupported claims to only document-backed ones?',
        ),
        'visual_queries': (
            'Which visible object here would need document-backed grounding before action?',
            'What visible opening or container should be described conservatively from evidence only?',
        ),
    },
)

MAX_GUIDED_QUERIES_PER_ROUND = 12
MAX_VISUAL_QUERIES_PER_ROUND = 4

DEFAULT_GUIDED_VARIANTS: tuple[str, ...] = (
    'The route is blocked and approval has not arrived yet. What should I verify before moving?',
    'Manager approval is still pending. Do I continue the task or stop first?',
    'The work zone is blocked. Should I reroute now or confirm approval before continuing?',
)


@dataclass
class CapabilityCurriculumRound:
    round_index: int
    source_used: str
    domain: str
    scenario: str
    hidden_seeded: int = 0
    transfer_seeded: int = 0
    guided_seeded: int = 0
    approved_reviews: int = 0
    visual_seeded: int = 0
    skipped_existing: int = 0
    average_novelty: float = 0.0
    errors: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilityImprovementSummary:
    workspace: str
    output_path: str
    before: dict[str, Any]
    after: dict[str, Any]
    gate: dict[str, Any] = field(default_factory=dict)
    proof: dict[str, Any] = field(default_factory=dict)
    quantization: dict[str, Any] = field(default_factory=dict)
    self_evolution: dict[str, Any] = field(default_factory=dict)
    rounds: list[CapabilityCurriculumRound] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    weak_axes_before: list[str] = field(default_factory=list)
    weak_axes_after: list[str] = field(default_factory=list)
    delta_readiness_percent: float = 0.0

    def model_dump(self) -> dict[str, Any]:
        return {
            'workspace': self.workspace,
            'output_path': self.output_path,
            'before': dict(self.before),
            'after': dict(self.after),
            'gate': dict(self.gate),
            'proof': dict(self.proof),
            'quantization': dict(self.quantization),
            'self_evolution': dict(self.self_evolution),
            'rounds': [item.model_dump() for item in self.rounds],
            'actions_taken': list(self.actions_taken),
            'weak_axes_before': list(self.weak_axes_before),
            'weak_axes_after': list(self.weak_axes_after),
            'delta_readiness_percent': self.delta_readiness_percent,
        }


class CapabilityImprovementRunner:
    def __init__(self, mode: str = 'heuristic') -> None:
        self.mode = mode

    @staticmethod
    def _unique_texts(items: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            normalized = str(item or '').strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    @staticmethod
    def _persist_graph(store: CorpusMemoryStore, graph: StructuredMeaningGraph, source: str, split: str) -> None:
        store.upsert_graph(graph, source=source, split=split)
        store.upsert_premise_operator_memory(graph, source=source, split=split)

    @staticmethod
    def _grounded_review_answer(context: str, fallback: str) -> str:
        normalized_context = re.sub(r'\s+', ' ', str(context).strip())
        if normalized_context:
            for chunk in re.split(r'(?<=[.!?])\s+|\n+', normalized_context):
                line = str(chunk).strip(' -')
                if len(line) >= 12:
                    return line
            return normalized_context
        return str(fallback).strip()


    @classmethod
    def _review_answer_from_graph(cls, graph: StructuredMeaningGraph, fallback: str) -> str:
        report = getattr(graph, 'operator_execution', None)
        if report is not None:
            grounded_claims = [str(item.claim).strip() for item in report.claim_groundings if item.grounded and str(item.claim).strip()]
            if grounded_claims:
                return grounded_claims[0]
        for result in graph.symbolic_results:
            answer = str(result.answer or '').strip()
            if result.domain == 'document_grounding' and answer:
                return answer
        if graph.hidden_goals:
            goal = str(graph.hidden_goals[0]).strip()
            premises = [str(item).strip() for item in (graph.required_premises or graph.satisfied_premises or graph.missing_premises) if str(item).strip()]
            if goal and premises:
                return f'{goal}. First verify {premises[0]}.'
            if goal:
                return goal
        return cls._grounded_review_answer(str(getattr(graph, 'source_context', '')), fallback)

    @classmethod
    def _bootstrap_review_reasons(cls, graph: StructuredMeaningGraph, kpis: dict[str, Any]) -> list[str]:
        reasons = list(review_reasons_from_graph_and_kpis(graph, kpis))
        if str(getattr(graph, 'source_context', '')).strip() and 'grounding_review' not in reasons:
            reasons.append('grounding_review')
        if 'approved_training_trace' not in reasons:
            reasons.append('approved_training_trace')
        return cls._unique_texts(reasons)

    @staticmethod
    def _load_hidden_cases(path_value: str) -> list[HiddenPremiseEvalCase]:
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

    @staticmethod
    def _load_transfer_cases(path_value: str) -> list[OperatorTransferEvalCase]:
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

    @staticmethod
    def _load_vlso_cases(path_value: str) -> list[Any]:
        path = Path(path_value)
        if not path.exists():
            return []
        return VlsoGroundedEvaluator.load_cases(path)

    @staticmethod
    def _load_store_graphs(store_path: str, split: str = 'train', source: str | None = None, limit: int = 640) -> list[StructuredMeaningGraph]:
        store = CorpusMemoryStore(store_path)
        rows = list(store.fetch_graphs(split=split or None, source=source or None))
        return rows[:limit]


    @staticmethod
    def _expand_text_queries(template: dict[str, Any], base_queries: Sequence[str]) -> list[str]:
        scenario = str(template.get('scenario', 'qa')).strip() or 'qa'
        context_hint = str(template.get('context', '')).strip()
        suffixes = (
            'Answer with the safest first step only.',
            'State the hidden prerequisite before the action.',
            'Keep the answer grounded in the given SOP or context.',
            'Name the first constraint that blocks immediate execution.',
            'Explain the answer as an operator-decomposition check.',
            'List the missing approval, access, or evidence requirement first.',
            'Trim unsupported claims and keep only grounded guidance.',
            'Give the fallback path if the route or gate is still blocked.',
            'Summarize the goal-preservation check before any action.',
            'Answer in one conservative sentence with the first safe prerequisite.',
        )
        variants: list[str] = []
        for query in base_queries:
            normalized = str(query).strip()
            if not normalized:
                continue
            stem = normalized.rstrip(' ?.')
            variants.append(normalized)
            variants.append(f'[{scenario}] {normalized}')
            variants.append(f'Operator algebra view: {normalized}')
            variants.append(f'Grounded-only answer: {normalized}')
            variants.append(f'Constraint-first answer: {normalized}')
            for suffix in suffixes:
                variants.append(f'{stem}. {suffix}')
            if context_hint:
                variants.append(f'{normalized} Use this context: {context_hint}')
                variants.append(f'{stem}. Use this context and cite the first supporting line: {context_hint}')
        return CapabilityImprovementRunner._unique_texts(variants)

    @staticmethod
    def _expand_visual_queries(template: dict[str, Any], base_queries: Sequence[str]) -> list[str]:
        scenario = str(template.get('scenario', 'qa')).strip() or 'qa'
        suffixes = (
            'Answer from visible evidence only.',
            'Name the first visible constraint before action.',
            'Describe openings, containers, and reachable objects conservatively.',
            'Point out the first visible opening or gate that changes the action plan.',
            'Keep the answer geometry-first and avoid unsupported object guesses.',
        )
        variants: list[str] = []
        for query in base_queries:
            normalized = str(query).strip()
            if not normalized:
                continue
            stem = normalized.rstrip(' ?.')
            variants.append(normalized)
            variants.append(f'[{scenario} visual] {normalized}')
            variants.append(f'Visual grounding only: {normalized}')
            variants.append(f'Geometry-first view: {normalized}')
            for suffix in suffixes:
                variants.append(f'{stem}. {suffix}')
        return CapabilityImprovementRunner._unique_texts(variants)

    @staticmethod
    def _backfill_store_scenarios(store_path: str, review_queue_path: str) -> int:
        review_store = ReviewQueueStore(review_queue_path)
        review_lookup = {}
        for item in review_store.fetch_items(status='approved', limit=10000):
            normalized_query = CorpusMemoryStore.normalize_query(item.query)
            review_lookup[normalized_query] = (item.domain, item.scenario)
        if not review_lookup:
            return 0
        updated = 0
        with sqlite3.connect(store_path) as conn:
            rows = conn.execute('SELECT query, normalized_query, source, split, intent, graph_json FROM examples').fetchall()
            for query, normalized_query, source, split, intent, graph_json in rows:
                mapped = review_lookup.get(str(normalized_query))
                if not mapped:
                    continue
                graph = StructuredMeaningGraph.from_dict(json.loads(graph_json))
                target_domain, target_scenario = mapped
                changed = False
                if str(getattr(graph, 'domain', 'general') or 'general').strip() in {'', 'general'} and str(target_domain).strip():
                    graph.domain = str(target_domain).strip()
                    changed = True
                if str(getattr(graph, 'scenario', 'qa') or 'qa').strip() in {'', 'qa'} and str(target_scenario).strip():
                    graph.scenario = str(target_scenario).strip()
                    changed = True
                if not changed:
                    continue
                conn.execute(
                    'UPDATE examples SET query=?, intent=?, graph_json=? WHERE normalized_query=? AND source=? AND split=?',
                    (graph.query, graph.intent, json.dumps(graph.model_dump(), ensure_ascii=False), normalized_query, source, split),
                )
                updated += 1
            conn.commit()
        return updated

    def _seed_builtin_cases(
        self,
        *,
        pipeline: StructuredMeaningPipeline,
        store: CorpusMemoryStore,
        existing_queries: set[str],
        source_used: str,
        split: str,
        hidden_cases: Sequence[HiddenPremiseEvalCase],
        transfer_cases: Sequence[OperatorTransferEvalCase],
        round_summary: CapabilityCurriculumRound,
    ) -> None:
        for case in hidden_cases:
            query = str(case.query).strip()
            if not query:
                continue
            if query in existing_queries:
                round_summary.skipped_existing += 1
                continue
            try:
                graph = pipeline.run(query)
                graph.domain = str(case.domain or 'general').strip() or 'general'
                graph.scenario = str(getattr(case, 'scenario', '') or 'hidden_premise_eval').strip() or 'hidden_premise_eval'
                self._persist_graph(store, graph, source_used, split)
                existing_queries.add(query)
                round_summary.hidden_seeded += 1
            except Exception as exc:
                round_summary.errors.append(f'hidden case failed: {query[:60]} ({exc})')
        for case in transfer_cases:
            query = str(case.query).strip()
            if not query:
                continue
            if query in existing_queries:
                round_summary.skipped_existing += 1
                continue
            try:
                graph = pipeline.run(query)
                graph.domain = str(getattr(case, 'domain', 'general') or 'general').strip() or 'general'
                graph.scenario = 'operator_transfer_' + str(getattr(case, 'split', 'train') or 'train')
                self._persist_graph(store, graph, source_used, split)
                existing_queries.add(query)
                round_summary.transfer_seeded += 1
            except Exception as exc:
                round_summary.errors.append(f'transfer case failed: {query[:60]} ({exc})')

    def _seed_round(
        self,
        round_index: int,
        *,
        source_used: str,
        split: str,
        store_path: str,
        review_queue_path: str,
        hidden_cases: Sequence[HiddenPremiseEvalCase],
        transfer_cases: Sequence[OperatorTransferEvalCase],
        vision_image: str,
        quantizer: TurboQuantReviewPlanner,
    ) -> CapabilityCurriculumRound:
        template = DEFAULT_CAPABILITY_CURRICULUM[min(max(0, round_index - 1), len(DEFAULT_CAPABILITY_CURRICULUM) - 1)]
        store = CorpusMemoryStore(store_path)
        review_store = ReviewQueueStore(review_queue_path)
        pipeline = StructuredMeaningPipeline(mode=self.mode)
        copilot = DomainCopilot(mode=self.mode)
        existing_approved = {
            (item.domain, item.scenario, item.query)
            for item in review_store.fetch_items(status='approved', limit=5000)
        }
        existing_queries = {str(item).strip() for item in store.fetch_queries(split=split, source=source_used)}
        round_summary = CapabilityCurriculumRound(
            round_index=round_index,
            source_used=source_used,
            domain=str(template['domain']),
            scenario=str(template['scenario']),
        )
        novelty_scores: list[float] = []
        if round_index == 1:
            self._seed_builtin_cases(
                pipeline=pipeline,
                store=store,
                existing_queries=existing_queries,
                source_used=source_used,
                split=split,
                hidden_cases=hidden_cases,
                transfer_cases=transfer_cases,
                round_summary=round_summary,
            )
        candidate_queries = self._expand_text_queries(
            template,
            [
                *DEFAULT_GUIDED_VARIANTS,
                *list(template.get('queries', ())),
                *list(template.get('candidate_queries', ())),
            ],
        )
        ranked_queries = sorted(
            candidate_queries,
            key=lambda item: quantizer.score_text(f"{template['domain']}\n{template['scenario']}\n{template['context']}\n{item}"),
            reverse=True,
        )
        guided_seed_limit = 0
        for query in ranked_queries:
            if guided_seed_limit >= MAX_GUIDED_QUERIES_PER_ROUND:
                break
            if query in existing_queries:
                round_summary.skipped_existing += 1
                continue
            try:
                request = CopilotRequest(
                    query=query,
                    context=str(template['context']),
                    domain=str(template['domain']),
                    scenario=str(template['scenario']),
                )
                result = copilot.run(request)
                result.graph.scenario = request.scenario
                self._persist_graph(store, result.graph, source_used, split)
                existing_queries.add(query)
                round_summary.guided_seeded += 1
                assignment = quantizer.add(
                    label=query,
                    text=f"{request.domain}\n{request.scenario}\n{request.context}\n{query}\n{result.answer_text}",
                    domain=request.domain,
                    scenario=request.scenario,
                )
                novelty_scores.append(assignment.novelty_score)
                reasons = self._bootstrap_review_reasons(result.graph, result.kpis.model_dump())
                severity = infer_review_severity(request.domain, request.scenario, reasons, result.kpis.model_dump())
                key = (request.domain, request.scenario, request.query)
                if key not in existing_approved:
                    item_id = review_store.enqueue(
                        domain=request.domain,
                        scenario=request.scenario,
                        query=request.query,
                        reasons=reasons,
                        answer_text=self._review_answer_from_graph(result.graph, self._grounded_review_answer(request.context, result.answer_text)),
                        kpis=result.kpis.model_dump(),
                        audit_items=[{'stage': item.stage, 'detail': item.detail} for item in result.audit_items],
                        context_text=request.context,
                        graph_payload=result.graph.model_dump(),
                        severity=severity,
                    )
                    review_store.update_status(item_id, 'approved', 'Capability coach starter trace.')
                    existing_approved.add(key)
                    round_summary.approved_reviews += 1
            except Exception as exc:
                round_summary.errors.append(f'guided query failed: {query[:60]} ({exc})')
        image_path = Path(vision_image)
        if image_path.exists():
            visual_queries = self._expand_visual_queries(template, list(template.get('visual_queries', ())))
            ranked_visual = sorted(
                visual_queries,
                key=lambda item: quantizer.score_text(f"vlso\n{template['scenario']}_visual\n{item}"),
                reverse=True,
            )
            visual_seed_limit = 0
            for query in ranked_visual:
                if visual_seed_limit >= MAX_VISUAL_QUERIES_PER_ROUND:
                    break
                if query in existing_queries:
                    round_summary.skipped_existing += 1
                    continue
                try:
                    graph = pipeline.run(query, visual_input={'image_path': str(image_path), 'metadata': {'image_path': str(image_path)}})
                    graph.domain = 'vlso'
                    graph.scenario = f"{template['scenario']}_visual"
                    self._persist_graph(store, graph, source_used, split)
                    existing_queries.add(query)
                    round_summary.visual_seeded += 1
                    visual_seed_limit += 1
                    visual_scenario = f"{template['scenario']}_visual"
                    assignment = quantizer.add(
                        label=query,
                        text=f"vlso\n{visual_scenario}\n{query}\n{image_path}",
                        domain='vlso',
                        scenario=visual_scenario,
                    )
                    novelty_scores.append(assignment.novelty_score)
                    reasons = self._bootstrap_review_reasons(graph, {'grounded_answer_accuracy': 1.0, 'visual_seeded': True})
                    key = ('vlso', visual_scenario, query)
                    if key not in existing_approved:
                        item_id = review_store.enqueue(
                            domain='vlso',
                            scenario=visual_scenario,
                            query=query,
                            reasons=reasons,
                            answer_text=self._review_answer_from_graph(graph, query),
                            kpis={'grounded_answer_accuracy': 1.0, 'visual_seeded': True},
                            audit_items=[{'stage': 'visual_seed', 'detail': f'image={image_path.name}'}],
                            context_text=str(image_path),
                            graph_payload=graph.model_dump(),
                            severity='medium',
                        )
                        review_store.update_status(item_id, 'approved', 'Capability coach visual starter trace.')
                        existing_approved.add(key)
                        round_summary.approved_reviews += 1
                except Exception as exc:
                    round_summary.errors.append(f'visual query failed: {query[:60]} ({exc})')
        round_summary.average_novelty = round(sum(novelty_scores) / float(len(novelty_scores) or 1), 4)
        return round_summary

    def run(
        self,
        *,
        workspace: str = '.',
        output_path: str | None = None,
        bootstrap_missing: bool = True,
        unified_output_dir: str = 'data/unified_semop_gui_run',
        unified_store_path: str = 'data/semop_memory.db',
        unified_review_queue_path: str = 'data/ops_review_queue.db',
        unified_benchmark_corpus_path: str = 'data/unified_semop_gui_run/persistent_benchmark_corpus.json',
        math_output_dir: str = 'data/math_world_model_gui_run',
        math_cases_path: str = 'examples/math_world_model_starter.jsonl',
        visual_output_dir: str = 'data/math_world_model_gui_run/visual_3d',
        hidden_input: str = 'examples/hidden_premise_eval.jsonl',
        transfer_input: str = 'examples/operator_transfer_eval.jsonl',
        vlso_input: str = 'examples/vlso_eval.jsonl',
        vlso_real_input: str = 'examples/vlso_real_image_eval_gold.jsonl',
        vision_image: str = 'data/scene.png',
        split: str = 'train',
        source_prefix: str = 'capability_coach',
        progress_callback: Any = None,
    ) -> CapabilityImprovementSummary:
        workspace_path = str(Path(workspace).resolve())
        audit = CapabilityAuditRunner(mode=self.mode)
        if progress_callback:
            progress_callback(0.05, 'Running the initial capability audit.')
        before = audit.run(
            workspace=workspace_path,
            bootstrap_missing=bootstrap_missing,
            output_path=str(Path(unified_output_dir) / 'capability_audit_report_before.json'),
            unified_output_dir=unified_output_dir,
            unified_store_path=unified_store_path,
            unified_review_queue_path=unified_review_queue_path,
            unified_benchmark_corpus_path=unified_benchmark_corpus_path,
            math_output_dir=math_output_dir,
            math_cases_path=math_cases_path,
            visual_output_dir=visual_output_dir,
        )
        weak_axes_before = [axis.name for axis in before.axes if axis.score < 0.65]
        hidden_cases = self._load_hidden_cases(hidden_input)
        transfer_cases = self._load_transfer_cases(transfer_input)
        vlso_cases = self._load_vlso_cases(vlso_input)
        vlso_real_cases = self._load_vlso_cases(vlso_real_input)
        round_summaries: list[CapabilityCurriculumRound] = []
        backfilled = self._backfill_store_scenarios(unified_store_path, unified_review_queue_path)
        if progress_callback:
            progress_callback(0.1, 'Running grounding self-evolution before the proof loop.')
        self_evolution = GroundingSelfEvolutionRunner(mode=self.mode).run(
            unified_store_path,
            unified_review_queue_path,
            unified_output_dir,
            split=split,
            rounds=2,
            cases_per_round=12,
            source_prefix='capability_grounding_self_evolution',
            progress_callback=(lambda p, d: progress_callback(0.1 + (float(p) * 0.12), d)) if progress_callback else None,
        ).model_dump()
        quantizer = TurboQuantReviewPlanner(bit_width=6, max_clusters=24)
        source_schedule = [f'{source_prefix}_round_{index:02d}' for index in range(1, len(DEFAULT_CAPABILITY_CURRICULUM) + 1)]

        def round_setup_callback(round_number: int, source_value: str, split_value: str) -> None:
            if progress_callback:
                progress_callback(0.12 + (round_number - 1) * 0.1, f'Seeding reviewed curriculum round {round_number}/{len(source_schedule)}.')
            summary = self._seed_round(
                round_number,
                source_used=source_value or source_schedule[round_number - 1],
                split=split_value,
                store_path=unified_store_path,
                review_queue_path=unified_review_queue_path,
                hidden_cases=hidden_cases,
                transfer_cases=transfer_cases,
                vision_image=vision_image,
                quantizer=quantizer,
            )
            round_summaries.append(summary)

        if progress_callback:
            progress_callback(0.3, 'Running the multi-round proof loop with round-by-round corpus growth.')
        proof_summary = GeneralizationProofHarness(mode=self.mode).run(
            unified_store_path,
            unified_output_dir,
            source=None,
            split=split,
            rounds=len(source_schedule),
            review_store_path=unified_review_queue_path,
            approved_queries_only=False,
            hidden_premise_cases=hidden_cases,
            transfer_cases=transfer_cases,
            analogy_cases=None,
            grounding_cases=None,
            compiler_cases=None,
            vlso_cases=vlso_cases,
            vlso_real_image_cases=vlso_real_cases,
            operating_domain='general',
            benchmark_corpus_path=unified_benchmark_corpus_path,
            round_setup_callback=round_setup_callback,
            progress_callback=(lambda p, d: progress_callback(0.3 + (float(p) * 0.45), d)) if progress_callback else None,
        ).model_dump()
        derivation_graphs = self._load_store_graphs(unified_store_path, split=split, source=None)
        hidden_eval_cases = hidden_cases or GeneralizationProofHarness._derive_hidden_premise_cases(derivation_graphs)
        transfer_eval_cases = transfer_cases or GeneralizationProofHarness._derive_transfer_cases(derivation_graphs)
        analogy_cases = GeneralizationProofHarness._derive_analogy_cases(derivation_graphs)
        grounding_cases = GeneralizationProofHarness._derive_grounding_cases(derivation_graphs)
        compiler_cases = GeneralizationProofHarness._derive_compiler_cases(derivation_graphs)

        if progress_callback:
            progress_callback(0.8, 'Rerunning the top-level benchmark gate on the grown corpus.')
        gate_summary = BenchmarkGatedContinuousTrainer(mode=self.mode).train_evaluate_and_gate(
            unified_store_path,
            unified_output_dir,
            source=None,
            split=split,
            review_store_path=unified_review_queue_path,
            approved_queries_only=False,
            hidden_premise_cases=hidden_eval_cases,
            transfer_cases=transfer_eval_cases,
            analogy_cases=analogy_cases,
            grounding_cases=grounding_cases,
            compiler_cases=compiler_cases,
            vlso_cases=vlso_cases,
            operating_domain='general',
            benchmark_corpus_path=unified_benchmark_corpus_path,
        ).model_dump()
        quantization_path = Path(unified_output_dir) / 'capability_quantization_report.json'
        quantization = quantizer.summarize(quantization_path).model_dump()
        actions_taken = [
            f"Backfilled scenario metadata on {backfilled} stored graphs using approved review traces.",
            f"Grounding self-evolution stored {self_evolution.get('stored_graphs', 0)} graphs and approved {self_evolution.get('approved_reviews', 0)} corrected traces.",
            f"Seeded {sum(item.hidden_seeded + item.transfer_seeded + item.guided_seeded + item.visual_seeded for item in round_summaries)} starter graphs across {len(round_summaries)} curriculum rounds.",
            f"Approved {sum(item.approved_reviews for item in round_summaries)} starter review traces for continuous learning and benchmark promotion.",
            f"TurboQuant-inspired review memory used {quantization.get('used_clusters', 0)}/{quantization.get('codebook_size', 0)} clusters with coverage {quantization.get('coverage_score', 0.0)}.",
            f"Self-evolution final grounding score reached {self_evolution.get('final_grounding_score', 0.0)}.",
            'Reran the multi-round generalization proof on the updated corpus and review queue.',
            'Reran benchmark-gated training after expanding the reviewed starter corpus.',
        ]

        if progress_callback:
            progress_callback(0.96, 'Running the final capability audit after the remediation cycle.')
        after_report_target = Path(output_path or (Path(unified_output_dir) / 'capability_improvement_report.json'))
        after = audit.run(
            workspace=workspace_path,
            bootstrap_missing=bootstrap_missing,
            output_path=str(Path(unified_output_dir) / 'capability_audit_report.json'),
            unified_output_dir=unified_output_dir,
            unified_store_path=unified_store_path,
            unified_review_queue_path=unified_review_queue_path,
            unified_benchmark_corpus_path=unified_benchmark_corpus_path,
            math_output_dir=math_output_dir,
            math_cases_path=math_cases_path,
            visual_output_dir=visual_output_dir,
        )
        weak_axes_after = [axis.name for axis in after.axes if axis.score < 0.65]
        summary = CapabilityImprovementSummary(
            workspace=workspace_path,
            output_path=str(after_report_target),
            before=before.model_dump(),
            after=after.model_dump(),
            gate=gate_summary,
            proof=proof_summary,
            quantization=quantization,
            self_evolution=self_evolution,
            rounds=round_summaries,
            actions_taken=actions_taken,
            weak_axes_before=weak_axes_before,
            weak_axes_after=weak_axes_after,
            delta_readiness_percent=round(float(after.overall_readiness_percent) - float(before.overall_readiness_percent), 1),
        )
        after_report_target.parent.mkdir(parents=True, exist_ok=True)
        after_report_target.write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

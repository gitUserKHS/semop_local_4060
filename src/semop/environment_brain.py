from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Sequence

from .corpus_store import CorpusMemoryStore
from .domain_copilot import CopilotRequest, CopilotResult, DomainCopilot
from .review_queue import ReviewQueueStore
from .unified_benchmark import UnifiedSemOpTrainer, UnifiedSemOpTrainingSummary
from .unified_world_solver_guidance import UnifiedWorldSolverGuidance, UnifiedWorldSolverGuidanceEngine
from .vlso import VLSOReasoner
from .structures import StructuredMeaningGraph


@dataclass
class EnvironmentConceptStat:
    label: str
    kind: str
    support_count: int

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentRoutineStat:
    label: str
    support_count: int
    required_premises: list[str] = field(default_factory=list)
    operator_families: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentProbe:
    query: str
    purpose: str
    modality: str = 'text'

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentBrainSummary:
    environment_name: str
    environment_source: str
    domain: str
    scenario: str
    source_context: str
    report_path: str
    visual_input: str = ''
    seeded_query_count: int = 0
    stored_graph_count: int = 0
    auto_approved_review_count: int = 0
    pending_review_count: int = 0
    mastery_scores: dict[str, float] = field(default_factory=dict)
    stable_concepts: list[EnvironmentConceptStat] = field(default_factory=list)
    stable_constraints: list[str] = field(default_factory=list)
    routine_patterns: list[EnvironmentRoutineStat] = field(default_factory=list)
    hazard_patterns: list[str] = field(default_factory=list)
    visual_entities: list[str] = field(default_factory=list)
    next_probes: list[EnvironmentProbe] = field(default_factory=list)
    training: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            'environment_name': self.environment_name,
            'environment_source': self.environment_source,
            'domain': self.domain,
            'scenario': self.scenario,
            'source_context': self.source_context,
            'report_path': self.report_path,
            'visual_input': self.visual_input,
            'seeded_query_count': self.seeded_query_count,
            'stored_graph_count': self.stored_graph_count,
            'auto_approved_review_count': self.auto_approved_review_count,
            'pending_review_count': self.pending_review_count,
            'mastery_scores': dict(self.mastery_scores),
            'stable_concepts': [item.model_dump() for item in self.stable_concepts],
            'stable_constraints': list(self.stable_constraints),
            'routine_patterns': [item.model_dump() for item in self.routine_patterns],
            'hazard_patterns': list(self.hazard_patterns),
            'visual_entities': list(self.visual_entities),
            'next_probes': [item.model_dump() for item in self.next_probes],
            'training': dict(self.training),
            'notes': list(self.notes),
        }


class EnvironmentBrainRunner:
    def __init__(self, mode: str = 'heuristic') -> None:
        self.mode = mode

    def run(
        self,
        *,
        environment_name: str,
        domain: str,
        scenario: str,
        context: str,
        store_path: str = 'data/semop_memory.db',
        review_queue_path: str | None = 'data/ops_review_queue.db',
        output_dir: str = 'data/unified_semop_gui_run/environment_brain',
        visual_input: str | None = None,
        seed_queries: Sequence[str] | None = None,
        concept_store_path: str | None = None,
        operator_store_path: str | None = None,
        affordance_weights_path: str | None = None,
        self_learning_plan: dict[str, Any] | None = None,
        integrated_reasoning: dict[str, Any] | None = None,
    ) -> EnvironmentBrainSummary:
        env_name = str(environment_name or '').strip() or f'{domain}_{scenario}'
        env_source = self._environment_source(env_name, domain, scenario)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        store = CorpusMemoryStore(store_path)
        review_store = ReviewQueueStore(review_queue_path) if review_queue_path else None
        copilot = DomainCopilot(mode=self.mode, review_queue_path=review_queue_path)
        guidance = UnifiedWorldSolverGuidanceEngine().build(
            plan=self_learning_plan,
            reasoning=integrated_reasoning,
            context=context,
            domain=domain,
            scenario=scenario,
        )

        queries = self._compose_queries(seed_queries, context, domain, scenario, guidance)
        results: list[CopilotResult] = []
        for query in queries:
            result = copilot.run(CopilotRequest(query=query, context=context, domain=domain, scenario=scenario))
            store.upsert_graph(result.graph, source=env_source, split='train')
            store.upsert_premise_operator_memory(result.graph, source=env_source, split='train')
            results.append(result)

        approved_count, pending_count = self._auto_review_environment_queries(review_store, results)
        graphs = list(store.fetch_graphs(split='train', source=env_source))

        visual_entities: list[str] = []
        if visual_input and Path(str(visual_input)).exists():
            visual_entities = self._visual_environment_entities(
                context=context,
                visual_input=str(visual_input),
                concept_store_path=concept_store_path,
                operator_store_path=operator_store_path,
                affordance_weights_path=affordance_weights_path,
            )

        training_output_dir = out_dir / 'environment_bundle'
        training_summary = UnifiedSemOpTrainer().train_from_store(
            store_path,
            training_output_dir,
            source=env_source,
            split='train',
            review_store_path=None,
            approved_queries_only=False,
            operating_domain=domain,
        )

        stable_concepts = self._stable_concepts(graphs, visual_entities)
        stable_constraints = self._stable_constraints(graphs)
        routine_patterns = self._routine_patterns(graphs)
        hazard_patterns = self._hazard_patterns(graphs)
        mastery_scores = self._mastery_scores(graphs, stable_concepts, routine_patterns, approved_count, visual_entities)
        next_probes = self._next_probes(mastery_scores, context, domain, scenario, guidance=guidance)
        notes = [
            f'Environment source: {env_source}',
            f'Training graphs: {len(graphs)}',
            f'Augmented graphs: {training_summary.augmented_graph_count}',
        ]
        if guidance.summary:
            notes.append(f'Shared-world solver guidance: {guidance.summary}')
        report_path = out_dir / 'environment_brain_report.json'
        summary = EnvironmentBrainSummary(
            environment_name=env_name,
            environment_source=env_source,
            domain=domain,
            scenario=scenario,
            source_context=context,
            report_path=str(report_path),
            visual_input=str(visual_input or ''),
            seeded_query_count=len(queries),
            stored_graph_count=len(graphs),
            auto_approved_review_count=approved_count,
            pending_review_count=pending_count,
            mastery_scores=mastery_scores,
            stable_concepts=stable_concepts,
            stable_constraints=stable_constraints,
            routine_patterns=routine_patterns,
            hazard_patterns=hazard_patterns,
            visual_entities=visual_entities,
            next_probes=next_probes,
            training=training_summary.model_dump(),
            notes=notes,
        )
        report_path.write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

    @staticmethod
    def _environment_source(environment_name: str, domain: str, scenario: str) -> str:
        raw = f'{environment_name}_{domain}_{scenario}'.strip().lower()
        slug = re.sub(r'[^a-z0-9]+', '_', raw).strip('_') or 'environment'
        return f'environment_brain_{slug}'

    @staticmethod
    def _compose_queries(
        seed_queries: Sequence[str] | None,
        context: str,
        domain: str,
        scenario: str,
        guidance: UnifiedWorldSolverGuidance | None,
    ) -> list[str]:
        queries = list(seed_queries or EnvironmentBrainRunner._default_queries(context, domain, scenario))
        if guidance is not None:
            queries = list(guidance.priority_queries) + queries
            if guidance.hidden_constraints:
                queries.append(
                    f'Which hidden constraint must stay explicit before acting here? Constraint: {guidance.hidden_constraints[0]}'
                )
            if guidance.operator_focus:
                queries.append(
                    f'Which local routine proves operator {guidance.operator_focus[0]} is valid in this environment?'
                )
        return EnvironmentBrainRunner._unique_texts(queries)[:8]

    @staticmethod
    def _default_queries(context: str, domain: str, scenario: str) -> list[str]:
        domain_phrase = str(domain or 'general environment').replace('_', ' ')
        scenario_phrase = str(scenario or 'routine').replace('_', ' ')
        context_hint = str(context or '').strip()
        if len(context_hint) > 180:
            context_hint = context_hint[:180].rsplit(' ', 1)[0] + '...'
        prefix = f'In this {domain_phrase} environment during {scenario_phrase},'
        suffix = f' Context hint: {context_hint}' if context_hint else ''
        return [
            f'{prefix} what should be verified before any action?{suffix}',
            f'{prefix} which hidden constraints most often block safe progress?{suffix}',
            f'{prefix} what routine seems to repeat across tasks here?{suffix}',
            f'{prefix} what hazards or failure modes matter most?{suffix}',
            f'{prefix} what evidence should ground an answer before execution?{suffix}',
            f'{prefix} which access, movement, or object-state checks define this environment?{suffix}',
        ]

    @staticmethod
    def _auto_review_environment_queries(review_store: ReviewQueueStore | None, results: Sequence[CopilotResult]) -> tuple[int, int]:
        if review_store is None:
            return 0, 0
        pending = review_store.fetch_pending(limit=max(20, len(results) * 4))
        pending_by_query = {item.query: item for item in pending}
        approved = 0
        still_pending = 0
        for result in results:
            item = pending_by_query.get(result.request.query)
            if item is None:
                continue
            if EnvironmentBrainRunner._should_auto_approve(result):
                review_store.update_status(item.id, 'approved', resolution_note='environment_brain_auto_approved')
                approved += 1
            else:
                still_pending += 1
        return approved, still_pending

    @staticmethod
    def _should_auto_approve(result: CopilotResult) -> bool:
        kpis = result.kpis
        return (
            float(kpis.invalid_advice_rate) <= 0.15
            and float(kpis.plan_executability) >= 0.72
            and float(kpis.context_misread_rate) <= 0.25
            and float(kpis.human_audit_usefulness) >= 0.62
        )

    def _visual_environment_entities(
        self,
        *,
        context: str,
        visual_input: str,
        concept_store_path: str | None,
        operator_store_path: str | None,
        affordance_weights_path: str | None,
    ) -> list[str]:
        query = 'What stable objects, paths, openings, and affordances define this environment?'
        if context.strip():
            query = f'{query} Context: {context.strip()}'
        world, _answer = VLSOReasoner(
            mode='heuristic',
            concept_store_path=concept_store_path,
            operator_store_path=operator_store_path,
            affordance_weights_path=affordance_weights_path,
            answer_mode='structured',
        ).answer(query, visual_input=visual_input, remember_visual=False)
        labels = []
        for entity in world.entities[:12]:
            label = str(getattr(entity, 'label', '') or getattr(entity, 'id', '')).strip()
            if label:
                labels.append(label)
        return self._unique_texts(labels)[:8]

    @staticmethod
    def _stable_concepts(graphs: Sequence[StructuredMeaningGraph], visual_entities: Sequence[str]) -> list[EnvironmentConceptStat]:
        counts: dict[tuple[str, str], int] = {}
        for graph in graphs:
            for node in graph.nodes:
                label = str(node.label or node.id or '').strip()
                kind = str(node.kind or 'entity').strip()
                if not label:
                    continue
                key = (label.lower(), kind.lower())
                counts[key] = counts.get(key, 0) + 1
        for label in visual_entities:
            key = (str(label).strip().lower(), 'visual_entity')
            counts[key] = counts.get(key, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0][0]))
        return [
            EnvironmentConceptStat(label=label, kind=kind, support_count=count)
            for (label, kind), count in ranked[:8]
        ]

    @staticmethod
    def _stable_constraints(graphs: Sequence[StructuredMeaningGraph]) -> list[str]:
        counts: dict[str, int] = {}
        for graph in graphs:
            for item in list(graph.required_premises) + list(graph.missing_premises) + list(graph.satisfied_premises):
                text = str(item).strip()
                if not text:
                    continue
                counts[text] = counts.get(text, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [text for text, _count in ranked[:8]]

    @staticmethod
    def _routine_patterns(graphs: Sequence[StructuredMeaningGraph]) -> list[EnvironmentRoutineStat]:
        routines: dict[str, EnvironmentRoutineStat] = {}
        for graph in graphs:
            families = [str(item.family).strip() for item in graph.induced_operators[:4] if str(item.family).strip()]
            required = [str(item).strip() for item in graph.required_premises[:4] if str(item).strip()]
            labels = [str(step.action).strip() for step in graph.plan[:4] if str(step.action).strip()]
            labels.extend(str(item).strip() for item in graph.inferred_scripts[:4] if str(item).strip())
            for label in labels:
                routine = routines.get(label)
                if routine is None:
                    routine = EnvironmentRoutineStat(label=label, support_count=0, required_premises=list(required), operator_families=list(families))
                    routines[label] = routine
                routine.support_count += 1
                routine.required_premises = EnvironmentBrainRunner._unique_texts(routine.required_premises + required)[:4]
                routine.operator_families = EnvironmentBrainRunner._unique_texts(routine.operator_families + families)[:4]
        ranked = sorted(routines.values(), key=lambda item: (-item.support_count, item.label))
        return ranked[:6]

    @staticmethod
    def _hazard_patterns(graphs: Sequence[StructuredMeaningGraph]) -> list[str]:
        hazards: list[str] = []
        for graph in graphs:
            hazards.extend(str(item).strip() for item in graph.warnings[:4] if str(item).strip())
            hazards.extend(str(item).strip() for item in graph.missing_premises[:4] if str(item).strip())
            hazards.extend(str(item).strip() for item in graph.invalid_advice[:4] if str(item).strip())
        return EnvironmentBrainRunner._unique_texts(hazards)[:8]

    @staticmethod
    def _mastery_scores(
        graphs: Sequence[StructuredMeaningGraph],
        stable_concepts: Sequence[EnvironmentConceptStat],
        routine_patterns: Sequence[EnvironmentRoutineStat],
        approved_count: int,
        visual_entities: Sequence[str],
    ) -> dict[str, float]:
        graph_count = float(max(1, len(graphs)))
        observation_density = min(1.0, (len(stable_concepts) + len(visual_entities)) / 10.0)
        repeated_routines = sum(1 for item in routine_patterns if int(item.support_count) >= 2)
        routine_stability = min(1.0, repeated_routines / 4.0 if routine_patterns else 0.0)
        safety_terms = []
        grounding_terms = []
        for graph in graphs:
            report = graph.operator_execution
            if report is not None:
                safety_terms.append(float(report.composition_score))
                grounding_terms.append(float(report.claim_grounding_score))
            safety_terms.append(0.0 if graph.invalid_advice else 1.0)
            grounding_edges = sum(1 for edge in graph.edges if edge.relation == 'GROUNDED_BY')
            grounding_terms.append(min(1.0, grounding_edges / 3.0))
        safety_alignment = sum(safety_terms) / float(len(safety_terms) or 1)
        grounding_strength = sum(grounding_terms) / float(len(grounding_terms) or 1)
        self_learning_strength = min(1.0, approved_count / graph_count)
        environment_mastery = (observation_density + routine_stability + safety_alignment + grounding_strength + self_learning_strength) / 5.0
        return {
            'observation_density': round(observation_density, 4),
            'routine_stability': round(routine_stability, 4),
            'safety_alignment': round(safety_alignment, 4),
            'grounding_strength': round(grounding_strength, 4),
            'self_learning_strength': round(self_learning_strength, 4),
            'environment_mastery': round(environment_mastery, 4),
        }

    @staticmethod
    def _next_probes(
        mastery_scores: dict[str, float],
        context: str,
        domain: str,
        scenario: str,
        guidance: UnifiedWorldSolverGuidance | None = None,
    ) -> list[EnvironmentProbe]:
        probes: list[EnvironmentProbe] = []
        if guidance is not None:
            for query in guidance.priority_queries[:2]:
                probes.append(EnvironmentProbe(query=query, purpose='follow shared self-learning plan'))
            if guidance.operator_focus:
                probes.append(EnvironmentProbe(
                    query=f'Which observation would verify operator {guidance.operator_focus[0]} in this environment?',
                    purpose='ground operator focus from the shared world model',
                ))
        if float(mastery_scores.get('grounding_strength', 0.0) or 0.0) < 0.72:
            probes.append(EnvironmentProbe(
                query=f'In this {domain} environment, which exact evidence must ground the next answer before action?',
                purpose='raise grounding fidelity',
            ))
        if float(mastery_scores.get('routine_stability', 0.0) or 0.0) < 0.6:
            probes.append(EnvironmentProbe(
                query=f'What repeated routine defines safe behavior in the {scenario} phase of this environment?',
                purpose='learn repeated local routine',
            ))
        if float(mastery_scores.get('safety_alignment', 0.0) or 0.0) < 0.74:
            probes.append(EnvironmentProbe(
                query=f'What failure mode in this environment should be checked before execution?',
                purpose='strengthen local safety prior',
            ))
        if float(mastery_scores.get('observation_density', 0.0) or 0.0) < 0.7:
            probes.append(EnvironmentProbe(
                query=f'Which objects, openings, paths, or state markers are stable in this environment? Context: {context[:120]}',
                purpose='grow stable environment concepts',
            ))
        if not probes:
            probes.append(EnvironmentProbe(
                query=f'What subtle variation in this {domain} environment would force a different plan next time?',
                purpose='expand environment-specific generalization',
            ))
        return probes[:4]

    @staticmethod
    def _unique_texts(items: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            normalized = str(item or '').strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            ordered.append(normalized)
        return ordered

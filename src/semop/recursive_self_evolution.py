from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

from .adaptive_environment_learning import AdaptiveEnvironmentLearningRunner, AdaptiveEnvironmentLearningSummary
from .concept_fusion import ConceptFusionEngine


@dataclass
class EvolvingImprovementProgram:
    program_id: str
    generation: int
    label: str
    focus_tags: list[str] = field(default_factory=list)
    seed_queries: list[str] = field(default_factory=list)
    self_evolution_rounds: int = 2
    cases_per_round: int = 8
    parent_ids: list[str] = field(default_factory=list)
    mutation_note: str = ""
    score: float = 0.0
    ready_axes: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    remaining_gaps: list[str] = field(default_factory=list)
    report_path: str = ""
    environment_variant: str = ""

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelfEvolutionGenerationSummary:
    generation_index: int
    candidate_count: int = 0
    best_program_id: str = ""
    best_score: float = 0.0
    average_score: float = 0.0
    score_delta: float = 0.0
    candidate_ids: list[str] = field(default_factory=list)
    mutation_notes: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecursiveSelfEvolutionSummary:
    environment_name: str
    domain: str
    scenario: str
    output_dir: str
    report_path: str
    generations: list[SelfEvolutionGenerationSummary] = field(default_factory=list)
    best_program: dict[str, Any] = field(default_factory=dict)
    initial_best_score: float = 0.0
    final_best_score: float = 0.0
    score_delta: float = 0.0
    research_principles: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    deployed_summary: dict[str, Any] = field(default_factory=dict)
    archive: list[dict[str, Any]] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            "environment_name": self.environment_name,
            "domain": self.domain,
            "scenario": self.scenario,
            "output_dir": self.output_dir,
            "report_path": self.report_path,
            "generations": [item.model_dump() for item in self.generations],
            "best_program": dict(self.best_program),
            "initial_best_score": self.initial_best_score,
            "final_best_score": self.final_best_score,
            "score_delta": self.score_delta,
            "research_principles": list(self.research_principles),
            "next_actions": list(self.next_actions),
            "deployed_summary": dict(self.deployed_summary),
            "archive": list(self.archive),
        }


class RecursiveSelfEvolutionRunner:
    def __init__(self, mode: str = 'heuristic') -> None:
        self.mode = mode
        self._fusion = ConceptFusionEngine()

    def run(
        self,
        *,
        environment_name: str,
        domain: str,
        scenario: str,
        context: str,
        store_path: str = 'data/semop_memory.db',
        review_queue_path: str | None = 'data/ops_review_queue.db',
        output_dir: str = 'data/unified_semop_gui_run/recursive_self_evolution',
        visual_input: str | None = None,
        concept_store_path: str | None = None,
        operator_store_path: str | None = None,
        affordance_weights_path: str | None = None,
        generations: int = 3,
        population_size: int = 3,
        children_per_generation: int = 4,
        progress_callback: Any = None,
    ) -> RecursiveSelfEvolutionSummary:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        archive: list[EvolvingImprovementProgram] = []
        generation_summaries: list[SelfEvolutionGenerationSummary] = []
        initial_best_score = 0.0

        seed_programs = self._seed_programs(
            environment_name=environment_name,
            domain=domain,
            scenario=scenario,
            context=context,
            visual_input=visual_input,
            population_size=population_size,
        )
        current_population = seed_programs
        for generation_index in range(1, max(1, int(generations)) + 1):
            if progress_callback:
                progress_callback(
                    min(0.92, 0.06 + ((generation_index - 1) / float(max(1, generations))) * 0.78),
                    f'Running recursive self-evolution generation {generation_index}/{generations}.',
                )
            if generation_index > 1:
                current_population = self._breed_programs(
                    parents=archive[:max(1, min(population_size, len(archive)))],
                    generation=generation_index,
                    domain=domain,
                    scenario=scenario,
                    context=context,
                    visual_input=visual_input,
                    children_per_generation=children_per_generation,
                )
            evaluated = [
                self._evaluate_program(
                    program=program,
                    environment_name=environment_name,
                    domain=domain,
                    scenario=scenario,
                    context=context,
                    store_path=store_path,
                    review_queue_path=review_queue_path,
                    output_root=root / f'generation_{generation_index:02d}',
                    visual_input=visual_input,
                    concept_store_path=concept_store_path,
                    operator_store_path=operator_store_path,
                    affordance_weights_path=affordance_weights_path,
                )
                for program in current_population
            ]
            scored = sorted(evaluated, key=lambda item: item.score, reverse=True)
            if generation_index == 1 and scored:
                initial_best_score = scored[0].score
            previous_best = archive[0].score if archive else 0.0
            archive = self._select_archive(archive + scored, population_size=max(population_size, 4))
            best = archive[0] if archive else None
            average = round(sum(item.score for item in scored) / float(len(scored) or 1), 4)
            generation_summaries.append(
                SelfEvolutionGenerationSummary(
                    generation_index=generation_index,
                    candidate_count=len(scored),
                    best_program_id=best.program_id if best is not None else '',
                    best_score=best.score if best is not None else 0.0,
                    average_score=average,
                    score_delta=round((best.score if best is not None else 0.0) - previous_best, 4),
                    candidate_ids=[item.program_id for item in scored],
                    mutation_notes=[item.mutation_note for item in scored if item.mutation_note][:6],
                )
            )

        best = archive[0] if archive else None
        deployed_summary = AdaptiveEnvironmentLearningRunner(mode=self.mode).run(
            environment_name=environment_name,
            domain=domain,
            scenario=scenario,
            context=context,
            store_path=store_path,
            review_queue_path=review_queue_path,
            output_dir=str(root / 'deployed_best'),
            visual_input=visual_input,
            concept_store_path=concept_store_path,
            operator_store_path=operator_store_path,
            affordance_weights_path=affordance_weights_path,
            seed_queries=list(best.seed_queries) if best is not None else None,
            self_evolution_rounds=int(best.self_evolution_rounds) if best is not None else 2,
            cases_per_round=int(best.cases_per_round) if best is not None else 8,
        )
        final_best_score = self._score_summary(deployed_summary)
        summary = RecursiveSelfEvolutionSummary(
            environment_name=environment_name,
            domain=domain,
            scenario=scenario,
            output_dir=str(root),
            report_path=str(root / 'recursive_self_evolution_report.json'),
            generations=generation_summaries,
            best_program=best.model_dump() if best is not None else {},
            initial_best_score=round(initial_best_score, 4),
            final_best_score=round(final_best_score, 4),
            score_delta=round(final_best_score - initial_best_score, 4),
            research_principles=self._research_principles(),
            next_actions=list(deployed_summary.next_actions[:8]),
            deployed_summary=deployed_summary.model_dump(),
            archive=[item.model_dump() for item in archive],
        )
        Path(summary.report_path).write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

    def _seed_programs(
        self,
        *,
        environment_name: str,
        domain: str,
        scenario: str,
        context: str,
        visual_input: str | None,
        population_size: int,
    ) -> list[EvolvingImprovementProgram]:
        tag_sets: list[list[str]] = [
            ['evidence_guard', 'hazard_guard'],
            ['routine_reuse', 'hidden_constraint'],
            ['evidence_guard', 'action_rehearsal'],
        ]
        if visual_input:
            tag_sets.append(['temporal_alignment', 'action_rehearsal', 'evidence_guard'])
        programs: list[EvolvingImprovementProgram] = []
        for index, tags in enumerate(tag_sets[: max(population_size, len(tag_sets))], start=1):
            label = f'Seed program {index}: ' + ', '.join(tags)
            programs.append(
                EvolvingImprovementProgram(
                    program_id=f'g01_seed_{index:02d}',
                    generation=1,
                    label=label,
                    focus_tags=list(tags),
                    seed_queries=self._build_seed_queries(context, domain, scenario, tags),
                    self_evolution_rounds=2,
                    cases_per_round=8,
                    mutation_note='Initial diverse seed based on environment reasoning, grounding, and action rehearsal.',
                )
            )
        return programs

    def _breed_programs(
        self,
        *,
        parents: Sequence[EvolvingImprovementProgram],
        generation: int,
        domain: str,
        scenario: str,
        context: str,
        visual_input: str | None,
        children_per_generation: int,
    ) -> list[EvolvingImprovementProgram]:
        ranked = sorted(parents, key=lambda item: item.score, reverse=True)
        if not ranked:
            return self._seed_programs(
                environment_name=f'{domain}_{scenario}',
                domain=domain,
                scenario=scenario,
                context=context,
                visual_input=visual_input,
                population_size=max(2, children_per_generation),
            )
        children: list[EvolvingImprovementProgram] = []
        for parent in ranked[:2]:
            children.append(self._mutate_program(parent, generation, domain, scenario, context, visual_input))
        if len(ranked) >= 2:
            children.append(self._recombine_program(ranked[0], ranked[1], generation, domain, scenario, context))
        children.append(self._exploration_program(ranked[0], generation, domain, scenario, context, visual_input))
        deduped: list[EvolvingImprovementProgram] = []
        seen: set[tuple[str, ...]] = set()
        for child in children:
            signature = tuple(sorted(child.focus_tags))
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(child)
        return deduped[: max(2, children_per_generation)]

    def _mutate_program(
        self,
        parent: EvolvingImprovementProgram,
        generation: int,
        domain: str,
        scenario: str,
        context: str,
        visual_input: str | None,
    ) -> EvolvingImprovementProgram:
        tags = set(parent.focus_tags)
        metrics = dict(parent.metrics)
        notes: list[str] = []
        rounds = int(parent.self_evolution_rounds)
        cases = int(parent.cases_per_round)
        if float(metrics.get('grounded_reasoning', 0.0)) < 0.7:
            tags.update({'evidence_guard', 'contradiction_probe'})
            rounds = min(4, rounds + 1)
            notes.append('Grounding was weak, so the mutation added evidence and contradiction probes.')
        if float(metrics.get('embodied_planning', 0.0)) < 0.64:
            tags.add('action_rehearsal')
            notes.append('Embodied planning was weak, so the mutation added action rehearsal focus.')
        if visual_input and float(metrics.get('multimodal_understanding', 0.0)) < 0.52:
            tags.add('temporal_alignment')
            notes.append('Multimodal understanding was weak, so the mutation added temporal alignment.')
        if float(metrics.get('contextual_reasoning', 0.0)) < 0.74:
            tags.add('hidden_constraint')
            notes.append('Context recovery was weak, so the mutation added hidden-constraint probes.')
        if float(metrics.get('self_reflection', 0.0)) < 0.6:
            cases = min(14, cases + 2)
            notes.append('Self-reflection was weak, so the mutation increased local correction pressure.')
        if not notes:
            tags.add('routine_reuse')
            notes.append('The parent was already strong, so the mutation pushed more reusable routine coverage.')
        ordered_tags = sorted(tags)
        return EvolvingImprovementProgram(
            program_id=f'g{generation:02d}_mut_{abs(hash((parent.program_id, tuple(ordered_tags)))) % 10000:04d}',
            generation=generation,
            label='Mutated program: ' + ', '.join(ordered_tags),
            focus_tags=ordered_tags,
            seed_queries=self._build_seed_queries(context, domain, scenario, ordered_tags),
            self_evolution_rounds=rounds,
            cases_per_round=cases,
            parent_ids=[parent.program_id],
            mutation_note=' '.join(notes),
        )

    def _recombine_program(
        self,
        left: EvolvingImprovementProgram,
        right: EvolvingImprovementProgram,
        generation: int,
        domain: str,
        scenario: str,
        context: str,
    ) -> EvolvingImprovementProgram:
        tags = sorted(set(left.focus_tags) | set(right.focus_tags))
        return EvolvingImprovementProgram(
            program_id=f'g{generation:02d}_rec_{abs(hash((left.program_id, right.program_id, tuple(tags)))) % 10000:04d}',
            generation=generation,
            label='Recombined program: ' + ', '.join(tags),
            focus_tags=tags,
            seed_queries=self._build_seed_queries(context, domain, scenario, tags),
            self_evolution_rounds=max(int(left.self_evolution_rounds), int(right.self_evolution_rounds)),
            cases_per_round=max(int(left.cases_per_round), int(right.cases_per_round)),
            parent_ids=[left.program_id, right.program_id],
            mutation_note='Recombined the best two parents to preserve strong ideas while widening focus coverage.',
        )

    def _exploration_program(
        self,
        parent: EvolvingImprovementProgram,
        generation: int,
        domain: str,
        scenario: str,
        context: str,
        visual_input: str | None,
    ) -> EvolvingImprovementProgram:
        tags = {'hazard_guard', 'hidden_constraint', 'contradiction_probe'}
        if visual_input:
            tags.add('temporal_alignment')
        fusion = self._fusion.build_summary(
            prompt=context or f'{domain} {scenario}',
            route='ops',
            prompt_understanding={
                'likely_domain': domain,
                'likely_scenario': scenario,
                'hidden_constraints': list(tags),
                'hidden_context': [parent.label],
            },
            payload={'parent_score': parent.score, 'parent_gaps': parent.remaining_gaps},
        )
        ordered_tags = sorted(tags)
        return EvolvingImprovementProgram(
            program_id=f'g{generation:02d}_exp_{abs(hash((parent.program_id, tuple(ordered_tags)))) % 10000:04d}',
            generation=generation,
            label='Exploration program: ' + ', '.join(ordered_tags),
            focus_tags=ordered_tags,
            seed_queries=self._build_seed_queries(context, domain, scenario, ordered_tags),
            self_evolution_rounds=min(4, int(parent.self_evolution_rounds) + 1),
            cases_per_round=min(14, int(parent.cases_per_round) + 1),
            parent_ids=[parent.program_id],
            mutation_note=(fusion.headline or 'Exploration program built to diversify search and avoid local optima.'),
        )

    def _evaluate_program(
        self,
        *,
        program: EvolvingImprovementProgram,
        environment_name: str,
        domain: str,
        scenario: str,
        context: str,
        store_path: str,
        review_queue_path: str | None,
        output_root: Path,
        visual_input: str | None,
        concept_store_path: str | None,
        operator_store_path: str | None,
        affordance_weights_path: str | None,
    ) -> EvolvingImprovementProgram:
        output_root.mkdir(parents=True, exist_ok=True)
        variant_name = f'{environment_name}_{program.program_id}'
        summary = AdaptiveEnvironmentLearningRunner(mode=self.mode).run(
            environment_name=variant_name,
            domain=domain,
            scenario=scenario,
            context=context,
            store_path=store_path,
            review_queue_path=review_queue_path,
            output_dir=str(output_root / program.program_id),
            visual_input=visual_input,
            concept_store_path=concept_store_path,
            operator_store_path=operator_store_path,
            affordance_weights_path=affordance_weights_path,
            seed_queries=list(program.seed_queries),
            self_evolution_rounds=int(program.self_evolution_rounds),
            cases_per_round=int(program.cases_per_round),
        )
        return EvolvingImprovementProgram(
            program_id=program.program_id,
            generation=program.generation,
            label=program.label,
            focus_tags=list(program.focus_tags),
            seed_queries=list(program.seed_queries),
            self_evolution_rounds=int(program.self_evolution_rounds),
            cases_per_round=int(program.cases_per_round),
            parent_ids=list(program.parent_ids),
            mutation_note=program.mutation_note,
            score=self._score_summary(summary),
            ready_axes=int(summary.ready_axes),
            metrics=dict(summary.capability_scores),
            remaining_gaps=list(summary.remaining_gaps),
            report_path=summary.report_path,
            environment_variant=variant_name,
        )

    @staticmethod
    def _score_summary(summary: AdaptiveEnvironmentLearningSummary) -> float:
        metrics = dict(summary.capability_scores)
        integrated_reasoning = getattr(summary, 'integrated_reasoning', {}) or {}
        approved_bonus = min(1.0, float(summary.approved_review_count) / 10.0)
        ready_bonus = min(1.0, float(summary.ready_axes) / float(summary.total_axes or 1))
        integration_bonus = float(metrics.get('world_model_integration', 0.0) or 0.0)
        if integration_bonus <= 0.0 and isinstance(integrated_reasoning, dict):
            integration_bonus = min(
                1.0,
                (min(1.0, len(integrated_reasoning.get('active_domains', []) or []) / 3.0) * 0.2)
                + (min(1.0, len(integrated_reasoning.get('blockers', []) or []) / 3.0) * 0.2)
                + (min(1.0, len(integrated_reasoning.get('prerequisites', []) or []) / 3.0) * 0.2)
                + (min(1.0, len(integrated_reasoning.get('evidence', []) or []) / 4.0) * 0.2)
                + (min(1.0, len(integrated_reasoning.get('next_steps', []) or []) / 3.0) * 0.2)
            )
        score = (
            float(metrics.get('local_intelligence', 0.0)) * 0.27
            + float(metrics.get('grounded_reasoning', 0.0)) * 0.15
            + float(metrics.get('self_reflection', 0.0)) * 0.12
            + float(metrics.get('embodied_planning', 0.0)) * 0.12
            + float(metrics.get('contextual_reasoning', 0.0)) * 0.1
            + float(metrics.get('multimodal_understanding', 0.0)) * 0.06
            + integration_bonus * 0.07
            + approved_bonus * 0.05
            + ready_bonus * 0.06
        )
        return round(min(1.0, score), 4)

    @staticmethod
    def _select_archive(programs: Sequence[EvolvingImprovementProgram], population_size: int) -> list[EvolvingImprovementProgram]:
        ranked = sorted(programs, key=lambda item: (item.score, item.ready_axes, len(item.focus_tags)), reverse=True)
        selected: list[EvolvingImprovementProgram] = []
        for program in ranked:
            if len(selected) >= max(1, population_size):
                break
            if not selected:
                selected.append(program)
                continue
            if all(RecursiveSelfEvolutionRunner._jaccard(program.focus_tags, existing.focus_tags) < 0.85 for existing in selected):
                selected.append(program)
        if len(selected) < max(1, population_size):
            for program in ranked:
                if len(selected) >= max(1, population_size):
                    break
                if program not in selected:
                    selected.append(program)
        return selected

    @staticmethod
    def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
        a = set(left)
        b = set(right)
        if not a and not b:
            return 1.0
        return len(a & b) / float(len(a | b) or 1)

    @staticmethod
    def _build_seed_queries(context: str, domain: str, scenario: str, focus_tags: Sequence[str]) -> list[str]:
        domain_phrase = str(domain or 'general environment').replace('_', ' ')
        scenario_phrase = str(scenario or 'routine').replace('_', ' ')
        hint = str(context or '').strip()
        if len(hint) > 180:
            hint = hint[:180].rsplit(' ', 1)[0] + '...'
        prefix = f'In this {domain_phrase} environment during {scenario_phrase},'
        suffix = f' Context hint: {hint}' if hint else ''
        templates = {
            'evidence_guard': f'{prefix} what exact evidence must be confirmed before action?{suffix}',
            'hazard_guard': f'{prefix} which failure mode breaks safe progress first?{suffix}',
            'routine_reuse': f'{prefix} which repeated routine should be reused instead of improvising?{suffix}',
            'hidden_constraint': f'{prefix} which hidden constraint is most likely to be missed by a first answer?{suffix}',
            'contradiction_probe': f'{prefix} if the first answer fails, which assumption is probably wrong?{suffix}',
            'temporal_alignment': f'{prefix} what changes over time and how should the action order adapt?{suffix}',
            'action_rehearsal': f'{prefix} what safe action order should be rehearsed before execution?{suffix}',
        }
        queries = [templates[tag] for tag in focus_tags if tag in templates]
        if not queries:
            queries.append(f'{prefix} what must be checked before acting?{suffix}')
        deduped: list[str] = []
        for query in queries:
            text = str(query).strip()
            if text and text not in deduped:
                deduped.append(text)
        return deduped[:8]

    @staticmethod
    def _research_principles() -> list[str]:
        return [
            'DreamerV3 and MuZero: improve policy quality by keeping a learned world model in the loop, not just a raw action trace.',
            'FunSearch and AlphaEvolve: keep a searchable population of programs and let evaluators decide what survives.',
            'Voyager, Reflexion, and Self-Refine: preserve reflection memory and reusable skills so failed attempts mutate the next proposal.',
            'RTX 4060 constraint: evolve symbolic improvement programs and evaluators first, then retrain only the best local bundle.',
        ]

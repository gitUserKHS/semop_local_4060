from __future__ import annotations

import json
import os

from dataclasses import asdict, dataclass, field
from typing import Any, List

from .hard_problem_engine import HardProblemEngine
from .leworldmodel_adapter import LeWorldModelAdapter
from .leworldmodel_planner import LeWorldModelTrajectoryPlanner
from .pipeline import StructuredMeaningPipeline
from .structures import StructuredMeaningGraph
from .vlso import VLSOReasoner
from .vlso.types import SharedWorldModel


GEOMETRY_TOKENS = (
    'triangle', 'circle', 'parallel', 'perpendicular', 'angle', 'angles', 'line', 'point', 'points',
    'quadrilateral', 'incircle', 'circumcircle', 'midpoint', 'bisector', 'chord', 'tangent',
)
NUMBER_THEORY_TOKENS = ('prime', 'divisible', 'mod', 'modulo', 'gcd', 'integer', 'remainder', 'coprime')
ALGEBRA_TOKENS = ('inequality', 'polynomial', 'factor', 'symmetric', 'substitution', 'equation')
COMBINATORICS_TOKENS = ('subset', 'coloring', 'count', 'arrangement', 'pigeonhole', 'invariant')
COMPUTATIONAL_GEOMETRY_TOKENS = ('convex hull', 'polygon', 'distance', 'cross product', 'orientation', 'segment', 'intersect')
RESEARCH_TOKENS = ('research', 'open problem', 'lemma', 'theorem', 'imo', 'olympiad', 'prove', 'show that')


@dataclass
class MathStrategyPrior:
    family: str
    difficulty: str
    recommended_operators: List[str] = field(default_factory=list)
    proposed_subgoals: List[str] = field(default_factory=list)
    latent_objects: List[str] = field(default_factory=list)
    latent_constraints: List[str] = field(default_factory=list)
    world_model_notes: List[str] = field(default_factory=list)
    research_anchors: List[str] = field(default_factory=list)
    leworldmodel_alignment: dict[str, Any] = field(default_factory=dict)
    leworldmodel_plan: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MathWorldCandidate:
    source: str
    kind: str
    answer: str
    confidence: float
    verification_score: float
    alignment_score: float
    score: float
    verified: bool
    support: List[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MathWorldCheck:
    name: str
    passed: bool
    score: float
    detail: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorldModelMathReport:
    query: str
    task_mode: str
    graph: StructuredMeaningGraph
    strategy_prior: MathStrategyPrior
    candidates: List[MathWorldCandidate]
    checks: List[MathWorldCheck]
    chosen_answer: str
    solved: bool
    verification_score: float
    beginner_summary: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)
    solution_process: List[str] = field(default_factory=list)
    operator_trace: List[str] = field(default_factory=list)
    matched_patterns: List[str] = field(default_factory=list)
    visual_world_model: SharedWorldModel | None = None
    visual_answer: dict[str, Any] | None = None
    leworldmodel_alignment: dict[str, Any] = field(default_factory=dict)
    leworldmodel_plan: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {
            'query': self.query,
            'task_mode': self.task_mode,
            'graph': self.graph.model_dump(),
            'strategy_prior': self.strategy_prior.model_dump(),
            'candidates': [item.model_dump() for item in self.candidates],
            'checks': [item.model_dump() for item in self.checks],
            'chosen_answer': self.chosen_answer,
            'solved': self.solved,
            'verification_score': self.verification_score,
            'beginner_summary': list(self.beginner_summary),
            'next_actions': list(self.next_actions),
            'solution_process': list(self.solution_process),
            'operator_trace': list(self.operator_trace),
            'matched_patterns': list(self.matched_patterns),
            'visual_world_model': self.visual_world_model.model_dump() if self.visual_world_model is not None else None,
            'visual_answer': dict(self.visual_answer or {}),
            'leworldmodel_alignment': dict(self.leworldmodel_alignment),
            'leworldmodel_plan': dict(self.leworldmodel_plan),
        }


class WorldModelMathReasoner:
    def __init__(
        self,
        *,
        mode: str = 'heuristic',
        model_id: str = 'Qwen/Qwen2.5-3B-Instruct',
        concept_store_path: str | None = None,
        operator_store_path: str | None = None,
        affordance_weights_path: str | None = None,
        logical_weight_path: str | None = 'data/logical_pattern_weights.json',
        strategy_memory_path: str | None = None,
        leworldmodel_path: str | None = None,
    ) -> None:
        self.strategy_memory_path = strategy_memory_path
        self.leworldmodel_path = leworldmodel_path
        self.strategy_memory = self._load_strategy_memory(strategy_memory_path)
        self.lewm = LeWorldModelAdapter()
        self.lewm_planner = LeWorldModelTrajectoryPlanner(adapter=self.lewm)
        self.leworldmodel_artifact = self.lewm.load(leworldmodel_path)
        self.pipeline = StructuredMeaningPipeline(mode=mode, model_id=model_id, logical_weight_path=logical_weight_path)
        self.hard_engine = HardProblemEngine(mode=mode, model_id=model_id, logical_weight_path=logical_weight_path)
        self.visual_reasoner = VLSOReasoner(
            mode='deep',
            answer_mode='structured',
            concept_store_path=concept_store_path,
            operator_store_path=operator_store_path,
            affordance_weights_path=affordance_weights_path,
        )

    def solve(
        self,
        query: str,
        *,
        source_context: str = '',
        visual_input: Any | None = None,
        task_mode: str = 'auto',
    ) -> WorldModelMathReport:
        graph = self.pipeline.run(query, source_context=source_context, visual_input=visual_input)
        hard_report = self.hard_engine.solve(query)
        visual_world: SharedWorldModel | None = None
        visual_answer_payload: dict[str, Any] | None = None
        visual_notes: list[str] = []
        if visual_input is not None:
            try:
                visual_world, visual_answer = self.visual_reasoner.answer(query, visual_input=visual_input)
                visual_answer_payload = visual_answer.model_dump()
            except Exception as exc:
                visual_notes.append(f'visual world model unavailable: {exc}')
        strategy_prior = self._build_strategy_prior(query, graph, visual_world, task_mode, visual_notes)
        lewm_alignment = self._score_leworldmodel_alignment(query, strategy_prior, graph, visual_world)
        strategy_prior.leworldmodel_alignment = dict(lewm_alignment)
        lewm_plan = self._plan_leworldmodel_trajectory(query, strategy_prior, graph, visual_world)
        strategy_prior.leworldmodel_plan = dict(lewm_plan)
        candidates = self._build_candidates(strategy_prior, hard_report, visual_world, visual_answer_payload, lewm_alignment, lewm_plan)
        if not candidates:
            candidates.append(
                MathWorldCandidate(
                    source='fallback',
                    kind='strategy_only',
                    answer='No verified answer was produced yet; follow the strategy prior and strongest next action.',
                    confidence=0.35,
                    verification_score=0.35,
                    alignment_score=0.4,
                    score=0.37,
                    verified=False,
                    support=['strategy-only fallback'],
                )
            )
        candidates.sort(key=lambda item: (-item.score, -item.verification_score, -item.confidence, item.source))
        chosen = candidates[0]
        checks = self._build_checks(graph, strategy_prior, chosen, hard_report, visual_world)
        verification_score = round(sum(item.score for item in checks) / float(len(checks) or 1), 4)
        solved = bool(chosen.verified or (chosen.score >= 0.74 and verification_score >= 0.7))
        beginner_summary = self._beginner_summary(strategy_prior, chosen, visual_world)
        next_actions = self._next_actions(strategy_prior, solved, visual_world)
        solution_process, operator_trace = self._solution_trace(graph, chosen, visual_world)
        return WorldModelMathReport(
            query=query,
            task_mode=task_mode,
            graph=graph,
            strategy_prior=strategy_prior,
            candidates=candidates,
            checks=checks,
            chosen_answer=chosen.answer,
            solved=solved,
            verification_score=verification_score,
            beginner_summary=beginner_summary,
            next_actions=next_actions,
            solution_process=solution_process,
            operator_trace=operator_trace,
            matched_patterns=list(hard_report.matched_patterns),
            visual_world_model=visual_world,
            visual_answer=visual_answer_payload,
            leworldmodel_alignment=lewm_alignment,
            leworldmodel_plan=lewm_plan,
        )

    def _build_strategy_prior(
        self,
        query: str,
        graph: StructuredMeaningGraph,
        visual_world: SharedWorldModel | None,
        task_mode: str,
        visual_notes: list[str],
    ) -> MathStrategyPrior:
        family = self._resolve_family(query, task_mode)
        difficulty = self._difficulty_label(query, family)
        latent_objects = [node.label for node in graph.nodes[:8] if node.label]
        if visual_world is not None:
            for entity in visual_world.entities[:6]:
                if entity.id not in latent_objects:
                    latent_objects.append(entity.id)
        latent_constraints = list(dict.fromkeys(
            [item for item in graph.hidden_goals if item]
            + [item for item in graph.required_premises if item]
            + ([item for item in visual_world.constraints] if visual_world is not None else [])
        ))[:10]
        recommended = self._recommended_operators(family, visual_world)
        subgoals = self._subgoals(family, visual_world)
        recommended, subgoals, learned_notes = self._apply_strategy_memory(family, recommended, subgoals)
        notes = [
            'Predict the latent proof or solution state first, then ask the solver ensemble.',
            'Use the world model to recover objects, constraints, and likely operators before choosing an answer.',
            'Keep the candidate that best matches the latent strategy and verification checks, not the first candidate.',
        ]
        notes.extend(visual_notes)
        notes.extend(learned_notes)
        if self.leworldmodel_artifact is not None:
            notes.append('LeWM-inspired latent prior is active: next-state prediction and stable latent geometry are reused during reranking.')
        anchors = [
            'LeWorldModel-style latent prediction: prefer stable next-embedding prediction over surface reconstruction when scoring candidate trajectories.',
            'AlphaGeometry-style hybrid reasoning: combine learned priors with symbolic proof search and constructions.',
            'AlphaProof-style verification: prefer candidates that keep an explicit, checkable trace.',
        ]
        return MathStrategyPrior(
            family=family,
            difficulty=difficulty,
            recommended_operators=recommended,
            proposed_subgoals=subgoals,
            latent_objects=latent_objects,
            latent_constraints=latent_constraints,
            world_model_notes=notes,
            research_anchors=anchors,
            leworldmodel_alignment={},
            leworldmodel_plan={},
        )

    def _score_leworldmodel_alignment(
        self,
        query: str,
        prior: MathStrategyPrior,
        graph: StructuredMeaningGraph,
        visual_world: SharedWorldModel | None,
    ) -> dict[str, Any]:
        if self.leworldmodel_artifact is None:
            return {}
        step_items = [
            [query],
            [prior.family, prior.difficulty] + list(prior.recommended_operators),
            list(prior.proposed_subgoals) + list(prior.latent_constraints) + [node.label for node in graph.nodes[:6] if node.label],
        ]
        if visual_world is not None:
            step_items.append([entity.id for entity in visual_world.entities[:6]] + [relation.relation for relation in visual_world.relations[:6]])
        record = self.lewm.make_record('math_probe', 'math_world_model', step_items)
        alignment = self.lewm.score_sequence(self.leworldmodel_artifact, record.steps)
        return alignment.model_dump()

    def _build_candidates(
        self,
        prior: MathStrategyPrior,
        hard_report: Any,
        visual_world: SharedWorldModel | None,
        visual_answer_payload: dict[str, Any] | None,
        lewm_alignment: dict[str, Any],
        lewm_plan: dict[str, Any],
    ) -> list[MathWorldCandidate]:
        candidates: list[MathWorldCandidate] = []
        for item in hard_report.candidates:
            alignment = self._candidate_alignment(prior, item.kind, item.source, visual_world)
            lewm_candidate_alignment = self._candidate_leworldmodel_alignment(
                prior,
                candidate_kind=item.kind,
                candidate_source=item.source,
                candidate_answer=item.answer,
                candidate_support=[item.detail, *list(hard_report.matched_patterns[:4])],
                visual_world=visual_world,
                base_alignment=lewm_alignment,
            )
            plan_bonus = 0.08 * self._plan_alignment_bonus(lewm_plan, item.kind, item.source, item.answer, [item.detail, *list(hard_report.matched_patterns[:4])])
            score = round(
                min(
                    0.99,
                    (0.48 * float(item.verification_score))
                    + (0.20 * float(item.confidence))
                    + (0.16 * alignment)
                    + (0.10 * lewm_candidate_alignment)
                    + plan_bonus,
                ),
                4,
            )
            support = [item.detail]
            if lewm_alignment:
                support.append(f"lewm global alignment={float(lewm_alignment.get('alignment_score', 0.0) or 0.0):.3f}")
            if self.leworldmodel_artifact is not None:
                support.append(f'lewm candidate alignment={lewm_candidate_alignment:.3f}')
                if lewm_plan:
                    support.append(f"lewm plan bonus={plan_bonus:.3f}")
            if hard_report.matched_patterns:
                support.append('matched patterns: ' + ', '.join(hard_report.matched_patterns[:4]))
            candidates.append(
                MathWorldCandidate(
                    source=item.source,
                    kind=item.kind,
                    answer=item.answer,
                    confidence=round(float(item.confidence), 4),
                    verification_score=round(float(item.verification_score), 4),
                    alignment_score=round(max(alignment, lewm_candidate_alignment), 4),
                    score=score,
                    verified=bool(item.verified),
                    support=support,
                )
            )
        if visual_world is not None:
            diagram_answer = self._diagram_summary(visual_world, visual_answer_payload)
            base_score = 0.78 if visual_world.operators or visual_world.relations else 0.46
            alignment = self._candidate_alignment(prior, 'diagram_world_model', 'vlso_reasoner', visual_world)
            lewm_candidate_alignment = self._candidate_leworldmodel_alignment(
                prior,
                candidate_kind='diagram_world_model',
                candidate_source='vlso_reasoner',
                candidate_answer=diagram_answer,
                candidate_support=['diagram-backed world model', *list(visual_world.inferred_steps[:4])],
                visual_world=visual_world,
                base_alignment=lewm_alignment,
            )
            plan_bonus = 0.08 * self._plan_alignment_bonus(lewm_plan, 'diagram_world_model', 'vlso_reasoner', diagram_answer, ['diagram-backed world model', *list(visual_world.inferred_steps[:4])])
            score = round(
                min(
                    0.99,
                    (0.48 * base_score)
                    + (0.20 * base_score)
                    + (0.16 * alignment)
                    + (0.12 * lewm_candidate_alignment)
                    + plan_bonus,
                ),
                4,
            )
            candidates.append(
                MathWorldCandidate(
                    source='vlso_reasoner',
                    kind='diagram_world_model',
                    answer=diagram_answer,
                    confidence=round(base_score, 4),
                    verification_score=round(base_score, 4),
                    alignment_score=round(max(alignment, lewm_candidate_alignment), 4),
                    score=score,
                    verified=base_score >= 0.72 and prior.family == 'geometry_proof',
                    support=[
                        'diagram-backed world model',
                        f'lewm candidate alignment={lewm_candidate_alignment:.3f}',
                        f'lewm plan bonus={plan_bonus:.3f}',
                    ],
                )
            )
        return candidates

    def _plan_leworldmodel_trajectory(
        self,
        query: str,
        prior: MathStrategyPrior,
        graph: StructuredMeaningGraph,
        visual_world: SharedWorldModel | None,
    ) -> dict[str, Any]:
        if self.leworldmodel_artifact is None:
            return {}
        candidate_actions = list(dict.fromkeys(
            list(prior.recommended_operators[:6])
            + list(prior.proposed_subgoals[:6])
            + list(prior.latent_constraints[:4])
            + ([relation.relation for relation in visual_world.relations[:4]] if visual_world is not None else [])
        ))
        initial_steps = [
            [query],
            [prior.family, prior.difficulty],
            list(prior.latent_constraints[:4]) + [node.label for node in graph.nodes[:4] if node.label],
        ]
        goal_tokens = list(prior.proposed_subgoals[:4]) + list(prior.latent_constraints[:4])
        plan = self.lewm_planner.plan(
            self.leworldmodel_artifact,
            domain='math_world_model',
            initial_steps=initial_steps,
            candidate_actions=candidate_actions,
            goal_tokens=goal_tokens,
            horizon=min(4, max(2, len(candidate_actions) // 2 or 2)),
            samples=24,
            elites=6,
            iterations=4,
            seed=len(query) + len(candidate_actions),
        )
        return plan.model_dump()

    @staticmethod
    def _plan_alignment_bonus(
        lewm_plan: dict[str, Any],
        candidate_kind: str,
        candidate_source: str,
        candidate_answer: str,
        candidate_support: list[str],
    ) -> float:
        if not isinstance(lewm_plan, dict) or not lewm_plan.get('plan_tokens'):
            return 0.0
        plan_tokens = {str(item).lower() for item in lewm_plan.get('plan_tokens', []) if str(item).strip()}
        probe = ' '.join([candidate_kind, candidate_source, candidate_answer, *candidate_support]).lower()
        hit_count = sum(1 for token in plan_tokens if token and token.lower() in probe)
        return min(1.0, hit_count / float(len(plan_tokens) or 1))

    def _candidate_leworldmodel_alignment(
        self,
        prior: MathStrategyPrior,
        *,
        candidate_kind: str,
        candidate_source: str,
        candidate_answer: str,
        candidate_support: list[str],
        visual_world: SharedWorldModel | None,
        base_alignment: dict[str, Any],
    ) -> float:
        global_alignment = float(base_alignment.get('alignment_score', 0.0) or 0.0) if isinstance(base_alignment, dict) else 0.0
        if self.leworldmodel_artifact is None:
            return round(global_alignment, 4)
        step_items = [
            [prior.family, prior.difficulty] + list(prior.recommended_operators[:4]),
            list(prior.proposed_subgoals[:4]) + list(prior.latent_constraints[:4]),
            [candidate_kind, candidate_source, candidate_answer] + list(candidate_support[:4]),
        ]
        if visual_world is not None:
            step_items.append([relation.relation for relation in visual_world.relations[:6]] + [entity.id for entity in visual_world.entities[:6]])
        record = self.lewm.make_record('candidate_probe', 'math_world_model', step_items)
        alignment = self.lewm.score_sequence(self.leworldmodel_artifact, record.steps)
        blended = (0.65 * float(alignment.alignment_score)) + (0.35 * global_alignment)
        return round(min(0.999, blended), 4)

    def _build_checks(
        self,
        graph: StructuredMeaningGraph,
        prior: MathStrategyPrior,
        chosen: MathWorldCandidate,
        hard_report: Any,
        visual_world: SharedWorldModel | None,
    ) -> list[MathWorldCheck]:
        checks = [
            MathWorldCheck(name=item.name, passed=bool(item.passed), score=round(float(item.score), 4), detail=item.detail)
            for item in hard_report.checks
        ]
        latent_score = 0.35
        if prior.latent_objects:
            latent_score += 0.15
        if prior.latent_constraints:
            latent_score += 0.2
        if graph.plan:
            latent_score += 0.15
        checks.append(
            MathWorldCheck(
                name='latent_structure_check',
                passed=latent_score >= 0.6,
                score=round(min(0.95, latent_score), 4),
                detail='Checks whether the world model recovered objects, constraints, and a usable plan before answer selection.',
            )
        )
        checks.append(
            MathWorldCheck(
                name='strategy_alignment_check',
                passed=chosen.alignment_score >= 0.58,
                score=round(chosen.alignment_score, 4),
                detail=f"Checks whether the chosen candidate matches the inferred family `{prior.family}`.",
            )
        )
        if visual_world is not None:
            diagram_score = 0.4
            if visual_world.operators:
                diagram_score += 0.2
            if visual_world.relations:
                diagram_score += 0.2
            if visual_world.inferred_steps:
                diagram_score += 0.1
            checks.append(
                MathWorldCheck(
                    name='diagram_consistency_check',
                    passed=diagram_score >= 0.65,
                    score=round(min(0.95, diagram_score), 4),
                    detail='Checks whether the diagram world model recovered meaningful geometry operators or relations.',
                )
            )
        return checks

    def _beginner_summary(
        self,
        prior: MathStrategyPrior,
        chosen: MathWorldCandidate,
        visual_world: SharedWorldModel | None,
    ) -> list[str]:
        rows = [
            f"1. Treat the problem as `{prior.family}` and recover latent objects and constraints first.",
            f"2. Start from operators like {', '.join(prior.recommended_operators[:4]) or 'structured decomposition'}.",
            f"3. Keep `{chosen.source}` only because its candidate best matches the strategy prior and verification checks.",
        ]
        lewm_alignment = float(prior.leworldmodel_alignment.get('alignment_score', 0.0) or 0.0) if isinstance(prior.leworldmodel_alignment, dict) else 0.0
        if prior.leworldmodel_alignment:
            rows.append(f'4. Check the LeWM latent prior: the current proof trajectory alignment is {lewm_alignment:.3f}, so use it as a reranking hint rather than as a proof by itself.')
        if prior.leworldmodel_plan:
            planned = ', '.join(str(item) for item in prior.leworldmodel_plan.get('plan_tokens', [])[:4]) or '-'
            rows.append(f'{len(rows) + 1}. Follow the latent plan first: {planned}.')
        if visual_world is not None:
            visual_index = len(rows) + 1
            rows.append(f'{visual_index}. Use the diagram world model as an extra proof hint instead of trusting raw pixels directly.')
        return rows

    def _next_actions(self, prior: MathStrategyPrior, solved: bool, visual_world: SharedWorldModel | None) -> list[str]:
        if solved:
            return [
                'Inspect the strongest candidate and its checks before trusting the final write-up.',
                'If needed, rerun with extra givens or a cleaner diagram to raise verification further.',
            ]
        actions = [
            'Add missing givens, lemmas, or explicit constraints in the context box.',
            'Try a narrower task mode if auto-routing picked the wrong mathematical family.',
        ]
        if prior.family == 'geometry_proof' and visual_world is None:
            actions.append('Attach a geometry diagram JSON or image so the world model can recover geometric relations.')
        if prior.family in {'research_math', 'number_theory_proof', 'algebra_proof', 'combinatorics_proof'}:
            actions.append('Break the problem into lemmas and rerun with one lemma or subgoal at a time.')
        return actions

    @staticmethod
    def _load_strategy_memory(path: str | None) -> dict[str, Any]:
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _apply_strategy_memory(
        self,
        family: str,
        recommended: list[str],
        subgoals: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        families = self.strategy_memory.get('families', {}) if isinstance(self.strategy_memory, dict) else {}
        family_payload = families.get(family, {}) if isinstance(families, dict) else {}
        learned_ops = list(family_payload.get('top_operators', [])) if isinstance(family_payload, dict) else []
        learned_hints = list(family_payload.get('top_hints', [])) if isinstance(family_payload, dict) else []
        learned_count = int(family_payload.get('successful_cases', 0)) if isinstance(family_payload, dict) else 0
        for operator in reversed(learned_ops):
            if operator in recommended:
                recommended.remove(operator)
            recommended.insert(0, operator)
        merged_recommended = list(dict.fromkeys(recommended))[:6]
        merged_subgoals = list(dict.fromkeys(subgoals + learned_hints))[:6]
        notes: list[str] = []
        if learned_count > 0:
            notes.append(f'Local training memory recovered {learned_count} successful `{family}` cases and reprioritized operators accordingly.')
        return merged_recommended, merged_subgoals, notes

    @staticmethod
    def _solution_trace(
        graph: StructuredMeaningGraph,
        chosen: MathWorldCandidate,
        visual_world: SharedWorldModel | None,
    ) -> tuple[list[str], list[str]]:
        process: list[str] = []
        operator_trace: list[str] = []
        for item in graph.symbolic_results:
            if item.evidence:
                process.extend(str(entry) for entry in item.evidence[:8])
            if item.equations:
                operator_trace.extend(str(entry) for entry in item.equations[:8])
        if visual_world is not None and getattr(visual_world, 'inferred_steps', None):
            process.extend(str(entry) for entry in list(visual_world.inferred_steps)[:6])
        if not process:
            process.extend(chosen.support[:6])
        if not operator_trace and chosen.support:
            operator_trace.extend(chosen.support[:4])
        return list(dict.fromkeys(process))[:10], list(dict.fromkeys(operator_trace))[:10]

    def _resolve_family(self, query: str, task_mode: str) -> str:
        if task_mode and task_mode != 'auto':
            return task_mode
        normalized = query.lower()
        if any(token in normalized for token in COMPUTATIONAL_GEOMETRY_TOKENS):
            return 'computational_geometry'
        if any(token in normalized for token in GEOMETRY_TOKENS):
            return 'geometry_proof'
        if any(token in normalized for token in NUMBER_THEORY_TOKENS):
            return 'number_theory_proof'
        if any(token in normalized for token in ALGEBRA_TOKENS):
            return 'algebra_proof'
        if any(token in normalized for token in COMBINATORICS_TOKENS):
            return 'combinatorics_proof'
        if any(token in normalized for token in RESEARCH_TOKENS):
            return 'research_math'
        return 'generic_math'

    @staticmethod
    def _difficulty_label(query: str, family: str) -> str:
        normalized = query.lower()
        if 'research' in normalized or 'open problem' in normalized:
            return 'research-grade'
        if any(token in normalized for token in ('imo', 'olympiad', 'theorem', 'lemma', 'prove', 'show that')):
            return 'olympiad-grade'
        if family == 'computational_geometry':
            return 'advanced-contest'
        return 'advanced'

    @staticmethod
    def _recommended_operators(family: str, visual_world: SharedWorldModel | None) -> list[str]:
        base = {
            'geometry_proof': ['AUXILIARY_CONSTRUCTION', 'ANGLE_CHASE', 'SIMILAR_TRIANGLES', 'AREA_RATIO'],
            'computational_geometry': ['ORIENTATION_TEST', 'CROSS_PRODUCT', 'SEGMENT_INTERSECTION', 'CONVEXITY_CHECK'],
            'number_theory_proof': ['MODULAR_REWRITE', 'DIVISIBILITY_SPLIT', 'CONTRADICTION_SETUP', 'PARITY_REWRITE'],
            'algebra_proof': ['SUBSTITUTION', 'FACTORIZATION', 'SYMMETRY_REWRITE', 'INEQUALITY_NORMALIZATION'],
            'combinatorics_proof': ['PIGEONHOLE', 'INVARIANT_TRACKING', 'DOUBLE_COUNTING', 'EXTREMAL_CHOICE'],
            'research_math': ['LEMMA_DECOMPOSITION', 'CASE_SPLIT', 'COUNTEREXAMPLE_SCAN', 'PROOF_STATE_CHECKPOINT'],
            'generic_math': ['GOAL_RESTATEMENT', 'SUBGOAL_DECOMPOSITION', 'INVARIANT_SCAN', 'CONTRADICTION_SETUP'],
        }.get(family, ['GOAL_RESTATEMENT', 'SUBGOAL_DECOMPOSITION'])
        if visual_world is not None:
            operator_names = {item.name for item in visual_world.operators}
            if any('PARALLEL' in name for name in operator_names) and 'PARALLEL_LINE_REASONING' not in base:
                base.append('PARALLEL_LINE_REASONING')
            if any('PERPENDICULAR' in name for name in operator_names) and 'RIGHT_ANGLE_REASONING' not in base:
                base.append('RIGHT_ANGLE_REASONING')
        return base[:6]

    @staticmethod
    def _subgoals(family: str, visual_world: SharedWorldModel | None) -> list[str]:
        goals = {
            'geometry_proof': ['identify the key invariant or angle relation', 'search for an auxiliary line', 'look for triangle similarity or area ratios'],
            'computational_geometry': ['recover coordinate objects', 'choose the robust predicate', 'verify degeneracy or boundary cases'],
            'number_theory_proof': ['rewrite the claim in modular or divisibility form', 'split into parity or residue cases', 'search for a contradiction'],
            'algebra_proof': ['normalize the expression', 'search for symmetry', 'try a factorization or substitution'],
            'combinatorics_proof': ['count the right objects', 'find an invariant or pigeonhole bottleneck', 'isolate a minimal obstruction'],
            'research_math': ['turn the statement into lemmas', 'check small examples and counterexamples', 'verify each lemma before composing the final proof'],
            'generic_math': ['restate the target cleanly', 'split the problem into lemmas', 'check which invariant survives each step'],
        }.get(family, ['restate the target', 'split the problem into subgoals'])
        if visual_world is not None and visual_world.relations:
            goals.append('use the recovered diagram relations as explicit proof hints')
        return goals[:6]

    @staticmethod
    def _candidate_alignment(
        prior: MathStrategyPrior,
        kind: str,
        source: str,
        visual_world: SharedWorldModel | None,
    ) -> float:
        score = 0.42
        family = prior.family
        lowered_kind = (kind or '').lower()
        lowered_source = (source or '').lower()
        if family == 'geometry_proof' and any(token in lowered_kind for token in ('geometry', 'olympiad', 'diagram')):
            score += 0.26
        if family == 'computational_geometry' and any(token in lowered_kind for token in ('contest', 'geometry')):
            score += 0.28
        if family == 'number_theory_proof' and 'olympiad' in lowered_kind:
            score += 0.18
        if family == 'research_math' and any(token in lowered_kind for token in ('olympiad', 'proof', 'document')):
            score += 0.14
        if 'vlso' in lowered_source or lowered_kind == 'diagram_world_model':
            score += 0.1 if family == 'geometry_proof' else 0.02
        if visual_world is not None and family == 'geometry_proof' and (visual_world.operators or visual_world.relations):
            score += 0.08
        return round(min(0.99, score), 4)

    @staticmethod
    def _diagram_summary(world: SharedWorldModel, visual_answer_payload: dict[str, Any] | None) -> str:
        entities = ', '.join(item.id for item in world.entities[:6]) or 'none'
        relations = ' | '.join(f"{item.source}-{item.relation}-{item.target}" for item in world.relations[:6]) or 'none'
        operators = ', '.join(item.name for item in world.operators[:6]) or 'none'
        answer_text = ''
        if isinstance(visual_answer_payload, dict):
            answer_text = str(visual_answer_payload.get('answer_text') or '')
        summary = f"Diagram world model: entities={entities}; relations={relations}; operators={operators}."
        if answer_text:
            summary += f" Grounded geometry note: {answer_text}"
        return summary

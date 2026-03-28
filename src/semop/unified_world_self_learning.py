from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

from .operator_algebra import OperatorAlgebraLearner
from .operator_evolution import OperatorSelfEvolutionEngine
from .structures import StructuredMeaningGraph
from .vlso.types import SharedWorldModel


@dataclass
class UnifiedWorldSelfLearningObjective:
    title: str
    north_star: str
    doctrine: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    success_metrics: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedWorldSelfLearningTarget:
    category: str
    label: str
    rationale: str
    priority: float = 0.0
    related_operators: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedWorldSelfLearningPlan:
    primary_goal: str = ''
    hidden_constraints: list[str] = field(default_factory=list)
    operator_algebra_targets: list[UnifiedWorldSelfLearningTarget] = field(default_factory=list)
    operator_functor_targets: list[UnifiedWorldSelfLearningTarget] = field(default_factory=list)
    retained_operator_candidates: list[dict[str, Any]] = field(default_factory=list)
    next_learning_queries: list[str] = field(default_factory=list)
    memory_notes: list[str] = field(default_factory=list)
    summary: str = ''

    def model_dump(self) -> dict[str, Any]:
        return {
            'primary_goal': self.primary_goal,
            'hidden_constraints': list(self.hidden_constraints),
            'operator_algebra_targets': [item.model_dump() for item in self.operator_algebra_targets],
            'operator_functor_targets': [item.model_dump() for item in self.operator_functor_targets],
            'retained_operator_candidates': list(self.retained_operator_candidates),
            'next_learning_queries': list(self.next_learning_queries),
            'memory_notes': list(self.memory_notes),
            'summary': self.summary,
        }


@dataclass
class UnifiedWorldSelfLearningReport:
    report_path: str = ''
    objective: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    operator_evolution: dict[str, Any] = field(default_factory=dict)
    integrated_world: dict[str, Any] = field(default_factory=dict)
    integrated_reasoning: dict[str, Any] = field(default_factory=dict)
    summary_text: str = ''

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class UnifiedWorldSelfLearningEngine:
    def __init__(
        self,
        *,
        operator_algebra_learner: OperatorAlgebraLearner | None = None,
        operator_evolution_engine: OperatorSelfEvolutionEngine | None = None,
    ) -> None:
        self.operator_algebra_learner = operator_algebra_learner or OperatorAlgebraLearner()
        self.operator_evolution_engine = operator_evolution_engine or OperatorSelfEvolutionEngine()

    def default_objective(self, *, domain: str = '', scenario: str = '') -> UnifiedWorldSelfLearningObjective:
        domain_text = str(domain or 'general').replace('_', ' ')
        scenario_text = str(scenario or 'qa').replace('_', ' ')
        return UnifiedWorldSelfLearningObjective(
            title='Unified World Self-Learning Objective',
            north_star=(
                'Build a general intelligence loop that reasons over one shared world model, '
                'secures logic with operator algebra, and improves itself through verified memory updates.'
            ),
            doctrine=[
                'Project every domain into one shared entity-relation-state-event world model.',
                'Recover hidden constraints and goals before acting or answering.',
                'Use operator algebra and functor mappings to explain why a plan or answer is valid.',
                'Promote only verifier-backed corrections into long-term memory and training traces.',
                'Prefer RTX 4060 friendly symbolic-first self-improvement over heavyweight generation-first loops.',
            ],
            constraints=[
                f'Current focus domain: {domain_text}.',
                f'Current focus scenario: {scenario_text}.',
                'Keep the loop auditable through world-model summaries, operator traces, and reviewed artifacts.',
            ],
            success_metrics=[
                'Integrated world summary keeps blockers, prerequisites, evidence, and next steps in one state.',
                'Operator decompositions and functor hypotheses become more reusable across runs.',
                'Self-learning produces better grounded answers and safer action plans without fragmenting by domain.',
            ],
        )

    def build_report(
        self,
        *,
        world: SharedWorldModel,
        reasoning: dict[str, Any],
        graphs: Sequence[StructuredMeaningGraph],
        context: str = '',
        domain: str = '',
        scenario: str = '',
        output_path: str | Path | None = None,
    ) -> UnifiedWorldSelfLearningReport:
        objective = self.default_objective(domain=domain, scenario=scenario)
        algebra_targets, functor_targets, retained_candidates = self._operator_learning_targets(graphs)
        plan = self._build_plan(
            world=world,
            reasoning=reasoning,
            context=context,
            algebra_targets=algebra_targets,
            functor_targets=functor_targets,
            retained_candidates=retained_candidates,
        )
        report = UnifiedWorldSelfLearningReport(
            report_path=str(output_path or ''),
            objective=objective.model_dump(),
            plan=plan.model_dump(),
            operator_evolution={
                'retained_operator_candidates': list(retained_candidates),
                'algebra_target_count': len(algebra_targets),
                'functor_target_count': len(functor_targets),
            },
            integrated_world=world.model_dump(),
            integrated_reasoning=dict(reasoning),
            summary_text=plan.summary,
        )
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
            report.report_path = str(path)
        return report

    def _operator_learning_targets(
        self,
        graphs: Sequence[StructuredMeaningGraph],
    ) -> tuple[list[UnifiedWorldSelfLearningTarget], list[UnifiedWorldSelfLearningTarget], list[dict[str, Any]]]:
        decomposition_targets: list[UnifiedWorldSelfLearningTarget] = []
        functor_targets: list[UnifiedWorldSelfLearningTarget] = []
        for graph in graphs:
            summary = self.operator_algebra_learner.analyze(graph)
            for item in summary.decompositions:
                decomposition_targets.append(
                    UnifiedWorldSelfLearningTarget(
                        category='operator_decomposition',
                        label=item.operator_name,
                        rationale=item.rationale or 'Recovered from operator algebra analysis.',
                        priority=float(item.confidence),
                        related_operators=list(item.basis_operators),
                    )
                )
            for item in summary.functor_hypotheses:
                functor_targets.append(
                    UnifiedWorldSelfLearningTarget(
                        category='functor_hypothesis',
                        label=item.name,
                        rationale=f'{item.source_category} -> {item.target_category}',
                        priority=float(item.confidence),
                        related_operators=sorted(set(item.morphism_map.values())),
                    )
                )
        retained_candidates: list[dict[str, Any]] = []
        if graphs:
            evolution = self.operator_evolution_engine.evolve(
                graphs,
                min_support=1 if len(graphs) < 4 else 2,
                utility_threshold=0.35,
            )
            retained_candidates = [item.model_dump() for item in evolution.proposals if item.retained][:6]
        return (
            self._dedupe_targets(decomposition_targets),
            self._dedupe_targets(functor_targets),
            retained_candidates,
        )

    def _build_plan(
        self,
        *,
        world: SharedWorldModel,
        reasoning: dict[str, Any],
        context: str,
        algebra_targets: list[UnifiedWorldSelfLearningTarget],
        functor_targets: list[UnifiedWorldSelfLearningTarget],
        retained_candidates: list[dict[str, Any]],
    ) -> UnifiedWorldSelfLearningPlan:
        primary_goal = str(reasoning.get('primary_goal') or world.query or '').strip()
        blockers = [str(item) for item in reasoning.get('blockers', []) if str(item).strip()]
        prerequisites = [str(item) for item in reasoning.get('prerequisites', []) if str(item).strip()]
        evidence = [str(item) for item in reasoning.get('evidence', []) if str(item).strip()]
        next_queries = self._next_learning_queries(primary_goal, blockers, prerequisites, evidence, context)
        memory_notes = self._memory_notes(primary_goal, blockers, prerequisites, retained_candidates)
        summary_bits: list[str] = []
        if primary_goal:
            summary_bits.append(f'Primary learning goal: {primary_goal}.')
        if blockers:
            summary_bits.append('Resolve blockers before broadening the domain: ' + ' / '.join(blockers[:3]) + '.')
        if algebra_targets:
            summary_bits.append('Operator algebra should focus on ' + ', '.join(item.label for item in algebra_targets[:3]) + '.')
        if retained_candidates:
            summary_bits.append('Retain and reuse evolved operators with verified utility first.')
        return UnifiedWorldSelfLearningPlan(
            primary_goal=primary_goal,
            hidden_constraints=self._dedupe_strings(blockers + prerequisites),
            operator_algebra_targets=algebra_targets,
            operator_functor_targets=functor_targets,
            retained_operator_candidates=retained_candidates,
            next_learning_queries=next_queries,
            memory_notes=memory_notes,
            summary=' '.join(summary_bits).strip(),
        )

    def _next_learning_queries(
        self,
        primary_goal: str,
        blockers: list[str],
        prerequisites: list[str],
        evidence: list[str],
        context: str,
    ) -> list[str]:
        queries: list[str] = []
        if blockers:
            queries.append(f'Which operator decomposition removes the blocker {blockers[0]} while preserving the main goal?')
        if prerequisites:
            queries.append(f'What evidence verifies prerequisite {prerequisites[0]} before action?')
        if primary_goal:
            queries.append(f'Which alternative action order still achieves {primary_goal} under the current constraints?')
        if evidence:
            queries.append(f'Which grounded relation best explains why {evidence[0]} should change the next decision?')
        if context.strip():
            snippet = context.strip()
            if len(snippet) > 120:
                snippet = snippet[:120].rsplit(' ', 1)[0] + '...'
            queries.append(f'Given this environment context, which reusable operator should be strengthened next? Context: {snippet}')
        return self._dedupe_strings(queries)[:6]

    @staticmethod
    def _memory_notes(
        primary_goal: str,
        blockers: list[str],
        prerequisites: list[str],
        retained_candidates: list[dict[str, Any]],
    ) -> list[str]:
        notes: list[str] = []
        if primary_goal:
            notes.append(f'Keep {primary_goal} as the long-horizon objective in future runs.')
        if blockers:
            notes.append('Cache blockers as first-class world-state facts: ' + ' / '.join(blockers[:3]))
        if prerequisites:
            notes.append('Cache prerequisites as verification gates: ' + ' / '.join(prerequisites[:3]))
        if retained_candidates:
            notes.append('Promote retained operators into reusable operator memory before widening the search space.')
        return UnifiedWorldSelfLearningEngine._dedupe_strings(notes)

    @staticmethod
    def _dedupe_targets(targets: Sequence[UnifiedWorldSelfLearningTarget]) -> list[UnifiedWorldSelfLearningTarget]:
        deduped: list[UnifiedWorldSelfLearningTarget] = []
        seen: set[tuple[str, str]] = set()
        for item in sorted(targets, key=lambda row: (-row.priority, row.category, row.label)):
            key = (item.category, item.label)
            if key in seen:
                continue
            deduped.append(item)
            seen.add(key)
        return deduped[:8]

    @staticmethod
    def _dedupe_strings(values: Sequence[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in values:
            normalized = str(item or '').strip()
            if not normalized or normalized in seen:
                continue
            deduped.append(normalized)
            seen.add(normalized)
        return deduped

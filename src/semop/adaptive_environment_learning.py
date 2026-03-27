from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

from .corpus_store import CorpusMemoryStore
from .environment_brain import EnvironmentBrainRunner, EnvironmentBrainSummary, EnvironmentRoutineStat
from .generalization_proof import GeneralizationProofHarness
from .grounding_self_evolution import GroundingSelfEvolutionRunner, GroundingSelfEvolutionSummary
from .multimodal_scene_understanding import TemporalSceneReasoner, TemporalSituationSummary
from .structures import StructuredMeaningGraph
from .unified_benchmark import GroundedExplanationEvalCase, UnifiedSemOpTrainer


@dataclass
class AdaptiveEnvironmentAxis:
    key: str
    label: str
    score: float
    target: float
    ready: bool
    summary: str
    next_step: str = ""

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveActionStep:
    order: int
    action: str
    rationale: str
    guard: str = ""

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveActionRehearsal:
    label: str
    trigger: str
    safety_goal: str
    confidence: float
    prerequisites: list[str] = field(default_factory=list)
    operator_families: list[str] = field(default_factory=list)
    failure_if_skipped: list[str] = field(default_factory=list)
    steps: list[AdaptiveActionStep] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "trigger": self.trigger,
            "safety_goal": self.safety_goal,
            "confidence": self.confidence,
            "prerequisites": list(self.prerequisites),
            "operator_families": list(self.operator_families),
            "failure_if_skipped": list(self.failure_if_skipped),
            "steps": [item.model_dump() for item in self.steps],
        }


@dataclass
class AdaptiveEnvironmentLearningSummary:
    environment_name: str
    environment_source: str
    domain: str
    scenario: str
    output_dir: str
    report_path: str
    visual_input: str = ""
    seeded_query_count: int = 0
    local_graph_count: int = 0
    approved_review_count: int = 0
    pending_review_count: int = 0
    grounding_cases_used: int = 0
    refined_graph_copies: int = 0
    improved_cases: int = 0
    capability_scores: dict[str, float] = field(default_factory=dict)
    ready_axes: int = 0
    total_axes: int = 0
    completed_skills: list[str] = field(default_factory=list)
    remaining_gaps: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    axes: list[AdaptiveEnvironmentAxis] = field(default_factory=list)
    action_rehearsals: list[AdaptiveActionRehearsal] = field(default_factory=list)
    environment_brain: dict[str, Any] = field(default_factory=dict)
    self_evolution: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    temporal_scene: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {
            "environment_name": self.environment_name,
            "environment_source": self.environment_source,
            "domain": self.domain,
            "scenario": self.scenario,
            "output_dir": self.output_dir,
            "report_path": self.report_path,
            "visual_input": self.visual_input,
            "seeded_query_count": self.seeded_query_count,
            "local_graph_count": self.local_graph_count,
            "approved_review_count": self.approved_review_count,
            "pending_review_count": self.pending_review_count,
            "grounding_cases_used": self.grounding_cases_used,
            "refined_graph_copies": self.refined_graph_copies,
            "improved_cases": self.improved_cases,
            "capability_scores": dict(self.capability_scores),
            "ready_axes": self.ready_axes,
            "total_axes": self.total_axes,
            "completed_skills": list(self.completed_skills),
            "remaining_gaps": list(self.remaining_gaps),
            "next_actions": list(self.next_actions),
            "notes": list(self.notes),
            "axes": [item.model_dump() for item in self.axes],
            "action_rehearsals": [item.model_dump() for item in self.action_rehearsals],
            "environment_brain": dict(self.environment_brain),
            "self_evolution": dict(self.self_evolution),
            "training": dict(self.training),
            "temporal_scene": dict(self.temporal_scene),
        }


class AdaptiveEnvironmentLearningRunner:
    def __init__(self, mode: str = "heuristic") -> None:
        self.mode = mode

    def run(
        self,
        *,
        environment_name: str,
        domain: str,
        scenario: str,
        context: str,
        store_path: str = "data/semop_memory.db",
        review_queue_path: str | None = "data/ops_review_queue.db",
        output_dir: str = "data/unified_semop_gui_run/adaptive_environment_learning",
        visual_input: str | None = None,
        concept_store_path: str | None = None,
        operator_store_path: str | None = None,
        affordance_weights_path: str | None = None,
        seed_queries: Sequence[str] | None = None,
        self_evolution_rounds: int = 2,
        cases_per_round: int = 8,
    ) -> AdaptiveEnvironmentLearningSummary:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)

        brain = EnvironmentBrainRunner(mode=self.mode).run(
            environment_name=environment_name,
            domain=domain,
            scenario=scenario,
            context=context,
            store_path=store_path,
            review_queue_path=review_queue_path,
            output_dir=str(root / "environment_brain"),
            visual_input=visual_input,
            concept_store_path=concept_store_path,
            operator_store_path=operator_store_path,
            affordance_weights_path=affordance_weights_path,
            seed_queries=seed_queries,
        )
        store = CorpusMemoryStore(store_path)
        local_graphs = self._environment_graphs(store, brain.environment_source)
        grounding_cases = self._derive_local_grounding_cases(local_graphs)

        evolution_prefix = f"{brain.environment_source}_self_evolution"
        self_evolution = GroundingSelfEvolutionRunner(mode=self.mode).run(
            store_path,
            review_queue_path or "data/ops_review_queue.db",
            root / "self_evolution",
            split="train",
            rounds=max(1, int(self_evolution_rounds)),
            cases_per_round=min(max(1, int(cases_per_round)), max(4, len(grounding_cases) or 4, int(cases_per_round))),
            source_prefix=evolution_prefix,
            cases=grounding_cases,
        )
        refined_graph_copies = self._merge_refined_graphs(
            store,
            destination_source=brain.environment_source,
            evolution_source_prefix=evolution_prefix,
            rounds=max(1, len(self_evolution.rounds)),
            split="train",
        )
        merged_graphs = self._environment_graphs(store, brain.environment_source)
        training = UnifiedSemOpTrainer().train_from_store(
            store_path,
            root / "adaptive_bundle",
            source=brain.environment_source,
            split="train",
            review_store_path=None,
            approved_queries_only=False,
            operating_domain=domain,
        )
        temporal_scene = self._summarize_temporal_scene(
            visual_input=visual_input,
            context=context,
            concept_store_path=concept_store_path,
            operator_store_path=operator_store_path,
            affordance_weights_path=affordance_weights_path,
        )
        action_rehearsals = self._action_rehearsals(brain, temporal_scene)
        capability_scores = self._capability_scores(brain, self_evolution, merged_graphs, temporal_scene, action_rehearsals)
        axes = self._axes(capability_scores, visual_input=visual_input, temporal_scene=temporal_scene)
        completed = [item.label for item in axes if item.ready]
        gaps = [item.label for item in axes if not item.ready]
        next_actions = self._next_actions(axes, brain, self_evolution, action_rehearsals, visual_input=visual_input)
        report_path = root / "adaptive_environment_learning_report.json"
        summary = AdaptiveEnvironmentLearningSummary(
            environment_name=brain.environment_name,
            environment_source=brain.environment_source,
            domain=brain.domain,
            scenario=brain.scenario,
            output_dir=str(root),
            report_path=str(report_path),
            visual_input=str(visual_input or ""),
            seeded_query_count=brain.seeded_query_count,
            local_graph_count=len(merged_graphs),
            approved_review_count=brain.auto_approved_review_count + int(self_evolution.approved_reviews),
            pending_review_count=brain.pending_review_count,
            grounding_cases_used=len(grounding_cases),
            refined_graph_copies=refined_graph_copies,
            improved_cases=int(self_evolution.improved_cases),
            capability_scores=capability_scores,
            ready_axes=sum(1 for item in axes if item.ready),
            total_axes=len(axes),
            completed_skills=completed,
            remaining_gaps=gaps,
            next_actions=next_actions,
            notes=[
                f"Environment source: {brain.environment_source}",
                f"Local grounding cases: {len(grounding_cases)}",
                f"Refined graph copies merged back into the same environment source: {refined_graph_copies}",
                f"Action rehearsals synthesized: {len(action_rehearsals)}",
            ],
            axes=axes,
            action_rehearsals=action_rehearsals,
            environment_brain=brain.model_dump(),
            self_evolution=self_evolution.model_dump(),
            training=training.model_dump(),
            temporal_scene=temporal_scene.model_dump() if temporal_scene is not None else {},
        )
        report_path.write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    @staticmethod
    def _environment_graphs(store: CorpusMemoryStore, source: str) -> list[StructuredMeaningGraph]:
        return list(store.fetch_graphs(split="train", source=source))

    @staticmethod
    def _derive_local_grounding_cases(graphs: list[StructuredMeaningGraph]) -> list[GroundedExplanationEvalCase]:
        return list(GeneralizationProofHarness._derive_grounding_cases(graphs, limit=max(12, len(graphs) * 3)))

    @staticmethod
    def _merge_refined_graphs(
        store: CorpusMemoryStore,
        *,
        destination_source: str,
        evolution_source_prefix: str,
        rounds: int,
        split: str,
    ) -> int:
        copied = 0
        for round_index in range(1, max(1, rounds) + 1):
            source_name = f"{evolution_source_prefix}_round_{round_index:02d}"
            for graph in store.fetch_graphs(split=split, source=source_name):
                store.upsert_graph(graph, source=destination_source, split=split)
                store.upsert_premise_operator_memory(graph, source=destination_source, split=split)
                copied += 1
        return copied

    def _summarize_temporal_scene(
        self,
        *,
        visual_input: str | None,
        context: str,
        concept_store_path: str | None,
        operator_store_path: str | None,
        affordance_weights_path: str | None,
    ) -> TemporalSituationSummary | None:
        candidate = str(visual_input or "").strip()
        if not candidate:
            return None
        path = Path(candidate)
        if not path.exists():
            return None
        try:
            query = "What changes, hazards, and repeated routines define this environment over time?"
            if context.strip():
                query = f"{query} Context: {context.strip()}"
            return TemporalSceneReasoner(
                mode="deep",
                answer_mode="structured",
                concept_store_path=concept_store_path,
                operator_store_path=operator_store_path,
                affordance_weights_path=affordance_weights_path,
            ).summarize(query, candidate)
        except Exception:
            return None

    @staticmethod
    def _action_rehearsals(
        brain: EnvironmentBrainSummary,
        temporal_scene: TemporalSituationSummary | None,
    ) -> list[AdaptiveActionRehearsal]:
        rehearsals: list[AdaptiveActionRehearsal] = []
        hazard_text = ', '.join(brain.hazard_patterns[:2]) if brain.hazard_patterns else 'avoid local failure modes'
        temporal_trigger = temporal_scene.situation_summary if temporal_scene is not None else ''
        for index, routine in enumerate(brain.routine_patterns[:4], start=1):
            trigger = temporal_trigger or f"When {brain.scenario.replace('_', ' ')} conditions appear in {brain.environment_name.replace('_', ' ')}"
            prerequisites = list(routine.required_premises[:4]) or list(brain.stable_constraints[:2])
            operator_families = list(routine.operator_families[:4])
            guard_text = prerequisites[0] if prerequisites else 'verify the local preconditions first'
            steps = [
                AdaptiveActionStep(
                    order=1,
                    action=guard_text,
                    rationale='Check the local prerequisite before any irreversible action.',
                    guard='precondition gate',
                ),
                AdaptiveActionStep(
                    order=2,
                    action=routine.label,
                    rationale='Execute the repeated local routine only after the guard passes.',
                    guard='routine execution',
                ),
                AdaptiveActionStep(
                    order=3,
                    action='Re-check the visible state and confirm the hazard is still controlled.',
                    rationale='Keep the plan grounded in the current environment rather than assuming it stayed safe.',
                    guard='post-action verification',
                ),
            ]
            confidence = round(min(1.0, (float(routine.support_count) / 4.0) * 0.6 + (len(prerequisites) / 4.0) * 0.25 + (len(operator_families) / 3.0) * 0.15), 4)
            rehearsals.append(
                AdaptiveActionRehearsal(
                    label=f"Routine rehearsal {index}: {routine.label}",
                    trigger=trigger,
                    safety_goal=f"Preserve safe execution while handling {hazard_text}.",
                    confidence=confidence,
                    prerequisites=prerequisites,
                    operator_families=operator_families,
                    failure_if_skipped=list(brain.hazard_patterns[:3]) or ['Ungrounded action can violate a hidden local constraint.'],
                    steps=steps,
                )
            )
        return rehearsals

    @staticmethod
    def _capability_scores(
        brain: EnvironmentBrainSummary,
        self_evolution: GroundingSelfEvolutionSummary,
        merged_graphs: list[StructuredMeaningGraph],
        temporal_scene: TemporalSituationSummary | None,
        action_rehearsals: Sequence[AdaptiveActionRehearsal],
    ) -> dict[str, float]:
        mastery = dict(brain.mastery_scores)
        contextual = float(mastery.get("environment_mastery", 0.0))
        safety = float(mastery.get("safety_alignment", 0.0))
        grounding = max(float(mastery.get("grounding_strength", 0.0)), float(self_evolution.final_grounding_score))
        multimodal = 0.0
        if temporal_scene is not None:
            stable_bonus = min(1.0, len(temporal_scene.stable_entities) / 4.0)
            change_bonus = min(1.0, len(temporal_scene.temporal_events) / 3.0)
            multimodal = round((stable_bonus * 0.55) + (change_bonus * 0.45), 4)
        elif brain.visual_entities:
            multimodal = min(1.0, 0.35 + (len(brain.visual_entities) / 10.0))
        reflection = round(
            min(
                1.0,
                (float(self_evolution.final_grounding_score) * 0.5)
                + (min(1.0, self_evolution.improved_cases / 6.0) * 0.3)
                + (min(1.0, self_evolution.approved_reviews / 6.0) * 0.2),
            ),
            4,
        )
        memory = round(
            min(
                1.0,
                (min(1.0, len(merged_graphs) / 12.0) * 0.55)
                + (min(1.0, len(brain.routine_patterns) / 5.0) * 0.25)
                + (min(1.0, len(brain.stable_concepts) / 6.0) * 0.2),
            ),
            4,
        )
        embodied = round(
            min(
                1.0,
                (safety * 0.35)
                + (min(1.0, len(action_rehearsals) / 3.0) * 0.3)
                + (min(1.0, sum(item.confidence for item in action_rehearsals) / max(1.0, len(action_rehearsals))) * 0.2)
                + (multimodal * 0.15),
            ),
            4,
        )
        local_intelligence = round(
            min(
                1.0,
                (contextual * 0.24)
                + (grounding * 0.22)
                + (safety * 0.13)
                + (reflection * 0.13)
                + (memory * 0.1)
                + (multimodal * 0.08)
                + (embodied * 0.1),
            ),
            4,
        )
        return {
            "contextual_reasoning": round(contextual, 4),
            "grounded_reasoning": round(grounding, 4),
            "safe_execution": round(safety, 4),
            "multimodal_understanding": round(multimodal, 4),
            "self_reflection": reflection,
            "environment_memory": memory,
            "embodied_planning": embodied,
            "local_intelligence": local_intelligence,
        }

    @staticmethod
    def _axes(
        scores: dict[str, float],
        *,
        visual_input: str | None,
        temporal_scene: TemporalSituationSummary | None,
    ) -> list[AdaptiveEnvironmentAxis]:
        multimodal_next = "Attach one image, frame folder, or short clip from this environment and rerun the self-improvement loop."
        if visual_input and temporal_scene is None:
            multimodal_next = "Use a frame folder or JSON manifest if direct video extraction is unavailable, then rerun the loop."
        return [
            AdaptiveEnvironmentAxis(
                key="contextual_reasoning",
                label="Hidden-context reasoning in this environment",
                score=float(scores.get("contextual_reasoning", 0.0)),
                target=0.74,
                ready=float(scores.get("contextual_reasoning", 0.0)) >= 0.74,
                summary="Repeated local prompts should converge on the same hidden constraints and repeated routines.",
                next_step="Add 3-5 harder local queries if this stays weak.",
            ),
            AdaptiveEnvironmentAxis(
                key="grounded_reasoning",
                label="Grounded reasoning and evidence discipline",
                score=float(scores.get("grounded_reasoning", 0.0)),
                target=0.7,
                ready=float(scores.get("grounded_reasoning", 0.0)) >= 0.7,
                summary="The system should stop echoing the question and ground its answer in local evidence.",
                next_step="Approve one or two grounded answers manually if automatic grounding remains weak.",
            ),
            AdaptiveEnvironmentAxis(
                key="multimodal_understanding",
                label="Multimodal understanding of the same environment",
                score=float(scores.get("multimodal_understanding", 0.0)),
                target=0.52,
                ready=float(scores.get("multimodal_understanding", 0.0)) >= 0.52,
                summary="The system should connect the local SOP and the visible environment instead of treating them as separate lanes.",
                next_step=multimodal_next,
            ),
            AdaptiveEnvironmentAxis(
                key="self_reflection",
                label="Self-reflection and correction",
                score=float(scores.get("self_reflection", 0.0)),
                target=0.6,
                ready=float(scores.get("self_reflection", 0.0)) >= 0.6,
                summary="The system should detect weak grounding, create corrected traces, and keep the better variants.",
                next_step="Run the loop again after approving a few corrected traces if this remains weak.",
            ),
            AdaptiveEnvironmentAxis(
                key="embodied_planning",
                label="Embodied action rehearsal in this environment",
                score=float(scores.get("embodied_planning", 0.0)),
                target=0.64,
                ready=float(scores.get("embodied_planning", 0.0)) >= 0.64,
                summary="The system should rehearse safe local action order instead of only describing the environment.",
                next_step="Use one image or clip from the same setting so the action rehearsal can anchor itself in visible state changes.",
            ),
            AdaptiveEnvironmentAxis(
                key="local_intelligence",
                label="Local autonomous intelligence",
                score=float(scores.get("local_intelligence", 0.0)),
                target=0.7,
                ready=float(scores.get("local_intelligence", 0.0)) >= 0.7,
                summary="This is the combined score for context, grounding, safety, memory, self-correction, and action rehearsal in one setting.",
                next_step="Close the weakest axis first, then rerun the same environment instead of widening to a new one.",
            ),
        ]

    @staticmethod
    def _next_actions(
        axes: list[AdaptiveEnvironmentAxis],
        brain: EnvironmentBrainSummary,
        self_evolution: GroundingSelfEvolutionSummary,
        action_rehearsals: Sequence[AdaptiveActionRehearsal],
        *,
        visual_input: str | None,
    ) -> list[str]:
        actions = [item.next_step for item in axes if not item.ready and item.next_step]
        if not actions and not brain.next_probes:
            actions.append("Ask a harder local question that mixes hidden constraints with action safety.")
        actions.extend(item.query for item in brain.next_probes[:3] if item.query)
        if not visual_input:
            actions.append("Attach one representative image or short clip from the same environment to add multimodal memory.")
        if self_evolution.approved_reviews <= 0:
            actions.append("Use Manual fast path once to approve a corrected local answer and rerun this loop.")
        if action_rehearsals:
            actions.append(f"Rehearse this action order locally: {action_rehearsals[0].label}.")
        deduped: list[str] = []
        for action in actions:
            text = str(action).strip()
            if text and text not in deduped:
                deduped.append(text)
        return deduped[:8]

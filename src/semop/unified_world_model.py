
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from .contest_programmer import ContestProblemStructure, ContestSolution
from .structures import StructuredMeaningGraph
from .vlso.types import SharedWorldModel, VLSOEntity, VLSOEvent, VLSOOperator, VLSORelation
from .world_model_math import WorldModelMathReport


@dataclass
class UnifiedWorldReasoning:
    query: str
    primary_goal: str = ""
    active_domains: list[str] = field(default_factory=list)
    active_modalities: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    operator_trace: list[str] = field(default_factory=list)
    summary: str = ""

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def to_text(self) -> str:
        lines: list[str] = []
        if self.active_domains:
            lines.append("공통 월드 모델이 묶은 도메인: " + ", ".join(self.active_domains[:4]))
        if self.primary_goal:
            lines.append(f"핵심 목표: {self.primary_goal}")
        if self.blockers:
            lines.append("막는 조건: " + " / ".join(self.blockers[:3]))
        if self.prerequisites:
            lines.append("선행 조건: " + " / ".join(self.prerequisites[:3]))
        if self.alternatives:
            lines.append("대안 경로: " + " / ".join(self.alternatives[:3]))
        if self.next_steps:
            lines.append("다음 추론 단계: " + " / ".join(self.next_steps[:3]))
        if self.evidence:
            lines.append("근거: " + " / ".join(self.evidence[:3]))
        if self.warnings:
            lines.append("주의: " + " / ".join(self.warnings[:2]))
        return "\n".join(lines).strip()


class UnifiedWorldModelEngine:
    _BLOCKER_RELATIONS = {"BLOCKED_BY", "CONFLICTS_WITH", "VIOLATES", "FAILS_WITH", "MISSING_FOR"}
    _PREREQUISITE_RELATIONS = {"REQUIRES", "DEPENDS_ON", "NEEDS", "PRECONDITION_FOR"}
    _ALTERNATIVE_RELATIONS = {"ALTERNATIVE", "FALLBACK", "BACKUP", "WORKAROUND"}
    _DOMAIN_RELATIONS = {"IN_DOMAIN", "MATCHES_FRAME", "TARGETS_GOAL", "USES_ALGORITHM"}

    def from_graph(self, graph: StructuredMeaningGraph | None, *, source: str = "ops") -> SharedWorldModel:
        if graph is None:
            return SharedWorldModel(query="")
        world = SharedWorldModel(query=graph.query)
        for node in graph.nodes:
            attrs = dict(node.attributes)
            if node.provenance:
                attrs["provenance"] = list(node.provenance)
            world.add_entity(VLSOEntity(node.id, node.label or node.id, source, node.kind or "unknown", attrs))
        for edge in graph.edges:
            attrs = dict(edge.attributes)
            if edge.provenance:
                attrs["provenance"] = list(edge.provenance)
            world.add_relation(VLSORelation(edge.source, edge.relation, edge.target, source, float(edge.confidence), attrs))
        for name in graph.semantic_operators:
            if name:
                world.add_operator(VLSOOperator(str(name), "semantic", "Operator recovered from the structured meaning graph.", source, 0.62))
        for item in graph.induced_operators:
            world.add_operator(VLSOOperator(item.name, item.family or "induced", item.description or "Induced operator candidate.", source, float(item.confidence)))
        for index, step in enumerate(graph.plan, start=1):
            world.inferred_steps = self._merge_unique(world.inferred_steps, [step.action])
            world.add_event(VLSOEvent(step.id or f"{source}_plan_{index}", step.action, "plan_step", source, None, 1.0, {"requires": list(step.requires)}, {"rationale": step.rationale, "status": step.status}))
        for index, result in enumerate(graph.symbolic_results, start=1):
            world.add_event(VLSOEvent(f"{source}_symbolic_{index}", result.domain or "symbolic_result", "symbolic_result", source, None, float(result.confidence), {"source": result.source}, {"answer": result.answer, "evidence": list(result.evidence), "equations": list(result.equations)}))
        frame = graph.context_frame
        goals = list(graph.hidden_goals)
        if frame is not None and frame.primary_goal:
            goals.insert(0, frame.primary_goal)
        world.goals = self._merge_unique(world.goals, goals)
        constraints: list[str] = list(graph.required_premises) + list(graph.missing_premises)
        if frame is not None:
            constraints.extend(frame.active_constraints)
            constraints.extend(frame.missing_requirements)
            constraints.extend(frame.risk_signals)
        world.constraints = self._merge_unique(world.constraints, constraints)
        world.warnings = self._merge_unique(world.warnings, list(graph.warnings) + list(graph.invalid_advice))
        world.audit_trace = self._merge_unique(world.audit_trace, list(graph.audit_trace))
        world.metadata = self._merge_metadata(world.metadata, {
            "source_routes": [source],
            "graph_intent": graph.intent,
            "graph_domain": graph.domain,
            "graph_scenario": graph.scenario,
            "candidate_actions": list(graph.candidate_actions),
            "creative_alternatives": list(graph.creative_alternatives),
            "context_frame": self._to_plain(frame),
            "operator_execution": self._to_plain(graph.operator_execution),
        })
        return world

    def from_shared_world(self, world_or_payload: SharedWorldModel | dict[str, Any] | None, *, source: str = "vision", query: str = "") -> SharedWorldModel:
        if world_or_payload is None:
            return SharedWorldModel(query=query)
        payload = world_or_payload.model_dump() if isinstance(world_or_payload, SharedWorldModel) else dict(world_or_payload)
        world = SharedWorldModel(query=str(payload.get("query") or query or ""))
        for item in payload.get("entities", []) or []:
            if not isinstance(item, dict):
                continue
            world.add_entity(VLSOEntity(str(item.get("id") or self._slug(item.get("label"), "entity")), str(item.get("label") or item.get("id") or "entity"), str(item.get("modality") or source), str(item.get("entity_type") or item.get("kind") or "unknown"), dict(item.get("attributes") or {})))
        for item in payload.get("relations", []) or []:
            if not isinstance(item, dict):
                continue
            world.add_relation(VLSORelation(str(item.get("source") or ""), str(item.get("relation") or ""), str(item.get("target") or ""), str(item.get("modality") or source), float(item.get("confidence", 1.0) or 1.0), dict(item.get("attributes") or {})))
        for item in payload.get("operators", []) or []:
            if not isinstance(item, dict):
                continue
            world.add_operator(VLSOOperator(str(item.get("name") or ""), str(item.get("axis") or "shared"), str(item.get("description") or ""), str(item.get("source_modality") or source), float(item.get("confidence", 0.6) or 0.6)))
        for item in payload.get("events", []) or []:
            if not isinstance(item, dict):
                continue
            world.add_event(VLSOEvent(str(item.get("id") or self._slug(item.get("label"), "event")), str(item.get("label") or item.get("id") or "event"), str(item.get("event_type") or "event"), str(item.get("modality") or source), item.get("frame_index"), float(item.get("confidence", 1.0) or 1.0), dict(item.get("participants") or {}), dict(item.get("attributes") or {})))
        world.goals = self._merge_unique(world.goals, [str(item) for item in payload.get("goals", []) or [] if str(item).strip()])
        world.constraints = self._merge_unique(world.constraints, [str(item) for item in payload.get("constraints", []) or [] if str(item).strip()])
        world.inferred_steps = self._merge_unique(world.inferred_steps, [str(item) for item in payload.get("inferred_steps", []) or [] if str(item).strip()])
        world.warnings = self._merge_unique(world.warnings, [str(item) for item in payload.get("warnings", []) or [] if str(item).strip()])
        world.audit_trace = self._merge_unique(world.audit_trace, [str(item) for item in payload.get("audit_trace", []) or [] if str(item).strip()])
        world.metadata = self._merge_metadata(world.metadata, payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {})
        world.metadata = self._merge_metadata(world.metadata, {"source_routes": [source]})
        return world

    def from_environment_brain_summary(self, summary: Any) -> SharedWorldModel:
        if summary is None:
            return SharedWorldModel(query="")
        payload = summary.model_dump() if hasattr(summary, "model_dump") else dict(summary)
        query = str(payload.get("source_context") or payload.get("environment_name") or "")
        world = SharedWorldModel(query=query)
        environment_name = str(payload.get("environment_name") or "environment")
        environment_id = self._slug(environment_name, "environment")
        domain = str(payload.get("domain") or "general_environment")
        scenario = str(payload.get("scenario") or "routine")
        world.add_entity(
            VLSOEntity(
                environment_id,
                environment_name,
                "ops",
                "environment",
                {
                    "domain": domain,
                    "scenario": scenario,
                    "environment_source": str(payload.get("environment_source") or ""),
                },
            )
        )
        for index, item in enumerate(payload.get("stable_concepts", []) or [], start=1):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or f"concept_{index}")
            concept_id = self._slug(label, f"concept_{index}")
            support = int(item.get("support_count", 0) or 0)
            world.add_entity(VLSOEntity(concept_id, label, "ops", str(item.get("kind") or "concept"), dict(item)))
            world.add_relation(VLSORelation(environment_id, "HAS_CONCEPT", concept_id, "ops", min(1.0, max(0.35, support / 5.0)), {"support_count": support}))
        for index, constraint in enumerate(payload.get("stable_constraints", []) or [], start=1):
            label = str(constraint or "").strip()
            if not label:
                continue
            constraint_id = self._slug(label, f"constraint_{index}")
            world.add_entity(VLSOEntity(constraint_id, label, "ops", "constraint", {}))
            world.add_relation(VLSORelation(environment_id, "REQUIRES", constraint_id, "ops", 1.0, {}))
        for index, item in enumerate(payload.get("routine_patterns", []) or [], start=1):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or f"routine_{index}")
            routine_id = self._slug(label, f"routine_{index}")
            support = int(item.get("support_count", 0) or 0)
            world.add_entity(VLSOEntity(routine_id, label, "ops", "routine", dict(item)))
            world.add_relation(VLSORelation(environment_id, "AFFORDS", routine_id, "ops", min(1.0, max(0.4, support / 5.0)), {"support_count": support}))
            for req_index, premise in enumerate(item.get("required_premises", []) or [], start=1):
                premise_text = str(premise or "").strip()
                if not premise_text:
                    continue
                premise_id = self._slug(premise_text, f"routine_{index}_requirement_{req_index}")
                world.add_entity(VLSOEntity(premise_id, premise_text, "ops", "requirement", {}))
                world.add_relation(VLSORelation(routine_id, "REQUIRES", premise_id, "ops", 1.0, {}))
            for family in item.get("operator_families", []) or []:
                family_text = str(family or "").strip()
                if not family_text:
                    continue
                world.add_operator(VLSOOperator(family_text, "environment_routine", "Recovered from environment routine statistics.", "ops", 0.72))
        for index, hazard in enumerate(payload.get("hazard_patterns", []) or [], start=1):
            label = str(hazard or "").strip()
            if not label:
                continue
            hazard_id = self._slug(label, f"hazard_{index}")
            world.add_entity(VLSOEntity(hazard_id, label, "ops", "hazard", {}))
            world.add_relation(VLSORelation(environment_id, "BLOCKED_BY", hazard_id, "ops", 0.92, {}))
        for index, label in enumerate(payload.get("visual_entities", []) or [], start=1):
            entity_label = str(label or "").strip()
            if not entity_label:
                continue
            entity_id = self._slug(entity_label, f"visual_entity_{index}")
            world.add_entity(VLSOEntity(entity_id, entity_label, "vision", "visual_anchor", {}))
            world.add_relation(VLSORelation(entity_id, "OBSERVED_IN", environment_id, "vision", 0.7, {}))
        for index, item in enumerate(payload.get("next_probes", []) or [], start=1):
            if not isinstance(item, dict):
                continue
            probe_query = str(item.get("query") or "").strip()
            if not probe_query:
                continue
            world.add_event(
                VLSOEvent(
                    f"environment_probe_{index}",
                    probe_query,
                    "next_probe",
                    str(item.get("modality") or "text"),
                    None,
                    0.72,
                    {"purpose": str(item.get("purpose") or "")},
                    dict(item),
                )
            )
            world.inferred_steps = self._merge_unique(world.inferred_steps, [probe_query])
        world.goals = self._merge_unique(world.goals, [f"master {domain.replace('_', ' ')} {scenario.replace('_', ' ')}"])
        world.constraints = self._merge_unique(world.constraints, [str(item) for item in payload.get("stable_constraints", []) or [] if str(item).strip()])
        world.warnings = self._merge_unique(world.warnings, [str(item) for item in payload.get("hazard_patterns", []) or [] if str(item).strip()])
        world.audit_trace = self._merge_unique(world.audit_trace, [str(item) for item in payload.get("notes", []) or [] if str(item).strip()])
        world.metadata = self._merge_metadata(world.metadata, {
            "source_routes": ["environment_brain"],
            "environment_brain": payload,
        })
        return world

    def from_grounding_self_evolution_summary(self, summary: Any, *, query: str = "") -> SharedWorldModel:
        if summary is None:
            return SharedWorldModel(query=query)
        payload = summary.model_dump() if hasattr(summary, "model_dump") else dict(summary)
        reflections = payload.get("reflections", []) or []
        resolved_query = str(query or (reflections[0].get("query") if reflections and isinstance(reflections[0], dict) else "") or "")
        world = SharedWorldModel(query=resolved_query)
        process_id = "grounding_self_evolution"
        world.add_entity(
            VLSOEntity(
                process_id,
                "grounding self evolution",
                "ops",
                "self_improvement_process",
                {
                    "improved_cases": int(payload.get("improved_cases", 0) or 0),
                    "approved_reviews": int(payload.get("approved_reviews", 0) or 0),
                    "stored_graphs": int(payload.get("stored_graphs", 0) or 0),
                    "final_grounding_score": float(payload.get("final_grounding_score", 0.0) or 0.0),
                },
            )
        )
        for index, item in enumerate(reflections, start=1):
            if not isinstance(item, dict):
                continue
            strategy = str(item.get("strategy") or f"strategy_{index}").strip()
            weakness = str(item.get("weakness") or "").strip()
            reflection_id = f"grounding_reflection_{index}"
            strategy_id = self._slug(strategy, f"strategy_{index}")
            world.add_entity(VLSOEntity(reflection_id, f"reflection {index}", "ops", "reflection", dict(item)))
            world.add_entity(VLSOEntity(strategy_id, strategy.replace("_", " "), "ops", "repair_strategy", {"strategy": strategy}))
            world.add_relation(VLSORelation(process_id, "WORKAROUND", strategy_id, "ops", 0.88, {}))
            if weakness:
                weakness_id = self._slug(weakness, f"weakness_{index}")
                world.add_entity(VLSOEntity(weakness_id, weakness, "ops", "weakness", {}))
                world.add_relation(VLSORelation(process_id, "BLOCKED_BY", weakness_id, "ops", 0.85, {}))
                world.add_relation(VLSORelation(strategy_id, "IMPROVES", weakness_id, "ops", 0.86, {}))
            for focus_index, focus in enumerate(item.get("evidence_focus", []) or [], start=1):
                focus_text = str(focus or "").strip()
                if not focus_text:
                    continue
                focus_id = self._slug(focus_text, f"reflection_{index}_focus_{focus_index}")
                world.add_entity(VLSOEntity(focus_id, focus_text, "ops", "evidence_focus", {}))
                world.add_relation(VLSORelation(process_id, "REQUIRES", focus_id, "ops", 0.84, {}))
            score_before = float(item.get("score_before", 0.0) or 0.0)
            score_after = float(item.get("score_after", 0.0) or 0.0)
            world.add_event(VLSOEvent(reflection_id, strategy.replace("_", " ") or f"reflection {index}", "self_reflection", "ops", None, max(0.4, score_after), {"strategy": strategy_id}, {"score_before": score_before, "score_after": score_after, "gain": round(score_after - score_before, 4)}))
            if score_after > score_before:
                world.inferred_steps = self._merge_unique(world.inferred_steps, [f"apply {strategy.replace('_', ' ')}"])
        for index, item in enumerate(payload.get("rounds", []) or [], start=1):
            if not isinstance(item, dict):
                continue
            world.add_event(VLSOEvent(f"self_evolution_round_{index}", f"round {item.get('round_index', index)}", "evolution_round", "ops", None, 0.76, {}, dict(item)))
        final_score = float(payload.get("final_grounding_score", 0.0) or 0.0)
        world.goals = self._merge_unique(world.goals, ["improve grounded reasoning", "promote corrected traces into long-term memory"])
        if final_score < 0.7:
            world.warnings = self._merge_unique(world.warnings, ["Grounding self-evolution still needs more verified corrections."])
        world.metadata = self._merge_metadata(world.metadata, {
            "source_routes": ["self_evolution"],
            "self_evolution": payload,
        })
        return world

    def from_video_payload(self, query: str, payload: dict[str, Any]) -> SharedWorldModel:
        world = SharedWorldModel(query=query)
        for index, item in enumerate(payload.get("stable_entities") or [], start=1):
            label = str(item.get("label") if isinstance(item, dict) else item or "").strip() or f"stable_entity_{index}"
            entity_id = str(item.get("id") if isinstance(item, dict) else "") or self._slug(label, f"stable_entity_{index}")
            world.add_entity(VLSOEntity(entity_id, label, "video", "temporal_anchor", dict(item) if isinstance(item, dict) else {}))
        for index, item in enumerate(payload.get("temporal_events") or [], start=1):
            label = str(item.get("label") if isinstance(item, dict) else item or "").strip() or f"temporal_event_{index}"
            world.add_event(VLSOEvent(str(item.get("id") if isinstance(item, dict) else "") or self._slug(label, f"temporal_event_{index}"), label, str(item.get("event_type") if isinstance(item, dict) else "temporal_event") or "temporal_event", "video", item.get("frame_index") if isinstance(item, dict) else None, float(item.get("confidence", 0.7) if isinstance(item, dict) else 0.7), dict(item.get("participants") or {}) if isinstance(item, dict) else {}, dict(item) if isinstance(item, dict) else {}))
        world.constraints = self._merge_unique(world.constraints, [str(payload.get("fallback_hint") or "").strip()] if payload.get("fallback_hint") else [])
        world.warnings = self._merge_unique(world.warnings, [str(item) for item in payload.get("warnings", []) or [] if str(item).strip()])
        world.metadata = self._merge_metadata(world.metadata, dict(payload))
        world.metadata = self._merge_metadata(world.metadata, {"source_routes": ["video"]})
        return world

    def from_reconstruction_payload(self, query: str, payload: dict[str, Any]) -> SharedWorldModel:
        world = SharedWorldModel(query=query)
        for index, item in enumerate(payload.get("primitives") or [], start=1):
            if not isinstance(item, dict):
                continue
            entity_id = str(item.get("id") or f"primitive_{index}")
            world.add_entity(VLSOEntity(entity_id, str(item.get("label") or entity_id), "visual_3d", str(item.get("kind") or "primitive"), dict(item)))
        for item in payload.get("relations") or []:
            if not isinstance(item, dict):
                continue
            world.add_relation(VLSORelation(str(item.get("source") or ""), str(item.get("relation") or ""), str(item.get("target") or ""), "visual_3d", float(item.get("confidence", 1.0) or 1.0), dict(item.get("attributes") or {})))
        world.warnings = self._merge_unique(world.warnings, [str(item) for item in payload.get("warnings", []) or [] if str(item).strip()])
        world.metadata = self._merge_metadata(world.metadata, dict(payload))
        world.metadata = self._merge_metadata(world.metadata, {"source_routes": ["visual_3d"]})
        return world

    def from_math_report(self, report: WorldModelMathReport | None) -> SharedWorldModel:
        if report is None:
            return SharedWorldModel(query="")
        worlds = [self.from_graph(report.graph, source="math_graph")]
        if report.visual_world_model is not None:
            worlds.append(self.from_shared_world(report.visual_world_model, source="vision"))
        world = self.merge(*worlds, query=report.query)
        for name in report.strategy_prior.recommended_operators:
            world.add_operator(VLSOOperator(str(name), "math_strategy", "Recommended by the math strategy prior.", "math", 0.74))
        for item in report.operator_trace:
            world.add_operator(VLSOOperator(str(item), "math_trace", "Observed in the math operator trace.", "math", 0.68))
        world.goals = self._merge_unique(world.goals, list(report.strategy_prior.proposed_subgoals))
        world.constraints = self._merge_unique(world.constraints, list(report.strategy_prior.latent_constraints))
        world.inferred_steps = self._merge_unique(world.inferred_steps, list(report.solution_process) + list(report.next_actions))
        if not report.solved:
            world.warnings = self._merge_unique(world.warnings, ["Math world model has not verified the final answer yet."])
        for index, candidate in enumerate(report.candidates, start=1):
            world.add_event(VLSOEvent(f"math_candidate_{index}", candidate.kind, "candidate_answer", "math", None, float(candidate.confidence), {"source": candidate.source}, {"answer": candidate.answer, "score": candidate.score, "verified": candidate.verified, "verification_score": candidate.verification_score}))
        for index, check in enumerate(report.checks, start=1):
            world.add_event(VLSOEvent(f"math_check_{index}", check.name, "verification_check", "math", None, float(check.score), {}, {"passed": check.passed, "detail": check.detail}))
            if not check.passed and check.detail:
                world.warnings = self._merge_unique(world.warnings, [str(check.detail)])
        world.metadata = self._merge_metadata(world.metadata, {
            "source_routes": ["math"],
            "math": {
                "task_mode": report.task_mode,
                "family": report.strategy_prior.family,
                "difficulty": report.strategy_prior.difficulty,
                "chosen_answer": report.chosen_answer,
                "solved": report.solved,
                "verification_score": report.verification_score,
                "matched_patterns": list(report.matched_patterns),
                "visual_answer": dict(report.visual_answer or {}),
            },
        })
        return world

    def from_cp_solution(self, query: str, solution: ContestSolution | None, *, structure: ContestProblemStructure | None = None) -> SharedWorldModel:
        world = SharedWorldModel(query=query)
        if solution is None:
            world.warnings = ["Competitive programming solver did not produce a candidate yet."]
            world.metadata = {"source_routes": ["cp"]}
            return world
        problem_id = "contest_problem"
        algorithm_id = self._slug(solution.category or "contest_strategy", "contest_strategy")
        world.add_entity(VLSOEntity(problem_id, "contest problem", "cp", "problem", {"query": query}))
        world.add_entity(VLSOEntity(algorithm_id, (solution.category or "contest strategy").replace("_", " "), "cp", "algorithm", {"approach": solution.approach, "confidence": solution.confidence, "time_complexity": solution.time_complexity, "memory_complexity": solution.memory_complexity}))
        world.add_relation(VLSORelation(problem_id, "USES_ALGORITHM", algorithm_id, "cp", float(solution.confidence), {}))
        goals = list(structure.goal_types if structure is not None else solution.goal_types)
        domains = list(structure.domain_tags if structure is not None else solution.domain_tags)
        frames = list(structure.logical_frames if structure is not None else solution.logical_frames)
        constraints = list(structure.extracted_constraints if structure is not None else solution.extracted_constraints)
        candidate_algorithms = list(structure.candidate_algorithms if structure is not None else [])
        for goal in goals:
            goal_id = self._slug(goal, "goal")
            world.add_entity(VLSOEntity(goal_id, goal.replace("_", " "), "cp", "goal", {}))
            world.add_relation(VLSORelation(problem_id, "TARGETS_GOAL", goal_id, "cp", 1.0, {}))
        for domain in domains:
            domain_id = self._slug(domain, "domain")
            world.add_entity(VLSOEntity(domain_id, domain.replace("_", " "), "cp", "domain", {}))
            world.add_relation(VLSORelation(problem_id, "IN_DOMAIN", domain_id, "cp", 1.0, {}))
        for frame in frames:
            frame_id = self._slug(frame, "frame")
            world.add_entity(VLSOEntity(frame_id, frame.replace("_", " "), "cp", "logical_frame", {}))
            world.add_relation(VLSORelation(problem_id, "MATCHES_FRAME", frame_id, "cp", 1.0, {}))
        for index, constraint in enumerate(constraints, start=1):
            constraint_id = f"constraint_{index}"
            world.add_entity(VLSOEntity(constraint_id, str(constraint), "cp", "constraint", {}))
            world.add_relation(VLSORelation(problem_id, "REQUIRES", constraint_id, "cp", 1.0, {}))
        for index, candidate in enumerate(candidate_algorithms[:5], start=1):
            candidate_id = self._slug(candidate, f"candidate_algorithm_{index}")
            world.add_entity(VLSOEntity(candidate_id, candidate.replace("_", " "), "cp", "algorithm_candidate", {}))
            world.add_relation(VLSORelation(problem_id, "CONSIDERS", candidate_id, "cp", 1.0, {}))
        for operator_name in solution.dsl_operators:
            world.add_operator(VLSOOperator(str(operator_name), "cp_dsl", "Competitive programming DSL operator recovered from the problem statement.", "cp", 0.75))
        world.goals = self._merge_unique(world.goals, [goal.replace("_", " ") for goal in goals] or ["solve competitive programming problem"])
        world.constraints = self._merge_unique(world.constraints, [str(item) for item in constraints] + [f"time complexity: {solution.time_complexity}", f"memory complexity: {solution.memory_complexity}"])
        world.inferred_steps = self._merge_unique(world.inferred_steps, list(solution.reasoning_steps))
        validation_ok = bool(solution.validation_report.get("overall_ok")) if isinstance(solution.validation_report, dict) else False
        world.add_event(VLSOEvent("cp_compile_check", "compile check", "verification_check", "cp", None, 1.0 if solution.compile_ok else 0.3, {"algorithm": algorithm_id}, {"passed": solution.compile_ok, "command": solution.compile_command, "stderr": solution.compile_stderr}))
        world.add_event(VLSOEvent("cp_validation_check", "validation check", "verification_check", "cp", None, 1.0 if validation_ok else 0.35, {"algorithm": algorithm_id}, dict(solution.validation_report)))
        if solution.repaired:
            world.add_event(VLSOEvent("cp_repair_attempts", "repair attempts", "repair", "cp", None, 0.7, {}, {"attempts": list(solution.repair_attempts)}))
        warnings: list[str] = []
        if not solution.compile_ok:
            warnings.append("C++ compile check did not pass.")
        if isinstance(solution.validation_report, dict) and solution.validation_report.get("overall_ok") is False:
            warnings.append("Validator found a behavioral issue in the current CP candidate.")
        if solution.repaired:
            warnings.append("The CP candidate needed repair before finalizing.")
        world.warnings = self._merge_unique(world.warnings, warnings)
        world.audit_trace = self._merge_unique(world.audit_trace, [str(item) for item in solution.reasoning_steps[:4]] + [str(item.get("category") or "") for item in solution.search_trace if isinstance(item, dict)])
        world.metadata = self._merge_metadata(world.metadata, {
            "source_routes": ["cp"],
            "cp": {
                "approach": solution.approach,
                "category": solution.category,
                "time_complexity": solution.time_complexity,
                "memory_complexity": solution.memory_complexity,
                "cues": list(solution.cues),
                "knowledge_sources": list(solution.knowledge_sources),
                "selection_strategy": solution.selection_strategy,
                "search_trace": list(solution.search_trace),
            },
            "cp_structure": self._to_plain(structure),
        })
        return world
    def merge(self, *worlds: SharedWorldModel | dict[str, Any] | None, query: str = "") -> SharedWorldModel:
        merged = SharedWorldModel(query=query)
        for item in worlds:
            world = self.from_shared_world(item, query=query) if not isinstance(item, SharedWorldModel) else self.from_shared_world(item, query=query)
            if not merged.query and world.query:
                merged.query = world.query
            for entity in world.entities:
                merged.add_entity(entity)
            for relation in world.relations:
                merged.add_relation(relation)
            for operator in world.operators:
                merged.add_operator(operator)
            for event in world.events:
                merged.add_event(event)
            merged.goals = self._merge_unique(merged.goals, world.goals)
            merged.constraints = self._merge_unique(merged.constraints, world.constraints)
            merged.inferred_steps = self._merge_unique(merged.inferred_steps, world.inferred_steps)
            merged.warnings = self._merge_unique(merged.warnings, world.warnings)
            merged.audit_trace = self._merge_unique(merged.audit_trace, world.audit_trace)
            merged.metadata = self._merge_metadata(merged.metadata, world.metadata)
        if query:
            merged.query = query
        return merged

    def reason(self, world: SharedWorldModel | dict[str, Any] | None) -> UnifiedWorldReasoning:
        shared = world if isinstance(world, SharedWorldModel) else self.from_shared_world(world)
        primary_goal = self._first_text(
            self._deep_get(shared.metadata, "context_frame", "primary_goal"),
            shared.goals[0] if shared.goals else "",
            self._deep_get(shared.metadata, "math", "chosen_answer"),
            shared.query,
        )
        active_domains = self._merge_unique([], [str(item) for item in shared.metadata.get("source_routes", []) if str(item).strip()] + self._labels_for_relation_targets(shared, self._DOMAIN_RELATIONS))
        active_modalities = self._merge_unique([], [entity.modality for entity in shared.entities] + [event.modality for event in shared.events] + [operator.source_modality for operator in shared.operators])
        blockers = self._relation_descriptions(shared, self._BLOCKER_RELATIONS)
        prerequisites = self._relation_descriptions(shared, self._PREREQUISITE_RELATIONS)
        alternatives = self._relation_descriptions(shared, self._ALTERNATIVE_RELATIONS)
        next_steps = self._merge_unique([], list(shared.inferred_steps))
        warnings = self._merge_unique([], list(shared.warnings))
        evidence = self._merge_unique([], self._relation_descriptions(shared, {"PART_OF", "TARGET_OF_ATTENTION", "AFFORDS", "USES_ALGORITHM", "IN_DOMAIN"}) + [event.label for event in shared.events[:4]])
        operator_trace = self._merge_unique([], [operator.name for operator in shared.operators] + [str(item) for item in shared.audit_trace[:4]])
        summary = UnifiedWorldReasoning(
            query=shared.query,
            primary_goal=self._humanize(primary_goal),
            active_domains=[self._humanize(item) for item in active_domains],
            active_modalities=[self._humanize(item) for item in active_modalities],
            blockers=[self._humanize(item) for item in blockers],
            prerequisites=[self._humanize(item) for item in prerequisites],
            alternatives=[self._humanize(item) for item in alternatives],
            next_steps=[self._humanize(item) for item in next_steps],
            warnings=[self._humanize(item) for item in warnings],
            evidence=[self._humanize(item) for item in evidence],
            operator_trace=[self._humanize(item) for item in operator_trace],
        )
        summary.summary = summary.to_text()
        return summary

    @staticmethod
    def _to_plain(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): UnifiedWorldModelEngine._to_plain(item) for key, item in value.items()}
        if isinstance(value, list):
            return [UnifiedWorldModelEngine._to_plain(item) for item in value]
        if is_dataclass(value):
            return {str(key): UnifiedWorldModelEngine._to_plain(item) for key, item in asdict(value).items()}
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return UnifiedWorldModelEngine._to_plain(model_dump())
        return str(value)

    @staticmethod
    def _merge_unique(base: list[str], incoming: list[str]) -> list[str]:
        seen = {str(item).strip() for item in base if str(item).strip()}
        merged = [str(item).strip() for item in base if str(item).strip()]
        for item in incoming:
            normalized = str(item).strip()
            if not normalized or normalized in seen:
                continue
            merged.append(normalized)
            seen.add(normalized)
        return merged

    @classmethod
    def _merge_metadata(cls, base: dict[str, Any], incoming: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(base)
        for key, value in (incoming or {}).items():
            if value in (None, "", [], {}):
                continue
            plain = cls._to_plain(value)
            if key not in merged:
                merged[key] = plain
            elif isinstance(merged[key], list) and isinstance(plain, list):
                merged[key] = cls._merge_unique([str(item) for item in merged[key]], [str(item) for item in plain])
            elif isinstance(merged[key], dict) and isinstance(plain, dict):
                merged[key] = cls._merge_metadata(merged[key], plain)
            elif merged[key] != plain:
                merged[f"{key}_merged"] = cls._merge_unique([str(item) for item in merged.get(f"{key}_merged", []) if str(item).strip()], [str(merged[key]), str(plain)])
        return merged

    @staticmethod
    def _slug(value: Any, prefix: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
        return normalized or prefix

    @staticmethod
    def _deep_get(mapping: dict[str, Any], *keys: str) -> Any:
        current: Any = mapping
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    @staticmethod
    def _first_text(*values: Any) -> str:
        for value in values:
            normalized = str(value or "").strip()
            if normalized:
                return normalized
        return ""

    @staticmethod
    def _humanize(value: str) -> str:
        text = str(value or "").strip().replace("_", " ")
        return re.sub(r"\s+", " ", text).strip()

    def _relation_descriptions(self, world: SharedWorldModel, relation_names: set[str]) -> list[str]:
        lines: list[str] = []
        for relation in world.relations:
            if relation.relation not in relation_names:
                continue
            lines.append(f"{self._entity_label(world, relation.source)} --{relation.relation}--> {self._entity_label(world, relation.target)}")
        return self._merge_unique([], lines)

    def _labels_for_relation_targets(self, world: SharedWorldModel, relation_names: set[str]) -> list[str]:
        labels: list[str] = []
        for relation in world.relations:
            if relation.relation in relation_names:
                labels.append(self._entity_label(world, relation.target))
        return self._merge_unique([], labels)

    @staticmethod
    def _entity_label(world: SharedWorldModel, entity_id: str) -> str:
        for entity in world.entities:
            if entity.id == entity_id:
                return entity.label or entity.id
        return entity_id

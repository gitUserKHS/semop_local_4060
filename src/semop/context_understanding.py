from __future__ import annotations

from .basis_operators import basis_operator_axis, canonicalize_basis_signature, infer_basis_operators
from .commonsense_kb import concept_label
from .structures import ContextFrame, StructuredMeaningGraph


class OperatorContextAnalyzer:
    def analyze(self, graph: StructuredMeaningGraph) -> ContextFrame:
        basis_signature = canonicalize_basis_signature(self._basis_signature(graph))
        primary_goal = self._primary_goal(graph)
        satisfied_requirements = list(dict.fromkeys(graph.satisfied_premises))
        missing_requirements = list(dict.fromkeys(graph.missing_premises))
        blockers = self._blockers(graph)
        active_constraints = list(
            dict.fromkeys(
                [
                    *[item for item in graph.required_premises if item not in satisfied_requirements],
                    *blockers,
                ]
            )
        )
        available_alternatives = self._alternatives(graph)
        risk_signals = self._risk_signals(graph)
        operator_view = [item.operator_name for item in graph.operator_decompositions[:4]]
        functor_view = [item.name for item in graph.functor_hypotheses[:4]]
        focus_entities = self._focus_entities(graph, primary_goal)
        frame_type = self._frame_type(graph, primary_goal, basis_signature)
        reasoning_mode = self._reasoning_mode(graph, active_constraints, risk_signals, available_alternatives)
        evidence = self._evidence(graph, basis_signature, operator_view, functor_view, risk_signals)
        summary = self._summary(
            graph=graph,
            frame_type=frame_type,
            reasoning_mode=reasoning_mode,
            primary_goal=primary_goal,
            active_constraints=active_constraints,
            satisfied_requirements=satisfied_requirements,
            available_alternatives=available_alternatives,
            risk_signals=risk_signals,
        )
        return ContextFrame(
            frame_type=frame_type,
            reasoning_mode=reasoning_mode,
            primary_goal=primary_goal,
            focus_entities=focus_entities,
            active_constraints=active_constraints,
            missing_requirements=missing_requirements,
            satisfied_requirements=satisfied_requirements,
            available_alternatives=available_alternatives,
            risk_signals=risk_signals,
            basis_signature=basis_signature,
            operator_view=operator_view,
            functor_view=functor_view,
            evidence=evidence,
            summary=summary,
        )

    @staticmethod
    def _basis_signature(graph: StructuredMeaningGraph) -> list[str]:
        runtime_basis = graph.operator_execution.basis_operator_hits if graph.operator_execution is not None else []
        return [item for item in list(runtime_basis) + infer_basis_operators(graph) if basis_operator_axis(item) != "unknown"]

    @staticmethod
    def _primary_goal(graph: StructuredMeaningGraph) -> str:
        if graph.hidden_goals:
            return graph.hidden_goals[0]
        for edge in graph.edges:
            if edge.relation == "GOAL_OF":
                return edge.source
        for node in graph.nodes:
            if node.kind in {"task", "hidden_goal"}:
                return node.id
        return graph.intent

    @staticmethod
    def _blockers(graph: StructuredMeaningGraph) -> list[str]:
        return list(dict.fromkeys(edge.target for edge in graph.edges if edge.relation == "BLOCKED_BY"))

    @staticmethod
    def _alternatives(graph: StructuredMeaningGraph) -> list[str]:
        symbolic = [edge.target for edge in graph.edges if edge.relation == "ALTERNATIVE"]
        return list(dict.fromkeys(symbolic + graph.creative_alternatives))[:6]

    @staticmethod
    def _risk_signals(graph: StructuredMeaningGraph) -> list[str]:
        risks = [
            f"{item.action}:{item.hidden_goal}:{item.status}"
            for item in graph.goal_preservation_checks
            if item.status in {"risk_high", "blocked", "invalid", "conditionally_valid"}
        ]
        if graph.clarification_needed:
            risks.append("clarification_needed")
        if graph.operator_execution is not None:
            risks.extend(graph.operator_execution.warnings[:3])
        return list(dict.fromkeys(risks))[:6]

    @staticmethod
    def _focus_entities(graph: StructuredMeaningGraph, primary_goal: str) -> list[str]:
        entities: list[str] = []
        if primary_goal:
            entities.append(primary_goal)
        for edge in graph.edges:
            if edge.relation in {"REQUIRES", "BLOCKED_BY", "ALTERNATIVE", "TYPICAL_FOR", "CONTAINS"}:
                entities.extend([edge.source, edge.target])
        for premise in graph.required_premises[:3]:
            entities.append(premise)
        return list(dict.fromkeys(item for item in entities if item))[:8]

    @staticmethod
    def _frame_type(graph: StructuredMeaningGraph, primary_goal: str, basis_signature: list[str]) -> str:
        lowered_goal = primary_goal.lower()
        basis = set(basis_signature)
        if {"PARALLEL", "PERPENDICULAR", "EQUAL_LENGTH"} & basis:
            return "geometry_reasoning"
        if "efficient_solution_goal" in lowered_goal or {"RANGE_QUERY", "OPTIMIZE", "FEASIBILITY_CHECK"} & basis:
            return "computational_reasoning"
        if any(token in lowered_goal for token in ["retrieve", "store", "contain", "access", "pour", "pass_through"]):
            return "access_reasoning"
        if any(token in lowered_goal for token in ["clean_car", "reservation", "booking"]) or {"vehicle_present", "traffic"} & set(graph.node_ids()):
            return "mobility_reasoning"
        if graph.source_context.strip() and graph.domain != "general":
            return "document_grounded_reasoning"
        return "generic_reasoning"

    @staticmethod
    def _reasoning_mode(
        graph: StructuredMeaningGraph,
        active_constraints: list[str],
        risk_signals: list[str],
        available_alternatives: list[str],
    ) -> str:
        if graph.clarification_needed:
            return "clarify_goal"
        if risk_signals or active_constraints:
            return "constraint_first"
        if graph.satisfied_premises or graph.plan:
            return "execution_ready"
        if available_alternatives:
            return "alternative_search"
        return "structure_first"

    def _evidence(
        self,
        graph: StructuredMeaningGraph,
        basis_signature: list[str],
        operator_view: list[str],
        functor_view: list[str],
        risk_signals: list[str],
    ) -> list[str]:
        evidence: list[str] = []
        if basis_signature:
            evidence.append("basis=" + ", ".join(basis_signature[:6]))
        if operator_view:
            evidence.append("decomposition=" + ", ".join(operator_view[:3]))
        if functor_view:
            evidence.append("functors=" + ", ".join(functor_view[:2]))
        if graph.operator_execution is not None and graph.operator_execution.derived_decisions:
            evidence.append("runtime_decisions=" + ", ".join(graph.operator_execution.derived_decisions[:2]))
        if risk_signals:
            evidence.append("risks=" + ", ".join(risk_signals[:2]))
        return evidence[:5]

    def _summary(
        self,
        graph: StructuredMeaningGraph,
        frame_type: str,
        reasoning_mode: str,
        primary_goal: str,
        active_constraints: list[str],
        satisfied_requirements: list[str],
        available_alternatives: list[str],
        risk_signals: list[str],
    ) -> str:
        goal_text = concept_label(primary_goal) if primary_goal else graph.intent.replace("_", " ")
        parts = [f"{goal_text} 중심의 {frame_type} 맥락이다."]
        if active_constraints:
            parts.append(
                "우선 제약을 해소해야 한다: "
                + ", ".join(concept_label(item) for item in active_constraints[:3])
                + "."
            )
        elif satisfied_requirements:
            parts.append(
                "핵심 전제는 이미 맞춰져 있다: "
                + ", ".join(concept_label(item) for item in satisfied_requirements[:3])
                + "."
            )
        if available_alternatives and reasoning_mode in {"constraint_first", "alternative_search"}:
            parts.append(
                "대안 경로도 열려 있다: "
                + ", ".join(concept_label(item) if "_" in item and " " not in item else item for item in available_alternatives[:2])
                + "."
            )
        if risk_signals:
            parts.append("현재 행동은 숨은 목표를 깨뜨릴 위험이 있다.")
        return " ".join(parts)



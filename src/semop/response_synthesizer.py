from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .commonsense_kb import concept_label
from .structures import StructuredMeaningGraph


@dataclass
class SynthesizedResponse:
    summary: str
    reasoning: List[str]
    symbolic_insights: List[str]
    hidden_premises: List[str]
    analogical_memories: List[str]
    context_frame: List[str]
    goal_checks: List[str]
    compiled_execution: List[str]
    plan: List[str]
    creative_options: List[str]
    induced_operators: List[str]
    grammar_hypotheses: List[str]
    cautions: List[str]

    def to_text(self) -> str:
        lines: List[str] = [f"핵심 판단: {self.summary}"]

        if self.reasoning:
            lines.append("")
            lines.append("논리적 근거:")
            lines.extend(f"- {item}" for item in self.reasoning)

        if self.symbolic_insights:
            lines.append("")
            lines.append("상징적 해석:")
            lines.extend(f"- {item}" for item in self.symbolic_insights)

        if self.hidden_premises:
            lines.append("")
            lines.append("숨은 전제:")
            lines.extend(f"- {item}" for item in self.hidden_premises)

        if self.analogical_memories:
            lines.append("")
            lines.append("연상된 사례:")
            lines.extend(f"- {item}" for item in self.analogical_memories)

        if self.context_frame:
            lines.append("")
            lines.append("맥락 프레임:")
            lines.extend(f"- {item}" for item in self.context_frame)

        if self.goal_checks:
            lines.append("")
            lines.append("목표 보존 검사:")
            lines.extend(f"- {item}" for item in self.goal_checks)

        if self.compiled_execution:
            lines.append("")
            lines.append("연산자 실행:")
            lines.extend(f"- {item}" for item in self.compiled_execution)

        if self.plan:
            lines.append("")
            lines.append("실행 계획:")
            lines.extend(f"{index}. {item}" for index, item in enumerate(self.plan, start=1))

        if self.creative_options:
            lines.append("")
            lines.append("창의적 대안:")
            lines.extend(f"- {item}" for item in self.creative_options)

        if self.induced_operators:
            lines.append("")
            lines.append("유도된 연산자:")
            lines.extend(f"- {item}" for item in self.induced_operators)

        if self.grammar_hypotheses:
            lines.append("")
            lines.append("유도된 문법 가설:")
            lines.extend(f"- {item}" for item in self.grammar_hypotheses)

        if self.cautions:
            lines.append("")
            lines.append("주의할 점:")
            lines.extend(f"- {item}" for item in self.cautions)

        return "\n".join(lines)


class ResponseSynthesizer:
    def synthesize(self, graph: StructuredMeaningGraph) -> SynthesizedResponse:
        return SynthesizedResponse(
            summary=self._summary(graph),
            reasoning=self._reasoning(graph),
            symbolic_insights=self._symbolic_insights(graph),
            hidden_premises=self._hidden_premise_lines(graph),
            analogical_memories=self._analogical_memories(graph),
            context_frame=self._context_frame(graph),
            goal_checks=self._goal_checks(graph),
            compiled_execution=self._compiled_execution(graph),
            plan=[step.action for step in graph.plan if step.status == "valid"],
            creative_options=list(dict.fromkeys(graph.creative_alternatives + self._alternative_relations(graph)))[:5],
            induced_operators=[self._format_operator(candidate) for candidate in graph.induced_operators[:5]],
            grammar_hypotheses=graph.grammar_hypotheses[:4],
            cautions=list(dict.fromkeys(graph.invalid_advice + graph.warnings))[:7],
        )

    def _hidden_premise_lines(self, graph: StructuredMeaningGraph) -> List[str]:
        lines = list(graph.hidden_assumptions[:3])
        for item in graph.premise_validations[:3]:
            suffix = f", requirement={item.requirement_state}" if item.requirement_state != 'unknown' else ''
            lines.append(f"{concept_label(item.premise)}: {item.status}{suffix} (support={item.support_score}, contradiction={item.contradiction_score})")
        if graph.satisfied_premises:
            lines.append('Satisfied: ' + ', '.join(concept_label(item) for item in graph.satisfied_premises[:3]))
        if graph.missing_premises:
            lines.append('Missing: ' + ', '.join(concept_label(item) for item in graph.missing_premises[:3]))
        if graph.clarification_score > 0.0:
            lines.append(f'Clarification score: {graph.clarification_score:.2f}')
        if graph.clarification_reasons:
            lines.extend(f'Clarification reason: {item}' for item in graph.clarification_reasons[:2])
        return lines[:10]

    def _analogical_memories(self, graph: StructuredMeaningGraph) -> List[str]:
        if not graph.analogical_matches:
            return []
        lines: List[str] = []
        for item in graph.analogical_matches[:3]:
            shared = []
            if item.shared_goals:
                shared.append('goal=' + ', '.join(concept_label(value) for value in item.shared_goals[:2]))
            if item.shared_requirements:
                shared.append('premise=' + ', '.join(concept_label(value) for value in item.shared_requirements[:2]))
            if item.shared_operator_families:
                shared.append('operator=' + ', '.join(item.shared_operator_families[:2]))
            detail = '; '.join(shared) if shared else ', '.join(item.shared_basis[:2])
            lines.append(f"{item.analogy_type}: {item.query} (score={item.score:.2f}; {detail})")
        return lines[:3]

    def _summary(self, graph: StructuredMeaningGraph) -> str:
        arithmetic = next((result for result in graph.symbolic_results if result.domain == "arithmetic"), None)
        if arithmetic is not None:
            return arithmetic.answer

        olympiad = next((result for result in graph.symbolic_results if result.domain == "olympiad_proof"), None)
        if olympiad is not None:
            return olympiad.answer

        document = next((result for result in graph.symbolic_results if result.domain == "document_grounding"), None)
        if document is not None:
            return document.answer

        if any(check.action == 'walk_without_car' and check.status == 'risk_high' for check in graph.goal_preservation_checks):
            return '걸어서 도착할 수는 있어도, 차를 세차하는 숨은 목표는 보통 깨질 가능성이 높다.'

        relations = graph.relation_tuples()
        if ("insert_book", "REQUIRES", "open_access") in relations:
            return "가방을 먼저 열고 공간을 확인한 뒤 책을 넣는 순서가 맞다."
        if ("drive_to_car_wash", "BLOCKED_BY", "traffic") in relations:
            return "교통 제약을 먼저 처리하고, 필요하면 더 가까운 세차장이나 출장 세차로 목표를 우회하는 편이 낫다."
        if graph.context_frame is not None and graph.context_frame.summary:
            return graph.context_frame.summary
        if graph.plan:
            return graph.plan[0].action
        return "질문을 구조적으로 분해해 실행 가능한 선택지를 추렸다."

    def _reasoning(self, graph: StructuredMeaningGraph) -> List[str]:
        items: List[str] = []
        for edge in graph.edges:
            source = self._label(graph, edge.source)
            target = self._label(graph, edge.target)
            if edge.relation == "REQUIRES":
                items.append(f"{source}에는 {target} 전제조건이 있다.")
            elif edge.relation == "BLOCKED_BY":
                items.append(f"{source}는 {target} 때문에 직접 실행이 막힌다.")
            elif edge.relation == "PART_OF":
                items.append(f"{source}는 {target}의 구성 요소다.")
            elif edge.relation == "ALTERNATIVE":
                items.append(f"{target}은 같은 목표를 위한 대안 후보다.")
            elif edge.relation == "TYPICAL_FOR":
                items.append(f"{source}는 보통 {target} 목적과 연결된다.")
        return list(dict.fromkeys(items))[:6]

    def _symbolic_insights(self, graph: StructuredMeaningGraph) -> List[str]:
        insights: List[str] = []
        for result in graph.symbolic_results:
            line = result.answer
            if result.equations:
                line += f" | 식: {result.equations[0]}"
            elif result.evidence:
                line += f" | 근거: {result.evidence[0]}"
            insights.append(line)
        return insights[:4]

    def _context_frame(self, graph: StructuredMeaningGraph) -> List[str]:
        frame = graph.context_frame
        if frame is None:
            return []
        lines = [frame.summary]
        if frame.reasoning_mode:
            lines.append(f"Mode: {frame.reasoning_mode}")
        if frame.basis_signature:
            lines.append("Basis: " + ", ".join(frame.basis_signature[:6]))
        if frame.operator_view:
            lines.append("Operator view: " + ", ".join(frame.operator_view[:3]))
        if frame.functor_view:
            lines.append("Functor view: " + ", ".join(frame.functor_view[:2]))
        if frame.active_constraints:
            lines.append("Active constraints: " + ", ".join(concept_label(item) for item in frame.active_constraints[:3]))
        return lines[:6]

    def _goal_checks(self, graph: StructuredMeaningGraph) -> List[str]:
        checks: List[str] = []
        for item in graph.goal_preservation_checks:
            goal = concept_label(item.hidden_goal)
            checks.append(f"{item.action}: {goal} 기준 {item.status} ({item.rationale})")
        if graph.clarification_needed and len(checks) < 4:
            checks.append('표면 행동만으로 목적이 확정되지 않아 목적 확인 질문이 유리하다.')
        return checks[:4]

    def _compiled_execution(self, graph: StructuredMeaningGraph) -> List[str]:
        report = graph.operator_execution
        if report is None:
            return []
        lines: List[str] = []
        if report.satisfied_facts:
            lines.append('Satisfied facts: ' + ', '.join(report.satisfied_facts[:4]))
        if report.missing_facts:
            lines.append('Missing facts: ' + ', '.join(report.missing_facts[:4]))
        if report.compiler_findings:
            lines.append('Compiler: ' + report.compiler_findings[0])
        if report.composition_score > 0.0:
            lines.append(f'Composition score: {report.composition_score:.2f}')
        if report.counterexample_repairs:
            lines.append('Repair: ' + report.counterexample_repairs[0])
        grounding = self._grounded_evidence_lines(graph)
        if grounding:
            lines.append('Grounding: ' + ' | '.join(grounding[:2]))
        if report.claim_groundings:
            grounded = sum(1 for item in report.claim_groundings if item.grounded)
            lines.append(f'Claim support: {grounded}/{len(report.claim_groundings)} grounded (score={report.claim_grounding_score:.2f})')
            for item in report.claim_groundings[:2]:
                if item.grounded and item.supports:
                    support = item.supports[0].strip().replace('\n', ' ')
                    lines.append(f'Claim: {item.claim} <- {support[:72]}')
                elif not item.grounded:
                    lines.append(f'Unsupported claim: {item.claim}')
        lines.extend(f'Decision: {item}' for item in report.derived_decisions[:4])
        return lines[:10]

    def _grounded_evidence_lines(self, graph: StructuredMeaningGraph) -> List[str]:
        evidence_lookup = {node.id: node for node in graph.nodes if node.kind == "evidence"}
        lines: List[str] = []
        for edge in graph.edges:
            if edge.source != "question" or edge.relation != "GROUNDED_BY":
                continue
            node = evidence_lookup.get(edge.target)
            if node is None:
                continue
            modality = str(node.attributes.get("modality", "evidence"))
            text = str(node.attributes.get("text", node.label)).strip()
            if not text:
                continue
            lines.append(f"{modality}: {text}")
        return list(dict.fromkeys(lines))[:3]

    def _alternative_relations(self, graph: StructuredMeaningGraph) -> List[str]:
        return [f"{self._label(graph, edge.target)} 쪽으로 계획을 바꿔 볼 수 있다." for edge in graph.edges if edge.relation == "ALTERNATIVE"]

    def _label(self, graph: StructuredMeaningGraph, concept_id: str) -> str:
        for node in graph.nodes:
            if node.id == concept_id and node.label:
                return node.label
        return concept_label(concept_id)

    @staticmethod
    def _format_operator(candidate) -> str:
        inputs = ", ".join(candidate.input_types) if candidate.input_types else "meaning_unit"
        suffixes: List[str] = []
        if any(tag.startswith("memory_prior:") for tag in candidate.provenance):
            suffixes.append("memory prior")
        if candidate.abstract_parents:
            suffixes.append("abstract: " + ",".join(candidate.abstract_parents[:2]))
        if any(tag.startswith("symbolic:") for tag in candidate.provenance):
            suffixes.append("symbolic")
        suffix = f" [{' / '.join(suffixes)}]" if suffixes else ""
        return f"{candidate.name}: {inputs} -> {candidate.output_type} ({candidate.description}){suffix}"


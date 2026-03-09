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
            plan=[step.action for step in graph.plan if step.status == "valid"],
            creative_options=list(dict.fromkeys(graph.creative_alternatives + self._alternative_relations(graph)))[:5],
            induced_operators=[self._format_operator(candidate) for candidate in graph.induced_operators[:5]],
            grammar_hypotheses=graph.grammar_hypotheses[:4],
            cautions=list(dict.fromkeys(graph.invalid_advice + graph.warnings))[:7],
        )

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

        relations = graph.relation_tuples()
        if ("insert_book", "REQUIRES", "open_access") in relations:
            return "가방을 먼저 열고 공간을 확인한 뒤 책을 넣는 순서가 맞다."
        if ("drive_to_car_wash", "BLOCKED_BY", "traffic") in relations:
            return "교통 제약을 먼저 처리하고, 필요하면 더 가까운 세차장이나 출장 세차로 목표를 우회하는 편이 낫다."
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

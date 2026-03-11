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
    goal_checks: List[str]
    plan: List[str]
    creative_options: List[str]
    induced_operators: List[str]
    grammar_hypotheses: List[str]
    cautions: List[str]

    def to_text(self) -> str:
        lines: List[str] = [f"\ud575\uc2ec \ud310\ub2e8: {self.summary}"]

        if self.reasoning:
            lines.append("")
            lines.append("\ub17c\ub9ac\uc801 \uadfc\uac70:")
            lines.extend(f"- {item}" for item in self.reasoning)

        if self.symbolic_insights:
            lines.append("")
            lines.append("\uc0c1\uc9d5\uc801 \ud574\uc11d:")
            lines.extend(f"- {item}" for item in self.symbolic_insights)

        if self.hidden_premises:
            lines.append("")
            lines.append("\uc228\uc740 \uc804\uc81c:")
            lines.extend(f"- {item}" for item in self.hidden_premises)

        if self.goal_checks:
            lines.append("")
            lines.append("\ubaa9\ud45c \ubcf4\uc874 \uac80\uc0ac:")
            lines.extend(f"- {item}" for item in self.goal_checks)

        if self.plan:
            lines.append("")
            lines.append("\uc2e4\ud589 \uacc4\ud68d:")
            lines.extend(f"{index}. {item}" for index, item in enumerate(self.plan, start=1))

        if self.creative_options:
            lines.append("")
            lines.append("\ucc3d\uc758\uc801 \ub300\uc548:")
            lines.extend(f"- {item}" for item in self.creative_options)

        if self.induced_operators:
            lines.append("")
            lines.append("\uc720\ub3c4\ub41c \uc5f0\uc0b0\uc790:")
            lines.extend(f"- {item}" for item in self.induced_operators)

        if self.grammar_hypotheses:
            lines.append("")
            lines.append("\uc720\ub3c4\ub41c \ubb38\ubc95 \uac00\uc124:")
            lines.extend(f"- {item}" for item in self.grammar_hypotheses)

        if self.cautions:
            lines.append("")
            lines.append("\uc8fc\uc758\ud560 \uc810:")
            lines.extend(f"- {item}" for item in self.cautions)

        return "\n".join(lines)


class ResponseSynthesizer:
    def synthesize(self, graph: StructuredMeaningGraph) -> SynthesizedResponse:
        return SynthesizedResponse(
            summary=self._summary(graph),
            reasoning=self._reasoning(graph),
            symbolic_insights=self._symbolic_insights(graph),
            hidden_premises=graph.hidden_assumptions[:4],
            goal_checks=self._goal_checks(graph),
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

        if any(check.action == 'walk_without_car' and check.status == 'risk_high' for check in graph.goal_preservation_checks):
            return '\uac78\uc5b4\uc11c \ub3c4\ucc29\ud560 \uc218\ub294 \uc788\uc5b4\ub3c4, \ucc28\ub97c \uc138\ucc28\ud558\ub294 \uc228\uc740 \ubaa9\ud45c\ub294 \ubcf4\ud1b5 \uae68\uc9c8 \uac00\ub2a5\uc131\uc774 \ub192\ub2e4.'

        relations = graph.relation_tuples()
        if ("insert_book", "REQUIRES", "open_access") in relations:
            return "\uac00\ubc29\uc744 \uba3c\uc800 \uc5f4\uace0 \uacf5\uac04\uc744 \ud655\uc778\ud55c \ub4a4 \ucc45\uc744 \ub123\ub294 \uc21c\uc11c\uac00 \ub9de\ub2e4."
        if ("drive_to_car_wash", "BLOCKED_BY", "traffic") in relations:
            return "\uad50\ud1b5 \uc81c\uc57d\uc744 \uba3c\uc800 \ucc98\ub9ac\ud558\uace0, \ud544\uc694\ud558\uba74 \ub354 \uac00\uae4c\uc6b4 \uc138\ucc28\uc7a5\uc774\ub098 \ucd9c\uc7a5 \uc138\ucc28\ub85c \ubaa9\ud45c\ub97c \uc6b0\ud68c\ud558\ub294 \ud3b8\uc774 \ub0ab\ub2e4."
        if graph.plan:
            return graph.plan[0].action
        return "\uc9c8\ubb38\uc744 \uad6c\uc870\uc801\uc73c\ub85c \ubd84\ud574\ud574 \uc2e4\ud589 \uac00\ub2a5\ud55c \uc120\ud0dd\uc9c0\ub97c \ucd94\ub838\ub2e4."

    def _reasoning(self, graph: StructuredMeaningGraph) -> List[str]:
        items: List[str] = []
        for edge in graph.edges:
            source = self._label(graph, edge.source)
            target = self._label(graph, edge.target)
            if edge.relation == "REQUIRES":
                items.append(f"{source}\uc5d0\ub294 {target} \uc804\uc81c\uc870\uac74\uc774 \uc788\ub2e4.")
            elif edge.relation == "BLOCKED_BY":
                items.append(f"{source}\ub294 {target} \ub54c\ubb38\uc5d0 \uc9c1\uc811 \uc2e4\ud589\uc774 \ub9c9\ud78c\ub2e4.")
            elif edge.relation == "PART_OF":
                items.append(f"{source}\ub294 {target}\uc758 \uad6c\uc131 \uc694\uc18c\ub2e4.")
            elif edge.relation == "ALTERNATIVE":
                items.append(f"{target}\uc740 \uac19\uc740 \ubaa9\ud45c\ub97c \uc704\ud55c \ub300\uc548 \ud6c4\ubcf4\ub2e4.")
            elif edge.relation == "TYPICAL_FOR":
                items.append(f"{source}\ub294 \ubcf4\ud1b5 {target} \ubaa9\uc801\uacfc \uc5f0\uacb0\ub41c\ub2e4.")
        return list(dict.fromkeys(items))[:6]

    def _symbolic_insights(self, graph: StructuredMeaningGraph) -> List[str]:
        insights: List[str] = []
        for result in graph.symbolic_results:
            line = result.answer
            if result.equations:
                line += f" | \uc2dd: {result.equations[0]}"
            elif result.evidence:
                line += f" | \uadfc\uac70: {result.evidence[0]}"
            insights.append(line)
        return insights[:4]

    def _goal_checks(self, graph: StructuredMeaningGraph) -> List[str]:
        checks: List[str] = []
        for item in graph.goal_preservation_checks:
            goal = concept_label(item.hidden_goal)
            checks.append(f"{item.action}: {goal} \uae30\uc900 {item.status} ({item.rationale})")
        if graph.clarification_needed and len(checks) < 4:
            checks.append('\ud45c\uba74 \ud589\ub3d9\ub9cc\uc73c\ub85c \ubaa9\uc801\uc774 \ud655\uc815\ub418\uc9c0 \uc54a\uc544 \ubaa9\uc801 \ud655\uc778 \uc9c8\ubb38\uc774 \uc720\ub9ac\ud558\ub2e4.')
        return checks[:4]

    def _alternative_relations(self, graph: StructuredMeaningGraph) -> List[str]:
        return [f"{self._label(graph, edge.target)} \ucabd\uc73c\ub85c \uacc4\ud68d\uc744 \ubc14\uafd4 \ubcfc \uc218 \uc788\ub2e4." for edge in graph.edges if edge.relation == "ALTERNATIVE"]

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

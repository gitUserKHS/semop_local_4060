from __future__ import annotations

from typing import List

from .olympiad_reasoner import OlympiadReasoner
from .structures import OperatorCandidate, PlanStep, StructuredMeaningGraph, SymbolicResult
from .symbolic_arithmetic import ArithmeticReasoner
from .symbolic_document import DocumentEvidenceReasoner


class SymbolicReasoner:
    def __init__(self) -> None:
        self.arithmetic = ArithmeticReasoner()
        self.olympiad = OlympiadReasoner()
        self.document = DocumentEvidenceReasoner()

    def reason(self, query: str, graph: StructuredMeaningGraph) -> List[SymbolicResult]:
        arithmetic = self.arithmetic.solve(query)
        if arithmetic is not None:
            return [arithmetic]
        olympiad = self.olympiad.solve(query)
        if olympiad is not None:
            return [olympiad]
        evidence = self.document.solve(query)
        if evidence is not None:
            return [evidence]
        return []

    def apply(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        results = self.reason(graph.query, graph)
        if not results:
            return graph

        existing_answers = {result.answer for result in graph.symbolic_results}
        for result in results:
            if result.answer in existing_answers:
                continue
            graph.symbolic_results.append(result)
            if result.domain == "arithmetic":
                self._attach_arithmetic_support(graph, result)
            elif result.domain == "olympiad_proof":
                self._attach_olympiad_support(graph, result)
            elif result.domain == "document_grounding":
                self._attach_document_support(graph, result)
        return graph

    def _attach_arithmetic_support(self, graph: StructuredMeaningGraph, result: SymbolicResult) -> None:
        if not any(candidate.family == "SYMBOLIC_ARITHMETIC" for candidate in graph.induced_operators):
            graph.induced_operators.append(
                OperatorCandidate(
                    name="symbolic_arithmetic_solver",
                    family="SYMBOLIC_ARITHMETIC",
                    arity=1,
                    input_types=["quantity_relation", "word_problem"],
                    output_type="numeric_answer",
                    description="Convert quantity relations into equations and solve them directly.",
                    examples=result.equations[:2] or [result.answer],
                    confidence=round(max(0.72, result.confidence), 2),
                    provenance=["symbolic:arithmetic"],
                )
            )
        if not any(step.id == "symbolic_math_parse" for step in graph.plan):
            graph.plan.insert(
                0,
                PlanStep(
                    id="symbolic_math_parse",
                    action="Extract quantities, prices, ratios, and multipliers into equations.",
                    rationale="Word problems become more reliable after relation-to-equation conversion.",
                ),
            )
        if not any(step.id == "symbolic_math_answer" for step in graph.plan):
            graph.plan.append(
                PlanStep(
                    id="symbolic_math_answer",
                    action=result.answer,
                    rationale="Use the symbolic arithmetic result as a candidate final answer.",
                )
            )
        note = "symbolic arithmetic solver applied"
        if note not in graph.warnings:
            graph.warnings.append(note)
        extra = "When units are mixed, build a table before calculating."
        if extra not in graph.creative_alternatives:
            graph.creative_alternatives.append(extra)

    def _attach_olympiad_support(self, graph: StructuredMeaningGraph, result: SymbolicResult) -> None:
        if not any(candidate.family == "SYMBOLIC_PROOF_SEARCH" for candidate in graph.induced_operators):
            graph.induced_operators.append(
                OperatorCandidate(
                    name="symbolic_olympiad_proof_search",
                    family="SYMBOLIC_PROOF_SEARCH",
                    arity=1,
                    input_types=["proof_problem", "constraint_pattern"],
                    output_type="proof_outline",
                    description="Build a proof state, apply meta-operators, and search for a valid proof sketch.",
                    examples=result.equations[:3] or result.evidence[:2] or [result.answer],
                    confidence=round(max(0.7, result.confidence), 2),
                    provenance=["symbolic:olympiad"],
                )
            )
        if not any(step.id == "symbolic_proof_parse" for step in graph.plan):
            graph.plan.insert(
                0,
                PlanStep(
                    id="symbolic_proof_parse",
                    action="Translate the statement into a proof state with goals, facts, and admissible meta-operators.",
                    rationale="Olympiad problems are closer to proof search than direct answer generation.",
                ),
            )
        if not any(step.id == "symbolic_proof_search" for step in graph.plan):
            graph.plan.append(
                PlanStep(
                    id="symbolic_proof_search",
                    action="Run proof search over parity, divisibility, pigeonhole, and case-split operators.",
                    rationale="Structured proof search makes the intermediate reasoning auditable.",
                )
            )
        if not any(step.id == "symbolic_proof_answer" for step in graph.plan):
            graph.plan.append(
                PlanStep(
                    id="symbolic_proof_answer",
                    action=result.answer,
                    rationale="Use the proof-search result as the current proof sketch.",
                )
            )
        note = "symbolic olympiad proof search applied"
        if note not in graph.warnings:
            graph.warnings.append(note)
        extra = "If the proof stalls, switch to a different meta-operator such as contradiction, invariant, or extremal choice."
        if extra not in graph.creative_alternatives:
            graph.creative_alternatives.append(extra)

    def _attach_document_support(self, graph: StructuredMeaningGraph, result: SymbolicResult) -> None:
        if not any(candidate.family == "SYMBOLIC_EVIDENCE" for candidate in graph.induced_operators):
            graph.induced_operators.append(
                OperatorCandidate(
                    name="symbolic_evidence_grounder",
                    family="SYMBOLIC_EVIDENCE",
                    arity=1,
                    input_types=["document_context", "question"],
                    output_type="evidence_span",
                    description="Retrieve the most relevant supporting sentence before answering.",
                    examples=result.evidence[:2] or [result.answer],
                    confidence=round(max(0.66, result.confidence), 2),
                    provenance=["symbolic:document"],
                )
            )
        if not any(step.id == "symbolic_doc_grounding" for step in graph.plan):
            graph.plan.insert(
                0,
                PlanStep(
                    id="symbolic_doc_grounding",
                    action="Rank document sentences by token overlap with the question.",
                    rationale="Document QA should ground on evidence before answer generation.",
                ),
            )
        note = "symbolic document grounding applied"
        if note not in graph.warnings:
            graph.warnings.append(note)
        extra = "If one sentence is weak, verify the adjacent sentence as well."
        if extra not in graph.creative_alternatives:
            graph.creative_alternatives.append(extra)

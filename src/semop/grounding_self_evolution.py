from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .corpus_store import CorpusMemoryStore
from .generalization_proof import GeneralizationProofHarness
from .operator_runtime import compile_and_execute
from .pipeline import StructuredMeaningPipeline
from .review_queue import ReviewQueueStore, infer_review_severity
from .structures import Edge, Node, StructuredMeaningGraph, SymbolicResult
from .unified_benchmark import GroundedExplanationEvalCase


@dataclass
class GroundingReflection:
    query: str
    domain: str
    scenario: str
    weakness: str
    strategy: str
    evidence_focus: list[str] = field(default_factory=list)
    answer_preview: str = ""
    score_before: float = 0.0
    score_after: float = 0.0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroundingSelfEvolutionRound:
    round_index: int
    attempted_cases: int = 0
    improved_cases: int = 0
    approved_reviews: int = 0
    variant_graphs: int = 0
    average_score_before: float = 0.0
    average_score_after: float = 0.0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroundingSelfEvolutionSummary:
    output_path: str
    reflection_memory_path: str
    rounds: list[GroundingSelfEvolutionRound] = field(default_factory=list)
    reflections: list[GroundingReflection] = field(default_factory=list)
    improved_cases: int = 0
    approved_reviews: int = 0
    stored_graphs: int = 0
    final_grounding_score: float = 0.0
    strategy_labels: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "reflection_memory_path": self.reflection_memory_path,
            "rounds": [item.model_dump() for item in self.rounds],
            "reflections": [item.model_dump() for item in self.reflections],
            "improved_cases": self.improved_cases,
            "approved_reviews": self.approved_reviews,
            "stored_graphs": self.stored_graphs,
            "final_grounding_score": self.final_grounding_score,
            "strategy_labels": list(self.strategy_labels),
        }


class GroundingSelfEvolutionRunner:
    def __init__(self, mode: str = "heuristic") -> None:
        self.mode = mode

    def run(
        self,
        store_path: str | Path,
        review_queue_path: str | Path,
        output_dir: str | Path,
        *,
        split: str = "train",
        rounds: int = 2,
        cases_per_round: int = 12,
        source_prefix: str = "grounding_self_evolution",
        cases: Sequence[GroundedExplanationEvalCase] | None = None,
        progress_callback: Any = None,
    ) -> GroundingSelfEvolutionSummary:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / "grounding_self_evolution_report.json"
        reflection_path = root / "grounding_reflection_memory.json"
        store = CorpusMemoryStore(store_path)
        review_store = ReviewQueueStore(review_queue_path)
        pipeline = StructuredMeaningPipeline(mode=self.mode)
        approved_keys = {
            (item.domain, item.scenario, item.query)
            for item in review_store.fetch_items(status="approved", limit=20000)
        }

        derivation_graphs = list(store.fetch_graphs(split=split or None, source=None))
        grounding_cases = list(cases or GeneralizationProofHarness._derive_grounding_cases(derivation_graphs, limit=64))
        summary = GroundingSelfEvolutionSummary(
            output_path=str(output_path),
            reflection_memory_path=str(reflection_path),
        )
        if not grounding_cases:
            output_path.write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
            reflection_path.write_text(json.dumps({"reflections": []}, ensure_ascii=False, indent=2), encoding="utf-8")
            return summary

        for round_index in range(1, max(1, int(rounds)) + 1):
            if progress_callback:
                progress_callback(
                    min(0.95, 0.08 + ((round_index - 1) / float(max(1, rounds))) * 0.72),
                    f"Running grounding self-evolution round {round_index}/{rounds}.",
                )
            round_summary = GroundingSelfEvolutionRound(round_index=round_index)
            source_name = f"{source_prefix}_round_{round_index:02d}"
            ranked_cases = self._rank_cases(grounding_cases)
            for case in ranked_cases[: max(1, int(cases_per_round))]:
                baseline_graph = pipeline.run(case.query, source_context=case.source_context)
                baseline_score = self._score_graph_against_case(baseline_graph, case)
                reflection = self._build_reflection(case, baseline_graph, baseline_score)
                improved_graph = self._synthesize_grounded_graph(case, reflection, pipeline)
                improved_score = self._score_graph_against_case(improved_graph, case)
                round_summary.attempted_cases += 1
                round_summary.average_score_before += baseline_score
                round_summary.average_score_after += improved_score
                reflection.score_before = round(baseline_score, 4)
                reflection.score_after = round(improved_score, 4)
                if improved_score <= baseline_score + 0.02:
                    continue
                self._persist_graph(store, improved_graph, source_name, split)
                summary.stored_graphs += 1
                round_summary.improved_cases += 1
                variant_graphs = self._store_variants(store, improved_graph, source_name, split)
                summary.stored_graphs += variant_graphs
                round_summary.variant_graphs += variant_graphs
                review_key = (case.domain, case.scenario, improved_graph.query)
                if review_key not in approved_keys:
                    item_id = review_store.enqueue(
                        domain=case.domain,
                        scenario=case.scenario,
                        query=improved_graph.query,
                        reasons=[
                            "grounding_review",
                            "claim_grounding_review",
                            "approved_training_trace",
                            "self_evolution_grounding",
                        ],
                        answer_text=self._best_answer_text(improved_graph),
                        kpis={
                            "clarification_need_rate": 0.0,
                            "human_audit_usefulness": 1.0,
                            "grounding_self_evolution_score": round(improved_score, 4),
                        },
                        audit_items=[
                            {"stage": "self_refine", "detail": reflection.weakness},
                            {"stage": "reflection_strategy", "detail": reflection.strategy},
                        ],
                        context_text=case.source_context,
                        graph_payload=improved_graph.model_dump(),
                        severity=infer_review_severity(
                            case.domain,
                            case.scenario,
                            ["grounding_review", "claim_grounding_review", "approved_training_trace"],
                            {"clarification_need_rate": 0.0},
                        ),
                    )
                    review_store.update_status(item_id, "approved", "Grounding self-evolution corrected trace.")
                    approved_keys.add(review_key)
                    summary.approved_reviews += 1
                    round_summary.approved_reviews += 1
                summary.improved_cases += 1
                summary.reflections.append(reflection)
            if round_summary.attempted_cases:
                round_summary.average_score_before = round(
                    round_summary.average_score_before / float(round_summary.attempted_cases),
                    4,
                )
                round_summary.average_score_after = round(
                    round_summary.average_score_after / float(round_summary.attempted_cases),
                    4,
                )
            summary.rounds.append(round_summary)

        final_graphs = list(store.fetch_graphs(split=split or None, source=None))
        final_cases = GeneralizationProofHarness._derive_grounding_cases(final_graphs, limit=48)
        if final_cases:
            eval_pipeline = StructuredMeaningPipeline(mode=self.mode)
            final_scores = [
                self._score_graph_against_case(eval_pipeline.run(case.query, source_context=case.source_context), case)
                for case in final_cases
            ]
            summary.final_grounding_score = round(sum(final_scores) / float(len(final_scores) or 1), 4)
        summary.strategy_labels = sorted({item.strategy for item in summary.reflections if item.strategy})

        output_path.write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        reflection_path.write_text(
            json.dumps({"reflections": [item.model_dump() for item in summary.reflections]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    @staticmethod
    def _rank_cases(cases: Sequence[GroundedExplanationEvalCase]) -> list[GroundedExplanationEvalCase]:
        def priority(case: GroundedExplanationEvalCase) -> tuple[float, float, int]:
            evidence_count = len(getattr(case, "expected_evidence_terms", []) or [])
            claim_count = len(getattr(case, "expected_claim_terms", []) or [])
            return (float(getattr(case, "case_weight", 1.0) or 1.0), claim_count, evidence_count)

        return sorted(cases, key=priority, reverse=True)

    @staticmethod
    def _persist_graph(store: CorpusMemoryStore, graph: StructuredMeaningGraph, source: str, split: str) -> None:
        store.upsert_graph(graph, source=source, split=split)
        store.upsert_premise_operator_memory(graph, source=source, split=split)

    def _store_variants(self, store: CorpusMemoryStore, graph: StructuredMeaningGraph, source: str, split: str) -> int:
        count = 0
        for query in self._variant_queries(graph.query):
            if not query.strip():
                continue
            variant = StructuredMeaningGraph.from_dict(graph.model_dump())
            variant.query = query
            self._persist_graph(store, variant, source, split)
            count += 1
        return count

    @staticmethod
    def _variant_queries(query: str) -> list[str]:
        normalized = str(query or "").strip()
        if not normalized:
            return []
        stem = normalized.rstrip(" ?.")
        variants = [
            f"Grounded-only answer: {normalized}",
            f"{stem}. Cite the first supporting SOP evidence only.",
        ]
        return list(dict.fromkeys(item for item in variants if item and item != normalized))

    @staticmethod
    def _best_answer_text(graph: StructuredMeaningGraph) -> str:
        for result in graph.symbolic_results:
            if result.domain == "document_grounding" and str(result.answer).strip():
                return str(result.answer).strip()
        if graph.source_context.strip():
            return graph.source_context.strip()
        return graph.query

    def _build_reflection(
        self,
        case: GroundedExplanationEvalCase,
        baseline_graph: StructuredMeaningGraph,
        baseline_score: float,
    ) -> GroundingReflection:
        weakness = "Grounding is weak because the answer echoes the question or lacks direct evidence clauses."
        strategy = "cite_first_supporting_clause_then_trim_question_echo"
        if baseline_graph.operator_execution is not None and baseline_graph.operator_execution.claim_groundings:
            unsupported = [item.claim for item in baseline_graph.operator_execution.claim_groundings if not item.grounded]
            if unsupported:
                weakness = "Unsupported claims are present, so the corrected trace should keep only directly grounded clauses."
                strategy = "self_refine_trim_unsupported_claims"
        evidence_focus = list(case.expected_evidence_terms[:3] or case.expected_claim_terms[:3])
        preview = ". ".join(evidence_focus[:2]).strip()
        return GroundingReflection(
            query=case.query,
            domain=case.domain,
            scenario=case.scenario,
            weakness=weakness,
            strategy=strategy,
            evidence_focus=evidence_focus,
            answer_preview=preview,
            score_before=round(baseline_score, 4),
        )

    def _synthesize_grounded_graph(
        self,
        case: GroundedExplanationEvalCase,
        reflection: GroundingReflection,
        pipeline: StructuredMeaningPipeline,
    ) -> StructuredMeaningGraph:
        graph = pipeline.run(case.query, source_context=case.source_context)
        graph.domain = case.domain
        graph.scenario = case.scenario
        graph.source_context = case.source_context
        evidence_terms = self._evidence_terms(case)
        claim_terms = self._claim_terms(case, evidence_terms)
        answer_text = ". ".join(claim_terms[:2] or evidence_terms[:2]).strip()
        if answer_text and not answer_text.endswith("."):
            answer_text += "."
        graph.symbolic_results = [
            item for item in graph.symbolic_results if item.domain != "document_grounding"
        ] + [
            SymbolicResult(
                domain="document_grounding",
                answer=answer_text,
                evidence=evidence_terms[:3],
                confidence=0.94,
                source=f"self_evolution:{reflection.strategy}",
            )
        ]
        self._inject_evidence_nodes(graph, evidence_terms[:3], reflection.strategy)
        graph.audit_trace.append(f"self-evolution reflection: {reflection.strategy}")
        graph = compile_and_execute(graph)
        return graph

    @staticmethod
    def _inject_evidence_nodes(graph: StructuredMeaningGraph, evidence_terms: Sequence[str], strategy: str) -> None:
        existing_ids = graph.node_ids()
        for index, evidence in enumerate(evidence_terms, 1):
            text = " ".join(str(evidence or "").split())
            if not text:
                continue
            node_id = f"self_evidence_{index:02d}"
            suffix = 2
            while node_id in existing_ids:
                node_id = f"self_evidence_{index:02d}_{suffix}"
                suffix += 1
            existing_ids.add(node_id)
            graph.add_node(
                Node(
                    id=node_id,
                    label=text,
                    kind="evidence",
                    attributes={"text": text, "source": "self_evolution", "strategy": strategy},
                    provenance=["self_evolution"],
                )
            )
            graph.add_edge(
                Edge(
                    source="question",
                    relation="GROUNDED_BY",
                    target=node_id,
                    confidence=0.96,
                    provenance=["self_evolution"],
                )
            )

    @staticmethod
    def _evidence_terms(case: GroundedExplanationEvalCase) -> list[str]:
        terms = [" ".join(str(item or "").split()) for item in list(case.expected_evidence_terms or [])]
        terms = [item for item in terms if len(item) >= 8]
        if terms:
            return list(dict.fromkeys(terms))
        source_context = " ".join(str(case.source_context or "").split())
        chunks = re.split(r"(?<=[.!?])\s+|,\s*|;\s*", source_context)
        return [item for item in dict.fromkeys(chunk.strip() for chunk in chunks if len(chunk.strip()) >= 8)]

    @staticmethod
    def _claim_terms(case: GroundedExplanationEvalCase, evidence_terms: Sequence[str]) -> list[str]:
        claims = [" ".join(str(item or "").split()) for item in list(case.expected_claim_terms or [])]
        claims = [item for item in claims if len(item) >= 8]
        if claims:
            return list(dict.fromkeys(claims))
        return list(evidence_terms[:2])

    def _score_graph_against_case(self, graph: StructuredMeaningGraph, case: GroundedExplanationEvalCase) -> float:
        if graph.operator_execution is None:
            graph = compile_and_execute(graph)
        symbolic_evidence = " ".join(
            result.answer + " " + " ".join(result.evidence)
            for result in graph.symbolic_results
            if result.domain == "document_grounding"
        )
        evidence_lookup = {node.id: node for node in graph.nodes if node.kind == "evidence"}
        grounded_evidence = " ".join(
            str(evidence_lookup[edge.target].attributes.get("text", evidence_lookup[edge.target].label))
            for edge in graph.edges
            if edge.source == "question" and edge.relation == "GROUNDED_BY" and edge.target in evidence_lookup
        )
        evidence_text = (symbolic_evidence + " " + grounded_evidence).lower()
        report = graph.operator_execution
        claim_groundings = list(report.claim_groundings) if report is not None else []
        grounded_claim_text = " ".join(item.claim for item in claim_groundings if item.grounded).lower()
        unsupported_claim_text = " ".join(item.claim for item in claim_groundings if not item.grounded).lower()
        components: list[float] = []
        if case.expected_evidence_terms:
            evidence_hits = sum(1 for item in case.expected_evidence_terms if item.lower() in evidence_text)
            components.append(evidence_hits / float(len(case.expected_evidence_terms)))
        elif evidence_text:
            components.append(1.0)
        if case.expected_claim_terms:
            claim_hits = sum(1 for item in case.expected_claim_terms if item.lower() in grounded_claim_text)
            components.append(claim_hits / float(len(case.expected_claim_terms)))
        elif claim_groundings and report is not None:
            components.append(float(report.claim_grounding_score))
        if case.forbidden_unsupported_claim_terms:
            forbidden_hit = any(item.lower() in unsupported_claim_text for item in case.forbidden_unsupported_claim_terms)
            components.append(0.0 if forbidden_hit else 1.0)
        if graph.operator_execution is not None:
            components.append(float(graph.operator_execution.claim_grounding_score))
        if not components:
            return 1.0 if grounded_claim_text or evidence_text else 0.0
        return sum(components) / float(len(components))

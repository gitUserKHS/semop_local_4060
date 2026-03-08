from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import List, Sequence

from .corpus_store import CorpusMemoryStore
from .memory_retrieval import QueryEmbeddingIndex
from .pipeline import StructuredMeaningPipeline
from .structures import StructuredMeaningGraph


@dataclass
class QueryPriorEffect:
    query: str
    baseline_family_reuse: float
    memory_family_reuse: float
    baseline_top_confidence: float
    memory_top_confidence: float
    promoted: bool
    injected: bool
    memory_prior_operator_count: int


@dataclass
class MemoryPriorEvaluationResult:
    train_size: int
    test_size: int
    evaluated_queries: int
    baseline_family_reuse_rate: float
    memory_family_reuse_rate: float
    family_reuse_gain: float
    baseline_grammar_reuse_rate: float
    memory_grammar_reuse_rate: float
    grammar_reuse_gain: float
    baseline_avg_top_confidence: float
    memory_avg_top_confidence: float
    avg_top_confidence_gain: float
    prior_promoted_query_rate: float
    prior_injected_query_rate: float
    average_memory_prior_operators: float
    samples: List[QueryPriorEffect]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)


class MemoryPriorEvaluator:
    def __init__(self, mode: str = "heuristic", model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.mode = mode
        self.model_id = model_id

    def evaluate_store(
        self,
        store_path: str | Path,
        source: str | None = None,
        max_test_queries: int = 100,
        sample_queries: int = 10,
    ) -> MemoryPriorEvaluationResult:
        store = CorpusMemoryStore(store_path)
        train_graphs = store.fetch_graphs(split="train", source=source)
        test_queries = store.fetch_queries(split="test", source=source)
        if max_test_queries > 0:
            test_queries = test_queries[:max_test_queries]
        return self.evaluate_graphs_and_queries(train_graphs, test_queries, sample_queries=sample_queries)

    def evaluate_graphs_and_queries(
        self,
        train_graphs: Sequence[StructuredMeaningGraph],
        test_queries: Sequence[str],
        sample_queries: int = 10,
    ) -> MemoryPriorEvaluationResult:
        train_graphs = list(train_graphs)
        test_queries = list(test_queries)
        train_families = {candidate.family for graph in train_graphs for candidate in graph.induced_operators}
        train_grammar_tokens = set(train_families)

        index = QueryEmbeddingIndex()
        index.build(train_graphs)

        baseline_pipeline = StructuredMeaningPipeline(mode=self.mode, model_id=self.model_id)
        memory_pipeline = StructuredMeaningPipeline(mode=self.mode, model_id=self.model_id)

        baseline_candidate_total = 0
        baseline_candidate_hits = 0
        memory_candidate_total = 0
        memory_candidate_hits = 0
        baseline_grammar_total = 0
        baseline_grammar_hits = 0
        memory_grammar_total = 0
        memory_grammar_hits = 0
        baseline_top_confidences: List[float] = []
        memory_top_confidences: List[float] = []
        promoted_queries = 0
        injected_queries = 0
        prior_operator_counts: List[int] = []
        samples: List[QueryPriorEffect] = []

        for query in test_queries:
            similar_graphs = index.search_indexed_graphs(query, top_k=3)
            baseline_graph = baseline_pipeline.run_with_memory_graphs(query, [])
            memory_graph = memory_pipeline.run_with_memory_graphs(query, similar_graphs)

            baseline_hits, baseline_total = self._family_hits(baseline_graph, train_families)
            memory_hits, memory_total = self._family_hits(memory_graph, train_families)
            baseline_candidate_hits += baseline_hits
            baseline_candidate_total += baseline_total
            memory_candidate_hits += memory_hits
            memory_candidate_total += memory_total

            baseline_ghits, baseline_gtotal = self._grammar_hits(baseline_graph, train_grammar_tokens)
            memory_ghits, memory_gtotal = self._grammar_hits(memory_graph, train_grammar_tokens)
            baseline_grammar_hits += baseline_ghits
            baseline_grammar_total += baseline_gtotal
            memory_grammar_hits += memory_ghits
            memory_grammar_total += memory_gtotal

            baseline_top = self._top_confidence(baseline_graph)
            memory_top = self._top_confidence(memory_graph)
            baseline_top_confidences.append(baseline_top)
            memory_top_confidences.append(memory_top)

            promoted = any("memory prior promoted families" in warning for warning in memory_graph.warnings)
            injected = any("memory prior injected families" in warning for warning in memory_graph.warnings)
            if promoted:
                promoted_queries += 1
            if injected:
                injected_queries += 1
            prior_operator_count = sum(1 for candidate in memory_graph.induced_operators if any(tag.startswith("memory_prior:") for tag in candidate.provenance))
            prior_operator_counts.append(prior_operator_count)

            if len(samples) < sample_queries:
                samples.append(
                    QueryPriorEffect(
                        query=query,
                        baseline_family_reuse=round(baseline_hits / max(1, baseline_total), 2),
                        memory_family_reuse=round(memory_hits / max(1, memory_total), 2),
                        baseline_top_confidence=baseline_top,
                        memory_top_confidence=memory_top,
                        promoted=promoted,
                        injected=injected,
                        memory_prior_operator_count=prior_operator_count,
                    )
                )

        baseline_family_reuse = round(baseline_candidate_hits / max(1, baseline_candidate_total), 2)
        memory_family_reuse = round(memory_candidate_hits / max(1, memory_candidate_total), 2)
        baseline_grammar_reuse = round(baseline_grammar_hits / max(1, baseline_grammar_total), 2)
        memory_grammar_reuse = round(memory_grammar_hits / max(1, memory_grammar_total), 2)
        baseline_avg_top = round(sum(baseline_top_confidences) / max(1, len(baseline_top_confidences)), 2)
        memory_avg_top = round(sum(memory_top_confidences) / max(1, len(memory_top_confidences)), 2)

        return MemoryPriorEvaluationResult(
            train_size=len(train_graphs),
            test_size=len(test_queries),
            evaluated_queries=len(test_queries),
            baseline_family_reuse_rate=baseline_family_reuse,
            memory_family_reuse_rate=memory_family_reuse,
            family_reuse_gain=round(memory_family_reuse - baseline_family_reuse, 2),
            baseline_grammar_reuse_rate=baseline_grammar_reuse,
            memory_grammar_reuse_rate=memory_grammar_reuse,
            grammar_reuse_gain=round(memory_grammar_reuse - baseline_grammar_reuse, 2),
            baseline_avg_top_confidence=baseline_avg_top,
            memory_avg_top_confidence=memory_avg_top,
            avg_top_confidence_gain=round(memory_avg_top - baseline_avg_top, 2),
            prior_promoted_query_rate=round(promoted_queries / max(1, len(test_queries)), 2),
            prior_injected_query_rate=round(injected_queries / max(1, len(test_queries)), 2),
            average_memory_prior_operators=round(sum(prior_operator_counts) / max(1, len(prior_operator_counts)), 2),
            samples=samples,
        )

    @staticmethod
    def _family_hits(graph: StructuredMeaningGraph, train_families: set[str]) -> tuple[int, int]:
        total = len(graph.induced_operators)
        hits = sum(1 for candidate in graph.induced_operators if candidate.family in train_families)
        return hits, total

    @staticmethod
    def _grammar_hits(graph: StructuredMeaningGraph, train_families: set[str]) -> tuple[int, int]:
        total = len(graph.grammar_hypotheses)
        hits = sum(1 for rule in graph.grammar_hypotheses if any(family in rule for family in train_families))
        return hits, total

    @staticmethod
    def _top_confidence(graph: StructuredMeaningGraph) -> float:
        if not graph.induced_operators:
            return 0.0
        return round(max(candidate.confidence for candidate in graph.induced_operators), 2)

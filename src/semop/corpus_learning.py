from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from .latent_abstraction import OperatorAbstractionClustering
from .logical_grammar import LogicalGrammarInducer, LogicalPattern
from .pipeline import StructuredMeaningPipeline
from .structures import OperatorCandidate, StructuredMeaningGraph


@dataclass
class CorpusExampleSummary:
    query: str
    intent: str
    relation_count: int
    induced_operator_count: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class LearnedOperatorFamily:
    name: str
    support: int
    member_operators: List[str]
    input_types: List[str]
    output_type: str
    example_queries: List[str]
    example_phrases: List[str]
    grammar_templates: List[str]
    confidence: float
    purity: float


@dataclass
class CorpusLearningResult:
    corpus_size: int
    intent_distribution: Dict[str, int]
    relation_distribution: Dict[str, int]
    learned_families: List[LearnedOperatorFamily]
    logical_patterns: List[LogicalPattern]
    examples: List[CorpusExampleSummary]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class CorpusReasoningLearner:
    """Learn reusable operator families from many queries rather than a single prompt."""

    def __init__(self, mode: str = "heuristic", model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.pipeline = StructuredMeaningPipeline(mode=mode, model_id=model_id)
        self.clusterer = OperatorAbstractionClustering()
        self.grammar_inducer = LogicalGrammarInducer()

    def learn_from_queries(self, queries: Sequence[str]) -> CorpusLearningResult:
        graphs: List[StructuredMeaningGraph] = []
        example_summaries: List[CorpusExampleSummary] = []
        intent_distribution: Dict[str, int] = {}
        relation_distribution: Dict[str, int] = {}

        for query in queries:
            graph = self.pipeline.run(query)
            graphs.append(graph)
            intent_distribution[graph.intent] = intent_distribution.get(graph.intent, 0) + 1
            for edge in graph.edges:
                relation_distribution[edge.relation] = relation_distribution.get(edge.relation, 0) + 1
            example_summaries.append(
                CorpusExampleSummary(
                    query=query,
                    intent=graph.intent,
                    relation_count=len(graph.edges),
                    induced_operator_count=len(graph.induced_operators),
                    warnings=list(graph.warnings),
                )
            )

        learned_families = self._learn_operator_families(graphs)
        logical_patterns = self.grammar_inducer.induce(graphs).patterns
        return CorpusLearningResult(
            corpus_size=len(queries),
            intent_distribution=dict(sorted(intent_distribution.items())),
            relation_distribution=dict(sorted(relation_distribution.items())),
            learned_families=learned_families,
            logical_patterns=logical_patterns,
            examples=example_summaries,
        )

    def learn_from_jsonl(self, path: str | Path) -> CorpusLearningResult:
        queries: List[str] = []
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    item = json.loads(line)
                    query = item.get("query") or item.get("text") or item.get("input")
                    if query:
                        queries.append(query)
                else:
                    queries.append(line)
        return self.learn_from_queries(queries)

    def save_result(self, result: CorpusLearningResult, output_path: str | Path) -> None:
        Path(output_path).write_text(result.to_json(indent=2), encoding="utf-8")

    def _learn_operator_families(self, graphs: Sequence[StructuredMeaningGraph]) -> List[LearnedOperatorFamily]:
        grouped: Dict[str, List[tuple[OperatorCandidate, StructuredMeaningGraph]]] = {}
        for graph in graphs:
            for candidate in graph.induced_operators:
                grouped.setdefault(candidate.family, []).append((candidate, graph))

        families: List[LearnedOperatorFamily] = []
        for family_name, entries in grouped.items():
            phrase_to_entries: Dict[str, List[tuple[OperatorCandidate, StructuredMeaningGraph]]] = {}
            for candidate, graph in entries:
                phrase = self._candidate_phrase(candidate)
                phrase_to_entries.setdefault(phrase, []).append((candidate, graph))

            clusters = self.clusterer.cluster(phrase_to_entries.keys(), threshold=0.42)
            for cluster in clusters:
                cluster_entries: List[tuple[OperatorCandidate, StructuredMeaningGraph]] = []
                for phrase in cluster.members:
                    cluster_entries.extend(phrase_to_entries.get(phrase, []))
                family = self._build_family(family_name, cluster_entries)
                if family.support > 0:
                    families.append(family)

        families.sort(key=lambda item: (-item.support, -item.confidence, item.name))
        return families

    @staticmethod
    def _candidate_phrase(candidate: OperatorCandidate) -> str:
        example = candidate.examples[0] if candidate.examples else candidate.description
        return f"{candidate.family} {candidate.output_type} {example}"

    def _build_family(
        self,
        family_name: str,
        entries: Sequence[tuple[OperatorCandidate, StructuredMeaningGraph]],
    ) -> LearnedOperatorFamily:
        operators = [candidate for candidate, _ in entries]
        graphs = [graph for _, graph in entries]
        member_names = list(dict.fromkeys(candidate.name for candidate in operators))
        input_types = self._merge_types(operators)
        output_type = self._majority([candidate.output_type for candidate in operators])
        example_queries = list(dict.fromkeys(graph.query for graph in graphs))[:6]
        example_phrases = list(dict.fromkeys(example for candidate in operators for example in candidate.examples))[:6]
        grammar_templates = list(
            dict.fromkeys(
                rule
                for graph in graphs
                for rule in graph.grammar_hypotheses
                if family_name in rule or any(candidate.name in rule for candidate in operators)
            )
        )[:6]
        support = len(example_queries)
        confidence = round(min(0.99, (sum(candidate.confidence for candidate in operators) / max(1, len(operators))) + 0.03 * min(6, support)), 2)
        purity = self._purity(operators)
        name = self._family_label(family_name, operators)
        return LearnedOperatorFamily(
            name=name,
            support=support,
            member_operators=member_names,
            input_types=input_types,
            output_type=output_type,
            example_queries=example_queries,
            example_phrases=example_phrases,
            grammar_templates=grammar_templates,
            confidence=confidence,
            purity=purity,
        )

    @staticmethod
    def _merge_types(operators: Sequence[OperatorCandidate]) -> List[str]:
        merged: List[str] = []
        for candidate in operators:
            for input_type in candidate.input_types:
                if input_type not in merged:
                    merged.append(input_type)
        return merged[:5]

    @staticmethod
    def _majority(values: Sequence[str]) -> str:
        counts: Dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return max(counts, key=counts.get) if counts else "meaning_state"

    @staticmethod
    def _family_label(family_name: str, operators: Sequence[OperatorCandidate]) -> str:
        top_operator = max(operators, key=lambda candidate: candidate.confidence)
        return f"FAMILY_{family_name}_{top_operator.output_type}"[:72]

    @staticmethod
    def _purity(operators: Sequence[OperatorCandidate]) -> float:
        if not operators:
            return 0.0
        output_counts: Dict[str, int] = {}
        signature_counts: Dict[str, int] = {}
        for candidate in operators:
            output_counts[candidate.output_type] = output_counts.get(candidate.output_type, 0) + 1
            signature = tuple(candidate.input_types)
            signature_counts[str(signature)] = signature_counts.get(str(signature), 0) + 1
        dominant_output = max(output_counts.values())
        dominant_signature = max(signature_counts.values())
        return round(((dominant_output / len(operators)) + (dominant_signature / len(operators))) / 2, 2)
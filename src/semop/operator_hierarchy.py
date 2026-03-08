from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from .latent_abstraction import OperatorAbstractionClustering
from .pipeline import StructuredMeaningPipeline
from .structures import OperatorCandidate, StructuredMeaningGraph


@dataclass
class OperatorHierarchyNode:
    level: str
    name: str
    support: int
    purity: float
    confidence: float
    parent_names: List[str] = field(default_factory=list)
    child_names: List[str] = field(default_factory=list)
    member_operators: List[str] = field(default_factory=list)
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    example_queries: List[str] = field(default_factory=list)
    composition_signatures: List[str] = field(default_factory=list)


@dataclass
class OperatorCompositionPattern:
    pattern: str
    support: int
    example_queries: List[str] = field(default_factory=list)


@dataclass
class OperatorHierarchyResult:
    corpus_size: int
    micro_nodes: List[OperatorHierarchyNode]
    family_nodes: List[OperatorHierarchyNode]
    abstract_nodes: List[OperatorHierarchyNode]
    composition_patterns: List[OperatorCompositionPattern]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class OperatorHierarchyLearner:
    def __init__(self, mode: str = "heuristic", model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.pipeline = StructuredMeaningPipeline(mode=mode, model_id=model_id)
        self.clusterer = OperatorAbstractionClustering()

    def learn_from_queries(self, queries: Sequence[str]) -> OperatorHierarchyResult:
        graphs = [self.pipeline.run(query) for query in queries]
        return self.learn_from_graphs(graphs)

    def learn_from_graphs(self, graphs: Sequence[StructuredMeaningGraph]) -> OperatorHierarchyResult:
        graphs = list(graphs)
        micro_nodes = self._build_micro_nodes(graphs)
        family_nodes = self._build_family_nodes(graphs, micro_nodes)
        abstract_nodes = self._build_abstract_nodes(family_nodes)
        composition_patterns = self._build_composition_patterns(graphs)
        return OperatorHierarchyResult(
            corpus_size=len(graphs),
            micro_nodes=micro_nodes,
            family_nodes=family_nodes,
            abstract_nodes=abstract_nodes,
            composition_patterns=composition_patterns,
        )

    def save_result(self, result: OperatorHierarchyResult, output_path: str | Path) -> None:
        Path(output_path).write_text(result.to_json(indent=2), encoding="utf-8")

    def _build_micro_nodes(self, graphs: Sequence[StructuredMeaningGraph]) -> List[OperatorHierarchyNode]:
        grouped: Dict[str, List[tuple[OperatorCandidate, StructuredMeaningGraph]]] = {}
        for graph in graphs:
            for candidate in graph.induced_operators:
                key = f"{candidate.name}|{candidate.family}|{','.join(candidate.input_types)}|{candidate.output_type}"
                grouped.setdefault(key, []).append((candidate, graph))

        nodes: List[OperatorHierarchyNode] = []
        for key, entries in grouped.items():
            candidates = [candidate for candidate, _ in entries]
            example_queries = list(dict.fromkeys(graph.query for _, graph in entries))[:6]
            member_names = list(dict.fromkeys(candidate.name for candidate in candidates))
            families = list(dict.fromkeys(candidate.family for candidate in candidates))
            input_types = self._merge_types(candidate.input_types for candidate in candidates)
            output_types = list(dict.fromkeys(candidate.output_type for candidate in candidates))
            composition_signatures = list(dict.fromkeys(self._candidate_signature(candidate) for candidate in candidates))[:6]
            purity = self._purity(candidates)
            confidence = round(sum(candidate.confidence for candidate in candidates) / max(1, len(candidates)), 2)
            node_name = f"L1_{families[0]}_{member_names[0]}"[:96]
            nodes.append(
                OperatorHierarchyNode(
                    level="L1",
                    name=node_name,
                    support=len(example_queries),
                    purity=purity,
                    confidence=confidence,
                    parent_names=families,
                    member_operators=member_names,
                    input_types=input_types,
                    output_types=output_types,
                    example_queries=example_queries,
                    composition_signatures=composition_signatures,
                )
            )
        nodes.sort(key=lambda item: (-item.support, -item.confidence, item.name))
        return nodes

    def _build_family_nodes(self, graphs: Sequence[StructuredMeaningGraph], micro_nodes: Sequence[OperatorHierarchyNode]) -> List[OperatorHierarchyNode]:
        grouped: Dict[str, List[tuple[OperatorCandidate, StructuredMeaningGraph]]] = {}
        micro_children: Dict[str, List[str]] = {}
        for node in micro_nodes:
            for parent in node.parent_names:
                micro_children.setdefault(parent, []).append(node.name)
        for graph in graphs:
            for candidate in graph.induced_operators:
                grouped.setdefault(candidate.family, []).append((candidate, graph))

        nodes: List[OperatorHierarchyNode] = []
        for family, entries in grouped.items():
            candidates = [candidate for candidate, _ in entries]
            example_queries = list(dict.fromkeys(graph.query for _, graph in entries))[:8]
            output_types = list(dict.fromkeys(candidate.output_type for candidate in candidates))
            input_types = self._merge_types(candidate.input_types for candidate in candidates)
            composition_signatures = list(dict.fromkeys(self._candidate_signature(candidate) for candidate in candidates))[:8]
            purity = self._purity(candidates)
            confidence = round(min(0.99, (sum(candidate.confidence for candidate in candidates) / max(1, len(candidates))) + 0.02 * min(5, len(example_queries))), 2)
            nodes.append(
                OperatorHierarchyNode(
                    level="L2",
                    name=f"L2_{family}"[:96],
                    support=len(example_queries),
                    purity=purity,
                    confidence=confidence,
                    child_names=sorted(set(micro_children.get(family, []))),
                    member_operators=list(dict.fromkeys(candidate.name for candidate in candidates)),
                    input_types=input_types,
                    output_types=output_types,
                    example_queries=example_queries,
                    composition_signatures=composition_signatures,
                )
            )
        nodes.sort(key=lambda item: (-item.support, -item.confidence, item.name))
        return nodes

    def _build_abstract_nodes(self, family_nodes: Sequence[OperatorHierarchyNode]) -> List[OperatorHierarchyNode]:
        if not family_nodes:
            return []
        phrase_to_nodes: Dict[str, List[OperatorHierarchyNode]] = {}
        for node in family_nodes:
            phrase = " ".join([node.name, *node.input_types, *node.output_types, *node.composition_signatures[:2]])
            phrase_to_nodes.setdefault(phrase, []).append(node)

        clusters = self.clusterer.cluster(phrase_to_nodes.keys(), threshold=0.38)
        abstract_nodes: List[OperatorHierarchyNode] = []
        for cluster in clusters:
            members: List[OperatorHierarchyNode] = []
            for phrase in cluster.members:
                members.extend(phrase_to_nodes.get(phrase, []))
            if not members:
                continue
            family_names = [node.name for node in members]
            family_labels = [name.removeprefix("L2_") for name in family_names]
            label = self._abstract_label(family_labels)
            support = len(set(query for node in members for query in node.example_queries))
            purity = round(sum(node.purity for node in members) / max(1, len(members)), 2)
            confidence = round(sum(node.confidence for node in members) / max(1, len(members)), 2)
            abstract_nodes.append(
                OperatorHierarchyNode(
                    level="L3",
                    name=f"L3_{label}"[:96],
                    support=support,
                    purity=purity,
                    confidence=confidence,
                    child_names=sorted(set(family_names)),
                    member_operators=sorted(set(operator for node in members for operator in node.member_operators)),
                    input_types=self._merge_types(node.input_types for node in members),
                    output_types=self._merge_types(node.output_types for node in members),
                    example_queries=list(dict.fromkeys(query for node in members for query in node.example_queries))[:8],
                    composition_signatures=list(dict.fromkeys(sig for node in members for sig in node.composition_signatures))[:8],
                )
            )
        abstract_nodes.sort(key=lambda item: (-item.support, -item.confidence, item.name))
        return abstract_nodes

    def _build_composition_patterns(self, graphs: Sequence[StructuredMeaningGraph]) -> List[OperatorCompositionPattern]:
        pattern_examples: Dict[str, List[str]] = {}
        for graph in graphs:
            sequence = list(dict.fromkeys(candidate.family for candidate in sorted(graph.induced_operators, key=lambda item: (-item.confidence, item.name))))
            if len(sequence) < 2:
                continue
            for width in [2, 3]:
                if len(sequence) < width:
                    continue
                for start in range(0, len(sequence) - width + 1):
                    pattern = " -> ".join(sequence[start:start + width])
                    pattern_examples.setdefault(pattern, []).append(graph.query)
        patterns = [
            OperatorCompositionPattern(
                pattern=pattern,
                support=len(set(queries)),
                example_queries=list(dict.fromkeys(queries))[:6],
            )
            for pattern, queries in pattern_examples.items()
        ]
        patterns.sort(key=lambda item: (-item.support, item.pattern))
        return patterns[:24]

    @staticmethod
    def _candidate_signature(candidate: OperatorCandidate) -> str:
        left = ",".join(candidate.input_types) if candidate.input_types else "meaning_unit"
        return f"{left}->{candidate.output_type}"

    @staticmethod
    def _merge_types(type_groups: Sequence[Sequence[str]]) -> List[str]:
        merged: List[str] = []
        for group in type_groups:
            for item in group:
                if item not in merged:
                    merged.append(item)
        return merged[:8]

    @staticmethod
    def _purity(candidates: Sequence[OperatorCandidate]) -> float:
        if not candidates:
            return 0.0
        output_counts: Dict[str, int] = {}
        signature_counts: Dict[str, int] = {}
        for candidate in candidates:
            output_counts[candidate.output_type] = output_counts.get(candidate.output_type, 0) + 1
            signature = ",".join(candidate.input_types)
            signature_counts[signature] = signature_counts.get(signature, 0) + 1
        dominant_output = max(output_counts.values())
        dominant_signature = max(signature_counts.values())
        return round(((dominant_output / len(candidates)) + (dominant_signature / len(candidates))) / 2, 2)

    @staticmethod
    def _abstract_label(family_labels: Sequence[str]) -> str:
        tokens: Dict[str, int] = {}
        for label in family_labels:
            for token in re.findall(r"[A-Z_]+", label.upper()):
                tokens[token] = tokens.get(token, 0) + 1
        ordered = sorted(tokens.items(), key=lambda item: (-item[1], item[0]))
        if ordered:
            return "_".join(token for token, _ in ordered[:2])
        return "ABSTRACT_OPERATOR"


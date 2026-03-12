from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .operator_evolution import OperatorSelfEvolutionEngine
from .operator_proposal import ModelProposedOperator, OperatorProposalEngine, OperatorProposalSummarizer
from .structures import StructuredMeaningGraph


@dataclass
class VisualSignalImpactSummary:
    baseline_retained_count: int
    ablated_retained_count: int
    baseline_proposal_count: int
    ablated_proposal_count: int
    retained_name_overlap: float
    lost_retained_names: list[str]
    gained_retained_names: list[str]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorProposalComparisonSummary:
    primary_label: str
    secondary_label: str
    primary_count: int
    secondary_count: int
    overlap_count: int
    primary_only: list[str]
    secondary_only: list[str]
    overlap_names: list[str]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)




@dataclass
class HybridOperatorProposalSummary:
    total_hybrid_count: int
    adopted_llm_count: int
    retained_names: list[str]
    llm_named_signatures: list[list[str]]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class HybridOperatorProposalPolicy:
    def build(
        self,
        graphs: Sequence[StructuredMeaningGraph],
        llm_model_id: str,
    ) -> HybridOperatorProposalSummary:
        heuristic_engine = OperatorProposalEngine()
        llm_engine = OperatorProposalEngine(OperatorProposalSummarizer(model_id=llm_model_id))
        heuristic = {tuple(item.basis_signature): item for item in heuristic_engine.propose(graphs)}
        llm = {tuple(item.basis_signature): item for item in llm_engine.propose(graphs)}
        retained_names: list[str] = []
        llm_named_signatures: list[list[str]] = []
        adopted_llm_count = 0
        for signature, item in heuristic.items():
            llm_item = llm.get(signature)
            if llm_item and llm_item.proposal_source == 'llm_operator_summarizer' and llm_item.normalized_name:
                retained_names.append(llm_item.normalized_name)
                llm_named_signatures.append(list(signature))
                adopted_llm_count += 1
            else:
                retained_names.append(item.normalized_name)
        return HybridOperatorProposalSummary(
            total_hybrid_count=len(retained_names),
            adopted_llm_count=adopted_llm_count,
            retained_names=sorted(retained_names),
            llm_named_signatures=sorted(llm_named_signatures),
        )


class VisualSignalImpactEvaluator:
    def __init__(self, engine: OperatorSelfEvolutionEngine | None = None) -> None:
        self.engine = engine or OperatorSelfEvolutionEngine()

    def evaluate(self, graphs: Sequence[StructuredMeaningGraph], min_support: int = 1, utility_threshold: float = 0.2) -> VisualSignalImpactSummary:
        baseline = self.engine.evolve(graphs, min_support=min_support, utility_threshold=utility_threshold)
        ablated_graphs = [self._ablate_visual_signals(graph) for graph in graphs]
        ablated = self.engine.evolve(ablated_graphs, min_support=min_support, utility_threshold=utility_threshold)
        baseline_names = {item.name for item in baseline.proposals if item.retained}
        ablated_names = {item.name for item in ablated.proposals if item.retained}
        overlap = baseline_names & ablated_names
        denom = len(baseline_names | ablated_names) or 1
        return VisualSignalImpactSummary(
            baseline_retained_count=len(baseline_names),
            ablated_retained_count=len(ablated_names),
            baseline_proposal_count=len(baseline.proposals),
            ablated_proposal_count=len(ablated.proposals),
            retained_name_overlap=round(len(overlap) / float(denom), 4),
            lost_retained_names=sorted(baseline_names - ablated_names),
            gained_retained_names=sorted(ablated_names - baseline_names),
        )

    @staticmethod
    def _ablate_visual_signals(graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        clone = copy.deepcopy(graph)
        for node in clone.nodes:
            attrs = node.attributes if isinstance(node.attributes, dict) else {}
            signatures = attrs.get('geometry_signature')
            if isinstance(signatures, list):
                attrs['geometry_signature'] = [
                    item for item in signatures
                    if item not in {'SYMMETRIC_STRUCTURE', 'AXIS_ALIGNED_STRUCTURE', 'CLOSED_BOUNDARY_STRUCTURE'}
                ]
            for key in ('symmetry_score', 'axis_alignment_score', 'closure_score'):
                attrs.pop(key, None)
            node.attributes = attrs
        return clone


class OperatorProposalComparator:
    def compare(self, graphs: Sequence[StructuredMeaningGraph], llm_model_id: str) -> OperatorProposalComparisonSummary:
        heuristic = OperatorProposalEngine()
        llm = OperatorProposalEngine(OperatorProposalSummarizer(model_id=llm_model_id))
        return self.compare_engines(graphs, heuristic, llm, 'heuristic', 'llm')

    def compare_with_hybrid(self, graphs: Sequence[StructuredMeaningGraph], llm_model_id: str) -> dict[str, Any]:
        comparison = self.compare(graphs, llm_model_id=llm_model_id)
        hybrid = HybridOperatorProposalPolicy().build(graphs, llm_model_id=llm_model_id)
        return {
            'comparison': comparison.model_dump(),
            'hybrid': hybrid.model_dump(),
        }

    @staticmethod
    def compare_engines(
        graphs: Sequence[StructuredMeaningGraph],
        primary_engine: OperatorProposalEngine,
        secondary_engine: OperatorProposalEngine,
        primary_label: str = 'primary',
        secondary_label: str = 'secondary',
    ) -> OperatorProposalComparisonSummary:
        primary = primary_engine.propose(graphs)
        secondary = secondary_engine.propose(graphs)
        primary_names = {item.normalized_name for item in primary}
        secondary_names = {item.normalized_name for item in secondary}
        overlap = primary_names & secondary_names
        return OperatorProposalComparisonSummary(
            primary_label=primary_label,
            secondary_label=secondary_label,
            primary_count=len(primary_names),
            secondary_count=len(secondary_names),
            overlap_count=len(overlap),
            primary_only=sorted(primary_names - secondary_names),
            secondary_only=sorted(secondary_names - primary_names),
            overlap_names=sorted(overlap),
        )

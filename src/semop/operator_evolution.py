
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .structures import FunctorHypothesis, OperatorDecomposition, StructuredMeaningGraph
from .operator_proposal import OperatorProposalEngine
from .unified_world_solver_guidance import UnifiedWorldSolverGuidance, UnifiedWorldSolverGuidanceEngine


@dataclass
class EvolvedOperatorProposal:
    name: str
    basis_signature: list[str]
    basis_operators: list[str]
    source_operator_names: list[str]
    source_domains: list[str]
    support: int
    domain_support: int
    confidence: float
    utility_score: float = 0.0
    retained: bool = False
    rationale: str = ''

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorEvolutionSummary:
    num_graphs: int
    proposals: list[EvolvedOperatorProposal]
    retained_count: int
    pruned_count: int
    merged_count: int

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorEvolutionRunResult:
    iteration: int
    source: str
    split: str
    summary: OperatorEvolutionSummary
    transfer_summary: "OperatorTransferEvalSummary | None" = None
    stored_run_id: int | None = None
    retained_operator_names: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class OperatorSelfEvolutionEngine:
    def __init__(self, proposal_engine: OperatorProposalEngine | None = None) -> None:
        self.proposal_engine = proposal_engine or OperatorProposalEngine()

    def evolve(
        self,
        graphs: Sequence[StructuredMeaningGraph],
        min_support: int = 2,
        utility_threshold: float = 0.45,
        self_learning_plan: dict[str, Any] | None = None,
        integrated_reasoning: dict[str, Any] | None = None,
    ) -> OperatorEvolutionSummary:
        proposals = self._propose(graphs)
        merged = self._merge(proposals)
        guidance = UnifiedWorldSolverGuidanceEngine().build(
            plan=self_learning_plan,
            reasoning=integrated_reasoning,
        )
        self._score_utility(merged, graphs, guidance=guidance)
        retained = 0
        pruned = 0
        for item in merged:
            item.retained = item.support >= min_support and item.utility_score >= utility_threshold
            if item.retained:
                retained += 1
            else:
                pruned += 1
        return OperatorEvolutionSummary(
            num_graphs=len(graphs),
            proposals=sorted(merged, key=lambda row: (-row.utility_score, -row.support, row.name)),
            retained_count=retained,
            pruned_count=pruned,
            merged_count=max(0, len(proposals) - len(merged)),
        )

    def _propose(self, graphs: Sequence[StructuredMeaningGraph]) -> list[EvolvedOperatorProposal]:
        proposals: list[EvolvedOperatorProposal] = []
        for proposal in self.proposal_engine.propose(graphs):
            proposals.append(EvolvedOperatorProposal(
                name=proposal.normalized_name,
                basis_signature=list(proposal.basis_signature),
                basis_operators=list(proposal.basis_operators),
                source_operator_names=list(proposal.source_operator_names),
                source_domains=list(proposal.source_domains),
                support=int(proposal.support),
                domain_support=len(set(proposal.source_domains)),
                confidence=float(proposal.confidence),
                rationale=proposal.rationale or 'Repeated decomposition pattern discovered from graph traces.',
            ))
        return proposals

    def _merge(self, proposals: Sequence[EvolvedOperatorProposal]) -> list[EvolvedOperatorProposal]:
        grouped: dict[tuple[str, ...], EvolvedOperatorProposal] = {}
        for proposal in proposals:
            key = tuple(proposal.basis_signature)
            current = grouped.get(key)
            if current is None:
                grouped[key] = EvolvedOperatorProposal(**proposal.model_dump())
                continue
            current.support += proposal.support
            current.source_operator_names = sorted(set(current.source_operator_names + proposal.source_operator_names))
            current.source_domains = sorted(set(current.source_domains + proposal.source_domains))
            current.domain_support = len(current.source_domains)
            current.confidence = round(max(current.confidence, proposal.confidence), 4)
            if len(proposal.name) < len(current.name):
                current.name = proposal.name
            if proposal.rationale and proposal.rationale not in current.rationale:
                current.rationale = (current.rationale + ' ' + proposal.rationale).strip()
        return list(grouped.values())

    def _score_utility(
        self,
        proposals: Sequence[EvolvedOperatorProposal],
        graphs: Sequence[StructuredMeaningGraph],
        guidance: UnifiedWorldSolverGuidance | None = None,
    ) -> None:
        guidance_engine = UnifiedWorldSolverGuidanceEngine()
        max_support = max((item.support for item in proposals), default=1)
        graph_domains = {graph.domain or 'general' for graph in graphs} or {'general'}
        max_domain_support = max(1, len(graph_domains))
        for item in proposals:
            support_score = item.support / float(max_support)
            transfer_score = item.domain_support / float(max_domain_support)
            compression_score = min(1.0, len(item.source_operator_names) / max(1.0, len(item.basis_signature)))
            confidence_score = min(1.0, max(0.0, float(item.confidence)))
            guidance_score = guidance_engine.operator_priority(
                guidance,
                name=item.name,
                basis_signature=item.basis_signature,
            )
            base_score = (0.3 * support_score) + (0.3 * transfer_score) + (0.15 * compression_score) + (0.15 * confidence_score)
            item.utility_score = round(min(1.0, base_score + (0.1 * guidance_score)), 4)
            if guidance_score > 0.0 and guidance is not None and guidance.summary and guidance.summary not in item.rationale:
                item.rationale = (item.rationale + ' Prioritized by shared-world solver guidance.').strip()
            item.domain_support = len(item.source_domains)

    @staticmethod
    def _proposal_name(decomp: OperatorDecomposition) -> str:
        basis = [item for item in decomp.basis_operators if item]
        if not basis:
            return decomp.operator_name
        if decomp.operator_name and decomp.operator_name not in {'GOAL_PRESERVATION_OPERATOR', 'SERVICE_GOAL_OPERATOR'}:
            return decomp.operator_name
        return 'EVOLVED_' + '_'.join(item.upper() for item in basis[:3])


@dataclass
class OperatorTransferEvalCase:
    query: str
    domain: str
    expected_operator_names: list[str] = field(default_factory=list)
    split: str = 'train'


@dataclass
class OperatorTransferEvalSummary:
    num_train: int
    num_test: int
    retained_operator_count: int
    transfer_recall: float
    domain_transfer_rate: float
    num_unseen_test_cases: int
    unseen_domain_transfer_rate: float
    retained_operator_names: list[str]
    results: list[dict[str, Any]]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class OperatorTransferEvaluator:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.evolution = OperatorSelfEvolutionEngine()

    def evaluate(self, cases: Sequence[OperatorTransferEvalCase]) -> OperatorTransferEvalSummary:
        train_cases = [case for case in cases if case.split == 'train']
        test_cases = [case for case in cases if case.split != 'train']
        train_graphs = [self.pipeline.run(case.query) for case in train_cases]
        test_graphs = [self.pipeline.run(case.query) for case in test_cases]
        evolution = self.evolution.evolve(train_graphs)
        retained = [item for item in evolution.proposals if item.retained]
        retained_names = {item.name for item in retained}
        retained_basis = {item.name: set(item.basis_signature) for item in retained}
        results: list[dict[str, Any]] = []
        recalls: list[float] = []
        train_domains = {case.domain for case in train_cases}
        cross_domain_hits = 0
        unseen_hits = 0
        unseen_test_cases = 0
        for case, graph in zip(test_cases, test_graphs):
            predicted_names = {item.operator_name for item in graph.operator_decompositions}
            graph_basis = {item.operator_name: set(item.basis_operators) for item in graph.operator_decompositions}
            gold = set(case.expected_operator_names)
            matched = set()
            for gold_name in gold:
                if gold_name in retained_names or gold_name in predicted_names:
                    matched.add(gold_name)
                    continue
                gold_basis = graph_basis.get(gold_name, set())
                if gold_basis and any(gold_basis == basis for basis in retained_basis.values()):
                    matched.add(gold_name)
            recall = 1.0 if not gold else len(matched) / float(len(gold))
            recalls.append(recall)
            if matched:
                cross_domain_hits += 1
            unseen = case.domain not in train_domains
            if unseen:
                unseen_test_cases += 1
                if matched:
                    unseen_hits += 1
            results.append({
                'query': case.query,
                'domain': case.domain,
                'expected_operator_names': sorted(gold),
                'matched_operator_names': sorted(matched),
                'transfer_recall': round(recall, 4),
                'unseen_domain': unseen,
            })
        num_test = len(test_cases)
        return OperatorTransferEvalSummary(
            num_train=len(train_cases),
            num_test=num_test,
            retained_operator_count=len(retained),
            transfer_recall=round(sum(recalls) / float(num_test or 1), 4),
            domain_transfer_rate=round(cross_domain_hits / float(num_test or 1), 4),
            num_unseen_test_cases=unseen_test_cases,
            unseen_domain_transfer_rate=round(unseen_hits / float(unseen_test_cases or 1), 4),
            retained_operator_names=sorted(item.name for item in retained),
            results=results,
        )

class OperatorSelfEvolutionLoop:
    def __init__(self, pipeline, store) -> None:
        self.pipeline = pipeline
        self.store = store
        self.engine = OperatorSelfEvolutionEngine()
        self.transfer_evaluator = OperatorTransferEvaluator(pipeline)

    def run(
        self,
        queries: Sequence[str],
        source: str = "operator_self_evolution",
        split: str = "train",
        iterations: int = 1,
        min_support: int = 2,
        utility_threshold: float = 0.45,
        transfer_cases: Sequence[OperatorTransferEvalCase] | None = None,
        self_learning_plan: dict[str, Any] | None = None,
        integrated_reasoning: dict[str, Any] | None = None,
    ) -> list[OperatorEvolutionRunResult]:
        results: list[OperatorEvolutionRunResult] = []
        for iteration in range(1, max(1, iterations) + 1):
            graphs = [self.pipeline.run(query) for query in queries]
            summary = self.engine.evolve(
                graphs,
                min_support=min_support,
                utility_threshold=utility_threshold,
                self_learning_plan=self_learning_plan,
                integrated_reasoning=integrated_reasoning,
            )
            retained = [item for item in summary.proposals if item.retained]
            run_id = self.store.store_operator_evolution_result(
                summary,
                source=source,
                split=split,
                iteration=iteration,
            )
            self.store.seed_evolved_operator_memory(
                retained,
                source=source,
                split=split,
            )
            transfer_summary = None
            if transfer_cases:
                transfer_summary = self.transfer_evaluator.evaluate(transfer_cases)
                self.store.store_operator_transfer_summary(
                    transfer_summary,
                    source=source,
                    split=split,
                    iteration=iteration,
                    evolution_run_id=run_id,
                )
            results.append(
                OperatorEvolutionRunResult(
                    iteration=iteration,
                    source=source,
                    split=split,
                    summary=summary,
                    transfer_summary=transfer_summary,
                    stored_run_id=run_id,
                    retained_operator_names=sorted(item.name for item in retained),
                )
            )
        return results


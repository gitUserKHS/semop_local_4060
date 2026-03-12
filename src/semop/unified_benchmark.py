from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .analogy_policy import AnalogyPolicyTrainer
from .continuous_learning import ContinuousLearningBundleBuilder
from .corpus_store import CorpusMemoryStore
from .graph_supervision import GraphSupervisionExporter
from .operator_evolution import OperatorTransferEvalCase, OperatorTransferEvaluator
from .multimodal_alignment_memory import MultimodalAlignmentTrainer
from .operator_repair_policy import OperatorRepairPolicyTrainer
from .retained_repair_programs import RetainedRepairProgramTrainer
from .operator_runtime import compile_and_execute
from .premise_eval import HiddenPremiseEvalCase, HiddenPremiseEvaluator
from .pipeline import StructuredMeaningPipeline
from .retained_operator_algebra import RetainedOperatorTrainer
from .structures import StructuredMeaningGraph
from .unified_parser import UnifiedParserTrainer
from .vlso.eval import VlsoEvalCase, VlsoGroundedEvaluator
from .vlso.reasoner import VLSOReasoner


@dataclass
class UnifiedSemOpArtifacts:
    analogy_policy_path: str = ''
    unified_parser_path: str = ''
    graph_supervision_path: str = ''
    retained_algebra_path: str = ''
    repair_policy_path: str = ''
    retained_repair_program_path: str = ''
    multimodal_alignment_path: str = ''
    continuous_learning_bundle_dir: str = ''

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedSemOpTrainingSummary:
    artifacts: UnifiedSemOpArtifacts
    trained_on_graphs: int
    analogy_policy: dict[str, Any]
    unified_parser: dict[str, Any]
    graph_supervision: dict[str, Any]
    retained_algebra: dict[str, Any]
    repair_policy: dict[str, Any]
    retained_repair_programs: dict[str, Any]
    multimodal_alignment: dict[str, Any]
    continuous_learning_bundle: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalogyEvalCase:
    query: str
    similar_graphs: list[StructuredMeaningGraph]
    expected_requirement: str = ''


@dataclass
class GroundedExplanationEvalCase:
    query: str
    source_context: str
    expected_evidence_terms: list[str] = field(default_factory=list)
    visual_input: Any | None = None


@dataclass
class CompilerRepairEvalCase:
    graph: StructuredMeaningGraph
    expected_repair_terms: list[str] = field(default_factory=list)


@dataclass
class UnifiedBenchmarkSummary:
    trained_pipeline: dict[str, Any]
    hidden_premise: dict[str, Any] | None = None
    transfer: dict[str, Any] | None = None
    vlso_grounded: dict[str, Any] | None = None
    unseen_transfer: float = 0.0
    analogy_usefulness: float = 0.0
    compiler_validity: float = 0.0
    grounded_explanation_fidelity: float = 0.0
    repair_success_rate: float = 0.0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class UnifiedSemOpTrainer:
    def train_from_store(
        self,
        store_path: str | Path,
        output_dir: str | Path,
        source: str | None = None,
        split: str = 'train',
    ) -> UnifiedSemOpTrainingSummary:
        store = CorpusMemoryStore(store_path)
        graphs = store.fetch_graphs(split=split, source=source)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        analogy_summary = AnalogyPolicyTrainer().train_from_memory(
            store,
            output_dir / 'analogy_policy.json',
            source=source,
            split=split,
        )
        unified_summary = UnifiedParserTrainer().train_from_graphs(
            graphs,
            output_dir / 'unified_parser.json',
        )
        graph_supervision_summary = GraphSupervisionExporter().export_from_graphs(
            graphs,
            output_dir / 'graph_supervision.jsonl',
        )
        retained_summary = RetainedOperatorTrainer().train_from_graphs(
            graphs,
            output_dir / 'retained_operator_algebra.json',
        )
        multimodal_alignment_summary = MultimodalAlignmentTrainer().train_from_graphs(
            graphs,
            output_dir / 'multimodal_alignment.json',
        )
        repair_policy_summary = OperatorRepairPolicyTrainer().train_from_graphs(
            graphs,
            output_dir / 'operator_repair_policy.json',
        )
        retained_repair_summary = RetainedRepairProgramTrainer().train_from_graphs(
            graphs,
            output_dir / 'retained_repair_programs.json',
        )
        continuous_learning_summary = ContinuousLearningBundleBuilder().build_from_graphs(
            graphs,
            output_dir / 'continuous_learning_bundle',
        )
        artifacts = UnifiedSemOpArtifacts(
            analogy_policy_path=str(output_dir / 'analogy_policy.json'),
            unified_parser_path=str(output_dir / 'unified_parser.json'),
            graph_supervision_path=str(output_dir / 'graph_supervision.jsonl'),
            retained_algebra_path=str(output_dir / 'retained_operator_algebra.json'),
            repair_policy_path=str(output_dir / 'operator_repair_policy.json'),
            retained_repair_program_path=str(output_dir / 'retained_repair_programs.json'),
            multimodal_alignment_path=str(output_dir / 'multimodal_alignment.json'),
            continuous_learning_bundle_dir=str(output_dir / 'continuous_learning_bundle'),
        )
        return UnifiedSemOpTrainingSummary(
            artifacts=artifacts,
            trained_on_graphs=len(graphs),
            analogy_policy=analogy_summary.model_dump(),
            unified_parser=unified_summary.model_dump(),
            graph_supervision=graph_supervision_summary.model_dump(),
            retained_algebra=retained_summary.model_dump(),
            repair_policy=repair_policy_summary.model_dump(),
            retained_repair_programs=retained_repair_summary.model_dump(),
            multimodal_alignment=multimodal_alignment_summary.model_dump(),
            continuous_learning_bundle=continuous_learning_summary.model_dump(),
        )


class UnifiedBenchmarkHarness:
    def __init__(self, mode: str = 'heuristic') -> None:
        self.mode = mode

    def evaluate(
        self,
        artifacts: UnifiedSemOpArtifacts,
        hidden_premise_cases: Sequence[HiddenPremiseEvalCase] | None = None,
        transfer_cases: Sequence[OperatorTransferEvalCase] | None = None,
        analogy_cases: Sequence[AnalogyEvalCase] | None = None,
        grounding_cases: Sequence[GroundedExplanationEvalCase] | None = None,
        compiler_cases: Sequence[CompilerRepairEvalCase] | None = None,
        vlso_cases: Sequence[VlsoEvalCase] | None = None,
    ) -> UnifiedBenchmarkSummary:
        pipeline = self._pipeline(artifacts)
        hidden_summary = HiddenPremiseEvaluator(pipeline).evaluate(hidden_premise_cases).__dict__ if hidden_premise_cases else None
        transfer_summary = OperatorTransferEvaluator(pipeline).evaluate(transfer_cases).model_dump() if transfer_cases else None
        vlso_summary = VlsoGroundedEvaluator(VLSOReasoner()).evaluate_cases(list(vlso_cases)).model_dump() if vlso_cases else None
        analogy_usefulness = self._evaluate_analogy_usefulness(pipeline, analogy_cases or [])
        grounded_fidelity = self._evaluate_grounded_explanations(pipeline, grounding_cases or [])
        compiler_validity = self._evaluate_compiler_validity(compiler_cases or [], hidden_summary)
        repair_success = self._evaluate_repairs(pipeline, compiler_cases or [])
        unseen_transfer = float(transfer_summary.get('unseen_domain_transfer_rate', 0.0)) if transfer_summary else 0.0
        return UnifiedBenchmarkSummary(
            trained_pipeline=artifacts.model_dump(),
            hidden_premise=hidden_summary,
            transfer=transfer_summary,
            vlso_grounded=vlso_summary,
            unseen_transfer=round(unseen_transfer, 4),
            analogy_usefulness=round(analogy_usefulness, 4),
            compiler_validity=round(compiler_validity, 4),
            grounded_explanation_fidelity=round(grounded_fidelity, 4),
            repair_success_rate=round(repair_success, 4),
        )

    def _pipeline(self, artifacts: UnifiedSemOpArtifacts) -> StructuredMeaningPipeline:
        return StructuredMeaningPipeline(
            mode=self.mode,
            analogy_policy_path=artifacts.analogy_policy_path or None,
            unified_parser_path=artifacts.unified_parser_path or None,
            retained_algebra_path=artifacts.retained_algebra_path or None,
            repair_policy_path=artifacts.repair_policy_path or None,
            repair_program_path=artifacts.retained_repair_program_path or None,
            multimodal_alignment_path=artifacts.multimodal_alignment_path or None,
        )

    @staticmethod
    def _evaluate_analogy_usefulness(pipeline: StructuredMeaningPipeline, cases: Sequence[AnalogyEvalCase]) -> float:
        if not cases:
            return 0.0
        scores: list[float] = []
        for case in cases:
            baseline = pipeline.run_with_memory_graphs(case.query, [])
            guided = pipeline.run_with_memory_graphs(case.query, case.similar_graphs)
            baseline_guard = any(step.id == 'analogy_requirement_guard' for step in baseline.plan)
            guided_guard = any(step.id == 'analogy_requirement_guard' for step in guided.plan)
            score = 1.0 if guided_guard and not baseline_guard else 0.0
            if case.expected_requirement:
                score = max(score, 1.0 if any(case.expected_requirement in step.requires for step in guided.plan) else 0.0)
            scores.append(score)
        return sum(scores) / float(len(scores))

    @staticmethod
    def _evaluate_grounded_explanations(pipeline: StructuredMeaningPipeline, cases: Sequence[GroundedExplanationEvalCase]) -> float:
        if not cases:
            return 0.0
        scores: list[float] = []
        for case in cases:
            graph = pipeline.run(case.query, source_context=case.source_context, visual_input=case.visual_input)
            symbolic_evidence = ' '.join(result.answer + ' ' + ' '.join(result.evidence) for result in graph.symbolic_results if result.domain == 'document_grounding')
            evidence_lookup = {node.id: node for node in graph.nodes if node.kind == 'evidence'}
            grounded_evidence = ' '.join(str(evidence_lookup[edge.target].attributes.get('text', evidence_lookup[edge.target].label)) for edge in graph.edges if edge.source == 'question' and edge.relation == 'GROUNDED_BY' and edge.target in evidence_lookup)
            evidence_text = (symbolic_evidence + ' ' + grounded_evidence).lower()
            if not case.expected_evidence_terms:
                scores.append(1.0 if evidence_text else 0.0)
                continue
            hits = sum(1 for item in case.expected_evidence_terms if item.lower() in evidence_text)
            scores.append(hits / float(len(case.expected_evidence_terms)))
        return sum(scores) / float(len(scores))

    @staticmethod
    def _evaluate_compiler_validity(cases: Sequence[CompilerRepairEvalCase], hidden_summary: dict[str, Any] | None) -> float:
        scores: list[float] = []
        for case in cases:
            graph = compile_and_execute(case.graph)
            if graph.operator_execution is not None:
                scores.append(float(graph.operator_execution.composition_score))
        if hidden_summary is not None:
            scores.append(float(hidden_summary.get('compiler_alignment_score', 0.0)))
        return sum(scores) / float(len(scores)) if scores else 0.0

    @staticmethod
    def _evaluate_repairs(pipeline: StructuredMeaningPipeline, cases: Sequence[CompilerRepairEvalCase]) -> float:
        if not cases:
            return 0.0
        hits = 0
        for case in cases:
            graph = StructuredMeaningGraph.from_dict(case.graph.model_dump())
            graph = pipeline.repair_engine.run(graph)
            report = graph.operator_execution
            repairs = ' '.join(report.counterexample_repairs if report is not None else []).lower()
            derived = ' '.join(report.derived_decisions if report is not None else []).lower()
            if not case.expected_repair_terms:
                hits += 1 if repairs or 'repair_applied:' in derived else 0
                continue
            if all(term.lower() in (repairs + ' ' + derived) for term in case.expected_repair_terms):
                hits += 1
        return hits / float(len(cases))

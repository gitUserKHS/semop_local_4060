from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .analogy_policy import AnalogyPolicyTrainer
from .continuous_learning import ContinuousLearningBundleBuilder
from .corpus_store import CorpusMemoryStore
from .graph_supervision import GraphSupervisionExporter
from .multimodal_alignment_memory import MultimodalAlignmentTrainer
from .operating_policies import infer_operating_domain, promoted_review_details, resolve_operating_policy, resolve_review_severity_weight, resolve_slice_balance_limit, resolve_slice_thresholds
from .operator_evolution import OperatorTransferEvalCase, OperatorTransferEvaluator
from .operator_repair import OperatorRepairEngine
from .operator_repair_policy import OperatorRepairPolicyTrainer
from .operator_runtime import compile_and_execute
from .pipeline import StructuredMeaningPipeline
from .premise_eval import HiddenPremiseEvalCase, HiddenPremiseEvaluator
from .repair_utility import RepairUtilityTrainer
from .retained_operator_algebra import RetainedOperatorTrainer
from .retained_repair_programs import RetainedRepairProgramTrainer
from .review_queue import severity_weight
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
    repair_utility_path: str = ''
    multimodal_alignment_path: str = ''
    continuous_learning_bundle_dir: str = ''

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedSemOpTrainingSummary:
    artifacts: UnifiedSemOpArtifacts
    trained_on_graphs: int
    promoted_review_graph_count: int
    repair_trace_graph_count: int
    augmented_graph_count: int
    analogy_policy: dict[str, Any]
    unified_parser: dict[str, Any]
    graph_supervision: dict[str, Any]
    retained_algebra: dict[str, Any]
    repair_policy: dict[str, Any]
    retained_repair_programs: dict[str, Any]
    repair_utility: dict[str, Any]
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
    expected_claim_terms: list[str] = field(default_factory=list)
    forbidden_unsupported_claim_terms: list[str] = field(default_factory=list)
    visual_input: Any | None = None
    domain: str = 'general'
    scenario: str = 'qa'
    severity: str = 'medium'
    case_weight: float = 0.0


@dataclass
class CompilerRepairEvalCase:
    graph: StructuredMeaningGraph
    expected_repair_terms: list[str] = field(default_factory=list)
    domain: str = 'general'
    scenario: str = 'qa'
    severity: str = 'medium'
    case_weight: float = 0.0


@dataclass
class BenchmarkSliceSummary:
    domain: str = 'general'
    scenario: str = 'qa'
    hidden_premise_quality: float = 0.0
    compiler_validity: float = 0.0
    grounded_explanation_fidelity: float = 0.0
    repair_success_rate: float = 0.0
    hidden_premise_case_count: int = 0
    grounding_case_count: int = 0
    compiler_case_count: int = 0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


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
    slice_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkGateThresholds:
    minimum_unseen_transfer: float = 0.0
    minimum_analogy_usefulness: float = 0.0
    minimum_compiler_validity: float = 0.55
    minimum_grounded_explanation_fidelity: float = 0.5
    minimum_repair_success_rate: float = 0.5
    regression_tolerance: float = 0.03
    require_improvement_if_baseline: bool = True

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_operating_domain(cls, domain: str | None) -> BenchmarkGateThresholds:
        policy = resolve_operating_policy(domain)
        return cls(
            minimum_unseen_transfer=policy.minimum_unseen_transfer,
            minimum_analogy_usefulness=policy.minimum_analogy_usefulness,
            minimum_compiler_validity=policy.minimum_compiler_validity,
            minimum_grounded_explanation_fidelity=policy.minimum_grounded_explanation_fidelity,
            minimum_repair_success_rate=policy.minimum_repair_success_rate,
            regression_tolerance=policy.regression_tolerance,
            require_improvement_if_baseline=policy.require_improvement_if_baseline,
        )


@dataclass
class BenchmarkGateDecision:
    accepted: bool
    improved_axes: list[str] = field(default_factory=list)
    regressed_axes: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    slice_blocking_reasons: list[str] = field(default_factory=list)
    candidate_metrics: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    baseline_slice_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    compared_against: str = ''
    decision_path: str = ''
    operating_domain: str = 'general'
    applied_thresholds: dict[str, Any] = field(default_factory=dict)
    slice_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromotedReviewBenchmarkCases:
    promoted_review_count: int = 0
    hidden_premise_cases: list[HiddenPremiseEvalCase] = field(default_factory=list)
    grounding_cases: list[GroundedExplanationEvalCase] = field(default_factory=list)
    compiler_cases: list[CompilerRepairEvalCase] = field(default_factory=list)
    output_path: str = ''

    def model_dump(self) -> dict[str, Any]:
        return {
            'promoted_review_count': self.promoted_review_count,
            'hidden_premise_case_count': len(self.hidden_premise_cases),
            'grounding_case_count': len(self.grounding_cases),
            'compiler_case_count': len(self.compiler_cases),
            'output_path': self.output_path,
        }

    def case_dump(self) -> dict[str, Any]:
        return {
            'promoted_review_count': self.promoted_review_count,
            'hidden_premise_cases': [asdict(item) for item in self.hidden_premise_cases],
            'grounding_cases': [asdict(item) for item in self.grounding_cases],
            'compiler_cases': [
                {
                    'graph': item.graph.model_dump(),
                    'expected_repair_terms': list(item.expected_repair_terms),
                    'domain': item.domain,
                    'scenario': item.scenario,
                    'severity': item.severity,
                    'case_weight': item.case_weight,
                }
                for item in self.compiler_cases
            ],
            'output_path': self.output_path,
        }


@dataclass
class PersistentBenchmarkCorpusSummary:
    hidden_premise_case_count: int = 0
    grounding_case_count: int = 0
    compiler_case_count: int = 0
    loaded_case_count: int = 0
    added_case_count: int = 0
    trimmed_case_count: int = 0
    slice_balance_limit: int = 0
    slice_balance_limits: dict[str, int] = field(default_factory=dict)
    output_path: str = ''

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkGatedTrainingSummary:
    training: UnifiedSemOpTrainingSummary
    benchmark: UnifiedBenchmarkSummary
    gate: BenchmarkGateDecision
    promoted_review_benchmarks: dict[str, Any] = field(default_factory=dict)
    benchmark_corpus: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class UnifiedSemOpTrainer:
    def train_from_store(
        self,
        store_path: str | Path,
        output_dir: str | Path,
        source: str | None = None,
        split: str = 'train',
        review_store_path: str | Path | None = None,
        approved_queries_only: bool = False,
        operating_domain: str | None = None,
    ) -> UnifiedSemOpTrainingSummary:
        store = CorpusMemoryStore(store_path)
        graphs = list(store.fetch_graphs(split=split, source=source))
        resolved_domain = operating_domain or infer_operating_domain(review_store_path)
        if approved_queries_only and review_store_path:
            promoted_queries = self._promoted_review_queries(review_store_path, operating_domain=resolved_domain)
            graphs = [graph for graph in graphs if graph.query in promoted_queries]
        review_graphs = self._promoted_review_graphs(review_store_path, operating_domain=resolved_domain)
        repair_trace_graphs = self._repair_trace_graphs(graphs + review_graphs)
        training_graphs = self._unique_graphs(graphs + review_graphs + repair_trace_graphs)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        analogy_summary = AnalogyPolicyTrainer().train_from_graphs(
            training_graphs,
            output_dir / 'analogy_policy.json',
        )
        unified_summary = UnifiedParserTrainer().train_from_graphs(
            training_graphs,
            output_dir / 'unified_parser.json',
        )
        graph_supervision_summary = GraphSupervisionExporter().export_from_graphs(
            training_graphs,
            output_dir / 'graph_supervision.jsonl',
        )
        retained_summary = RetainedOperatorTrainer().train_from_graphs(
            training_graphs,
            output_dir / 'retained_operator_algebra.json',
        )
        multimodal_alignment_summary = MultimodalAlignmentTrainer().train_from_graphs(
            training_graphs,
            output_dir / 'multimodal_alignment.json',
        )
        repair_policy_summary = OperatorRepairPolicyTrainer().train_from_graphs(
            training_graphs,
            output_dir / 'operator_repair_policy.json',
        )
        retained_repair_summary = RetainedRepairProgramTrainer().train_from_graphs(
            training_graphs,
            output_dir / 'retained_repair_programs.json',
        )
        repair_utility_summary = RepairUtilityTrainer().train_from_graphs(
            training_graphs,
            output_dir / 'repair_utility.json',
        )
        continuous_learning_summary = ContinuousLearningBundleBuilder().build_from_graphs(
            training_graphs,
            output_dir / 'continuous_learning_bundle',
            review_store_path=review_store_path,
            operating_domain=resolved_domain,
        )
        artifacts = UnifiedSemOpArtifacts(
            analogy_policy_path=str(output_dir / 'analogy_policy.json'),
            unified_parser_path=str(output_dir / 'unified_parser.json'),
            graph_supervision_path=str(output_dir / 'graph_supervision.jsonl'),
            retained_algebra_path=str(output_dir / 'retained_operator_algebra.json'),
            repair_policy_path=str(output_dir / 'operator_repair_policy.json'),
            retained_repair_program_path=str(output_dir / 'retained_repair_programs.json'),
            repair_utility_path=str(output_dir / 'repair_utility.json'),
            multimodal_alignment_path=str(output_dir / 'multimodal_alignment.json'),
            continuous_learning_bundle_dir=str(output_dir / 'continuous_learning_bundle'),
        )
        return UnifiedSemOpTrainingSummary(
            artifacts=artifacts,
            trained_on_graphs=len(graphs),
            promoted_review_graph_count=len(review_graphs),
            repair_trace_graph_count=len(repair_trace_graphs),
            augmented_graph_count=len(training_graphs),
            analogy_policy=analogy_summary.model_dump(),
            unified_parser=unified_summary.model_dump(),
            graph_supervision=graph_supervision_summary.model_dump(),
            retained_algebra=retained_summary.model_dump(),
            repair_policy=repair_policy_summary.model_dump(),
            retained_repair_programs=retained_repair_summary.model_dump(),
            repair_utility=repair_utility_summary.model_dump(),
            multimodal_alignment=multimodal_alignment_summary.model_dump(),
            continuous_learning_bundle=continuous_learning_summary.model_dump(),
        )

    @staticmethod
    def _promoted_review_queries(review_store_path: str | Path, operating_domain: str | None = None) -> set[str]:
        return {
            str(detail.get('query', '')).strip()
            for detail in promoted_review_details(review_store_path, domain=operating_domain)
            if str(detail.get('query', '')).strip()
        }

    @staticmethod
    def _promoted_review_graphs(review_store_path: str | Path | None, operating_domain: str | None = None) -> list[StructuredMeaningGraph]:
        if not review_store_path:
            return []
        graphs: list[StructuredMeaningGraph] = []
        for detail in promoted_review_details(review_store_path, domain=operating_domain):
            payload = detail.get('graph')
            if not isinstance(payload, dict):
                continue
            if not payload.get('query') or not payload.get('intent'):
                continue
            graphs.append(StructuredMeaningGraph.from_dict(payload))
        return graphs

    @staticmethod
    def _repair_trace_graphs(graphs: Sequence[StructuredMeaningGraph]) -> list[StructuredMeaningGraph]:
        engine = OperatorRepairEngine()
        repair_graphs: list[StructuredMeaningGraph] = []
        for graph in graphs:
            repaired = engine.run(StructuredMeaningGraph.from_dict(graph.model_dump()))
            report = repaired.operator_execution
            if report is None:
                continue
            if any(item.startswith(('repair_applied:', 'repair_program:', 'repair_rejected:')) for item in report.derived_decisions):
                repair_graphs.append(repaired)
        return repair_graphs

    @staticmethod
    def _unique_graphs(graphs: Sequence[StructuredMeaningGraph]) -> list[StructuredMeaningGraph]:
        unique: dict[str, StructuredMeaningGraph] = {}
        for graph in graphs:
            signature = hashlib.sha1(json.dumps(graph.model_dump(), ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
            unique[signature] = graph
        return list(unique.values())

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
        hidden_cases = list(hidden_premise_cases or [])
        grounding_eval_cases = list(grounding_cases or [])
        compiler_eval_cases = list(compiler_cases or [])
        hidden_summary = HiddenPremiseEvaluator(pipeline).evaluate(hidden_cases).__dict__ if hidden_cases else None
        transfer_summary = OperatorTransferEvaluator(pipeline).evaluate(transfer_cases).model_dump() if transfer_cases else None
        vlso_summary = VlsoGroundedEvaluator(VLSOReasoner()).evaluate_cases(list(vlso_cases)).model_dump() if vlso_cases else None
        analogy_usefulness = self._evaluate_analogy_usefulness(pipeline, analogy_cases or [])
        grounded_fidelity = self._evaluate_grounded_explanations(pipeline, grounding_eval_cases)
        compiler_validity = self._evaluate_compiler_validity(compiler_eval_cases, hidden_summary)
        repair_success = self._evaluate_repairs(pipeline, compiler_eval_cases)
        unseen_transfer = float(transfer_summary.get('unseen_domain_transfer_rate', 0.0)) if transfer_summary else 0.0
        slice_metrics = self._evaluate_slice_metrics(
            pipeline,
            hidden_cases,
            grounding_eval_cases,
            compiler_eval_cases,
        )
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
            slice_metrics=slice_metrics,
        )

    def _pipeline(self, artifacts: UnifiedSemOpArtifacts) -> StructuredMeaningPipeline:
        return StructuredMeaningPipeline(
            mode=self.mode,
            analogy_policy_path=artifacts.analogy_policy_path or None,
            unified_parser_path=artifacts.unified_parser_path or None,
            retained_algebra_path=artifacts.retained_algebra_path or None,
            repair_policy_path=artifacts.repair_policy_path or None,
            repair_program_path=artifacts.retained_repair_program_path or None,
            repair_utility_path=artifacts.repair_utility_path or None,
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
    def _case_weight(case: Any) -> float:
        weight = float(getattr(case, 'case_weight', 0.0) or 0.0)
        if weight > 0.0:
            return weight
        return severity_weight(getattr(case, 'severity', 'medium'), getattr(case, 'domain', 'general'), getattr(case, 'scenario', 'qa'))

    @staticmethod
    def _weighted_average(weighted_terms: Sequence[tuple[float, float]]) -> float:
        if not weighted_terms:
            return 0.0
        total_weight = sum(float(weight) for _value, weight in weighted_terms)
        if total_weight <= 0.0:
            return sum(float(value) for value, _weight in weighted_terms) / float(len(weighted_terms))
        return sum(float(value) * float(weight) for value, weight in weighted_terms) / float(total_weight)

    @staticmethod
    def _evaluate_hidden_premise_case(pipeline: StructuredMeaningPipeline, case: HiddenPremiseEvalCase) -> float:
        graph = pipeline.run(case.query)
        predicted_risky = [check.action for check in graph.goal_preservation_checks if check.status in {'risk_high', 'invalid'}]
        scores = [
            HiddenPremiseEvaluator._recall(graph.required_premises, case.expected_required_premises),
            HiddenPremiseEvaluator._recall(graph.hidden_goals, case.expected_hidden_goals),
            HiddenPremiseEvaluator._recall(predicted_risky, case.expected_risky_actions),
            HiddenPremiseEvaluator._unsupported_precision(graph.required_premises, case.forbidden_premises),
            1.0 if bool(graph.clarification_needed) == bool(case.expected_clarification_needed) else 0.0,
            HiddenPremiseEvaluator._requirement_state_accuracy(
                graph.satisfied_premises,
                graph.missing_premises,
                case.expected_satisfied_premises,
                case.expected_missing_premises,
            ),
        ]
        if case.expected_support_operators is not None:
            supported = {item.name for item in graph.induced_operators} | {item.operator_name for item in graph.operator_decompositions}
            scores.append(HiddenPremiseEvaluator._recall(supported, case.expected_support_operators))
        return sum(scores) / float(len(scores)) if scores else 0.0

    @staticmethod
    def _evaluate_grounded_case(pipeline: StructuredMeaningPipeline, case: GroundedExplanationEvalCase) -> float:
        graph = pipeline.run(case.query, source_context=case.source_context, visual_input=case.visual_input)
        if graph.operator_execution is None:
            graph = compile_and_execute(graph)
        symbolic_evidence = ' '.join(
            result.answer + ' ' + ' '.join(result.evidence)
            for result in graph.symbolic_results
            if result.domain == 'document_grounding'
        )
        evidence_lookup = {node.id: node for node in graph.nodes if node.kind == 'evidence'}
        grounded_evidence = ' '.join(
            str(evidence_lookup[edge.target].attributes.get('text', evidence_lookup[edge.target].label))
            for edge in graph.edges
            if edge.source == 'question' and edge.relation == 'GROUNDED_BY' and edge.target in evidence_lookup
        )
        evidence_text = (symbolic_evidence + ' ' + grounded_evidence).lower()
        if not case.expected_claim_terms and not case.forbidden_unsupported_claim_terms:
            if not case.expected_evidence_terms:
                return 1.0 if evidence_text else 0.0
            hits = sum(1 for item in case.expected_evidence_terms if item.lower() in evidence_text)
            return hits / float(len(case.expected_evidence_terms))
        components: list[float] = []
        if case.expected_evidence_terms:
            evidence_hits = sum(1 for item in case.expected_evidence_terms if item.lower() in evidence_text)
            components.append(evidence_hits / float(len(case.expected_evidence_terms)))
        elif evidence_text:
            components.append(1.0)
        report = graph.operator_execution
        claim_groundings = list(report.claim_groundings) if report is not None else []
        grounded_claim_text = ' '.join(item.claim for item in claim_groundings if item.grounded).lower()
        unsupported_claim_text = ' '.join(item.claim for item in claim_groundings if not item.grounded).lower()
        if case.expected_claim_terms:
            claim_hits = sum(1 for item in case.expected_claim_terms if item.lower() in grounded_claim_text)
            components.append(claim_hits / float(len(case.expected_claim_terms)))
        elif claim_groundings and report is not None:
            components.append(float(report.claim_grounding_score))
        if case.forbidden_unsupported_claim_terms:
            forbidden_hit = any(item.lower() in unsupported_claim_text for item in case.forbidden_unsupported_claim_terms)
            components.append(0.0 if forbidden_hit else 1.0)
        if not components:
            return 1.0 if grounded_claim_text or evidence_text else 0.0
        return sum(components) / float(len(components))

    @staticmethod
    def _evaluate_compiler_case(case: CompilerRepairEvalCase) -> float:
        graph = compile_and_execute(StructuredMeaningGraph.from_dict(case.graph.model_dump()))
        if graph.operator_execution is None:
            return 0.0
        return float(graph.operator_execution.composition_score)

    @staticmethod
    def _evaluate_repair_case(pipeline: StructuredMeaningPipeline, case: CompilerRepairEvalCase) -> float:
        graph = StructuredMeaningGraph.from_dict(case.graph.model_dump())
        graph = pipeline.repair_engine.run(graph)
        report = graph.operator_execution
        repairs = ' '.join(report.counterexample_repairs if report is not None else []).lower()
        derived = ' '.join(report.derived_decisions if report is not None else []).lower()
        if not case.expected_repair_terms:
            return 1.0 if repairs or 'repair_applied:' in derived else 0.0
        return 1.0 if all(term.lower() in (repairs + ' ' + derived) for term in case.expected_repair_terms) else 0.0

    @classmethod
    def _evaluate_slice_metrics(
        cls,
        pipeline: StructuredMeaningPipeline,
        hidden_cases: Sequence[HiddenPremiseEvalCase],
        grounding_cases: Sequence[GroundedExplanationEvalCase],
        compiler_cases: Sequence[CompilerRepairEvalCase],
    ) -> dict[str, dict[str, Any]]:
        buckets: dict[str, dict[str, Any]] = {}

        def bucket_for(domain: str, scenario: str) -> dict[str, Any]:
            key = cls._slice_key(domain, scenario)
            if key not in buckets:
                buckets[key] = {
                    'domain': domain,
                    'scenario': scenario,
                    'hidden_terms': [],
                    'grounding_terms': [],
                    'compiler_terms': [],
                    'repair_terms': [],
                }
            return buckets[key]

        for case in hidden_cases:
            bucket_for(case.domain, case.scenario)['hidden_terms'].append((cls._evaluate_hidden_premise_case(pipeline, case), cls._case_weight(case)))
        for case in grounding_cases:
            bucket_for(case.domain, case.scenario)['grounding_terms'].append((cls._evaluate_grounded_case(pipeline, case), cls._case_weight(case)))
        for case in compiler_cases:
            bucket = bucket_for(case.domain, case.scenario)
            weight = cls._case_weight(case)
            bucket['compiler_terms'].append((cls._evaluate_compiler_case(case), weight))
            bucket['repair_terms'].append((cls._evaluate_repair_case(pipeline, case), weight))

        summaries: dict[str, dict[str, Any]] = {}
        for key, bucket in buckets.items():
            hidden_terms = bucket['hidden_terms']
            grounding_terms = bucket['grounding_terms']
            compiler_terms = bucket['compiler_terms']
            repair_terms = bucket['repair_terms']
            summaries[key] = BenchmarkSliceSummary(
                domain=bucket['domain'],
                scenario=bucket['scenario'],
                hidden_premise_quality=round(cls._weighted_average(hidden_terms), 4) if hidden_terms else 0.0,
                compiler_validity=round(cls._weighted_average(compiler_terms), 4) if compiler_terms else 0.0,
                grounded_explanation_fidelity=round(cls._weighted_average(grounding_terms), 4) if grounding_terms else 0.0,
                repair_success_rate=round(cls._weighted_average(repair_terms), 4) if repair_terms else 0.0,
                hidden_premise_case_count=len(hidden_terms),
                grounding_case_count=len(grounding_terms),
                compiler_case_count=len(compiler_terms),
            ).model_dump()
        return summaries

    @staticmethod
    def _slice_key(domain: str, scenario: str) -> str:
        resolved_domain = str(domain or 'general').strip() or 'general'
        resolved_scenario = str(scenario or 'qa').strip() or 'qa'
        return f'{resolved_domain}::{resolved_scenario}'

    @classmethod
    def _evaluate_grounded_explanations(cls, pipeline: StructuredMeaningPipeline, cases: Sequence[GroundedExplanationEvalCase]) -> float:
        if not cases:
            return 0.0
        weighted_terms = [(cls._evaluate_grounded_case(pipeline, case), cls._case_weight(case)) for case in cases]
        return cls._weighted_average(weighted_terms)

    @classmethod
    def _evaluate_compiler_validity(cls, cases: Sequence[CompilerRepairEvalCase], hidden_summary: dict[str, Any] | None) -> float:
        weighted_terms = [(cls._evaluate_compiler_case(case), cls._case_weight(case)) for case in cases]
        if hidden_summary is not None:
            weighted_terms.append((float(hidden_summary.get('compiler_alignment_score', 0.0)), 1.0))
        return cls._weighted_average(weighted_terms)

    @classmethod
    def _evaluate_repairs(cls, pipeline: StructuredMeaningPipeline, cases: Sequence[CompilerRepairEvalCase]) -> float:
        if not cases:
            return 0.0
        weighted_terms = [(cls._evaluate_repair_case(pipeline, case), cls._case_weight(case)) for case in cases]
        return cls._weighted_average(weighted_terms)


class BenchmarkGatedContinuousTrainer:
    def __init__(self, mode: str = 'heuristic') -> None:
        self.mode = mode
        self.trainer = UnifiedSemOpTrainer()
        self.harness = UnifiedBenchmarkHarness(mode=mode)

    def train_evaluate_and_gate(
        self,
        store_path: str | Path,
        output_dir: str | Path,
        source: str | None = None,
        split: str = 'train',
        review_store_path: str | Path | None = None,
        approved_queries_only: bool = False,
        hidden_premise_cases: Sequence[HiddenPremiseEvalCase] | None = None,
        transfer_cases: Sequence[OperatorTransferEvalCase] | None = None,
        analogy_cases: Sequence[AnalogyEvalCase] | None = None,
        grounding_cases: Sequence[GroundedExplanationEvalCase] | None = None,
        compiler_cases: Sequence[CompilerRepairEvalCase] | None = None,
        vlso_cases: Sequence[VlsoEvalCase] | None = None,
        baseline_summary_path: str | Path | None = None,
        thresholds: BenchmarkGateThresholds | None = None,
        operating_domain: str | None = None,
        benchmark_corpus_path: str | Path | None = None,
    ) -> BenchmarkGatedTrainingSummary:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_domain = operating_domain or infer_operating_domain(review_store_path)
        applied_thresholds = thresholds or BenchmarkGateThresholds.from_operating_domain(resolved_domain)
        benchmark_corpus = PersistentBenchmarkCorpusSummary()
        promoted_review_benchmarks = self._derive_promoted_review_benchmarks(
            review_store_path,
            operating_domain=resolved_domain,
            output_path=output_dir / 'promoted_review_benchmark_cases.json',
        )
        benchmark_cases = promoted_review_benchmarks
        if benchmark_corpus_path:
            benchmark_cases, benchmark_corpus = self._merge_with_benchmark_corpus(
                promoted_review_benchmarks,
                benchmark_corpus_path,
            )
        hidden_cases = list(hidden_premise_cases or []) + benchmark_cases.hidden_premise_cases
        grounding_eval_cases = list(grounding_cases or []) + benchmark_cases.grounding_cases
        compiler_eval_cases = list(compiler_cases or []) + benchmark_cases.compiler_cases
        training = self.trainer.train_from_store(
            store_path,
            output_dir,
            source=source,
            split=split,
            review_store_path=review_store_path,
            approved_queries_only=approved_queries_only,
            operating_domain=resolved_domain,
        )
        benchmark = self.harness.evaluate(
            training.artifacts,
            hidden_premise_cases=hidden_cases,
            transfer_cases=transfer_cases,
            analogy_cases=analogy_cases,
            grounding_cases=grounding_eval_cases,
            compiler_cases=compiler_eval_cases,
            vlso_cases=vlso_cases,
        )
        baseline_payload = self._load_baseline_payload(baseline_summary_path)
        baseline_metrics = self._metric_view(baseline_payload) if baseline_payload else None
        baseline_slice_metrics = self._load_baseline_slice_metrics(baseline_payload)
        gate = self._gate_decision(
            benchmark,
            baseline_metrics=baseline_metrics,
            baseline_slice_metrics=baseline_slice_metrics,
            thresholds=applied_thresholds,
            compared_against=str(baseline_summary_path or ''),
            operating_domain=resolved_domain,
        )
        decision_path = output_dir / 'benchmark_gate.json'
        gate.decision_path = str(decision_path)
        payload = {
            'training': training.model_dump(),
            'benchmark': benchmark.model_dump(),
            'gate': gate.model_dump(),
            'promoted_review_benchmarks': promoted_review_benchmarks.model_dump(),
            'benchmark_corpus': benchmark_corpus.model_dump(),
        }
        decision_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        if gate.accepted:
            (output_dir / 'accepted_benchmark_summary.json').write_text(
                json.dumps(benchmark.model_dump(), ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
        return BenchmarkGatedTrainingSummary(
            training=training,
            benchmark=benchmark,
            gate=gate,
            promoted_review_benchmarks=promoted_review_benchmarks.model_dump(),
            benchmark_corpus=benchmark_corpus.model_dump(),
        )

    @staticmethod
    def _gate_decision(
        benchmark: UnifiedBenchmarkSummary,
        *,
        baseline_metrics: dict[str, float] | None,
        baseline_slice_metrics: dict[str, dict[str, Any]] | None,
        thresholds: BenchmarkGateThresholds,
        compared_against: str = '',
        operating_domain: str = 'general',
    ) -> BenchmarkGateDecision:
        candidate_metrics = BenchmarkGatedContinuousTrainer._metric_view(benchmark.model_dump())
        improved_axes: list[str] = []
        regressed_axes: list[str] = []
        blocking_reasons: list[str] = []
        for metric_name, threshold in BenchmarkGatedContinuousTrainer._threshold_view(thresholds).items():
            value = candidate_metrics.get(metric_name, 0.0)
            if value + 1e-9 < threshold:
                blocking_reasons.append(f'{metric_name} below threshold: {value:.4f} < {threshold:.4f}')
        if baseline_metrics:
            for metric_name, value in candidate_metrics.items():
                baseline_value = float(baseline_metrics.get(metric_name, 0.0))
                if value > baseline_value + thresholds.regression_tolerance:
                    improved_axes.append(metric_name)
                elif value < baseline_value - thresholds.regression_tolerance:
                    regressed_axes.append(metric_name)
                    blocking_reasons.append(
                        f'{metric_name} regressed against baseline: {value:.4f} < {baseline_value:.4f}'
                    )
        slice_blocking_reasons, slice_improvements = BenchmarkGatedContinuousTrainer._slice_gate_reasons(
            benchmark.slice_metrics,
            baseline_slice_metrics or {},
            thresholds.regression_tolerance,
        )
        if thresholds.require_improvement_if_baseline and (baseline_metrics or baseline_slice_metrics) and not improved_axes and not slice_improvements:
            blocking_reasons.append('no benchmark axis improved beyond tolerance')
        return BenchmarkGateDecision(
            accepted=not (blocking_reasons or slice_blocking_reasons),
            improved_axes=improved_axes + slice_improvements,
            regressed_axes=regressed_axes,
            blocking_reasons=blocking_reasons,
            slice_blocking_reasons=slice_blocking_reasons,
            candidate_metrics=candidate_metrics,
            baseline_metrics=baseline_metrics or {},
            baseline_slice_metrics=baseline_slice_metrics or {},
            compared_against=compared_against,
            operating_domain=operating_domain,
            applied_thresholds=thresholds.model_dump(),
            slice_metrics=dict(benchmark.slice_metrics),
        )

    @staticmethod
    def _threshold_view(thresholds: BenchmarkGateThresholds) -> dict[str, float]:
        return {
            'unseen_transfer': thresholds.minimum_unseen_transfer,
            'analogy_usefulness': thresholds.minimum_analogy_usefulness,
            'compiler_validity': thresholds.minimum_compiler_validity,
            'grounded_explanation_fidelity': thresholds.minimum_grounded_explanation_fidelity,
            'repair_success_rate': thresholds.minimum_repair_success_rate,
        }

    @staticmethod
    def _metric_view(payload: dict[str, Any]) -> dict[str, float]:
        return {
            'unseen_transfer': float(payload.get('unseen_transfer', 0.0)),
            'analogy_usefulness': float(payload.get('analogy_usefulness', 0.0)),
            'compiler_validity': float(payload.get('compiler_validity', 0.0)),
            'grounded_explanation_fidelity': float(payload.get('grounded_explanation_fidelity', 0.0)),
            'repair_success_rate': float(payload.get('repair_success_rate', 0.0)),
        }

    @staticmethod
    def _slice_gate_reasons(
        slice_metrics: dict[str, Any],
        baseline_slice_metrics: dict[str, dict[str, Any]],
        regression_tolerance: float,
    ) -> tuple[list[str], list[str]]:
        reasons: list[str] = []
        improvements: list[str] = []
        for slice_key, metrics in sorted(slice_metrics.items()):
            if not isinstance(metrics, dict):
                continue
            domain = str(metrics.get('domain', 'general')).strip() or 'general'
            scenario = str(metrics.get('scenario', 'qa')).strip() or 'qa'
            thresholds = resolve_slice_thresholds(domain, scenario)
            baseline_metrics = baseline_slice_metrics.get(slice_key, {}) if isinstance(baseline_slice_metrics, dict) else {}
            compiler_case_count = int(metrics.get('compiler_case_count', 0) or 0)
            grounding_case_count = int(metrics.get('grounding_case_count', 0) or 0)
            baseline_compiler_case_count = int(baseline_metrics.get('compiler_case_count', 0) or 0)
            baseline_grounding_case_count = int(baseline_metrics.get('grounding_case_count', 0) or 0)
            if compiler_case_count > 0:
                compiler_value = float(metrics.get('compiler_validity', 0.0))
                compiler_threshold = float(thresholds.get('minimum_compiler_validity', 0.0))
                if compiler_value + 1e-9 < compiler_threshold:
                    reasons.append(
                        f'slice {slice_key} compiler_validity below threshold: {compiler_value:.4f} < {compiler_threshold:.4f}'
                    )
                elif baseline_compiler_case_count > 0:
                    baseline_compiler_value = float(baseline_metrics.get('compiler_validity', 0.0))
                    if compiler_value > baseline_compiler_value + regression_tolerance:
                        improvements.append(f'slice:{slice_key}:compiler_validity')
                    elif compiler_value < baseline_compiler_value - regression_tolerance:
                        reasons.append(
                            f'slice {slice_key} compiler_validity regressed against baseline: {compiler_value:.4f} < {baseline_compiler_value:.4f}'
                        )
                repair_value = float(metrics.get('repair_success_rate', 0.0))
                repair_threshold = float(thresholds.get('minimum_repair_success_rate', 0.0))
                if repair_value + 1e-9 < repair_threshold:
                    reasons.append(
                        f'slice {slice_key} repair_success_rate below threshold: {repair_value:.4f} < {repair_threshold:.4f}'
                    )
                elif baseline_compiler_case_count > 0:
                    baseline_repair_value = float(baseline_metrics.get('repair_success_rate', 0.0))
                    if repair_value > baseline_repair_value + regression_tolerance:
                        improvements.append(f'slice:{slice_key}:repair_success_rate')
                    elif repair_value < baseline_repair_value - regression_tolerance:
                        reasons.append(
                            f'slice {slice_key} repair_success_rate regressed against baseline: {repair_value:.4f} < {baseline_repair_value:.4f}'
                        )
            if grounding_case_count > 0:
                grounding_value = float(metrics.get('grounded_explanation_fidelity', 0.0))
                grounding_threshold = float(thresholds.get('minimum_grounded_explanation_fidelity', 0.0))
                if grounding_value + 1e-9 < grounding_threshold:
                    reasons.append(
                        f'slice {slice_key} grounded_explanation_fidelity below threshold: {grounding_value:.4f} < {grounding_threshold:.4f}'
                    )
                elif baseline_grounding_case_count > 0:
                    baseline_grounding_value = float(baseline_metrics.get('grounded_explanation_fidelity', 0.0))
                    if grounding_value > baseline_grounding_value + regression_tolerance:
                        improvements.append(f'slice:{slice_key}:grounded_explanation_fidelity')
                    elif grounding_value < baseline_grounding_value - regression_tolerance:
                        reasons.append(
                            f'slice {slice_key} grounded_explanation_fidelity regressed against baseline: {grounding_value:.4f} < {baseline_grounding_value:.4f}'
                        )
        return reasons, improvements

    @staticmethod
    def _load_baseline_payload(summary_path: str | Path | None) -> dict[str, Any] | None:
        if not summary_path:
            return None
        path = Path(summary_path)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(payload, dict) and isinstance(payload.get('benchmark'), dict):
            payload = payload['benchmark']
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _load_baseline_slice_metrics(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        if not payload or not isinstance(payload.get('slice_metrics'), dict):
            return {}
        return {
            str(key): value
            for key, value in payload.get('slice_metrics', {}).items()
            if isinstance(value, dict)
        }

    @staticmethod
    def _slice_key(domain: str, scenario: str) -> str:
        return UnifiedBenchmarkHarness._slice_key(domain, scenario)

    @staticmethod
    def _case_weight(case: Any) -> float:
        return UnifiedBenchmarkHarness._case_weight(case)


    @classmethod
    def _merge_with_benchmark_corpus(
        cls,
        derived: PromotedReviewBenchmarkCases,
        benchmark_corpus_path: str | Path,
    ) -> tuple[PromotedReviewBenchmarkCases, PersistentBenchmarkCorpusSummary]:
        existing = cls._load_benchmark_corpus(benchmark_corpus_path)
        merged = PromotedReviewBenchmarkCases(
            promoted_review_count=existing.promoted_review_count + derived.promoted_review_count,
            hidden_premise_cases=cls._dedupe_hidden_cases(existing.hidden_premise_cases + derived.hidden_premise_cases),
            grounding_cases=cls._dedupe_grounding_cases(existing.grounding_cases + derived.grounding_cases),
            compiler_cases=cls._dedupe_compiler_cases(existing.compiler_cases + derived.compiler_cases),
            output_path=str(benchmark_corpus_path),
        )
        balanced = PromotedReviewBenchmarkCases(
            promoted_review_count=merged.promoted_review_count,
            hidden_premise_cases=cls._balance_hidden_cases(merged.hidden_premise_cases),
            grounding_cases=cls._balance_grounding_cases(merged.grounding_cases),
            compiler_cases=cls._balance_compiler_cases(merged.compiler_cases),
            output_path=str(benchmark_corpus_path),
        )
        output = Path(benchmark_corpus_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(balanced.case_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        loaded_case_count = len(existing.hidden_premise_cases) + len(existing.grounding_cases) + len(existing.compiler_cases)
        merged_case_count = len(merged.hidden_premise_cases) + len(merged.grounding_cases) + len(merged.compiler_cases)
        balanced_case_count = len(balanced.hidden_premise_cases) + len(balanced.grounding_cases) + len(balanced.compiler_cases)
        slice_balance_limits = cls._slice_balance_limits(balanced)
        return balanced, PersistentBenchmarkCorpusSummary(
            hidden_premise_case_count=len(balanced.hidden_premise_cases),
            grounding_case_count=len(balanced.grounding_cases),
            compiler_case_count=len(balanced.compiler_cases),
            loaded_case_count=loaded_case_count,
            added_case_count=max(0, balanced_case_count - loaded_case_count),
            trimmed_case_count=max(0, merged_case_count - balanced_case_count),
            slice_balance_limit=max(slice_balance_limits.values(), default=0),
            slice_balance_limits=slice_balance_limits,
            output_path=str(output),
        )

    @classmethod
    def _load_benchmark_corpus(cls, benchmark_corpus_path: str | Path) -> PromotedReviewBenchmarkCases:
        path = Path(benchmark_corpus_path)
        if not path.exists():
            return PromotedReviewBenchmarkCases(output_path=str(path))
        payload = json.loads(path.read_text(encoding='utf-8'))
        hidden_cases = [
            HiddenPremiseEvalCase(
                query=str(item.get('query', '')),
                expected_hidden_goals=list(item.get('expected_hidden_goals', [])),
                expected_required_premises=list(item.get('expected_required_premises', [])),
                expected_satisfied_premises=list(item.get('expected_satisfied_premises', [])),
                expected_missing_premises=list(item.get('expected_missing_premises', [])),
                expected_risky_actions=list(item.get('expected_risky_actions', [])),
                forbidden_premises=list(item.get('forbidden_premises', [])),
                expected_clarification_needed=bool(item.get('expected_clarification_needed', False)),
                expected_clarification_score=item.get('expected_clarification_score'),
                expected_support_operators=list(item.get('expected_support_operators', [])) if item.get('expected_support_operators') is not None else None,
                domain=str(item.get('domain', 'general')).strip() or 'general',
                scenario=str(item.get('scenario', 'qa')).strip() or 'qa',
                severity=str(item.get('severity', 'medium')).strip() or 'medium',
                case_weight=float(item.get('case_weight', severity_weight(item.get('severity', 'medium'), item.get('domain', 'general'), item.get('scenario', 'qa'))) or 0.0),
            )
            for item in payload.get('hidden_premise_cases', [])
            if isinstance(item, dict) and item.get('query')
        ]
        grounding_cases = [
            GroundedExplanationEvalCase(
                query=str(item.get('query', '')),
                source_context=str(item.get('source_context', '')),
                expected_evidence_terms=list(item.get('expected_evidence_terms', [])),
                expected_claim_terms=list(item.get('expected_claim_terms', [])),
                forbidden_unsupported_claim_terms=list(item.get('forbidden_unsupported_claim_terms', [])),
                visual_input=item.get('visual_input'),
                domain=str(item.get('domain', 'general')).strip() or 'general',
                scenario=str(item.get('scenario', 'qa')).strip() or 'qa',
                severity=str(item.get('severity', 'medium')).strip() or 'medium',
                case_weight=float(item.get('case_weight', severity_weight(item.get('severity', 'medium'), item.get('domain', 'general'), item.get('scenario', 'qa'))) or 0.0),
            )
            for item in payload.get('grounding_cases', [])
            if isinstance(item, dict) and item.get('query') and item.get('source_context')
        ]
        compiler_cases = [
            CompilerRepairEvalCase(
                graph=StructuredMeaningGraph.from_dict(item.get('graph', {})),
                expected_repair_terms=list(item.get('expected_repair_terms', [])),
                domain=str(item.get('domain', 'general')).strip() or 'general',
                scenario=str(item.get('scenario', 'qa')).strip() or 'qa',
                severity=str(item.get('severity', 'medium')),
                case_weight=float(item.get('case_weight', severity_weight(item.get('severity', 'medium'), item.get('domain', 'general'), item.get('scenario', 'qa'))) or 0.0),
            )
            for item in payload.get('compiler_cases', [])
            if isinstance(item, dict) and isinstance(item.get('graph'), dict) and item['graph'].get('query') and item['graph'].get('intent')
        ]
        return PromotedReviewBenchmarkCases(
            promoted_review_count=int(payload.get('promoted_review_count', 0) or 0),
            hidden_premise_cases=cls._balance_hidden_cases(cls._dedupe_hidden_cases(hidden_cases)),
            grounding_cases=cls._balance_grounding_cases(cls._dedupe_grounding_cases(grounding_cases)),
            compiler_cases=cls._balance_compiler_cases(cls._dedupe_compiler_cases(compiler_cases)),
            output_path=str(path),
        )

    @classmethod
    def _dedupe_hidden_cases(cls, cases: Sequence[HiddenPremiseEvalCase]) -> list[HiddenPremiseEvalCase]:
        unique: dict[str, HiddenPremiseEvalCase] = {}
        for case in cases:
            unique[cls._case_signature(asdict(case))] = case
        return list(unique.values())

    @classmethod
    def _dedupe_grounding_cases(cls, cases: Sequence[GroundedExplanationEvalCase]) -> list[GroundedExplanationEvalCase]:
        unique: dict[str, GroundedExplanationEvalCase] = {}
        for case in cases:
            unique[cls._case_signature(asdict(case))] = case
        return list(unique.values())

    @classmethod
    def _dedupe_compiler_cases(cls, cases: Sequence[CompilerRepairEvalCase]) -> list[CompilerRepairEvalCase]:
        unique: dict[str, CompilerRepairEvalCase] = {}
        for case in cases:
            payload = {
                'graph': case.graph.model_dump(),
                'expected_repair_terms': list(case.expected_repair_terms),
                'domain': case.domain,
                'scenario': case.scenario,
                'severity': case.severity,
                'case_weight': case.case_weight,
            }
            unique[cls._case_signature(payload)] = case
        return list(unique.values())

    @classmethod
    def _balance_hidden_cases(cls, cases: Sequence[HiddenPremiseEvalCase]) -> list[HiddenPremiseEvalCase]:
        return cls._balance_cases_by_slice(cases)

    @classmethod
    def _balance_grounding_cases(cls, cases: Sequence[GroundedExplanationEvalCase]) -> list[GroundedExplanationEvalCase]:
        return cls._balance_cases_by_slice(cases)

    @classmethod
    def _balance_compiler_cases(cls, cases: Sequence[CompilerRepairEvalCase]) -> list[CompilerRepairEvalCase]:
        return cls._balance_cases_by_slice(cases)

    @classmethod
    def _balance_cases_by_slice(cls, cases: Sequence[Any]) -> list[Any]:
        grouped: dict[str, list[tuple[float, int, Any]]] = {}
        for index, case in enumerate(cases):
            key = cls._slice_key(getattr(case, 'domain', 'general'), getattr(case, 'scenario', 'qa'))
            grouped.setdefault(key, []).append((cls._case_weight(case), index, case))
        selected: list[Any] = []
        for slice_key in sorted(grouped):
            ranked = sorted(grouped[slice_key], key=lambda item: (item[0], item[1]), reverse=True)
            sample_case = ranked[0][2]
            limit = resolve_slice_balance_limit(getattr(sample_case, 'domain', 'general'), getattr(sample_case, 'scenario', 'qa'))
            selected.extend(case for _weight, _index, case in ranked[:limit])
        return selected

    @classmethod
    def _slice_balance_limits(cls, benchmark_cases: PromotedReviewBenchmarkCases) -> dict[str, int]:
        limits: dict[str, int] = {}
        for case in list(benchmark_cases.hidden_premise_cases) + list(benchmark_cases.grounding_cases) + list(benchmark_cases.compiler_cases):
            slice_key = cls._slice_key(getattr(case, 'domain', 'general'), getattr(case, 'scenario', 'qa'))
            limits[slice_key] = resolve_slice_balance_limit(getattr(case, 'domain', 'general'), getattr(case, 'scenario', 'qa'))
        return limits

    @staticmethod
    def _case_signature(payload: dict[str, Any]) -> str:
        normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(normalized.encode('utf-8')).hexdigest()

    @classmethod
    def _derive_promoted_review_benchmarks(
        cls,
        review_store_path: str | Path | None,
        *,
        operating_domain: str | None = None,
        output_path: str | Path | None = None,
    ) -> PromotedReviewBenchmarkCases:
        derived = PromotedReviewBenchmarkCases(output_path=str(output_path or ''))
        if not review_store_path:
            return derived
        details = promoted_review_details(review_store_path, domain=operating_domain)
        derived.promoted_review_count = len(details)
        for detail in details:
            graph = cls._graph_from_review_detail(detail)
            reasons = {str(item).strip() for item in detail.get('reasons', []) if str(item).strip()}
            eval_query = cls._review_eval_query(detail)
            domain = str(detail.get('domain', 'general')).strip() or 'general'
            scenario = str(detail.get('scenario', 'qa')).strip() or 'qa'
            severity = str(detail.get('severity', 'medium')).strip() or 'medium'
            weight = float(detail.get('severity_weight', resolve_review_severity_weight(domain, scenario, severity)) or 0.0)
            if graph is not None and (graph.hidden_goals or graph.required_premises or graph.missing_premises or graph.satisfied_premises):
                risky_actions = [check.action for check in graph.goal_preservation_checks if check.status in {'risk_high', 'invalid'}]
                derived.hidden_premise_cases.append(
                    HiddenPremiseEvalCase(
                        query=eval_query,
                        expected_hidden_goals=list(graph.hidden_goals[:3]),
                        expected_required_premises=list(graph.required_premises[:4]),
                        expected_satisfied_premises=list(graph.satisfied_premises[:4]),
                        expected_missing_premises=list(graph.missing_premises[:4]),
                        expected_risky_actions=risky_actions[:3],
                        forbidden_premises=[],
                        expected_clarification_needed=bool(graph.clarification_needed),
                        expected_clarification_score=float(graph.clarification_score),
                        domain=domain,
                        scenario=scenario,
                        severity=severity,
                        case_weight=weight,
                    )
                )
            source_context = cls._review_source_context(detail, graph)
            if source_context and ('grounding_review' in reasons or (graph is not None and cls._has_grounding_signal(graph))):
                derived.grounding_cases.append(
                    GroundedExplanationEvalCase(
                        query=str(detail.get('query', '')),
                        source_context=source_context,
                        expected_evidence_terms=cls._grounding_expected_terms(detail, graph, source_context),
                        expected_claim_terms=cls._claim_grounding_expected_terms(detail, graph),
                        forbidden_unsupported_claim_terms=cls._forbidden_unsupported_claim_terms(graph),
                        domain=domain,
                        scenario=scenario,
                        severity=severity,
                        case_weight=weight,
                    )
                )
            if graph is not None and (reasons & {'compiler_validity_gap', 'repair_failure', 'grounding_review', 'claim_grounding_review'} or cls._has_compiler_signal(graph)):
                broken = cls._broken_graph_from_review_graph(graph, reasons)
                derived.compiler_cases.append(
                    CompilerRepairEvalCase(
                        graph=broken,
                        expected_repair_terms=cls._expected_repair_terms(broken, reasons),
                        domain=domain,
                        scenario=scenario,
                        severity=severity,
                        case_weight=weight,
                    )
                )
        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(derived.case_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
            derived.output_path = str(output)
        return derived

    @staticmethod
    def _graph_from_review_detail(detail: dict[str, Any]) -> StructuredMeaningGraph | None:
        payload = detail.get('graph')
        if not isinstance(payload, dict):
            return None
        if not payload.get('query') or not payload.get('intent'):
            return None
        return StructuredMeaningGraph.from_dict(payload)

    @staticmethod
    def _review_eval_query(detail: dict[str, Any]) -> str:
        query = str(detail.get('query', '')).strip()
        context = str(detail.get('context_text', '')).strip()
        if not context:
            return query
        domain = str(detail.get('domain', 'general')).strip() or 'general'
        scenario = str(detail.get('scenario', 'qa')).strip() or 'qa'
        return (
            f'[Domain: {domain}]\n'
            f'[Scenario: {scenario}]\n'
            f'SOP Context:\n{context}\n\n'
            f'Question:\n{query}'
        )

    @staticmethod
    def _review_source_context(detail: dict[str, Any], graph: StructuredMeaningGraph | None) -> str:
        context = str(detail.get('context_text', '')).strip()
        if context:
            return context
        if graph is not None and graph.source_context.strip():
            return graph.source_context
        return ''

    @classmethod
    def _grounding_expected_terms(cls, detail: dict[str, Any], graph: StructuredMeaningGraph | None, source_context: str) -> list[str]:
        candidates: list[str] = []
        if graph is not None:
            evidence_lookup = {node.id: node for node in graph.nodes if node.kind == 'evidence'}
            for edge in graph.edges:
                if edge.source != 'question' or edge.relation != 'GROUNDED_BY':
                    continue
                node = evidence_lookup.get(edge.target)
                if node is None:
                    continue
                text = str(node.attributes.get('text', node.label)).strip()
                if text:
                    candidates.append(text)
            for result in graph.symbolic_results:
                candidates.extend(result.evidence[:2])
                candidates.extend(result.equations[:1])
        if source_context.strip():
            candidates.extend(line.strip() for line in source_context.splitlines() if line.strip())
        answer_text = str(detail.get('answer_text', '')).strip()
        if answer_text:
            candidates.append(answer_text)
        return cls._extract_text_phrases(candidates)

    @classmethod
    def _claim_grounding_expected_terms(cls, detail: dict[str, Any], graph: StructuredMeaningGraph | None) -> list[str]:
        candidates: list[str] = []
        answer_text = str(detail.get('answer_text', '')).strip()
        if answer_text:
            candidates.append(answer_text)
        if graph is not None and graph.operator_execution is not None:
            candidates.extend(item.claim for item in graph.operator_execution.claim_groundings if item.grounded)
        if graph is not None and not candidates:
            candidates.extend(result.answer for result in graph.symbolic_results if result.domain == 'document_grounding' and result.answer.strip())
        return cls._extract_text_phrases(candidates)

    @classmethod
    def _forbidden_unsupported_claim_terms(cls, graph: StructuredMeaningGraph | None) -> list[str]:
        if graph is None or graph.operator_execution is None:
            return []
        candidates = [item.claim for item in graph.operator_execution.claim_groundings if not item.grounded and item.claim.strip()]
        return cls._extract_text_phrases(candidates)

    @staticmethod
    def _extract_text_phrases(candidates: Sequence[str], limit: int = 2) -> list[str]:
        phrases: list[str] = []
        seen: set[str] = set()
        for text in candidates:
            normalized = re.sub(r'\s+', ' ', str(text).strip()).strip(' -:|')
            if len(normalized) < 6:
                continue
            for piece in re.split(r'[.;|]\s*', normalized):
                phrase = piece.strip().lower()
                if len(phrase) < 6 or phrase in seen:
                    continue
                seen.add(phrase)
                phrases.append(phrase)
                tokens = re.findall(r"[^\W_]+", phrase)
                short_phrase = ' '.join(tokens[:4]).strip()
                if len(short_phrase) >= 6 and short_phrase not in seen:
                    seen.add(short_phrase)
                    phrases.append(short_phrase)
                if len(phrases) >= limit:
                    return phrases[:limit]
        return phrases[:limit]

    @staticmethod
    def _has_grounding_signal(graph: StructuredMeaningGraph) -> bool:
        report = graph.operator_execution
        return any(edge.source == 'question' and edge.relation == 'GROUNDED_BY' for edge in graph.edges) or any(
            result.domain == 'document_grounding' for result in graph.symbolic_results
        ) or (report is not None and bool(report.claim_groundings))

    @staticmethod
    def _has_compiler_signal(graph: StructuredMeaningGraph) -> bool:
        report = graph.operator_execution
        return report is not None and bool(report.compiler_findings or report.counterexample_repairs)

    @staticmethod
    def _broken_graph_from_review_graph(graph: StructuredMeaningGraph, reasons: set[str]) -> StructuredMeaningGraph:
        broken = StructuredMeaningGraph.from_dict(graph.model_dump())
        broken.operator_execution = None
        broken.audit_trace = []
        if 'grounding_review' in reasons and 'claim_grounding_review' not in reasons:
            broken.edges = [
                edge for edge in broken.edges
                if not (edge.source == 'question' and edge.relation == 'GROUNDED_BY')
            ]
        return broken

    @staticmethod
    def _expected_repair_terms(graph: StructuredMeaningGraph, reasons: set[str]) -> list[str]:
        if 'claim_grounding_review' in reasons:
            return ['trim_unsupported_claims']
        if 'grounding_review' in reasons:
            if graph.source_context.strip():
                return ['document']
            if any(node.id == 'visual_scene' or node.kind == 'scene' for node in graph.nodes):
                return ['visual']
        return []



































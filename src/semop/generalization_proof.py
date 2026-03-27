from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .corpus_store import CorpusMemoryStore
from .operating_policies import infer_operating_domain
from .operator_evolution import OperatorTransferEvalCase
from .premise_eval import HiddenPremiseEvalCase
from .review_queue import ReviewQueueStore
from .structures import StructuredMeaningGraph
from .understanding_eval import SemOpUnderstandingEvaluator
from .unified_benchmark import (
    AnalogyEvalCase,
    BenchmarkGatedContinuousTrainer,
    CompilerRepairEvalCase,
    GroundedExplanationEvalCase,
)
from .vlso.eval import VlsoEvalCase


@dataclass
class GeneralizationProofRound:
    round_index: int
    source: str = ""
    split: str = "train"
    output_dir: str = ""
    corpus_graph_count: int = 0
    approved_review_count: int = 0
    promoted_review_count: int = 0
    trained_on_graphs: int = 0
    augmented_graph_count: int = 0
    gate_accepted: bool = False
    benchmark: dict[str, Any] = field(default_factory=dict)
    gate: dict[str, Any] = field(default_factory=dict)
    understanding: dict[str, Any] = field(default_factory=dict)
    metric_deltas: dict[str, float] = field(default_factory=dict)
    domains_seen: list[str] = field(default_factory=list)
    scenarios_seen: list[str] = field(default_factory=list)
    derived_case_counts: dict[str, int] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneralizationProofEvidence:
    learned_generalization_score: float = 0.0
    reviewed_corpus_growth_score: float = 0.0
    multimodal_transfer_score: float = 0.0
    domain_coverage_score: float = 0.0
    strong_model_score: float = 0.0
    learned_generalization_verified: bool = False
    reviewed_corpus_growth_verified: bool = False
    unseen_multimodal_transfer_verified: bool = False
    domain_coverage_verified: bool = False
    strong_model_ready: bool = False
    headline: str = ""
    strengths: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneralizationGoalAxis:
    key: str
    label: str
    score: float = 0.0
    target: float = 0.0
    verified: bool = False
    gap: float = 0.0
    summary: str = ""
    next_step: str = ""

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneralizationGoalTracker:
    readiness_percent: float = 0.0
    ready_axes: int = 0
    total_axes: int = 0
    priority_focus: str = ""
    completed_items: list[str] = field(default_factory=list)
    remaining_items: list[str] = field(default_factory=list)
    domains_seen: list[str] = field(default_factory=list)
    scenarios_seen: list[str] = field(default_factory=list)
    axes: list[GeneralizationGoalAxis] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneralizationProofSummary:
    rounds: list[GeneralizationProofRound] = field(default_factory=list)
    accepted_rounds: int = 0
    final_gate_accepted: bool = False
    best_round_index: int = 0
    metric_trends: dict[str, list[float]] = field(default_factory=dict)
    corpus_growth: dict[str, list[int]] = field(default_factory=dict)
    evidence: GeneralizationProofEvidence = field(default_factory=GeneralizationProofEvidence)
    goal_tracker: GeneralizationGoalTracker = field(default_factory=GeneralizationGoalTracker)
    report_path: str = ""

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class GeneralizationProofHarness:
    def __init__(self, mode: str = "heuristic") -> None:
        self.mode = mode
        self.gated = BenchmarkGatedContinuousTrainer(mode=mode)
        self.understanding = SemOpUnderstandingEvaluator()

    def run(
        self,
        store_path: str | Path,
        output_dir: str | Path,
        *,
        source: str | None = None,
        source_schedule: Sequence[str] | None = None,
        split: str = "train",
        split_schedule: Sequence[str] | None = None,
        rounds: int = 3,
        review_store_path: str | Path | None = None,
        approved_queries_only: bool = False,
        hidden_premise_cases: Sequence[HiddenPremiseEvalCase] | None = None,
        transfer_cases: Sequence[OperatorTransferEvalCase] | None = None,
        analogy_cases: Sequence[AnalogyEvalCase] | None = None,
        grounding_cases: Sequence[GroundedExplanationEvalCase] | None = None,
        compiler_cases: Sequence[CompilerRepairEvalCase] | None = None,
        vlso_cases: Sequence[VlsoEvalCase] | None = None,
        vlso_real_image_cases: Sequence[VlsoEvalCase] | None = None,
        baseline_summary_path: str | Path | None = None,
        operating_domain: str | None = None,
        benchmark_corpus_path: str | Path | None = None,
        report_path: str | Path | None = None,
        round_setup_callback: Any = None,
        progress_callback: Any = None,
    ) -> GeneralizationProofSummary:
        resolved_domain = operating_domain or infer_operating_domain(review_store_path)
        total_rounds = max(1, int(rounds))
        sources = list(source_schedule or [])
        if not sources:
            sources = [str(source or "").strip()] * total_rounds
        while len(sources) < total_rounds:
            sources.append(sources[-1] if sources else str(source or "").strip())
        splits = list(split_schedule or [])
        if not splits:
            splits = [split] * total_rounds
        while len(splits) < total_rounds:
            splits.append(splits[-1] if splits else split)

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        proof_root = root / "generalization_proof"
        proof_root.mkdir(parents=True, exist_ok=True)

        current_baseline = str(baseline_summary_path) if baseline_summary_path else ""
        rounds_out: list[GeneralizationProofRound] = []
        previous_metrics: dict[str, float] | None = None
        review_store = ReviewQueueStore(review_store_path) if review_store_path else None
        store = CorpusMemoryStore(store_path)
        for round_index in range(1, total_rounds + 1):
            source_value = str(sources[round_index - 1] or "").strip()
            split_value = str(splits[round_index - 1] or split).strip() or split
            round_dir = proof_root / f"round_{round_index:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            if round_setup_callback is not None:
                round_setup_callback(round_index, source_value, split_value)
            if progress_callback:
                progress_callback(
                    min(0.92, ((round_index - 1) / float(total_rounds)) + 0.05),
                    f"Running generalization proof round {round_index}/{total_rounds}.",
                )

            source_graphs = self._fetch_graphs(store, split_value, source_value or None)
            derivation_graphs = self._fetch_graphs(store, split_value, None) or source_graphs
            resolved_hidden_cases = list(hidden_premise_cases or self._derive_hidden_premise_cases(derivation_graphs))
            resolved_transfer_cases = list(transfer_cases or self._derive_transfer_cases(derivation_graphs))
            resolved_analogy_cases = list(analogy_cases or self._derive_analogy_cases(derivation_graphs))
            resolved_grounding_cases = list(grounding_cases or self._derive_grounding_cases(derivation_graphs))
            resolved_compiler_cases = list(compiler_cases or self._derive_compiler_cases(derivation_graphs))

            gated_summary = self.gated.train_evaluate_and_gate(
                store_path,
                round_dir,
                source=source_value or None,
                split=split_value,
                review_store_path=review_store_path,
                approved_queries_only=approved_queries_only,
                hidden_premise_cases=resolved_hidden_cases,
                transfer_cases=resolved_transfer_cases,
                analogy_cases=resolved_analogy_cases,
                grounding_cases=resolved_grounding_cases,
                compiler_cases=resolved_compiler_cases,
                vlso_cases=vlso_cases,
                baseline_summary_path=current_baseline or None,
                operating_domain=resolved_domain,
                benchmark_corpus_path=benchmark_corpus_path,
            )
            understanding_summary = self.understanding.evaluate(
                hidden_premise_cases=resolved_hidden_cases,
                cp_examples=None,
                cp_hidden_examples=None,
                vlso_cases=list(vlso_cases or []) or None,
                vlso_real_image_cases=list(vlso_real_image_cases or []) or None,
                cp_mode=self.mode,
            )
            benchmark_payload = gated_summary.benchmark.model_dump()
            understanding_payload = understanding_summary.model_dump()
            combined_metrics = self._combined_metric_view(benchmark_payload, understanding_payload)
            deltas = {
                key: round(combined_metrics.get(key, 0.0) - float(previous_metrics.get(key, 0.0)), 4)
                for key in combined_metrics
            } if previous_metrics else {}
            graph_count = len(source_graphs)
            approved_reviews = review_store.fetch_stats().get("approved", 0) if review_store is not None else 0
            domains_seen = sorted({self._normalize_domain(getattr(graph, 'domain', 'general')) for graph in derivation_graphs})
            scenarios_seen = sorted({self._normalize_scenario(getattr(graph, 'scenario', 'qa')) for graph in derivation_graphs})
            rounds_out.append(
                GeneralizationProofRound(
                    round_index=round_index,
                    source=source_value,
                    split=split_value,
                    output_dir=str(round_dir),
                    corpus_graph_count=graph_count,
                    approved_review_count=approved_reviews,
                    promoted_review_count=int(gated_summary.training.promoted_review_graph_count),
                    trained_on_graphs=int(gated_summary.training.trained_on_graphs),
                    augmented_graph_count=int(gated_summary.training.augmented_graph_count),
                    gate_accepted=bool(gated_summary.gate.accepted),
                    benchmark=benchmark_payload,
                    gate=gated_summary.gate.model_dump(),
                    understanding=understanding_payload,
                    metric_deltas=deltas,
                    domains_seen=domains_seen,
                    scenarios_seen=scenarios_seen,
                    derived_case_counts={
                        "hidden_premise": len(resolved_hidden_cases),
                        "transfer": len(resolved_transfer_cases),
                        "analogy": len(resolved_analogy_cases),
                        "grounding": len(resolved_grounding_cases),
                        "compiler": len(resolved_compiler_cases),
                        "vlso": len(list(vlso_cases or [])),
                        "vlso_real": len(list(vlso_real_image_cases or [])),
                    },
                )
            )
            previous_metrics = combined_metrics
            accepted_path = round_dir / "accepted_benchmark_summary.json"
            if accepted_path.exists():
                current_baseline = str(accepted_path)
            elif not current_baseline:
                current_baseline = str(round_dir / "benchmark_gate.json")

        summary = self._summarize(rounds_out)
        summary.report_path = str(report_path or (proof_root / "generalization_proof_report.json"))
        if progress_callback:
            progress_callback(0.97, "Writing the generalization proof report.")
        Path(summary.report_path).write_text(
            json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    @staticmethod
    def _fetch_graphs(
        store: CorpusMemoryStore,
        split_value: str,
        source_value: str | None,
        limit: int = 640,
    ) -> list[StructuredMeaningGraph]:
        rows = list(store.fetch_graphs(split=split_value or None, source=source_value or None))
        return rows[:limit]

    @classmethod
    def _derive_hidden_premise_cases(
        cls,
        graphs: Sequence[StructuredMeaningGraph],
        limit: int = 12,
    ) -> list[HiddenPremiseEvalCase]:
        cases: list[HiddenPremiseEvalCase] = []
        for graph in graphs:
            if not graph.query.strip():
                continue
            if not graph.hidden_goals and not graph.required_premises:
                continue
            risky_actions = [
                check.action
                for check in graph.goal_preservation_checks
                if check.status in {"risk_high", "invalid"} and check.action
            ]
            support_operators = sorted({
                item.operator_name
                for item in graph.operator_decompositions
                if item.operator_name
            }) or None
            cases.append(
                HiddenPremiseEvalCase(
                    query=graph.query,
                    expected_hidden_goals=cls._unique(graph.hidden_goals),
                    expected_required_premises=cls._unique(graph.required_premises),
                    expected_satisfied_premises=cls._unique(graph.satisfied_premises),
                    expected_missing_premises=cls._unique(graph.missing_premises),
                    expected_risky_actions=cls._unique(risky_actions),
                    forbidden_premises=[],
                    expected_clarification_needed=bool(graph.clarification_needed),
                    expected_clarification_score=float(graph.clarification_score),
                    expected_support_operators=support_operators,
                    domain=cls._normalize_domain(getattr(graph, 'domain', 'general')),
                    scenario=cls._normalize_scenario(getattr(graph, 'scenario', 'qa')),
                    severity="medium",
                    case_weight=1.0,
                )
            )
            if len(cases) >= limit:
                break
        return cases

    @classmethod
    def _derive_transfer_cases(
        cls,
        graphs: Sequence[StructuredMeaningGraph],
        limit: int = 12,
    ) -> list[OperatorTransferEvalCase]:
        by_domain: dict[str, list[StructuredMeaningGraph]] = {}
        for graph in graphs:
            operator_names = cls._graph_operator_names(graph)
            if not graph.query.strip() or not operator_names:
                continue
            by_domain.setdefault(cls._normalize_domain(getattr(graph, 'domain', 'general')), []).append(graph)
        if not by_domain:
            return []
        domains = sorted(by_domain)
        primary_domain = domains[0]
        cases: list[OperatorTransferEvalCase] = []
        for graph in by_domain.get(primary_domain, [])[:2]:
            cases.append(
                OperatorTransferEvalCase(
                    query=graph.query,
                    domain=primary_domain,
                    expected_operator_names=cls._graph_operator_names(graph),
                    split="train",
                )
            )
        for domain in domains[1:]:
            for graph in by_domain.get(domain, [])[: max(1, limit - len(cases))]:
                cases.append(
                    OperatorTransferEvalCase(
                        query=graph.query,
                        domain=domain,
                        expected_operator_names=cls._graph_operator_names(graph),
                        split="test",
                    )
                )
                if len(cases) >= limit:
                    return cases
        if len(cases) < 2:
            fallback_graphs = [graph for graphs_in_domain in by_domain.values() for graph in graphs_in_domain]
            for index, graph in enumerate(fallback_graphs[:limit]):
                cases.append(
                    OperatorTransferEvalCase(
                        query=graph.query,
                        domain=cls._normalize_domain(getattr(graph, 'domain', 'general')),
                        expected_operator_names=cls._graph_operator_names(graph),
                        split="train" if index == 0 else "test",
                    )
                )
        return cases[:limit]

    @classmethod
    def _derive_analogy_cases(
        cls,
        graphs: Sequence[StructuredMeaningGraph],
        limit: int = 10,
    ) -> list[AnalogyEvalCase]:
        cases: list[AnalogyEvalCase] = []
        for graph in graphs:
            if not graph.query.strip():
                continue
            anchor_requirements = [item for item in graph.required_premises if item]
            anchor_goals = [item for item in graph.hidden_goals if item]
            if not anchor_requirements and not anchor_goals:
                continue
            anchor = set(anchor_requirements + anchor_goals)
            similar_graphs: list[StructuredMeaningGraph] = []
            for other in graphs:
                if other.query == graph.query:
                    continue
                overlap = anchor & set(other.required_premises + other.hidden_goals)
                if not overlap:
                    continue
                similar_graphs.append(StructuredMeaningGraph.from_dict(other.model_dump()))
                if len(similar_graphs) >= 2:
                    break
            if not similar_graphs:
                continue
            cases.append(
                AnalogyEvalCase(
                    query=graph.query,
                    similar_graphs=similar_graphs,
                    expected_requirement=anchor_requirements[0] if anchor_requirements else "",
                )
            )
            if len(cases) >= limit:
                break
        return cases

    @classmethod
    def _derive_grounding_cases(
        cls,
        graphs: Sequence[StructuredMeaningGraph],
        limit: int = 10,
    ) -> list[GroundedExplanationEvalCase]:
        cases: list[GroundedExplanationEvalCase] = []
        for graph in graphs:
            source_context = str(getattr(graph, 'source_context', '')).strip()
            if not graph.query.strip() or not source_context:
                continue
            grounded_claims: list[str] = []
            unsupported_claims: list[str] = []
            if graph.operator_execution is not None:
                for item in graph.operator_execution.claim_groundings:
                    claim = cls._clean_grounding_term(item.claim, graph.query)
                    if not claim:
                        continue
                    if item.grounded:
                        grounded_claims.append(claim)
                    else:
                        unsupported_claims.append(claim)
            evidence_terms: list[str] = []
            context_terms = cls._context_grounding_terms(source_context, graph.query)
            evidence_terms.extend(context_terms[:3])
            for result in graph.symbolic_results:
                if result.domain != "document_grounding":
                    continue
                for evidence in result.evidence:
                    snippet = cls._clean_grounding_term(evidence, graph.query, max_words=5)
                    if snippet and snippet not in evidence_terms:
                        evidence_terms.append(snippet)
                if not grounded_claims and result.answer.strip():
                    for chunk in re.split(r"and|[.;]\s*", result.answer):
                        snippet = cls._clean_grounding_term(chunk, graph.query)
                        if snippet and snippet not in grounded_claims:
                            grounded_claims.append(snippet)
            if not evidence_terms:
                for node in graph.nodes:
                    if node.kind != "evidence":
                        continue
                    snippet = cls._clean_grounding_term(str(node.attributes.get("text", node.label)), graph.query)
                    if snippet and snippet not in evidence_terms:
                        evidence_terms.append(snippet)
            if not grounded_claims:
                grounded_claims.extend(context_terms[:2])
            grounded_claims = cls._unique(grounded_claims)
            evidence_terms = cls._unique(evidence_terms)
            unsupported_claims = cls._unique(unsupported_claims)
            if not grounded_claims and not evidence_terms:
                continue
            cases.append(
                GroundedExplanationEvalCase(
                    query=graph.query,
                    source_context=source_context,
                    expected_evidence_terms=evidence_terms[:3],
                    expected_claim_terms=grounded_claims[:3],
                    forbidden_unsupported_claim_terms=unsupported_claims[:2],
                    domain=cls._normalize_domain(getattr(graph, 'domain', 'general')),
                    scenario=cls._normalize_scenario(getattr(graph, 'scenario', 'qa')),
                    severity="medium",
                    case_weight=1.0,
                )
            )
            if len(cases) >= limit:
                break
        return cases

    @classmethod
    def _derive_compiler_cases(
        cls,
        graphs: Sequence[StructuredMeaningGraph],
        limit: int = 10,
    ) -> list[CompilerRepairEvalCase]:
        cases: list[CompilerRepairEvalCase] = []
        for graph in graphs:
            if not graph.query.strip():
                continue
            expected_terms: list[str] = []
            if graph.hidden_goals and (graph.required_premises or graph.satisfied_premises or graph.missing_premises):
                expected_terms.append("goal_preservation")
            if str(getattr(graph, 'source_context', '')).strip():
                expected_terms.append("document_grounding")
            if getattr(graph, 'visual_input', None) is not None:
                expected_terms.append("visual_grounding")
            if not expected_terms:
                continue
            cases.append(
                CompilerRepairEvalCase(
                    graph=StructuredMeaningGraph.from_dict(graph.model_dump()),
                    expected_repair_terms=expected_terms,
                    domain=cls._normalize_domain(getattr(graph, 'domain', 'general')),
                    scenario=cls._normalize_scenario(getattr(graph, 'scenario', 'qa')),
                    severity="medium",
                    case_weight=1.0,
                )
            )
            if len(cases) >= limit:
                break
        return cases

    @classmethod
    def _context_grounding_terms(cls, source_context: str, query: str) -> list[str]:
        terms: list[str] = []
        for chunk in re.split(r'(?<=[.!?])\s+|,\s*|;\s*', str(source_context or '').strip()):
            snippet = cls._clean_grounding_term(chunk, query)
            if snippet and snippet not in terms:
                terms.append(snippet)
        return terms

    @classmethod
    def _clean_grounding_term(cls, text: str, query: str, max_words: int = 6) -> str:
        cleaned = str(text or '').strip()
        if not cleaned:
            return ''
        cleaned = re.sub(r'^(?:(?:Question|SOP Context|Document evidence points to):\s*)+', '', cleaned, flags=re.I)
        cleaned = ' '.join(cleaned.split())
        if not cleaned:
            return ''
        query_lower = ' '.join(str(query or '').split()).lower()
        lowered = cleaned.lower()
        if lowered == query_lower or lowered.startswith('question:'):
            return ''
        if len(lowered) < 10:
            return ''
        snippet = cls._snippet(cleaned, max_words=max_words)
        if snippet.lower() == query_lower:
            return ''
        return snippet

    @staticmethod
    def _snippet(text: str, max_words: int = 6, max_chars: int = 72) -> str:
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return ""
        words = cleaned.split()
        short = " ".join(words[:max_words])
        if len(short) > max_chars:
            return short[: max_chars - 3].rstrip() + "..."
        return short

    @staticmethod
    def _graph_operator_names(graph: StructuredMeaningGraph) -> list[str]:
        names = [item.operator_name for item in graph.operator_decompositions if item.operator_name]
        if names:
            return sorted(dict.fromkeys(names))
        induced = [item.name for item in graph.induced_operators if item.name]
        return sorted(dict.fromkeys(induced))

    @staticmethod
    def _unique(values: Sequence[str]) -> list[str]:
        ordered: list[str] = []
        for value in values:
            item = str(value or "").strip()
            if item and item not in ordered:
                ordered.append(item)
        return ordered

    @staticmethod
    def _normalize_domain(value: str | None) -> str:
        return str(value or "general").strip() or "general"

    @staticmethod
    def _normalize_scenario(value: str | None) -> str:
        return str(value or "qa").strip() or "qa"

    @staticmethod
    def _combined_metric_view(benchmark: dict[str, Any], understanding: dict[str, Any]) -> dict[str, float]:
        progress = understanding.get("progress", {}) if isinstance(understanding.get("progress"), dict) else {}
        snapshot = understanding.get("snapshot", {}) if isinstance(understanding.get("snapshot"), dict) else {}
        vlso_grounded = snapshot.get("vlso_grounded", {}) if isinstance(snapshot.get("vlso_grounded"), dict) else {}
        vlso_real = snapshot.get("vlso_real_image", {}) if isinstance(snapshot.get("vlso_real_image"), dict) else {}
        return {
            "unseen_transfer": float(benchmark.get("unseen_transfer", 0.0) or 0.0),
            "analogy_usefulness": float(benchmark.get("analogy_usefulness", 0.0) or 0.0),
            "compiler_validity": float(benchmark.get("compiler_validity", 0.0) or 0.0),
            "grounded_explanation_fidelity": float(benchmark.get("grounded_explanation_fidelity", 0.0) or 0.0),
            "repair_success_rate": float(benchmark.get("repair_success_rate", 0.0) or 0.0),
            "robust_understanding": float(progress.get("robust_general_intelligence_overall", 0.0) or 0.0),
            "vlso_grounded_accuracy": float(vlso_grounded.get("grounded_answer_accuracy", 0.0) or 0.0),
            "vlso_real_grounding": float(vlso_real.get("grounded_answer_accuracy", 0.0) or 0.0),
        }

    @classmethod
    def _summarize(cls, rounds: Sequence[GeneralizationProofRound]) -> GeneralizationProofSummary:
        if not rounds:
            return GeneralizationProofSummary()
        metric_names = [
            "unseen_transfer",
            "analogy_usefulness",
            "compiler_validity",
            "grounded_explanation_fidelity",
            "repair_success_rate",
            "robust_understanding",
            "vlso_grounded_accuracy",
            "vlso_real_grounding",
        ]
        metric_trends = {
            name: [cls._combined_metric_view(round_.benchmark, round_.understanding).get(name, 0.0) for round_ in rounds]
            for name in metric_names
        }
        corpus_growth = {
            "corpus_graph_count": [int(round_.corpus_graph_count) for round_ in rounds],
            "approved_review_count": [int(round_.approved_review_count) for round_ in rounds],
            "promoted_review_count": [int(round_.promoted_review_count) for round_ in rounds],
        }
        best_round = max(
            rounds,
            key=lambda round_: cls._round_priority(cls._combined_metric_view(round_.benchmark, round_.understanding), round_.gate_accepted),
        )
        evidence = cls._build_evidence(rounds, metric_trends, corpus_growth)
        goal_tracker = cls._build_goal_tracker(rounds, evidence)
        return GeneralizationProofSummary(
            rounds=list(rounds),
            accepted_rounds=sum(1 for round_ in rounds if round_.gate_accepted),
            final_gate_accepted=bool(rounds[-1].gate_accepted),
            best_round_index=int(best_round.round_index),
            metric_trends=metric_trends,
            corpus_growth=corpus_growth,
            evidence=evidence,
            goal_tracker=goal_tracker,
        )

    @classmethod
    def _build_evidence(
        cls,
        rounds: Sequence[GeneralizationProofRound],
        metric_trends: dict[str, list[float]],
        corpus_growth: dict[str, list[int]],
    ) -> GeneralizationProofEvidence:
        final_metrics = cls._combined_metric_view(rounds[-1].benchmark, rounds[-1].understanding)
        first_metrics = cls._combined_metric_view(rounds[0].benchmark, rounds[0].understanding)
        learned_generalization_score = round(cls._average([
            final_metrics["unseen_transfer"],
            final_metrics["analogy_usefulness"],
            final_metrics["compiler_validity"],
            final_metrics["grounded_explanation_fidelity"],
            final_metrics["repair_success_rate"],
        ]), 4)
        reviewed_corpus_growth_score = round(cls._reviewed_corpus_growth_score(metric_trends, corpus_growth), 4)
        multimodal_transfer_score = round(cls._average([
            final_metrics["unseen_transfer"],
            final_metrics["vlso_grounded_accuracy"],
            max(final_metrics["vlso_real_grounding"], final_metrics["grounded_explanation_fidelity"]),
            final_metrics["robust_understanding"],
        ]), 4)
        domain_coverage_score = round(cls._domain_coverage_score(rounds), 4)
        strong_model_score = round(cls._average([
            learned_generalization_score,
            reviewed_corpus_growth_score,
            multimodal_transfer_score,
            domain_coverage_score,
            final_metrics["robust_understanding"],
        ]), 4)

        learned_generalization_verified = learned_generalization_score >= 0.6 and bool(rounds[-1].gate_accepted)
        reviewed_corpus_growth_verified = reviewed_corpus_growth_score >= 0.55
        unseen_multimodal_transfer_verified = multimodal_transfer_score >= 0.5
        domain_coverage_verified = domain_coverage_score >= 0.6
        strong_model_ready = strong_model_score >= 0.64 and bool(rounds[-1].gate_accepted)

        strengths: list[str] = []
        risks: list[str] = []
        next_steps: list[str] = []

        if learned_generalization_verified:
            strengths.append("Repeated benchmark-gated rounds now show a usable learned-generalization signal.")
        else:
            risks.append("Learned generalization is still below the evidence threshold on the current proof loop.")
            next_steps.append("Grow reviewed traces and hold-out transfer cases until unseen transfer and grounding move together.")

        if reviewed_corpus_growth_verified:
            strengths.append("Metric trends improved while the approved review corpus and stored graph count grew.")
        else:
            risks.append("Corpus growth is not yet translating into a stable benchmark lift across rounds.")
            next_steps.append("Add more approved review traces per round so the continuous-learning bundle has real new evidence.")

        if unseen_multimodal_transfer_verified:
            strengths.append("Text transfer and visual grounding are moving together instead of improving in isolation.")
        else:
            risks.append("Multimodal transfer is still the biggest gap between the symbolic stack and real scenes.")
            next_steps.append("Seed more reviewed visual graphs and keep the real-image VLSO benchmark in the proof loop.")

        if domain_coverage_verified:
            strengths.append("The proof loop now covers several domains and scenarios instead of a single narrow slice.")
        else:
            risks.append("Domain coverage is still thin, so broad general-model claims would be premature.")
            next_steps.append("Add more reviewed domains and scenarios so the proof loop covers wider unseen slices.")

        if strong_model_ready:
            strengths.append("The current bundle clears a stronger evidence bar than the basic operating gate alone.")
        else:
            risks.append("The current system is still a research stack, not a strongly validated general model yet.")
            next_steps.append("Push the robust-understanding and real-image grounding axes higher before calling the model broadly general.")

        overall_delta = cls._average([
            max(0.0, final_metrics[key] - first_metrics.get(key, 0.0))
            for key in (
                "unseen_transfer",
                "compiler_validity",
                "grounded_explanation_fidelity",
                "repair_success_rate",
                "robust_understanding",
            )
        ])
        if strong_model_ready:
            headline = "The proof loop now shows a stronger, repeatable generalization signal across symbolic and visual checks."
        elif overall_delta >= 0.05:
            headline = "The stack is improving across repeated rounds, but the stronger general-model evidence bar is not cleared yet."
        else:
            headline = "The proof loop is running, but it is not yet demonstrating strong enough improvement to claim broad generalization."

        if not next_steps:
            next_steps.append("Keep running reviewed corpus growth rounds and only keep bundles that raise the proof scores.")

        return GeneralizationProofEvidence(
            learned_generalization_score=learned_generalization_score,
            reviewed_corpus_growth_score=reviewed_corpus_growth_score,
            multimodal_transfer_score=multimodal_transfer_score,
            domain_coverage_score=domain_coverage_score,
            strong_model_score=strong_model_score,
            learned_generalization_verified=learned_generalization_verified,
            reviewed_corpus_growth_verified=reviewed_corpus_growth_verified,
            unseen_multimodal_transfer_verified=unseen_multimodal_transfer_verified,
            domain_coverage_verified=domain_coverage_verified,
            strong_model_ready=strong_model_ready,
            headline=headline,
            strengths=strengths,
            risks=risks,
            next_steps=next_steps,
        )

    @classmethod
    def _build_goal_tracker(
        cls,
        rounds: Sequence[GeneralizationProofRound],
        evidence: GeneralizationProofEvidence,
    ) -> GeneralizationGoalTracker:
        domains_seen = sorted({domain for round_ in rounds for domain in round_.domains_seen if domain})
        scenarios_seen = sorted({scenario for round_ in rounds for scenario in round_.scenarios_seen if scenario})
        axes = [
            GeneralizationGoalAxis(
                key="learned_generalization",
                label="Learned generalization verification",
                score=evidence.learned_generalization_score,
                target=0.6,
                verified=evidence.learned_generalization_verified,
                gap=round(max(0.0, 0.6 - evidence.learned_generalization_score), 4),
                summary="Repeated gated rounds should keep unseen transfer, grounding, compiler validity, and repair above the proof bar.",
                next_step="Add more approved reviewed traces and hold-out transfer cases until unseen transfer rises above the threshold.",
            ),
            GeneralizationGoalAxis(
                key="reviewed_corpus_growth",
                label="Reviewed corpus improvement proof",
                score=evidence.reviewed_corpus_growth_score,
                target=0.55,
                verified=evidence.reviewed_corpus_growth_verified,
                gap=round(max(0.0, 0.55 - evidence.reviewed_corpus_growth_score), 4),
                summary="The approved review corpus should grow while benchmark scores keep improving instead of drifting sideways.",
                next_step="Keep promoting only strong review traces and reject corpus growth that does not lift the benchmark.",
            ),
            GeneralizationGoalAxis(
                key="multimodal_transfer",
                label="Unseen multimodal transfer",
                score=evidence.multimodal_transfer_score,
                target=0.5,
                verified=evidence.unseen_multimodal_transfer_verified,
                gap=round(max(0.0, 0.5 - evidence.multimodal_transfer_score), 4),
                summary="Text transfer, grounded explanation, and visual grounding should move together on unseen slices.",
                next_step="Expand reviewed vision-language traces and keep real-image VLSO cases in the proof loop.",
            ),
            GeneralizationGoalAxis(
                key="domain_coverage",
                label="Wide domain coverage",
                score=evidence.domain_coverage_score,
                target=0.6,
                verified=evidence.domain_coverage_verified,
                gap=round(max(0.0, 0.6 - evidence.domain_coverage_score), 4),
                summary="The proof loop should cover several domains and scenarios before claiming broad operator-algebra intelligence.",
                next_step="Seed more reviewed domains and scenario slices into the same benchmark-gated curriculum.",
            ),
            GeneralizationGoalAxis(
                key="strong_model_evidence",
                label="Strong general-model evidence",
                score=evidence.strong_model_score,
                target=0.64,
                verified=evidence.strong_model_ready,
                gap=round(max(0.0, 0.64 - evidence.strong_model_score), 4),
                summary="The whole stack should clear a stronger proof bar than the operating gate alone.",
                next_step="Keep only bundles that improve both the gate and the stronger proof score across repeated rounds.",
            ),
        ]
        readiness_terms = [min(1.0, (axis.score / axis.target)) if axis.target > 0 else 1.0 for axis in axes]
        readiness_percent = round(cls._average(readiness_terms) * 100.0, 1)
        completed_items = [axis.label for axis in axes if axis.verified]
        remaining_items = [axis.label for axis in axes if not axis.verified]
        priority_focus = remaining_items[0] if remaining_items else "Keep growing reviewed traces while holding the proof score."
        return GeneralizationGoalTracker(
            readiness_percent=readiness_percent,
            ready_axes=sum(1 for axis in axes if axis.verified),
            total_axes=len(axes),
            priority_focus=priority_focus,
            completed_items=completed_items,
            remaining_items=remaining_items,
            domains_seen=domains_seen,
            scenarios_seen=scenarios_seen,
            axes=axes,
        )

    @staticmethod
    def _average(values: Sequence[float]) -> float:
        numeric = [float(value) for value in values]
        if not numeric:
            return 0.0
        return sum(numeric) / float(len(numeric))

    @classmethod
    def _reviewed_corpus_growth_score(
        cls,
        metric_trends: dict[str, list[float]],
        corpus_growth: dict[str, list[int]],
    ) -> float:
        first_graphs = int(corpus_growth.get("corpus_graph_count", [0])[0] if corpus_growth.get("corpus_graph_count") else 0)
        last_graphs = int(corpus_growth.get("corpus_graph_count", [0])[-1] if corpus_growth.get("corpus_graph_count") else 0)
        first_reviews = int(corpus_growth.get("approved_review_count", [0])[0] if corpus_growth.get("approved_review_count") else 0)
        last_reviews = int(corpus_growth.get("approved_review_count", [0])[-1] if corpus_growth.get("approved_review_count") else 0)
        graph_growth = min(1.0, max(0, last_graphs - first_graphs) / float(max(3, first_graphs or 1)))
        review_growth = min(1.0, max(0, last_reviews - first_reviews) / float(max(2, first_reviews or 1)))
        metric_growth = cls._average([
            min(1.0, max(0.0, values[-1] - values[0]) / 0.12)
            for values in metric_trends.values()
            if values
        ])
        return cls._average([graph_growth, review_growth, metric_growth])

    @classmethod
    def _domain_coverage_score(cls, rounds: Sequence[GeneralizationProofRound]) -> float:
        domains = sorted({domain for round_ in rounds for domain in round_.domains_seen if domain})
        scenarios = sorted({scenario for round_ in rounds for scenario in round_.scenarios_seen if scenario})
        domain_term = min(1.0, len(domains) / 4.0)
        scenario_term = min(1.0, len(scenarios) / 6.0)
        accepted_term = sum(1 for round_ in rounds if round_.gate_accepted) / float(max(1, len(rounds)))
        return cls._average([domain_term, scenario_term, accepted_term])

    @staticmethod
    def _round_priority(metrics: dict[str, float], gate_accepted: bool) -> float:
        return (
            (1.0 if gate_accepted else 0.0) * 2.0
            + float(metrics.get("robust_understanding", 0.0))
            + float(metrics.get("unseen_transfer", 0.0))
            + float(metrics.get("grounded_explanation_fidelity", 0.0))
        )

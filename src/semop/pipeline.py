from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Dict, List

from .commonsense_kb import concept_label, kb_relations_for_concept, normalize_concept, scripts_for_concept
from .context_chunks import split_context_into_chunks
from .unified_parser import LearnedUnifiedParser
from .context_understanding import OperatorContextAnalyzer
from .analogy_policy import AnalogyPolicyScorer
from .memory_analogies import AnalogicalMemoryBuilder
from .multimodal_alignment_memory import MultimodalAlignmentMemory
from .corpus_store import CorpusMemoryStore
from .emergent_operators import EmergentOperatorInducer
from .heuristic_extractors import HeuristicExtractor
from .llm_client import LocalLLMConfig, LocalTransformersExtractor
from .logical_grammar import LogicalPattern, LogicalPatternMatcher
from .operator_registry import TypedOperatorRegistry
from .operator_algebra import OperatorAlgebraLearner
from .retained_operator_algebra import RetainedOperatorAlgebra
from .operator_repair import OperatorRepairEngine
from .operator_runtime import compile_and_execute
from .premise_explorer import HiddenPremiseExplorer
from .semantic_operators import apply_many
from .structures import Edge, Node, OperatorCandidate, PlanStep, StructuredMeaningGraph
from .symbolic_reasoners import SymbolicReasoner
from .validation import validate_graph


class StructuredMeaningPipeline:
    def __init__(
        self,
        mode: str = "heuristic",
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        memory_store_path: str | None = None,
        memory_source: str | None = None,
        logical_weight_path: str | None = None,
        script_compatibility_model_path: str | None = None,
        analogy_policy_path: str | None = None,
        unified_parser_path: str | None = None,
        retained_algebra_path: str | None = None,
        repair_policy_path: str | None = None,
        repair_program_path: str | None = None,
        repair_utility_path: str | None = None,
        multimodal_alignment_path: str | None = None,
        operator_backend: str = "shadow",
    ):
        self.mode = mode
        self.heuristic = HeuristicExtractor()
        self.inducer = EmergentOperatorInducer()
        self.symbolic = SymbolicReasoner()
        self.memory_store = CorpusMemoryStore(memory_store_path) if memory_store_path else None
        self.memory_source = memory_source
        self.llm = LocalTransformersExtractor(LocalLLMConfig(model_id=model_id)) if mode == "llm" else None
        self._registry_cache: TypedOperatorRegistry | None = None
        self._logical_pattern_cache: List[LogicalPattern] | None = None
        self._logical_matcher = LogicalPatternMatcher()
        self._visual_parser = None
        self.premise_explorer = HiddenPremiseExplorer(
            memory_store=self.memory_store,
            memory_source=self.memory_source,
            compatibility_model_path=script_compatibility_model_path,
        )
        self.operator_algebra = OperatorAlgebraLearner()
        self.context_analyzer = OperatorContextAnalyzer()
        self.analogy_builder = AnalogicalMemoryBuilder()
        self.analogy_policy = AnalogyPolicyScorer(model_path=analogy_policy_path) if analogy_policy_path else None
        self.unified_parser = LearnedUnifiedParser(model_path=unified_parser_path) if unified_parser_path else None
        self.parser_dominance_threshold = self.unified_parser.model.dominance_threshold if self.unified_parser is not None else 0.0
        self.retained_algebra = RetainedOperatorAlgebra(model_path=retained_algebra_path) if retained_algebra_path else None
        self.repair_engine = OperatorRepairEngine(repair_policy_path=repair_policy_path, repair_program_path=repair_program_path, repair_utility_path=repair_utility_path)
        self.multimodal_alignment = MultimodalAlignmentMemory(model_path=multimodal_alignment_path) if multimodal_alignment_path else None
        self.retained_algebra_path = retained_algebra_path
        self.logical_weight_path = logical_weight_path
        self._logical_weight_cache: Dict[str, float] | None = None
        if operator_backend not in {"legacy", "shadow", "typed"}:
            raise ValueError("operator_backend must be legacy, shadow, or typed")
        self.operator_backend = operator_backend

    def run(self, query: str, source_context: str = "", visual_input: Any | None = None) -> StructuredMeaningGraph:
        similar_graphs = self._retrieve_similar_graphs(query, source_context=source_context, visual_input=visual_input)
        return self.run_with_memory_graphs(query, similar_graphs, source_context=source_context, visual_input=visual_input)

    def run_with_memory_graphs(
        self,
        query: str,
        similar_graphs: List[StructuredMeaningGraph] | None = None,
        source_context: str = "",
        visual_input: Any | None = None,
    ) -> StructuredMeaningGraph:
        similar_graphs = similar_graphs or []
        graph = self._prepare_graph(query, source_context=source_context, visual_input=visual_input)
        if self.multimodal_alignment is not None:
            graph = self.multimodal_alignment.enrich(graph)
        graph = self.premise_explorer.enrich(graph)
        graph = self._attach_premise_memory_hints(graph)
        graph = self._apply_logical_grammar_priors(graph)
        graph = self._attach_memory_hints(graph, similar_graphs)
        graph = self.inducer.induce(graph)
        graph = self._annotate_with_registry(graph)
        graph = self._apply_memory_priors(graph, similar_graphs)
        graph.analogical_matches = self.analogy_builder.build(graph, similar_graphs)
        if graph.analogical_matches:
            graph.audit_trace.append(f"analogical memory: selected {len(graph.analogical_matches)} structurally similar cases")
        graph = self._apply_analogy_guidance(graph)
        return self._finalize_graph(graph)

    def _prepare_graph(self, query: str, source_context: str = "", visual_input: Any | None = None) -> StructuredMeaningGraph:
        if self.mode == "heuristic":
            parser_bootstrap = None
            if self.unified_parser is not None:
                prediction = self.unified_parser.predict(query, source_context=source_context)
                if prediction.confidence >= self.parser_dominance_threshold:
                    parser_bootstrap = self.unified_parser.bootstrap_graph(query, source_context=source_context, prediction=prediction)
            graph = self._prepare_multimodal_graph(query, source_context=source_context, visual_input=visual_input, base_graph=parser_bootstrap)
            graph = self._apply_logical_relation_priors(graph)
            graph = self._apply_unified_parser_priors(graph)
            graph = self._attach_document_grounding_context(graph)
            return validate_graph(graph)
        try:
            raw = self.llm.extract(query) if self.llm is not None else None
        except Exception as exc:
            graph = self._prepare_multimodal_graph(query, source_context=source_context, visual_input=visual_input)
            graph.warnings.append(f"llm extraction failed; fallback to heuristic: {exc}")
            graph = self._apply_logical_relation_priors(graph)
            graph = self._apply_unified_parser_priors(graph)
            graph = self._attach_document_grounding_context(graph)
            return validate_graph(graph)

        graph = self._build_graph_from_llm(query, raw or {})
        graph = self._prepare_multimodal_graph(query, source_context=source_context, visual_input=visual_input, base_graph=graph)
        graph = self._apply_logical_relation_priors(graph)
        graph = self._apply_unified_parser_priors(graph)
        graph = self._attach_document_grounding_context(graph)
        return validate_graph(graph)

    def _finalize_graph(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        graph = self.symbolic.apply(graph)
        graph = self.operator_algebra.enrich(graph)
        if self.retained_algebra is not None:
            graph = self.retained_algebra.enrich(graph)
        graph = self.repair_engine.run(graph)
        if self.operator_backend != "legacy":
            from .kernel.adapters import TypedKernelBridge

            graph = TypedKernelBridge().run(
                graph, mode=self.operator_backend
            ).graph
        graph.context_frame = self.context_analyzer.analyze(graph)
        graph = self._annotate_with_registry(graph)
        graph = self._ensure_plan(graph)
        graph.induced_operators.sort(key=lambda candidate: (-candidate.confidence, candidate.name))
        graph = validate_graph(graph)
        self._store_runtime_memory(graph)
        return graph

    def _retrieve_similar_graphs(self, query: str, source_context: str = "", visual_input: Any | None = None) -> List[StructuredMeaningGraph]:
        if self.memory_store is None:
            return []
        probe_graph = self._build_memory_probe_graph(query, source_context=source_context, visual_input=visual_input)
        if self.analogy_policy is not None:
            graphs = self.memory_store.fetch_graphs(split="train", source=self.memory_source)
            return self.analogy_policy.rank_graphs(query, probe_graph, graphs, top_k=6)
        return self.memory_store.search_similar_graphs(query, split="train", source=self.memory_source, top_k=6, query_graph=probe_graph)

    def _build_memory_probe_graph(self, query: str, source_context: str = "", visual_input: Any | None = None) -> StructuredMeaningGraph:
        graph = self._prepare_multimodal_graph(query, source_context=source_context, visual_input=visual_input)
        graph = self._apply_logical_relation_priors(graph)
        graph = self._apply_unified_parser_priors(graph)
        graph = self._attach_document_grounding_context(graph)
        graph = validate_graph(graph)
        graph = self.premise_explorer.enrich(graph)
        return graph

    def _load_registry(self) -> TypedOperatorRegistry | None:
        if self.memory_store is None:
            return None
        if self._registry_cache is not None:
            return self._registry_cache

        summary = None
        if self.memory_source is not None:
            summary = self.memory_store.fetch_latest_hierarchy_summary(source=self.memory_source, split="train")
        if summary is None:
            summary = self.memory_store.fetch_latest_hierarchy_summary(split="train")
        registry = TypedOperatorRegistry.from_summary(summary)
        self._registry_cache = None if registry.is_empty() else registry
        return self._registry_cache

    def _load_logical_patterns(self) -> List[LogicalPattern]:
        if self.memory_store is None:
            return []
        if self._logical_pattern_cache is not None:
            return self._logical_pattern_cache

        summary = None
        if self.memory_source is not None:
            summary = self.memory_store.fetch_latest_learning_summary(source=self.memory_source)
        if summary is None:
            summary = self.memory_store.fetch_latest_learning_summary()
        raw_patterns = summary.get("logical_patterns", []) if summary else []
        self._logical_pattern_cache = [LogicalPattern(**item) for item in raw_patterns]
        return self._logical_pattern_cache

    def _load_logical_weights(self) -> Dict[str, float]:
        if self._logical_weight_cache is not None:
            return self._logical_weight_cache
        if not self.logical_weight_path:
            self._logical_weight_cache = {}
            return self._logical_weight_cache
        try:
            with open(self.logical_weight_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            self._logical_weight_cache = {}
            return self._logical_weight_cache
        except Exception:
            self._logical_weight_cache = {}
            return self._logical_weight_cache
        weights = payload.get("weights", {}) if isinstance(payload, dict) else {}
        self._logical_weight_cache = {str(key): float(value.get("weight", 1.0) if isinstance(value, dict) else value) for key, value in weights.items()}
        return self._logical_weight_cache

    def _pattern_weight(self, pattern: LogicalPattern) -> float:
        weights = self._load_logical_weights()
        key = f"{pattern.family}::{pattern.connector.lower()}"
        return weights.get(key, 1.0)

    def _apply_logical_relation_priors(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        patterns = self._load_logical_patterns()
        if not patterns:
            return graph
        matches = self._logical_matcher.match(graph.query, graph, patterns)
        if not matches:
            return graph
        for match in matches:
            weighted_confidence = min(0.97, match.pattern.confidence * self._pattern_weight(match.pattern))
            graph.add_edge(
                Edge(
                    source=match.source_id,
                    relation=match.relation,
                    target=match.target_id,
                    confidence=round(weighted_confidence, 2),
                    provenance=[f"grammar_prior_relation:{match.pattern.connector}", f"grammar_family:{match.pattern.family}"],
                )
            )
            note = f"logical relation prior injected: {match.source_id} {match.relation} {match.target_id}"
            if note not in graph.warnings:
                graph.warnings.append(note)
        if matches:
            graph.audit_trace.append("logical relation priors injected before operator induction")
        return graph

    def _apply_logical_grammar_priors(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        patterns = self._load_logical_patterns()
        if not patterns:
            return graph

        lowered = graph.query.lower()
        matched: List[LogicalPattern] = []
        existing_names = {candidate.name for candidate in graph.induced_operators}
        for pattern in patterns:
            connector = pattern.connector.lower().strip()
            if not connector or connector not in lowered:
                continue
            matched.append(pattern)
            if pattern.grammar_template not in graph.grammar_hypotheses:
                graph.grammar_hypotheses.append(pattern.grammar_template)
            note = f"logical grammar prior matched: {pattern.connector} -> {pattern.family}"
            if note not in graph.warnings:
                graph.warnings.append(note)
            if pattern.induced_operator_name not in existing_names:
                graph.induced_operators.append(
                    OperatorCandidate(
                        name=pattern.induced_operator_name,
                        family=pattern.family,
                        arity=2,
                        input_types=pattern.concept_slots[:4] or [graph.intent],
                        output_type="relation_frame",
                        description=self._logical_pattern_description(pattern),
                        examples=pattern.example_frames[:3] or pattern.example_queries[:2],
                        confidence=round(min(0.92, (pattern.confidence + 0.03) * self._pattern_weight(pattern)), 2),
                        provenance=[f"grammar_prior:{pattern.connector}", f"grammar_support:{pattern.support}"],
                    )
                )
                existing_names.add(pattern.induced_operator_name)
            self._inject_plan_prior(graph, pattern)

        if matched:
            summary = ", ".join(sorted({pattern.family for pattern in matched})[:4])
            audit = f"logical grammar priors applied: {summary}"
            if audit not in graph.audit_trace:
                graph.audit_trace.append(audit)
        return graph

    @staticmethod
    def _logical_pattern_description(pattern: LogicalPattern) -> str:
        relation_text = ", ".join(pattern.relation_hints[:3]) if pattern.relation_hints else "structured relation"
        return f"Corpus-induced logical grammar prior for connector '{pattern.connector}' with relation hints {relation_text}."

    def _inject_plan_prior(self, graph: StructuredMeaningGraph, pattern: LogicalPattern) -> None:
        step_id = f"grammar_{pattern.family.lower()}"
        if any(step.id == step_id for step in graph.plan):
            return
        if pattern.family == "PRECONDITION_FRAME":
            graph.plan.insert(0, PlanStep(id=step_id, action="Check prerequisites before executing the main action.", rationale="The query uses a precondition-style connector learned from corpus patterns."))
        elif pattern.family == "CONDITIONAL_FRAME":
            graph.plan.insert(0, PlanStep(id=step_id, action="Branch the reasoning by condition before choosing an action.", rationale="The query uses an if/when-style conditional connector."))
        elif pattern.family == "TEMPORAL_ORDER_FRAME":
            graph.plan.insert(0, PlanStep(id=step_id, action="Preserve the stated temporal order of sub-actions.", rationale="Before/after connectors imply order constraints."))
        elif pattern.family == "CAPABILITY_FRAME":
            graph.plan.insert(0, PlanStep(id=step_id, action="Verify that the object or agent actually affords the requested action.", rationale="Capability connectors usually map to affordance checks."))
        elif pattern.family == "CONTAINMENT_FRAME":
            graph.plan.insert(0, PlanStep(id=step_id, action="Interpret the problem as a containment or inside/outside relation first.", rationale="Containment connectors often hide structural constraints."))

    def _annotate_with_registry(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        registry = self._load_registry()
        if registry is None:
            return graph
        abstract_hits: List[str] = []
        for candidate in graph.induced_operators:
            registry.annotate_candidate(candidate)
            abstract_hits.extend(candidate.abstract_parents)
        if abstract_hits:
            note = "registry attached abstract parents: " + ", ".join(sorted(set(abstract_hits))[:4])
            if note not in graph.warnings:
                graph.warnings.append(note)
        return graph


    def _attach_premise_memory_hints(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        if self.memory_store is None:
            return graph
        premise_hits = self.memory_store.search_premise_support(graph.query, top_k=5, source=self.memory_source)
        operator_hits = self.memory_store.search_operator_support(graph.query, top_k=5, source=self.memory_source)
        for hit in premise_hits:
            note = f"premise memory hint: {hit['premise']} ({hit.get('hidden_goal', '')})".strip()
            if note not in graph.warnings:
                graph.warnings.append(note)
        existing_names = {candidate.name for candidate in graph.induced_operators}
        for hit in operator_hits:
            note = f"operator memory hint: {hit['operator_name']}"
            if note not in graph.warnings:
                graph.warnings.append(note)
            if hit['operator_name'] not in existing_names:
                graph.induced_operators.append(
                    OperatorCandidate(
                        name=str(hit['operator_name']),
                        family=str(hit.get('operator_family', 'memory_family')),
                        arity=2,
                        input_types=[graph.intent],
                        output_type='relation_frame',
                        description='Retrieved from SQLite operator memory.',
                        confidence=float(hit.get('support_score', 0.55)),
                        provenance=['memory_operator_store'],
                    )
                )
                existing_names.add(hit['operator_name'])
        if premise_hits or operator_hits:
            graph.audit_trace.append('sqlite premise/operator memory hints applied')
        return graph

    def _store_runtime_memory(self, graph: StructuredMeaningGraph) -> None:
        if self.memory_store is None:
            return
        source = self.memory_source or 'runtime'
        self.memory_store.upsert_graph(graph, source=source, split='runtime')
        self.memory_store.upsert_premise_operator_memory(graph, source=source, split='runtime')

    def _attach_memory_hints(self, graph: StructuredMeaningGraph, similar_graphs: List[StructuredMeaningGraph]) -> StructuredMeaningGraph:
        if not similar_graphs:
            return graph

        registry = self._load_registry()
        hinted_abstracts: List[str] = []
        for similar in similar_graphs:
            hint = f"memory hint from similar query: {similar.query}"
            if hint not in graph.warnings:
                graph.warnings.append(hint)

            for operator in similar.induced_operators[:2]:
                note = f"retrieved operator family {operator.family} from memory"
                if note not in graph.warnings:
                    graph.warnings.append(note)
                if registry is not None:
                    hinted_abstracts.extend(registry.abstract_parents_for_family(operator.family))

            for alternative in similar.creative_alternatives[:1]:
                if alternative not in graph.creative_alternatives:
                    graph.creative_alternatives.append(alternative)

            if not graph.grammar_hypotheses:
                graph.grammar_hypotheses.extend(similar.grammar_hypotheses[:2])

        if hinted_abstracts:
            note = "memory hint abstract operators: " + ", ".join(sorted(set(hinted_abstracts))[:4])
            if note not in graph.warnings:
                graph.warnings.append(note)
        return graph

    def _apply_memory_priors(self, graph: StructuredMeaningGraph, similar_graphs: List[StructuredMeaningGraph]) -> StructuredMeaningGraph:
        if not similar_graphs:
            return graph

        registry = self._load_registry()
        family_support: Dict[str, int] = {}
        exemplar_by_family: Dict[str, OperatorCandidate] = {}
        grammar_by_family: Dict[str, str] = {}
        for similar in similar_graphs:
            for candidate in similar.induced_operators:
                family_support[candidate.family] = family_support.get(candidate.family, 0) + 1
                exemplar = exemplar_by_family.get(candidate.family)
                if exemplar is None or candidate.confidence > exemplar.confidence:
                    exemplar_by_family[candidate.family] = deepcopy(candidate)
            for rule in similar.grammar_hypotheses:
                for family in family_support:
                    if family in rule and family not in grammar_by_family:
                        grammar_by_family[family] = rule

        if not family_support:
            return graph

        abstract_support = registry.abstract_support_map(family_support) if registry is not None else {}
        promoted_families: List[str] = []
        promoted_abstracts: List[str] = []
        existing_families: set[str] = set()
        for candidate in graph.induced_operators:
            existing_families.add(candidate.family)
            support = family_support.get(candidate.family, 0)
            if support > 0:
                candidate.confidence = round(min(0.99, candidate.confidence + min(0.18, 0.04 * support)), 2)
                prior_tag = f"memory_prior:support:{support}"
                if prior_tag not in candidate.provenance:
                    candidate.provenance.append(prior_tag)
                exemplar = exemplar_by_family.get(candidate.family)
                if exemplar is not None:
                    for example in exemplar.examples[:2]:
                        if example not in candidate.examples:
                            candidate.examples.append(example)
                    if candidate.output_type == "meaning_state" and exemplar.output_type != "meaning_state":
                        candidate.output_type = exemplar.output_type
                promoted_families.append(candidate.family)
            if registry is not None:
                registry.annotate_candidate(candidate)
                for parent in candidate.abstract_parents:
                    parent_support = abstract_support.get(parent, 0)
                    if parent_support <= 0:
                        continue
                    candidate.confidence = round(min(0.99, candidate.confidence + min(0.12, 0.02 * parent_support)), 2)
                    abstract_tag = f"memory_prior:abstract:{parent}:{parent_support}"
                    if abstract_tag not in candidate.provenance:
                        candidate.provenance.append(abstract_tag)
                    promoted_abstracts.append(parent)

        injected_families: List[str] = []
        for family, support in sorted(family_support.items(), key=lambda item: (-item[1], item[0])):
            if family in existing_families or support < 2:
                continue
            exemplar = exemplar_by_family.get(family)
            if exemplar is None:
                continue
            injected = deepcopy(exemplar)
            injected.confidence = round(min(0.82, 0.5 + 0.05 * support), 2)
            injected.provenance = list(dict.fromkeys(injected.provenance + [f"memory_prior:injected:{support}"]))
            if registry is not None:
                registry.annotate_candidate(injected)
            graph.induced_operators.append(injected)
            existing_families.add(family)
            injected_families.append(family)
            if len(injected_families) >= 2:
                break

        abstract_injected_families: List[str] = []
        if registry is not None:
            ordered_candidates = sorted(graph.induced_operators, key=lambda item: (-item.confidence, item.name))
            for candidate in ordered_candidates:
                for related_family in registry.related_families(candidate):
                    if related_family in existing_families:
                        continue
                    support = family_support.get(related_family, 0)
                    if support < 2:
                        continue
                    exemplar = exemplar_by_family.get(related_family)
                    if exemplar is None:
                        continue
                    injected = deepcopy(exemplar)
                    injected.confidence = round(min(0.78, 0.48 + 0.04 * support), 2)
                    injected.provenance = list(dict.fromkeys(injected.provenance + [f"memory_prior:abstract_injected:{support}"]))
                    registry.annotate_candidate(injected)
                    graph.induced_operators.append(injected)
                    existing_families.add(related_family)
                    abstract_injected_families.append(related_family)
                    if len(abstract_injected_families) >= 2:
                        break
                if len(abstract_injected_families) >= 2:
                    break

        for family in list(dict.fromkeys(promoted_families + injected_families + abstract_injected_families)):
            rule = grammar_by_family.get(family)
            if rule and rule not in graph.grammar_hypotheses:
                graph.grammar_hypotheses.append(rule)

        if promoted_families:
            note = "memory prior promoted families: " + ", ".join(sorted(set(promoted_families)))
            if note not in graph.warnings:
                graph.warnings.append(note)
        if injected_families:
            note = "memory prior injected families: " + ", ".join(sorted(set(injected_families)))
            if note not in graph.warnings:
                graph.warnings.append(note)
        if promoted_abstracts:
            note = "memory prior promoted abstract operators: " + ", ".join(sorted(set(promoted_abstracts)))
            if note not in graph.warnings:
                graph.warnings.append(note)
        if abstract_injected_families:
            note = "memory prior abstract sibling injection: " + ", ".join(sorted(set(abstract_injected_families)))
            if note not in graph.warnings:
                graph.warnings.append(note)

        graph.induced_operators.sort(
            key=lambda candidate: (
                -candidate.confidence,
                -family_support.get(candidate.family, 0),
                -len(candidate.abstract_parents),
                candidate.name,
            )
        )
        return graph


    def _apply_analogy_guidance(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        if not graph.analogical_matches:
            return graph
        graph = self._apply_analogy_aware_planning(graph)
        graph = self._apply_analogy_aware_operator_priorities(graph)
        graph = self._apply_analogy_aware_verifier(graph)
        return graph

    def _apply_analogy_aware_planning(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        requirement_support = self._analogy_requirement_support(graph)
        if not requirement_support:
            return graph
        plan_weight = self.analogy_policy.model.plan_guard_weight if self.analogy_policy is not None else 1.0
        if not any(step.id == 'analogy_requirement_guard' for step in graph.plan):
            requirement_labels = ', '.join(concept_label(item) for item in list(requirement_support.keys())[:3])
            graph.plan.insert(
                0,
                PlanStep(
                    id='analogy_requirement_guard',
                    action=f"Recall similar cases and verify {requirement_labels} before direct execution.",
                    rationale=f'Several structurally similar cases shared the same prerequisite pattern and failed when it was skipped (weight={plan_weight:.2f}).',
                    requires=list(requirement_support.keys())[:3],
                ),
            )
        if len(graph.analogical_matches) >= 2 and not any(step.id == 'analogy_compare_cases' for step in graph.plan):
            graph.plan.insert(
                1,
                PlanStep(
                    id='analogy_compare_cases',
                    action='Compare recalled cases to separate shared constraints from surface wording differences.',
                    rationale='Multiple analogies are available, so the plan should preserve the common structure rather than copy a single example.',
                ),
            )
        note = 'analogy-aware planning: inserted prerequisite guard from structurally similar cases'
        if note not in graph.audit_trace:
            graph.audit_trace.append(note)
        return graph

    def _apply_analogy_aware_operator_priorities(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        family_support: Dict[str, float] = {}
        weight = self.analogy_policy.model.operator_priority_weight if self.analogy_policy is not None else 0.7
        for item in graph.analogical_matches:
            for family in item.shared_operator_families:
                family_support[family] = family_support.get(family, 0.0) + float(item.score)
        if not family_support:
            return graph
        boosted: List[str] = []
        for candidate in graph.induced_operators:
            support = family_support.get(candidate.family, 0.0)
            if support <= 0.0:
                continue
            candidate.confidence = round(min(0.99, candidate.confidence + min(0.16, 0.05 * support * weight)), 2)
            tag = f'analogy_policy:operator_priority:{round(support, 2)}'
            if tag not in candidate.provenance:
                candidate.provenance.append(tag)
            boosted.append(candidate.family)
        if boosted:
            note = 'analogy-aware operator priorities: boosted families from learned analogy policy'
            if note not in graph.audit_trace:
                graph.audit_trace.append(note)
        return graph

    def _apply_analogy_aware_verifier(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        requirement_support = self._analogy_requirement_support(graph)
        if not requirement_support:
            return graph

        verifier_weight = self.analogy_policy.model.verifier_weight if self.analogy_policy is not None else 1.0
        strengthened_premises: List[str] = []
        for item in graph.premise_validations:
            support = requirement_support.get(item.premise, 0)
            if support <= 0 or item.status not in {'missing', 'unsupported', 'uncertain'}:
                continue
            item.support_score = round(min(0.99, item.support_score + min(0.22, 0.04 * support * verifier_weight)), 2)
            reason = f'Analogical memory found {support} structurally similar cases that also depended on {concept_label(item.premise)}.'
            if reason not in item.rationale:
                item.rationale = (item.rationale + ' ' + reason).strip()
            strengthened_premises.append(item.premise)

        action_requirements = {
            'walk_without_car': 'vehicle_present',
            'insert_without_opening': 'open_access',
            'retrieve_without_opening': 'open_access',
            'pour_without_uncapping': 'open_access',
            'proceed_without_opening': 'open_access',
            'recompute_each_query': 'subquadratic_complexity',
        }
        strengthened_checks: List[str] = []
        for item in graph.goal_preservation_checks:
            required_premise = action_requirements.get(item.action)
            support = requirement_support.get(required_premise or '', 0)
            if support <= 0 or item.status not in {'risk_high', 'conditionally_valid'}:
                continue
            item.confidence = round(min(0.99, item.confidence + min(0.16, 0.03 * support * verifier_weight)), 2)
            note = f'Analogical memory recalled {support} structurally similar cases that failed without {concept_label(required_premise)}.'
            if note not in item.rationale:
                item.rationale = (item.rationale + ' ' + note).strip()
            strengthened_checks.append(item.action)

        for premise, support in requirement_support.items():
            if premise not in graph.missing_premises:
                continue
            warning = f'analogy verifier: {support} similar cases point to {concept_label(premise)} as a recurring prerequisite.'
            if warning not in graph.warnings:
                graph.warnings.append(warning)

        if strengthened_premises or strengthened_checks:
            note = 'analogy-aware verifier: strengthened premise and goal-risk checks from recalled cases'
            if note not in graph.audit_trace:
                graph.audit_trace.append(note)
        return graph

    @staticmethod
    def _analogy_requirement_support(graph: StructuredMeaningGraph) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in graph.analogical_matches:
            if item.analogy_type not in {'goal_premise_analogy', 'failure_analogy', 'operator_analogy'}:
                continue
            for premise in item.shared_requirements:
                counts[premise] = counts.get(premise, 0) + 1
        return counts

    def _prepare_multimodal_graph(
        self,
        query: str,
        source_context: str = "",
        visual_input: Any | None = None,
        base_graph: StructuredMeaningGraph | None = None,
    ) -> StructuredMeaningGraph:
        graph = deepcopy(base_graph) if base_graph is not None else self.heuristic.extract(query)
        graph.query = query
        graph.source_context = source_context or graph.source_context
        if not any(node.id == "question" for node in graph.nodes):
            graph.add_node(Node(id="question", label=query, kind="query", attributes={"text": query}, provenance=["pipeline:query"]))
        if source_context.strip():
            graph.domain = graph.domain if graph.domain != "general" else "document_grounded_reasoning"
        if visual_input is not None:
            graph = self._merge_visual_world(graph, visual_input)
        return graph

    def _apply_unified_parser_priors(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        if self.unified_parser is None:
            return graph
        return self.unified_parser.enrich(graph)

    def _attach_document_grounding_context(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        if not graph.source_context.strip():
            return graph
        graph.add_node(
            Node(
                id="source_document",
                label="source_document",
                kind="document",
                attributes={"pdf_like": self._looks_like_pdf_context(graph.source_context)},
                provenance=["document:source_context"],
            )
        )
        graph.add_edge(Edge(source="question", relation="USES_CONTEXT", target="source_document", confidence=0.92, provenance=["document:source_context"]))
        chunks = split_context_into_chunks(graph.source_context)
        for index, chunk in enumerate(chunks, start=1):
            graph.add_node(
                Node(
                    id=chunk.chunk_id,
                    label=chunk.text[:96],
                    kind="evidence",
                    attributes={"text": chunk.text, "source": chunk.source, "modality": "document"},
                    provenance=["document:chunk"],
                )
            )
            graph.add_edge(Edge(source="source_document", relation="HAS_EVIDENCE", target=chunk.chunk_id, confidence=0.9, provenance=["document:chunk"]))
            if index <= 2:
                graph.add_edge(Edge(source="question", relation="GROUNDED_BY", target=chunk.chunk_id, confidence=0.74, provenance=["document:grounding_context"]))
        note = f"document grounding context attached: {len(chunks)} chunks"
        if note not in graph.audit_trace:
            graph.audit_trace.append(note)
        return graph

    @staticmethod
    def _looks_like_pdf_context(text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in ["page ", "section", "appendix", ".pdf", "figure ", "table "])

    def _merge_visual_world(self, graph: StructuredMeaningGraph, visual_input: Any) -> StructuredMeaningGraph:
        parser = self._visual_parser_instance()
        world, observation = parser.parse(visual_input)
        graph.domain = graph.domain if graph.domain != "general" else "multimodal_reasoning"
        graph.add_node(Node(id="visual_scene", label="visual_scene", kind="scene", attributes={"source": observation.metadata.get("source", "vision")}, provenance=["multimodal:vision_scene"]))
        graph.add_edge(Edge(source="question", relation="CONDITIONS_ON", target="visual_scene", confidence=0.88, provenance=["multimodal:vision_scene"]))
        for entity in world.entities:
            graph.add_node(
                Node(
                    id=str(entity.id),
                    label=str(entity.label),
                    kind=str(entity.entity_type or "object"),
                    attributes=dict(entity.attributes),
                    provenance=[f"multimodal:{entity.modality}_entity"],
                )
            )
        for relation in world.relations:
            graph.add_edge(
                Edge(
                    source=str(relation.source),
                    relation=str(relation.relation),
                    target=str(relation.target),
                    confidence=float(relation.confidence),
                    attributes=dict(relation.attributes),
                    provenance=[f"multimodal:{relation.modality}_relation"],
                )
            )
        existing_names = {candidate.name for candidate in graph.induced_operators}
        for operator in world.operators:
            if operator.name in existing_names:
                continue
            graph.induced_operators.append(
                OperatorCandidate(
                    name=str(operator.name),
                    family=str(operator.name if str(operator.name).endswith("_OPERATOR") else f"VISUAL_{operator.axis.upper()}"),
                    arity=1,
                    input_types=["visual_structure", graph.intent],
                    output_type="visual_relation_frame",
                    description=str(operator.description),
                    examples=[str(operator.axis)],
                    confidence=round(float(operator.confidence), 2),
                    provenance=[f"multimodal:{operator.source_modality}_operator"],
                )
            )
            existing_names.add(operator.name)
        for goal in world.goals:
            if goal not in graph.hidden_goals:
                graph.hidden_goals.append(goal)
        for item in world.inferred_steps[:3]:
            if item not in graph.creative_alternatives:
                graph.creative_alternatives.append(item)
        for item in world.constraints[:4]:
            if item not in graph.warnings:
                graph.warnings.append(item)
        self._merge_visual_premise_support(graph, world, observation)
        self._attach_visual_grounding_context(graph, world, observation)
        for item in world.audit_trace[:6]:
            if item not in graph.audit_trace:
                graph.audit_trace.append(item)
        for item in world.warnings[:4]:
            if item not in graph.warnings:
                graph.warnings.append(item)
        note = f"multimodal merge: fused {len(world.entities)} visual entities and {len(world.operators)} operators"
        if note not in graph.audit_trace:
            graph.audit_trace.append(note)
        return graph

    def _merge_visual_premise_support(self, graph: StructuredMeaningGraph, world, observation) -> None:
        operator_names = {str(item.name).upper() for item in world.operators}
        state_values = {str(item.get("value", item.get("state", ""))).upper() for item in observation.states if isinstance(item, dict)}
        if operator_names & {"ACCESS_PORT_OPERATOR", "CONTAINER_ACCESS_OPERATOR", "CONTROLLED_ACCESS_OPERATOR", "OPENABLE"}:
            if "OPEN" in state_values:
                if "open_access" not in graph.satisfied_premises:
                    graph.satisfied_premises.append("open_access")
            elif "open_access" not in graph.required_premises and "open_access" not in graph.satisfied_premises:
                graph.required_premises.append("open_access")
        if operator_names & {"CONTAINER_BODY_OPERATOR", "MANIPULABLE_CONTAINER_OPERATOR", "CARRIABLE_CONTAINER_OPERATOR"}:
            if "available_space" not in graph.satisfied_premises:
                graph.satisfied_premises.append("available_space")
        if operator_names & {"ATTACHED_GRASP_OPERATOR", "MANIPULABLE_CONTAINER_OPERATOR"}:
            if "manipulable_grasp" not in graph.satisfied_premises:
                graph.satisfied_premises.append("manipulable_grasp")
        for premise in list(dict.fromkeys(graph.satisfied_premises)):
            if premise in graph.required_premises:
                graph.required_premises = [item for item in graph.required_premises if item != premise]
            if premise in graph.missing_premises:
                graph.missing_premises = [item for item in graph.missing_premises if item != premise]

    def _attach_visual_grounding_context(self, graph: StructuredMeaningGraph, world, observation) -> None:
        candidates = self._visual_grounding_candidates(graph, world)
        if not candidates:
            return
        source = str(observation.metadata.get("source", "vision"))
        for index, entity in enumerate(candidates, start=1):
            evidence_id = f"visual_evidence_{index:03d}_{entity.id}"
            text = self._visual_evidence_text(entity)
            graph.add_node(
                Node(
                    id=evidence_id,
                    label=text[:96],
                    kind="evidence",
                    attributes={"text": text, "source": source, "modality": "vision", "entity_id": str(entity.id)},
                    provenance=["multimodal:vision_evidence"],
                )
            )
            graph.add_edge(Edge(source="visual_scene", relation="HAS_EVIDENCE", target=evidence_id, confidence=0.82, provenance=["multimodal:vision_evidence"]))
            graph.add_edge(Edge(source="question", relation="GROUNDED_BY", target=evidence_id, confidence=0.76, provenance=["multimodal:vision_grounding"]))
        note = f"visual grounding context attached: {len(candidates)} evidence nodes"
        if note not in graph.audit_trace:
            graph.audit_trace.append(note)

    def _visual_grounding_candidates(self, graph: StructuredMeaningGraph, world, limit: int = 4):
        query_tokens = set(re.findall(r"[0-9a-z_]+", graph.query.lower()))
        scored = []
        for entity in world.entities:
            haystack_parts = [str(entity.id), str(entity.label), str(entity.entity_type)]
            haystack_parts.extend(str(value) for value in entity.attributes.values())
            haystack = " ".join(haystack_parts).lower()
            score = float(sum(1 for token in query_tokens if token and token in haystack))
            if entity.entity_type in {"part", "container", "object", "state"}:
                score += 0.5
            if any(keyword in haystack for keyword in ["handle", "opening", "drawer", "cabinet", "door", "zipper", "lid", "bottle"]):
                score += 1.5
            if entity.attributes.get("structural_role"):
                score += 1.0
            scored.append((score, str(entity.id), entity))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [entity for score, _entity_id, entity in scored if score > 0][:limit]
        if selected:
            return selected
        return [entity for _score, _entity_id, entity in scored[:limit]]

    @staticmethod
    def _visual_evidence_text(entity) -> str:
        parts = [str(entity.label or entity.id)]
        role = entity.attributes.get("structural_role")
        if role:
            parts.append(f"role={role}")
        state = entity.attributes.get("value") or entity.attributes.get("state")
        if state:
            parts.append(f"state={state}")
        parent = entity.attributes.get("part_of")
        if parent:
            parts.append(f"part_of={parent}")
        return "visual evidence: " + ", ".join(parts)

    def _visual_parser_instance(self):
        if self._visual_parser is None:
            from .vlso.visual_parser import VLSOVisualParser

            self._visual_parser = VLSOVisualParser()
        return self._visual_parser

    def _build_graph_from_llm(self, query: str, raw: Dict[str, Any]) -> StructuredMeaningGraph:
        graph = StructuredMeaningGraph(query=query, intent=raw.get("intent", "generic_reasoning"))
        for entity in raw.get("entities", []):
            concept_id = normalize_concept(entity.get("id", entity.get("label", "concept")))
            graph.add_node(
                Node(
                    id=concept_id,
                    label=entity.get("label", concept_label(concept_id)),
                    kind=entity.get("kind", "concept"),
                    attributes=entity.get("attributes", {}),
                    provenance=[f"llm:entity:{concept_id}"],
                )
            )
            apply_many(graph, kb_relations_for_concept(concept_id))
            for script in scripts_for_concept(concept_id):
                if script not in graph.inferred_scripts:
                    graph.inferred_scripts.append(script)
        for rel in raw.get("relations", []):
            graph.add_edge(
                Edge(
                    source=normalize_concept(rel["source"]),
                    relation=rel["relation"],
                    target=normalize_concept(rel["target"]),
                    confidence=float(rel.get("confidence", 0.8)),
                    provenance=[f"llm:relation:{rel['relation']}"],
                )
            )
        graph.inferred_scripts.extend(raw.get("scripts", []))
        graph.candidate_actions.extend(raw.get("candidate_actions", []))
        graph.warnings.extend(raw.get("constraints", []))
        if raw.get("missing_knowledge"):
            graph.warnings.append("Missing knowledge: " + "; ".join(raw["missing_knowledge"]))
        return graph

    def _ensure_plan(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        node_ids = graph.node_ids()
        if graph.plan:
            return graph
        if {"bag", "book"}.issubset(node_ids):
            bag_graph = self.heuristic.extract(graph.query)
            graph.plan = bag_graph.plan
            graph.invalid_advice.extend(bag_graph.invalid_advice)
            graph.creative_alternatives.extend(bag_graph.creative_alternatives)
        elif "car_wash" in node_ids:
            car_wash_graph = self.heuristic.extract(graph.query)
            graph.plan = car_wash_graph.plan
            graph.invalid_advice.extend(car_wash_graph.invalid_advice)
            graph.creative_alternatives.extend(car_wash_graph.creative_alternatives)
        else:
            graph.plan = self._generic_plan_from_edges(graph)
        return graph

    @staticmethod
    def _generic_plan_from_edges(graph: StructuredMeaningGraph) -> list[PlanStep]:
        steps = [
            PlanStep(
                id="step1",
                action="Break the query into goals, entities, and constraints first.",
                rationale="Structured reasoning starts by separating targets from blockers.",
            )
        ]
        for edge in graph.edges:
            if edge.relation == "REQUIRES":
                steps.append(
                    PlanStep(
                        id=f"require_{edge.source}_{edge.target}",
                        action=f"Satisfy {concept_label(edge.target)} before executing {concept_label(edge.source)}.",
                        rationale="A prerequisite must hold before the downstream action can execute.",
                        requires=[edge.target],
                    )
                )
            if edge.relation == "BLOCKED_BY":
                steps.append(
                    PlanStep(
                        id=f"block_{edge.source}_{edge.target}",
                        action=f"Reduce or route around the blocker {concept_label(edge.target)}.",
                        rationale="Handling blockers is higher priority than forcing direct execution.",
                        requires=[edge.target],
                    )
                )
        steps.append(
            PlanStep(
                id="step_final",
                action="Choose the option that still satisfies the goal under current constraints.",
                rationale="Logical planning should keep only executable candidates.",
            )
        )
        return steps








from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Dict, List

from .commonsense_kb import concept_label, kb_relations_for_concept, normalize_concept, scripts_for_concept
from .corpus_store import CorpusMemoryStore
from .emergent_operators import EmergentOperatorInducer
from .heuristic_extractors import HeuristicExtractor
from .llm_client import LocalLLMConfig, LocalTransformersExtractor
from .logical_grammar import LogicalPattern, LogicalPatternMatcher
from .operator_registry import TypedOperatorRegistry
from .semantic_operators import apply_many
from .structures import Edge, Node, OperatorCandidate, PlanStep, StructuredMeaningGraph
from .symbolic_reasoners import SymbolicReasoner
from .validation import validate_graph


class StructuredMeaningPipeline:
    def __init__(self, mode: str = "heuristic", model_id: str = "Qwen/Qwen2.5-3B-Instruct", memory_store_path: str | None = None, memory_source: str | None = None, logical_weight_path: str | None = None):
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
        self.logical_weight_path = logical_weight_path
        self._logical_weight_cache: Dict[str, float] | None = None

    def run(self, query: str) -> StructuredMeaningGraph:
        similar_graphs = self._retrieve_similar_graphs(query)
        return self.run_with_memory_graphs(query, similar_graphs)

    def run_with_memory_graphs(self, query: str, similar_graphs: List[StructuredMeaningGraph] | None = None) -> StructuredMeaningGraph:
        similar_graphs = similar_graphs or []
        graph = self._prepare_graph(query)
        graph = self._apply_logical_grammar_priors(graph)
        graph = self._attach_memory_hints(graph, similar_graphs)
        graph = self.inducer.induce(graph)
        graph = self._annotate_with_registry(graph)
        graph = self._apply_memory_priors(graph, similar_graphs)
        return self._finalize_graph(graph)

    def _prepare_graph(self, query: str) -> StructuredMeaningGraph:
        if self.mode == "heuristic":
            graph = self.heuristic.extract(query)
            graph = self._apply_logical_relation_priors(graph)
            return validate_graph(graph)

        try:
            raw = self.llm.extract(query) if self.llm is not None else None
        except Exception as exc:
            graph = self.heuristic.extract(query)
            graph.warnings.append(f"llm extraction failed; fallback to heuristic: {exc}")
            return validate_graph(graph)

        graph = self._build_graph_from_llm(query, raw or {})
        graph = self._apply_logical_relation_priors(graph)
        graph = self._ensure_plan(graph)
        return validate_graph(graph)

    def _finalize_graph(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        graph = self.symbolic.apply(graph)
        graph = self._annotate_with_registry(graph)
        graph.induced_operators.sort(key=lambda candidate: (-candidate.confidence, candidate.name))
        return graph

    def _retrieve_similar_graphs(self, query: str) -> List[StructuredMeaningGraph]:
        if self.memory_store is None:
            return []
        return self.memory_store.search_similar_graphs(query, split="train", source=self.memory_source, top_k=3)

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

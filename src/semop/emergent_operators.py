from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List

from .latent_abstraction import OperatorAbstractionClustering
from .structures import OperatorCandidate, StructuredMeaningGraph


@dataclass
class EvidenceItem:
    text: str
    family_hint: str
    relation: str
    input_types: List[str]
    output_type: str
    provenance: List[str]


class EmergentOperatorInducer:
    """Induce reusable operator and grammar candidates from graph relations and action phrases."""

    def __init__(self, threshold: float = 0.34):
        self.threshold = threshold
        self.clusterer = OperatorAbstractionClustering()

    def induce(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        evidence = self._collect_evidence(graph)
        if not evidence:
            return graph

        text_to_items: Dict[str, List[EvidenceItem]] = {}
        for item in evidence:
            text_to_items.setdefault(item.text, []).append(item)

        phrase_clusters = self.clusterer.cluster(text_to_items.keys(), threshold=self.threshold)
        candidates: List[OperatorCandidate] = []
        grammar_rules: List[str] = []

        for cluster in phrase_clusters:
            members: List[EvidenceItem] = []
            for text in cluster.members:
                members.extend(text_to_items.get(text, []))
            if not members:
                continue
            candidate = self._candidate_from_cluster(cluster.label, members)
            candidates.append(candidate)
            grammar_rules.append(self._grammar_rule(candidate))

        graph.induced_operators = self._dedupe_candidates(graph.induced_operators + candidates)
        graph.grammar_hypotheses = list(dict.fromkeys(graph.grammar_hypotheses + grammar_rules))
        return graph

    def _collect_evidence(self, graph: StructuredMeaningGraph) -> List[EvidenceItem]:
        evidence: List[EvidenceItem] = []
        context_types = self._graph_context_types(graph)

        for edge in graph.edges:
            typed_inputs = self._dedupe_types([edge.source, edge.target, *context_types[:2]])
            if edge.relation == "REQUIRES":
                evidence.append(EvidenceItem(
                    text=f"{edge.source} requires {edge.target}",
                    family_hint="PRECONDITION",
                    relation=edge.relation,
                    input_types=typed_inputs,
                    output_type="constraint_checked_plan",
                    provenance=edge.provenance or [f"relation:{edge.relation}"],
                ))
            elif edge.relation == "BLOCKED_BY":
                evidence.append(EvidenceItem(
                    text=f"{edge.source} blocked by {edge.target}",
                    family_hint="REDIRECT",
                    relation=edge.relation,
                    input_types=typed_inputs,
                    output_type="redirected_plan",
                    provenance=edge.provenance or [f"relation:{edge.relation}"],
                ))
            elif edge.relation == "ALTERNATIVE":
                evidence.append(EvidenceItem(
                    text=f"{edge.source} alternative {edge.target}",
                    family_hint="ALTERNATIVE_SEARCH",
                    relation=edge.relation,
                    input_types=typed_inputs,
                    output_type="alternative_plan",
                    provenance=edge.provenance or [f"relation:{edge.relation}"],
                ))
            elif edge.relation in {"CONTAINS", "PART_OF"}:
                evidence.append(EvidenceItem(
                    text=f"{edge.source} structural relation {edge.target}",
                    family_hint="STRUCTURE_RECOVERY",
                    relation=edge.relation,
                    input_types=typed_inputs,
                    output_type="structured_state",
                    provenance=edge.provenance or [f"relation:{edge.relation}"],
                ))

        for action in graph.candidate_actions:
            evidence.append(EvidenceItem(
                text=action,
                family_hint="ACTION_REWRITE",
                relation="ACTION",
                input_types=context_types,
                output_type="candidate_action",
                provenance=["graph:candidate_actions"],
            ))

        for phrase in graph.creative_alternatives:
            evidence.append(EvidenceItem(
                text=phrase,
                family_hint="CREATIVE_REFRAME",
                relation="CREATIVE",
                input_types=context_types,
                output_type="creative_option",
                provenance=["graph:creative_alternatives"],
            ))

        for script in graph.inferred_scripts:
            evidence.append(EvidenceItem(
                text=script.replace("_", " "),
                family_hint="SCRIPT_PATTERN",
                relation="SCRIPT",
                input_types=context_types,
                output_type="script_step",
                provenance=["graph:inferred_scripts"],
            ))

        return evidence

    def _candidate_from_cluster(self, cluster_label: str, members: List[EvidenceItem]) -> OperatorCandidate:
        family = self._family(cluster_label, members)
        arity = 2 if sum(1 for member in members if len(member.input_types) >= 2) >= max(1, len(members) // 2) else 1
        input_types = self._merge_types(members)
        output_type = self._majority([member.output_type for member in members])
        provenance = list(dict.fromkeys(p for member in members for p in member.provenance))
        examples = list(dict.fromkeys(member.text for member in members))[:4]
        description = self._description(family, members)
        name = self._name_operator(family, examples)
        confidence = min(0.95, 0.43 + 0.07 * len(examples) + 0.03 * len(provenance) + 0.02 * max(0, len(input_types) - 1))
        return OperatorCandidate(
            name=name,
            family=family,
            arity=arity,
            input_types=input_types,
            output_type=output_type,
            description=description,
            examples=examples,
            confidence=round(confidence, 2),
            provenance=provenance,
        )

    def _family(self, cluster_label: str, members: List[EvidenceItem]) -> str:
        dominant = self._majority([member.family_hint for member in members]) or "EMERGENT_OPERATOR"
        subtype = self._subtype(members)
        qualifiers: List[str] = []
        if subtype != "GENERIC":
            qualifiers.append(subtype)
        if cluster_label not in {"", "ABSTRACT_OPERATOR", dominant}:
            qualifiers.append(cluster_label)
        return "_".join([dominant, *dict.fromkeys(qualifiers)])[:72]

    def _subtype(self, members: List[EvidenceItem]) -> str:
        text = " ".join(member.text.lower() for member in members)
        input_types = {item for member in members for item in member.input_types}

        if any(token in text for token in ["car_wash", "traffic", "drive", "route", "vehicle", "mobile_detailer"]):
            return "MOBILITY"
        if any(token in text for token in ["bag", "book", "zipper", "container", "open_access", "available_space"]):
            return "CONTAINMENT"
        if any(token in text for token in ["revenue", "income", "cash flow", "metric", "office", "desk", "statement"]):
            return "DOCUMENT"
        if any(token in text for token in ["hour", "day", "week", "month", "time", "schedule", "delay", "later"]):
            return "TEMPORAL"
        if any(token in text for token in ["%", "$", "average", "total", "split", "cost", "price", "amount"]):
            return "QUANTITATIVE"
        if {"service_place", "route_strategy", "vehicle", "mobility_reasoning"} & input_types:
            return "MOBILITY"
        if {"container", "closure", "space", "containment_reasoning"} & input_types:
            return "CONTAINMENT"
        if {"document_reasoning", "table", "report"} & input_types:
            return "DOCUMENT"
        if {"quantitative_reasoning", "number", "quantity"} & input_types:
            return "QUANTITATIVE"
        if {"time_strategy", "temporal_reasoning"} & input_types:
            return "TEMPORAL"
        return "GENERIC"

    def _graph_context_types(self, graph: StructuredMeaningGraph) -> List[str]:
        lowered = graph.query.lower()
        context: List[str] = [graph.intent]

        for node in graph.nodes:
            if node.kind not in {"concept", "user", "task"} and node.kind not in context:
                context.append(node.kind)
            if node.id not in context:
                context.append(node.id)

        if re.search(r"\d", graph.query) or any(token in lowered for token in ["$", "%", "cost", "price", "sum", "total", "average"]):
            context.append("quantitative_reasoning")
        if any(token in lowered for token in ["metric", "revenue", "income", "cash flow", "office", "desk", "report"]) or "|" in graph.query:
            context.append("document_reasoning")
        if any(token in lowered for token in ["bag", "book", "zipper", "container", "storage"]):
            context.append("containment_reasoning")
        if any(token in lowered for token in ["car wash", "traffic", "drive", "route", "vehicle"]):
            context.append("mobility_reasoning")
        if any(token in lowered for token in ["time", "day", "week", "hour", "month", "delay", "schedule"]):
            context.append("temporal_reasoning")
        return self._dedupe_types(context)[:6]

    def _name_operator(self, family: str, examples: List[str]) -> str:
        tokens = [
            token
            for example in examples
            for token in re.findall(r"[A-Za-z_]+", example.lower())
            if len(token) >= 2
        ]
        stopwords = {"requires", "blocked", "alternative", "structural", "relation", "graph", "from", "with"}
        tokens = [token for token in tokens if token not in stopwords]
        top = list(dict.fromkeys(tokens))[:2]
        suffix = "_".join(token.upper() for token in top) if top else family
        return f"OP_{family}_{suffix}"[:80]

    def _description(self, family: str, members: List[EvidenceItem]) -> str:
        dominant_relation = self._majority([member.relation for member in members])
        subtype = self._subtype(members).lower()
        if dominant_relation == "REQUIRES":
            return f"A {subtype} operator candidate that checks prerequisites before the next action."
        if dominant_relation == "BLOCKED_BY":
            return f"A {subtype} operator candidate that redirects execution around a blocking condition."
        if dominant_relation == "ALTERNATIVE":
            return f"A {subtype} operator candidate that rewrites the goal through an alternative path."
        if dominant_relation in {"CONTAINS", "PART_OF"}:
            return f"A {subtype} operator candidate that recovers structural or containment relations."
        if family.startswith("CREATIVE_REFRAME"):
            return f"A {subtype} creative operator candidate that reframes the path while preserving the goal."
        return f"An unsupervised {subtype} operator candidate induced from recurring graph and phrase patterns."

    def _grammar_rule(self, candidate: OperatorCandidate) -> str:
        left = ", ".join(candidate.input_types) if candidate.input_types else "meaning_unit"
        return f"{left} -> {candidate.name}({left}) -> {candidate.output_type}"

    @staticmethod
    def _merge_types(members: List[EvidenceItem]) -> List[str]:
        types: List[str] = []
        for member in members:
            for input_type in member.input_types:
                if input_type not in types:
                    types.append(input_type)
        return types[:6] or ["meaning_unit"]

    @staticmethod
    def _majority(values: List[str]) -> str:
        counts: Dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return max(counts, key=counts.get) if counts else ""

    @staticmethod
    def _dedupe_types(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value for value in values if value))

    @staticmethod
    def _dedupe_candidates(candidates: List[OperatorCandidate]) -> List[OperatorCandidate]:
        kept: Dict[str, OperatorCandidate] = {}
        for candidate in candidates:
            existing = kept.get(candidate.name)
            if existing is None or candidate.confidence > existing.confidence:
                kept[candidate.name] = candidate
        return sorted(kept.values(), key=lambda item: (-item.confidence, item.name))

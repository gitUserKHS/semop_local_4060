from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Sequence

from .structures import Edge, Node, OperatorCandidate, OperatorDecomposition, StructuredMeaningGraph

if TYPE_CHECKING:
    from .corpus_store import CorpusMemoryStore


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9a-z_]+", text.lower())


@dataclass
class UnifiedParserModel:
    token_intents: dict[str, dict[str, float]] = field(default_factory=dict)
    token_domains: dict[str, dict[str, float]] = field(default_factory=dict)
    token_goals: dict[str, dict[str, float]] = field(default_factory=dict)
    token_required_premises: dict[str, dict[str, float]] = field(default_factory=dict)
    token_operator_families: dict[str, dict[str, float]] = field(default_factory=dict)
    token_decompositions: dict[str, dict[str, float]] = field(default_factory=dict)
    min_vote: float = 0.35
    dominance_threshold: float = 0.62
    trained_on_graphs: int = 0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: str | Path | None) -> "UnifiedParserModel | None":
        if not path:
            return None
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict) and "weights" in payload:
            payload = payload["weights"]
        return cls(**payload)


@dataclass
class UnifiedParserPrediction:
    intent: str = ""
    domain: str = ""
    hidden_goals: list[str] = field(default_factory=list)
    required_premises: list[str] = field(default_factory=list)
    operator_families: list[tuple[str, float]] = field(default_factory=list)
    operator_decompositions: list[tuple[str, float]] = field(default_factory=list)
    confidence: float = 0.0
    matched_tokens: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedParserTrainingSummary:
    output_path: str
    trained_on_graphs: int
    model: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class UnifiedParserTrainer:
    def train_from_memory(
        self,
        memory_store: CorpusMemoryStore,
        output_path: str | Path,
        source: str | None = None,
        split: str = "train",
    ) -> UnifiedParserTrainingSummary:
        graphs = memory_store.fetch_graphs(split=split, source=source)
        return self.train_from_graphs(graphs, output_path)

    def train_from_graphs(
        self,
        graphs: Sequence[StructuredMeaningGraph],
        output_path: str | Path,
    ) -> UnifiedParserTrainingSummary:
        token_intents: dict[str, dict[str, float]] = {}
        token_domains: dict[str, dict[str, float]] = {}
        token_goals: dict[str, dict[str, float]] = {}
        token_required_premises: dict[str, dict[str, float]] = {}
        token_operator_families: dict[str, dict[str, float]] = {}
        token_decompositions: dict[str, dict[str, float]] = {}

        graphs = list(graphs)
        for graph in graphs:
            tokens = set(_tokenize(f"{graph.query} {graph.source_context}".strip()))
            if not tokens:
                continue
            weight = float(graph.operator_execution.composition_score) if graph.operator_execution is not None and graph.operator_execution.composition_score > 0.0 else 1.0
            weight = max(0.5, min(1.0, weight))
            for token in tokens:
                self._bump(token_intents, token, graph.intent, weight)
                self._bump(token_domains, token, graph.domain or "general", weight)
                for goal in graph.hidden_goals[:3]:
                    self._bump(token_goals, token, goal, weight)
                for premise in graph.required_premises[:4]:
                    self._bump(token_required_premises, token, premise, weight)
                for candidate in graph.induced_operators[:6]:
                    self._bump(token_operator_families, token, candidate.family, candidate.confidence)
                for decomposition in graph.operator_decompositions[:4]:
                    self._bump(token_decompositions, token, decomposition.operator_name, decomposition.confidence)

        model = UnifiedParserModel(
            token_intents=token_intents,
            token_domains=token_domains,
            token_goals=token_goals,
            token_required_premises=token_required_premises,
            token_operator_families=token_operator_families,
            token_decompositions=token_decompositions,
            trained_on_graphs=len(graphs),
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"weights": model.model_dump()}, ensure_ascii=False, indent=2), encoding="utf-8")
        return UnifiedParserTrainingSummary(
            output_path=str(output),
            trained_on_graphs=len(graphs),
            model=model.model_dump(),
        )

    @staticmethod
    def _bump(table: dict[str, dict[str, float]], token: str, key: str, amount: float) -> None:
        if not token or not key:
            return
        bucket = table.setdefault(token, {})
        bucket[key] = round(float(bucket.get(key, 0.0)) + float(amount), 4)


class LearnedUnifiedParser:
    def __init__(self, model: UnifiedParserModel | None = None, model_path: str | Path | None = None) -> None:
        self.model = model or UnifiedParserModel.from_path(model_path) or UnifiedParserModel()

    def predict(self, query: str, source_context: str = "") -> UnifiedParserPrediction:
        tokens = _tokenize(f"{query} {source_context}".strip())
        if not tokens:
            return UnifiedParserPrediction()
        matched_tokens = sorted({
            token
            for token in tokens
            if token in self.model.token_intents
            or token in self.model.token_domains
            or token in self.model.token_goals
            or token in self.model.token_required_premises
            or token in self.model.token_operator_families
            or token in self.model.token_decompositions
        })
        intent_votes = self._rank_votes(tokens, self.model.token_intents)
        domain_votes = self._rank_votes(tokens, self.model.token_domains)
        goal_votes = self._rank_votes(tokens, self.model.token_goals)
        premise_votes = self._rank_votes(tokens, self.model.token_required_premises)
        family_votes = self._rank_votes(tokens, self.model.token_operator_families)
        decomposition_votes = self._rank_votes(tokens, self.model.token_decompositions)

        confidence_signals = [
            self._confidence_from_votes(intent_votes, len(tokens)),
            self._confidence_from_votes(domain_votes, len(tokens)),
            self._confidence_from_votes(goal_votes, len(tokens)),
            self._confidence_from_votes(premise_votes, len(tokens)),
            self._confidence_from_votes(decomposition_votes, len(tokens)),
        ]
        active_signals = [value for value in confidence_signals if value > 0.0]
        coverage = len(matched_tokens) / float(len(set(tokens)) or 1)
        confidence = round((0.55 * (sum(active_signals) / float(len(active_signals) or 1))) + (0.45 * coverage), 4) if active_signals else 0.0
        return UnifiedParserPrediction(
            intent=intent_votes[0][0] if intent_votes else "",
            domain=domain_votes[0][0] if domain_votes else "",
            hidden_goals=[goal for goal, score in goal_votes[:3] if score >= self.model.min_vote],
            required_premises=[premise for premise, score in premise_votes[:4] if score >= self.model.min_vote],
            operator_families=[item for item in family_votes[:4] if item[1] >= self.model.min_vote],
            operator_decompositions=[item for item in decomposition_votes[:3] if item[1] >= self.model.min_vote],
            confidence=confidence,
            matched_tokens=matched_tokens,
        )

    def bootstrap_graph(self, query: str, source_context: str = "", prediction: UnifiedParserPrediction | None = None) -> StructuredMeaningGraph:
        prediction = prediction or self.predict(query, source_context=source_context)
        graph = StructuredMeaningGraph(
            query=query,
            intent=prediction.intent or "generic_reasoning",
            domain=prediction.domain or "general",
            source_context=source_context,
        )
        graph.add_node(Node(id="question", label=query, kind="query", attributes={"text": query}, provenance=["unified_parser:bootstrap"]))
        for goal in prediction.hidden_goals:
            if goal not in graph.hidden_goals:
                graph.hidden_goals.append(goal)
        for premise in prediction.required_premises:
            if premise not in graph.required_premises:
                graph.required_premises.append(premise)
        goal_source = graph.hidden_goals[0] if graph.hidden_goals else "question"
        for premise in graph.required_premises[:4]:
            graph.add_edge(Edge(source=goal_source, relation="REQUIRES", target=premise, confidence=round(min(0.88, 0.45 + prediction.confidence * 0.35), 2), provenance=["unified_parser:bootstrap"]))
        for family, score in prediction.operator_families[:3]:
            graph.induced_operators.append(
                OperatorCandidate(
                    name=f"UNIFIED_{family.upper()}_BOOTSTRAP",
                    family=family,
                    arity=2,
                    input_types=[graph.intent],
                    output_type="relation_frame",
                    description="Parser-first operator-family prior recovered from learned graph-slot supervision.",
                    confidence=round(min(0.9, 0.42 + 0.1 * score), 2),
                    provenance=["unified_parser:bootstrap_family"],
                )
            )
        for operator_name, score in prediction.operator_decompositions[:2]:
            basis = self._decomposition_basis(operator_name, graph)
            if not basis:
                continue
            graph.operator_decompositions.append(
                OperatorDecomposition(
                    operator_name=operator_name,
                    basis_operators=basis,
                    rationale="unified parser bootstrap: retained from learned graph-slot supervision.",
                    confidence=round(min(0.88, 0.44 + 0.1 * score), 2),
                )
            )
        note = f"unified parser: parser-first bootstrap activated (confidence={prediction.confidence:.2f})"
        if note not in graph.audit_trace:
            graph.audit_trace.append(note)
        return graph

    def enrich(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        prediction = self.predict(graph.query, source_context=graph.source_context)
        if prediction.intent and graph.intent in {"generic_reasoning", "goal_directed_reasoning", "constraint_reasoning"}:
            graph.intent = prediction.intent
        if prediction.domain and graph.domain == "general":
            graph.domain = prediction.domain
        for goal in prediction.hidden_goals:
            if goal not in graph.hidden_goals:
                graph.hidden_goals.append(goal)
        for premise in prediction.required_premises:
            if premise not in graph.required_premises and premise not in graph.satisfied_premises:
                graph.required_premises.append(premise)
        goal_source = graph.hidden_goals[0] if graph.hidden_goals else "question"
        if goal_source:
            existing_relations = {(edge.source, edge.relation, edge.target) for edge in graph.edges}
            for premise in graph.required_premises[:4]:
                triple = (goal_source, "REQUIRES", premise)
                if triple in existing_relations:
                    continue
                graph.add_edge(Edge(source=goal_source, relation="REQUIRES", target=premise, confidence=round(min(0.82, 0.4 + prediction.confidence * 0.3), 2), provenance=["unified_parser:required_premise"]))
        existing_families = {candidate.family for candidate in graph.induced_operators}
        for family, score in prediction.operator_families[:4]:
            if family in existing_families:
                continue
            graph.induced_operators.append(
                OperatorCandidate(
                    name=f"UNIFIED_{family.upper()}_PRIOR",
                    family=family,
                    arity=2,
                    input_types=[graph.intent],
                    output_type="relation_frame",
                    description="Learned unified-parser prior recovered from graph-slot supervision.",
                    confidence=round(min(0.86, 0.4 + 0.1 * score), 2),
                    provenance=["unified_parser:token_family"],
                )
            )
            existing_families.add(family)
        existing_decompositions = {item.operator_name for item in graph.operator_decompositions}
        for operator_name, score in prediction.operator_decompositions[:3]:
            if operator_name in existing_decompositions:
                continue
            basis = self._decomposition_basis(operator_name, graph)
            if not basis:
                continue
            graph.operator_decompositions.append(
                OperatorDecomposition(
                    operator_name=operator_name,
                    basis_operators=basis,
                    rationale="unified parser prior: predicted from learned graph-slot supervision.",
                    confidence=round(min(0.84, 0.4 + 0.08 * score), 2),
                )
            )
            existing_decompositions.add(operator_name)
        note = f"unified parser: applied learned graph-slot priors (confidence={prediction.confidence:.2f})"
        if note not in graph.audit_trace:
            graph.audit_trace.append(note)
        return graph

    @staticmethod
    def _decomposition_basis(operator_name: str, graph: StructuredMeaningGraph) -> list[str]:
        if operator_name == "GOAL_PRESERVATION_OPERATOR":
            basis = ["HIDDEN_GOAL"] if graph.hidden_goals else []
            if graph.required_premises or any(edge.relation == "REQUIRES" for edge in graph.edges):
                basis.append("REQUIRES")
            if graph.missing_premises or any(edge.relation == "BLOCKED_BY" for edge in graph.edges):
                basis.append("BLOCKED_BY")
            return list(dict.fromkeys(basis))
        if operator_name == "symbolic_evidence_grounder" and graph.source_context.strip():
            return ["DOCUMENT_CONTEXT", "REQUIRES"]
        return []

    @staticmethod
    def _confidence_from_votes(ranked: list[tuple[str, float]], token_count: int) -> float:
        if not ranked:
            return 0.0
        top = float(ranked[0][1])
        second = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        coverage = min(1.0, top / float(max(1, token_count)))
        margin = 1.0 if top <= 0.0 else max(0.0, (top - second) / top)
        return round((0.55 * coverage) + (0.45 * margin), 4)

    @staticmethod
    def _rank_votes(tokens: list[str], table: dict[str, dict[str, float]]) -> list[tuple[str, float]]:
        votes: dict[str, float] = {}
        for token in tokens:
            for key, value in table.get(token, {}).items():
                votes[key] = float(votes.get(key, 0.0)) + float(value)
        return sorted(votes.items(), key=lambda item: (-item[1], item[0]))

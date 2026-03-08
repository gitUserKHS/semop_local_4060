from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .structures import Edge, StructuredMeaningGraph


@dataclass(frozen=True)
class SemanticOperator:
    name: str
    arity: int
    description: str
    input_types: Sequence[str] = ()
    output_type: str = "meaning_state"

    def apply(self, graph: StructuredMeaningGraph, args: List[str], provenance: List[str] | None = None) -> None:
        if len(args) != self.arity:
            raise ValueError(f"{self.name} expects {self.arity} args, got {len(args)}")
        if self.arity == 2:
            graph.add_edge(
                Edge(
                    source=args[0],
                    relation=self.name,
                    target=args[1],
                    provenance=list(dict.fromkeys((provenance or []) + [f"operator:{self.name}"])),
                )
            )
        elif self.arity == 1:
            graph.add_edge(
                Edge(
                    source=args[0],
                    relation=self.name,
                    target="TRUE",
                    provenance=list(dict.fromkeys((provenance or []) + [f"operator:{self.name}"])),
                )
            )
        graph.semantic_operators.append(self.name)


OPERATORS: Dict[str, SemanticOperator] = {
    "PART_OF": SemanticOperator("PART_OF", 2, "part-whole relation", ["part", "whole"], "structured_state"),
    "CONTAINS": SemanticOperator("CONTAINS", 2, "containment relation", ["container", "content"], "structured_state"),
    "AFFORDS": SemanticOperator("AFFORDS", 2, "entity affords action", ["entity", "action"], "action_space"),
    "REQUIRES": SemanticOperator("REQUIRES", 2, "entity or plan requires precondition", ["plan", "condition"], "constraint_checked_plan"),
    "BLOCKED_BY": SemanticOperator("BLOCKED_BY", 2, "plan or route is blocked by obstacle", ["plan", "obstacle"], "redirected_plan"),
    "ALTERNATIVE": SemanticOperator("ALTERNATIVE", 2, "alternative plan relation", ["plan", "alternative"], "alternative_plan"),
    "GOAL_OF": SemanticOperator("GOAL_OF", 2, "goal relation", ["task", "agent"], "goal_binding"),
}


def register_operator(name: str, arity: int, description: str, input_types: Sequence[str] = (), output_type: str = "meaning_state") -> None:
    OPERATORS[name] = SemanticOperator(name, arity, description, input_types, output_type)


def apply_operator(graph: StructuredMeaningGraph, name: str, *args: str, provenance: List[str] | None = None) -> None:
    op = OPERATORS.get(name)
    if op is None:
        raise KeyError(f"Unknown operator: {name}")
    op.apply(graph, list(args), provenance=provenance)


def apply_many(graph: StructuredMeaningGraph, triples: Iterable[tuple]) -> None:
    for item in triples:
        if len(item) == 4:
            source, relation, target, provenance = item
        else:
            source, relation, target = item
            provenance = []
        if relation not in OPERATORS:
            graph.add_edge(Edge(source=source, relation=relation, target=target, provenance=list(provenance)))
        else:
            apply_operator(graph, relation, source, target, provenance=list(provenance))
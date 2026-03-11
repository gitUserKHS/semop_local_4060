from __future__ import annotations

from ..pipeline import StructuredMeaningPipeline
from ..structures import StructuredMeaningGraph
from .types import SharedWorldModel, VLSOEntity, VLSOOperator, VLSORelation


class VLSOLanguageParser:
    QUESTION_STOPWORDS = {
        "a",
        "an",
        "are",
        "do",
        "does",
        "did",
        "here",
        "how",
        "is",
        "or",
        "the",
        "there",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
    }

    def __init__(self, mode: str = "heuristic") -> None:
        self.pipeline = StructuredMeaningPipeline(mode=mode)

    def parse(self, query: str) -> tuple[SharedWorldModel, StructuredMeaningGraph]:
        graph = self.pipeline.run(query)
        model = SharedWorldModel(query=query)
        model.goals.extend(graph.hidden_goals or [])
        model.goals.extend(item for item in (graph.candidate_actions or []) if item not in model.goals)
        model.constraints.extend(graph.invalid_advice)
        model.constraints.extend(item for item in graph.required_premises if item not in model.constraints)
        model.warnings.extend(graph.warnings)
        model.metadata['hidden_premises'] = list(graph.hidden_assumptions)
        model.metadata['goal_preservation_checks'] = [item.__dict__ for item in graph.goal_preservation_checks]
        model.metadata['operator_decompositions'] = [item.__dict__ for item in graph.operator_decompositions]
        model.metadata['functor_hypotheses'] = [item.__dict__ for item in graph.functor_hypotheses]
        model.audit_trace.extend(graph.audit_trace)
        referenced_node_ids = {edge.source for edge in graph.edges} | {edge.target for edge in graph.edges}
        for node in graph.nodes:
            if self._should_skip_node(node.id, node.label, node.kind, node.attributes, referenced_node_ids):
                continue
            model.add_entity(
                VLSOEntity(
                    id=node.id,
                    label=node.label,
                    modality="language",
                    entity_type=node.kind or "unknown",
                    attributes=dict(node.attributes),
                )
            )
        for edge in graph.edges:
            model.add_relation(
                VLSORelation(
                    source=edge.source,
                    relation=edge.relation,
                    target=edge.target,
                    modality="language",
                    confidence=edge.confidence,
                    attributes=dict(edge.attributes),
                )
            )
        for item in graph.semantic_operators:
            model.add_operator(VLSOOperator(name=item, axis="action", description=item, source_modality="language", confidence=0.7))
        for item in graph.induced_operators:
            model.add_operator(
                VLSOOperator(
                    name=item.name,
                    axis=item.family,
                    description=item.description,
                    source_modality="language",
                    confidence=item.confidence,
                )
            )
        for plan_step in graph.plan:
            model.inferred_steps.append(plan_step.action)
        return model, graph

    def _should_skip_node(
        self,
        node_id: str,
        label: str,
        kind: str | None,
        attributes: dict,
        referenced_node_ids: set[str],
    ) -> bool:
        lowered = (label or node_id).strip().lower()
        if lowered not in self.QUESTION_STOPWORDS:
            return False
        if node_id in referenced_node_ids:
            return False
        if attributes:
            return False
        return kind in {None, "", "concept", "unknown"}

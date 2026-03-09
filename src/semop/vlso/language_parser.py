from __future__ import annotations

from ..pipeline import StructuredMeaningPipeline
from ..structures import StructuredMeaningGraph
from .types import SharedWorldModel, VLSOEntity, VLSOOperator, VLSORelation


class VLSOLanguageParser:
    def __init__(self, mode: str = "heuristic") -> None:
        self.pipeline = StructuredMeaningPipeline(mode=mode)

    def parse(self, query: str) -> tuple[SharedWorldModel, StructuredMeaningGraph]:
        graph = self.pipeline.run(query)
        model = SharedWorldModel(query=query)
        model.goals.extend(graph.candidate_actions or [])
        model.constraints.extend(graph.invalid_advice)
        model.warnings.extend(graph.warnings)
        model.audit_trace.extend(graph.audit_trace)
        for node in graph.nodes:
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

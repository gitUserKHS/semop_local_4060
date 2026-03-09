from __future__ import annotations

from .types import SharedWorldModel, VLSOEntity, VLSORelation


class VLSOAligner:
    def align(self, language_model: SharedWorldModel, visual_model: SharedWorldModel | None = None) -> SharedWorldModel:
        world = SharedWorldModel(query=language_model.query)
        self._merge(world, language_model)
        if visual_model is not None:
            self._merge(world, visual_model)
            world.audit_trace.append("merged language and visual operator spaces")
        self._infer_cross_modal_steps(world)
        return world

    def _merge(self, destination: SharedWorldModel, source: SharedWorldModel) -> None:
        for entity in source.entities:
            destination.add_entity(entity)
        for relation in source.relations:
            destination.add_relation(relation)
        for operator in source.operators:
            destination.add_operator(operator)
        destination.goals.extend(item for item in source.goals if item not in destination.goals)
        destination.constraints.extend(item for item in source.constraints if item not in destination.constraints)
        destination.warnings.extend(item for item in source.warnings if item not in destination.warnings)
        destination.audit_trace.extend(item for item in source.audit_trace if item not in destination.audit_trace)
        destination.inferred_steps.extend(item for item in source.inferred_steps if item not in destination.inferred_steps)

    def _infer_cross_modal_steps(self, world: SharedWorldModel) -> None:
        relations = world.relation_tuples()
        requires_open = any(relation == "REQUIRES" and target == "open_access" for _, relation, target in relations)
        zipper_closed = any(relation == "STATE" and target.endswith(":closed") for _, relation, target in relations)
        zipper_part = any(relation == "PART_OF" and target == "bag" for _, relation, target in relations)
        openable = any(operator.name == "OPENABLE" for operator in world.operators)
        if requires_open and zipper_closed and zipper_part and openable:
            world.inferred_steps.insert(0, "Open the zipper before attempting the insertion.")
            world.constraints.append("open_access_required")
            world.audit_trace.append("cross-modal inference: language prerequisite matched visual closed zipper")
        if any(item == "path_blocked" for item in world.constraints):
            world.warnings.append("visual context reports a blocked path; route selection or delay may be required")
        if any(relation == "PART_OF" and target == "bag" for _, relation, target in relations):
            world.add_relation(VLSORelation(source="bag", relation="HAS", target="interior", modality="shared", confidence=0.6))
            world.add_entity(VLSOEntity(id="interior", label="interior", modality="shared", entity_type="region"))

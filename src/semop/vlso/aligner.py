from __future__ import annotations

from .types import SharedWorldModel, VLSOEntity, VLSOEvent, VLSORelation


class VLSOAligner:
    def align(self, language_model: SharedWorldModel, visual_model: SharedWorldModel | None = None) -> SharedWorldModel:
        world = SharedWorldModel(query=language_model.query)
        self._merge(world, language_model)
        if visual_model is not None:
            self._merge(world, visual_model)
            world.audit_trace.append("merged language and visual operator spaces")
        self._infer_cross_modal_steps(world)
        self._align_hidden_premises(world)
        return world

    def _merge(self, destination: SharedWorldModel, source: SharedWorldModel) -> None:
        for entity in source.entities:
            destination.add_entity(entity)
        for relation in source.relations:
            destination.add_relation(relation)
        for operator in source.operators:
            destination.add_operator(operator)
        for event in getattr(source, 'events', []):
            destination.add_event(event)
        destination.goals.extend(item for item in source.goals if item not in destination.goals)
        destination.constraints.extend(item for item in source.constraints if item not in destination.constraints)
        destination.warnings.extend(item for item in source.warnings if item not in destination.warnings)
        destination.audit_trace.extend(item for item in source.audit_trace if item not in destination.audit_trace)
        destination.inferred_steps.extend(item for item in source.inferred_steps if item not in destination.inferred_steps)
        for key, value in source.metadata.items():
            if key not in destination.metadata:
                destination.metadata[key] = value
            elif isinstance(destination.metadata[key], list) and isinstance(value, list):
                for item in value:
                    if item not in destination.metadata[key]:
                        destination.metadata[key].append(item)
            elif isinstance(destination.metadata[key], dict) and isinstance(value, dict):
                merged = dict(destination.metadata[key])
                merged.update(value)
                destination.metadata[key] = merged

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
        if any(relation == "AFFORDS" and target == "open_access_action" for _, relation, target in relations):
            world.inferred_steps.append("The visual world model includes an access-opening affordance.")
        if getattr(world, 'events', []):
            world.audit_trace.append(f'cross-modal temporal merge preserved {len(world.events)} event(s)')

    def _align_hidden_premises(self, world: SharedWorldModel) -> None:
        hidden_premises = world.metadata.get('hidden_premises', []) if isinstance(world.metadata, dict) else []
        required_premises = world.metadata.get('required_premises', []) if isinstance(world.metadata, dict) else []
        premise_validations = world.metadata.get('premise_validations', []) if isinstance(world.metadata, dict) else []
        goal_checks = world.metadata.get('goal_preservation_checks', []) if isinstance(world.metadata, dict) else []
        clarification_score = float(world.metadata.get('clarification_score', 0.0)) if isinstance(world.metadata, dict) else 0.0
        clarification_reasons = world.metadata.get('clarification_reasons', []) if isinstance(world.metadata, dict) else []
        functors = world.metadata.get('functor_hypotheses', []) if isinstance(world.metadata, dict) else []
        operator_names = {item.name for item in world.operators}
        hidden_goals = {str(item) for item in world.goals}
        if hidden_premises:
            world.audit_trace.append('cross-modal alignment: hidden premises projected into shared world model')
        for premise in required_premises:
            if premise not in world.constraints:
                world.constraints.append(premise)
        if any(item.get('name') == 'VisualStructureToActionFunctor' for item in functors if isinstance(item, dict)):
            if 'ACCESS_PORT_OPERATOR' in operator_names and 'ACCESS_CONTROL_OPERATOR' in operator_names:
                world.inferred_steps.append('Visual structure and language preconditions both indicate an access-first action sequence.')
                if 'cross_modal_access_alignment' not in world.constraints:
                    world.constraints.append('cross_modal_access_alignment')
        if any(item.get('premise') == 'open_access' and item.get('status') != 'contradicted' for item in premise_validations if isinstance(item, dict)):
            if 'open_access_required' not in world.constraints:
                world.constraints.append('open_access_required')
        if any(item.get('premise') == 'vehicle_present' and item.get('status') == 'supported' for item in premise_validations if isinstance(item, dict)):
            world.warnings.append('Goal-preservation alignment: service success still depends on the target vehicle being present.')
        if any(item.get('action') == 'walk_without_car' and item.get('status') == 'risk_high' for item in goal_checks if isinstance(item, dict)):
            world.warnings.append('Hidden-goal check: movement without the target object may fail the real service goal.')
        if any(item.get('action') == 'insert_without_opening' and item.get('status') == 'risk_high' for item in goal_checks if isinstance(item, dict)):
            world.warnings.append('Hidden-goal check: insertion without opening access may fail the containment goal.')
        if any(item.get('action') == 'retrieve_without_opening' and item.get('status') == 'risk_high' for item in goal_checks if isinstance(item, dict)):
            world.warnings.append('Hidden-goal check: retrieval without opening access may fail the access goal.')
        if any(item.get('action') == 'pour_without_uncapping' and item.get('status') == 'risk_high' for item in goal_checks if isinstance(item, dict)):
            world.warnings.append('Hidden-goal check: pouring without removing the cap or lid may fail the access goal.')
        if 'retrieve_item_from_cabinet_goal' in hidden_goals and {'ACCESS_CONTROL_OPERATOR', 'ACCESS_PORT_OPERATOR'} & operator_names:
            world.inferred_steps.append('Open the cabinet access-control part before retrieving the item inside.')
            if 'cabinet_access_alignment' not in world.constraints:
                world.constraints.append('cabinet_access_alignment')
        access_goal_labels = {
            'retrieve_item_from_box_goal': 'Open the box access-control part before retrieving the item inside.',
            'retrieve_item_from_bin_goal': 'Open or lift the bin cover before retrieving the item inside.',
            'retrieve_item_from_pouch_goal': 'Open the pouch access-control part before retrieving the item inside.',
            'retrieve_item_from_suitcase_goal': 'Open the suitcase access-control part before retrieving the item inside.',
        }
        for goal_name, step_text in access_goal_labels.items():
            if goal_name in hidden_goals and {'ACCESS_CONTROL_OPERATOR', 'ACCESS_PORT_OPERATOR'} & operator_names:
                world.inferred_steps.append(step_text)
                constraint_name = goal_name.replace('retrieve_item_from_', '').replace('_goal', '') + '_access_alignment'
                if constraint_name not in world.constraints:
                    world.constraints.append(constraint_name)
        if 'pour_from_bottle_goal' in hidden_goals and {'ACCESS_CONTROL_OPERATOR', 'ACCESS_PORT_OPERATOR'} & operator_names:
            world.inferred_steps.append('Remove or open the cap/control part before pouring from the container.')
            if 'capped_access_alignment' not in world.constraints:
                world.constraints.append('capped_access_alignment')
        if clarification_score >= 0.5:
            world.warnings.append(f'Clarification recommended: ambiguity remains high (score={clarification_score:.2f}).')
            for reason in clarification_reasons[:2]:
                world.audit_trace.append('clarification cue: ' + str(reason))

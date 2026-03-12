from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .affordance_classifier import WeakAffordanceClassifier
from .affordance_features import VisualAffordanceFeatureExtractor
from .concept_memory import VisualConceptMemory
from .hybrid_memory import VisualHybridMemory
from .operator_learning import VisualOperatorMemory
from .predictive_priors import JepaStructuralPredictor
from .structural_operators import VisualStructuralOperatorInducer
from .types import VisualObservation


@dataclass
class VisualObjectReasoningResult:
    inferred_parts: List[dict]
    inferred_affordances: List[dict]
    inferred_constraints: List[str]
    structural_operators: List[dict]
    audit_trace: List[str]


class VisualObjectReasoner:
    def __init__(
        self,
        classifier: WeakAffordanceClassifier | None = None,
        feature_extractor: VisualAffordanceFeatureExtractor | None = None,
        concept_memory: VisualConceptMemory | None = None,
        operator_memory: VisualOperatorMemory | None = None,
    ) -> None:
        self.classifier = classifier or WeakAffordanceClassifier()
        self.feature_extractor = feature_extractor or VisualAffordanceFeatureExtractor()
        self.concept_memory = concept_memory
        self.operator_memory = operator_memory
        self.hybrid_memory = VisualHybridMemory(concept_memory=concept_memory, operator_memory=operator_memory) if (concept_memory is not None or operator_memory is not None) else None
        self.structural_inducer = VisualStructuralOperatorInducer()
        self.predictive_prior = JepaStructuralPredictor(operator_memory=operator_memory) if operator_memory is not None else None

    def analyze(self, observation: VisualObservation) -> VisualObjectReasoningResult:
        candidates = self.feature_extractor.extract(observation)
        inferred_parts: List[dict] = []
        inferred_affordances: List[dict] = []
        inferred_constraints: List[str] = []
        audit_trace: List[str] = []
        classifier_outputs = []
        concept_outputs = []
        operator_outputs = []
        structural_result = self.structural_inducer.analyze(observation, candidates)
        inferred_parts.extend(structural_result.inferred_parts)
        inferred_affordances.extend(structural_result.inferred_affordances)
        inferred_constraints.extend(structural_result.inferred_constraints)
        structural_bindings = list(structural_result.bindings)
        audit_trace.extend(structural_result.audit_trace)
        if self.predictive_prior is not None:
            predictive_result = self.predictive_prior.predict(observation, candidates)
            structural_bindings.extend(predictive_result.bindings)
            inferred_affordances.extend(predictive_result.inferred_affordances)
            inferred_constraints.extend(predictive_result.inferred_constraints)
            audit_trace.extend(predictive_result.audit_trace)
            if predictive_result.bindings:
                observation.metadata['predictive_operator_priors'] = [item.model_dump() for item in predictive_result.bindings]
        structural_operators = [item.model_dump() for item in structural_bindings]
        self._inject_real_image_archetypes(candidates, inferred_parts, inferred_affordances, inferred_constraints)
        for candidate in candidates:
            predictions = self.classifier.predict(candidate.features, limit=4, threshold=0.58)
            if predictions:
                predictions = [item for item in predictions if self._accept_prediction(item.label, candidate)]
            if predictions:
                classifier_outputs.append(
                    {
                        'subject': candidate.subject,
                        'parent': candidate.parent,
                        'labels': [
                            {'label': item.label, 'confidence': item.confidence, 'score': item.score}
                            for item in predictions
                        ],
                    }
                )
                for prediction in predictions:
                    inferred_affordances.append({'subject': candidate.subject, 'value': prediction.label})
                    self._apply_label_side_effects(prediction.label, candidate.parent, inferred_parts, inferred_affordances, inferred_constraints, candidate.subject)

            signature = self._candidate_signature(candidate)
            if self.hybrid_memory is not None:
                hybrid = self.hybrid_memory.retrieve(candidate.features, signature, limit=3)
                accepted_local = [item for item in hybrid.local_matches if self._accept_prediction(item.label, candidate)]
                accepted_fused = []
                for row in hybrid.fused_labels:
                    label = str(row.get('label', ''))
                    if label and self._accept_prediction(label, candidate):
                        accepted_fused.append(row)
                if accepted_local:
                    concept_outputs.append(
                        {
                            'subject': candidate.subject,
                            'parent': candidate.parent,
                            'matches': [item.model_dump() for item in accepted_local],
                        }
                    )
                    for match in accepted_local:
                        inferred_affordances.append({'subject': candidate.subject, 'value': match.label})
                        self._apply_label_side_effects(match.label, candidate.parent, inferred_parts, inferred_affordances, inferred_constraints, candidate.subject)
                if hybrid.global_matches:
                    operator_outputs.append(
                        {
                            'subject': candidate.subject,
                            'parent': candidate.parent,
                            'signature': signature,
                            'matches': [item.model_dump() for item in hybrid.global_matches],
                            'fused_labels': accepted_fused,
                        }
                    )
                for row in accepted_fused:
                    label = str(row.get('label', ''))
                    score = float(row.get('score', 0.0) or 0.0)
                    if not label or score < 0.22:
                        continue
                    inferred_affordances.append({'subject': candidate.subject, 'value': label})
                    self._apply_label_side_effects(label, candidate.parent, inferred_parts, inferred_affordances, inferred_constraints, candidate.subject)
                for match in hybrid.global_matches:
                    self._apply_operator_match(match.metadata, candidate.subject, candidate.parent, inferred_parts, inferred_affordances, inferred_constraints)

            if candidate.features.get('boundary_attached', 0.0) >= 1.0 and candidate.features.get('inside_parent', 0.0) >= 1.0:
                inferred_affordances.append({'subject': candidate.subject, 'value': 'EDGE_OPENING'})
            if candidate.features.get('horizontal_elongation', 0.0) >= 2.4 or candidate.features.get('vertical_elongation', 0.0) >= 2.4:
                inferred_affordances.append({'subject': candidate.subject, 'value': 'GRASPABLE_PART'})
        if inferred_parts:
            inferred_constraints.append('nested_object_structure_detected')
        if classifier_outputs:
            observation.metadata['affordance_classifier'] = classifier_outputs
            audit_trace.append('weak affordance classifier added learned priors')
        if concept_outputs:
            observation.metadata['concept_memory_matches'] = concept_outputs
            audit_trace.append('few-shot visual concept memory matched prior exemplars')
        if operator_outputs:
            observation.metadata['operator_memory_matches'] = operator_outputs
            audit_trace.append('visual operator prototypes matched relational scene patterns')
        if concept_outputs and operator_outputs:
            audit_trace.append('hybrid memory fused local exemplars with compressed operator prototypes')
        if inferred_affordances and not audit_trace:
            audit_trace.append('visual object reasoner inferred part and affordance hypotheses')
        return VisualObjectReasoningResult(
            inferred_parts=self._dedupe_dicts(inferred_parts),
            inferred_affordances=self._dedupe_dicts(inferred_affordances),
            inferred_constraints=list(dict.fromkeys(inferred_constraints)),
            structural_operators=structural_operators,
            audit_trace=list(dict.fromkeys(audit_trace)),
        )

    def enrich_observation(self, observation: VisualObservation) -> VisualObservation:
        result = self.analyze(observation)
        object_map = {
            str(item.get('id') or item.get('label') or item.get('name')): item
            for item in observation.objects
            if item.get('id') or item.get('label') or item.get('name')
        }
        for item in result.inferred_parts:
            parent = object_map.get(item['parent'])
            child = object_map.get(item['child'])
            if not parent or not child:
                continue
            parent.setdefault('parts', [])
            if item['child'] not in parent['parts']:
                parent['parts'].append(item['child'])
            if parent.get('kind') in {None, 'object', 'shape'}:
                parent['kind'] = 'container'
            if child.get('kind') in {None, 'object', 'shape'}:
                child['kind'] = 'part'
        for affordance in result.inferred_affordances:
            if affordance not in observation.affordances:
                observation.affordances.append(affordance)
            subject = object_map.get(affordance['subject'])
            if subject is not None:
                subject.setdefault('concept_labels', [])
                if affordance['value'] not in subject['concept_labels']:
                    subject['concept_labels'].append(affordance['value'])
                kind = self._label_to_kind(affordance['value'])
                if kind and subject.get('kind') in {None, 'object', 'shape'}:
                    subject['kind'] = kind
        for constraint in result.inferred_constraints:
            if constraint not in observation.constraints:
                observation.constraints.append(constraint)
        if result.structural_operators:
            observation.metadata['structural_operators'] = result.structural_operators
        if result.audit_trace:
            observation.metadata.setdefault('object_reasoner_audit', [])
            for item in result.audit_trace:
                if item not in observation.metadata['object_reasoner_audit']:
                    observation.metadata['object_reasoner_audit'].append(item)
        return observation

    def _apply_label_side_effects(
        self,
        label: str,
        parent: str,
        inferred_parts: List[dict],
        inferred_affordances: List[dict],
        inferred_constraints: List[str],
        subject: str,
    ) -> None:
        upper = label.upper()
        is_container_like = any(token in upper for token in {'CONTAINER', 'BOX', 'DRAWER', 'CABINET', 'BOTTLE', 'CASE', 'BIN', 'JAR', 'POUCH', 'SUITCASE'})
        is_opening_like = any(token in upper for token in {'ZIPPER', 'OPENING', 'HINGE', 'CAP', 'LID', 'DOOR', 'PANEL'})
        is_grasp_like = any(token in upper for token in {'HANDLE', 'STRAP', 'GRASP', 'KNOB', 'PULL', 'HOOK', 'GRIP'})
        is_tool_like = any(token in upper for token in {'TOOL', 'BLADE', 'HEAD', 'HAMMER', 'SCREWDRIVER'})
        if is_container_like:
            inferred_constraints.append('container_like_object_detected')
            inferred_affordances.append({'subject': subject, 'value': 'HAS_INTERIOR'})
            if any(token in upper for token in {'BAG', 'SUITCASE', 'POUCH', 'CASE'}):
                inferred_affordances.append({'subject': subject, 'value': 'PORTABLE_CONTAINER'})
        if is_opening_like or is_grasp_like or 'WHEEL' in upper or is_tool_like:
            if parent:
                inferred_parts.append({'parent': parent, 'child': subject})
        if is_opening_like:
            inferred_constraints.append('opening_candidate_detected')
            if any(token in upper for token in {'CAP', 'LID', 'HINGE', 'DOOR'}):
                inferred_affordances.append({'subject': subject, 'value': 'ACCESS_CONTROL_PART'})
            if any(token in upper for token in {'HINGE', 'DOOR'}):
                inferred_constraints.append('hinged_access_detected')
            if any(token in upper for token in {'CAP', 'LID'}):
                inferred_constraints.append('closure_part_detected')
        if is_grasp_like:
            inferred_affordances.append({'subject': subject, 'value': 'GRASPABLE_PART'})
            if any(token in upper for token in {'HANDLE', 'KNOB', 'PULL', 'GRIP'}):
                inferred_constraints.append('handle_like_part_detected')
        if is_tool_like:
            inferred_constraints.append('tool_like_object_detected')
            if is_grasp_like:
                inferred_affordances.append({'subject': subject, 'value': 'TOOL_CONTROL_PART'})

    def _inject_real_image_archetypes(
        self,
        candidates,
        inferred_parts: List[dict],
        inferred_affordances: List[dict],
        inferred_constraints: List[str],
    ) -> None:
        children_by_parent: dict[str, list] = {}
        for candidate in candidates:
            if candidate.parent:
                children_by_parent.setdefault(candidate.parent, []).append(candidate)
        for candidate in candidates:
            features = candidate.features
            if features.get('is_dominant', 0.0) >= 1.0 and (
                features.get('large_region', 0.0) >= 1.0
                or features.get('rectilinear_bias', 0.0) >= 0.45
                or features.get('hole_count_norm', 0.0) > 0.0
            ):
                inferred_affordances.append({'subject': candidate.subject, 'value': 'STRUCTURAL_CONTAINER_CANDIDATE'})
                inferred_affordances.append({'subject': candidate.subject, 'value': 'HAS_INTERIOR'})
                inferred_constraints.append('container_like_object_detected')
        for parent, children in children_by_parent.items():
            opening_like: list[str] = []
            grasp_like: list[str] = []
            for child in children:
                features = child.features
                opening_rule = (
                    features.get('explicit_opening_hint', 0.0) >= 1.0
                    or features.get('top_strip_candidate', 0.0) >= 1.0
                    or (features.get('boundary_attached', 0.0) >= 1.0 and features.get('near_top_band', 0.0) >= 1.0 and features.get('horizontal_elongation', 0.0) >= 2.0)
                    or features.get('hole_count_norm', 0.0) > 0.0
                )
                control_rule = (
                    features.get('explicit_control_hint', 0.0) >= 1.0
                    or features.get('side_strip_candidate', 0.0) >= 1.0
                    or (features.get('boundary_attached', 0.0) >= 1.0 and features.get('near_side_band', 0.0) >= 1.0 and max(features.get('vertical_elongation', 0.0), features.get('horizontal_elongation', 0.0)) >= 1.8)
                )
                grasp_rule = (
                    features.get('explicit_grasp_hint', 0.0) >= 1.0
                    or features.get('side_strip_candidate', 0.0) >= 1.0
                    or (features.get('near_side_band', 0.0) >= 1.0 and max(features.get('vertical_elongation', 0.0), features.get('horizontal_elongation', 0.0)) >= 1.8)
                    or (features.get('compactness', 0.0) <= 0.5 and features.get('small_region', 0.0) >= 1.0)
                )
                if opening_rule:
                    opening_like.append(child.subject)
                    inferred_affordances.append({'subject': child.subject, 'value': 'ACCESS_OPENING_CANDIDATE'})
                    inferred_constraints.append('opening_candidate_detected')
                if control_rule:
                    inferred_affordances.append({'subject': child.subject, 'value': 'ACCESS_CONTROL_PART'})
                    inferred_constraints.append('closure_part_detected')
                if grasp_rule:
                    grasp_like.append(child.subject)
                    inferred_affordances.append({'subject': child.subject, 'value': 'GRASPABLE_PART'})
                    inferred_affordances.append({'subject': child.subject, 'value': 'HANDLE_CANDIDATE'})
                    inferred_constraints.append('handle_like_part_detected')
                if opening_rule or control_rule or grasp_rule:
                    inferred_parts.append({'parent': parent, 'child': child.subject})
            if opening_like and grasp_like:
                inferred_affordances.append({'subject': parent, 'value': 'ACCESSIBLE_INTERIOR_PATH'})
                inferred_constraints.append('controlled_access_structure_detected')

    def _apply_operator_match(
        self,
        metadata: dict,
        subject: str,
        parent: str,
        inferred_parts: List[dict],
        inferred_affordances: List[dict],
        inferred_constraints: List[str],
    ) -> None:
        implied = metadata.get('implied_labels', [])
        if isinstance(implied, list):
            for row in implied:
                if not isinstance(row, dict):
                    continue
                label = str(row.get('label', ''))
                confidence = float(row.get('confidence', 0.0) or 0.0)
                if not label or confidence < 0.45:
                    continue
                if 'STRAP_LIKE_PART' in label.upper() and any(existing.get('value') == 'ACCESS_OPENING_CANDIDATE' and existing.get('subject') == subject for existing in inferred_affordances):
                    continue
                inferred_affordances.append({'subject': subject, 'value': label})
                self._apply_label_side_effects(label, parent, inferred_parts, inferred_affordances, inferred_constraints, subject)
        record_type = str(metadata.get('record_type', ''))
        if record_type == 'operator_prototype':
            inferred_constraints.append('operator_prototype_matched')

    @staticmethod
    def _label_to_kind(label: str) -> str | None:
        upper = label.upper()
        if any(token in upper for token in {'CONTAINER', 'BOX', 'DRAWER', 'CABINET', 'BOTTLE', 'JAR', 'BIN', 'POUCH', 'SUITCASE'}):
            return 'container'
        if any(token in upper for token in {'PART', 'HANDLE', 'ZIPPER', 'STRAP', 'WHEEL', 'OPENING', 'HINGE', 'DOOR', 'CAP', 'LID', 'KNOB', 'GRIP'}):
            return 'part'
        return None

    @staticmethod
    def _dedupe_dicts(items: List[dict]) -> List[dict]:
        seen = set()
        output = []
        for item in items:
            key = tuple(sorted(item.items()))
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
        return output

    @staticmethod
    def _candidate_signature(candidate) -> List[str]:
        features = candidate.features
        signature: list[str] = []
        if candidate.parent:
            signature.append('HAS_PARENT')
        if features.get('inside_parent', 0.0) >= 1.0:
            signature.append('INSIDE_PARENT')
        if features.get('boundary_attached', 0.0) >= 1.0:
            signature.append('BOUNDARY_ATTACHED')
        if features.get('near_top_band', 0.0) >= 1.0:
            signature.append('TOP_BAND')
        if features.get('near_side_band', 0.0) >= 1.0 and features.get('near_top_band', 0.0) < 1.0:
            signature.append('SIDE_BAND')
        if features.get('horizontal_elongation', 0.0) >= 2.3:
            signature.append('HORIZONTAL_ELONGATION')
        if features.get('vertical_elongation', 0.0) >= 2.3:
            signature.append('VERTICAL_ELONGATION')
        if features.get('touches_border', 0.0) >= 1.0:
            signature.append('TOUCHES_BORDER')
        return sorted(signature)

    @staticmethod
    def _accept_prediction(label: str, candidate) -> bool:
        upper = label.upper()
        features = candidate.features
        if any(token in upper for token in {"ZIPPER", "OPENING", "STRAP", "HANDLE", "GRASP"}):
            if not candidate.parent and features.get("touches_border", 0.0) >= 1.0:
                if features.get("area_ratio", 0.0) <= 0.03 and features.get('segmentation_confidence', 0.0) < 0.55:
                    return False
            if 'STRAP' in upper and features.get('top_strip_candidate', 0.0) >= 1.0 and features.get('explicit_grasp_hint', 0.0) < 1.0:
                return False
            if 'ZIPPER' in upper and features.get('top_strip_candidate', 0.0) < 1.0 and features.get('explicit_control_hint', 0.0) < 1.0 and features.get('hole_count_norm', 0.0) <= 0.0:
                return False
            if 'HANDLE' in upper and not candidate.parent and features.get('small_region', 0.0) >= 1.0:
                return False
        return True

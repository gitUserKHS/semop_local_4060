from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .affordance_classifier import WeakAffordanceClassifier
from .affordance_features import VisualAffordanceFeatureExtractor
from .concept_memory import VisualConceptMemory
from .types import VisualObservation


@dataclass
class VisualObjectReasoningResult:
    inferred_parts: List[dict]
    inferred_affordances: List[dict]
    inferred_constraints: List[str]
    audit_trace: List[str]


class VisualObjectReasoner:
    def __init__(
        self,
        classifier: WeakAffordanceClassifier | None = None,
        feature_extractor: VisualAffordanceFeatureExtractor | None = None,
        concept_memory: VisualConceptMemory | None = None,
    ) -> None:
        self.classifier = classifier or WeakAffordanceClassifier()
        self.feature_extractor = feature_extractor or VisualAffordanceFeatureExtractor()
        self.concept_memory = concept_memory

    def analyze(self, observation: VisualObservation) -> VisualObjectReasoningResult:
        candidates = self.feature_extractor.extract(observation)
        inferred_parts: List[dict] = []
        inferred_affordances: List[dict] = []
        inferred_constraints: List[str] = []
        audit_trace: List[str] = []
        classifier_outputs = []
        concept_outputs = []
        for candidate in candidates:
            predictions = self.classifier.predict(candidate.features, limit=4, threshold=0.58)
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

            if self.concept_memory is not None:
                matches = self.concept_memory.search(candidate.features, limit=3)
                strong_matches = [item for item in matches if item.score >= 0.86]
                if strong_matches:
                    concept_outputs.append(
                        {
                            'subject': candidate.subject,
                            'parent': candidate.parent,
                            'matches': [item.model_dump() for item in strong_matches],
                        }
                    )
                    for match in strong_matches:
                        inferred_affordances.append({'subject': candidate.subject, 'value': match.label})
                        self._apply_label_side_effects(match.label, candidate.parent, inferred_parts, inferred_affordances, inferred_constraints, candidate.subject)
                        for co_label in match.metadata.get('co_labels', []):
                            if isinstance(co_label, dict):
                                other_label = str(co_label.get('label', ''))
                                confidence = float(co_label.get('confidence', 0.0) or 0.0)
                                if other_label and confidence >= 0.5:
                                    inferred_affordances.append({'subject': candidate.subject, 'value': other_label})
                                    self._apply_label_side_effects(other_label, candidate.parent, inferred_parts, inferred_affordances, inferred_constraints, candidate.subject)

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
        if inferred_affordances and not audit_trace:
            audit_trace.append('visual object reasoner inferred part and affordance hypotheses')
        return VisualObjectReasoningResult(
            inferred_parts=self._dedupe_dicts(inferred_parts),
            inferred_affordances=self._dedupe_dicts(inferred_affordances),
            inferred_constraints=list(dict.fromkeys(inferred_constraints)),
            audit_trace=audit_trace,
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
        if 'CONTAINER' in upper:
            inferred_constraints.append('container_like_object_detected')
            inferred_affordances.append({'subject': subject, 'value': 'HAS_INTERIOR'})
        if any(token in upper for token in {'ZIPPER', 'STRAP', 'HANDLE', 'WHEEL', 'OPENING', 'GRASP'}):
            if parent:
                inferred_parts.append({'parent': parent, 'child': subject})
        if any(token in upper for token in {'ZIPPER', 'OPENING'}):
            inferred_constraints.append('opening_candidate_detected')
        if any(token in upper for token in {'HANDLE', 'STRAP', 'GRASP'}):
            inferred_affordances.append({'subject': subject, 'value': 'GRASPABLE_PART'})

    @staticmethod
    def _label_to_kind(label: str) -> str | None:
        upper = label.upper()
        if 'CONTAINER' in upper:
            return 'container'
        if any(token in upper for token in {'PART', 'HANDLE', 'ZIPPER', 'STRAP', 'WHEEL', 'OPENING'}):
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

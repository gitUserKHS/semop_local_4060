from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from .affordance_features import VisualAffordanceCandidate
from .types import VisualObservation


@dataclass
class StructuralOperatorBinding:
    operator_name: str
    subject: str
    parent: str = ''
    confidence: float = 0.7
    evidence: List[str] | None = None
    implied_affordances: List[str] | None = None
    implied_constraints: List[str] | None = None

    def model_dump(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['evidence'] = list(self.evidence or [])
        payload['implied_affordances'] = list(self.implied_affordances or [])
        payload['implied_constraints'] = list(self.implied_constraints or [])
        return payload


@dataclass
class VisualStructuralReasoningResult:
    bindings: List[StructuralOperatorBinding]
    inferred_affordances: List[dict]
    inferred_parts: List[dict]
    inferred_constraints: List[str]
    audit_trace: List[str]


class VisualStructuralOperatorInducer:
    def analyze(self, observation: VisualObservation, candidates: List[VisualAffordanceCandidate]) -> VisualStructuralReasoningResult:
        bindings: List[StructuralOperatorBinding] = []
        affordances: List[dict] = []
        parts: List[dict] = []
        constraints: List[str] = []
        audit: List[str] = []
        object_by_id = {
            str(item.get('id') or item.get('label') or item.get('name') or ''): item
            for item in observation.objects
            if item.get('id') or item.get('label') or item.get('name')
        }
        children_by_parent: Dict[str, List[VisualAffordanceCandidate]] = {}
        for candidate in candidates:
            if candidate.parent:
                children_by_parent.setdefault(candidate.parent, []).append(candidate)

        for candidate in candidates:
            subject = candidate.subject
            features = candidate.features
            children = children_by_parent.get(subject, [])
            if (features.get('is_dominant', 0.0) >= 1.0 and children) or self._is_structural_container_candidate(features):
                evidence = self._signature(candidate)
                bindings.append(StructuralOperatorBinding(
                    operator_name='CONTAINER_BODY_OPERATOR',
                    subject=subject,
                    confidence=0.82,
                    evidence=evidence,
                    implied_affordances=['HAS_INTERIOR', 'STRUCTURAL_CONTAINER_CANDIDATE'],
                    implied_constraints=['container_like_object_detected'],
                ))
                affordances.extend([
                    {'subject': subject, 'value': 'HAS_INTERIOR'},
                    {'subject': subject, 'value': 'STRUCTURAL_CONTAINER_CANDIDATE'},
                ])
                constraints.append('container_like_object_detected')
                if any(self._is_grasp_candidate(child.features) for child in children):
                    bindings.append(StructuralOperatorBinding(
                        operator_name='MANIPULABLE_CONTAINER_OPERATOR',
                        subject=subject,
                        confidence=0.72,
                        evidence=evidence,
                        implied_affordances=['MANIPULABLE_CONTAINER'],
                        implied_constraints=[],
                    ))
                    affordances.append({'subject': subject, 'value': 'MANIPULABLE_CONTAINER'})

            if candidate.parent:
                parts.append({'parent': candidate.parent, 'child': subject})
                if self._is_access_port_candidate(features):
                    bindings.append(StructuralOperatorBinding(
                        operator_name='ACCESS_PORT_OPERATOR',
                        subject=subject,
                        parent=candidate.parent,
                        confidence=0.78,
                        evidence=self._signature(candidate),
                        implied_affordances=['ACCESS_OPENING_CANDIDATE', 'ACCESS_PORT_CANDIDATE'],
                        implied_constraints=['opening_candidate_detected'],
                    ))
                    affordances.extend([
                        {'subject': subject, 'value': 'ACCESS_OPENING_CANDIDATE'},
                        {'subject': subject, 'value': 'ACCESS_PORT_CANDIDATE'},
                    ])
                    constraints.append('opening_candidate_detected')
                if self._is_access_control_candidate(features):
                    bindings.append(StructuralOperatorBinding(
                        operator_name='ACCESS_CONTROL_OPERATOR',
                        subject=subject,
                        parent=candidate.parent,
                        confidence=0.74,
                        evidence=self._signature(candidate),
                        implied_affordances=['ACCESS_CONTROL_PART'],
                        implied_constraints=['closure_part_detected'],
                    ))
                    affordances.append({'subject': subject, 'value': 'ACCESS_CONTROL_PART'})
                    constraints.append('closure_part_detected')
                if self._is_grasp_candidate(features):
                    bindings.append(StructuralOperatorBinding(
                        operator_name='ATTACHED_GRASP_OPERATOR',
                        subject=subject,
                        parent=candidate.parent,
                        confidence=0.76,
                        evidence=self._signature(candidate),
                        implied_affordances=['GRASPABLE_PART', 'HANDLE_CANDIDATE'],
                        implied_constraints=['handle_like_part_detected'],
                    ))
                    affordances.extend([
                        {'subject': subject, 'value': 'GRASPABLE_PART'},
                        {'subject': subject, 'value': 'HANDLE_CANDIDATE'},
                    ])
                    constraints.append('handle_like_part_detected')

        for parent, children in children_by_parent.items():
            has_access = any(self._is_access_port_candidate(child.features) or self._is_access_control_candidate(child.features) for child in children)
            has_grasp = any(self._is_grasp_candidate(child.features) for child in children)
            if has_access and has_grasp:
                bindings.append(StructuralOperatorBinding(
                    operator_name='CONTROLLED_ACCESS_OPERATOR',
                    subject=parent,
                    confidence=0.8,
                    evidence=['HAS_ACCESS_PART', 'HAS_GRASP_PART'],
                    implied_affordances=['ACCESSIBLE_INTERIOR_PATH'],
                    implied_constraints=['controlled_access_structure_detected'],
                ))
                affordances.append({'subject': parent, 'value': 'ACCESSIBLE_INTERIOR_PATH'})
                constraints.append('controlled_access_structure_detected')

        if any(binding.operator_name == 'CONTAINER_BODY_OPERATOR' for binding in bindings):
            audit.append('structural operator inducer derived container/access hypotheses from geometry-topology features')
        return VisualStructuralReasoningResult(
            bindings=self._dedupe_bindings(bindings),
            inferred_affordances=self._dedupe_dicts(affordances),
            inferred_parts=self._dedupe_dicts(parts),
            inferred_constraints=list(dict.fromkeys(constraints)),
            audit_trace=audit,
        )

    @staticmethod
    def _is_access_port_candidate(features: Dict[str, float]) -> bool:
        return (
            (features.get('inside_parent', 0.0) >= 1.0 or features.get('explicit_opening_hint', 0.0) >= 1.0)
            and (
                features.get('boundary_attached', 0.0) >= 1.0
                or features.get('hole_count_norm', 0.0) > 0.0
                or features.get('explicit_opening_hint', 0.0) >= 1.0
            )
            and (
                features.get('near_top_band', 0.0) >= 1.0
                or features.get('horizontal_elongation', 0.0) >= 2.1
                or features.get('relative_area', 0.0) <= 0.18
                or features.get('hole_count_norm', 0.0) > 0.0
                or features.get('explicit_opening_hint', 0.0) >= 1.0
            )
        )

    @staticmethod
    def _is_access_control_candidate(features: Dict[str, float]) -> bool:
        return (
            (features.get('inside_parent', 0.0) >= 1.0 or features.get('explicit_opening_hint', 0.0) >= 1.0)
            and (
                features.get('boundary_attached', 0.0) >= 1.0
                or features.get('segmentation_confidence', 0.0) >= 0.5
            )
            and (
                features.get('near_side_band', 0.0) >= 1.0
                or features.get('vertical_elongation', 0.0) >= 1.9
                or features.get('horizontal_elongation', 0.0) >= 2.8
                or features.get('compactness', 0.0) <= 0.42
                or features.get('explicit_control_hint', 0.0) >= 1.0
            )
        )

    @staticmethod
    def _is_grasp_candidate(features: Dict[str, float]) -> bool:
        return (
            features.get('explicit_grasp_hint', 0.0) >= 1.0
            or features.get('near_side_band', 0.0) >= 1.0
            or max(features.get('horizontal_elongation', 0.0), features.get('vertical_elongation', 0.0)) >= 2.4
            or (features.get('segmentation_confidence', 0.0) >= 0.5 and features.get('compactness', 0.0) <= 0.5)
        )

    @staticmethod
    def _is_structural_container_candidate(features: Dict[str, float]) -> bool:
        return (
            features.get('is_dominant', 0.0) >= 1.0
            and (
                features.get('explicit_parent', 0.0) >= 1.0
                or features.get('bbox_fill_ratio', 0.0) >= 0.42
                or features.get('hull_fill_ratio', 0.0) >= 0.55
                or features.get('hole_count_norm', 0.0) > 0.0
                or features.get('explicit_opening_hint', 0.0) >= 1.0
                or features.get('right_angle_count_norm', 0.0) >= 0.3
                or features.get('parallel_edge_pair_norm', 0.0) >= 0.2
            )
        )

    @staticmethod
    def _signature(candidate: VisualAffordanceCandidate) -> List[str]:
        features = candidate.features
        signature: List[str] = []
        if candidate.parent:
            signature.append('HAS_PARENT')
        if features.get('inside_parent', 0.0) >= 1.0:
            signature.append('INSIDE_PARENT')
        if features.get('boundary_attached', 0.0) >= 1.0:
            signature.append('BOUNDARY_ATTACHED')
        if features.get('near_top_band', 0.0) >= 1.0:
            signature.append('TOP_BAND')
        if features.get('near_side_band', 0.0) >= 1.0:
            signature.append('SIDE_BAND')
        if features.get('horizontal_elongation', 0.0) >= 2.3:
            signature.append('HORIZONTAL_ELONGATION')
        if features.get('vertical_elongation', 0.0) >= 2.3:
            signature.append('VERTICAL_ELONGATION')
        if features.get('is_dominant', 0.0) >= 1.0:
            signature.append('DOMINANT_REGION')
        return signature

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
    def _dedupe_bindings(items: List[StructuralOperatorBinding]) -> List[StructuralOperatorBinding]:
        seen = set()
        output: List[StructuralOperatorBinding] = []
        for item in items:
            key = (item.operator_name, item.subject, item.parent)
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
        return output

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from .affordance_features import VisualAffordanceCandidate
from .operator_learning import VisualOperatorMemory
from .structural_operators import StructuralOperatorBinding
from .types import VisualObservation


@dataclass
class PredictivePriorResult:
    bindings: List[StructuralOperatorBinding]
    inferred_affordances: List[dict[str, Any]]
    inferred_constraints: List[str]
    audit_trace: List[str]

    def model_dump(self) -> dict[str, Any]:
        return {
            'bindings': [item.model_dump() for item in self.bindings],
            'inferred_affordances': list(self.inferred_affordances),
            'inferred_constraints': list(self.inferred_constraints),
            'audit_trace': list(self.audit_trace),
        }


class JepaStructuralPredictor:
    """JEPA-inspired latent structural prediction.

    Predict structural operator priors from context features rather than
    reconstructing pixels or relying on explicit labels. This is not a full
    JEPA training implementation; it is a runtime world-model layer aligned
    with the JEPA principle of context-to-latent prediction.
    """

    def __init__(self, operator_memory: VisualOperatorMemory | None = None) -> None:
        self.operator_memory = operator_memory

    def predict(self, observation: VisualObservation, candidates: List[VisualAffordanceCandidate]) -> PredictivePriorResult:
        if self.operator_memory is None or not candidates:
            return PredictivePriorResult(bindings=[], inferred_affordances=[], inferred_constraints=[], audit_trace=[])
        bindings: list[StructuralOperatorBinding] = []
        affordances: list[dict[str, Any]] = []
        constraints: list[str] = []
        audit: list[str] = []
        children_by_parent: dict[str, list[VisualAffordanceCandidate]] = {}
        candidate_by_subject = {item.subject: item for item in candidates}
        for candidate in candidates:
            if candidate.parent:
                children_by_parent.setdefault(candidate.parent, []).append(candidate)
        for parent, children in children_by_parent.items():
            parent_candidate = candidate_by_subject.get(parent)
            context_vector = self._context_vector(parent_candidate, children)
            signature = self._context_signature(parent_candidate, children)
            matches = self.operator_memory.search(context_vector, signature, limit=3)
            accepted = [item for item in matches if item.score >= 0.4]
            if not accepted:
                continue
            audit.append('JEPA-inspired structural predictor inferred latent operator priors from context')
            for match in accepted:
                binding = self._binding_from_match(parent, match.operator_name, children, match.score)
                if binding is None:
                    continue
                bindings.append(binding)
                for label in binding.implied_affordances or []:
                    affordances.append({'subject': binding.subject, 'value': label})
                for row in binding.implied_constraints or []:
                    constraints.append(row)
        return PredictivePriorResult(
            bindings=self._dedupe_bindings(bindings),
            inferred_affordances=self._dedupe_dicts(affordances),
            inferred_constraints=list(dict.fromkeys(constraints)),
            audit_trace=list(dict.fromkeys(audit)),
        )

    def _context_vector(self, parent: VisualAffordanceCandidate | None, children: List[VisualAffordanceCandidate]) -> Dict[str, float]:
        rows = [child.features for child in children]
        summary: dict[str, float] = {}
        keys = sorted({key for row in rows for key in row})
        for key in keys:
            summary[key] = sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows))
        summary['child_count_norm'] = min(1.0, len(children) / 5.0)
        summary['has_parent_container'] = 1.0 if parent is not None else 0.0
        if parent is not None:
            for key in ('bbox_fill_ratio', 'hull_fill_ratio', 'hole_count_norm', 'right_angle_count_norm', 'parallel_edge_pair_norm'):
                summary[f'parent_{key}'] = float(parent.features.get(key, 0.0))
        return summary

    def _context_signature(self, parent: VisualAffordanceCandidate | None, children: List[VisualAffordanceCandidate]) -> List[str]:
        signature: list[str] = []
        if parent is not None:
            signature.append('HAS_PARENT_CONTAINER')
            if parent.features.get('right_angle_count_norm', 0.0) >= 0.3:
                signature.append('PARENT_RIGHT_ANGLE_STRUCTURE')
            if parent.features.get('parallel_edge_pair_norm', 0.0) >= 0.2:
                signature.append('PARENT_PARALLEL_EDGE_STRUCTURE')
            if parent.features.get('hole_count_norm', 0.0) > 0.0:
                signature.append('PARENT_HOLE_STRUCTURE')
        if any(child.features.get('near_top_band', 0.0) >= 1.0 for child in children):
            signature.append('TOP_ACCESS_PATTERN')
        if any(child.features.get('near_side_band', 0.0) >= 1.0 for child in children):
            signature.append('SIDE_GRASP_PATTERN')
        if any(child.features.get('boundary_attached', 0.0) >= 1.0 for child in children):
            signature.append('BOUNDARY_ATTACHED_CHILD')
        if any(child.features.get('right_angle_count_norm', 0.0) >= 0.3 for child in children):
            signature.append('CHILD_RIGHT_ANGLE_STRUCTURE')
        if any(child.features.get('parallel_edge_pair_norm', 0.0) >= 0.2 for child in children):
            signature.append('CHILD_PARALLEL_EDGE_STRUCTURE')
        return sorted(signature)

    def _binding_from_match(
        self,
        parent: str,
        operator_name: str,
        children: List[VisualAffordanceCandidate],
        score: float,
    ) -> StructuralOperatorBinding | None:
        upper = operator_name.upper()
        if upper == 'CONTAINER_ACCESS_OPERATOR':
            return StructuralOperatorBinding(
                operator_name=operator_name,
                subject=parent,
                confidence=round(score, 4),
                evidence=['JEPA_CONTEXT_PREDICTION'],
                implied_affordances=['ACCESSIBLE_INTERIOR_PATH'],
                implied_constraints=['predictive_access_structure_detected'],
            )
        if upper == 'CONTROLLED_ACCESS_OPERATOR':
            return StructuralOperatorBinding(
                operator_name=operator_name,
                subject=parent,
                confidence=round(score, 4),
                evidence=['JEPA_CONTEXT_PREDICTION'],
                implied_affordances=['CONTROLLED_ACCESS_PATH'],
                implied_constraints=['predictive_control_structure_detected'],
            )
        if upper == 'CARRIABLE_CONTAINER_OPERATOR':
            return StructuralOperatorBinding(
                operator_name=operator_name,
                subject=parent,
                confidence=round(score, 4),
                evidence=['JEPA_CONTEXT_PREDICTION'],
                implied_affordances=['PORTABLE_CONTAINER'],
                implied_constraints=[],
            )
        if upper == 'ATTACHED_GRASP_OPERATOR':
            child = self._best_child(children, mode='grasp')
            if child is None:
                return None
            return StructuralOperatorBinding(
                operator_name=operator_name,
                subject=child.subject,
                parent=parent,
                confidence=round(score, 4),
                evidence=['JEPA_CONTEXT_PREDICTION'],
                implied_affordances=['GRASPABLE_PART'],
                implied_constraints=['predictive_grasp_part_detected'],
            )
        if upper in {'OPENING_CONTROL_OPERATOR', 'ACCESS_PORT_OPERATOR', 'ACCESS_CONTROL_OPERATOR'}:
            child = self._best_child(children, mode='opening')
            if child is None:
                return None
            implied = ['ACCESS_OPENING_CANDIDATE'] if upper != 'ACCESS_CONTROL_OPERATOR' else ['ACCESS_CONTROL_PART']
            return StructuralOperatorBinding(
                operator_name='ACCESS_PORT_OPERATOR' if upper == 'OPENING_CONTROL_OPERATOR' else operator_name,
                subject=child.subject,
                parent=parent,
                confidence=round(score, 4),
                evidence=['JEPA_CONTEXT_PREDICTION'],
                implied_affordances=implied,
                implied_constraints=['predictive_opening_structure_detected'],
            )
        return None

    @staticmethod
    def _best_child(children: List[VisualAffordanceCandidate], mode: str) -> VisualAffordanceCandidate | None:
        if not children:
            return None
        if mode == 'grasp':
            return max(children, key=lambda item: item.features.get('near_side_band', 0.0) + max(item.features.get('horizontal_elongation', 0.0), item.features.get('vertical_elongation', 0.0)))
        return max(children, key=lambda item: item.features.get('near_top_band', 0.0) + item.features.get('boundary_attached', 0.0) + item.features.get('hole_count_norm', 0.0))

    @staticmethod
    def _dedupe_dicts(items: List[dict[str, Any]]) -> List[dict[str, Any]]:
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
        output: list[StructuralOperatorBinding] = []
        for item in items:
            key = (item.operator_name, item.subject, item.parent)
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
        return output

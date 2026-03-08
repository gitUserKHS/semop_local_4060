from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .structures import OperatorCandidate


@dataclass
class RegistryNode:
    level: str
    name: str
    support: int
    purity: float
    confidence: float
    parent_names: List[str] = field(default_factory=list)
    child_names: List[str] = field(default_factory=list)
    member_operators: List[str] = field(default_factory=list)
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    example_queries: List[str] = field(default_factory=list)
    composition_signatures: List[str] = field(default_factory=list)


class TypedOperatorRegistry:
    def __init__(
        self,
        micro_nodes: Dict[str, RegistryNode] | None = None,
        family_nodes: Dict[str, RegistryNode] | None = None,
        abstract_nodes: Dict[str, RegistryNode] | None = None,
        composition_patterns: List[dict] | None = None,
    ):
        self.micro_nodes = micro_nodes or {}
        self.family_nodes = family_nodes or {}
        self.abstract_nodes = abstract_nodes or {}
        self.composition_patterns = composition_patterns or []
        self._family_to_abstract: Dict[str, List[str]] = {}
        for abstract in self.abstract_nodes.values():
            for child in abstract.child_names:
                if child.startswith('L2_'):
                    family = child.removeprefix('L2_')
                    self._family_to_abstract.setdefault(family, []).append(abstract.name)

    @classmethod
    def from_summary(cls, summary: dict | None) -> TypedOperatorRegistry:
        if not summary:
            return cls()
        return cls(
            micro_nodes={item['name']: RegistryNode(**item) for item in summary.get('micro_nodes', [])},
            family_nodes={item['name']: RegistryNode(**item) for item in summary.get('family_nodes', [])},
            abstract_nodes={item['name']: RegistryNode(**item) for item in summary.get('abstract_nodes', [])},
            composition_patterns=list(summary.get('composition_patterns', [])),
        )

    def is_empty(self) -> bool:
        return not (self.micro_nodes or self.family_nodes or self.abstract_nodes)

    def family_node(self, family: str) -> RegistryNode | None:
        return self.family_nodes.get(f'L2_{family}')

    def abstract_parents_for_family(self, family: str) -> List[str]:
        return list(dict.fromkeys(self._family_to_abstract.get(family, [])))

    def annotate_candidate(self, candidate: OperatorCandidate) -> OperatorCandidate:
        family_node = self.family_node(candidate.family)
        if family_node is not None:
            if candidate.output_type == 'meaning_state' and family_node.output_types:
                candidate.output_type = family_node.output_types[0]
            for input_type in family_node.input_types:
                if input_type not in candidate.input_types:
                    candidate.input_types.append(input_type)
            registry_tag = f'registry:family:{candidate.family}'
            if registry_tag not in candidate.provenance:
                candidate.provenance.append(registry_tag)
        for parent in self.abstract_parents_for_family(candidate.family):
            if parent not in candidate.abstract_parents:
                candidate.abstract_parents.append(parent)
            registry_tag = f'registry:abstract:{parent}'
            if registry_tag not in candidate.provenance:
                candidate.provenance.append(registry_tag)
        candidate.input_types = list(dict.fromkeys(candidate.input_types))[:6]
        candidate.abstract_parents = list(dict.fromkeys(candidate.abstract_parents))[:4]
        return candidate

    def related_families(self, candidate: OperatorCandidate) -> List[str]:
        related: List[str] = []
        candidate_inputs = set(candidate.input_types)
        for parent_name in self.abstract_parents_for_family(candidate.family):
            parent = self.abstract_nodes.get(parent_name)
            if parent is None:
                continue
            for child_name in parent.child_names:
                if not child_name.startswith('L2_'):
                    continue
                family = child_name.removeprefix('L2_')
                if family == candidate.family:
                    continue
                node = self.family_nodes.get(child_name)
                if node is None:
                    continue
                outputs = set(node.output_types)
                inputs = set(node.input_types)
                if candidate.output_type in outputs or candidate_inputs & inputs or candidate_inputs & outputs:
                    related.append(family)
        return list(dict.fromkeys(related))[:4]

    def abstract_support_map(self, family_support: Dict[str, int]) -> Dict[str, int]:
        support: Dict[str, int] = {}
        for family, count in family_support.items():
            for parent in self.abstract_parents_for_family(family):
                support[parent] = support.get(parent, 0) + count
        return support

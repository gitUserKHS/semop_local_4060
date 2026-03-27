from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class VLSOEntity:
    id: str
    label: str
    modality: str
    entity_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class VLSORelation:
    source: str
    relation: str
    target: str
    modality: str
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class VLSOOperator:
    name: str
    axis: str
    description: str
    source_modality: str = "shared"
    confidence: float = 0.6

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class VLSOEvent:
    id: str
    label: str
    event_type: str
    modality: str = "shared"
    frame_index: int | None = None
    confidence: float = 1.0
    participants: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class VisualObservation:
    objects: List[dict] = field(default_factory=list)
    relations: List[dict] = field(default_factory=list)
    affordances: List[dict] = field(default_factory=list)
    states: List[dict] = field(default_factory=list)
    geometry: List[dict] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SharedWorldModel:
    query: str
    entities: List[VLSOEntity] = field(default_factory=list)
    relations: List[VLSORelation] = field(default_factory=list)
    operators: List[VLSOOperator] = field(default_factory=list)
    events: List[VLSOEvent] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    inferred_steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    audit_trace: List[str] = field(default_factory=list)

    def add_entity(self, entity: VLSOEntity) -> None:
        for existing in self.entities:
            if existing.id == entity.id:
                existing.attributes.update(entity.attributes)
                if existing.entity_type == "unknown" and entity.entity_type != "unknown":
                    existing.entity_type = entity.entity_type
                if (existing.label == existing.id or existing.label.lower().startswith(("shape_", "polygon_"))) and entity.label:
                    existing.label = entity.label
                return
        self.entities.append(entity)

    def add_relation(self, relation: VLSORelation) -> None:
        for existing in self.relations:
            if (existing.source, existing.relation, existing.target) == (relation.source, relation.relation, relation.target):
                existing.confidence = max(existing.confidence, relation.confidence)
                existing.attributes.update(relation.attributes)
                return
        self.relations.append(relation)

    def add_operator(self, operator: VLSOOperator) -> None:
        key = (operator.name, operator.axis, operator.source_modality)
        for existing in self.operators:
            if (existing.name, existing.axis, existing.source_modality) == key:
                existing.confidence = max(existing.confidence, operator.confidence)
                return
        self.operators.append(operator)

    def add_event(self, event: VLSOEvent) -> None:
        for existing in self.events:
            if existing.id == event.id:
                existing.confidence = max(existing.confidence, event.confidence)
                existing.participants.update(event.participants)
                existing.attributes.update(event.attributes)
                if existing.frame_index is None and event.frame_index is not None:
                    existing.frame_index = event.frame_index
                return
        self.events.append(event)

    def relation_tuples(self) -> set[tuple[str, str, str]]:
        return {(item.source, item.relation, item.target) for item in self.relations}

    def model_dump(self) -> dict:
        return {
            "query": self.query,
            "entities": [item.model_dump() for item in self.entities],
            "relations": [item.model_dump() for item in self.relations],
            "operators": [item.model_dump() for item in self.operators],
            "events": [item.model_dump() for item in self.events],
            "goals": list(self.goals),
            "constraints": list(self.constraints),
            "metadata": dict(self.metadata),
            "inferred_steps": list(self.inferred_steps),
            "warnings": list(self.warnings),
            "audit_trace": list(self.audit_trace),
        }

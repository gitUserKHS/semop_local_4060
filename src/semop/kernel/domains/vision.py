from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from ..catalog import normalize_predicate_name, register_transitive_relation
from ..grounding import (
    GroundingAuthority,
    GroundingDisposition,
    GroundingTrace,
    grounding_payload_digest,
    make_grounding_record,
)
from ..model import (
    AssertionStatus,
    EvidenceStatus,
    Fact,
    FactStatus,
    Goal,
    SolveResult,
    WorldState,
)
from ..registry import KernelRegistry
from .base import DomainInstance


@dataclass(frozen=True)
class VisionRelationGoal:
    predicate: str
    source: str
    target: str
    label: str = ""


@dataclass(frozen=True)
class VisionProblem:
    world: Any
    goals: tuple[VisionRelationGoal, ...]

    def __post_init__(self) -> None:
        if not self.goals:
            raise ValueError("vision problem requires at least one explicit typed goal")


class VisionWorldAdapter:
    """Convert a symbolic scene into facts without trusting model confidence."""

    _SUPPORTED_RELATIONS = {
        "ABOVE",
        "CONNECTED_TO",
        "INSIDE",
        "LEFT_OF",
        "OVERLAPS",
        "SAME_OBJECT",
        "SUPPORTS",
        "TOUCHING",
    }
    _TRANSITIVE_RELATIONS = {"ABOVE", "INSIDE", "LEFT_OF"}
    _SYMMETRIC_RELATIONS = {
        "CONNECTED_TO",
        "OVERLAPS",
        "SAME_OBJECT",
        "TOUCHING",
    }
    _INVERSES = {
        "BELOW": ("ABOVE", True),
        "CONTAINS": ("INSIDE", True),
        "RIGHT_OF": ("LEFT_OF", True),
    }
    _VERIFICATION_KEYS = (
        "verified",
        "geometry_verified",
        "symbolically_verified",
        "adapter_verified",
        "external_verified",
    )

    def __init__(self, *, minimum_verified_confidence: float = 0.5) -> None:
        if not 0.0 <= minimum_verified_confidence <= 1.0:
            raise ValueError("minimum verified confidence must be between 0 and 1")
        self.minimum_verified_confidence = minimum_verified_confidence

    def adapt(self, value: VisionProblem) -> DomainInstance:
        if not isinstance(value, VisionProblem):
            raise TypeError("vision payload must be a VisionProblem")
        return self.adapt_world(value.world, value.goals)

    def adapt_world(
        self,
        world: Any,
        goals: tuple[VisionRelationGoal, ...] = (),
    ) -> DomainInstance:
        """Adapt a scene independently from optional downstream goal extensions."""

        if not hasattr(world, "entities") or not hasattr(world, "relations"):
            raise TypeError("vision world must expose entities and relations")

        registry = KernelRegistry()
        entity = registry.types.register("Entity")
        visual_entity = registry.types.register("VisualEntity", entity)
        registry.register_predicate("VISUAL_ENTITY", (visual_entity,))

        explicit_ids = {
            str(item.id).strip()
            for item in world.entities
            if str(getattr(item, "id", "")).strip()
        }
        relation_ids = {
            endpoint
            for relation in world.relations
            for endpoint in (
                str(getattr(relation, "source", "")).strip(),
                str(getattr(relation, "target", "")).strip(),
            )
            if endpoint
        }
        known_ids = explicit_ids | relation_ids
        for goal in goals:
            missing = {
                endpoint
                for endpoint in (goal.source.strip(), goal.target.strip())
                if endpoint not in known_ids
            }
            if missing:
                raise ValueError(
                    "vision goal references unknown entities: "
                    + ", ".join(sorted(missing))
                )

        symbols = {
            name: registry.symbol(name, visual_entity) for name in sorted(known_ids)
        }
        relation_records = []
        predicate_names: set[str] = set()
        for relation in world.relations:
            predicate, source, target = self._canonical_relation(
                str(getattr(relation, "relation", "")),
                str(getattr(relation, "source", "")).strip(),
                str(getattr(relation, "target", "")).strip(),
            )
            if not source or not target:
                continue
            predicate_names.add(predicate)
            relation_records.append((predicate, source, target, relation))

        goal_records = []
        for goal in goals:
            predicate, source, target = self._canonical_relation(
                goal.predicate, goal.source.strip(), goal.target.strip()
            )
            predicate_names.add(predicate)
            goal_records.append((predicate, source, target, goal.label))

        unverified_relations: set[str] = set()
        for predicate_name in sorted(predicate_names):
            verified = predicate_name in self._SUPPORTED_RELATIONS
            symmetry = (
                ((0, 1),)
                if predicate_name in self._SYMMETRIC_RELATIONS
                else ()
            )
            registry.register_predicate(
                predicate_name,
                (visual_entity, visual_entity),
                symmetry_groups=symmetry,
                verified=verified,
                metadata=None if verified else {"status": "unverified"},
            )
            if not verified:
                unverified_relations.add(predicate_name)

        for predicate_name in sorted(
            predicate_names & self._TRANSITIVE_RELATIONS
        ):
            register_transitive_relation(
                registry,
                predicate_name,
                visual_entity,
                tags=("vision", "spatial"),
                description_ko=(
                    "검증된 공간 관계를 이어 {x}와 {z} 사이의 관계를 도출한다."
                ),
            )

        facts: list[Fact] = []
        grounding_records = []
        input_digest = _vision_input_digest(world)
        entity_by_id = {
            str(item.id).strip(): item
            for item in world.entities
            if str(getattr(item, "id", "")).strip()
        }
        for entity_id, symbol in symbols.items():
            source_entity = entity_by_id.get(entity_id)
            attributes = dict(getattr(source_entity, "attributes", {}) or {})
            status = (
                FactStatus.OBSERVED
                if self._independently_verified(attributes)
                else FactStatus.PROPOSED
            )
            atom = registry.atom("VISUAL_ENTITY", symbol)
            record = make_grounding_record(
                domain="vision",
                statement=f"entity:{entity_id}:{atom}",
                atom=atom,
                producer_id="vision_world_adapter",
                source="vision_entity",
                disposition=(
                    GroundingDisposition.OBSERVED
                    if status is FactStatus.OBSERVED
                    else GroundingDisposition.PROPOSED
                ),
                authority=self._grounding_authority(attributes, status),
                assertion_status=AssertionStatus.IMPORTED,
                evidence_status=self._evidence_status(attributes),
                rationale=(
                    "vision input carried an independent verification marker"
                    if status is FactStatus.OBSERVED
                    else "unverified visual entity was retained as a proposal"
                ),
                input_digest=input_digest,
                evidence=(f"input:{input_digest}",),
            )
            grounding_records.append(record)
            if record.fact is not None:
                facts.append(record.fact)

        verified_count = 0
        proposed_count = 0
        for relation_index, (predicate_name, source, target, relation) in enumerate(
            relation_records
        ):
            predicate = registry.predicates[predicate_name]
            attributes = dict(getattr(relation, "attributes", {}) or {})
            confidence = _clamp_confidence(getattr(relation, "confidence", 0.0))
            if attributes.get("contradicted") is True:
                status = FactStatus.CONTRADICTED
            elif (
                predicate.verified
                and self._independently_verified(attributes)
                and confidence >= self.minimum_verified_confidence
            ):
                status = FactStatus.OBSERVED
                verified_count += 1
            else:
                status = FactStatus.PROPOSED
                proposed_count += 1
            atom = registry.atom(predicate, symbols[source], symbols[target])
            record = make_grounding_record(
                domain="vision",
                statement=f"relation:{relation_index}:{atom}",
                atom=atom,
                producer_id="vision_world_adapter",
                source="vision_relation",
                disposition={
                    FactStatus.OBSERVED: GroundingDisposition.OBSERVED,
                    FactStatus.PROPOSED: GroundingDisposition.PROPOSED,
                    FactStatus.CONTRADICTED: GroundingDisposition.CONTRADICTED,
                }[status],
                authority=self._grounding_authority(attributes, status),
                assertion_status=AssertionStatus.IMPORTED,
                evidence_status=self._evidence_status(attributes),
                rationale=(
                    "visual relation passed its declared independent verifier"
                    if status is FactStatus.OBSERVED
                    else (
                        "visual relation was explicitly marked contradicted"
                        if status is FactStatus.CONTRADICTED
                        else "visual relation lacked independent verification"
                    )
                ),
                input_digest=input_digest,
                evidence=(f"input:{input_digest}",),
                confidence=confidence,
            )
            grounding_records.append(record)
            if record.fact is not None:
                facts.append(record.fact)

        goals = tuple(
            Goal(
                registry.atom(predicate, symbols[source], symbols[target]),
                label=label,
            )
            for predicate, source, target, label in goal_records
        )
        grounding_trace = GroundingTrace(tuple(grounding_records))
        return DomainInstance(
            registry=registry,
            state=WorldState(tuple(facts)),
            goals=goals,
            domain="vision",
            metadata={
                "query": str(getattr(world, "query", "")),
                "verified_relation_count": verified_count,
                "proposed_relation_count": proposed_count,
                "implicit_entities": tuple(sorted(relation_ids - explicit_ids)),
                "unverified_relations": tuple(sorted(unverified_relations)),
                "reviewed_examples": 0,
                "grounding": grounding_trace.to_dict(include_records=False),
            },
            grounding_trace=grounding_trace,
        )

    @classmethod
    def _canonical_relation(
        cls, predicate: str, source: str, target: str
    ) -> tuple[str, str, str]:
        normalized = normalize_predicate_name(predicate)
        inverse = cls._INVERSES.get(normalized)
        if inverse is not None:
            canonical, swap = inverse
            return (canonical, target, source) if swap else (canonical, source, target)
        return normalized, source, target

    @classmethod
    def _independently_verified(cls, attributes: dict[str, Any]) -> bool:
        return any(attributes.get(key) is True for key in cls._VERIFICATION_KEYS)

    @classmethod
    def _evidence_status(cls, attributes: dict[str, Any]) -> EvidenceStatus:
        if attributes.get("external_verified") is True:
            return EvidenceStatus.EXTERNAL_VERIFIED
        if attributes.get("adapter_verified") is True:
            return EvidenceStatus.ADAPTER_VERIFIED
        return EvidenceStatus.UNVERIFIED

    @classmethod
    def _grounding_authority(
        cls,
        attributes: dict[str, Any],
        status: FactStatus,
    ) -> GroundingAuthority:
        if status is FactStatus.PROPOSED:
            if attributes.get("perception_proposal") is True:
                return GroundingAuthority.HEURISTIC_PROPOSAL
            return GroundingAuthority.IMPORTED_PROPOSAL
        evidence_status = cls._evidence_status(attributes)
        if evidence_status is EvidenceStatus.EXTERNAL_VERIFIED:
            return GroundingAuthority.EXTERNAL_VERIFIER
        if evidence_status is EvidenceStatus.ADAPTER_VERIFIED:
            return GroundingAuthority.DETERMINISTIC_ADAPTER
        return GroundingAuthority.EXPLICIT_INPUT

    @staticmethod
    def project(value: VisionProblem, result: SolveResult) -> None:
        """Project only replay-verified traces into the legacy scene audit fields."""

        world = value.world
        summary = (
            "typed_kernel: "
            f"success={str(result.success).lower()} "
            f"verified={str(result.verified).lower()} "
            f"steps={len(result.proof)} rounds={result.inference_rounds} "
            f"expansions={result.expansions}"
        )
        if summary not in world.audit_trace:
            world.audit_trace.append(summary)
        if not result.verified:
            return
        for step in result.proof:
            projected = (
                f"typed:{step.index}:{step.action.operator.name}:"
                + ",".join(str(effect) for effect in step.effects)
            )
            if projected not in world.inferred_steps:
                world.inferred_steps.append(projected)
        world.metadata["typed_kernel"] = {
            "success": result.success,
            "verified": result.verified,
            "expansions": result.expansions,
            "proof_steps": len(result.proof),
            "inference_rounds": result.inference_rounds,
        }


def _clamp_confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _vision_input_digest(world: Any) -> str:
    supplied = str(getattr(world, "input_digest", "")).strip().lower()
    if len(supplied) == 64 and all(
        character in "0123456789abcdef" for character in supplied
    ):
        return supplied
    payload = {
        "query": str(getattr(world, "query", "")),
        "entities": [
            {
                "id": str(getattr(item, "id", "")),
                "attributes": dict(getattr(item, "attributes", {}) or {}),
            }
            for item in getattr(world, "entities", ())
        ],
        "relations": [
            {
                "source": str(getattr(item, "source", "")),
                "relation": str(getattr(item, "relation", "")),
                "target": str(getattr(item, "target", "")),
                "confidence": _clamp_confidence(
                    getattr(item, "confidence", 0.0)
                ),
                "attributes": dict(getattr(item, "attributes", {}) or {}),
            }
            for item in getattr(world, "relations", ())
        ],
    }
    return grounding_payload_digest(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )

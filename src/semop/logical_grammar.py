from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Dict, List, Sequence

from .structures import StructuredMeaningGraph


@dataclass
class LogicalPattern:
    connector: str
    family: str
    support: int
    relation_hints: List[str]
    concept_slots: List[str]
    example_frames: List[str]
    example_queries: List[str]
    induced_operator_name: str
    grammar_template: str
    confidence: float


@dataclass
class GrammarInductionResult:
    patterns: List[LogicalPattern]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class _PatternAccumulator:
    connector: str
    family: str
    relation_hints: List[str] = field(default_factory=list)
    concept_slots: List[str] = field(default_factory=list)
    example_frames: List[str] = field(default_factory=list)
    example_queries: List[str] = field(default_factory=list)


@dataclass
class LogicalPatternMatch:
    pattern: LogicalPattern
    relation: str
    source_id: str
    target_id: str
    connector_start: int
    connector_end: int


class LogicalPatternMatcher:
    FAMILY_TO_RELATION = {
        "PRECONDITION_FRAME": "REQUIRES",
        "CONSTRAINT_FRAME": "BLOCKED_BY",
        "CONDITIONAL_FRAME": "CONDITION_ON",
        "TEMPORAL_ORDER_FRAME": "BEFORE",
        "POSSESSION_FRAME": "HAS",
        "STATE_ASSERTION_FRAME": "STATE",
        "CAPABILITY_FRAME": "AFFORDS",
        "CONTAINMENT_FRAME": "CONTAINS",
        "CAUSAL_FRAME": "CAUSES",
        "ALTERNATIVE_FRAME": "ALTERNATIVE",
    }

    def match(self, query: str, graph: StructuredMeaningGraph, patterns: Sequence[LogicalPattern]) -> List[LogicalPatternMatch]:
        lowered = query.lower()
        mentions = self._mentions(lowered, graph)
        matches: List[LogicalPatternMatch] = []
        for pattern in patterns:
            connector = pattern.connector.lower().strip()
            if not connector:
                continue
            start = 0
            while True:
                index = lowered.find(connector, start)
                if index < 0:
                    break
                source_id, target_id = self._resolve_pair(index, index + len(connector), mentions, pattern.family)
                relation = self._relation_for(pattern)
                if source_id and target_id and source_id != target_id and relation:
                    matches.append(
                        LogicalPatternMatch(
                            pattern=pattern,
                            relation=relation,
                            source_id=source_id,
                            target_id=target_id,
                            connector_start=index,
                            connector_end=index + len(connector),
                        )
                    )
                start = index + len(connector)
        deduped: Dict[tuple[str, str, str, str], LogicalPatternMatch] = {}
        for item in matches:
            key = (item.pattern.family, item.relation, item.source_id, item.target_id)
            existing = deduped.get(key)
            if existing is None or item.pattern.confidence > existing.pattern.confidence:
                deduped[key] = item
        return list(deduped.values())

    def _resolve_pair(self, start: int, end: int, mentions: Sequence[tuple[int, int, str]], family: str) -> tuple[str | None, str | None]:
        left_mentions = [item for item in mentions if item[1] <= start]
        right_mentions = [item for item in mentions if item[0] >= end]
        all_ids = []
        for _s, _e, node_id in mentions:
            if node_id not in all_ids:
                all_ids.append(node_id)

        left_id = left_mentions[-1][2] if left_mentions else None
        right_id = right_mentions[0][2] if right_mentions else None
        if family == "CONDITIONAL_FRAME":
            condition_id = right_mentions[0][2] if right_mentions else (all_ids[0] if all_ids else None)
            action_id = right_mentions[1][2] if len(right_mentions) > 1 else (all_ids[1] if len(all_ids) > 1 else left_id)
            return action_id, condition_id
        if family == "TEMPORAL_ORDER_FRAME":
            if left_id is None and len(all_ids) > 0:
                left_id = all_ids[0]
            if right_id is None and len(all_ids) > 1:
                right_id = all_ids[1]
            return left_id, right_id
        if left_id is None and len(all_ids) > 0:
            left_id = all_ids[0]
        if right_id is None and len(all_ids) > 1:
            fallback = all_ids[1] if all_ids[1] != left_id else (all_ids[2] if len(all_ids) > 2 else None)
            right_id = fallback
        return left_id, right_id

    def _relation_for(self, pattern: LogicalPattern) -> str:
        if pattern.family == "TEMPORAL_ORDER_FRAME" and pattern.connector.lower().strip() in {"after", "??", "??"}:
            return "AFTER"
        for hint in pattern.relation_hints:
            if hint in {"REQUIRES", "BLOCKED_BY", "ALTERNATIVE", "CONTAINS", "PART_OF", "AFFORDS", "CAUSES", "HAS", "STATE", "BEFORE", "AFTER"}:
                return hint
        return self.FAMILY_TO_RELATION.get(pattern.family, "")

    @staticmethod
    def _mentions(lowered_query: str, graph: StructuredMeaningGraph) -> List[tuple[int, int, str]]:
        mentions: List[tuple[int, int, str]] = []
        for node in graph.nodes:
            aliases = {node.id.lower().replace('_', ' '), node.label.lower()}
            for alias in aliases:
                alias = alias.strip()
                if not alias:
                    continue
                start = 0
                while True:
                    index = lowered_query.find(alias, start)
                    if index < 0:
                        break
                    mentions.append((index, index + len(alias), node.id))
                    start = index + len(alias)
        mentions.sort(key=lambda item: (item[0], item[1], item[2]))
        return mentions


class LogicalGrammarInducer:
    CONNECTOR_RULES = [
        (r"\brequires?\b|\bneed(?:s)?\b|\bmust\b|필요|해야", "PRECONDITION_FRAME", ["REQUIRES"]),
        (r"\bblocked by\b|막히|불가|못하", "CONSTRAINT_FRAME", ["BLOCKED_BY"]),
        (r"\bif\b|\bwhen\b|라면|경우", "CONDITIONAL_FRAME", ["CONDITION", "BRANCH"]),
        (r"\bbefore\b|먼저|이전", "TEMPORAL_ORDER_FRAME", ["BEFORE"]),
        (r"\bafter\b|이후|다음", "TEMPORAL_ORDER_FRAME", ["AFTER"]),
        (r"\bhas\b|\bhave\b|가지|있고", "POSSESSION_FRAME", ["HAS", "CONTAINS"]),
        (r"\bis\b|\bare\b|이다|이다\.|인\b", "STATE_ASSERTION_FRAME", ["IS_A", "STATE"]),
        (r"\bcan\b|가능|할 수", "CAPABILITY_FRAME", ["AFFORDS", "CAN"]),
        (r"\bcontains?\b|\binside\b|into\b|안에|넣", "CONTAINMENT_FRAME", ["CONTAINS", "PART_OF"]),
        (r"\bbecause\b|\bso\b|때문|그래서", "CAUSAL_FRAME", ["CAUSES", "EXPLAINS"]),
        (r"\bor\b|대안|다른", "ALTERNATIVE_FRAME", ["ALTERNATIVE"]),
    ]

    def induce(self, graphs: Sequence[StructuredMeaningGraph]) -> GrammarInductionResult:
        grouped: Dict[tuple[str, str], _PatternAccumulator] = {}
        for graph in graphs:
            query = graph.query
            lowered = query.lower()
            concept_slots = self._concept_slots(graph, query)
            relation_hints = list(dict.fromkeys(edge.relation for edge in graph.edges))

            for pattern, family, default_hints in self.CONNECTOR_RULES:
                for match in re.finditer(pattern, lowered):
                    connector = match.group(0)
                    frame = self._frame(query, match.start(), match.end(), concept_slots)
                    key = (family, connector)
                    bucket = grouped.setdefault(key, _PatternAccumulator(connector=connector, family=family))
                    for hint in relation_hints + default_hints:
                        if hint not in bucket.relation_hints:
                            bucket.relation_hints.append(hint)
                    for slot in concept_slots:
                        if slot not in bucket.concept_slots:
                            bucket.concept_slots.append(slot)
                    if frame not in bucket.example_frames:
                        bucket.example_frames.append(frame)
                    if query not in bucket.example_queries:
                        bucket.example_queries.append(query)

        patterns: List[LogicalPattern] = []
        for (_family, _connector), bucket in grouped.items():
            support = len(bucket.example_queries)
            if support <= 0:
                continue
            concept_slots = bucket.concept_slots[:6] or ["concept_a", "concept_b"]
            relation_hints = bucket.relation_hints[:4] or ["STATE"]
            operator_name = self._operator_name(bucket.family, bucket.connector)
            grammar_template = self._grammar_template(bucket.family, bucket.connector, concept_slots)
            confidence = min(0.96, 0.42 + 0.08 * min(5, support) + 0.03 * min(4, len(relation_hints)))
            patterns.append(
                LogicalPattern(
                    connector=bucket.connector,
                    family=bucket.family,
                    support=support,
                    relation_hints=relation_hints,
                    concept_slots=concept_slots,
                    example_frames=bucket.example_frames[:5],
                    example_queries=bucket.example_queries[:5],
                    induced_operator_name=operator_name,
                    grammar_template=grammar_template,
                    confidence=round(confidence, 2),
                )
            )

        patterns.sort(key=lambda item: (-item.support, -item.confidence, item.family, item.connector))
        return GrammarInductionResult(patterns=patterns)

    @staticmethod
    def _concept_slots(graph: StructuredMeaningGraph, query: str) -> List[str]:
        slots: List[str] = []
        for node in graph.nodes:
            if node.id not in slots:
                slots.append(node.id)
            if node.kind not in slots and node.kind not in {"concept", "task", "user"}:
                slots.append(node.kind)
        raw_tokens = re.findall(r"[A-Za-z_]+", query)
        for token in raw_tokens:
            token = token.lower()
            if len(token) < 3:
                continue
            if token not in slots and token not in {"what", "when", "where", "which", "that", "with", "from"}:
                slots.append(token)
        return slots[:8]

    @staticmethod
    def _frame(query: str, start: int, end: int, concept_slots: Sequence[str]) -> str:
        left = query[max(0, start - 24):start].strip()
        connector = query[start:end].strip()
        right = query[end:min(len(query), end + 24)].strip()
        slot_left = concept_slots[0] if concept_slots else "concept_a"
        slot_right = concept_slots[1] if len(concept_slots) > 1 else "concept_b"
        if left:
            left = left.split()[-1]
        else:
            left = slot_left
        if right:
            right = right.split()[0]
        else:
            right = slot_right
        return f"{left} {connector} {right}"

    @staticmethod
    def _operator_name(family: str, connector: str) -> str:
        token = re.sub(r"[^A-Za-z가-힣0-9]+", "_", connector.strip()).strip("_") or "connector"
        return f"GRAMMAR_{family}_{token}"[:72]

    @staticmethod
    def _grammar_template(family: str, connector: str, concept_slots: Sequence[str]) -> str:
        left = concept_slots[0] if concept_slots else "concept_a"
        right = concept_slots[1] if len(concept_slots) > 1 else "concept_b"
        return f"{left} --[{connector}:{family}]--> {right}"

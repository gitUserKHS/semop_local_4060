from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .structures import StructuredMeaningGraph


@dataclass
class OperatorProposalPattern:
    basis_signature: list[str]
    basis_operators: list[str]
    source_operator_names: list[str]
    source_domains: list[str]
    support: int
    hidden_goals: list[str] = field(default_factory=list)
    evidence_patterns: list[str] = field(default_factory=list)
    visual_signatures: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelProposedOperator:
    proposed_name: str
    normalized_name: str
    basis_signature: list[str]
    basis_operators: list[str]
    source_operator_names: list[str]
    source_domains: list[str]
    support: int
    confidence: float
    proposal_source: str = 'pattern_proposer'
    rationale: str = ''
    evidence_patterns: list[str] = field(default_factory=list)
    visual_signatures: list[str] = field(default_factory=list)
    hidden_goals: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class OperatorProposalSummarizer:
    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id

    def summarize(self, pattern: OperatorProposalPattern) -> tuple[str, str, str]:
        name = self._fallback_name(pattern)
        rationale = self._fallback_rationale(pattern)
        proposal_source = 'heuristic_pattern_proposer'
        if self.model_id:
            try:
                llm_name, llm_rationale = self._llm_summarize(pattern)
            except Exception:
                return name, rationale, proposal_source
            if llm_name:
                return llm_name, llm_rationale or rationale, 'llm_operator_summarizer'
        return name, rationale, proposal_source

    def _llm_summarize(self, pattern: OperatorProposalPattern) -> tuple[str, str]:
        from .vlso.qa import LocalTextGenerator
        from .llm_client import LocalLLMConfig

        generator = LocalTextGenerator(LocalLLMConfig(model_id=self.model_id, max_new_tokens=120, temperature=0.1))
        system = (
            'You propose short reusable operator names from repeated structural/logical basis patterns. '
            'Return a JSON object with keys name and rationale. Keep the name uppercase snake case and under 5 words.'
        )
        user = (
            'basis_signature=' + ','.join(pattern.basis_signature) + '\n'
            'basis_operators=' + ','.join(pattern.basis_operators) + '\n'
            'domains=' + ','.join(pattern.source_domains) + '\n'
            'hidden_goals=' + ','.join(pattern.hidden_goals[:4]) + '\n'
            'visual_signatures=' + ','.join(pattern.visual_signatures[:6]) + '\n'
            'evidence=' + ' | '.join(pattern.evidence_patterns[:6])
        )
        raw = generator.generate(system, user)
        import json
        import re
        first = raw.find('{')
        last = raw.rfind('}')
        if first == -1 or last == -1 or first >= last:
            raise RuntimeError('llm summarizer returned non-json')
        payload = json.loads(raw[first:last + 1])
        return str(payload.get('name', '')).strip(), str(payload.get('rationale', '')).strip()

    @staticmethod
    def _fallback_name(pattern: OperatorProposalPattern) -> str:
        signature = set(pattern.basis_signature)
        goals = set(pattern.hidden_goals)
        if {'ACCESS_PORT_OPERATOR', 'ACCESS_CONTROL_OPERATOR'} <= signature:
            return 'CONTROLLED_ACCESS_SCHEMA'
        if {'CONTAINER_BODY_OPERATOR', 'ATTACHED_GRASP_OPERATOR'} <= signature:
            return 'MANIPULABLE_CONTAINER_SCHEMA'
        if 'GOAL_PRESERVATION_OPERATOR' in signature and any('retrieve_item' in item for item in goals):
            return 'RETRIEVAL_GOAL_SCHEMA'
        if 'GOAL_PRESERVATION_OPERATOR' in signature and any('pour_from' in item for item in goals):
            return 'POURING_GOAL_SCHEMA'
        if any('RIGHT_ANGLE_STRUCTURE' in item for item in pattern.visual_signatures):
            return 'RECTILINEAR_STRUCTURE_SCHEMA'
        if any('PARALLEL_EDGE_STRUCTURE' in item for item in pattern.visual_signatures):
            return 'PARALLEL_BOUNDARY_SCHEMA'
        if any('SYMMETRIC_STRUCTURE' in item for item in pattern.visual_signatures):
            return 'SYMMETRIC_STRUCTURE_SCHEMA'
        if any('AXIS_ALIGNED_STRUCTURE' in item for item in pattern.visual_signatures):
            return 'AXIS_ALIGNED_STRUCTURE_SCHEMA'
        if pattern.basis_signature:
            return 'EVOLVED_' + '_'.join(item.replace('_OPERATOR', '') for item in pattern.basis_signature[:3])
        return 'EVOLVED_OPERATOR_SCHEMA'

    @staticmethod
    def _fallback_rationale(pattern: OperatorProposalPattern) -> str:
        domains = ', '.join(sorted(set(pattern.source_domains))[:4]) or 'general'
        basis = ', '.join(pattern.basis_signature[:4]) or 'unknown basis'
        return f'Repeated pattern over {basis} observed across domains: {domains}.'


class OperatorProposalEngine:
    def __init__(self, summarizer: OperatorProposalSummarizer | None = None) -> None:
        self.summarizer = summarizer or OperatorProposalSummarizer()

    def propose(self, graphs: Sequence[StructuredMeaningGraph]) -> list[ModelProposedOperator]:
        patterns = self.collect_patterns(graphs)
        proposals: list[ModelProposedOperator] = []
        for pattern in patterns:
            name, rationale, proposal_source = self.summarizer.summarize(pattern)
            normalized = self._normalize_name(name)
            confidence = self._proposal_confidence(pattern)
            proposals.append(
                ModelProposedOperator(
                    proposed_name=name,
                    normalized_name=normalized,
                    basis_signature=list(pattern.basis_signature),
                    basis_operators=list(pattern.basis_operators),
                    source_operator_names=list(pattern.source_operator_names),
                    source_domains=list(pattern.source_domains),
                    support=pattern.support,
                    confidence=confidence,
                    proposal_source=proposal_source,
                    rationale=rationale,
                    evidence_patterns=list(pattern.evidence_patterns),
                    visual_signatures=list(pattern.visual_signatures),
                    hidden_goals=list(pattern.hidden_goals),
                )
            )
        return self._merge_proposals(proposals)

    def collect_patterns(self, graphs: Sequence[StructuredMeaningGraph]) -> list[OperatorProposalPattern]:
        grouped: dict[tuple[str, ...], OperatorProposalPattern] = {}
        for graph in graphs:
            domain = graph.domain or 'general'
            visual_signatures = self._graph_visual_signatures(graph)
            for decomp in graph.operator_decompositions:
                basis_signature = sorted(dict.fromkeys(str(item) for item in decomp.basis_operators if item))
                if len(basis_signature) < 2:
                    continue
                key = tuple(basis_signature)
                current = grouped.get(key)
                if current is None:
                    grouped[key] = OperatorProposalPattern(
                        basis_signature=basis_signature,
                        basis_operators=list(dict.fromkeys(decomp.basis_operators)),
                        source_operator_names=[decomp.operator_name],
                        source_domains=[domain],
                        support=1,
                        hidden_goals=list(dict.fromkeys(graph.hidden_goals)),
                        evidence_patterns=[decomp.rationale] if decomp.rationale else [],
                        visual_signatures=visual_signatures,
                    )
                    continue
                current.support += 1
                current.source_operator_names = sorted(set(current.source_operator_names + [decomp.operator_name]))
                current.source_domains = sorted(set(current.source_domains + [domain]))
                current.hidden_goals = sorted(set(current.hidden_goals + list(graph.hidden_goals)))
                current.visual_signatures = sorted(set(current.visual_signatures + visual_signatures))
                if decomp.rationale and decomp.rationale not in current.evidence_patterns:
                    current.evidence_patterns.append(decomp.rationale)
        return sorted(grouped.values(), key=lambda item: (-item.support, item.basis_signature))

    @staticmethod
    def _graph_visual_signatures(graph: StructuredMeaningGraph) -> list[str]:
        values: list[str] = []
        for node in graph.nodes:
            signatures = node.attributes.get('geometry_signature') if isinstance(node.attributes, dict) else None
            if isinstance(signatures, list):
                for item in signatures:
                    text = str(item)
                    if text:
                        values.append(text)
        return sorted(set(values))

    @staticmethod
    def _normalize_name(name: str) -> str:
        text = ''.join(ch if ch.isalnum() else '_' for ch in name.upper()).strip('_')
        while '__' in text:
            text = text.replace('__', '_')
        return text or 'EVOLVED_OPERATOR_SCHEMA'

    @staticmethod
    def _proposal_confidence(pattern: OperatorProposalPattern) -> float:
        support_score = min(1.0, pattern.support / 4.0)
        domain_score = min(1.0, len(set(pattern.source_domains)) / 3.0)
        visual_bonus = 0.1 if pattern.visual_signatures else 0.0
        goal_bonus = 0.1 if pattern.hidden_goals else 0.0
        return round(min(0.95, 0.45 + 0.2 * support_score + 0.15 * domain_score + visual_bonus + goal_bonus), 4)

    @staticmethod
    def _merge_proposals(items: Sequence[ModelProposedOperator]) -> list[ModelProposedOperator]:
        merged: dict[tuple[str, ...], ModelProposedOperator] = {}
        for item in items:
            key = tuple(item.basis_signature)
            current = merged.get(key)
            if current is None:
                merged[key] = ModelProposedOperator(**item.model_dump())
                continue
            current.support += item.support
            current.source_operator_names = sorted(set(current.source_operator_names + item.source_operator_names))
            current.source_domains = sorted(set(current.source_domains + item.source_domains))
            current.hidden_goals = sorted(set(current.hidden_goals + item.hidden_goals))
            current.visual_signatures = sorted(set(current.visual_signatures + item.visual_signatures))
            current.evidence_patterns = current.evidence_patterns + [row for row in item.evidence_patterns if row not in current.evidence_patterns]
            current.confidence = max(current.confidence, item.confidence)
            if current.proposal_source != 'llm_operator_summarizer' and item.proposal_source == 'llm_operator_summarizer':
                current.proposed_name = item.proposed_name
                current.normalized_name = item.normalized_name
                current.proposal_source = item.proposal_source
                current.rationale = item.rationale or current.rationale
        return sorted(merged.values(), key=lambda item: (-item.support, -item.confidence, item.normalized_name))

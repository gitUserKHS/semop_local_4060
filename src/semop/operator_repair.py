from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from .basis_operators import canonicalize_basis_signature
from .context_chunks import split_context_into_chunks
from .operator_repair_policy import OperatorRepairPolicyScorer
from .operator_runtime import compile_and_execute
from .retained_repair_programs import RetainedRepairProgramLibrary
from .structures import Edge, Node, OperatorDecomposition, StructuredMeaningGraph


@dataclass
class OperatorRepairAttempt:
    action: str
    detail: str
    applied: bool = True

    def model_dump(self) -> dict:
        return asdict(self)


class OperatorRepairEngine:
    def __init__(self, repair_policy_path: str | Path | None = None, repair_program_path: str | Path | None = None) -> None:
        self.policy = OperatorRepairPolicyScorer(model_path=repair_policy_path) if repair_policy_path else None
        self.program_library = RetainedRepairProgramLibrary(model_path=repair_program_path)

    def run(self, graph: StructuredMeaningGraph, max_rounds: int = 2) -> StructuredMeaningGraph:
        graph = compile_and_execute(graph)
        applied_actions: list[str] = []
        for _ in range(max(1, max_rounds)):
            report = graph.operator_execution
            if report is None:
                break
            if report.composition_score >= 0.65 and not report.counterexample_repairs:
                break
            round_actions, program_ids = self._apply_repairs(graph, report.compiler_findings, report.counterexample_repairs)
            if not round_actions:
                break
            applied_actions.extend(round_actions)
            note = 'operator repair: ' + ', '.join(round_actions[:3])
            if note not in graph.audit_trace:
                graph.audit_trace.append(note)
            if program_ids:
                program_note = 'repair synthesis: ' + ', '.join(program_ids[:2])
                if program_note not in graph.audit_trace:
                    graph.audit_trace.append(program_note)
            graph = compile_and_execute(graph)
            if graph.operator_execution is not None:
                for action in round_actions[:3]:
                    decision = f'repair_applied:{action}'
                    if decision not in graph.operator_execution.derived_decisions:
                        graph.operator_execution.derived_decisions.insert(0, decision)
                for program_id in program_ids[:2]:
                    decision = f'repair_program:{program_id}'
                    if decision not in graph.operator_execution.derived_decisions:
                        graph.operator_execution.derived_decisions.insert(0, decision)
        return graph

    def _apply_repairs(self, graph: StructuredMeaningGraph, findings: List[str], counterexample_repairs: List[str]) -> tuple[List[str], List[str]]:
        actions: List[str] = []
        text = ' '.join(findings + counterexample_repairs).lower()
        handlers = {
            'add_goal_preservation_decomposition': lambda: self._ensure_goal_preservation_decomposition(graph) if ('no explicit operator decomposition' in text or ('missing basis' in text and graph.hidden_goals and graph.required_premises)) else False,
            'bind_requires_edges': lambda: self._ensure_requires_edges(graph) if (('missing basis' in text or 'input types unsupported' in text or 'no explicit operator decomposition' in text) and (graph.required_premises or graph.satisfied_premises or graph.missing_premises)) else False,
            'attach_document_context_nodes': lambda: self._ensure_document_context(graph) if (('document_context' in text or 'document context' in text or 'evidence_span' in text) and graph.source_context.strip()) else False,
            'attach_visual_scene': lambda: self._ensure_visual_scene(graph) if (('visual structure' in text or 'visual source objects' in text) and self._has_visual_signal(graph)) else False,
            'rebind_functor_object_map': lambda: self._rebind_functors(graph) if 'object map is detached' in text else False,
        }
        synthesized_actions, program_ids = self.program_library.synthesize_actions(graph, findings, counterexample_repairs)
        ranked_actions = self.policy.rank_actions(graph, findings) if self.policy is not None else list(handlers.keys())
        candidate_order = list(dict.fromkeys(synthesized_actions + ranked_actions + list(handlers.keys())))
        for action in candidate_order:
            handler = handlers.get(action)
            if handler is None:
                continue
            if handler():
                actions.append(action)
        return actions, program_ids

    @staticmethod
    def _ensure_goal_preservation_decomposition(graph: StructuredMeaningGraph) -> bool:
        if any(item.operator_name == 'GOAL_PRESERVATION_OPERATOR' for item in graph.operator_decompositions):
            return False
        basis = ['HIDDEN_GOAL', 'REQUIRES']
        if graph.missing_premises or any(edge.relation == 'BLOCKED_BY' for edge in graph.edges):
            basis.append('BLOCKED_BY')
        graph.operator_decompositions.append(
            OperatorDecomposition(
                operator_name='GOAL_PRESERVATION_OPERATOR',
                basis_operators=canonicalize_basis_signature(basis),
                rationale='repair loop: recovered missing goal-preservation decomposition from hidden-goal and prerequisite state.',
                confidence=0.78,
            )
        )
        return True

    @staticmethod
    def _ensure_requires_edges(graph: StructuredMeaningGraph) -> bool:
        source = graph.hidden_goals[0] if graph.hidden_goals else 'question'
        existing = {(edge.source, edge.relation, edge.target) for edge in graph.edges}
        added = False
        for premise in list(dict.fromkeys(graph.required_premises + graph.satisfied_premises + graph.missing_premises)):
            triple = (source, 'REQUIRES', premise)
            if triple in existing:
                continue
            graph.add_edge(
                Edge(
                    source=source,
                    relation='REQUIRES',
                    target=premise,
                    confidence=0.72,
                    provenance=['repair_loop:requires_edge'],
                )
            )
            added = True
        return added

    @staticmethod
    def _ensure_document_context(graph: StructuredMeaningGraph) -> bool:
        added = False
        if not any(node.id == 'source_document' for node in graph.nodes):
            graph.add_node(
                Node(
                    id='source_document',
                    label='source_document',
                    kind='document',
                    attributes={'pdf_like': True},
                    provenance=['repair_loop:document_context'],
                )
            )
            graph.add_edge(Edge(source='question', relation='USES_CONTEXT', target='source_document', confidence=0.86, provenance=['repair_loop:document_context']))
            added = True
        existing_ids = {node.id for node in graph.nodes}
        for index, chunk in enumerate(split_context_into_chunks(graph.source_context), start=1):
            if chunk.chunk_id not in existing_ids:
                graph.add_node(
                    Node(
                        id=chunk.chunk_id,
                        label=chunk.text[:96],
                        kind='evidence',
                        attributes={'text': chunk.text, 'source': chunk.source, 'modality': 'document'},
                        provenance=['repair_loop:document_chunk'],
                    )
                )
                graph.add_edge(Edge(source='source_document', relation='HAS_EVIDENCE', target=chunk.chunk_id, confidence=0.82, provenance=['repair_loop:document_chunk']))
                added = True
            if index <= 2:
                before = len(graph.edges)
                graph.add_edge(Edge(source='question', relation='GROUNDED_BY', target=chunk.chunk_id, confidence=0.72, provenance=['repair_loop:document_grounding']))
                if len(graph.edges) != before:
                    added = True
        return added

    @staticmethod
    def _has_visual_signal(graph: StructuredMeaningGraph) -> bool:
        return any(any(tag.startswith('multimodal:vision') for tag in node.provenance) for node in graph.nodes)

    @staticmethod
    def _ensure_visual_scene(graph: StructuredMeaningGraph) -> bool:
        added = False
        if not any(node.id == 'visual_scene' for node in graph.nodes):
            graph.add_node(Node(id='visual_scene', label='visual_scene', kind='scene', provenance=['repair_loop:visual_scene']))
            graph.add_edge(Edge(source='question', relation='CONDITIONS_ON', target='visual_scene', confidence=0.8, provenance=['repair_loop:visual_scene']))
            added = True
        visual_nodes = [node for node in graph.nodes if any(tag.startswith('multimodal:vision') for tag in node.provenance) and node.kind != 'evidence']
        for index, node in enumerate(visual_nodes[:4], start=1):
            evidence_id = f'repair_visual_evidence_{index:03d}_{node.id}'
            before_nodes = len(graph.nodes)
            graph.add_node(Node(id=evidence_id, label=node.label[:96], kind='evidence', attributes={'text': f"visual evidence: {node.label}", 'source': 'vision', 'modality': 'vision', 'entity_id': node.id}, provenance=['repair_loop:visual_evidence']))
            if len(graph.nodes) != before_nodes:
                added = True
            before_edges = len(graph.edges)
            graph.add_edge(Edge(source='visual_scene', relation='HAS_EVIDENCE', target=evidence_id, confidence=0.76, provenance=['repair_loop:visual_evidence']))
            graph.add_edge(Edge(source='question', relation='GROUNDED_BY', target=evidence_id, confidence=0.74, provenance=['repair_loop:visual_grounding']))
            if len(graph.edges) != before_edges:
                added = True
        return added

    @staticmethod
    def _rebind_functors(graph: StructuredMeaningGraph) -> bool:
        changed = False
        node_ids = graph.node_ids()
        hidden = graph.hidden_goals[0] if graph.hidden_goals else ''
        premise = (graph.required_premises + graph.satisfied_premises + graph.missing_premises)
        premise_target = premise[0] if premise else ''
        for functor in graph.functor_hypotheses:
            if functor.name == 'ServiceGoalToConstraintFunctor' and ('car_wash' in node_ids or any(edge.relation == 'TYPICAL_FOR' for edge in graph.edges)):
                new_map = {'car_wash' if 'car_wash' in node_ids else 'service_place': hidden or 'hidden_goal'}
                if premise_target:
                    new_map['vehicle_present' if premise_target == 'vehicle_present' else premise_target] = premise_target
                if functor.object_map != new_map:
                    functor.object_map = new_map
                    changed = True
            elif hidden and premise_target and not functor.object_map:
                functor.object_map = {'hidden_goal': hidden, 'required_premise': premise_target}
                changed = True
        return changed
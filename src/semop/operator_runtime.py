from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .basis_operators import BASIS_OPERATOR_AXES, basis_operator_axis, canonicalize_basis_signature, infer_basis_operators, normalize_operator_symbol
from .commonsense_kb import concept_label
from .structures import OperatorExecutionReport, OperatorInstruction, StructuredMeaningGraph


@dataclass
class OperatorCompiler:
    def compile(self, graph: StructuredMeaningGraph) -> List[OperatorInstruction]:
        instructions: List[OperatorInstruction] = []
        basis_hits = infer_basis_operators(graph)
        for symbol in basis_hits:
            instructions.append(
                OperatorInstruction(
                    opcode='DECLARE_BASIS',
                    arguments={'operator': symbol, 'axis': basis_operator_axis(symbol)},
                    evidence=[f'basis:{symbol}'],
                    confidence=0.95,
                )
            )
        for hidden_goal in graph.hidden_goals:
            instructions.append(
                OperatorInstruction(
                    opcode='SET_GOAL',
                    arguments={'hidden_goal': hidden_goal},
                    evidence=[f'hidden_goal:{hidden_goal}'],
                    confidence=0.9,
                )
            )
        for item in graph.premise_validations:
            instructions.append(
                OperatorInstruction(
                    opcode='ASSERT_PREMISE',
                    arguments={
                        'premise': item.premise,
                        'hidden_goal': item.hidden_goal,
                        'status': item.status,
                        'requirement_state': item.requirement_state,
                        'support_score': item.support_score,
                        'contradiction_score': item.contradiction_score,
                    },
                    evidence=[item.rationale] if item.rationale else [],
                    confidence=max(0.0, min(1.0, item.support_score - item.contradiction_score + 0.5)),
                )
            )
        for item in graph.analogical_matches[:3]:
            instructions.append(
                OperatorInstruction(
                    opcode='ANALOGY_SIGNAL',
                    arguments={
                        'analogy_type': item.analogy_type,
                        'query': item.query,
                        'shared_requirements': item.shared_requirements[:3],
                        'shared_goals': item.shared_goals[:2],
                        'score': item.score,
                    },
                    evidence=[item.query] + item.shared_requirements[:2],
                    confidence=item.score,
                )
            )
        for item in graph.goal_preservation_checks:
            instructions.append(
                OperatorInstruction(
                    opcode='CHECK_GOAL_PRESERVATION',
                    arguments={
                        'action': item.action,
                        'hidden_goal': item.hidden_goal,
                        'status': item.status,
                        'rationale': item.rationale,
                    },
                    evidence=[item.rationale],
                    confidence=item.confidence,
                )
            )
        for item in graph.operator_decompositions:
            instructions.append(
                OperatorInstruction(
                    opcode='DEFINE_OPERATOR',
                    arguments={
                        'operator_name': item.operator_name,
                        'basis_operators': canonicalize_basis_signature(item.basis_operators),
                    },
                    evidence=[item.rationale] if item.rationale else [],
                    confidence=item.confidence,
                )
            )
        for edge in graph.edges:
            if edge.relation not in {'REQUIRES', 'BLOCKED_BY', 'TYPICAL_FOR', 'PART_OF', 'ALTERNATIVE', 'STRUCTURAL_PART_OF', 'CONTAINS', 'GROUNDED_BY', 'USES_CONTEXT'}:
                continue
            instructions.append(
                OperatorInstruction(
                    opcode='BIND_RELATION',
                    arguments={
                        'source': edge.source,
                        'relation': normalize_operator_symbol(edge.relation),
                        'target': edge.target,
                    },
                    evidence=[f'{edge.source}:{edge.relation}:{edge.target}'],
                    confidence=edge.confidence,
                )
            )
        return instructions


@dataclass
class OperatorExecutor:
    def execute(self, instructions: List[OperatorInstruction]) -> OperatorExecutionReport:
        satisfied: List[str] = []
        missing: List[str] = []
        decisions: List[str] = []
        warnings: List[str] = []
        support_trace: List[str] = []
        basis_hits: List[str] = []
        basis_axes: Dict[str, str] = {}
        referenced_basis: List[str] = []
        satisfied_ids: set[str] = set()
        missing_ids: set[str] = set()

        for instruction in instructions:
            args: Dict[str, Any] = instruction.arguments
            if instruction.opcode == 'DECLARE_BASIS':
                operator = str(args.get('operator', ''))
                axis = str(args.get('axis', 'unknown'))
                if operator and operator not in basis_hits:
                    basis_hits.append(operator)
                    basis_axes[operator] = axis
                    support_trace.append(f'basis:{operator}:{axis}')
                continue

            if instruction.opcode == 'SET_GOAL':
                support_trace.append(f"goal:{concept_label(str(args.get('hidden_goal', '')))}")
                referenced_basis.append('HIDDEN_GOAL')
                continue

            if instruction.opcode == 'ASSERT_PREMISE':
                premise_id = str(args.get('premise', ''))
                premise = concept_label(premise_id)
                state = str(args.get('requirement_state', 'unknown'))
                status = str(args.get('status', 'supported'))
                referenced_basis.append('REQUIRES')
                if state in {'satisfied', 'available'} and status in {'supported', 'recovered'}:
                    if premise not in satisfied:
                        satisfied.append(premise)
                        satisfied_ids.add(premise_id)
                elif state in {'missing', 'blocked'} or status in {'unsupported', 'contradicted'}:
                    if premise not in missing:
                        missing.append(premise)
                        missing_ids.add(premise_id)
                        warnings.append(f'missing premise: {premise}')
                    referenced_basis.append('BLOCKED_BY')
                support_trace.append(f'premise:{premise}:{status}:{state}')
                continue

            if instruction.opcode == 'ANALOGY_SIGNAL':
                analogy_type = str(args.get('analogy_type', 'surface_analogy'))
                shared_requirements = [str(item) for item in args.get('shared_requirements', [])]
                shared_goals = [str(item) for item in args.get('shared_goals', [])]
                risky_requirements = [item for item in shared_requirements if item in missing_ids]
                if not risky_requirements and analogy_type in {'goal_premise_analogy', 'failure_analogy'}:
                    risky_requirements = [item for item in shared_requirements if item not in satisfied_ids]
                if risky_requirements:
                    labels = ', '.join(concept_label(item) for item in risky_requirements[:3])
                    warnings.append(f'analogy risk: similar cases also depended on {labels}')
                    decisions.append(f'analogy_guard:{analogy_type}:{risky_requirements[0]}')
                    referenced_basis.extend(['HIDDEN_GOAL', 'REQUIRES'])
                    referenced_basis.append('BLOCKED_BY')
                if shared_goals:
                    support_trace.append(f"analogy:{analogy_type}:{shared_goals[0]}")
                else:
                    support_trace.append(f"analogy:{analogy_type}")
                continue

            if instruction.opcode == 'CHECK_GOAL_PRESERVATION':
                action = str(args.get('action', ''))
                goal = concept_label(str(args.get('hidden_goal', '')))
                status = str(args.get('status', 'unknown'))
                decisions.append(f'{action}:{goal}:{status}')
                referenced_basis.extend(['HIDDEN_GOAL', 'REQUIRES'])
                if status in {'risk_high', 'invalid', 'blocked'}:
                    warnings.append(f'goal risk: {action} threatens {goal}')
                    referenced_basis.append('BLOCKED_BY')
                support_trace.append(f'goal_check:{action}:{goal}:{status}')
                continue

            if instruction.opcode == 'DEFINE_OPERATOR':
                name = str(args.get('operator_name', ''))
                basis = canonicalize_basis_signature(args.get('basis_operators', []))
                referenced_basis.extend(basis)
                support_trace.append(f"operator:{name}<-{','.join(basis)}")
                continue

            if instruction.opcode == 'BIND_RELATION':
                relation = normalize_operator_symbol(str(args.get('relation', '')))
                if relation:
                    referenced_basis.append(relation)
                support_trace.append(f"relation:{args.get('source','')}:{relation}:{args.get('target','')}")

        declared_basis = set(basis_hits)
        referenced_basis_set = set(canonicalize_basis_signature(referenced_basis))
        if referenced_basis_set:
            compiler_alignment_score = len(declared_basis & referenced_basis_set) / float(len(referenced_basis_set))
        else:
            compiler_alignment_score = 1.0 if declared_basis else 0.0

        return OperatorExecutionReport(
            satisfied_facts=satisfied,
            missing_facts=missing,
            derived_decisions=decisions,
            warnings=list(dict.fromkeys(warnings)),
            support_trace=support_trace[:40],
            basis_operator_hits=basis_hits,
            basis_operator_axes=basis_axes,
            compiler_alignment_score=round(compiler_alignment_score, 4),
        )


@dataclass
class OperatorCompositionVerifier:
    def verify(self, graph: StructuredMeaningGraph, report: OperatorExecutionReport) -> OperatorExecutionReport:
        findings: List[str] = []
        repairs: List[str] = []
        scores: List[float] = []
        available = self._available_basis(graph)
        available_types = self._available_types(graph)
        candidate_lookup = {item.name: item for item in graph.induced_operators}

        if graph.hidden_goals and graph.required_premises and not graph.operator_decompositions:
            findings.append('compiler warning: hidden-goal reasoning has no explicit operator decomposition.')
            repairs.append('repair: emit an explicit GOAL_PRESERVATION_OPERATOR before execution.')
            scores.append(0.0)

        for item in graph.operator_decompositions:
            basis = canonicalize_basis_signature(item.basis_operators)
            if not basis:
                continue
            score = 1.0
            missing = [name for name in basis if name not in available]
            if missing:
                score -= len(missing) / float(len(basis))
                findings.append('compiler warning: ' + item.operator_name + ' missing basis ' + ', '.join(missing))
                repairs.append(self._repair_hint_for_missing_basis(item.operator_name, missing))
            illegal = [name for name in basis if not self._is_legal_basis_symbol(name, candidate_lookup)]
            if illegal:
                score -= min(0.35, 0.15 * len(illegal))
                findings.append('compiler warning: ' + item.operator_name + ' uses illegal basis ' + ', '.join(illegal))
                repairs.append(f"repair: replace illegal basis in {item.operator_name} with canonical basis operators or retained operators.")
            candidate = candidate_lookup.get(item.operator_name)
            if candidate is not None:
                missing_inputs = [name for name in candidate.input_types if not self._type_supported(name, available_types)]
                if missing_inputs:
                    score -= min(0.3, 0.1 * len(missing_inputs))
                    findings.append('compiler warning: ' + item.operator_name + ' input types unsupported: ' + ', '.join(missing_inputs))
                    repairs.append(f"repair: provide {missing_inputs[0]} evidence before invoking {item.operator_name}.")
                if not self._output_legal(candidate.output_type, available_types):
                    score -= 0.12
                    findings.append(f'compiler warning: {item.operator_name} output type {candidate.output_type} is not grounded in the current category frame.')
                    repairs.append(f"repair: add a downstream consumer or grounding edge for {candidate.output_type}.")
            scores.append(max(0.0, score))
            if score >= 0.999:
                findings.append(f'compiler verified: {item.operator_name} composition closes over declared basis')

        for functor in graph.functor_hypotheses:
            valid, detail, repair = self._check_functor_consistency(functor, graph, available_types)
            if not valid:
                findings.append('compiler warning: ' + detail)
                repairs.append(repair)
                scores.append(0.45)
            else:
                scores.append(0.9)

        grounding_score, grounding_findings, grounding_repairs = self._check_grounding_fidelity(graph)
        if grounding_findings:
            findings.extend(grounding_findings)
        if grounding_repairs:
            repairs.extend(grounding_repairs)
        if graph.source_context.strip() or any(any(tag.startswith('multimodal:vision') for tag in node.provenance) for node in graph.nodes):
            scores.append(grounding_score)

        composition_score = round(sum(scores) / float(len(scores)), 4) if scores else 1.0
        report.compiler_findings = findings[:10]
        report.counterexample_repairs = list(dict.fromkeys(repairs))[:8]
        report.composition_score = composition_score
        if composition_score < 0.65:
            warning = f'compiler composition risk: score={composition_score:.2f}'
            if warning not in report.warnings:
                report.warnings.append(warning)
        for repair in report.counterexample_repairs[:3]:
            repair_warning = 'counterexample repair: ' + repair
            if repair_warning not in report.warnings:
                report.warnings.append(repair_warning)
        return report

    @staticmethod
    def _available_basis(graph: StructuredMeaningGraph) -> set[str]:
        available: set[str] = set()
        if graph.hidden_goals:
            available.add('HIDDEN_GOAL')
        if graph.required_premises or any(edge.relation == 'REQUIRES' for edge in graph.edges):
            available.add('REQUIRES')
        if graph.missing_premises or any(edge.relation == 'BLOCKED_BY' for edge in graph.edges) or any(item.status in {'risk_high', 'blocked', 'invalid'} for item in graph.goal_preservation_checks):
            available.add('BLOCKED_BY')
        if any(edge.relation == 'TYPICAL_FOR' for edge in graph.edges):
            available.add('TYPICAL_FOR')
        if any(edge.relation == 'CONTAINS' for edge in graph.edges):
            available.add('CONTAINS')
        if graph.source_context.strip():
            available.add('DOCUMENT_CONTEXT')
        if any(any(tag.startswith('multimodal:vision') for tag in node.provenance) for node in graph.nodes):
            available.add('VISUAL_STRUCTURE')
        return available

    @staticmethod
    def _available_types(graph: StructuredMeaningGraph) -> set[str]:
        types = {graph.intent, graph.domain, 'question'}
        types.update(node.kind for node in graph.nodes if node.kind)
        if graph.source_context.strip():
            types.update({'document_context', 'evidence_span'})
        if graph.hidden_goals:
            types.add('goal_directed_reasoning')
            types.add('hidden_goal')
        if graph.required_premises or graph.satisfied_premises or graph.missing_premises:
            types.add('premise_state')
        if any(any(tag.startswith('multimodal:vision') for tag in node.provenance) for node in graph.nodes):
            types.add('visual_structure')
        if any(result.domain == 'document_grounding' for result in graph.symbolic_results):
            types.add('document_grounding')
        return {item for item in types if item}

    @staticmethod
    def _is_legal_basis_symbol(symbol: str, candidate_lookup: Dict[str, Any]) -> bool:
        normalized = normalize_operator_symbol(symbol)
        return normalized in BASIS_OPERATOR_AXES or symbol in candidate_lookup or normalized in {'DOCUMENT_CONTEXT', 'VISUAL_STRUCTURE'}

    @staticmethod
    def _type_supported(input_type: str, available_types: set[str]) -> bool:
        lowered = input_type.lower()
        if input_type in available_types:
            return True
        if lowered in {'relation_frame', 'meaning_state'}:
            return True
        if lowered == 'document_context':
            return 'document_context' in available_types
        if lowered == 'question':
            return 'question' in available_types
        if lowered.startswith('visual'):
            return 'visual_structure' in available_types
        if lowered in {'goal_directed_reasoning', 'constraint_reasoning'}:
            return any(item in available_types for item in {'goal_directed_reasoning', 'constraint_reasoning', 'generic_reasoning'})
        if lowered in {'hidden_goal', 'required_premise', 'premise_state'}:
            return any(item in available_types for item in {'hidden_goal', 'premise_state'})
        return False

    @staticmethod
    def _output_legal(output_type: str, available_types: set[str]) -> bool:
        lowered = output_type.lower()
        if lowered in {'relation_frame', 'meaning_state', 'proof_outline', 'numeric_answer', 'visual_relation_frame'}:
            return True
        if lowered == 'evidence_span':
            return 'document_context' in available_types or 'evidence_span' in available_types
        return output_type in available_types

    def _check_functor_consistency(self, functor, graph: StructuredMeaningGraph, available_types: set[str]) -> tuple[bool, str, str]:
        source = functor.source_category.lower()
        target = functor.target_category.lower()
        node_ids = graph.node_ids()
        if 'service' in source and not ({'car_wash', 'service_place'} & node_ids or any(edge.relation == 'TYPICAL_FOR' for edge in graph.edges)):
            return True, '', ''
        if 'visual' in source and 'visual_structure' not in available_types:
            return False, f'{functor.name} expects visual source objects but the graph has no visual structure.', 'provide visual entities or structural operators before applying the functor.'
        if 'goal' in target and not graph.hidden_goals:
            return False, f'{functor.name} targets goal-preservation logic but no hidden goal was recovered.', 'recover or assert a hidden goal before applying the functor.'
        if 'constraint' in target and not (graph.required_premises or graph.missing_premises or graph.satisfied_premises):
            return False, f'{functor.name} maps into constraints but the graph has no premise state.', 'derive premise states before functor composition.'
        if functor.object_map and not any(key in node_ids or value in graph.hidden_goals + graph.required_premises + graph.satisfied_premises + graph.missing_premises for key, value in functor.object_map.items()):
            return True, '', ''
        return True, '', ''

    @staticmethod
    def _check_grounding_fidelity(graph: StructuredMeaningGraph) -> tuple[float, List[str], List[str]]:
        findings: List[str] = []
        repairs: List[str] = []
        evidence_lookup = {node.id: node for node in graph.nodes if node.kind == 'evidence'}
        grounded_nodes = [
            evidence_lookup[edge.target]
            for edge in graph.edges
            if edge.source == 'question' and edge.relation == 'GROUNDED_BY' and edge.target in evidence_lookup
        ]
        has_document_context = bool(graph.source_context.strip())
        has_visual_context = any(any(tag.startswith('multimodal:vision') for tag in node.provenance) for node in graph.nodes)
        document_grounded = any(
            str(node.attributes.get('modality', '')).lower() == 'document'
            or 'document' in ' '.join(node.provenance).lower()
            or str(node.attributes.get('source', '')).lower().startswith('context:')
            for node in grounded_nodes
        )
        visual_grounded = any(
            str(node.attributes.get('modality', '')).lower() == 'vision'
            or 'vision' in ' '.join(node.provenance).lower()
            for node in grounded_nodes
        )
        score = 1.0
        if has_document_context and not document_grounded:
            score -= 0.35
            findings.append('grounding compiler warning: document-backed reasoning has no explicit GROUNDED_BY evidence edge.')
            repairs.append('repair: attach document evidence nodes and ground the question on them before answering.')
        if has_visual_context and not visual_grounded:
            score -= 0.35
            findings.append('grounding compiler warning: visual reasoning has no explicit GROUNDED_BY visual evidence edge.')
            repairs.append('repair: attach visual evidence nodes and ground the question on them before answering.')
        if any(result.domain == 'document_grounding' for result in graph.symbolic_results) and not document_grounded:
            score -= 0.1
        return max(0.0, score), findings[:3], repairs[:3]

    @staticmethod
    def _repair_hint_for_missing_basis(operator_name: str, missing: List[str]) -> str:
        hints = {
            'HIDDEN_GOAL': 'recover the hidden goal signal from query or memory before composing.',
            'REQUIRES': 'bind the prerequisite edges explicitly before composition.',
            'BLOCKED_BY': 'add blocker or contradiction evidence before trusting the operator.',
            'CONTAINS': 'ground an interior or containment relation from text or vision first.',
            'TYPICAL_FOR': 'recover the supporting script or typical-purpose relation from memory.',
            'DOCUMENT_CONTEXT': 'attach document chunks or PDF paragraphs before evidence composition.',
            'VISUAL_STRUCTURE': 'attach visual entities and structural operators before multimodal composition.',
        }
        details = [hints.get(item, 'supply the missing basis operator before execution.') for item in missing]
        return f"{operator_name}: {' '.join(dict.fromkeys(details))}"


def compile_and_execute(graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
    compiler = OperatorCompiler()
    instructions = compiler.compile(graph)
    report = OperatorExecutor().execute(instructions)
    report = OperatorCompositionVerifier().verify(graph, report)
    graph.operator_instructions = instructions
    graph.operator_execution = report
    if instructions:
        graph.audit_trace.append(f'operator runtime: compiled {len(instructions)} instructions')
    if report.derived_decisions or report.basis_operator_hits:
        graph.audit_trace.append('operator runtime: executed compiled program')
    for warning in report.warnings[:6]:
        if warning not in graph.warnings:
            graph.warnings.append(warning)
    return graph
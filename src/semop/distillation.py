from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .contest_programmer import CompetitiveProgrammingReasoner
from .operator_evolution import OperatorSelfEvolutionEngine
from .pipeline import StructuredMeaningPipeline
from .structures import OperatorDecomposition, FunctorHypothesis
from .vlso.reasoner import VLSOReasoner


@dataclass
class TeacherTraceRecord:
    task: str
    input_text: str
    input_payload: dict[str, Any]
    teacher_trace: dict[str, Any]
    completion_payload: dict[str, Any]
    metadata: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DistillationSftRecord:
    prompt: str
    completion: str
    task: str
    metadata: dict[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class TeacherTraceExporter:
    def __init__(
        self,
        premise_pipeline: StructuredMeaningPipeline | None = None,
        cp_reasoner: CompetitiveProgrammingReasoner | None = None,
        vlso_reasoner: VLSOReasoner | None = None,
    ) -> None:
        self.premise_pipeline = premise_pipeline or StructuredMeaningPipeline(mode='heuristic')
        self.cp_reasoner = cp_reasoner or CompetitiveProgrammingReasoner()
        self.vlso_reasoner = vlso_reasoner or VLSOReasoner(mode='hybrid', language_mode='heuristic', answer_mode='structured')

    def export_hidden_premise_eval(self, path: str | Path) -> list[TeacherTraceRecord]:
        records: list[TeacherTraceRecord] = []
        for payload in self._iter_jsonl(path):
            query = str(payload.get('query', '')).strip()
            if not query:
                continue
            graph = self.premise_pipeline.run(query)
            teacher_trace = {
                'surface_parse': {
                    'intent': graph.intent,
                    'domain': graph.domain,
                    'nodes': [node.id for node in graph.nodes[:10]],
                    'edges': [f"{edge.source}:{edge.relation}:{edge.target}" for edge in graph.edges[:12]],
                },
                'hidden_goals': list(graph.hidden_goals),
                'required_premises': list(graph.required_premises),
                'satisfied_premises': list(graph.satisfied_premises),
                'missing_premises': list(graph.missing_premises),
                'optional_interpretations': list(graph.optional_interpretations),
                'premise_candidates': [asdict(item) for item in graph.premise_candidates[:12]],
                'premise_validations': [asdict(item) for item in graph.premise_validations[:12]],
                'goal_preservation_checks': [asdict(item) for item in graph.goal_preservation_checks[:8]],
                'operator_support': [item.name for item in graph.induced_operators[:8]],
                'clarification': {
                    'needed': graph.clarification_needed,
                    'score': graph.clarification_score,
                    'reasons': list(graph.clarification_reasons),
                },
            }
            completion_payload = {
                'hidden_goals': list(graph.hidden_goals),
                'required_premises': list(graph.required_premises),
                'satisfied_premises': list(graph.satisfied_premises),
                'missing_premises': list(graph.missing_premises),
                'goal_preservation_checks': [asdict(item) for item in graph.goal_preservation_checks[:6]],
                'clarification_needed': graph.clarification_needed,
                'clarification_score': graph.clarification_score,
            }
            records.append(
                TeacherTraceRecord(
                    task='hidden_premise',
                    input_text=query,
                    input_payload={'query': query},
                    teacher_trace=teacher_trace,
                    completion_payload=completion_payload,
                    metadata={
                        'source': str(path),
                        'gold_reference': payload,
                        'mode': self.premise_pipeline.mode,
                    },
                )
            )
        return records

    def export_cp_parser_eval(self, path: str | Path) -> list[TeacherTraceRecord]:
        records: list[TeacherTraceRecord] = []
        for payload in self._iter_jsonl(path):
            statement = str(payload.get('statement', '')).strip()
            if not statement:
                continue
            structure = self.cp_reasoner.parse_problem(statement)
            if structure is None:
                continue
            teacher_trace = {
                'goal_types': list(structure.goal_types),
                'domain_tags': list(structure.domain_tags),
                'extracted_constraints': list(structure.extracted_constraints),
                'detected_phrases': list(structure.detected_phrases),
                'hidden_concepts': list(structure.hidden_concepts),
                'logical_frames': list(structure.logical_frames),
                'dsl_operators': list(structure.dsl_operators),
                'candidate_algorithms': list(structure.candidate_algorithms),
                'episode_priors': dict(structure.episode_priors),
            }
            completion_payload = {
                'goal_types': list(structure.goal_types),
                'domain_tags': list(structure.domain_tags),
                'logical_frames': list(structure.logical_frames),
                'dsl_operators': list(structure.dsl_operators),
                'target_algorithm': payload.get('target_algorithm', structure.candidate_algorithms[0] if structure.candidate_algorithms else ''),
                'reasoning_sketch': payload.get('reasoning_sketch', ''),
            }
            records.append(
                TeacherTraceRecord(
                    task='cp_structuring',
                    input_text=statement,
                    input_payload={'statement': statement},
                    teacher_trace=teacher_trace,
                    completion_payload=completion_payload,
                    metadata={
                        'source': str(path),
                        'gold_reference': payload,
                    },
                )
            )
        return records

    def export_vlso_eval(self, path: str | Path, base_dir: str | Path | None = None) -> list[TeacherTraceRecord]:
        base_path = Path(base_dir) if base_dir else Path(path).parent
        records: list[TeacherTraceRecord] = []
        for payload in self._iter_jsonl(path):
            query = str(payload.get('query', '')).strip()
            visual_ref = payload.get('visual_json') or payload.get('image_path')
            if not query or not visual_ref:
                continue
            visual_path = Path(str(visual_ref))
            if not visual_path.is_absolute():
                visual_path = base_path / visual_ref
            world, answer = self.vlso_reasoner.answer(query, visual_input=str(visual_path))
            teacher_trace = {
                'entities': [entity.model_dump() for entity in world.entities[:16]],
                'relations': [relation.model_dump() for relation in world.relations[:16]],
                'operators': [operator.model_dump() for operator in world.operators[:12]],
                'constraints': list(world.constraints),
                'inferred_steps': list(world.inferred_steps),
                'warnings': list(world.warnings),
                'answer': answer.model_dump(),
            }
            completion_payload = {
                'answer_text': answer.answer_text,
                'entities': [entity.label for entity in world.entities if entity.modality == 'vision'][:10],
                'required_relations': [relation.model_dump() for relation in world.relations[:8]],
                'operators': [operator.name for operator in world.operators[:8]],
            }
            records.append(
                TeacherTraceRecord(
                    task='vlso_grounded_qa',
                    input_text=query,
                    input_payload={'query': query, 'visual_input': str(visual_path)},
                    teacher_trace=teacher_trace,
                    completion_payload=completion_payload,
                    metadata={
                        'source': str(path),
                        'case_id': payload.get('case_id', ''),
                        'gold_reference': payload,
                    },
                )
            )
        return records


    def export_operator_transfer_eval(self, path: str | Path) -> list[TeacherTraceRecord]:
        records: list[TeacherTraceRecord] = []
        cases = self._iter_jsonl(path)
        graphs = []
        train_graphs = []
        for payload in cases:
            query = str(payload.get('query', '')).strip()
            if not query:
                continue
            graph = self.premise_pipeline.run(query)
            graphs.append((payload, graph))
            if str(payload.get('split', 'train')) == 'train':
                train_graphs.append(graph)
            teacher_trace = {
                'domain': payload.get('domain', 'general'),
                'split': payload.get('split', 'train'),
                'hidden_goals': list(graph.hidden_goals),
                'required_premises': list(graph.required_premises),
                'operator_decompositions': [asdict(item) for item in graph.operator_decompositions[:12]],
                'functor_hypotheses': [asdict(item) for item in graph.functor_hypotheses[:8]],
                'induced_operators': [item.name for item in graph.induced_operators[:12]],
                'premise_validations': [asdict(item) for item in graph.premise_validations[:10]],
            }
            completion_payload = {
                'proposed_operator_names': [item.operator_name for item in graph.operator_decompositions[:8]],
                'operator_decompositions': [asdict(item) for item in graph.operator_decompositions[:8]],
                'functor_hypotheses': [asdict(item) for item in graph.functor_hypotheses[:4]],
            }
            records.append(
                TeacherTraceRecord(
                    task='operator_proposal',
                    input_text=query,
                    input_payload={'query': query, 'domain': payload.get('domain', 'general')},
                    teacher_trace=teacher_trace,
                    completion_payload=completion_payload,
                    metadata={
                        'source': str(path),
                        'gold_reference': payload,
                    },
                )
            )
        if train_graphs:
            evolution = OperatorSelfEvolutionEngine().evolve(train_graphs)
            records.append(
                TeacherTraceRecord(
                    task='operator_self_evolution',
                    input_text='self-evolution summary',
                    input_payload={'num_train_graphs': len(train_graphs)},
                    teacher_trace={
                        'train_graph_count': len(train_graphs),
                        'proposal_count': len(evolution.proposals),
                        'retained_count': evolution.retained_count,
                        'proposals': [item.model_dump() for item in evolution.proposals[:24]],
                    },
                    completion_payload={
                        'retained_operator_names': [item.name for item in evolution.proposals if item.retained],
                        'retained_operator_count': evolution.retained_count,
                        'proposal_summaries': [item.model_dump() for item in evolution.proposals[:12]],
                    },
                    metadata={'source': str(path)},
                )
            )
        return records

    @staticmethod
    def to_sft_records(records: Iterable[TeacherTraceRecord]) -> list[DistillationSftRecord]:
        output: list[DistillationSftRecord] = []
        for record in records:
            output.append(
                DistillationSftRecord(
                    prompt=TeacherTraceExporter._prompt_for(record),
                    completion=json.dumps(record.completion_payload, ensure_ascii=False),
                    task=record.task,
                )
            )
        return output

    @staticmethod
    def save_jsonl(path: str | Path, rows: Iterable[TeacherTraceRecord | DistillationSftRecord]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row.model_dump(), ensure_ascii=False) + '\n')

    @staticmethod
    def _iter_jsonl(path: str | Path) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with Path(path).open('r', encoding='utf-8-sig') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    @staticmethod
    def _prompt_for(record: TeacherTraceRecord) -> str:
        if record.task == 'hidden_premise':
            return (
                'Analyze the query and return structured hidden-premise reasoning. '
                'Output JSON fields: hidden_goals, required_premises, satisfied_premises, missing_premises, '
                'goal_preservation_checks, clarification_needed, clarification_score.\n\n'
                f"Query:\n{record.input_text}\n\n"
                f"Teacher trace context:\n{json.dumps(record.teacher_trace, ensure_ascii=False)}"
            )
        if record.task == 'cp_structuring':
            return (
                'Read the competitive-programming statement and return structured parsing fields. '
                'Output JSON fields: goal_types, domain_tags, logical_frames, dsl_operators, target_algorithm, reasoning_sketch.\n\n'
                f"Statement:\n{record.input_text}\n\n"
                f"Teacher trace context:\n{json.dumps(record.teacher_trace, ensure_ascii=False)}"
            )
        if record.task == 'operator_proposal':
            return (
                'Read the query and return operator proposal fields. '
                'Output JSON fields: proposed_operator_names, operator_decompositions, functor_hypotheses.\n\n'
                f"Query:\n{record.input_text}\n\n"
                f"Teacher trace context:\n{json.dumps(record.teacher_trace, ensure_ascii=False)}"
            )
        if record.task == 'operator_self_evolution':
            return (
                'Read the train-domain operator summaries and return retained operator evolution output. '
                'Output JSON fields: retained_operator_names, retained_operator_count, proposal_summaries.\n\n'
                f"Context:\n{record.input_text}\n\n"
                f"Teacher trace context:\n{json.dumps(record.teacher_trace, ensure_ascii=False)}"
            )
        return (
            'Answer the visual question using grounded structured reasoning. '
            'Output JSON fields: answer_text, entities, required_relations, operators.\n\n'
            f"Question:\n{record.input_text}\n\n"
            f"Teacher trace context:\n{json.dumps(record.teacher_trace, ensure_ascii=False)}"
        )


from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .operator_repair_policy import REPAIR_ACTIONS
from .operator_runtime import compile_and_execute
from .structures import StructuredMeaningGraph


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9a-z_]+", text.lower())


@dataclass
class RepairUtilityModel:
    action_feature_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    action_bias: dict[str, float] = field(default_factory=dict)
    program_feature_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    program_bias: dict[str, float] = field(default_factory=dict)
    action_min_utility: float = -0.02
    program_min_utility: float = -0.04
    trained_on_examples: int = 0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: str | Path | None) -> RepairUtilityModel | None:
        if not path:
            return None
        payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if isinstance(payload, dict) and 'weights' in payload:
            payload = payload['weights']
        return cls(**payload)


@dataclass
class RepairUtilityTrainingSummary:
    output_path: str
    trained_on_examples: int
    model: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class RepairUtilityTrainer:
    def train_from_graphs(
        self,
        graphs: Sequence[StructuredMeaningGraph],
        output_path: str | Path,
    ) -> RepairUtilityTrainingSummary:
        from .operator_repair import OperatorRepairEngine

        graphs = list(graphs)
        engine = OperatorRepairEngine()
        action_examples: list[tuple[list[str], str, float]] = []
        program_examples: list[tuple[list[str], str, float]] = []
        for graph in graphs:
            action_examples.extend(self._trace_action_examples(graph))
            program_examples.extend(self._trace_program_examples(graph))
            for broken in self._synthetic_breakages(graph):
                broken = compile_and_execute(broken)
                repaired = engine.run(StructuredMeaningGraph.from_dict(broken.model_dump()))
                features = self._features(broken)
                utility = self._benchmark_delta(broken, repaired)
                report = repaired.operator_execution
                if report is None:
                    continue
                for action in self._applied_actions(repaired):
                    action_examples.append((features, action, utility))
                for program_id in self._applied_programs(repaired):
                    program_examples.append((features, program_id, utility))
                rejected_penalty = min(-0.25, utility if utility < 0.0 else -0.25)
                for action in self._rejected_actions(repaired):
                    action_examples.append((features, action, rejected_penalty))

        action_feature_weights: dict[str, dict[str, float]] = {action: {} for action in REPAIR_ACTIONS}
        action_bias: dict[str, float] = {action: 0.0 for action in REPAIR_ACTIONS}
        for features, action, utility in action_examples:
            action_bias[action] = round(float(action_bias.get(action, 0.0)) + float(utility), 4)
            for feature in features:
                bucket = action_feature_weights.setdefault(action, {})
                bucket[feature] = round(float(bucket.get(feature, 0.0)) + float(utility), 4)

        program_feature_weights: dict[str, dict[str, float]] = {}
        program_bias: dict[str, float] = {}
        for features, program_id, utility in program_examples:
            program_bias[program_id] = round(float(program_bias.get(program_id, 0.0)) + float(utility), 4)
            bucket = program_feature_weights.setdefault(program_id, {})
            for feature in features:
                bucket[feature] = round(float(bucket.get(feature, 0.0)) + float(utility), 4)

        model = RepairUtilityModel(
            action_feature_weights=action_feature_weights,
            action_bias=action_bias,
            program_feature_weights=program_feature_weights,
            program_bias=program_bias,
            trained_on_examples=len(action_examples) + len(program_examples),
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'weights': model.model_dump()}, ensure_ascii=False, indent=2), encoding='utf-8')
        return RepairUtilityTrainingSummary(
            output_path=str(output),
            trained_on_examples=model.trained_on_examples,
            model=model.model_dump(),
        )

    @staticmethod
    def _clone_graph(graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        return StructuredMeaningGraph.from_dict(graph.model_dump())

    def _synthetic_breakages(self, graph: StructuredMeaningGraph) -> list[StructuredMeaningGraph]:
        broken_graphs: list[StructuredMeaningGraph] = []
        if graph.hidden_goals and graph.required_premises:
            broken = self._clone_graph(graph)
            broken.operator_decompositions = []
            broken_graphs.append(broken)
        if graph.required_premises or graph.satisfied_premises or graph.missing_premises:
            broken = self._clone_graph(graph)
            broken.edges = [edge for edge in broken.edges if edge.relation != 'REQUIRES']
            broken_graphs.append(broken)
        if graph.source_context.strip():
            broken = self._clone_graph(graph)
            broken.nodes = [node for node in broken.nodes if node.id not in {'source_document'} and node.kind != 'evidence']
            broken.edges = [edge for edge in broken.edges if edge.relation not in {'USES_CONTEXT', 'HAS_EVIDENCE', 'GROUNDED_BY'}]
            broken_graphs.append(broken)
        if any(any(tag.startswith('multimodal:vision') for tag in node.provenance) for node in graph.nodes):
            broken = self._clone_graph(graph)
            broken.nodes = [node for node in broken.nodes if node.id != 'visual_scene']
            broken.edges = [edge for edge in broken.edges if not (edge.source == 'question' and edge.relation == 'CONDITIONS_ON' and edge.target == 'visual_scene')]
            broken_graphs.append(broken)
        if any(result.domain == 'document_grounding' and result.answer.strip() for result in graph.symbolic_results):
            positive = self._clone_graph(graph)
            for result in positive.symbolic_results:
                if result.domain == 'document_grounding' and result.answer.strip() and ' and ' not in result.answer:
                    result.answer = result.answer.rstrip('. ') + ' and inspect the hidden sensor.'
            broken_graphs.append(positive)
            negative = self._clone_graph(graph)
            for result in negative.symbolic_results:
                if result.domain == 'document_grounding' and result.answer.strip():
                    result.answer = 'Inspect the hidden sensor.'
            broken_graphs.append(negative)
        if graph.functor_hypotheses:
            broken = self._clone_graph(graph)
            for functor in broken.functor_hypotheses:
                functor.object_map = {}
            broken_graphs.append(broken)
        return broken_graphs

    def _trace_action_examples(self, graph: StructuredMeaningGraph) -> list[tuple[list[str], str, float]]:
        if graph.operator_execution is None:
            return []
        features = self._features(graph)
        utility = self._trace_utility(graph)
        examples = [(features, action, utility) for action in self._applied_actions(graph)]
        examples.extend((features, action, min(-0.25, -abs(utility) or -0.25)) for action in self._rejected_actions(graph))
        return examples

    def _trace_program_examples(self, graph: StructuredMeaningGraph) -> list[tuple[list[str], str, float]]:
        if graph.operator_execution is None:
            return []
        features = self._features(graph)
        utility = self._trace_utility(graph)
        return [(features, program_id, utility) for program_id in self._applied_programs(graph)]

    @classmethod
    def _features(cls, graph: StructuredMeaningGraph) -> list[str]:
        report = graph.operator_execution
        findings = ' '.join((report.compiler_findings + report.counterexample_repairs) if report is not None else [])
        features = set(_tokenize(findings))
        if graph.hidden_goals:
            features.add('has_hidden_goal')
        if graph.required_premises or graph.satisfied_premises or graph.missing_premises:
            features.add('has_premise_state')
        if graph.source_context.strip():
            features.add('has_document_context')
        if any(any(tag.startswith('multimodal:vision') for tag in node.provenance) for node in graph.nodes):
            features.add('has_visual_signal')
        if graph.functor_hypotheses:
            features.add('has_functor')
        if report is not None and report.claim_groundings:
            features.add('has_claim_grounding')
            grounded_claims = [item for item in report.claim_groundings if item.grounded]
            unsupported_claims = [item for item in report.claim_groundings if not item.grounded]
            if grounded_claims:
                features.add('has_grounded_claim')
            if unsupported_claims:
                features.add('has_unsupported_claim')
                if cls._claim_trim_safe(graph):
                    features.add('claim_trim_safe')
                else:
                    features.add('claim_trim_unsafe')
        return sorted(features)

    @staticmethod
    def _applied_actions(graph: StructuredMeaningGraph) -> list[str]:
        report = graph.operator_execution
        if report is None:
            return []
        return list(dict.fromkeys(item.split(':', 1)[1] for item in report.derived_decisions if item.startswith('repair_applied:')))

    @staticmethod
    def _applied_programs(graph: StructuredMeaningGraph) -> list[str]:
        report = graph.operator_execution
        if report is None:
            return []
        return list(dict.fromkeys(item.split(':', 1)[1] for item in report.derived_decisions if item.startswith('repair_program:')))

    @staticmethod
    def _rejected_actions(graph: StructuredMeaningGraph) -> list[str]:
        report = graph.operator_execution
        if report is None:
            return []
        return list(dict.fromkeys(item.split(':', 1)[1] for item in report.derived_decisions if item.startswith('repair_rejected:')))

    @classmethod
    def _benchmark_delta(cls, broken: StructuredMeaningGraph, repaired: StructuredMeaningGraph) -> float:
        before_report = broken.operator_execution
        after_report = repaired.operator_execution
        before_comp = float(before_report.composition_score) if before_report is not None else 0.0
        after_comp = float(after_report.composition_score) if after_report is not None else 0.0
        before_claim = float(before_report.claim_grounding_score) if before_report is not None else 0.0
        after_claim = float(after_report.claim_grounding_score) if after_report is not None else 0.0
        before_findings = len(before_report.compiler_findings) + len(before_report.counterexample_repairs) if before_report is not None else 0
        after_findings = len(after_report.compiler_findings) + len(after_report.counterexample_repairs) if after_report is not None else 0
        finding_delta = 0.0
        if before_findings > 0:
            finding_delta = (before_findings - after_findings) / float(before_findings)
        repair_signal = 1.0 if cls._applied_actions(repaired) else 0.0
        return round((0.4 * (after_comp - before_comp)) + (0.3 * (after_claim - before_claim)) + (0.2 * finding_delta) + (0.1 * repair_signal), 4)

    @classmethod
    def _trace_utility(cls, graph: StructuredMeaningGraph) -> float:
        report = graph.operator_execution
        if report is None:
            return 0.0
        applied_actions = cls._applied_actions(graph)
        repair_bonus = 0.15 if applied_actions else 0.0
        return round(
            max(0.0, float(report.composition_score) - 0.55) * 0.6
            + max(0.0, float(report.claim_grounding_score) - 0.5) * 0.25
            + repair_bonus,
            4,
        )

    @staticmethod
    def _claim_trim_safe(graph: StructuredMeaningGraph) -> bool:
        report = graph.operator_execution
        if report is None or not report.claim_groundings:
            return False
        unsupported = [item.claim.strip() for item in report.claim_groundings if not item.grounded and item.claim.strip()]
        grounded = [item.claim.strip() for item in report.claim_groundings if item.grounded and item.claim.strip()]
        if not unsupported:
            return False
        if grounded:
            return True
        for result in graph.symbolic_results:
            if result.domain != 'document_grounding' or not result.answer.strip():
                continue
            answer = result.answer
            for claim in unsupported:
                if claim and claim in answer:
                    answer = answer.replace(claim, ' ')
            answer = ' '.join(answer.split()).strip(' ,.;|-')
            if answer:
                return True
        return False


class RepairUtilityScorer:
    def __init__(self, model: RepairUtilityModel | None = None, model_path: str | Path | None = None) -> None:
        self.model = model or RepairUtilityModel.from_path(model_path) or RepairUtilityModel()

    def score_action(self, graph: StructuredMeaningGraph, action: str, findings: Sequence[str]) -> float:
        features = RepairUtilityTrainer._features(graph)
        score = float(self.model.action_bias.get(action, 0.0))
        for feature in features + _tokenize(' '.join(findings)):
            score += float(self.model.action_feature_weights.get(action, {}).get(feature, 0.0))
        return round(score, 4)

    def score_program(self, graph: StructuredMeaningGraph, program_id: str, findings: Sequence[str]) -> float:
        features = RepairUtilityTrainer._features(graph)
        score = float(self.model.program_bias.get(program_id, 0.0))
        for feature in features + _tokenize(' '.join(findings)):
            score += float(self.model.program_feature_weights.get(program_id, {}).get(feature, 0.0))
        return round(score, 4)

    def rank_actions(
        self,
        graph: StructuredMeaningGraph,
        findings: Sequence[str],
        actions: Sequence[str],
        action_to_program_ids: dict[str, list[str]] | None = None,
    ) -> list[str]:
        action_to_program_ids = action_to_program_ids or {}
        scored: list[tuple[str, float]] = []
        for action in actions:
            action_score = self.score_action(graph, action, findings)
            program_scores = [self.score_program(graph, program_id, findings) for program_id in action_to_program_ids.get(action, [])]
            combined = action_score + (max(program_scores) if program_scores else 0.0)
            scored.append((action, combined))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [action for action, _score in scored]

    def rejection_reason(
        self,
        graph: StructuredMeaningGraph,
        action: str,
        findings: Sequence[str],
        program_ids: Sequence[str] | None = None,
    ) -> str | None:
        action_score = self.score_action(graph, action, findings)
        features = set(RepairUtilityTrainer._features(graph))
        if action == 'trim_unsupported_claims' and 'claim_trim_unsafe' in features:
            return f'low_expected_utility({action_score:.2f})'
        if action_score < self.model.action_min_utility:
            return f'low_expected_utility({action_score:.2f})'
        program_scores = [self.score_program(graph, program_id, findings) for program_id in (program_ids or [])]
        if program_scores and max(program_scores) < self.model.program_min_utility:
            return f'low_program_utility({max(program_scores):.2f})'
        return None

def train_repair_utility_from_graphs(graphs: Sequence[StructuredMeaningGraph], output_path: str | Path) -> RepairUtilityTrainingSummary:
    return RepairUtilityTrainer().train_from_graphs(graphs, output_path)





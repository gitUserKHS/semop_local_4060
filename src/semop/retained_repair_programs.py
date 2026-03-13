from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .basis_operators import infer_basis_operators
from .operator_runtime import compile_and_execute
from .structures import StructuredMeaningGraph


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9a-z_]+", text.lower())


@dataclass
class RetainedRepairProgramRecord:
    program_id: str
    actions: list[str] = field(default_factory=list)
    trigger_terms: list[str] = field(default_factory=list)
    basis_signature: list[str] = field(default_factory=list)
    required_features: list[str] = field(default_factory=list)
    support: int = 0
    success_rate: float = 0.0
    average_score: float = 0.0
    utility_delta: float = 0.0
    sequence_length: int = 1
    rationale: str = ""

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetainedRepairProgramModel:
    records: list[RetainedRepairProgramRecord] = field(default_factory=list)
    min_match_score: float = 0.38
    trained_on_graphs: int = 0

    def model_dump(self) -> dict[str, Any]:
        return {
            "records": [item.model_dump() for item in self.records],
            "min_match_score": self.min_match_score,
            "trained_on_graphs": self.trained_on_graphs,
        }

    @classmethod
    def from_path(cls, path: str | Path | None) -> "RetainedRepairProgramModel | None":
        if not path:
            return None
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict) and "weights" in payload:
            payload = payload["weights"]
        return cls(
            records=[RetainedRepairProgramRecord(**item) for item in payload.get("records", [])],
            min_match_score=float(payload.get("min_match_score", 0.38)),
            trained_on_graphs=int(payload.get("trained_on_graphs", 0)),
        )


@dataclass
class RetainedRepairProgramTrainingSummary:
    output_path: str
    trained_on_graphs: int
    retained_program_count: int
    model: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class RetainedRepairProgramTrainer:
    def train_from_graphs(
        self,
        graphs: Sequence[StructuredMeaningGraph],
        output_path: str | Path,
        min_support: int = 1,
    ) -> RetainedRepairProgramTrainingSummary:
        from .operator_repair import OperatorRepairEngine

        graphs = list(graphs)
        engine = OperatorRepairEngine()
        grouped: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]] = {}
        for graph in graphs:
            self._harvest_repair_trace(graph, grouped)
            for broken in self._synthetic_breakages(graph):
                broken = compile_and_execute(broken)
                before_score = float(broken.operator_execution.composition_score) if broken.operator_execution is not None else 0.0
                findings = list(broken.operator_execution.compiler_findings if broken.operator_execution is not None else [])
                counterexample_repairs = list(broken.operator_execution.counterexample_repairs if broken.operator_execution is not None else [])
                repaired = engine.run(StructuredMeaningGraph.from_dict(broken.model_dump()))
                report = repaired.operator_execution
                if report is None:
                    continue
                actions = self._applied_actions(repaired)
                if not actions:
                    continue
                after_score = float(report.composition_score)
                utility_delta = self._utility_delta(broken, repaired)
                if after_score < before_score and utility_delta < 0.0:
                    continue
                flags = tuple(self._feature_flags(broken))
                key = (tuple(dict.fromkeys(actions)), flags)
                bucket = grouped.setdefault(
                    key,
                    {
                        "actions": list(dict.fromkeys(actions)),
                        "required_features": list(flags),
                        "trigger_counts": {},
                        "basis_counts": {},
                        "support": 0,
                        "success_sum": 0.0,
                        "score_sum": 0.0,
                        "utility_sum": 0.0,
                        "rationales": [],
                    },
                )
                for token in self._trigger_terms(findings, counterexample_repairs):
                    bucket["trigger_counts"][token] = int(bucket["trigger_counts"].get(token, 0)) + 1
                for basis in self._basis_signature(broken):
                    bucket["basis_counts"][basis] = int(bucket["basis_counts"].get(basis, 0)) + 1
                bucket["support"] += 1
                bucket["success_sum"] += 1.0 if after_score >= before_score else 0.0
                bucket["score_sum"] += after_score
                bucket["utility_sum"] += utility_delta
                rationale = " ".join(findings[:2] + counterexample_repairs[:1]).strip()
                if rationale:
                    bucket["rationales"].append(rationale)

        records: list[RetainedRepairProgramRecord] = []
        for index, bucket in enumerate(grouped.values(), start=1):
            support = int(bucket["support"])
            if support < min_support:
                continue
            trigger_terms = self._top_terms(bucket["trigger_counts"])
            basis_signature = self._top_terms(bucket["basis_counts"], limit=5)
            average_score = round(float(bucket["score_sum"]) / float(max(1, support)), 4)
            utility_delta = round(float(bucket["utility_sum"]) / float(max(1, support)), 4)
            success_rate = round(float(bucket["success_sum"]) / float(max(1, support)), 4)
            actions = list(bucket["actions"])
            program_name = actions[0] if actions else "repair_program"
            records.append(
                RetainedRepairProgramRecord(
                    program_id=f"{program_name}_{index}",
                    actions=actions,
                    trigger_terms=trigger_terms,
                    basis_signature=basis_signature,
                    required_features=list(bucket["required_features"]),
                    support=support,
                    success_rate=success_rate,
                    average_score=average_score,
                    utility_delta=utility_delta,
                    sequence_length=len(actions) or 1,
                    rationale=" ".join(dict.fromkeys(bucket["rationales"]))[:240] or "Retained repair program distilled from successful compiler-guided repairs.",
                )
            )

        records.sort(key=lambda item: (-item.utility_delta, -item.support, -item.average_score, item.program_id))
        model = RetainedRepairProgramModel(records=records, trained_on_graphs=len(graphs))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"weights": model.model_dump()}, ensure_ascii=False, indent=2), encoding="utf-8")
        return RetainedRepairProgramTrainingSummary(
            output_path=str(output),
            trained_on_graphs=len(graphs),
            retained_program_count=len(records),
            model=model.model_dump(),
        )

    @staticmethod
    def _applied_actions(graph: StructuredMeaningGraph) -> list[str]:
        report = graph.operator_execution
        if report is None:
            return []
        return list(dict.fromkeys(item.split(":", 1)[1] for item in report.derived_decisions if item.startswith("repair_applied:")))

    def _harvest_repair_trace(self, graph: StructuredMeaningGraph, grouped: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]]) -> None:
        report = graph.operator_execution
        if report is None:
            return
        actions = self._applied_actions(graph)
        if not actions:
            return
        flags = tuple(self._feature_flags(graph))
        key = (tuple(actions), flags)
        bucket = grouped.setdefault(
            key,
            {
                "actions": list(actions),
                "required_features": list(flags),
                "trigger_counts": {},
                "basis_counts": {},
                "support": 0,
                "success_sum": 0.0,
                "score_sum": 0.0,
                "utility_sum": 0.0,
                "rationales": [],
            },
        )
        for token in self._trigger_terms(list(report.compiler_findings), list(report.counterexample_repairs)):
            bucket["trigger_counts"][token] = int(bucket["trigger_counts"].get(token, 0)) + 1
        for basis in self._basis_signature(graph):
            bucket["basis_counts"][basis] = int(bucket["basis_counts"].get(basis, 0)) + 1
        bucket["support"] += 1
        bucket["success_sum"] += 1.0
        bucket["score_sum"] += float(report.composition_score)
        bucket["utility_sum"] += self._trace_utility(graph)
        rationale = " ".join(list(report.compiler_findings[:2]) + list(report.counterexample_repairs[:1]) + list(graph.audit_trace[:1])).strip()
        if rationale:
            bucket["rationales"].append(rationale)

    def _synthetic_breakages(self, graph: StructuredMeaningGraph) -> list[StructuredMeaningGraph]:
        broken_graphs: list[StructuredMeaningGraph] = []
        if graph.hidden_goals and graph.required_premises:
            broken = StructuredMeaningGraph.from_dict(graph.model_dump())
            broken.operator_decompositions = []
            broken_graphs.append(broken)
        if graph.required_premises or graph.satisfied_premises or graph.missing_premises:
            broken = StructuredMeaningGraph.from_dict(graph.model_dump())
            broken.edges = [edge for edge in broken.edges if edge.relation != "REQUIRES"]
            broken_graphs.append(broken)
        if graph.source_context.strip():
            broken = StructuredMeaningGraph.from_dict(graph.model_dump())
            broken.nodes = [node for node in broken.nodes if node.id not in {"source_document"} and node.kind != "evidence"]
            broken.edges = [edge for edge in broken.edges if edge.relation not in {"USES_CONTEXT", "HAS_EVIDENCE", "GROUNDED_BY"}]
            broken_graphs.append(broken)
        if any(result.domain == "document_grounding" and result.answer.strip() for result in graph.symbolic_results):
            broken = StructuredMeaningGraph.from_dict(graph.model_dump())
            for result in broken.symbolic_results:
                if result.domain != "document_grounding" or not result.answer.strip():
                    continue
                if " and " not in result.answer:
                    result.answer = result.answer.rstrip(". ") + " and inspect the hidden sensor."
            broken_graphs.append(broken)
        if any(any(tag.startswith("multimodal:vision") for tag in node.provenance) for node in graph.nodes):
            broken = StructuredMeaningGraph.from_dict(graph.model_dump())
            broken.nodes = [node for node in broken.nodes if node.id != "visual_scene"]
            broken.edges = [edge for edge in broken.edges if not (edge.source == "question" and edge.relation == "CONDITIONS_ON" and edge.target == "visual_scene")]
            broken_graphs.append(broken)
        if graph.functor_hypotheses:
            broken = StructuredMeaningGraph.from_dict(graph.model_dump())
            for functor in broken.functor_hypotheses:
                functor.object_map = {}
            broken_graphs.append(broken)
        return broken_graphs

    @staticmethod
    def _feature_flags(graph: StructuredMeaningGraph) -> list[str]:
        flags: list[str] = []
        if graph.hidden_goals:
            flags.append("has_hidden_goal")
        if graph.required_premises or graph.satisfied_premises or graph.missing_premises:
            flags.append("has_premise_state")
        if graph.source_context.strip():
            flags.append("has_document_context")
        if any(any(tag.startswith("multimodal:vision") for tag in node.provenance) for node in graph.nodes):
            flags.append("has_visual_signal")
        if graph.functor_hypotheses:
            flags.append("has_functor")
        if graph.operator_execution is not None and graph.operator_execution.claim_groundings:
            flags.append("has_claim_grounding")
            if any(item.grounded for item in graph.operator_execution.claim_groundings):
                flags.append("has_grounded_claim")
            if any(not item.grounded for item in graph.operator_execution.claim_groundings):
                flags.append("has_unsupported_claim")
        return flags

    @staticmethod
    def _trigger_terms(findings: list[str], counterexample_repairs: list[str]) -> list[str]:
        raw = " ".join(findings + counterexample_repairs)
        tokens = [token for token in _tokenize(raw) if len(token) > 2 and token not in {"compiler", "warning", "repair"}]
        return list(dict.fromkeys(tokens))[:10]

    @staticmethod
    def _basis_signature(graph: StructuredMeaningGraph) -> list[str]:
        basis = set(infer_basis_operators(graph))
        if graph.source_context.strip():
            basis.add("DOCUMENT_CONTEXT")
        if any(any(tag.startswith("multimodal:vision") for tag in node.provenance) for node in graph.nodes):
            basis.add("VISUAL_STRUCTURE")
        return sorted(basis)

    @staticmethod
    def _top_terms(counts: dict[str, int], limit: int = 8) -> list[str]:
        items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [item[0] for item in items[:limit]]

    @classmethod
    def _utility_delta(cls, broken: StructuredMeaningGraph, repaired: StructuredMeaningGraph) -> float:
        before_report = broken.operator_execution
        after_report = repaired.operator_execution
        before_comp = float(before_report.composition_score) if before_report is not None else 0.0
        after_comp = float(after_report.composition_score) if after_report is not None else 0.0
        before_claim = float(before_report.claim_grounding_score) if before_report is not None else 0.0
        after_claim = float(after_report.claim_grounding_score) if after_report is not None else 0.0
        return round((0.6 * (after_comp - before_comp)) + (0.4 * (after_claim - before_claim)), 4)

    @classmethod
    def _trace_utility(cls, graph: StructuredMeaningGraph) -> float:
        report = graph.operator_execution
        if report is None:
            return 0.0
        return round((0.7 * float(report.composition_score)) + (0.3 * float(report.claim_grounding_score)), 4)


class RetainedRepairProgramLibrary:
    def __init__(self, model: RetainedRepairProgramModel | None = None, model_path: str | Path | None = None) -> None:
        self.model = model or RetainedRepairProgramModel.from_path(model_path) or RetainedRepairProgramModel()

    def matched_programs(
        self,
        graph: StructuredMeaningGraph,
        findings: list[str],
        counterexample_repairs: list[str] | None = None,
    ) -> list[tuple[RetainedRepairProgramRecord, float]]:
        counterexample_repairs = counterexample_repairs or []
        flags = set(RetainedRepairProgramTrainer._feature_flags(graph))
        trigger_terms = set(RetainedRepairProgramTrainer._trigger_terms(findings, counterexample_repairs))
        basis = set(RetainedRepairProgramTrainer._basis_signature(graph))
        scored: list[tuple[RetainedRepairProgramRecord, float]] = []
        for record in self.model.records:
            trigger_overlap = len(trigger_terms & set(record.trigger_terms)) / float(len(record.trigger_terms) or 1)
            flag_overlap = len(flags & set(record.required_features)) / float(len(record.required_features) or 1)
            basis_overlap = len(basis & set(record.basis_signature)) / float(len(record.basis_signature) or 1)
            utility_bonus = max(0.0, float(record.utility_delta))
            sequence_bonus = min(1.0, float(record.sequence_length) / 3.0)
            score = (0.35 * trigger_overlap) + (0.2 * flag_overlap) + (0.15 * basis_overlap) + (0.1 * record.success_rate) + (0.1 * utility_bonus) + (0.1 * sequence_bonus)
            if score >= self.model.min_match_score:
                scored.append((record, round(score, 4)))
        scored.sort(key=lambda item: (-item[1], -item[0].utility_delta, -item[0].support, item[0].program_id))
        return scored

    def synthesize_actions(
        self,
        graph: StructuredMeaningGraph,
        findings: list[str],
        counterexample_repairs: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        counterexample_repairs = counterexample_repairs or []
        actions: list[str] = []
        program_ids: list[str] = []
        for record, _score in self.matched_programs(graph, findings, counterexample_repairs):
            program_ids.append(record.program_id)
            for action in record.actions:
                if action not in actions:
                    actions.append(action)
        synthesized = self._actions_from_counterexamples(" ".join(findings + counterexample_repairs).lower())
        for action in synthesized:
            if action not in actions:
                actions.append(action)
        return actions, program_ids

    @staticmethod
    def _actions_from_counterexamples(text: str) -> list[str]:
        actions: list[str] = []
        if "goal" in text or "hidden" in text or "decomposition" in text:
            actions.append("add_goal_preservation_decomposition")
        if "requires" in text or "prerequisite" in text or "input types unsupported" in text:
            actions.append("bind_requires_edges")
        if "document" in text or "evidence" in text or "pdf" in text:
            actions.append("attach_document_context_nodes")
        if "visual" in text or "scene" in text:
            actions.append("attach_visual_scene")
        if "functor" in text or "object map" in text:
            actions.append("rebind_functor_object_map")
        if "claim" in text or "unsupported" in text or "sensor" in text:
            actions.append("trim_unsupported_claims")
        return actions


def train_retained_repair_programs_from_graphs(
    graphs: Sequence[StructuredMeaningGraph],
    output_path: str | Path,
) -> RetainedRepairProgramTrainingSummary:
    return RetainedRepairProgramTrainer().train_from_graphs(graphs, output_path)

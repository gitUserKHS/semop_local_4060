from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .generalization_proof import GeneralizationProofHarness
from .runtime_ops import build_runtime_doctor_report
from .visual_geometry_3d import VisualGeometry3DWorkbench
from .world_model_math_service import ProductionMathServiceConfig, WorldModelMathProductionService
from .world_model_math_training import WorldModelMathTrainer, ensure_starter_math_cases


@dataclass
class CapabilityAxisStatus:
    name: str
    score: float
    status: str
    evidence: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilityAuditSummary:
    workspace: str
    report_path: str
    runtime_doctor: dict[str, Any]
    benchmark_gate: dict[str, Any] = field(default_factory=dict)
    generalization_proof: dict[str, Any] = field(default_factory=dict)
    math_self_test: dict[str, Any] = field(default_factory=dict)
    math_training: dict[str, Any] = field(default_factory=dict)
    visual_reconstruction: dict[str, Any] = field(default_factory=dict)
    generated_artifacts: list[str] = field(default_factory=list)
    axes: list[CapabilityAxisStatus] = field(default_factory=list)
    overall_readiness_percent: float = 0.0
    overall_status: str = 'unknown'
    priority_improvements: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            'workspace': self.workspace,
            'report_path': self.report_path,
            'runtime_doctor': dict(self.runtime_doctor),
            'benchmark_gate': dict(self.benchmark_gate),
            'generalization_proof': dict(self.generalization_proof),
            'math_self_test': dict(self.math_self_test),
            'math_training': dict(self.math_training),
            'visual_reconstruction': dict(self.visual_reconstruction),
            'generated_artifacts': list(self.generated_artifacts),
            'axes': [item.model_dump() for item in self.axes],
            'overall_readiness_percent': self.overall_readiness_percent,
            'overall_status': self.overall_status,
            'priority_improvements': list(self.priority_improvements),
        }


class CapabilityAuditRunner:
    def __init__(self, mode: str = 'heuristic') -> None:
        self.mode = mode

    @staticmethod
    def _read_json(path: str | Path) -> dict[str, Any]:
        target = Path(path)
        if not target.exists():
            return {}
        try:
            payload = json.loads(target.read_text(encoding='utf-8'))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, round(float(value), 4)))

    @staticmethod
    def _status_from_score(score: float) -> str:
        if score >= 0.85:
            return 'strong'
        if score >= 0.65:
            return 'usable'
        if score >= 0.45:
            return 'partial'
        return 'weak'

    def run(
        self,
        *,
        workspace: str = '.',
        bootstrap_missing: bool = False,
        output_path: str | None = None,
        unified_output_dir: str = 'data/unified_semop_gui_run',
        unified_store_path: str = 'data/semop_memory.db',
        unified_review_queue_path: str = 'data/ops_review_queue.db',
        unified_benchmark_corpus_path: str = 'data/unified_semop_gui_run/persistent_benchmark_corpus.json',
        math_output_dir: str = 'data/math_world_model_gui_run',
        math_cases_path: str = 'examples/math_world_model_starter.jsonl',
        visual_output_dir: str = 'data/math_world_model_gui_run/visual_3d',
    ) -> CapabilityAuditSummary:
        workspace_path = str(Path(workspace).resolve())
        runtime_doctor = build_runtime_doctor_report(workspace=workspace_path).model_dump()
        generated_artifacts: list[str] = []

        benchmark_gate_path = Path(unified_output_dir) / 'benchmark_gate.json'
        benchmark_gate = self._read_json(benchmark_gate_path)

        proof_report_path = Path(unified_output_dir) / 'generalization_proof' / 'generalization_proof_report.json'
        generalization_proof = self._read_json(proof_report_path)
        if bootstrap_missing and not generalization_proof and Path(unified_store_path).exists():
            proof = GeneralizationProofHarness(mode=self.mode).run(
                unified_store_path,
                unified_output_dir,
                rounds=1,
                review_store_path=unified_review_queue_path if Path(unified_review_queue_path).exists() else None,
                benchmark_corpus_path=unified_benchmark_corpus_path if Path(unified_benchmark_corpus_path).exists() else None,
                approved_queries_only=False,
            )
            generalization_proof = proof.model_dump()
            generated_artifacts.append(str(proof.report_path))

        math_training_path = Path(math_output_dir) / 'math_training_summary.json'
        math_training = self._read_json(math_training_path)
        if bootstrap_missing and not math_training:
            cases_path = ensure_starter_math_cases(math_cases_path)
            trainer = WorldModelMathTrainer(mode=self.mode)
            training = trainer.train_from_cases(cases_path, math_output_dir, epochs=1, bootstrap_starter=False)
            math_training = training.model_dump()
            generated_artifacts.append(str(math_training_path))

        math_service = WorldModelMathProductionService(
            ProductionMathServiceConfig(
                mode=self.mode,
                concept_store_path='data/vlso_visual_prototypes.db',
                operator_store_path='data/vlso_visual_operators.db',
                affordance_weights_path='data/vlso_samples/trained_affordance_weights.json',
                logical_weight_path=str(Path(math_output_dir) / 'logical_pattern_weights.json'),
                strategy_memory_path=str(Path(math_output_dir) / 'math_strategy_memory.json'),
                leworldmodel_path=str(Path(math_output_dir) / 'math_leworldmodel_prior.json'),
                audit_log_path=str(Path(math_output_dir) / 'audit_log.jsonl'),
            )
        )
        math_self_test = math_service.self_test().model_dump()

        visual_recon_path = Path(visual_output_dir) / 'scene_3d_reconstruction.json'
        visual_reconstruction = self._read_json(visual_recon_path)
        if bootstrap_missing and not visual_reconstruction:
            workbench = VisualGeometry3DWorkbench()
            if not (Path(visual_output_dir) / 'visual_geometry_concepts.db').exists():
                workbench.train_starter_bundle(visual_output_dir, limit_scenes=1)
            reconstruction = workbench.reconstruct_scene(
                'Reconstruct the visible geometry and topology into 3D.',
                'examples/vlso/geometry_scene.json',
                visual_output_dir,
            )
            visual_reconstruction = reconstruction.model_dump()
            generated_artifacts.append(str(visual_recon_path))

        axes = self._build_axes(runtime_doctor, benchmark_gate, generalization_proof, math_self_test, math_training, visual_reconstruction)
        readiness = round(sum(item.score for item in axes) / float(len(axes) or 1) * 100.0, 1)
        overall_status = 'strong' if readiness >= 85 else 'usable' if readiness >= 65 else 'partial' if readiness >= 45 else 'weak'
        priority_improvements = self._priority_improvements(axes, benchmark_gate, generalization_proof, math_self_test)

        report_target = Path(output_path or (Path(unified_output_dir) / 'capability_audit_report.json'))
        report_target.parent.mkdir(parents=True, exist_ok=True)
        summary = CapabilityAuditSummary(
            workspace=workspace_path,
            report_path=str(report_target),
            runtime_doctor=runtime_doctor,
            benchmark_gate=benchmark_gate,
            generalization_proof=generalization_proof,
            math_self_test=math_self_test,
            math_training=math_training,
            visual_reconstruction=visual_reconstruction,
            generated_artifacts=generated_artifacts,
            axes=axes,
            overall_readiness_percent=readiness,
            overall_status=overall_status,
            priority_improvements=priority_improvements,
        )
        report_target.write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

    def _build_axes(
        self,
        runtime_doctor: dict[str, Any],
        benchmark_gate: dict[str, Any],
        generalization_proof: dict[str, Any],
        math_self_test: dict[str, Any],
        math_training: dict[str, Any],
        visual_reconstruction: dict[str, Any],
    ) -> list[CapabilityAxisStatus]:
        axes: list[CapabilityAxisStatus] = []
        dependency = runtime_doctor.get('dependency_status', {}) if isinstance(runtime_doctor.get('dependency_status'), dict) else {}
        surfaces = runtime_doctor.get('surfaces', []) if isinstance(runtime_doctor.get('surfaces'), list) else []
        runtime_score = 0.4 * (1.0 if dependency.get('llm_ready') else 0.0) + 0.3 * (1.0 if dependency.get('training_ready') else 0.0) + 0.3 * (min(1.0, len([item for item in surfaces if isinstance(item, dict) and item.get('ready')]) / float(max(1, len(surfaces)))))
        axes.append(CapabilityAxisStatus(
            name='local_runtime',
            score=self._clip(runtime_score),
            status=self._status_from_score(runtime_score),
            evidence=[
                'hardware profile: ' + str((runtime_doctor.get('hardware_profile') or {}).get('detected_profile', 'unknown')),
                'llm_ready=' + str(dependency.get('llm_ready', False)),
                'training_ready=' + str(dependency.get('training_ready', False)),
                'qlora_ready=' + str(dependency.get('qlora_ready', False)),
            ],
            recommendations=[str(item) for item in runtime_doctor.get('recommended_actions', [])[:4]],
        ))

        gate = benchmark_gate.get('gate', {}) if isinstance(benchmark_gate.get('gate'), dict) else benchmark_gate.get('decision', {}) if isinstance(benchmark_gate.get('decision'), dict) else {}
        benchmark = benchmark_gate.get('benchmark', {}) if isinstance(benchmark_gate.get('benchmark'), dict) else {}
        candidate_metrics = gate.get('candidate_metrics', {}) if isinstance(gate.get('candidate_metrics'), dict) else {}
        reasoning_terms = [
            float(candidate_metrics.get('unseen_transfer', benchmark.get('unseen_transfer', 0.0)) or 0.0),
            float(candidate_metrics.get('analogy_usefulness', benchmark.get('analogy_usefulness', 0.0)) or 0.0),
            float(candidate_metrics.get('compiler_validity', benchmark.get('compiler_validity', 0.0)) or 0.0),
            float(candidate_metrics.get('grounded_explanation_fidelity', benchmark.get('grounded_explanation_fidelity', 0.0)) or 0.0),
            float(candidate_metrics.get('repair_success_rate', benchmark.get('repair_success_rate', 0.0)) or 0.0),
        ]
        reasoning_score = sum(reasoning_terms) / float(len(reasoning_terms) or 1)
        axes.append(CapabilityAxisStatus(
            name='semop_reasoning',
            score=self._clip(reasoning_score),
            status=self._status_from_score(reasoning_score),
            evidence=[
                'gate accepted=' + str(gate.get('accepted', False)),
                'unseen_transfer=' + str(round(reasoning_terms[0], 4)),
                'analogy_usefulness=' + str(round(reasoning_terms[1], 4)),
                'compiler_validity=' + str(round(reasoning_terms[2], 4)),
                'grounded_fidelity=' + str(round(reasoning_terms[3], 4)),
                'repair_success=' + str(round(reasoning_terms[4], 4)),
            ],
            recommendations=[str(item) for item in gate.get('blocking_reasons', [])[:4]],
        ))

        goal_tracker = generalization_proof.get('goal_tracker', {}) if isinstance(generalization_proof.get('goal_tracker'), dict) else {}
        proof_score = float(goal_tracker.get('readiness_percent', 0.0) or 0.0) / 100.0
        axes.append(CapabilityAxisStatus(
            name='generalization_proof',
            score=self._clip(proof_score),
            status=self._status_from_score(proof_score),
            evidence=[
                'readiness_percent=' + str(goal_tracker.get('readiness_percent', 0.0)),
                'priority_focus=' + str(goal_tracker.get('priority_focus', '-')),
                'domains_seen=' + ', '.join(str(item) for item in goal_tracker.get('domains_seen', [])[:4]),
            ],
            recommendations=[str(item) for item in goal_tracker.get('remaining_items', [])[:4]],
        ))

        self_test_pass = float(math_self_test.get('passed_count', 0) or 0) / float(max(1, int(math_self_test.get('total_count', 0) or 0)))
        training_score = float(math_training.get('final_average_score', 0.0) or 0.0)
        math_score = 0.6 * self_test_pass + 0.4 * training_score
        axes.append(CapabilityAxisStatus(
            name='math_world_model',
            score=self._clip(math_score),
            status=self._status_from_score(math_score),
            evidence=[
                'self_test=' + f"{math_self_test.get('passed_count', 0)}/{math_self_test.get('total_count', 0)}",
                'training_final_average=' + str(round(training_score, 4)),
                'accepted_cases=' + str(math_training.get('accepted_cases', 0)),
                'exact_match_cases=' + str(math_training.get('exact_match_cases', 0)),
            ],
            recommendations=[str(item) for item in ((math_self_test.get('readiness') or {}).get('recommended_actions', []) if isinstance(math_self_test.get('readiness'), dict) else [])[:4]],
        ))

        primitive_count = len(visual_reconstruction.get('primitives', []) or []) if isinstance(visual_reconstruction.get('primitives'), list) else 0
        has_obj = bool(((visual_reconstruction.get('export_paths') or {}).get('obj_path')) if isinstance(visual_reconstruction.get('export_paths'), dict) else False)
        warning_penalty = min(0.4, 0.1 * len(visual_reconstruction.get('warnings', []) or []))
        visual_score = min(1.0, (0.5 if primitive_count > 0 else 0.0) + (0.3 if has_obj else 0.0) + (0.2 if visual_reconstruction else 0.0)) - warning_penalty
        axes.append(CapabilityAxisStatus(
            name='visual_3d_reconstruction',
            score=self._clip(visual_score),
            status=self._status_from_score(visual_score),
            evidence=[
                'primitive_count=' + str(primitive_count),
                'has_obj=' + str(has_obj),
                'warnings=' + str(len(visual_reconstruction.get('warnings', []) or [])),
            ],
            recommendations=[str(item) for item in (visual_reconstruction.get('warnings', []) or [])[:4]],
        ))
        return axes

    @staticmethod
    def _priority_improvements(
        axes: list[CapabilityAxisStatus],
        benchmark_gate: dict[str, Any],
        generalization_proof: dict[str, Any],
        math_self_test: dict[str, Any],
    ) -> list[str]:
        ordered = sorted(axes, key=lambda item: item.score)
        recommendations: list[str] = []
        for axis in ordered[:3]:
            if axis.recommendations:
                recommendations.extend(axis.recommendations[:2])
        gate = benchmark_gate.get('gate', {}) if isinstance(benchmark_gate.get('gate'), dict) else {}
        recommendations.extend(str(item) for item in gate.get('blocking_reasons', [])[:2])
        tracker = generalization_proof.get('goal_tracker', {}) if isinstance(generalization_proof.get('goal_tracker'), dict) else {}
        recommendations.extend(str(item) for item in tracker.get('remaining_items', [])[:2])
        readiness = math_self_test.get('readiness', {}) if isinstance(math_self_test.get('readiness'), dict) else {}
        recommendations.extend(str(item) for item in readiness.get('recommended_actions', [])[:2])
        deduped: list[str] = []
        for item in recommendations:
            if item and item not in deduped:
                deduped.append(item)
        return deduped[:8]

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .capability_audit import CapabilityAuditRunner
from .concept_fusion import ConceptFusionEngine, ConceptFusionSummary


@dataclass
class UltimateAGIAxis:
    key: str
    label: str
    score: float
    target: float
    status: str
    summary: str
    blockers: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UltimateAGIReadinessSummary:
    workspace: str
    report_path: str
    overall_readiness_percent: float
    commercial_status: str
    headline: str
    axes: list[UltimateAGIAxis] = field(default_factory=list)
    completed_axes: list[str] = field(default_factory=list)
    remaining_axes: list[str] = field(default_factory=list)
    priority_focus: str = ''
    product_blockers: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    source_reports: dict[str, str] = field(default_factory=dict)
    concept_fusion_preview: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {
            'workspace': self.workspace,
            'report_path': self.report_path,
            'overall_readiness_percent': self.overall_readiness_percent,
            'commercial_status': self.commercial_status,
            'headline': self.headline,
            'axes': [item.model_dump() for item in self.axes],
            'completed_axes': list(self.completed_axes),
            'remaining_axes': list(self.remaining_axes),
            'priority_focus': self.priority_focus,
            'product_blockers': list(self.product_blockers),
            'next_steps': list(self.next_steps),
            'source_reports': dict(self.source_reports),
            'concept_fusion_preview': dict(self.concept_fusion_preview),
        }


class UltimateAGIReadinessRunner:
    def __init__(self, mode: str = 'heuristic') -> None:
        self.mode = mode
        self._fusion = ConceptFusionEngine()

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, round(float(value), 4)))

    @staticmethod
    def _status(score: float, target: float) -> str:
        if score >= target:
            return 'ready'
        if score >= max(0.45, target - 0.15):
            return 'advancing'
        return 'early'

    @staticmethod
    def _avg(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / float(len(values))

    def run(
        self,
        *,
        workspace: str = '.',
        output_path: str | None = None,
        bootstrap_missing: bool = True,
        unified_output_dir: str = 'data/unified_semop_gui_run',
        unified_store_path: str = 'data/semop_memory.db',
        unified_review_queue_path: str = 'data/ops_review_queue.db',
        unified_benchmark_corpus_path: str = 'data/unified_semop_gui_run/persistent_benchmark_corpus.json',
        math_output_dir: str = 'data/math_world_model_gui_run',
        math_cases_path: str = 'examples/math_world_model_starter.jsonl',
        visual_output_dir: str = 'data/math_world_model_gui_run/visual_3d',
    ) -> UltimateAGIReadinessSummary:
        workspace_path = str(Path(workspace).resolve())
        audit = CapabilityAuditRunner(mode=self.mode).run(
            workspace=workspace_path,
            bootstrap_missing=bootstrap_missing,
            unified_output_dir=unified_output_dir,
            unified_store_path=unified_store_path,
            unified_review_queue_path=unified_review_queue_path,
            unified_benchmark_corpus_path=unified_benchmark_corpus_path,
            math_output_dir=math_output_dir,
            math_cases_path=math_cases_path,
            visual_output_dir=visual_output_dir,
        )
        audit_payload = audit.model_dump()
        gate = audit_payload.get('benchmark_gate', {}) if isinstance(audit_payload.get('benchmark_gate'), dict) else {}
        gate_metrics = gate.get('benchmark', {}) if isinstance(gate.get('benchmark'), dict) else gate.get('candidate_metrics', {}) if isinstance(gate.get('candidate_metrics'), dict) else {}
        proof = audit_payload.get('generalization_proof', {}) if isinstance(audit_payload.get('generalization_proof'), dict) else {}
        evidence = proof.get('evidence', {}) if isinstance(proof.get('evidence'), dict) else {}
        goal = proof.get('goal_tracker', {}) if isinstance(proof.get('goal_tracker'), dict) else {}
        math_self_test = audit_payload.get('math_self_test', {}) if isinstance(audit_payload.get('math_self_test'), dict) else {}
        math_training = audit_payload.get('math_training', {}) if isinstance(audit_payload.get('math_training'), dict) else {}
        visual = audit_payload.get('visual_reconstruction', {}) if isinstance(audit_payload.get('visual_reconstruction'), dict) else {}
        runtime = audit_payload.get('runtime_doctor', {}) if isinstance(audit_payload.get('runtime_doctor'), dict) else {}
        runtime_dep = runtime.get('dependency_status', {}) if isinstance(runtime.get('dependency_status'), dict) else {}

        math_self_score = float(math_self_test.get('passed_count', 0) or 0) / float(max(1, int(math_self_test.get('total_count', 0) or 0)))
        math_train_score = float(math_training.get('final_average_score', 0.0) or 0.0)
        visual_grounding_score = self._avg([
            float(gate_metrics.get('grounded_explanation_fidelity', 0.0) or 0.0),
            min(1.0, len(visual.get('primitives', []) or []) / 5.0),
            min(1.0, len(visual.get('relations', []) or []) / 4.0),
        ])
        embodied_score = self._avg([
            float(gate_metrics.get('compiler_validity', 0.0) or 0.0),
            float(evidence.get('multimodal_transfer_score', 0.0) or 0.0),
            visual_grounding_score,
        ])
        creativity_score = self._avg([
            float(gate_metrics.get('analogy_usefulness', 0.0) or 0.0),
            float(evidence.get('domain_coverage_score', 0.0) or 0.0),
            float(evidence.get('learned_generalization_score', 0.0) or 0.0),
        ])
        self_improvement_score = self._avg([
            float(evidence.get('reviewed_corpus_growth_score', 0.0) or 0.0),
            float(evidence.get('strong_model_score', 0.0) or 0.0),
            float(goal.get('readiness_percent', 0.0) or 0.0) / 100.0,
        ])
        commercial_score = self._avg([
            audit.overall_readiness_percent / 100.0,
            1.0 if runtime_dep.get('llm_ready') else 0.0,
            1.0 if runtime_dep.get('training_ready') else 0.0,
            float(gate_metrics.get('grounded_explanation_fidelity', 0.0) or 0.0),
        ])

        axes = [
            self._axis(
                key='contextual_reasoning',
                label='Prompt understanding and logical reasoning',
                score=float(next((item.score for item in audit.axes if item.name == 'semop_reasoning'), 0.0)),
                target=0.78,
                summary='Hidden context, intent recovery, logical response quality, and operator-grounded reasoning.',
                blockers=self._blockers_from_gate(gate),
                next_steps=['Keep improving grounded fidelity and approved review coverage for general-domain prompts.'],
            ),
            self._axis(
                key='multimodal_understanding',
                label='Image and video multimodal understanding',
                score=visual_grounding_score,
                target=0.72,
                summary='Grounded image/video understanding that can describe situations and support planning.',
                blockers=['Grounded multimodal explanations are still weaker than compiler validity.'] if visual_grounding_score < 0.72 else [],
                next_steps=['Collect more approved image/video traces and keep real-scene grounding cases in the loop.'],
            ),
            self._axis(
                key='mathematical_reasoning',
                label='Mathematical problem solving and proof',
                score=self._avg([math_self_score, math_train_score]),
                target=0.72,
                summary='Math world-model solving, proof guidance, and verification-backed answer quality.',
                blockers=['Math self-test and training scores still need broader difficult cases.'] if self._avg([math_self_score, math_train_score]) < 0.72 else [],
                next_steps=['Grow harder reviewed proof cases and keep diagram-grounded math in the training loop.'],
            ),
            self._axis(
                key='embodied_autonomy',
                label='Embodied autonomy readiness for driving and robotics',
                score=embodied_score,
                target=0.68,
                summary='World-model perception, action-safe planning, and compiler-checked multimodal control readiness.',
                blockers=['Embodied planning is still missing stronger grounded scene-to-action supervision.'] if embodied_score < 0.68 else [],
                next_steps=['Expand visual-control traces, safety reviews, and action-affordance verification.'],
            ),
            self._axis(
                key='creativity_concept_fusion',
                label='Creativity and concept fusion',
                score=creativity_score,
                target=0.62,
                summary='Novel but auditable concept fusion across domains, world models, memory, and operators.',
                blockers=['Creative recombination is not strong enough until fusion ideas improve verification or transfer.'] if creativity_score < 0.62 else [],
                next_steps=['Keep only fused priors that improve planning, transfer, or grounded verification.'],
            ),
            self._axis(
                key='self_improvement',
                label='Self-improvement and reviewed-corpus growth',
                score=self_improvement_score,
                target=0.64,
                summary='Reviewed-corpus growth, stronger proof loops, and repeated improvement without drifting quality.',
                blockers=['The stronger proof loop is not yet high enough to claim durable generalization.'] if self_improvement_score < 0.64 else [],
                next_steps=['Keep benchmark-gated growth loops and reject review growth that does not improve proof scores.'],
            ),
            self._axis(
                key='commercialization',
                label='Commercialization readiness',
                score=commercial_score,
                target=0.7,
                summary='Deployable runtime, beginner operation, grounded safety, and product-style auditability.',
                blockers=['Commercialization is blocked until grounding fidelity and proof readiness rise together.'] if commercial_score < 0.7 else [],
                next_steps=['Keep one-click operation, safety gates, and production-style reports aligned with stronger grounding metrics.'],
            ),
        ]

        overall = round(self._avg([item.score for item in axes]) * 100.0, 1)
        completed = [item.label for item in axes if item.status == 'ready']
        remaining = [item.label for item in axes if item.status != 'ready']
        if overall >= 85.0:
            commercial_status = 'commercializing'
        elif overall >= 70.0:
            commercial_status = 'preproduct'
        elif overall >= 55.0:
            commercial_status = 'prototype'
        else:
            commercial_status = 'research'
        if commercial_status == 'commercializing':
            headline = 'The stack is nearing a commercializable multi-domain reasoning product, but the remaining weak axes still need disciplined verification.'
        elif commercial_status == 'preproduct':
            headline = 'The stack is beyond pure research and is moving into pre-product readiness, but it is not yet safe to call it a broad AGI product.'
        elif commercial_status == 'prototype':
            headline = 'The stack is a serious prototype with multiple AGI-aligned subsystems, but broad deployment is still premature.'
        else:
            headline = 'The stack is still in the research stage for the full AGI ambition.'

        blockers = []
        for axis in axes:
            blockers.extend(axis.blockers[:1])
        blockers = blockers[:6]
        next_steps = []
        for axis in axes:
            next_steps.extend(axis.next_steps[:1])
        next_steps = list(dict.fromkeys(next_steps))[:6]
        priority_focus = remaining[0] if remaining else 'Keep raising the same axes without losing verification quality.'
        fusion_preview = self._fusion.build_summary(
            'Build a commercializable AGI that can reason, see, control robots, drive, solve math, and invent new ideas.',
            'ops',
            {
                'likely_domain': 'agi_productization',
                'likely_scenario': 'cross_domain_world_model',
                'hidden_constraints': blockers[:3],
            },
            {'notes': next_steps[:3]},
        )

        report_target = Path(output_path or (Path(unified_output_dir) / 'ultimate_agi_readiness.json'))
        report_target.parent.mkdir(parents=True, exist_ok=True)
        summary = UltimateAGIReadinessSummary(
            workspace=workspace_path,
            report_path=str(report_target),
            overall_readiness_percent=overall,
            commercial_status=commercial_status,
            headline=headline,
            axes=axes,
            completed_axes=completed,
            remaining_axes=remaining,
            priority_focus=priority_focus,
            product_blockers=blockers,
            next_steps=next_steps,
            source_reports={
                'capability_audit': audit.report_path,
                'generalization_proof': str(Path(unified_output_dir) / 'generalization_proof' / 'generalization_proof_report.json'),
                'benchmark_gate': str(Path(unified_output_dir) / 'benchmark_gate.json'),
            },
            concept_fusion_preview=fusion_preview.model_dump(),
        )
        report_target.write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

    def _axis(
        self,
        *,
        key: str,
        label: str,
        score: float,
        target: float,
        summary: str,
        blockers: list[str],
        next_steps: list[str],
    ) -> UltimateAGIAxis:
        clipped = self._clip(score)
        return UltimateAGIAxis(
            key=key,
            label=label,
            score=clipped,
            target=target,
            status=self._status(clipped, target),
            summary=summary,
            blockers=list(dict.fromkeys(str(item) for item in blockers if str(item).strip())),
            next_steps=list(dict.fromkeys(str(item) for item in next_steps if str(item).strip())),
        )

    @staticmethod
    def _blockers_from_gate(gate: dict[str, Any]) -> list[str]:
        gate_info = gate.get('gate', {}) if isinstance(gate.get('gate'), dict) else gate.get('decision', {}) if isinstance(gate.get('decision'), dict) else {}
        blockers = gate_info.get('blocking_reasons', []) if isinstance(gate_info.get('blocking_reasons'), list) else []
        return [str(item) for item in blockers[:4]]

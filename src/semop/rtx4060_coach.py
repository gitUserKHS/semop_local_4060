from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .capability_audit import CapabilityAuditRunner, CapabilityAuditSummary
from .capability_coach import CapabilityImprovementRunner, CapabilityImprovementSummary
from .review_queue import ReviewQueueStore


@dataclass
class RTX4060CollectionLane:
    label: str
    current_count: int
    suggested_next_batch: int
    preferred_surface: str
    why: str
    examples: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RTX4060OptimizationSummary:
    mode: str
    report_path: str
    detected_profile: str
    operator_algebra_mode: str
    overall_readiness_percent: float
    overall_status: str
    goal_readiness_percent: float
    weak_axes: list[str] = field(default_factory=list)
    remaining_goal_items: list[str] = field(default_factory=list)
    benchmark_blockers: list[str] = field(default_factory=list)
    approved_review_total: int = 0
    approved_domain_slices: int = 0
    grounded_review_total: int = 0
    data_collection_lanes: list[RTX4060CollectionLane] = field(default_factory=list)
    performance_tactics: list[str] = field(default_factory=list)
    beginner_actions: list[str] = field(default_factory=list)
    source_reports: dict[str, str] = field(default_factory=dict)
    before_readiness_percent: float = 0.0
    after_readiness_percent: float = 0.0
    delta_readiness_percent: float = 0.0
    improvement_summary: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {
            'mode': self.mode,
            'report_path': self.report_path,
            'detected_profile': self.detected_profile,
            'operator_algebra_mode': self.operator_algebra_mode,
            'overall_readiness_percent': self.overall_readiness_percent,
            'overall_status': self.overall_status,
            'goal_readiness_percent': self.goal_readiness_percent,
            'weak_axes': list(self.weak_axes),
            'remaining_goal_items': list(self.remaining_goal_items),
            'benchmark_blockers': list(self.benchmark_blockers),
            'approved_review_total': self.approved_review_total,
            'approved_domain_slices': self.approved_domain_slices,
            'grounded_review_total': self.grounded_review_total,
            'data_collection_lanes': [item.model_dump() for item in self.data_collection_lanes],
            'performance_tactics': list(self.performance_tactics),
            'beginner_actions': list(self.beginner_actions),
            'source_reports': dict(self.source_reports),
            'before_readiness_percent': self.before_readiness_percent,
            'after_readiness_percent': self.after_readiness_percent,
            'delta_readiness_percent': self.delta_readiness_percent,
            'improvement_summary': dict(self.improvement_summary),
        }


class RTX4060ReasoningCoach:
    def __init__(self, mode: str = 'heuristic') -> None:
        self.mode = mode

    def assess(
        self,
        *,
        workspace: str = '.',
        bootstrap_missing: bool = False,
        output_path: str = 'data/unified_semop_gui_run/rtx4060_reasoning_coach.json',
        unified_output_dir: str = 'data/unified_semop_gui_run',
        unified_store_path: str = 'data/semop_memory.db',
        unified_review_queue_path: str = 'data/ops_review_queue.db',
        unified_benchmark_corpus_path: str = 'data/unified_semop_gui_run/persistent_benchmark_corpus.json',
        math_output_dir: str = 'data/math_world_model_gui_run',
        math_cases_path: str = 'examples/math_world_model_starter.jsonl',
        visual_output_dir: str = 'data/math_world_model_gui_run/visual_3d',
    ) -> RTX4060OptimizationSummary:
        audit = CapabilityAuditRunner(mode=self.mode).run(
            workspace=workspace,
            bootstrap_missing=bootstrap_missing,
            output_path=str(Path(unified_output_dir) / 'capability_audit_report.json'),
            unified_output_dir=unified_output_dir,
            unified_store_path=unified_store_path,
            unified_review_queue_path=unified_review_queue_path,
            unified_benchmark_corpus_path=unified_benchmark_corpus_path,
            math_output_dir=math_output_dir,
            math_cases_path=math_cases_path,
            visual_output_dir=visual_output_dir,
        )
        summary = self._build_summary(
            audit,
            output_path=output_path,
            unified_output_dir=unified_output_dir,
            unified_review_queue_path=unified_review_queue_path,
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

    def improve(
        self,
        *,
        workspace: str = '.',
        bootstrap_missing: bool = False,
        output_path: str = 'data/unified_semop_gui_run/rtx4060_reasoning_improvement.json',
        unified_output_dir: str = 'data/unified_semop_gui_run',
        unified_store_path: str = 'data/semop_memory.db',
        unified_review_queue_path: str = 'data/ops_review_queue.db',
        unified_benchmark_corpus_path: str = 'data/unified_semop_gui_run/persistent_benchmark_corpus.json',
        math_output_dir: str = 'data/math_world_model_gui_run',
        math_cases_path: str = 'examples/math_world_model_starter.jsonl',
        visual_output_dir: str = 'data/math_world_model_gui_run/visual_3d',
        hidden_input: str = 'examples/hidden_premise_eval.jsonl',
        transfer_input: str = 'examples/operator_transfer_eval.jsonl',
        vlso_input: str = 'examples/vlso_eval.jsonl',
        vlso_real_input: str = 'examples/vlso_real_image_eval_gold.jsonl',
        vision_image: str = 'data/scene.png',
    ) -> RTX4060OptimizationSummary:
        before = CapabilityAuditRunner(mode=self.mode).run(
            workspace=workspace,
            bootstrap_missing=bootstrap_missing,
            output_path=str(Path(unified_output_dir) / 'capability_audit_report_before_4060.json'),
            unified_output_dir=unified_output_dir,
            unified_store_path=unified_store_path,
            unified_review_queue_path=unified_review_queue_path,
            unified_benchmark_corpus_path=unified_benchmark_corpus_path,
            math_output_dir=math_output_dir,
            math_cases_path=math_cases_path,
            visual_output_dir=visual_output_dir,
        )
        improvement = CapabilityImprovementRunner(mode=self.mode).run(
            workspace=workspace,
            output_path=str(Path(unified_output_dir) / 'capability_improvement_report.json'),
            bootstrap_missing=bootstrap_missing,
            unified_output_dir=unified_output_dir,
            unified_store_path=unified_store_path,
            unified_review_queue_path=unified_review_queue_path,
            unified_benchmark_corpus_path=unified_benchmark_corpus_path,
            math_output_dir=math_output_dir,
            math_cases_path=math_cases_path,
            visual_output_dir=visual_output_dir,
            hidden_input=hidden_input,
            transfer_input=transfer_input,
            vlso_input=vlso_input,
            vlso_real_input=vlso_real_input,
            vision_image=vision_image,
        )
        after = CapabilityAuditRunner(mode=self.mode).run(
            workspace=workspace,
            bootstrap_missing=bootstrap_missing,
            output_path=str(Path(unified_output_dir) / 'capability_audit_report.json'),
            unified_output_dir=unified_output_dir,
            unified_store_path=unified_store_path,
            unified_review_queue_path=unified_review_queue_path,
            unified_benchmark_corpus_path=unified_benchmark_corpus_path,
            math_output_dir=math_output_dir,
            math_cases_path=math_cases_path,
            visual_output_dir=visual_output_dir,
        )
        summary = self._build_summary(
            after,
            output_path=output_path,
            unified_output_dir=unified_output_dir,
            unified_review_queue_path=unified_review_queue_path,
            before=before,
            improvement=improvement,
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

    def _build_summary(
        self,
        audit: CapabilityAuditSummary,
        *,
        output_path: str,
        unified_output_dir: str,
        unified_review_queue_path: str,
        before: CapabilityAuditSummary | None = None,
        improvement: CapabilityImprovementSummary | None = None,
    ) -> RTX4060OptimizationSummary:
        runtime = audit.runtime_doctor if isinstance(audit.runtime_doctor, dict) else {}
        hardware = runtime.get('hardware_profile', {}) if isinstance(runtime.get('hardware_profile'), dict) else {}
        benchmark_gate = audit.benchmark_gate if isinstance(audit.benchmark_gate, dict) else {}
        gate = benchmark_gate.get('gate', {}) if isinstance(benchmark_gate.get('gate'), dict) else benchmark_gate.get('decision', {}) if isinstance(benchmark_gate.get('decision'), dict) else {}
        proof = audit.generalization_proof if isinstance(audit.generalization_proof, dict) else {}
        goal_tracker = proof.get('goal_tracker', {}) if isinstance(proof.get('goal_tracker'), dict) else {}
        review_stats = self._approved_review_stats(unified_review_queue_path)
        lanes = self._collection_lanes(audit, review_stats)
        tactics = self._performance_tactics(audit, hardware)
        actions = self._beginner_actions(audit, lanes)
        source_reports = {
            'capability_audit': str(audit.report_path),
            'benchmark_gate': str(Path(unified_output_dir) / 'benchmark_gate.json'),
            'generalization_proof': str(Path(unified_output_dir) / 'generalization_proof' / 'generalization_proof_report.json'),
        }
        if improvement is not None:
            source_reports['capability_improvement'] = str(improvement.output_path)
        return RTX4060OptimizationSummary(
            mode='improvement' if improvement is not None else 'assessment',
            report_path=str(output_path),
            detected_profile=str(hardware.get('detected_profile', 'unknown')),
            operator_algebra_mode=str(runtime.get('operator_algebra_mode', hardware.get('operator_algebra_mode', 'symbolic_cpu_first'))),
            overall_readiness_percent=float(audit.overall_readiness_percent),
            overall_status=str(audit.overall_status),
            goal_readiness_percent=float(goal_tracker.get('readiness_percent', 0.0) or 0.0),
            weak_axes=[axis.name for axis in audit.axes if float(axis.score) < 0.85],
            remaining_goal_items=[str(item) for item in goal_tracker.get('remaining_items', [])[:6]],
            benchmark_blockers=[str(item) for item in gate.get('blocking_reasons', [])[:6]],
            approved_review_total=review_stats['approved_total'],
            approved_domain_slices=review_stats['approved_domain_slices'],
            grounded_review_total=review_stats['grounded_review_total'],
            data_collection_lanes=lanes,
            performance_tactics=tactics,
            beginner_actions=actions,
            source_reports=source_reports,
            before_readiness_percent=float(before.overall_readiness_percent) if before is not None else float(audit.overall_readiness_percent),
            after_readiness_percent=float(audit.overall_readiness_percent),
            delta_readiness_percent=round(float(audit.overall_readiness_percent) - float(before.overall_readiness_percent), 1) if before is not None else 0.0,
            improvement_summary=improvement.model_dump() if improvement is not None else {},
        )

    @staticmethod
    def _approved_review_stats(review_queue_path: str) -> dict[str, int]:
        path = Path(review_queue_path)
        if not path.exists():
            return {
                'approved_total': 0,
                'approved_domain_slices': 0,
                'grounded_review_total': 0,
            }
        store = ReviewQueueStore(path)
        items = store.fetch_items(status='approved', limit=5000)
        slices = {(item.domain, item.scenario) for item in items}
        grounded = 0
        for item in items:
            reason_set = {str(reason).strip() for reason in item.reasons}
            if {'grounding_review', 'claim_grounding_review', 'approved_training_trace'} & reason_set:
                grounded += 1
        return {
            'approved_total': len(items),
            'approved_domain_slices': len(slices),
            'grounded_review_total': grounded,
        }

    @staticmethod
    def _axis_score(audit: CapabilityAuditSummary, name: str) -> float:
        for axis in audit.axes:
            if axis.name == name:
                return float(axis.score)
        return 0.0

    def _collection_lanes(self, audit: CapabilityAuditSummary, review_stats: dict[str, int]) -> list[RTX4060CollectionLane]:
        semop_score = self._axis_score(audit, 'semop_reasoning')
        proof_score = self._axis_score(audit, 'generalization_proof')
        math_score = self._axis_score(audit, 'math_world_model')
        visual_score = self._axis_score(audit, 'visual_3d_reconstruction')
        lanes = [
            RTX4060CollectionLane(
                label='Grounded SOP reviews',
                current_count=int(review_stats.get('grounded_review_total', 0)),
                suggested_next_batch=12 if semop_score < 0.85 else 4,
                preferred_surface='SemOp Studio -> Unified chat + Manual fast path',
                why='grounded_explanation_fidelity is the most common gate blocker, so grounded claim traces give the fastest reliability gain.',
                examples=[
                    'SOP를 보고 이 답변을 근거 문장만 남겨서 다시 써줘',
                    '이 상황에서 멈춰야 하는 이유를 SOP 근거와 같이 설명해줘',
                ],
            ),
            RTX4060CollectionLane(
                label='Cross-domain approved reviews',
                current_count=int(review_stats.get('approved_domain_slices', 0)),
                suggested_next_batch=10 if proof_score < 0.85 else 4,
                preferred_surface='SemOp Studio -> Unified chat across different domains',
                why='generalization proof rises when approved traces cover more domain and scenario slices, not just one warehouse path.',
                examples=[
                    'service 문서 근거형 질문 3-4개 승인',
                    'general access reasoning 질문 3-4개 승인',
                ],
            ),
            RTX4060CollectionLane(
                label='Math proof cases',
                current_count=int((audit.math_training or {}).get('total_cases', 0) if isinstance(audit.math_training, dict) else 0),
                suggested_next_batch=8 if math_score < 0.82 else 3,
                preferred_surface='math_world_model_gui.py or SemOp Studio unified chat',
                why='4060에서는 operator-algebra-first 수학 추론이 강점이라, 고품질 증명 사례를 조금씩 더 넣는 편이 효율적입니다.',
                examples=[
                    '정수/조합론 증명 5개',
                    '그림이 있는 기하 문제 3개',
                ],
            ),
            RTX4060CollectionLane(
                label='Visual geometry and 3D corrections',
                current_count=int((audit.visual_reconstruction or {}).get('topology_summary', {}).get('primitive_count', 0) if isinstance(audit.visual_reconstruction, dict) else 0),
                suggested_next_batch=8 if visual_score < 0.85 else 3,
                preferred_surface='math_world_model_gui.py image-to-3D flow',
                why='diagram and 3D correction examples improve geometry transfer and keep figure-heavy questions from dropping into review.',
                examples=[
                    '삼각형/원 도형 JSON 또는 이미지 4개',
                    '단순 실사진 또는 컨테이너 장면 4개',
                ],
            ),
            RTX4060CollectionLane(
                label='Compiler and repair traces',
                current_count=int(review_stats.get('approved_total', 0)),
                suggested_next_batch=6 if semop_score < 0.9 else 2,
                preferred_surface='SemOp Studio -> Manual fast path after blocked answers',
                why='repair and compiler traces are cheap to collect on a 4060 because symbolic verification does most of the work.',
                examples=[
                    '근거 부족 답변을 승인/수정해서 다시 학습',
                    'blocked gate 결과를 follow-up 후 승인',
                ],
            ),
        ]
        return lanes

    @staticmethod
    def _performance_tactics(audit: CapabilityAuditSummary, hardware: dict[str, Any]) -> list[str]:
        tactics = [
            'Keep the RTX 4060 profile in symbolic_first_gpu_assist mode so operator algebra, retrieval, and compiler checks stay first and the GPU only assists where it helps.',
            'Use batch size 1, short generations, and QLoRA/4-bit only for compact local fine-tuning; do not spend 8GB VRAM on long free-form generation first.',
            'After each approved-review batch, rerun the benchmark gate and generalization proof instead of collecting data blindly.',
        ]
        if float(audit.overall_readiness_percent) < 85.0:
            tactics.append('Do not chase larger models first. Raise grounded explanation fidelity and cross-domain approved traces before touching model scale.')
        if str(hardware.get('detected_profile', '')) == 'rtx_4060_8gb':
            tactics.append('This machine already supports QLoRA, so the bottleneck is data quality and review coverage, not missing dependencies.')
        return tactics

    @staticmethod
    def _beginner_actions(audit: CapabilityAuditSummary, lanes: list[RTX4060CollectionLane]) -> list[str]:
        actions = [
            'Open SemOp Studio and start in Unified chat. Ask real prompts instead of filling each lab by hand.',
            'When an answer is good, use Manual fast path -> Save + approve current result.',
            'Collect one small focused batch from the first two lanes below, then run Improve weak areas + re-audit.',
            'Use One-click all-domain training only after you have approved new traces, so the run actually has fresh data to learn from.',
        ]
        if any(item.label == 'Grounded SOP reviews' and item.suggested_next_batch >= 10 for item in lanes):
            actions.append('Prioritize grounded SOP reviews first, because that is the fastest path to unblocking the benchmark gate.')
        return actions

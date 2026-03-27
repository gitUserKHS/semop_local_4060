from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .hardware_profiles import detect_local_hardware, detect_local_ml_stack
from .world_model_math import MathWorldCandidate, WorldModelMathReasoner, WorldModelMathReport


@dataclass
class ProductionMathServiceConfig:
    mode: str = 'heuristic'
    model_id: str = 'Qwen/Qwen2.5-3B-Instruct'
    concept_store_path: str | None = None
    operator_store_path: str | None = None
    affordance_weights_path: str | None = None
    logical_weight_path: str | None = 'data/logical_pattern_weights.json'
    strategy_memory_path: str | None = None
    leworldmodel_path: str | None = None
    audit_log_path: str = 'data/math_world_model_service/audit_log.jsonl'
    accept_score_threshold: float = 0.68
    accept_verification_threshold: float = 0.7
    accept_alignment_threshold: float = 0.58
    accept_leworldmodel_threshold: float = 0.4
    accept_check_pass_rate: float = 0.6
    review_score_threshold: float = 0.5
    require_visual_for_figure_queries: bool = True
    hardware_profile: str = 'auto'

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProductionMathDecision:
    request_id: str
    status: str
    accepted: bool
    elapsed_ms: int
    gate_message: str
    reasons: list[str] = field(default_factory=list)
    top_candidate_source: str = ''
    top_candidate_kind: str = ''
    top_candidate_score: float = 0.0
    top_candidate_verified: bool = False
    verification_score: float = 0.0
    alignment_score: float = 0.0
    check_pass_rate: float = 0.0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProductionMathReadiness:
    ready: bool
    status: str
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    path_checks: dict[str, str] = field(default_factory=dict)
    hardware_profile: dict[str, Any] = field(default_factory=dict)
    dependency_status: dict[str, Any] = field(default_factory=dict)
    operator_algebra_mode: str = 'symbolic_cpu_first'

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProductionMathSelfTestCase:
    name: str
    request: dict[str, Any]
    expected_status: str = 'accepted'

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProductionMathSelfTestResult:
    name: str
    passed: bool
    status: str
    reasons: list[str] = field(default_factory=list)
    elapsed_ms: int = 0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProductionMathSelfTestSummary:
    ready: bool
    status: str
    passed_count: int
    total_count: int
    results: list[ProductionMathSelfTestResult] = field(default_factory=list)
    readiness: ProductionMathReadiness | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            'ready': self.ready,
            'status': self.status,
            'passed_count': self.passed_count,
            'total_count': self.total_count,
            'results': [item.model_dump() for item in self.results],
            'readiness': self.readiness.model_dump() if self.readiness is not None else None,
        }


@dataclass
class ProductionMathMetrics:
    total_requests: int = 0
    accepted_requests: int = 0
    review_requests: int = 0
    error_requests: int = 0
    self_test_runs: int = 0
    last_status: str = 'idle'
    last_request_id: str = ''

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProductionMathServiceResponse:
    request_id: str
    status: str
    accepted: bool
    safe_answer: str
    decision: ProductionMathDecision
    report: WorldModelMathReport | None = None
    warnings: list[str] = field(default_factory=list)
    audit_log_path: str = ''

    def model_dump(self) -> dict[str, Any]:
        return {
            'request_id': self.request_id,
            'status': self.status,
            'accepted': self.accepted,
            'safe_answer': self.safe_answer,
            'decision': self.decision.model_dump(),
            'report': self.report.model_dump() if self.report is not None else None,
            'warnings': list(self.warnings),
            'audit_log_path': self.audit_log_path,
        }


class WorldModelMathProductionService:
    def __init__(self, config: ProductionMathServiceConfig | None = None) -> None:
        self.config = config or ProductionMathServiceConfig()
        self.hardware_profile = detect_local_hardware(self.config.hardware_profile)
        self.dependency_status = detect_local_ml_stack(self.hardware_profile)
        self.reasoner = WorldModelMathReasoner(
            mode=self.config.mode,
            model_id=self.config.model_id,
            concept_store_path=self.config.concept_store_path,
            operator_store_path=self.config.operator_store_path,
            affordance_weights_path=self.config.affordance_weights_path,
            logical_weight_path=self.config.logical_weight_path,
            strategy_memory_path=self.config.strategy_memory_path,
            leworldmodel_path=self.config.leworldmodel_path,
        )
        self.metrics = ProductionMathMetrics()

    def health(self) -> dict[str, Any]:
        readiness = self.readiness()
        return {
            'service': 'world_model_math_production',
            'status': 'ok' if readiness.ready else 'degraded',
            'audit_log_path': self.config.audit_log_path,
            'metrics': self.metrics.model_dump(),
            'hardware_profile': self.hardware_profile.model_dump(),
            'dependency_status': self.dependency_status.model_dump(),
            'operator_algebra_mode': self.hardware_profile.operator_algebra_mode,
            'readiness': readiness.model_dump(),
        }

    def readiness(self) -> ProductionMathReadiness:
        blockers: list[str] = []
        warnings: list[str] = []
        recommended: list[str] = []
        path_checks = {
            'concept_store': self._path_status(self.config.concept_store_path),
            'operator_store': self._path_status(self.config.operator_store_path),
            'affordance_weights': self._path_status(self.config.affordance_weights_path),
            'logical_weights': self._path_status(self.config.logical_weight_path),
            'strategy_memory': self._path_status(self.config.strategy_memory_path),
            'leworldmodel_prior': self._path_status(self.config.leworldmodel_path),
            'audit_dir': self._ensure_audit_dir(),
        }
        if path_checks['audit_dir'] != 'ready':
            blockers.append('Audit log directory is not writable.')
        if path_checks['concept_store'] != 'ready':
            warnings.append('Concept store is missing; geometry grounding will be weaker.')
            recommended.append('Point the service to a populated visual concept store before production geometry use.')
        if path_checks['operator_store'] != 'ready':
            warnings.append('Operator store is missing; diagram-backed operator recovery will be weaker.')
            recommended.append('Point the service to a populated visual operator store before production geometry use.')
        if path_checks['affordance_weights'] != 'ready':
            warnings.append('Affordance weights are missing; visual ranking will fall back to weaker defaults.')
            recommended.append('Provide trained affordance weights if you want stronger visual gating.')
        if path_checks['logical_weights'] != 'ready':
            warnings.append('Logical pattern weights are missing; training improvements will not be reused yet.')
            recommended.append('Run the math training loop once so the world model can reuse learned logical weights.')
        if path_checks['strategy_memory'] != 'ready':
            warnings.append('Strategy memory is missing; successful proof patterns will not yet be reused across similar problems.')
            recommended.append('Run the math training loop once so successful operators and proof hints are stored in local strategy memory.')
        if path_checks['leworldmodel_prior'] != 'ready':
            warnings.append('LeWM latent prior is missing; latent trajectory reranking is disabled.')
            recommended.append('Run the math training loop once so the LeWM-inspired latent prior is exported and reused during solving.')
        if self.hardware_profile.low_vram:
            recommended.append('This low-VRAM CUDA profile stays symbolic/operator-algebra first and should prefer LoRA or QLoRA for any local fine-tuning.')
        if self.dependency_status.missing_core:
            blockers.append('Core ML packages are missing: ' + ', '.join(self.dependency_status.missing_core))
            recommended.append('Install torch and transformers before using local math world-model inference.')
        elif not self.dependency_status.training_ready:
            warnings.append('Parameter-efficient fine-tuning dependencies are incomplete.')
            recommended.append('Install peft to enable LoRA-based local math training.')
        if self.hardware_profile.low_vram and not self.dependency_status.qlora_ready:
            warnings.append('QLoRA dependencies are incomplete for the low-VRAM profile.')
            recommended.append('Install bitsandbytes and accelerate if you want 4-bit QLoRA on RTX 4060-class GPUs.')
        ready = not blockers
        status = 'ready' if ready and not warnings else 'conditional' if ready else 'blocked'
        if self.config.require_visual_for_figure_queries:
            recommended.append('Keep figure-dependent geometry questions in review mode unless a diagram path is attached.')
        return ProductionMathReadiness(
            ready=ready,
            status=status,
            warnings=warnings,
            blockers=blockers,
            recommended_actions=list(dict.fromkeys(recommended)),
            path_checks=path_checks,
            hardware_profile=self.hardware_profile.model_dump(),
            dependency_status=self.dependency_status.model_dump(),
            operator_algebra_mode=self.hardware_profile.operator_algebra_mode,
        )

    def self_test(self) -> ProductionMathSelfTestSummary:
        self.metrics.self_test_runs += 1
        cases = self._self_test_cases()
        results: list[ProductionMathSelfTestResult] = []
        for case in cases:
            started = time.perf_counter()
            try:
                response = self.solve_request(**case.request, metadata={'self_test_case': case.name})
                passed = response.status == case.expected_status
                results.append(
                    ProductionMathSelfTestResult(
                        name=case.name,
                        passed=passed,
                        status=response.status,
                        reasons=list(response.decision.reasons),
                        elapsed_ms=int((time.perf_counter() - started) * 1000),
                    )
                )
            except Exception as exc:
                results.append(
                    ProductionMathSelfTestResult(
                        name=case.name,
                        passed=False,
                        status='error',
                        reasons=[str(exc)],
                        elapsed_ms=int((time.perf_counter() - started) * 1000),
                    )
                )
        passed_count = sum(1 for item in results if item.passed)
        readiness = self.readiness()
        ready = readiness.ready and passed_count == len(results)
        status = 'ready' if ready else 'conditional' if readiness.ready else 'blocked'
        return ProductionMathSelfTestSummary(
            ready=ready,
            status=status,
            passed_count=passed_count,
            total_count=len(results),
            results=results,
            readiness=readiness,
        )

    def solve_request(
        self,
        query: str,
        *,
        source_context: str = '',
        visual_input: Any | None = None,
        task_mode: str = 'auto',
        metadata: dict[str, Any] | None = None,
    ) -> ProductionMathServiceResponse:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        self.metrics.total_requests += 1
        self.metrics.last_request_id = request_id
        if not str(query or '').strip():
            self.metrics.error_requests += 1
            self.metrics.last_status = 'error'
            decision = ProductionMathDecision(
                request_id=request_id,
                status='error',
                accepted=False,
                elapsed_ms=0,
                gate_message='A non-empty problem statement is required.',
                reasons=['empty_query'],
            )
            response = ProductionMathServiceResponse(
                request_id=request_id,
                status='error',
                accepted=False,
                safe_answer='No answer was produced because the problem statement was empty.',
                decision=decision,
                report=None,
                warnings=['Provide a concrete math problem before solving.'],
                audit_log_path=self.config.audit_log_path,
            )
            self._append_audit_log(response, query=query, source_context=source_context, visual_input=visual_input, task_mode=task_mode, metadata=metadata)
            return response
        report = self.reasoner.solve(
            query,
            source_context=source_context,
            visual_input=visual_input,
            task_mode=task_mode,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        decision = self._gate_report(request_id, report, elapsed_ms=elapsed_ms, visual_input=visual_input)
        if decision.accepted:
            self.metrics.accepted_requests += 1
        elif decision.status == 'error':
            self.metrics.error_requests += 1
        else:
            self.metrics.review_requests += 1
        self.metrics.last_status = decision.status
        response = ProductionMathServiceResponse(
            request_id=request_id,
            status=decision.status,
            accepted=decision.accepted,
            safe_answer=self._safe_answer(report, decision),
            decision=decision,
            report=report,
            warnings=self._warnings_from_report(report, decision),
            audit_log_path=self.config.audit_log_path,
        )
        self._append_audit_log(response, query=query, source_context=source_context, visual_input=visual_input, task_mode=task_mode, metadata=metadata)
        return response

    def _gate_report(
        self,
        request_id: str,
        report: WorldModelMathReport,
        *,
        elapsed_ms: int,
        visual_input: Any | None,
    ) -> ProductionMathDecision:
        chosen = report.candidates[0] if report.candidates else MathWorldCandidate(
            source='none',
            kind='none',
            answer='',
            confidence=0.0,
            verification_score=0.0,
            alignment_score=0.0,
            score=0.0,
            verified=False,
            support=[],
        )
        check_pass_rate = self._check_pass_rate(report)
        reasons: list[str] = []
        figure_query = self._expects_visual_context(report.query, report.strategy_prior.family)
        if figure_query and self.config.require_visual_for_figure_queries and not visual_input:
            reasons.append('figure_dependent_query_without_visual')
        if visual_input and report.strategy_prior.family == 'geometry_proof' and report.visual_world_model is None:
            reasons.append('visual_input_did_not_produce_world_model')
        if chosen.score < self.config.accept_score_threshold:
            reasons.append('top_candidate_score_below_accept_threshold')
        if report.verification_score < self.config.accept_verification_threshold:
            reasons.append('verification_score_below_accept_threshold')
        if chosen.alignment_score < self.config.accept_alignment_threshold:
            reasons.append('alignment_score_below_accept_threshold')
        lewm_alignment = float(report.leworldmodel_alignment.get('alignment_score', 0.0) or 0.0) if isinstance(report.leworldmodel_alignment, dict) else 0.0
        if report.leworldmodel_alignment and lewm_alignment < self.config.accept_leworldmodel_threshold:
            reasons.append('leworldmodel_alignment_below_accept_threshold')
        if check_pass_rate < self.config.accept_check_pass_rate:
            reasons.append('check_pass_rate_below_accept_threshold')
        accepted = bool(
            report.solved
            and chosen.score >= self.config.accept_score_threshold
            and report.verification_score >= self.config.accept_verification_threshold
            and chosen.alignment_score >= self.config.accept_alignment_threshold
            and (not report.leworldmodel_alignment or lewm_alignment >= self.config.accept_leworldmodel_threshold)
            and check_pass_rate >= self.config.accept_check_pass_rate
            and 'figure_dependent_query_without_visual' not in reasons
            and 'visual_input_did_not_produce_world_model' not in reasons
        )
        review_signal = (
            chosen.score >= self.config.review_score_threshold
            or report.verification_score >= self.config.review_score_threshold
            or bool(report.beginner_summary)
        )
        status = 'accepted' if accepted else 'review' if review_signal else 'rejected'
        gate_message = self._gate_message(report, accepted, reasons)
        return ProductionMathDecision(
            request_id=request_id,
            status=status,
            accepted=accepted,
            elapsed_ms=elapsed_ms,
            gate_message=gate_message,
            reasons=reasons,
            top_candidate_source=chosen.source,
            top_candidate_kind=chosen.kind,
            top_candidate_score=round(float(chosen.score), 4),
            top_candidate_verified=bool(chosen.verified),
            verification_score=round(float(report.verification_score), 4),
            alignment_score=round(float(chosen.alignment_score), 4),
            check_pass_rate=round(check_pass_rate, 4),
        )

    @staticmethod
    def _check_pass_rate(report: WorldModelMathReport) -> float:
        if not report.checks:
            return 0.0
        passed = sum(1 for item in report.checks if item.passed)
        return round(passed / float(len(report.checks)), 4)

    @staticmethod
    def _expects_visual_context(query: str, family: str) -> bool:
        normalized = query.lower()
        tokens = ('figure', 'diagram', 'shown', 'visible', 'configuration', 'in the given')
        return family == 'geometry_proof' and any(token in normalized for token in tokens)

    @staticmethod
    def _gate_message(report: WorldModelMathReport, accepted: bool, reasons: list[str]) -> str:
        if accepted:
            return 'Accepted for direct use because the chosen proof path cleared the verification and consistency gates.'
        if not reasons:
            return 'Review is required because the system produced only a partial strategy.'
        if 'figure_dependent_query_without_visual' in reasons:
            return 'Review is required because the problem appears to depend on a figure, but no diagram input was attached.'
        if 'visual_input_did_not_produce_world_model' in reasons:
            return 'Review is required because the diagram input did not produce a stable geometry world model.'
        if 'leworldmodel_alignment_below_accept_threshold' in reasons:
            return 'Review is required because the learned latent trajectory prior did not agree strongly enough with the current proof path.'
        if report.strategy_prior.family == 'research_math':
            return 'Review is required because research-style claims need a stronger explicit proof trace before deployment.'
        return 'Review is required because the current answer did not clear the production verification bar.'

    @staticmethod
    def _safe_answer(report: WorldModelMathReport, decision: ProductionMathDecision) -> str:
        if decision.accepted:
            return report.chosen_answer
        summary = ' '.join(report.beginner_summary[:2]).strip()
        next_step = report.next_actions[0] if report.next_actions else 'Add clearer givens and rerun.'
        return (
            'Review required. The system found a plausible direction but did not clear the production gate. '
            f'Strategy hint: {summary or "Recover the latent structure first."} '
            f'Next step: {next_step}'
        )

    @staticmethod
    def _warnings_from_report(report: WorldModelMathReport, decision: ProductionMathDecision) -> list[str]:
        warnings = []
        if report.strategy_prior.family == 'geometry_proof' and report.visual_world_model is None:
            warnings.append('Geometry proof ran without a grounded diagram world model.')
        if not decision.accepted:
            warnings.append('This response should be reviewed before direct operational use.')
        return warnings

    def _append_audit_log(
        self,
        response: ProductionMathServiceResponse,
        *,
        query: str,
        source_context: str,
        visual_input: Any | None,
        task_mode: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        audit_dir = os.path.dirname(self.config.audit_log_path)
        if audit_dir:
            os.makedirs(audit_dir, exist_ok=True)
        record = {
            'timestamp_ms': int(time.time() * 1000),
            'request_id': response.request_id,
            'status': response.status,
            'accepted': response.accepted,
            'query': query,
            'source_context': source_context,
            'visual_input': visual_input,
            'task_mode': task_mode,
            'decision': response.decision.model_dump(),
            'warnings': list(response.warnings),
            'safe_answer': response.safe_answer,
            'report': response.report.model_dump() if response.report is not None else None,
            'metadata': dict(metadata or {}),
        }
        with open(self.config.audit_log_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    @staticmethod
    def _path_status(path: str | None) -> str:
        if not path:
            return 'not_configured'
        return 'ready' if os.path.exists(path) else 'missing'

    def _ensure_audit_dir(self) -> str:
        audit_dir = os.path.dirname(self.config.audit_log_path)
        if not audit_dir:
            return 'ready'
        try:
            os.makedirs(audit_dir, exist_ok=True)
        except OSError:
            return 'blocked'
        return 'ready'

    def _self_test_cases(self) -> list[ProductionMathSelfTestCase]:
        cases = [
            ProductionMathSelfTestCase(
                name='number_theory_parity',
                request={
                    'query': 'Prove that the sum of two odd integers is even.',
                    'task_mode': 'number_theory_proof',
                    'source_context': 'Use parity rewriting and explicit divisibility steps.',
                },
                expected_status='accepted',
            ),
            ProductionMathSelfTestCase(
                name='research_average_argument',
                request={
                    'query': 'Show that every finite set of real numbers contains an element that is at most the average of the set.',
                    'task_mode': 'research_math',
                    'source_context': 'Use the extremal element and contradiction if needed.',
                },
                expected_status='review',
            ),
        ]
        geometry_scene = 'examples/vlso/geometry_scene.json'
        if os.path.exists(geometry_scene):
            cases.append(
                ProductionMathSelfTestCase(
                    name='geometry_figure_route',
                    request={
                        'query': 'In the given geometry configuration, what proof strategy should I try first to show two angles are equal?',
                        'task_mode': 'geometry_proof',
                        'source_context': 'Recover angle relations and similar-triangle routes first.',
                        'visual_input': geometry_scene,
                    },
                    expected_status='accepted',
                )
            )
        return cases

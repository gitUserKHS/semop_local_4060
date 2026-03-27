from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .hard_problem_engine import HardProblemEngine
from .hardware_profiles import detect_local_hardware
from .leworldmodel_adapter import LeWorldModelAdapter
from .world_model_math_service import ProductionMathServiceConfig, WorldModelMathProductionService


@dataclass
class MathTrainingCase:
    case_id: str
    query: str
    source_context: str = ''
    visual_input: str = ''
    task_mode: str = 'auto'
    expected_family: str = ''
    expected_status: str = 'accepted'
    required_terms: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)
    exact_answer: str = ''
    exact_answer_aliases: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'MathTrainingCase':
        return cls(
            case_id=str(payload.get('case_id') or ''),
            query=str(payload.get('query') or ''),
            source_context=str(payload.get('source_context') or ''),
            visual_input=str(payload.get('visual_input') or ''),
            task_mode=str(payload.get('task_mode') or 'auto'),
            expected_family=str(payload.get('expected_family') or ''),
            expected_status=str(payload.get('expected_status') or 'accepted'),
            required_terms=[str(item) for item in payload.get('required_terms', []) or []],
            forbidden_terms=[str(item) for item in payload.get('forbidden_terms', []) or []],
            exact_answer=str(payload.get('exact_answer') or ''),
            exact_answer_aliases=[str(item) for item in payload.get('exact_answer_aliases', []) or []],
        )


@dataclass
class MathCaseEvaluation:
    case_id: str
    status: str
    accepted: bool
    solved: bool
    family: str
    verification_score: float
    top_candidate_score: float
    required_terms_hit: list[str] = field(default_factory=list)
    missing_required_terms: list[str] = field(default_factory=list)
    forbidden_terms_hit: list[str] = field(default_factory=list)
    exact_match: bool = False
    family_match: bool = False
    status_match: bool = False
    process_steps: int = 0
    operator_steps: int = 0
    overall_score: float = 0.0
    passed: bool = False
    safe_answer: str = ''
    chosen_answer: str = ''
    solution_process: list[str] = field(default_factory=list)
    operator_trace: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MathWorldModelEvaluationSummary:
    input_path: str
    output_path: str
    total_cases: int
    passed_cases: int
    accepted_cases: int
    solved_cases: int
    exact_match_cases: int
    average_score: float
    results: list[MathCaseEvaluation] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            'input_path': self.input_path,
            'output_path': self.output_path,
            'total_cases': self.total_cases,
            'passed_cases': self.passed_cases,
            'accepted_cases': self.accepted_cases,
            'solved_cases': self.solved_cases,
            'exact_match_cases': self.exact_match_cases,
            'average_score': self.average_score,
            'results': [item.model_dump() for item in self.results],
        }


@dataclass
class MathWorldModelTrainingSummary:
    input_path: str
    output_dir: str
    epochs: int
    starter_created: bool
    total_cases: int
    baseline_average_score: float
    final_average_score: float
    improved_case_count: int
    passed_cases: int
    accepted_cases: int
    exact_match_cases: int
    hardware_profile: dict[str, Any] = field(default_factory=dict)
    operator_algebra_mode: str = 'symbolic_cpu_first'
    leworldmodel_metrics: dict[str, Any] = field(default_factory=dict)
    leworldmodel_effect: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    final_results: list[MathCaseEvaluation] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            'input_path': self.input_path,
            'output_dir': self.output_dir,
            'epochs': self.epochs,
            'starter_created': self.starter_created,
            'total_cases': self.total_cases,
            'baseline_average_score': self.baseline_average_score,
            'final_average_score': self.final_average_score,
            'improved_case_count': self.improved_case_count,
            'passed_cases': self.passed_cases,
            'accepted_cases': self.accepted_cases,
            'exact_match_cases': self.exact_match_cases,
            'hardware_profile': dict(self.hardware_profile),
            'operator_algebra_mode': self.operator_algebra_mode,
            'leworldmodel_metrics': dict(self.leworldmodel_metrics),
            'leworldmodel_effect': dict(self.leworldmodel_effect),
            'artifact_paths': dict(self.artifact_paths),
            'final_results': [item.model_dump() for item in self.final_results],
        }


def _starter_cases() -> list[MathTrainingCase]:
    return [
        MathTrainingCase(
            case_id='odd_sum_even',
            query='Prove that the sum of two odd integers is even.',
            source_context='Use parity rewriting and explicit divisibility steps.',
            task_mode='number_theory_proof',
            expected_family='number_theory_proof',
            expected_status='accepted',
            required_terms=['odd integers', 'even'],
            exact_answer='therefore the sum of two odd integers is even',
        ),
        MathTrainingCase(
            case_id='odd_square_odd',
            query='Prove that the square of an odd integer is odd.',
            source_context='Prefer parity rewriting and keep the 2k+1 form visible.',
            task_mode='number_theory_proof',
            expected_family='number_theory_proof',
            expected_status='accepted',
            required_terms=['odd integer', 'odd'],
            exact_answer='therefore the square of an odd integer is odd',
        ),
        MathTrainingCase(
            case_id='consecutive_product_even',
            query='Prove that the product of two consecutive integers is even.',
            source_context='Split the pair into n and n+1 and isolate the parity claim.',
            task_mode='number_theory_proof',
            expected_family='number_theory_proof',
            expected_status='accepted',
            required_terms=['consecutive integers', 'even'],
            exact_answer='therefore the product of two consecutive integers is even',
        ),
        MathTrainingCase(
            case_id='three_consecutive_sum_mod3',
            query='Prove that the sum of three consecutive integers is divisible by 3.',
            source_context='Parameterize the integers and factor the sum cleanly.',
            task_mode='number_theory_proof',
            expected_family='number_theory_proof',
            expected_status='accepted',
            required_terms=['consecutive integers', 'divisible by 3'],
            exact_answer='therefore the sum of three consecutive integers is divisible by 3',
        ),
        MathTrainingCase(
            case_id='induction_sum_formula',
            query='Prove that 1+2+...+n = n(n+1)/2 for all positive integers n.',
            source_context='Use base case, induction hypothesis, and the k+1 step explicitly.',
            task_mode='research_math',
            expected_family='research_math',
            expected_status='accepted',
            required_terms=['base case', 'induction'],
            exact_answer='n(n+1)/2',
        ),
        MathTrainingCase(
            case_id='infinitely_many_primes',
            query='Prove that there are infinitely many primes.',
            source_context='Prefer contradiction plus an Euclid-style auxiliary number.',
            task_mode='number_theory_proof',
            expected_family='number_theory_proof',
            expected_status='accepted',
            required_terms=['infinitely many primes', 'contradiction'],
            exact_answer='therefore there are infinitely many primes',
        ),
        MathTrainingCase(
            case_id='extremal_average_bound',
            query='Show that every finite set of real numbers contains an element that is at most the average of the set.',
            source_context='Choose the extremal element and compare it to the average.',
            task_mode='research_math',
            expected_family='research_math',
            expected_status='accepted',
            required_terms=['average', 'finite set'],
            exact_answer='therefore some element is at most the average',
        ),
        MathTrainingCase(
            case_id='mutilated_chessboard',
            query='Show that a chessboard with two opposite corners removed cannot be tiled by dominoes.',
            source_context='Use checkerboard coloring and a preserved invariant.',
            task_mode='combinatorics_proof',
            expected_family='combinatorics_proof',
            expected_status='accepted',
            required_terms=['domino', 'chessboard'],
            exact_answer='cannot be tiled by dominoes',
        ),
    ]


def ensure_starter_math_cases(path: str = 'examples/math_world_model_starter.jsonl') -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with target.open('w', encoding='utf-8') as handle:
            for case in _starter_cases():
                handle.write(json.dumps(case.model_dump(), ensure_ascii=False) + '\n')
    return str(target)


def load_math_training_cases(path: str) -> list[MathTrainingCase]:
    cases: list[MathTrainingCase] = []
    with open(path, 'r', encoding='utf-8') as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                cases.append(MathTrainingCase.from_dict(payload))
    return cases


class WorldModelMathTrainer:
    def __init__(
        self,
        *,
        mode: str = 'heuristic',
        model_id: str = 'Qwen/Qwen2.5-3B-Instruct',
        concept_store_path: str | None = None,
        operator_store_path: str | None = None,
        affordance_weights_path: str | None = None,
        hardware_profile: str = 'auto',
    ) -> None:
        self.mode = mode
        self.model_id = model_id
        self.concept_store_path = concept_store_path
        self.operator_store_path = operator_store_path
        self.affordance_weights_path = affordance_weights_path
        self.hardware_profile_name = hardware_profile
        self.hardware_profile = detect_local_hardware(hardware_profile)
        self.lewm = LeWorldModelAdapter()

    def train_from_cases(
        self,
        input_path: str,
        output_dir: str,
        *,
        epochs: int = 2,
        bootstrap_starter: bool = False,
    ) -> MathWorldModelTrainingSummary:
        starter_created = False
        if bootstrap_starter:
            ensure_starter_math_cases(input_path)
            starter_created = True
        cases = load_math_training_cases(input_path)
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        baseline = self.evaluate_cases(
            input_path,
            output_dir,
            summary_name='math_eval_baseline.json',
            use_strategy_memory=False,
            use_leworldmodel=False,
        )
        logical_weight_path = str(output_root / 'logical_pattern_weights.json')
        learning_engine = HardProblemEngine(mode=self.mode, model_id=self.model_id, logical_weight_path=logical_weight_path)
        for _ in range(max(1, int(epochs))):
            for case in cases:
                response = self._service(output_dir, use_strategy_memory=False, use_leworldmodel=False).solve_request(
                    case.query,
                    source_context=case.source_context,
                    visual_input=case.visual_input or None,
                    task_mode=case.task_mode,
                    metadata={'surface': 'math_training_loop', 'case_id': case.case_id},
                )
                evaluation = self._evaluate_case(case, response)
                hard_report = learning_engine.solve(case.query)
                learning_engine.learn_from_report(hard_report, success=evaluation.passed)
        preliminary_final = self.evaluate_cases(input_path, output_dir, summary_name='math_eval_after_weights.json')
        strategy_memory = self._build_strategy_memory(preliminary_final.results)
        strategy_memory_path = Path(output_dir) / 'math_strategy_memory.json'
        strategy_memory_path.write_text(json.dumps(strategy_memory, ensure_ascii=False, indent=2), encoding='utf-8')
        leworldmodel_path = Path(output_dir) / 'math_leworldmodel_prior.json'
        self._fit_leworldmodel_prior(preliminary_final.results, str(leworldmodel_path))
        leworldmodel_metrics = self._load_leworldmodel_metrics(str(leworldmodel_path))
        no_lewm_eval = self.evaluate_cases(
            input_path,
            output_dir,
            summary_name='math_eval_no_lewm.json',
            use_strategy_memory=True,
            use_leworldmodel=False,
        )
        final_eval = self.evaluate_cases(input_path, output_dir, summary_name='math_eval_final.json')
        leworldmodel_effect = self._build_leworldmodel_effect(no_lewm_eval, final_eval, leworldmodel_metrics)
        leworldmodel_effect_path = output_root / 'math_leworldmodel_effect.json'
        leworldmodel_effect_path.write_text(json.dumps(leworldmodel_effect, ensure_ascii=False, indent=2), encoding='utf-8')
        improved_case_count = self._improved_case_count(baseline.results, final_eval.results)
        summary = MathWorldModelTrainingSummary(
            input_path=input_path,
            output_dir=output_dir,
            epochs=max(1, int(epochs)),
            starter_created=starter_created,
            total_cases=final_eval.total_cases,
            baseline_average_score=baseline.average_score,
            final_average_score=final_eval.average_score,
            improved_case_count=improved_case_count,
            passed_cases=final_eval.passed_cases,
            accepted_cases=final_eval.accepted_cases,
            exact_match_cases=final_eval.exact_match_cases,
            hardware_profile=self.hardware_profile.model_dump(),
            operator_algebra_mode=self.hardware_profile.operator_algebra_mode,
            leworldmodel_metrics=leworldmodel_metrics,
            leworldmodel_effect=leworldmodel_effect,
            artifact_paths={
                'baseline_eval': baseline.output_path,
                'weights_eval': str(output_root / 'math_eval_after_weights.json'),
                'no_lewm_eval': no_lewm_eval.output_path,
                'final_eval': final_eval.output_path,
                'logical_weights': logical_weight_path,
                'strategy_memory': str(strategy_memory_path),
                'leworldmodel_prior': str(leworldmodel_path),
                'leworldmodel_effect': str(leworldmodel_effect_path),
                'audit_log': str(output_root / 'audit_log.jsonl'),
                'training_summary': str(output_root / 'math_training_summary.json'),
                'starter_cases': input_path,
            },
            final_results=final_eval.results,
        )
        summary_path = Path(summary.artifact_paths['training_summary'])
        summary_path.write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

    def evaluate_cases(
        self,
        input_path: str,
        output_dir: str,
        *,
        summary_name: str = 'math_eval_summary.json',
        use_strategy_memory: bool = True,
        use_leworldmodel: bool = True,
    ) -> MathWorldModelEvaluationSummary:
        cases = load_math_training_cases(input_path)
        results: list[MathCaseEvaluation] = []
        service = self._service(output_dir, use_strategy_memory=use_strategy_memory, use_leworldmodel=use_leworldmodel)
        for case in cases:
            response = service.solve_request(
                case.query,
                source_context=case.source_context,
                visual_input=case.visual_input or None,
                task_mode=case.task_mode,
                metadata={'surface': 'math_evaluation', 'case_id': case.case_id},
            )
            results.append(self._evaluate_case(case, response))
        summary = MathWorldModelEvaluationSummary(
            input_path=input_path,
            output_path=str(Path(output_dir) / summary_name),
            total_cases=len(results),
            passed_cases=sum(1 for item in results if item.passed),
            accepted_cases=sum(1 for item in results if item.accepted),
            solved_cases=sum(1 for item in results if item.solved),
            exact_match_cases=sum(1 for item in results if item.exact_match),
            average_score=round(sum(item.overall_score for item in results) / float(len(results) or 1), 4),
            results=results,
        )
        Path(summary.output_path).write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

    def _service(
        self,
        output_dir: str,
        *,
        use_strategy_memory: bool = True,
        use_leworldmodel: bool = True,
    ) -> WorldModelMathProductionService:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        return WorldModelMathProductionService(
            ProductionMathServiceConfig(
                model_id=self.model_id,
                mode=self.mode,
                concept_store_path=self.concept_store_path,
                operator_store_path=self.operator_store_path,
                affordance_weights_path=self.affordance_weights_path,
                logical_weight_path=str(output_root / 'logical_pattern_weights.json'),
                strategy_memory_path=str(output_root / 'math_strategy_memory.json') if use_strategy_memory else None,
                leworldmodel_path=str(output_root / 'math_leworldmodel_prior.json') if use_leworldmodel else None,
                audit_log_path=str(output_root / 'audit_log.jsonl'),
                hardware_profile=self.hardware_profile_name,
            )
        )

    @staticmethod
    def _aggregate_text(response: Any) -> str:
        report = response.report
        chunks: list[str] = [response.safe_answer]
        if report is not None:
            chunks.append(report.chosen_answer)
            chunks.extend(report.beginner_summary)
            chunks.extend(report.next_actions)
            chunks.extend(report.solution_process)
            chunks.extend(report.operator_trace)
        return ' '.join(str(item) for item in chunks if item).lower()

    def _evaluate_case(self, case: MathTrainingCase, response: Any) -> MathCaseEvaluation:
        report = response.report
        aggregate_text = self._aggregate_text(response)
        required_hit = [term for term in case.required_terms if term.lower() in aggregate_text]
        missing_required = [term for term in case.required_terms if term.lower() not in aggregate_text]
        forbidden_hit = [term for term in case.forbidden_terms if term.lower() in aggregate_text]
        exact_targets = [item for item in [case.exact_answer, *case.exact_answer_aliases] if item]
        exact_match = True if not exact_targets else any(term.lower() in aggregate_text for term in exact_targets)
        family = report.strategy_prior.family if report is not None else ''
        family_match = not case.expected_family or family == case.expected_family
        status_match = response.status == case.expected_status
        term_ratio = len(required_hit) / float(len(case.required_terms) or 1)
        forbidden_score = 1.0 if not forbidden_hit else 0.0
        exact_score = 1.0 if exact_match else 0.0
        process_steps = len(report.solution_process) if report is not None else 0
        operator_steps = len(report.operator_trace) if report is not None else 0
        process_score = 1.0 if process_steps > 0 else 0.0
        overall = round(
            (0.25 * (1.0 if status_match else 0.0))
            + (0.15 * (1.0 if family_match else 0.0))
            + (0.25 * min(1.0, term_ratio))
            + (0.15 * forbidden_score)
            + (0.1 * exact_score)
            + (0.1 * process_score),
            4,
        )
        passed = bool(status_match and family_match and not forbidden_hit and term_ratio >= 0.5 and overall >= 0.72)
        return MathCaseEvaluation(
            case_id=case.case_id,
            status=response.status,
            accepted=bool(response.accepted),
            solved=bool(report.solved) if report is not None else False,
            family=family,
            verification_score=round(float(response.decision.verification_score), 4),
            top_candidate_score=round(float(response.decision.top_candidate_score), 4),
            required_terms_hit=required_hit,
            missing_required_terms=missing_required,
            forbidden_terms_hit=forbidden_hit,
            exact_match=exact_match,
            family_match=family_match,
            status_match=status_match,
            process_steps=process_steps,
            operator_steps=operator_steps,
            overall_score=overall,
            passed=passed,
            safe_answer=str(response.safe_answer),
            chosen_answer=str(report.chosen_answer) if report is not None else '',
            solution_process=list(report.solution_process) if report is not None else [],
            operator_trace=list(report.operator_trace) if report is not None else [],
            matched_patterns=list(report.matched_patterns) if report is not None else [],
        )

    def _fit_leworldmodel_prior(self, results: list[MathCaseEvaluation], output_path: str) -> None:
        records = []
        for item in results:
            if not item.passed:
                continue
            step_items = [
                [item.family, item.status],
                item.solution_process[:6],
                item.operator_trace[:8],
                item.matched_patterns[:6],
                item.required_terms_hit[:6],
            ]
            record = self.lewm.make_record(item.case_id, 'math_world_model', step_items, metadata={'overall_score': item.overall_score})
            if record.steps:
                records.append(record)
        if not records:
            return
        self.lewm.fit(
            records,
            output_path=output_path,
            domain='math_world_model',
            source_summary={'passed_case_count': len(records)},
        )

    @staticmethod
    def _load_leworldmodel_metrics(path: str) -> dict[str, Any]:
        if not path or not os.path.exists(path):
            return {}
        try:
            payload = json.loads(Path(path).read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            'artifact_path': str(payload.get('artifact_path') or path),
            'domain': str(payload.get('domain') or 'math_world_model'),
            'example_count': int(payload.get('example_count') or 0),
            'step_count': int(payload.get('step_count') or 0),
            'sigreg_score': round(float(payload.get('sigreg_score') or 0.0), 4),
            'predictive_consistency': round(float(payload.get('predictive_consistency') or 0.0), 4),
            'straightness_score': round(float(payload.get('straightness_score') or 0.0), 4),
            'top_tokens': [str(item) for item in payload.get('top_tokens', []) or []][:12],
        }

    @staticmethod
    def _build_leworldmodel_effect(
        without_lewm: MathWorldModelEvaluationSummary,
        with_lewm: MathWorldModelEvaluationSummary,
        leworldmodel_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        without_by_case = {item.case_id: item for item in without_lewm.results}
        improved_cases: list[str] = []
        regressed_cases: list[str] = []
        changed_cases: list[str] = []
        for item in with_lewm.results:
            baseline = without_by_case.get(item.case_id)
            if baseline is None:
                continue
            if item.overall_score > baseline.overall_score:
                improved_cases.append(item.case_id)
                changed_cases.append(item.case_id)
            elif item.overall_score < baseline.overall_score:
                regressed_cases.append(item.case_id)
                changed_cases.append(item.case_id)
        return {
            'enabled': bool(leworldmodel_metrics),
            'with_lewm_average_score': with_lewm.average_score,
            'without_lewm_average_score': without_lewm.average_score,
            'average_score_delta': round(with_lewm.average_score - without_lewm.average_score, 4),
            'accepted_case_delta': with_lewm.accepted_cases - without_lewm.accepted_cases,
            'passed_case_delta': with_lewm.passed_cases - without_lewm.passed_cases,
            'exact_match_delta': with_lewm.exact_match_cases - without_lewm.exact_match_cases,
            'improved_cases': improved_cases,
            'regressed_cases': regressed_cases,
            'changed_case_count': len(changed_cases),
            'leworldmodel_metrics': dict(leworldmodel_metrics),
        }

    @staticmethod
    def _build_strategy_memory(results: list[MathCaseEvaluation]) -> dict[str, Any]:
        families: dict[str, dict[str, Any]] = {}
        for item in results:
            if not item.passed or not item.family:
                continue
            payload = families.setdefault(item.family, {'successful_cases': 0, 'operator_counts': {}, 'hint_counts': {}})
            payload['successful_cases'] += 1
            for operator in item.operator_trace:
                key = str(operator)
                payload['operator_counts'][key] = int(payload['operator_counts'].get(key, 0)) + 1
            for hint in item.solution_process[:4]:
                key = str(hint)
                payload['hint_counts'][key] = int(payload['hint_counts'].get(key, 0)) + 1
        summary_families: dict[str, dict[str, Any]] = {}
        for family, payload in families.items():
            operator_counts = payload.get('operator_counts', {})
            hint_counts = payload.get('hint_counts', {})
            top_operators = [key for key, _value in sorted(operator_counts.items(), key=lambda item: (-item[1], item[0]))[:6]]
            top_hints = [key for key, _value in sorted(hint_counts.items(), key=lambda item: (-item[1], item[0]))[:4]]
            summary_families[family] = {
                'successful_cases': int(payload.get('successful_cases', 0)),
                'top_operators': top_operators,
                'top_hints': top_hints,
            }
        return {'families': summary_families}

    @staticmethod
    def _improved_case_count(left: list[MathCaseEvaluation], right: list[MathCaseEvaluation]) -> int:
        baseline = {item.case_id: item.overall_score for item in left}
        return sum(1 for item in right if item.overall_score > baseline.get(item.case_id, 0.0))

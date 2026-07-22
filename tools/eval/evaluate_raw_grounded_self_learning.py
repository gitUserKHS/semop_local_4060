from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter, process_time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_low_resource_transfer import _peak_rss_bytes
from semop.kernel import (
    GroundedLearningBatch,
    LearningMetrics,
    LearningSplit,
    LearningTask,
    OperatorKernel,
    PolicyLearner,
    RasterImage,
    RasterVisionProblem,
    RawExperienceGrounder,
    RawLearningExample,
    RawSelfLearningLoop,
    SelfLearningBudget,
    SelfLearningLoop,
    SolveBudget,
    TaskEvaluation,
    TypedDomainRequest,
    VisionPropertyGoal,
    summarize_task_evaluations,
    task_evaluation_from_result,
)
from semop.kernel.engine import ActionPolicy
from semop.kernel.model import SolveResult
from semop.tiny_controller import (
    NumpyTinyController,
    StructuralLinearPolicy,
    StructuralPolicyLearner,
    TinyControllerConfig,
    TinyControllerPolicyLearner,
)


CONTROLLER_PROFILES = (
    "sparse",
    "recurrent-diagnostic",
    "recurrent-compact",
    "recurrent-full",
)
WHITE = (255, 255, 255)
RED = (255, 0, 0)


def evaluate_raw_grounded_self_learning(
    *,
    controller_profile: str = "recurrent-diagnostic",
    controller_epochs: int = 5,
    controller_learning_rate: float = 1e-3,
    device: str = "cpu",
    seed: int = 31,
    min_transfer_expansion_reduction: float = 0.30,
    artifact_output: str | Path | None = None,
) -> dict[str, Any]:
    """Train on raw language and evaluate untouched raw math and pixels."""

    learner = _controller_learner(
        controller_profile,
        epochs=controller_epochs,
        learning_rate=controller_learning_rate,
        device=device,
        seed=seed,
    )
    grounder = RawExperienceGrounder(hard_negatives_per_example=4)
    training_examples, promotion_examples = _raw_language_learning_split()
    transfer_examples = _raw_math_vision_transfer_examples()
    learning_loop = SelfLearningLoop(
        learner=learner,
        budget=SelfLearningBudget(
            solve_budget=SolveBudget(
                max_steps=32,
                max_expansions=20_000,
                timeout_seconds=10.0,
            ),
            min_expansion_reduction=0.01,
        ),
    )

    peak_before = _peak_rss_bytes()
    started = perf_counter()
    raw_result = RawSelfLearningLoop(
        learning_loop,
        grounder=grounder,
    ).run(
        training_examples,
        promotion_examples,
        namespace=f"raw-grounded-{seed}",
    )

    transfer_batch = grounder.ground(
        transfer_examples,
        namespace=f"raw-transfer-{seed}",
    )
    transfer_tasks = transfer_batch.require_complete()
    baseline_metrics, baseline_tasks, _baseline_results = _evaluate_tasks(
        transfer_tasks,
        None,
    )
    policy = raw_result.learning.final_policy
    candidate_metrics, candidate_tasks, candidate_results = _evaluate_tasks(
        transfer_tasks,
        policy,
    )
    wall_seconds = perf_counter() - started
    peak_after = _peak_rss_bytes()

    active_candidate = raw_result.learning.active_candidate
    restored_metrics: LearningMetrics | None = None
    restored_results: dict[str, SolveResult] = {}
    artifact_round_trip = False
    portable_cpu_runtime = False
    restored_runtime_name: str | None = None
    if active_candidate is not None:
        restored = learner.restore(active_candidate.artifact)
        restored_runtime_name = type(restored).__name__
        portable_cpu_runtime = isinstance(
            restored,
            (NumpyTinyController, StructuralLinearPolicy),
        )
        restored_metrics, _restored_tasks, restored_results = _evaluate_tasks(
            transfer_tasks,
            restored,
        )
        artifact_round_trip = _same_verified_behavior(
            transfer_tasks,
            candidate_results,
            restored_results,
        )

    primitive_replay = _all_successes_replay(
        transfer_tasks,
        candidate_results,
    )
    training_domains = tuple(
        sorted(task.domain for task in raw_result.grounding.training.tasks)
    )
    transfer_domains = tuple(sorted({task.domain for task in transfer_tasks}))
    learning_fingerprints = _fingerprints(
        raw_result.grounding.training,
        raw_result.grounding.heldout,
    )
    transfer_fingerprints = _fingerprints(transfer_batch)
    final_input_disjoint = learning_fingerprints[0].isdisjoint(
        transfer_fingerprints[0]
    )
    final_semantic_disjoint = learning_fingerprints[1].isdisjoint(
        transfer_fingerprints[1]
    )
    per_domain_reduction = {
        domain: _domain_expansion_reduction(
            baseline_metrics,
            candidate_metrics,
            domain,
        )
        for domain in transfer_domains
    }
    promotion_iteration = raw_result.learning.iterations[-1]
    artifact_bytes = (
        active_candidate.artifact_bytes if active_candidate is not None else 0
    )
    parameter_count = (
        active_candidate.parameter_count if active_candidate is not None else 0
    )
    inference_resources = _measure_portable_inference(active_candidate, seed=seed)
    gates = {
        "raw_grounding_complete": (
            raw_result.grounding.training.complete
            and raw_result.grounding.heldout.complete
            and transfer_batch.complete
        ),
        "promotion_split_leakage_free": raw_result.grounding.leakage_free,
        "final_inputs_untouched": final_input_disjoint and final_semantic_disjoint,
        "language_only_training": set(training_domains) == {"language"},
        "math_vision_only_transfer": set(transfer_domains) == {"math", "vision"},
        "controller_promoted_on_language_holdout": raw_result.promoted,
        "verified_training_traces_recorded": bool(
            raw_result.learning.corpus.records
        ),
        "transfer_solve_rate_not_regressed": (
            candidate_metrics.verified_solve_rate
            >= baseline_metrics.verified_solve_rate - 0.01
        ),
        "proof_soundness_100_percent": candidate_metrics.proof_soundness == 1.0,
        "false_positives_zero": candidate_metrics.false_positives == 0,
        "math_expansions_reduced": per_domain_reduction["math"]
        >= min_transfer_expansion_reduction,
        "vision_expansions_reduced": per_domain_reduction["vision"]
        >= min_transfer_expansion_reduction,
        "primitive_full_replay_verified": primitive_replay,
        "portable_artifact_round_trip": artifact_round_trip,
        "portable_cpu_runtime": portable_cpu_runtime,
        "fresh_process_inference_verified": (
            inference_resources.get("verified") is True
        ),
        "parameters_under_15m": parameter_count <= 15_000_000,
        "artifact_under_64mb": artifact_bytes <= 64 * 1024 * 1024,
        "transfer_p95_under_10s": candidate_metrics.p95_cpu_seconds <= 10.0,
        "additional_peak_rss_under_512mb": int(
            inference_resources.get("additional_peak_rss_bytes", 2**63)
        )
        <= 512 * 1024 * 1024,
    }
    all_gates_passed = all(gates.values())
    persisted_artifact = _persist_candidate_artifact(
        active_candidate,
        artifact_output,
        allowed=all_gates_passed,
    )
    return {
        "schema_version": 1,
        "suite": "raw-grounded-self-learning",
        "seed": seed,
        "controller": {
            "profile": controller_profile,
            "epochs": 0 if controller_profile == "sparse" else controller_epochs,
            "learning_rate": (
                0.0
                if controller_profile == "sparse"
                else controller_learning_rate
            ),
            "device": "dependency-free" if controller_profile == "sparse" else device,
            "kind": active_candidate.kind if active_candidate is not None else None,
            "parameters": parameter_count,
            "training_updates": (
                active_candidate.training_updates
                if active_candidate is not None
                else 0
            ),
        },
        "data": {
            "training_raw_examples": len(training_examples),
            "promotion_raw_examples": len(promotion_examples),
            "transfer_raw_examples": len(transfer_examples),
            "training_domains": training_domains,
            "transfer_domains": transfer_domains,
            "training_grounding_failures": len(
                raw_result.grounding.training.failures
            ),
            "promotion_grounding_failures": len(
                raw_result.grounding.heldout.failures
            ),
            "transfer_grounding_failures": len(transfer_batch.failures),
            "promotion_input_overlap": raw_result.grounding.input_overlap,
            "promotion_semantic_overlap": raw_result.grounding.semantic_overlap,
            "final_input_disjoint": final_input_disjoint,
            "final_semantic_disjoint": final_semantic_disjoint,
        },
        "learning": {
            "promoted": raw_result.promoted,
            "rejection_reasons": promotion_iteration.rejection_reasons,
            "verified_training_traces": promotion_iteration.verified_training_traces,
            "decision_cases": promotion_iteration.decision_cases,
            "corpus_records": len(raw_result.learning.corpus.records),
            "promotion_expansion_reduction": (
                promotion_iteration.expansion_reduction
            ),
        },
        "transfer": {
            "baseline": baseline_metrics.to_dict(),
            "candidate": candidate_metrics.to_dict(),
            "restored": (
                restored_metrics.to_dict() if restored_metrics is not None else None
            ),
            "per_domain_expansion_reduction": per_domain_reduction,
            "baseline_tasks": tuple(asdict(item) for item in baseline_tasks),
            "candidate_tasks": tuple(asdict(item) for item in candidate_tasks),
        },
        "artifact": {
            "bytes": artifact_bytes,
            "sha256": (
                active_candidate.artifact_sha256
                if active_candidate is not None
                else None
            ),
            "round_trip_verified": artifact_round_trip,
            "runtime": restored_runtime_name,
            "candidate_only": True,
            "persisted_path": persisted_artifact,
        },
        "resources": {
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": peak_after,
            "training_additional_peak_rss_bytes": max(
                0, peak_after - peak_before
            ),
            "cpu_inference": inference_resources,
            "additional_peak_rss_bytes": int(
                inference_resources.get("additional_peak_rss_bytes", 0)
            ),
        },
        "gates": {**gates, "all_passed": all_gates_passed},
    }


def _measure_portable_inference(
    candidate: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    if candidate is None:
        return {"available": False, "error": "no promoted candidate"}
    with tempfile.TemporaryDirectory() as directory:
        artifact = Path(directory) / f"controller{candidate.artifact_suffix}"
        artifact.write_bytes(candidate.artifact)
        command = [
            sys.executable,
            str(ROOT / "tools" / "eval" / "measure_raw_controller_inference.py"),
            "--artifact",
            str(artifact),
            "--kind",
            candidate.kind,
            "--seed",
            str(seed),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=60.0,
                check=False,
            )
            payload = json.loads(completed.stdout.strip())
        except (json.JSONDecodeError, OSError, subprocess.TimeoutExpired) as exc:
            return {"available": False, "error": str(exc)}
    if completed.returncode != 0:
        return {
            "available": False,
            "error": payload.get("error", completed.stderr.strip()),
        }
    return dict(payload)


def _persist_candidate_artifact(
    candidate: Any,
    output: str | Path | None,
    *,
    allowed: bool,
) -> str | None:
    if output is None or candidate is None or not allowed:
        return None
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(candidate.artifact)
    temporary.replace(destination)
    return str(destination.resolve())


def _controller_learner(
    profile: str,
    *,
    epochs: int,
    learning_rate: float,
    device: str,
    seed: int,
) -> PolicyLearner:
    if profile == "sparse":
        return StructuralPolicyLearner()
    if profile == "recurrent-diagnostic":
        config = TinyControllerConfig.diagnostic()
    elif profile == "recurrent-compact":
        config = TinyControllerConfig.compact()
    elif profile == "recurrent-full":
        config = TinyControllerConfig()
    else:
        raise ValueError(
            "controller_profile must be sparse, recurrent-diagnostic, "
            "recurrent-compact, or recurrent-full"
        )
    return TinyControllerPolicyLearner(
        config=config,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
    )


def _raw_language_learning_split() -> tuple[
    tuple[RawLearningExample, ...],
    tuple[RawLearningExample, ...],
]:
    training_texts = (
        "Goal: deploy; Requires: tests, approval; "
        "Satisfied: tests; Satisfied: approval",
        "Goal: publish; Requires: review, license; "
        "Satisfied: review; Satisfied: license",
        "Goal: archive; Requires: checksum, backup; "
        "Satisfied: checksum; Satisfied: backup",
    )
    training = tuple(
        RawLearningExample(
            f"language-train-{index}",
            TypedDomainRequest("language", text, "shadow"),
            split=LearningSplit.TRAIN,
            capability="raw-requirement-reasoning",
            structure_key="raw-language-two-requirements",
        )
        for index, text in enumerate(training_texts)
    )
    heldout = (
        RawLearningExample(
            "language-promotion-positive",
            TypedDomainRequest(
                "language",
                "Goal: launch; Requires: audit, signoff; "
                "Satisfied: audit; Satisfied: signoff",
                "shadow",
            ),
            split=LearningSplit.HELDOUT,
            capability="raw-requirement-reasoning",
            structure_key="raw-language-heldout-positive",
        ),
        RawLearningExample(
            "language-promotion-negative",
            TypedDomainRequest(
                "language",
                "Goal: release; Requires: audit, signoff; Satisfied: audit",
                "shadow",
            ),
            expected_solved=False,
            split=LearningSplit.HELDOUT,
            capability="raw-missing-premise-control",
            structure_key="raw-language-heldout-negative",
        ),
    )
    return training, heldout


def _raw_math_vision_transfer_examples() -> tuple[RawLearningExample, ...]:
    return (
        RawLearningExample(
            "math-transfer-positive",
            TypedDomainRequest("math", "3*x + 2 = 17", "shadow"),
            split=LearningSplit.HELDOUT,
            capability="raw-linear-equation",
            structure_key="raw-math-positive",
        ),
        RawLearningExample(
            "math-transfer-negative",
            TypedDomainRequest("math", "2 + 2 == 5", "shadow"),
            expected_solved=False,
            split=LearningSplit.HELDOUT,
            capability="raw-false-comparison-control",
            structure_key="raw-math-negative",
        ),
        RawLearningExample(
            "vision-transfer-positive",
            TypedDomainRequest(
                "vision",
                RasterVisionProblem(
                    _square_image(),
                    (VisionPropertyGoal("SQUARE", "red"),),
                ),
                "shadow",
            ),
            split=LearningSplit.HELDOUT,
            capability="raw-pixel-shape",
            structure_key="raw-vision-positive",
        ),
        RawLearningExample(
            "vision-transfer-negative",
            TypedDomainRequest(
                "vision",
                RasterVisionProblem(
                    _incomplete_square_image(),
                    (VisionPropertyGoal("SQUARE", "red"),),
                ),
                "shadow",
            ),
            expected_solved=False,
            split=LearningSplit.HELDOUT,
            capability="raw-pixel-shape-control",
            structure_key="raw-vision-negative",
        ),
    )


def _square_image() -> RasterImage:
    return RasterImage.from_rows(
        (
            [WHITE] * 7,
            [WHITE, WHITE, RED, RED, RED, WHITE, WHITE],
            [WHITE, WHITE, RED, RED, RED, WHITE, WHITE],
            [WHITE, WHITE, RED, RED, RED, WHITE, WHITE],
            [WHITE] * 7,
        )
    )


def _incomplete_square_image() -> RasterImage:
    return RasterImage.from_rows(
        (
            [WHITE] * 6,
            [WHITE, RED, RED, WHITE, WHITE, WHITE],
            [WHITE, RED, WHITE, WHITE, WHITE, WHITE],
            [WHITE] * 6,
        )
    )


def _evaluate_tasks(
    tasks: Sequence[LearningTask],
    policy: ActionPolicy | None,
) -> tuple[
    LearningMetrics,
    tuple[TaskEvaluation, ...],
    dict[str, SolveResult],
]:
    evaluations: list[TaskEvaluation] = []
    results: dict[str, SolveResult] = {}
    for task in tasks:
        started = process_time()
        result = OperatorKernel(task.instance.registry).solve(
            task.instance.state,
            task.instance.goals,
            policy=policy,
            budget=SolveBudget(
                max_steps=32,
                max_expansions=20_000,
                timeout_seconds=10.0,
            ),
        )
        results[task.task_id] = result
        evaluations.append(
            task_evaluation_from_result(
                task,
                result,
                process_time() - started,
            )
        )
    resolved = tuple(evaluations)
    return summarize_task_evaluations(resolved), resolved, results


def _fingerprints(
    *batches: GroundedLearningBatch,
) -> tuple[set[str], set[str]]:
    return (
        {
            fingerprint
            for batch in batches
            for _identifier, fingerprint in batch.input_fingerprints
        },
        {
            fingerprint
            for batch in batches
            for _identifier, fingerprint in batch.semantic_fingerprints
        },
    )


def _domain_expansion_reduction(
    baseline: LearningMetrics,
    candidate: LearningMetrics,
    domain: str,
) -> float:
    baseline_domain = baseline.for_domain(domain)
    candidate_domain = candidate.for_domain(domain)
    if baseline_domain.positive_expansions == 0:
        return (
            0.0
            if candidate_domain.positive_expansions == 0
            else -1.0
        )
    return (
        baseline_domain.positive_expansions
        - candidate_domain.positive_expansions
    ) / baseline_domain.positive_expansions


def _all_successes_replay(
    tasks: Sequence[LearningTask],
    results: dict[str, SolveResult],
) -> bool:
    for task in tasks:
        result = results[task.task_id]
        if not result.success:
            continue
        if not result.verified:
            return False
        if any(
            step.action.operator.name not in task.instance.registry.operators
            for step in result.proof
        ):
            return False
        if not OperatorKernel(task.instance.registry).replay(
            task.instance.state,
            task.instance.goals,
            result.proof,
        ).verified:
            return False
    return True


def _same_verified_behavior(
    tasks: Sequence[LearningTask],
    first: dict[str, SolveResult],
    second: dict[str, SolveResult],
) -> bool:
    return all(
        (
            first[task.task_id].success,
            first[task.task_id].verified,
            first[task.task_id].expansions,
            tuple(
                step.action.canonical_key()
                for step in first[task.task_id].proof
            ),
        )
        == (
            second[task.task_id].success,
            second[task.task_id].verified,
            second[task.task_id].expansions,
            tuple(
                step.action.canonical_key()
                for step in second[task.task_id].proof
            ),
        )
        for task in tasks
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small verifier-gated controller on raw language and test "
            "untouched raw math and pixel transfer"
        )
    )
    parser.add_argument(
        "--controller-profile",
        choices=CONTROLLER_PROFILES,
        default="recurrent-diagnostic",
    )
    parser.add_argument("--controller-epochs", type=int, default=5)
    parser.add_argument("--controller-learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--min-transfer-expansion-reduction",
        type=float,
        default=0.30,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--artifact-output",
        type=Path,
        help="write the candidate controller only when every evaluation gate passes",
    )
    args = parser.parse_args()
    report = evaluate_raw_grounded_self_learning(
        controller_profile=args.controller_profile,
        controller_epochs=args.controller_epochs,
        controller_learning_rate=args.controller_learning_rate,
        device=args.device,
        seed=args.seed,
        min_transfer_expansion_reduction=(
            args.min_transfer_expansion_reduction
        ),
        artifact_output=args.artifact_output,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["gates"]["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

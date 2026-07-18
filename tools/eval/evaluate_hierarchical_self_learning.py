from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_low_resource_transfer import _peak_rss_bytes
from semop.kernel import (
    HierarchicalLearningBudget,
    HierarchicalOperatorBrain,
    HierarchicalSelfLearningLoop,
    OperatorKernel,
    PolicyLearner,
    generate_hierarchical_brain_transfer_split,
    hierarchical_learning_tasks_from_curriculum,
)
from semop.tiny_controller import (
    StructuralPolicyLearner,
    TinyControllerConfig,
    TinyControllerPolicyLearner,
)


CONTROLLER_PROFILES = (
    "sparse",
    "recurrent-diagnostic",
    "recurrent-full",
)


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
    elif profile == "recurrent-full":
        config = TinyControllerConfig()
    else:
        raise ValueError(
            "controller_profile must be sparse, recurrent-diagnostic, "
            "or recurrent-full"
        )
    return TinyControllerPolicyLearner(
        config=config,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
    )


def evaluate_hierarchical_self_learning(
    *,
    examples_per_domain: int = 3,
    validation_per_domain: int = 1,
    heldout_per_domain: int = 1,
    seed: int = 23,
    controller_domain: str = "language",
    controller_profile: str = "sparse",
    controller_epochs: int = 10,
    controller_learning_rate: float = 1e-3,
    device: str = "cpu",
    min_joint_expansion_reduction: float = 0.50,
) -> dict[str, Any]:
    split = generate_hierarchical_brain_transfer_split(
        examples_per_domain,
        validation_per_domain=validation_per_domain,
        heldout_per_domain=heldout_per_domain,
        seed=seed,
    )
    tasks = hierarchical_learning_tasks_from_curriculum(
        split,
        controller_domain=controller_domain,
        namespace=f"hierarchical-{seed}",
    )
    controller_training = tasks.controller_training
    controller_heldout = tasks.controller_heldout
    macro_training = tasks.macro_training
    macro_validation = tasks.macro_validation
    macro_heldout = tasks.macro_heldout
    joint_positive = tasks.joint_positive
    joint_negative = tasks.joint_negative
    learner = _controller_learner(
        controller_profile,
        epochs=controller_epochs,
        learning_rate=controller_learning_rate,
        device=device,
        seed=seed,
    )

    peak_before = _peak_rss_bytes()
    started = perf_counter()
    result = HierarchicalSelfLearningLoop(
        learner=learner,
        budget=HierarchicalLearningBudget(
            min_joint_expansion_reduction=min_joint_expansion_reduction,
        )
    ).run_tasks(tasks)
    wall_seconds = perf_counter() - started
    peak_after = _peak_rss_bytes()

    controller_names = {
        str(task.instance.metadata["completion_operator"])
        for task in controller_training + controller_heldout
    }
    heldout_names = {
        str(problem.instance.metadata["completion_operator"])
        for problem in split.joint_heldout
    }
    positive_groups = (
        split.controller_training,
        split.controller_heldout,
        split.macro_training,
        split.macro_validation,
        split.macro_heldout,
        split.joint_heldout,
    )
    vision_digest_sets = tuple(
        {
            str(problem.instance.metadata["image_digest"])
            for problem in group
            if problem.domain == "vision"
        }
        for group in positive_groups
    )
    vision_inputs_disjoint = all(
        left.isdisjoint(right)
        for left_index, left in enumerate(vision_digest_sets)
        for right in vision_digest_sets[left_index + 1 :]
    )

    primitive_replay = True
    unseen_completion_used = True
    macro_is_indirect = True
    brain = result.active_brain
    for task in joint_positive:
        solved = result.joint.results[task.task_id]
        primitive_replay = primitive_replay and solved.success and solved.verified
        primitive_replay = primitive_replay and all(
            step.action.operator.name in task.instance.registry.operators
            and not step.action.operator.name.startswith("macro_")
            for step in solved.proof
        )
        primitive_replay = primitive_replay and OperatorKernel(
            task.instance.registry
        ).replay(
            task.instance.state,
            task.instance.goals,
            solved.proof,
        ).verified
        unseen_completion_used = unseen_completion_used and any(
            step.action.operator.name in heldout_names
            and step.action.operator.name not in controller_names
            and step.action.operator.family == "verify"
            for step in solved.proof
        )
        if brain is not None:
            final_signature = (
                f"{task.instance.goals[0].atom.predicate.name}("
                + ",".join(
                    argument.type.name
                    for argument in task.instance.goals[0].atom.arguments
                )
                + ")"
            )
            policy = brain.policy_for(task.instance.registry)
            macro_is_indirect = macro_is_indirect and bool(
                policy.activation.programs
            ) and all(
                final_signature not in program.effect_signature
                for program in policy.activation.programs
            )

    artifact_round_trip = False
    artifact_bytes = 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "hierarchical-brain.json"
        persisted = HierarchicalSelfLearningLoop.persist_promoted(result, path)
        if persisted is not None and brain is not None:
            artifact_bytes = path.stat().st_size
            restored = HierarchicalOperatorBrain.load(
                path,
                learner,
            )
            artifact_round_trip = restored.to_artifact() == brain.to_artifact()

    ablations = {
        "deterministic": result.deterministic.metrics.to_dict(),
        "controller_only": result.controller_only.metrics.to_dict(),
        "macro_only": result.macro_only.metrics.to_dict(),
        "joint": result.joint.metrics.to_dict(),
    }
    controller_weights = dict(
        result.active_brain.base_policy.weights
        if result.active_brain is not None
        and hasattr(result.active_brain.base_policy, "weights")
        else ()
    )
    active_candidate = result.controller_learning.active_candidate
    controller_training_updates = (
        active_candidate.training_updates if active_candidate is not None else 0
    )
    sparse_family_signal = (
        controller_weights.get("operator:family:verify", 0.0) > 0.0
        and controller_weights.get("operator:family:search", 0.0) < 0.0
    )
    zero_shot_controller_domains = tuple(
        sorted(
            {task.domain for task in joint_positive}
            - {task.domain for task in controller_training}
        )
    )
    joint_expansions = result.joint.metrics.positive_expansions
    gates = {
        "controller_component_promoted": (
            result.controller_learning.active_candidate is not None
        ),
        "macro_component_promoted": result.macro_learning.promoted,
        "joint_brain_promoted": result.promoted,
        "completion_operator_names_held_out": controller_names.isdisjoint(
            heldout_names
        ),
        "controller_leaves_two_domains_out": len(zero_shot_controller_domains)
        == 2,
        "controller_received_verified_updates": controller_training_updates > 0,
        "shared_operator_signal_learned": (
            sparse_family_signal
            if controller_profile == "sparse"
            else controller_training_updates > 0
        ),
        "shared_verify_family_transfers": unseen_completion_used
        and all(
            result.controller_only.metrics.for_domain(domain).positive_expansions
            < result.deterministic.metrics.for_domain(domain).positive_expansions
            for domain in zero_shot_controller_domains
        ),
        "vision_inputs_disjoint": vision_inputs_disjoint,
        "macro_reused_as_indirect_subprogram": macro_is_indirect,
        "joint_beats_controller_only": joint_expansions
        < result.controller_only.metrics.positive_expansions,
        "joint_beats_macro_only": joint_expansions
        < result.macro_only.metrics.positive_expansions,
        "minimum_joint_expansion_reduction": (
            result.joint_expansion_reduction
            >= min_joint_expansion_reduction
        ),
        "all_three_domains_active": set(result.active_domains)
        == {"language", "math", "vision"},
        "proof_soundness_100_percent": result.joint.metrics.proof_soundness
        == 1.0,
        "false_positives_zero": result.joint.metrics.false_positives == 0,
        "primitive_full_replay_verified": primitive_replay,
        "artifact_round_trip_verified": artifact_round_trip,
        "parameters_under_15m": brain is not None
        and brain.parameter_count <= 15_000_000,
        "artifact_under_64mb": artifact_bytes <= 64 * 1024 * 1024,
        "joint_p95_under_10s": result.joint.metrics.p95_cpu_seconds <= 10.0,
        "additional_peak_rss_under_512mb": max(0, peak_after - peak_before)
        <= 512 * 1024 * 1024,
    }
    return {
        "schema_version": 1,
        "suite": "typed-hierarchical-self-learning",
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
        },
        "split": {
            "controller_training_pool": len(split.controller_training),
            "controller_training_selected": len(controller_training),
            "controller_training_domain": controller_domain,
            "controller_heldout_positive": sum(
                task.expected_solved for task in controller_heldout
            ),
            "controller_heldout_negative": sum(
                not task.expected_solved for task in controller_heldout
            ),
            "zero_shot_controller_domains": zero_shot_controller_domains,
            "macro_training": len(macro_training),
            "macro_validation": len(macro_validation),
            "macro_heldout_positive": len(split.macro_heldout),
            "macro_heldout_negative": len(split.macro_negative_controls),
            "joint_heldout_positive": len(joint_positive),
            "joint_heldout_negative": len(joint_negative),
            "controller_completion_operators": tuple(sorted(controller_names)),
            "heldout_completion_operators": tuple(sorted(heldout_names)),
            "vision_image_digests": tuple(
                tuple(sorted(values)) for values in vision_digest_sets
            ),
        },
        "learning": {
            "promoted": result.promoted,
            "rejection_reasons": result.rejection_reasons,
            "controller_kind": (
                active_candidate.kind
                if active_candidate is not None
                else None
            ),
            "controller_parameters": (
                brain.parameter_count if brain is not None else 0
            ),
            "controller_family_weights": {
                name: value
                for name, value in sorted(controller_weights.items())
                if name.startswith("operator:family:")
            },
            "controller_training_updates": controller_training_updates,
            "controller_diagnostics": (
                active_candidate.diagnostics if active_candidate is not None else ()
            ),
            "active_macros": brain.macro_count if brain is not None else 0,
            "active_domains": result.active_domains,
        },
        "ablations": ablations,
        "ab": {
            "positive_expansions": {
                name: metrics["positive_expansions"]
                for name, metrics in ablations.items()
            },
            "joint_expansion_reduction": result.joint_expansion_reduction,
        },
        "artifact": {
            "format_version": HierarchicalOperatorBrain.FORMAT_VERSION,
            "bytes": artifact_bytes,
            "sha256": brain.artifact_sha256 if brain is not None else None,
            "round_trip_verified": artifact_round_trip,
        },
        "resources": {
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": peak_after,
            "additional_peak_rss_bytes": max(0, peak_after - peak_before),
        },
        "gates": {**gates, "all_passed": all(gates.values())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a shared controller plus verified procedural memory"
    )
    parser.add_argument("--examples-per-domain", type=int, default=3)
    parser.add_argument("--validation-per-domain", type=int, default=1)
    parser.add_argument("--heldout-per-domain", type=int, default=1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--controller-domain",
        choices=("language", "math", "vision"),
        default="language",
    )
    parser.add_argument(
        "--controller-profile",
        choices=CONTROLLER_PROFILES,
        default="sparse",
    )
    parser.add_argument("--controller-epochs", type=int, default=10)
    parser.add_argument(
        "--controller-learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--min-joint-expansion-reduction",
        type=float,
        default=0.50,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_hierarchical_self_learning(
        examples_per_domain=args.examples_per_domain,
        validation_per_domain=args.validation_per_domain,
        heldout_per_domain=args.heldout_per_domain,
        seed=args.seed,
        controller_domain=args.controller_domain,
        controller_profile=args.controller_profile,
        controller_epochs=args.controller_epochs,
        controller_learning_rate=args.controller_learning_rate,
        device=args.device,
        min_joint_expansion_reduction=args.min_joint_expansion_reduction,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["gates"]["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

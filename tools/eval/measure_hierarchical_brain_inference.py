from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EVAL = ROOT / "tools" / "eval"
for path in (SRC, EVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_hierarchical_self_learning import _controller_learner
from evaluate_low_resource_transfer import _peak_rss_bytes
from semop.kernel import (
    HierarchicalOperatorBrain,
    OperatorKernel,
    generate_hierarchical_brain_transfer_split,
    hierarchical_learning_tasks_from_curriculum,
)


def measure(
    artifact: Path,
    *,
    profile: str,
    seed: int,
    examples_per_domain: int,
    validation_per_domain: int,
    heldout_per_domain: int,
    controller_domain: str,
) -> dict[str, object]:
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
    learner = _controller_learner(
        profile,
        epochs=1,
        learning_rate=1e-3,
        device="cpu",
        seed=seed,
    )
    baseline_peak = _peak_rss_bytes()
    brain = HierarchicalOperatorBrain.load(artifact, learner)
    elapsed: list[float] = []
    verified = True
    for task in tasks.joint_heldout:
        started = perf_counter()
        result = brain.solve(task.instance)
        elapsed.append(perf_counter() - started)
        if task.expected_solved:
            verified = verified and result.success and result.verified
            verified = verified and OperatorKernel(
                task.instance.registry
            ).replay(
                task.instance.state,
                task.instance.goals,
                result.proof,
            ).verified
        else:
            verified = verified and not result.success
    peak = _peak_rss_bytes()
    ordered = sorted(elapsed)
    p95 = ordered[max(0, len(ordered) - 1)] if ordered else 0.0
    return {
        "available": True,
        "verified": verified,
        "runtime": type(brain.base_policy).__name__,
        "baseline_peak_rss_bytes": baseline_peak,
        "peak_rss_bytes": peak,
        "additional_peak_rss_bytes": max(0, peak - baseline_peak),
        "p95_cpu_seconds": p95,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure a SemOp hierarchical brain in a fresh CPU process"
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--examples-per-domain", type=int, required=True)
    parser.add_argument("--validation-per-domain", type=int, required=True)
    parser.add_argument("--heldout-per-domain", type=int, required=True)
    parser.add_argument("--controller-domain", required=True)
    args = parser.parse_args()
    try:
        result = measure(
            args.artifact,
            profile=args.profile,
            seed=args.seed,
            examples_per_domain=args.examples_per_domain,
            validation_per_domain=args.validation_per_domain,
            heldout_per_domain=args.heldout_per_domain,
            controller_domain=args.controller_domain,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"available": False, "error": str(exc)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

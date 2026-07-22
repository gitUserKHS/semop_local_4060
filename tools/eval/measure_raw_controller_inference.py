from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EVAL = ROOT / "tools" / "eval"
for path in (SRC, EVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_low_resource_transfer import _peak_rss_bytes
from evaluate_raw_grounded_self_learning import (
    _all_successes_replay,
    _evaluate_tasks,
    _raw_math_vision_transfer_examples,
)
from semop.kernel import RawExperienceGrounder
from semop.tiny_controller import NumpyTinyController, StructuralLinearPolicy


def measure(artifact: Path, *, kind: str, seed: int) -> dict[str, object]:
    batch = RawExperienceGrounder(hard_negatives_per_example=4).ground(
        _raw_math_vision_transfer_examples(),
        namespace=f"raw-transfer-{seed}",
    )
    tasks = batch.require_complete()
    baseline_peak = _peak_rss_bytes()
    payload = artifact.read_bytes()
    if kind == "tiny-controller-v6":
        policy = NumpyTinyController.from_artifact(payload)
    elif kind == StructuralLinearPolicy.KIND:
        policy = StructuralLinearPolicy.from_artifact(payload)
    else:
        raise ValueError(f"unsupported controller artifact kind: {kind}")
    metrics, _evaluations, results = _evaluate_tasks(tasks, policy)
    peak = _peak_rss_bytes()
    return {
        "available": True,
        "verified": (
            metrics.proof_soundness == 1.0
            and metrics.false_positives == 0
            and _all_successes_replay(tasks, results)
        ),
        "runtime": type(policy).__name__,
        "baseline_peak_rss_bytes": baseline_peak,
        "peak_rss_bytes": peak,
        "additional_peak_rss_bytes": max(0, peak - baseline_peak),
        "p95_cpu_seconds": metrics.p95_cpu_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure a SemOp controller in a fresh NumPy inference process"
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    try:
        result = measure(args.artifact, kind=args.kind, seed=args.seed)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"available": False, "error": str(exc)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    OperatorKernel,
    SolveBudget,
    TraceCorpus,
    build_decision_training_cases,
    generate_symbolic_curriculum,
    hard_negative_records,
    symbolic_curriculum_domains,
)
from semop.tiny_controller import ControllerFeatureProfile, TinyControllerConfig
from semop.tiny_controller.training import (
    TorchTinyController,
    encode_training_example,
    train_controller,
)


def train(
    output: Path,
    *,
    examples_per_domain: int = 2,
    epochs: int = 1,
    seed: int = 0,
    device: str = "cpu",
    debug_small: bool = False,
    curriculum: str = "language-math-vision",
    domains: Sequence[str] | None = None,
    feature_profile: ControllerFeatureProfile | str = (
        ControllerFeatureProfile.TYPED_STRUCTURE
    ),
) -> dict:
    import torch

    torch.manual_seed(seed)
    base_config = (
        TinyControllerConfig.diagnostic()
        if debug_small
        else TinyControllerConfig()
    )
    config = replace(
        base_config,
        feature_profile=ControllerFeatureProfile(feature_profile),
    )
    corpus = TraceCorpus()
    encoded = []
    verified_count = 0
    verified_by_domain: dict[str, int] = {}
    decision_count = 0
    hard_negative_count = 0
    started = perf_counter()
    available_domains = symbolic_curriculum_domains(curriculum)
    requested_domains = (
        available_domains if domains is None else tuple(dict.fromkeys(domains))
    )
    for synthetic in generate_symbolic_curriculum(
        examples_per_domain,
        seed=seed,
        curriculum=curriculum,
        domains=requested_domains,
    ):
        instance = synthetic.instance
        kernel = OperatorKernel(instance.registry)
        result = kernel.solve(
            instance.state,
            instance.goals,
            budget=SolveBudget(max_steps=6, max_expansions=20_000),
        )
        if not result.success or not result.verified or len(result.proof) > 6:
            continue
        cases = build_decision_training_cases(
            kernel, result, negatives_per_positive=4
        )
        negatives = hard_negative_records(cases)
        hard_negative_count += len(negatives)
        corpus.add_result(
            synthetic.problem_id,
            synthetic.domain,
            result,
            source="synthetic",
            hard_negatives=negatives,
        )
        verified_count += 1
        verified_by_domain[synthetic.domain] = (
            verified_by_domain.get(synthetic.domain, 0) + 1
        )
        proof_length = max(1, len(result.proof))
        for case in cases:
            remaining = proof_length - case.step + 1
            encoded.append(
                encode_training_example(
                    case.state,
                    case.goals,
                    case.actions,
                    target_action=case.target_action,
                    target_halt=False,
                    target_value=max(0.0, 1.0 - remaining / 6.0),
                    config=config,
                )
            )
            decision_count += 1
        encoded.append(
            encode_training_example(
                result.final_state,
                tuple(outcome.goal for outcome in result.goals),
                (),
                target_action=-1,
                target_halt=True,
                target_value=1.0,
                config=config,
            )
        )

    if not encoded:
        raise RuntimeError("curriculum produced no verified training examples")
    missing_domains = sorted(set(requested_domains) - set(verified_by_domain))
    if missing_domains:
        raise RuntimeError(
            "curriculum produced no verified trace for requested domains: "
            + ", ".join(missing_domains)
        )
    model = TorchTinyController(config)
    history = train_controller(
        model,
        encoded,
        epochs=epochs,
        device=device,
    )
    output = output.with_suffix(".npz")
    model.export_numpy(output)
    trace_path = output.with_suffix(".traces.jsonl")
    corpus.save_jsonl(trace_path)
    summary = {
        "schema_version": 1,
        "output": str(output),
        "trace_path": str(trace_path),
        "debug_small": debug_small,
        "seed": seed,
        "device": device,
        "curriculum": curriculum,
        "feature_profile": config.feature_profile.value,
        "curriculum_domains": list(available_domains),
        "requested_domains": list(requested_domains),
        "trained_domains": sorted(verified_by_domain),
        "held_out_domains": sorted(set(available_domains) - set(requested_domains)),
        "training_scope": (
            "all_domains"
            if set(requested_domains) == set(available_domains)
            else "domain_holdout"
        ),
        "verified_traces_by_domain": dict(sorted(verified_by_domain.items())),
        "examples_per_domain": examples_per_domain,
        "reviewed_examples": 0,
        "reviewed_examples_per_domain": 0,
        "verified_synthetic_traces": verified_count,
        "decision_examples": decision_count,
        "hard_negative_records": hard_negative_count,
        "terminal_halt_examples": verified_count,
        "epochs": epochs,
        "loss_history": history,
        "parameter_count": model.parameter_count,
        "artifact_bytes": output.stat().st_size,
        "elapsed_seconds": perf_counter() - started,
        "caps": {
            "synthetic_per_domain": 5_000,
            "operator_depth": 6,
            "hard_negatives_per_positive": 4,
            "parameters": 15_000_000,
            "artifact_bytes": 64 * 1024 * 1024,
        },
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the verifier-first SemOp tiny controller"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-domain", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--curriculum",
        choices=(
            "operator-v1",
            "language-math-vision",
            "language-math-vision-composed",
        ),
        default="language-math-vision",
    )
    parser.add_argument(
        "--debug-small",
        action="store_true",
        help="use a tiny architecture for pipeline smoke tests only",
    )
    parser.add_argument(
        "--feature-profile",
        choices=tuple(item.value for item in ControllerFeatureProfile),
        default=ControllerFeatureProfile.TYPED_STRUCTURE.value,
        help="controller input identity profile",
    )
    parser.add_argument(
        "--domain",
        action="append",
        choices=(
            "geometry",
            "hidden_premise",
            "grid",
            "language",
            "math",
            "vision",
            "composed",
        ),
        help="train only this curriculum domain; repeat for multiple domains",
    )
    args = parser.parse_args()
    summary = train(
        args.output,
        examples_per_domain=args.examples_per_domain,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        debug_small=args.debug_small,
        curriculum=args.curriculum,
        domains=args.domain,
        feature_profile=args.feature_profile,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

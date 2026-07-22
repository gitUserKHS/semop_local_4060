from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.artifact_promotion import artifact_sha256
from semop.semantic_development_evaluation import (
    SemanticDevelopmentMetrics,
    development_candidate_passes,
    evaluate_generated_semantic_model,
)
from semop.semantic_distillation import (
    DistillationDecision,
    SemanticDistillationCorpus,
    SemanticDistillationRecord,
)
from semop.semantic_student_evaluation import (
    LocalSemanticRunner,
    SemanticRuntimeObservation,
    semantic_completion_signature,
)
from semop.semantic_models import QWEN_ECONOMY
from semop.verified_semantic_curriculum import GENERATOR_VERSION


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the pinned 0.8B base and an optional LoRA on generator-known "
            "heldout Korean arithmetic prompts. Results are never promotion evidence."
        )
    )
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help="reuse a previous digest-matched baseline report instead of rerunning 0.8B",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/verified-semantic-development.json"),
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        corpus = SemanticDistillationCorpus.load_jsonl(args.corpus)
        positives = tuple(
            record
            for record in corpus.records
            if record.decision is not DistillationDecision.HARD_NEGATIVE
        )
        if not positives:
            raise ValueError("development evaluation requires positive records")
        probe = _observed_edge_probe(positives[0].data_split)
        corpus_digest = _file_digest(args.corpus)
        if args.baseline_report is None:
            baseline_runner = LocalSemanticRunner(
                model_id=QWEN_ECONOMY.model_id,
                device=args.device,
            )
            try:
                baseline_observations = _run_unique(baseline_runner, positives)
                baseline_probe = baseline_runner.run(probe)
                baseline_model_id = baseline_runner.model_id
            finally:
                baseline_runner.unload()
            baseline = evaluate_generated_semantic_model(
                corpus,
                baseline_observations,
                model_id=baseline_model_id,
            )
            baseline_probe_match = _probe_matches(probe, baseline_probe)
        else:
            baseline, baseline_probe_match = _load_baseline_report(
                args.baseline_report,
                corpus_digest=corpus_digest,
                expected_split=positives[0].data_split,
            )

        payload: dict[str, Any] = {
            "schema_version": "semop.verified-semantic-development.v1",
            "promotion_eligible": False,
            "corpus_sha256": corpus_digest,
            "baseline": baseline.to_dict(),
            "observed_edge_probe": {
                "prompt": probe.prompt,
                "baseline_match": baseline_probe_match,
            },
        }
        exit_code = 0
        if args.adapter is not None:
            _validate_candidate(args.adapter)
            candidate_digest = artifact_sha256(args.adapter)
            candidate_runner = LocalSemanticRunner(
                model_id=QWEN_ECONOMY.model_id,
                device=args.device,
                adapter_path=str(args.adapter.resolve()),
                adapter_sha256=candidate_digest,
            )
            try:
                candidate_observations = _run_unique(candidate_runner, positives)
                candidate_probe = candidate_runner.run(probe)
                candidate_model_id = candidate_runner.model_id
            finally:
                candidate_runner.unload()
            candidate = evaluate_generated_semantic_model(
                corpus,
                candidate_observations,
                model_id=candidate_model_id,
            )
            candidate_probe_match = _probe_matches(probe, candidate_probe)
            passed = development_candidate_passes(
                baseline,
                candidate,
                observed_probe_match=candidate_probe_match,
            )
            payload.update(
                {
                    "candidate_sha256": candidate_digest,
                    "candidate": candidate.to_dict(),
                    "observed_edge_probe": {
                        **payload["observed_edge_probe"],
                        "candidate_match": candidate_probe_match,
                    },
                    "development_passed": passed,
                    "candidate_action": (
                        "retain_for_human_sealed_review"
                        if passed
                        else "discard_or_retrain"
                    ),
                }
            )
            exit_code = 0 if passed else 2
        _atomic_json(args.output, payload)
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"verified semantic development evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


def _run_unique(
    runner: LocalSemanticRunner,
    positives: Sequence[SemanticDistillationRecord],
) -> tuple[SemanticRuntimeObservation, ...]:
    return tuple(runner.run(record) for record in positives)


def _observed_edge_probe(split: str) -> SemanticDistillationRecord:
    completion = json.dumps(
        {
            "domain": "math",
            "confidence": 1.0,
            "operator_program": ["DECOMPOSE", "INFER", "VERIFY", "EXPLAIN"],
            "payload": {"expression": "12 - 5"},
            "answer": "7",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return SemanticDistillationRecord(
        record_id="",
        domain="math",
        prompt="사과가 12개 있었는데 5개 먹었어. 몇 개 남았어?",
        completion=completion,
        decision=DistillationDecision.ACCEPTED,
        proposal_fingerprint=sha256(
            (GENERATOR_VERSION + completion).encode("utf-8")
        ).hexdigest(),
        replay_verified=True,
        data_split=split,
        source=f"{GENERATOR_VERSION}:observed_edge_probe:000",
    )


def _probe_matches(
    probe: SemanticDistillationRecord,
    observation: SemanticRuntimeObservation,
) -> bool:
    return bool(
        observation.model_used
        and observation.proof_eligible
        and observation.replay_verified
        and semantic_completion_signature(observation.completion_json)
        == semantic_completion_signature(probe.completion)
    )


def _validate_candidate(adapter: Path) -> None:
    summary = json.loads(
        (adapter / "semop_training_summary.json").read_text(encoding="utf-8")
    )
    if not isinstance(summary, dict) or summary.get("candidate_only") is not True:
        raise ValueError("adapter is not a SemOp candidate")
    if summary.get("student_model_id") != QWEN_ECONOMY.model_id:
        raise ValueError("adapter uses the wrong student base model")
    if summary.get("student_model_revision") != QWEN_ECONOMY.revision:
        raise ValueError("adapter uses the wrong student revision")


def _load_baseline_report(
    path: Path,
    *,
    corpus_digest: str,
    expected_split: str,
) -> tuple[SemanticDevelopmentMetrics, bool]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("baseline report must be a JSON object")
    if value.get("schema_version") != "semop.verified-semantic-development.v1":
        raise ValueError("baseline report has an unsupported schema")
    if value.get("promotion_eligible") is not False:
        raise ValueError("baseline report is not development-only")
    if value.get("corpus_sha256") != corpus_digest:
        raise ValueError("baseline report corpus digest does not match")
    baseline_value = value.get("baseline")
    probe_value = value.get("observed_edge_probe")
    if not isinstance(baseline_value, dict) or not isinstance(probe_value, dict):
        raise ValueError("baseline report lacks metrics or probe evidence")
    baseline = SemanticDevelopmentMetrics.from_dict(baseline_value)
    if baseline.split != expected_split:
        raise ValueError("baseline report split does not match")
    return baseline, bool(probe_value.get("baseline_match"))


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())

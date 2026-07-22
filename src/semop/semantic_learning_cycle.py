from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence
import unicodedata

from .artifact_promotion import (
    ArtifactCandidate,
    ArtifactPromotionStore,
    artifact_sha256,
)
from .semantic_distillation import (
    SemanticDistillationCorpus,
    SemanticDistillationRecord,
)
from .semantic_experience import (
    SemanticTraceSplit,
    SemanticTraceStats,
    SemanticTraceStore,
)
from .semantic_student_evaluation import (
    SemanticStudentEvaluation,
    run_local_semantic_student_evaluation,
    semantic_completion_signature,
)
from .semantic_models import QWEN_ECONOMY
from .semantic_student_training import train_semantic_student


class SemanticCorpusSource(Protocol):
    def stats(self) -> SemanticTraceStats: ...

    def distillation_corpus(
        self,
        *,
        splits: Sequence[SemanticTraceSplit | str],
    ) -> SemanticDistillationCorpus: ...


class SemanticStudentTrainer(Protocol):
    def __call__(
        self,
        corpus_path: str | Path,
        output_path: str | Path,
        *,
        epochs: int,
        max_records: int,
        device: str,
        seed: int,
    ) -> Mapping[str, Any]: ...


class SemanticStudentEvaluator(Protocol):
    def __call__(
        self,
        corpus_path: str | Path,
        adapter_path: str | Path,
        *,
        output_path: str | Path | None,
        device: str,
    ) -> SemanticStudentEvaluation: ...


@dataclass(frozen=True)
class SemanticLearningRoleSummary:
    split: str
    records: int
    positives: int
    hard_negatives: int
    domains: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticLearningReadiness:
    ready: bool
    journal: SemanticTraceStats
    roles: tuple[SemanticLearningRoleSummary, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    prompt_overlap_count: int
    semantic_target_overlap_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "journal": self.journal.to_dict(),
            "roles": {role.split: role.to_dict() for role in self.roles},
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "prompt_overlap_count": self.prompt_overlap_count,
            "semantic_target_overlap_count": self.semantic_target_overlap_count,
            "next_action": (
                "run_candidate_training"
                if self.ready
                else "review_more_balanced_train_and_sealed_traces"
            ),
        }


class SemanticLearningCycleStatus(str, Enum):
    NOT_READY = "not_ready"
    GATE_REJECTED = "gate_rejected"
    STAGED = "staged"


@dataclass(frozen=True)
class SemanticLearningCycleResult:
    status: SemanticLearningCycleStatus
    readiness: SemanticLearningReadiness
    run_id: str = ""
    run_path: str = ""
    train_corpus_path: str = ""
    sealed_corpus_path: str = ""
    candidate_path: str = ""
    evaluation_path: str = ""
    candidate: ArtifactCandidate | None = None
    evaluation: SemanticStudentEvaluation | None = None

    @property
    def staged(self) -> bool:
        return self.status is SemanticLearningCycleStatus.STAGED

    def to_dict(self) -> dict[str, Any]:
        evaluation = self.evaluation
        return {
            "schema_version": "semop.semantic-learning-cycle.v1",
            "status": self.status.value,
            "readiness": self.readiness.to_dict(),
            "run_id": self.run_id,
            "run_path": self.run_path,
            "train_corpus_path": self.train_corpus_path,
            "sealed_corpus_path": self.sealed_corpus_path,
            "candidate_path": self.candidate_path,
            "evaluation_path": self.evaluation_path,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "evaluation": (
                {
                    "passed": evaluation.passed,
                    "positive_records": evaluation.positive_records,
                    "hard_negative_records": evaluation.hard_negative_records,
                    "semantic_matches": evaluation.semantic_matches,
                    "candidate_model_coverage": evaluation.candidate_model_coverage,
                    "teacher_model_coverage": evaluation.teacher_model_coverage,
                    "report": evaluation.report.to_dict(),
                    "rejection_reasons": list(
                        evaluation.report.rejection_reasons("semantic_model")
                    ),
                }
                if evaluation is not None
                else None
            ),
            "activation_required": self.candidate is not None,
            "activated": False,
        }


class SemanticStudentLearningCycle:
    """Coordinate reviewed export, training, sealed evaluation, and staging."""

    def __init__(
        self,
        trace_store: SemanticCorpusSource,
        promotion_store: ArtifactPromotionStore,
        *,
        trainer: SemanticStudentTrainer = train_semantic_student,
        evaluator: SemanticStudentEvaluator = run_local_semantic_student_evaluation,
    ) -> None:
        self.trace_store = trace_store
        self.promotion_store = promotion_store
        self.trainer = trainer
        self.evaluator = evaluator

    def readiness(self) -> SemanticLearningReadiness:
        readiness, _ = _readiness_snapshot(self.trace_store)
        return readiness

    def run(
        self,
        *,
        confirm_training: bool,
        device: str = "auto",
        epochs: int = 2,
        max_records: int = 0,
        seed: int = 4060,
    ) -> SemanticLearningCycleResult:
        readiness, corpora = _readiness_snapshot(self.trace_store)
        if not readiness.ready:
            return SemanticLearningCycleResult(
                status=SemanticLearningCycleStatus.NOT_READY,
                readiness=readiness,
            )
        if confirm_training is not True:
            raise ValueError("candidate training requires explicit --confirm")
        if epochs <= 0 or max_records < 0:
            raise ValueError("epochs must be positive and max_records cannot be negative")
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda, or cpu")

        run_id = _new_run_id(corpora, epochs=epochs, max_records=max_records, seed=seed)
        run_path = self.promotion_store.root / "runs" / run_id
        if run_path.exists():
            raise ValueError("semantic learning run already exists")
        run_path.mkdir(parents=True, exist_ok=False)
        train_path = corpora[SemanticTraceSplit.TRAIN].save_jsonl(
            run_path / "train.jsonl"
        )
        sealed_path = corpora[SemanticTraceSplit.SEALED].save_jsonl(
            run_path / "sealed.jsonl"
        )
        candidate_path = run_path / "candidate"
        evaluation_path = run_path / "sealed-evaluation.json"

        training_summary = dict(
            self.trainer(
                train_path,
                candidate_path,
                epochs=epochs,
                max_records=max_records,
                device=device,
                seed=seed,
            )
        )
        train_digest = _file_sha256(train_path)
        _validate_training_result(training_summary, train_digest, candidate_path)

        evaluation = self.evaluator(
            sealed_path,
            candidate_path,
            output_path=evaluation_path,
            device=device,
        )
        sealed_digest = _file_sha256(sealed_path)
        candidate_digest = artifact_sha256(candidate_path)
        if evaluation.corpus_sha256 != sealed_digest:
            raise ValueError("sealed evaluation is bound to a different corpus")
        if evaluation.candidate_sha256 != candidate_digest:
            raise ValueError("sealed evaluation is bound to a different candidate")
        _atomic_json(evaluation_path, evaluation.to_dict())

        common = {
            "readiness": readiness,
            "run_id": run_id,
            "run_path": str(run_path.resolve()),
            "train_corpus_path": str(train_path.resolve()),
            "sealed_corpus_path": str(sealed_path.resolve()),
            "candidate_path": str(candidate_path.resolve()),
            "evaluation_path": str(evaluation_path.resolve()),
            "evaluation": evaluation,
        }
        if not evaluation.passed:
            return SemanticLearningCycleResult(
                status=SemanticLearningCycleStatus.GATE_REJECTED,
                **common,
            )

        candidate = self.promotion_store.stage(
            candidate_path,
            kind="semantic_model",
            report=evaluation.report,
            metadata={
                "schema_version": "semop.semantic-learning-cycle-metadata.v1",
                "run_id": run_id,
                "base_model_id": training_summary["student_model_id"],
                "base_model_revision": training_summary["student_model_revision"],
                "train_corpus_sha256": train_digest,
                "sealed_corpus_sha256": sealed_digest,
                "training_positive_records": int(
                    training_summary.get("positive_records", 0)
                ),
                "human_activation_required": True,
            },
        )
        return SemanticLearningCycleResult(
            status=SemanticLearningCycleStatus.STAGED,
            candidate=candidate,
            **common,
        )


def inspect_semantic_learning_readiness(
    trace_store: SemanticCorpusSource,
) -> SemanticLearningReadiness:
    readiness, _ = _readiness_snapshot(trace_store)
    return readiness


def _readiness_snapshot(
    trace_store: SemanticCorpusSource,
) -> tuple[
    SemanticLearningReadiness,
    dict[SemanticTraceSplit, SemanticDistillationCorpus],
]:
    corpora = {
        split: trace_store.distillation_corpus(splits=(split,))
        for split in SemanticTraceSplit
    }
    roles = tuple(_role_summary(split, corpora[split]) for split in SemanticTraceSplit)
    by_split = {role.split: role for role in roles}
    blockers: list[str] = []
    warnings: list[str] = []
    if by_split[SemanticTraceSplit.TRAIN.value].positives == 0:
        blockers.append("train_positive_records_missing")
    if by_split[SemanticTraceSplit.SEALED.value].positives == 0:
        blockers.append("sealed_positive_records_missing")
    if by_split[SemanticTraceSplit.VALIDATION.value].positives == 0:
        warnings.append("validation_positive_records_missing")
    if by_split[SemanticTraceSplit.SEALED.value].positives < 20:
        warnings.append("sealed_positive_records_below_20_gold_target")

    prompt_overlaps = _cross_split_overlaps(corpora, _prompt_fingerprint)
    target_overlaps = _cross_split_overlaps(corpora, _semantic_target_fingerprint)
    if prompt_overlaps:
        blockers.append("raw_prompt_overlap_across_splits")
    if target_overlaps:
        blockers.append("semantic_target_overlap_across_splits")
    journal = trace_store.stats()
    if journal.accepted + journal.rejected > journal.exportable:
        warnings.append("reviewed_image_labels_are_excluded_from_text_student_training")
    return (
        SemanticLearningReadiness(
            ready=not blockers,
            journal=journal,
            roles=roles,
            blockers=tuple(blockers),
            warnings=tuple(warnings),
            prompt_overlap_count=len(prompt_overlaps),
            semantic_target_overlap_count=len(target_overlaps),
        ),
        corpora,
    )


def _role_summary(
    split: SemanticTraceSplit,
    corpus: SemanticDistillationCorpus,
) -> SemanticLearningRoleSummary:
    positives = len(corpus.positives)
    return SemanticLearningRoleSummary(
        split=split.value,
        records=len(corpus.records),
        positives=positives,
        hard_negatives=len(corpus.records) - positives,
        domains=tuple(sorted({record.domain for record in corpus.records})),
    )


def _cross_split_overlaps(
    corpora: Mapping[SemanticTraceSplit, SemanticDistillationCorpus],
    fingerprint: Callable[[SemanticDistillationRecord], str],
) -> set[str]:
    owners: dict[str, set[SemanticTraceSplit]] = {}
    for split, corpus in corpora.items():
        for record in corpus.records:
            value = fingerprint(record)
            owners.setdefault(value, set()).add(split)
    return {value for value, splits in owners.items() if len(splits) > 1}


def _prompt_fingerprint(record: SemanticDistillationRecord) -> str:
    normalized = " ".join(
        unicodedata.normalize("NFKC", record.prompt).casefold().split()
    )
    return sha256(normalized.encode("utf-8")).hexdigest()


def _semantic_target_fingerprint(record: SemanticDistillationRecord) -> str:
    signature = semantic_completion_signature(record.completion) or record.completion
    return sha256(signature.encode("utf-8")).hexdigest()


def _new_run_id(
    corpora: Mapping[SemanticTraceSplit, SemanticDistillationCorpus],
    *,
    epochs: int,
    max_records: int,
    seed: int,
) -> str:
    identity = json.dumps(
        {
            split.value: [record.record_id for record in corpus.records]
            for split, corpus in corpora.items()
        }
        | {"epochs": epochs, "max_records": max_records, "seed": seed},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    short_digest = sha256(identity.encode("utf-8")).hexdigest()[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{short_digest}"


def _validate_training_result(
    summary: Mapping[str, Any],
    train_digest: str,
    candidate_path: Path,
) -> None:
    if summary.get("candidate_only") is not True:
        raise ValueError("student trainer did not produce a candidate-only artifact")
    if summary.get("corpus_sha256") != train_digest:
        raise ValueError("student training summary is bound to a different corpus")
    if summary.get("student_model_id") != QWEN_ECONOMY.model_id:
        raise ValueError("student training summary uses the wrong base model")
    if summary.get("student_model_revision") != QWEN_ECONOMY.revision:
        raise ValueError("student training summary uses the wrong base model revision")
    if not candidate_path.is_dir():
        raise ValueError("student trainer did not create a candidate directory")
    summary_path = candidate_path / "semop_training_summary.json"
    if not summary_path.is_file():
        raise ValueError("student candidate has no training summary")
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    if persisted != dict(summary):
        raise ValueError("persisted student training summary does not match the runner")


def _file_sha256(path: Path) -> str:
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or run the reviewed semantic student cycle. A passing run is "
            "staged for human review and is never activated automatically."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/experience/beginner-experience.db"),
    )
    parser.add_argument(
        "--promotion-root",
        type=Path,
        default=Path("artifacts/promotion/semantic-model"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="show train/validation/sealed readiness")
    run = commands.add_parser(
        "run",
        help="train, run sealed evaluation, and stage only a passing candidate",
    )
    run.add_argument("--confirm", action="store_true")
    run.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    run.add_argument("--epochs", type=int, default=2)
    run.add_argument("--max-records", type=int, default=0)
    run.add_argument("--seed", type=int, default=4060)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        cycle = SemanticStudentLearningCycle(
            SemanticTraceStore(args.db),
            ArtifactPromotionStore(args.promotion_root),
        )
        if args.command == "status":
            payload = cycle.readiness().to_dict()
            exit_code = 0
        else:
            result = cycle.run(
                confirm_training=args.confirm,
                device=args.device,
                epochs=args.epochs,
                max_records=args.max_records,
                seed=args.seed,
            )
            payload = result.to_dict()
            exit_code = 0 if result.staged else 2
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"semantic student cycle failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

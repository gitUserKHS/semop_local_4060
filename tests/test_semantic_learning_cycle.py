from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from semop.artifact_promotion import ArtifactPromotionStore, PromotionGateReport, artifact_sha256
from semop.semantic_distillation import SemanticDistillationCorpus, SemanticDistillationRecord
from semop.semantic_experience import (
    SemanticTraceSplit,
    SemanticTraceStats,
    semantic_request_split,
)
from semop.semantic_learning_cycle import (
    SemanticLearningCycleStatus,
    SemanticStudentLearningCycle,
    inspect_semantic_learning_readiness,
    main,
)
from semop.semantic_student_evaluation import SemanticStudentEvaluation
from semop.semantic_models import QWEN_ECONOMY


def _record(
    name: str,
    split: SemanticTraceSplit,
    *,
    prompt: str | None = None,
    expression: str | None = None,
) -> SemanticDistillationRecord:
    completion = json.dumps(
        {
            "domain": "math",
            "confidence": 0.8,
            "operator_program": ["DECOMPOSE", "INFER", "VERIFY", "EXPLAIN"],
            "payload": {"expression": expression or f"{len(name)}+1"},
            "answer": str(len(name) + 1),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return SemanticDistillationRecord(
        record_id="",
        domain="math",
        prompt=prompt or f"{name} 문제",
        completion=completion,
        decision="accepted",
        proposal_fingerprint=sha256(f"proposal:{name}:{split.value}".encode()).hexdigest(),
        replay_verified=True,
        semantic_review_digest=sha256(f"review:{name}:{split.value}".encode()).hexdigest(),
        data_split=split.value,
        source="local_semantic_trace_review",
    )


class _CorpusSource:
    def __init__(
        self,
        train: tuple[SemanticDistillationRecord, ...] = (),
        validation: tuple[SemanticDistillationRecord, ...] = (),
        sealed: tuple[SemanticDistillationRecord, ...] = (),
    ) -> None:
        self.corpora = {
            SemanticTraceSplit.TRAIN: SemanticDistillationCorpus(train),
            SemanticTraceSplit.VALIDATION: SemanticDistillationCorpus(validation),
            SemanticTraceSplit.SEALED: SemanticDistillationCorpus(sealed),
        }

    def stats(self) -> SemanticTraceStats:
        counts = {split: len(corpus.records) for split, corpus in self.corpora.items()}
        total = sum(counts.values())
        return SemanticTraceStats(
            total=total,
            pending=0,
            accepted=total,
            rejected=0,
            exportable=total,
            train=counts[SemanticTraceSplit.TRAIN],
            validation=counts[SemanticTraceSplit.VALIDATION],
            sealed=counts[SemanticTraceSplit.SEALED],
        )

    def distillation_corpus(
        self,
        *,
        splits: tuple[SemanticTraceSplit | str, ...],
    ) -> SemanticDistillationCorpus:
        selected: list[SemanticDistillationRecord] = []
        for split in splits:
            selected.extend(self.corpora[SemanticTraceSplit(split)].records)
        return SemanticDistillationCorpus(tuple(selected))


def _trainer(calls: list[str]):
    def train(
        corpus_path: str | Path,
        output_path: str | Path,
        **_: object,
    ) -> dict[str, object]:
        calls.append("train")
        corpus_file = Path(corpus_path)
        output = Path(output_path)
        output.mkdir()
        (output / "adapter_model.safetensors").write_bytes(b"candidate")
        summary: dict[str, object] = {
            "schema_version": "semop.semantic-student-training.v1",
            "candidate_only": True,
            "corpus_sha256": sha256(corpus_file.read_bytes()).hexdigest(),
            "student_model_id": QWEN_ECONOMY.model_id,
            "student_model_revision": QWEN_ECONOMY.revision,
            "positive_records": len(
                SemanticDistillationCorpus.load_jsonl(corpus_file).positives
            ),
        }
        (output / "semop_training_summary.json").write_text(
            json.dumps(summary, sort_keys=True),
            encoding="utf-8",
        )
        return summary

    return train


def _evaluator(calls: list[str], *, passed: bool = True):
    def evaluate(
        corpus_path: str | Path,
        adapter_path: str | Path,
        **_: object,
    ) -> SemanticStudentEvaluation:
        calls.append("evaluate")
        corpus_file = Path(corpus_path)
        corpus = SemanticDistillationCorpus.load_jsonl(corpus_file)
        positives = len(corpus.positives)
        report = PromotionGateReport(
            sealed_evaluation_digest="e" * 64,
            replay_integrity=1.0,
            false_acceptances=0,
            teacher_retention=1.0 if passed else 0.5,
            latency_reduction=0.5,
            artifact_bytes=1,
        )
        return SemanticStudentEvaluation(
            schema_version="semop.semantic-student-evaluation.v1",
            corpus_sha256=sha256(corpus_file.read_bytes()).hexdigest(),
            candidate_sha256=artifact_sha256(adapter_path),
            candidate_model_id="student#adapter",
            teacher_model_id="teacher",
            positive_records=positives,
            hard_negative_records=len(corpus.records) - positives,
            semantic_matches=positives if passed else 0,
            teacher_live_matches=positives,
            candidate_model_coverage=1.0,
            teacher_model_coverage=1.0,
            median_candidate_seconds=1.0,
            median_teacher_seconds=2.0,
            candidate_peak_vram_bytes=100,
            teacher_peak_vram_bytes=200,
            outcomes=(),
            report=report,
        )

    return evaluate


class _TrackingPromotionStore(ArtifactPromotionStore):
    def __init__(self, root: Path, calls: list[str]) -> None:
        super().__init__(root)
        self.calls = calls

    def stage(self, *args: object, **kwargs: object):
        self.calls.append("stage")
        return super().stage(*args, **kwargs)


def test_empty_journal_is_not_ready_and_never_loads_models(tmp_path: Path) -> None:
    calls: list[str] = []
    cycle = SemanticStudentLearningCycle(
        _CorpusSource(),
        ArtifactPromotionStore(tmp_path / "promotion"),
        trainer=_trainer(calls),
        evaluator=_evaluator(calls),
    )

    result = cycle.run(confirm_training=True)

    assert result.status is SemanticLearningCycleStatus.NOT_READY
    assert result.readiness.blockers == (
        "train_positive_records_missing",
        "sealed_positive_records_missing",
    )
    assert calls == []
    assert not (tmp_path / "promotion" / "runs").exists()


def test_cycle_trains_evaluates_and_stages_without_activation(tmp_path: Path) -> None:
    source = _CorpusSource(
        train=(_record("train-a", SemanticTraceSplit.TRAIN),),
        validation=(_record("validation-a", SemanticTraceSplit.VALIDATION),),
        sealed=(_record("sealed-a", SemanticTraceSplit.SEALED),),
    )
    calls: list[str] = []
    promotion = _TrackingPromotionStore(tmp_path / "promotion", calls)
    cycle = SemanticStudentLearningCycle(
        source,
        promotion,
        trainer=_trainer(calls),
        evaluator=_evaluator(calls),
    )

    result = cycle.run(confirm_training=True, device="cpu", epochs=1)

    assert result.status is SemanticLearningCycleStatus.STAGED
    assert calls == ["train", "evaluate", "stage"]
    assert result.candidate is not None
    assert promotion.load_candidate(result.candidate.candidate_id).passed is True
    assert result.candidate.metadata["base_model_id"] == QWEN_ECONOMY.model_id
    assert result.candidate.metadata["base_model_revision"] == QWEN_ECONOMY.revision
    assert not promotion.active_path.exists()
    assert result.to_dict()["activation_required"] is True
    assert result.to_dict()["activated"] is False
    assert Path(result.train_corpus_path).is_file()
    assert Path(result.sealed_corpus_path).is_file()
    assert Path(result.evaluation_path).is_file()


def test_failed_sealed_gate_keeps_candidate_out_of_staging(tmp_path: Path) -> None:
    source = _CorpusSource(
        train=(_record("train-b", SemanticTraceSplit.TRAIN),),
        sealed=(_record("sealed-b", SemanticTraceSplit.SEALED),),
    )
    calls: list[str] = []
    promotion = _TrackingPromotionStore(tmp_path / "promotion", calls)
    cycle = SemanticStudentLearningCycle(
        source,
        promotion,
        trainer=_trainer(calls),
        evaluator=_evaluator(calls, passed=False),
    )

    result = cycle.run(confirm_training=True)

    assert result.status is SemanticLearningCycleStatus.GATE_REJECTED
    assert calls == ["train", "evaluate"]
    assert result.candidate is None
    assert not promotion.candidates_root.exists()
    assert not promotion.active_path.exists()


def test_cross_split_prompt_and_target_leakage_blocks_training() -> None:
    source = _CorpusSource(
        train=(
            _record(
                "train-c",
                SemanticTraceSplit.TRAIN,
                prompt="  같은   질문 ",
                expression="12 - 5",
            ),
        ),
        sealed=(
            _record(
                "sealed-c",
                SemanticTraceSplit.SEALED,
                prompt="같은 질문",
                expression="12-5",
            ),
        ),
    )

    readiness = inspect_semantic_learning_readiness(source)

    assert readiness.ready is False
    assert "raw_prompt_overlap_across_splits" in readiness.blockers
    assert "semantic_target_overlap_across_splits" in readiness.blockers
    assert readiness.prompt_overlap_count == 1
    assert readiness.semantic_target_overlap_count == 1


def test_raw_request_split_ignores_model_tier_and_text_spacing() -> None:
    first = json.dumps(
        {
            "text": "  같은   질문 ",
            "images": [],
            "workspace": None,
            "resource_tier": "balanced",
            "judge_opt_in": False,
        }
    )
    second = json.dumps(
        {
            "text": "같은 질문",
            "images": [],
            "workspace": None,
            "resource_tier": "economy",
            "judge_opt_in": True,
        }
    )

    assert semantic_request_split(first) is semantic_request_split(second)


def test_ready_cycle_still_requires_explicit_training_confirmation(tmp_path: Path) -> None:
    cycle = SemanticStudentLearningCycle(
        _CorpusSource(
            train=(_record("train-d", SemanticTraceSplit.TRAIN),),
            sealed=(_record("sealed-d", SemanticTraceSplit.SEALED),),
        ),
        ArtifactPromotionStore(tmp_path / "promotion"),
        trainer=_trainer([]),
        evaluator=_evaluator([]),
    )

    with pytest.raises(ValueError, match="explicit --confirm"):
        cycle.run(confirm_training=False)


def test_cycle_cli_reports_empty_local_journal_without_model_loading(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = [
        "--db",
        str(tmp_path / "experience.db"),
        "--promotion-root",
        str(tmp_path / "promotion"),
    ]

    assert main([*arguments, "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["ready"] is False
    assert status["roles"]["train"]["positives"] == 0
    assert status["roles"]["sealed"]["positives"] == 0

    assert main([*arguments, "run", "--confirm"]) == 2
    run = json.loads(capsys.readouterr().out)
    assert run["status"] == "not_ready"
    assert not (tmp_path / "promotion" / "runs").exists()

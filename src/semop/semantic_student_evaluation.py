from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import median
import sys
from typing import Any, Iterable, Mapping, Sequence

from .artifact_promotion import PromotionGateReport, artifact_sha256
from .assistant import SemOpAssistant
from .prompt_api import PromptRequest
from .prompt_compiler import PromptCompiler
from .semantic_distillation import (
    DistillationDecision,
    SemanticDistillationCorpus,
    SemanticDistillationRecord,
)
from .semantic_experience import canonical_completion_from_proposal
from .semantic_models import (
    E5OperatorRetriever,
    MODEL_SPECS,
    MULTILINGUAL_E5,
    QWEN_BALANCED,
    QWEN_ECONOMY,
    QwenSemanticBackend,
    QwenSemanticConfig,
)


@dataclass(frozen=True)
class SemanticRuntimeObservation:
    record_id: str
    completion_json: str
    model_id: str
    model_used: bool
    proof_eligible: bool
    replay_verified: bool
    elapsed_seconds: float
    peak_vram_bytes: int | None = None

    def __post_init__(self) -> None:
        _require_digest(self.record_id, "semantic record")
        if self.completion_json:
            value = _json_object(self.completion_json, "runtime completion")
            object.__setattr__(self, "completion_json", _canonical_json(value))
        if self.model_used and not self.model_id.strip():
            raise ValueError("a model-backed observation requires a model id")
        if type(self.model_used) is not bool:
            raise TypeError("model_used must be boolean")
        if type(self.proof_eligible) is not bool:
            raise TypeError("proof_eligible must be boolean")
        if type(self.replay_verified) is not bool:
            raise TypeError("replay_verified must be boolean")
        if self.elapsed_seconds < 0:
            raise ValueError("semantic evaluation latency cannot be negative")
        if self.peak_vram_bytes is not None and self.peak_vram_bytes < 0:
            raise ValueError("semantic evaluation VRAM cannot be negative")


@dataclass(frozen=True)
class SemanticStudentOutcome:
    record_id: str
    domain: str
    decision: str
    candidate_model_used: bool
    proof_eligible: bool
    replay_verified: bool
    semantic_match: bool
    repeated_rejected_semantics: bool
    expected_signature_sha256: str
    candidate_signature_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticStudentEvaluation:
    schema_version: str
    corpus_sha256: str
    candidate_sha256: str
    candidate_model_id: str
    teacher_model_id: str
    positive_records: int
    hard_negative_records: int
    semantic_matches: int
    teacher_live_matches: int
    candidate_model_coverage: float
    teacher_model_coverage: float
    median_candidate_seconds: float
    median_teacher_seconds: float
    candidate_peak_vram_bytes: int | None
    teacher_peak_vram_bytes: int | None
    outcomes: tuple[SemanticStudentOutcome, ...]
    report: PromotionGateReport

    @property
    def passed(self) -> bool:
        return not self.report.rejection_reasons("semantic_model")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "corpus_sha256": self.corpus_sha256,
            "candidate_sha256": self.candidate_sha256,
            "candidate_model_id": self.candidate_model_id,
            "teacher_model_id": self.teacher_model_id,
            "positive_records": self.positive_records,
            "hard_negative_records": self.hard_negative_records,
            "semantic_matches": self.semantic_matches,
            "teacher_live_matches": self.teacher_live_matches,
            "candidate_model_coverage": self.candidate_model_coverage,
            "teacher_model_coverage": self.teacher_model_coverage,
            "median_candidate_seconds": self.median_candidate_seconds,
            "median_teacher_seconds": self.median_teacher_seconds,
            "candidate_peak_vram_bytes": self.candidate_peak_vram_bytes,
            "teacher_peak_vram_bytes": self.teacher_peak_vram_bytes,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "report": self.report.to_dict(),
            "passed": self.passed,
            "rejection_reasons": list(
                self.report.rejection_reasons("semantic_model")
            ),
        }


def evaluate_semantic_student(
    corpus: SemanticDistillationCorpus,
    student_observations: Sequence[SemanticRuntimeObservation],
    teacher_observations: Sequence[SemanticRuntimeObservation],
    *,
    corpus_sha256: str,
    candidate_sha256: str,
    candidate_model_id: str,
    teacher_model_id: str,
    artifact_bytes: int,
) -> SemanticStudentEvaluation:
    """Evaluate one candidate only against a precommitted sealed text corpus."""

    corpus.validate_budget()
    corpus.require_split("sealed")
    _require_digest(corpus_sha256, "sealed corpus")
    _require_digest(candidate_sha256, "student candidate")
    if artifact_bytes < 0:
        raise ValueError("candidate artifact bytes cannot be negative")
    if not candidate_model_id.strip() or not teacher_model_id.strip():
        raise ValueError("semantic evaluation requires candidate and teacher model ids")
    if any(not record.semantic_review_digest for record in corpus.records):
        raise ValueError("sealed semantic evaluation requires human review digests")

    positives = tuple(
        record
        for record in corpus.records
        if record.decision is not DistillationDecision.HARD_NEGATIVE
    )
    negatives = tuple(
        record
        for record in corpus.records
        if record.decision is DistillationDecision.HARD_NEGATIVE
    )
    if not positives:
        raise ValueError("sealed semantic evaluation requires positive records")

    student = _observation_map(
        student_observations,
        expected_ids={record.record_id for record in corpus.records},
        label="student",
    )
    teacher = _observation_map(
        teacher_observations,
        expected_ids={record.record_id for record in positives},
        label="teacher",
    )
    if any(
        observation.model_used and observation.model_id != candidate_model_id
        for observation in student.values()
    ):
        raise ValueError("student observations do not match the candidate model id")
    if any(
        observation.model_used and observation.model_id != teacher_model_id
        for observation in teacher.values()
    ):
        raise ValueError("teacher observations do not match the teacher model id")

    outcomes: list[SemanticStudentOutcome] = []
    semantic_matches = 0
    false_acceptances = 0
    proof_eligible = 0
    replayed_proofs = 0
    for record in corpus.records:
        observation = student[record.record_id]
        expected_signature = semantic_completion_signature(record.completion)
        candidate_signature = semantic_completion_signature(observation.completion_json)
        is_positive = record.decision is not DistillationDecision.HARD_NEGATIVE
        semantic_match = bool(
            is_positive
            and observation.model_used
            and observation.proof_eligible
            and observation.replay_verified
            and candidate_signature
            and candidate_signature == expected_signature
        )
        repeated_rejected = bool(
            not is_positive
            and observation.model_used
            and observation.proof_eligible
            and observation.replay_verified
            and candidate_signature
            and candidate_signature == expected_signature
        )
        semantic_matches += int(semantic_match)
        false_acceptances += int(repeated_rejected)
        if observation.proof_eligible:
            proof_eligible += 1
            replayed_proofs += int(observation.replay_verified)
        outcomes.append(
            SemanticStudentOutcome(
                record_id=record.record_id,
                domain=record.domain,
                decision=record.decision.value,
                candidate_model_used=observation.model_used,
                proof_eligible=observation.proof_eligible,
                replay_verified=observation.replay_verified,
                semantic_match=semantic_match,
                repeated_rejected_semantics=repeated_rejected,
                expected_signature_sha256=_text_digest(expected_signature),
                candidate_signature_sha256=(
                    _text_digest(candidate_signature) if candidate_signature else ""
                ),
            )
        )

    teacher_live_matches = sum(
        bool(
            teacher[record.record_id].model_used
            and teacher[record.record_id].proof_eligible
            and teacher[record.record_id].replay_verified
            and semantic_completion_signature(teacher[record.record_id].completion_json)
            == semantic_completion_signature(record.completion)
        )
        for record in positives
    )
    candidate_coverage = sum(
        student[record.record_id].model_used for record in corpus.records
    ) / len(corpus.records)
    teacher_coverage = sum(
        teacher[record.record_id].model_used for record in positives
    ) / len(positives)
    replay_integrity = (
        replayed_proofs / proof_eligible if proof_eligible else 0.0
    )
    teacher_retention = semantic_matches / len(positives)

    student_positive = tuple(student[record.record_id] for record in positives)
    teacher_positive = tuple(teacher[record.record_id] for record in positives)
    candidate_latency = _median_latency(student_positive)
    teacher_latency = _median_latency(teacher_positive)
    candidate_vram = _peak_vram(student_positive)
    teacher_vram = _peak_vram(teacher_positive)
    resource_baseline_valid = teacher_coverage == 1.0
    latency_reduction = (
        _reduction(candidate_latency, teacher_latency)
        if resource_baseline_valid
        else 0.0
    )
    vram_reduction = (
        _reduction(candidate_vram, teacher_vram)
        if resource_baseline_valid
        else 0.0
    )

    outcome_payload = [item.to_dict() for item in outcomes]
    sealed_digest = _text_digest(
        _canonical_json(
            {
                "schema_version": "semop.semantic-student-seal.v1",
                "corpus_sha256": corpus_sha256,
                "candidate_sha256": candidate_sha256,
                "candidate_model_id": candidate_model_id,
                "teacher_model_id": teacher_model_id,
                "outcomes": outcome_payload,
                "candidate_latency": candidate_latency,
                "teacher_latency": teacher_latency,
                "candidate_vram": candidate_vram,
                "teacher_vram": teacher_vram,
            }
        )
    )
    report = PromotionGateReport(
        sealed_evaluation_digest=sealed_digest,
        replay_integrity=replay_integrity,
        false_acceptances=false_acceptances,
        teacher_retention=teacher_retention,
        latency_reduction=latency_reduction,
        vram_reduction=vram_reduction,
        artifact_bytes=artifact_bytes,
        notes=(
            f"sealed_positive_records={len(positives)}",
            f"sealed_hard_negatives={len(negatives)}",
            f"candidate_model_coverage={candidate_coverage:.6f}",
            f"teacher_model_coverage={teacher_coverage:.6f}",
            f"teacher_live_semantic_matches={teacher_live_matches}/{len(positives)}",
            "semantic match compares reviewed typed domain and payload; confidence and prose are ignored",
        ),
    )
    return SemanticStudentEvaluation(
        schema_version="semop.semantic-student-evaluation.v1",
        corpus_sha256=corpus_sha256,
        candidate_sha256=candidate_sha256,
        candidate_model_id=candidate_model_id,
        teacher_model_id=teacher_model_id,
        positive_records=len(positives),
        hard_negative_records=len(negatives),
        semantic_matches=semantic_matches,
        teacher_live_matches=teacher_live_matches,
        candidate_model_coverage=candidate_coverage,
        teacher_model_coverage=teacher_coverage,
        median_candidate_seconds=candidate_latency,
        median_teacher_seconds=teacher_latency,
        candidate_peak_vram_bytes=candidate_vram,
        teacher_peak_vram_bytes=teacher_vram,
        outcomes=tuple(outcomes),
        report=report,
    )


class LocalSemanticRunner:
    """Run one pinned local backend and expose only evaluator-safe measurements."""

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        adapter_path: str = "",
        adapter_sha256: str = "",
    ) -> None:
        spec = MODEL_SPECS[model_id]
        self.backend = QwenSemanticBackend(
            QwenSemanticConfig(
                model_id=model_id,
                max_context=spec.max_context,
                max_new_tokens=512,
                local_files_only=True,
                device=device,
                adapter_path=adapter_path,
                adapter_sha256=adapter_sha256,
            )
        )
        self.retriever = E5OperatorRetriever(
            MULTILINGUAL_E5.model_id,
            local_files_only=True,
            device="cpu",
        )
        self.assistant = SemOpAssistant(
            compiler=PromptCompiler(
                backend=self.backend,
                retriever=self.retriever,
            )
        )

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    def run(self, record: SemanticDistillationRecord) -> SemanticRuntimeObservation:
        _reset_peak_vram()
        request = PromptRequest(record.prompt, resource_tier="balanced")
        answer = self.assistant.solve(request)
        _synchronize_cuda()
        model_proposal = next(
            (
                proposal
                for proposal in answer.proposals
                if not proposal.deterministic
                and proposal.model_id.startswith(self.backend.config.model_id)
            ),
            None,
        )
        completion = (
            _canonical_json(canonical_completion_from_proposal(request, model_proposal))
            if model_proposal is not None
            else ""
        )
        proof_eligible = bool(answer.results) and all(
            result.success for result in answer.results
        )
        return SemanticRuntimeObservation(
            record_id=record.record_id,
            completion_json=completion,
            model_id=model_proposal.model_id if model_proposal is not None else "",
            model_used=model_proposal is not None,
            proof_eligible=proof_eligible,
            replay_verified=bool(
                answer.provenance.get("logical_replay_verified", False)
            ),
            elapsed_seconds=answer.metrics.elapsed_seconds,
            peak_vram_bytes=answer.metrics.peak_vram_bytes,
        )

    def unload(self) -> None:
        self.backend.unload()
        self.assistant = None  # type: ignore[assignment]
        self.retriever = None  # type: ignore[assignment]
        gc.collect()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a candidate Qwen3.5-0.8B LoRA on a human-reviewed sealed "
            "SemOp corpus. The command writes a gate report but never activates it."
        )
    )
    parser.add_argument("corpus", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/student-sealed-report.json"),
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def run_local_semantic_student_evaluation(
    corpus_path: str | Path,
    adapter_path: str | Path,
    *,
    output_path: str | Path | None = None,
    device: str = "auto",
) -> SemanticStudentEvaluation:
    """Run the pinned student and teacher against one human-reviewed sealed corpus."""

    corpus_file = Path(corpus_path)
    adapter = Path(adapter_path)
    _validate_device(device)
    corpus = SemanticDistillationCorpus.load_jsonl(corpus_file)
    corpus.validate_budget()
    corpus.require_split("sealed")
    positives = tuple(
        record
        for record in corpus.records
        if record.decision is not DistillationDecision.HARD_NEGATIVE
    )
    if not positives:
        raise ValueError("sealed semantic evaluation requires positive records")
    if any(not record.semantic_review_digest for record in corpus.records):
        raise ValueError("sealed semantic evaluation requires human review digests")
    _validate_training_summary(adapter)
    candidate_digest = artifact_sha256(adapter)
    corpus_digest = _file_digest(corpus_file)

    student_runner = LocalSemanticRunner(
        model_id=QWEN_ECONOMY.model_id,
        device=device,
        adapter_path=str(adapter.resolve()),
        adapter_sha256=candidate_digest,
    )
    try:
        student_observations = tuple(
            student_runner.run(record) for record in corpus.records
        )
        student_model_id = student_runner.model_id
    finally:
        student_runner.unload()

    teacher_runner = LocalSemanticRunner(
        model_id=QWEN_BALANCED.model_id,
        device=device,
    )
    try:
        teacher_observations = tuple(
            teacher_runner.run(record) for record in positives
        )
        teacher_model_id = teacher_runner.model_id
    finally:
        teacher_runner.unload()

    evaluation = evaluate_semantic_student(
        corpus,
        student_observations,
        teacher_observations,
        corpus_sha256=corpus_digest,
        candidate_sha256=candidate_digest,
        candidate_model_id=student_model_id,
        teacher_model_id=teacher_model_id,
        artifact_bytes=_artifact_bytes(adapter),
    )
    if output_path is not None:
        _atomic_json(Path(output_path), evaluation.to_dict())
    return evaluation


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        evaluation = run_local_semantic_student_evaluation(
            args.corpus,
            args.adapter,
            output_path=args.output,
            device=args.device,
        )
        payload = evaluation.to_dict()
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"semantic student evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if evaluation.passed else 2


def _observation_map(
    observations: Sequence[SemanticRuntimeObservation],
    *,
    expected_ids: set[str],
    label: str,
) -> dict[str, SemanticRuntimeObservation]:
    indexed: dict[str, SemanticRuntimeObservation] = {}
    for observation in observations:
        if observation.record_id in indexed:
            raise ValueError(f"duplicate {label} semantic observation")
        indexed[observation.record_id] = observation
    actual_ids = set(indexed)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"{label} semantic observations do not match sealed records: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return indexed


def semantic_completion_signature(completion_json: str) -> str:
    if not completion_json:
        return ""
    value = _json_object(completion_json, "semantic completion")
    domain = str(value.get("domain", "")).strip().lower()
    payload = value.get("payload")
    if not domain or not isinstance(payload, Mapping):
        return ""
    normalized_payload = dict(payload)
    if domain == "math" and isinstance(normalized_payload.get("expression"), str):
        normalized_payload["expression"] = re.sub(
            r"\s+", "", normalized_payload["expression"]
        )
    return _canonical_json({"domain": domain, "payload": normalized_payload})


def _median_latency(
    observations: Sequence[SemanticRuntimeObservation],
) -> float:
    return float(median(item.elapsed_seconds for item in observations))


def _peak_vram(
    observations: Sequence[SemanticRuntimeObservation],
) -> int | None:
    values = tuple(
        item.peak_vram_bytes
        for item in observations
        if item.peak_vram_bytes is not None
    )
    return max(values) if values else None


def _reduction(candidate: float | int | None, baseline: float | int | None) -> float:
    if candidate is None or baseline is None or baseline <= 0:
        return 0.0
    return min(1.0, max(0.0, 1.0 - float(candidate) / float(baseline)))


def _validate_training_summary(adapter: Path) -> None:
    summary_path = adapter / "semop_training_summary.json"
    value = _json_object(
        summary_path.read_text(encoding="utf-8"),
        "student training summary",
    )
    if value.get("candidate_only") is not True:
        raise ValueError("student adapter is not marked candidate-only")
    if value.get("student_model_id") != QWEN_ECONOMY.model_id:
        raise ValueError("student adapter has the wrong base model id")
    if value.get("student_model_revision") != QWEN_ECONOMY.revision:
        raise ValueError("student adapter has the wrong base model revision")


def _validate_device(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA evaluation requires PyTorch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def _reset_peak_vram() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _synchronize_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _artifact_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_object(value: str, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return decoded


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} digest must be SHA-256")


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

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROMOTION_APPROVAL_ATTESTATION = (
    "I reviewed the sealed SemOp promotion report and approve activation."
)


@dataclass(frozen=True)
class PromotionGateReport:
    sealed_evaluation_digest: str
    replay_integrity: float
    false_acceptances: int
    solve_rate_delta: float = 0.0
    expansion_reduction_domains: int = 0
    teacher_retention: float = 1.0
    latency_reduction: float = 0.0
    vram_reduction: float = 0.0
    artifact_bytes: int = 0
    parameter_count: int = 0
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _is_sha256(self.sealed_evaluation_digest):
            raise ValueError("sealed evaluation digest must be a SHA-256 hex digest")
        rates = (
            self.replay_integrity,
            self.teacher_retention,
            self.latency_reduction,
            self.vram_reduction,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("promotion rates must be between 0 and 1")
        if self.false_acceptances < 0 or self.expansion_reduction_domains < 0:
            raise ValueError("promotion counts cannot be negative")
        if self.artifact_bytes < 0 or self.parameter_count < 0:
            raise ValueError("artifact resources cannot be negative")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    def rejection_reasons(self, kind: str) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.replay_integrity != 1.0:
            reasons.append("replay_integrity_below_100_percent")
        if self.false_acceptances != 0:
            reasons.append("proof_eligible_false_acceptance_detected")
        if self.artifact_bytes > 64 * 1024 * 1024 and kind != "semantic_model":
            reasons.append("controller_or_macro_artifact_exceeds_64_mib")
        if self.parameter_count > 15_000_000 and kind != "semantic_model":
            reasons.append("controller_parameter_count_exceeds_15m")
        if kind == "controller":
            if self.solve_rate_delta < -0.01:
                reasons.append("controller_solve_rate_regression_exceeds_1pp")
            if self.expansion_reduction_domains < 2:
                reasons.append("controller_lacks_30_percent_gain_in_two_domains")
        if kind == "semantic_model":
            if self.teacher_retention < 0.95:
                reasons.append("student_retains_less_than_95_percent_of_teacher")
            if max(self.latency_reduction, self.vram_reduction) < 0.30:
                reasons.append("student_resource_reduction_below_30_percent")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactCandidate:
    candidate_id: str
    kind: str
    artifact_path: str
    artifact_sha256: str
    report: PromotionGateReport
    created_at: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not _is_sha256(self.candidate_id):
            raise ValueError("candidate id must be a SHA-256 digest")
        if not _is_sha256(self.artifact_sha256):
            raise ValueError("candidate artifact digest must be SHA-256")
        expected = _candidate_id(
            self.kind,
            self.artifact_sha256,
            self.report,
            self.metadata,
        )
        if self.candidate_id != expected:
            raise ValueError("candidate id does not match its sealed identity")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def passed(self) -> bool:
        return not self.report.rejection_reasons(self.kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "report": self.report.to_dict(),
            "passed": self.passed,
            "rejection_reasons": list(self.report.rejection_reasons(self.kind)),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


class ArtifactPromotionStore:
    """Digest-bound candidate staging with explicit human activation."""

    ALLOWED_KINDS = {"controller", "macro", "semantic_model"}

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.candidates_root = self.root / "candidates"
        self.active_path = self.root / "active.json"

    def stage(
        self,
        artifact_path: str | Path,
        *,
        kind: str,
        report: PromotionGateReport,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactCandidate:
        normalized_kind = kind.strip().lower()
        if normalized_kind not in self.ALLOWED_KINDS:
            raise ValueError(f"unsupported artifact kind: {kind!r}")
        artifact = self._contained_path(artifact_path)
        if not artifact.exists() or not (artifact.is_file() or artifact.is_dir()):
            raise ValueError("candidate artifact does not exist")
        artifact_digest = _artifact_sha256(artifact)
        candidate_metadata = dict(metadata or {})
        created_at = _timestamp()
        candidate_id = _candidate_id(
            normalized_kind,
            artifact_digest,
            report,
            candidate_metadata,
        )
        candidate = ArtifactCandidate(
            candidate_id=candidate_id,
            kind=normalized_kind,
            artifact_path=str(artifact.relative_to(self.root)).replace("\\", "/"),
            artifact_sha256=artifact_digest,
            report=report,
            created_at=created_at,
            metadata=candidate_metadata,
        )
        destination = self.candidates_root / f"{candidate_id}.json"
        _atomic_json(destination, candidate.to_dict())
        return candidate

    def load_candidate(self, candidate_id: str) -> ArtifactCandidate:
        if not _is_sha256(candidate_id):
            raise ValueError("candidate id must be a SHA-256 digest")
        path = self.candidates_root / f"{candidate_id}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise KeyError(candidate_id) from exc
        report = PromotionGateReport(**value["report"])
        candidate = ArtifactCandidate(
            candidate_id=value["candidate_id"],
            kind=value["kind"],
            artifact_path=value["artifact_path"],
            artifact_sha256=value["artifact_sha256"],
            report=report,
            created_at=value["created_at"],
            metadata=dict(value.get("metadata", {})),
        )
        if candidate.candidate_id != candidate_id:
            raise ValueError("candidate record id does not match its file name")
        return candidate

    def approve(
        self,
        candidate_id: str,
        *,
        reviewer: str,
        attestation: str,
    ) -> dict[str, Any]:
        if attestation != PROMOTION_APPROVAL_ATTESTATION:
            raise ValueError("artifact activation requires the exact human attestation")
        if not reviewer.strip():
            raise ValueError("artifact activation requires a reviewer identity")
        candidate = self.load_candidate(candidate_id)
        rejections = candidate.report.rejection_reasons(candidate.kind)
        if rejections:
            raise ValueError(
                "candidate failed promotion gates: " + ", ".join(rejections)
            )
        artifact = self._contained_path(candidate.artifact_path)
        if _artifact_sha256(artifact) != candidate.artifact_sha256:
            raise ValueError("candidate artifact digest changed after evaluation")
        previous = self.active() if self.active_path.exists() else None
        active = {
            "schema_version": "semop.active-artifact.v1",
            "candidate": candidate.to_dict(),
            "approved_by": reviewer.strip(),
            "approved_at": _timestamp(),
            "attestation": attestation,
            "previous": previous,
        }
        _atomic_json(self.active_path, active)
        return active

    def active(self) -> dict[str, Any]:
        try:
            value = json.loads(self.active_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise KeyError("no active artifact") from exc
        if not isinstance(value, dict):
            raise ValueError("active artifact pointer is not an object")
        return value

    def active_candidate(
        self,
        *,
        kind: str | None = None,
    ) -> tuple[ArtifactCandidate, Path]:
        active = self.active()
        if active.get("schema_version") != "semop.active-artifact.v1":
            raise ValueError("active artifact pointer has an unsupported schema")
        if active.get("attestation") != PROMOTION_APPROVAL_ATTESTATION:
            raise ValueError("active artifact pointer has no valid approval attestation")
        if not str(active.get("approved_by", "")).strip():
            raise ValueError("active artifact pointer has no reviewer identity")
        candidate_value = active.get("candidate")
        if not isinstance(candidate_value, Mapping):
            raise ValueError("active artifact pointer has no candidate")
        candidate_id = str(candidate_value.get("candidate_id", ""))
        candidate = self.load_candidate(candidate_id)
        canonical_candidate = json.loads(
            json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True)
        )
        if dict(candidate_value) != canonical_candidate:
            raise ValueError("active artifact candidate record does not match staging")
        if kind is not None and candidate.kind != kind.strip().lower():
            raise ValueError(f"active artifact is not a {kind} candidate")
        if not candidate.passed:
            raise ValueError("active artifact no longer passes promotion gates")
        artifact = self._contained_path(candidate.artifact_path)
        if _artifact_sha256(artifact) != candidate.artifact_sha256:
            raise ValueError("active artifact digest no longer matches")
        return candidate, artifact

    def rollback(self, *, reviewer: str, reason: str) -> dict[str, Any]:
        current = self.active()
        previous = current.get("previous")
        if not isinstance(previous, dict):
            raise ValueError("active artifact has no rollback target")
        restored = {
            **previous,
            "rollback": {
                "reviewer": reviewer.strip(),
                "reason": reason.strip(),
                "rolled_back_at": _timestamp(),
            },
        }
        if not reviewer.strip() or not reason.strip():
            raise ValueError("rollback requires reviewer and reason")
        _atomic_json(self.active_path, restored)
        return restored

    def _contained_path(self, value: str | Path) -> Path:
        path = Path(value)
        resolved = (
            path.resolve()
            if path.is_absolute()
            else (self.root / path).resolve()
        )
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("artifact path escapes the promotion store") from exc
        return resolved


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_sha256(path: Path) -> str:
    if path.is_file():
        if path.is_symlink():
            raise ValueError("candidate artifact cannot be a symbolic link")
        return _file_sha256(path)
    if not path.is_dir():
        raise ValueError("candidate artifact does not exist")
    files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
    if not files:
        raise ValueError("candidate artifact directory is empty")
    digest = sha256()
    for item in files:
        if item.is_symlink():
            raise ValueError("candidate artifact cannot contain symbolic links")
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = item.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(_file_sha256(item)))
    return digest.hexdigest()


def artifact_sha256(path: str | Path) -> str:
    """Return the deterministic digest used for promotion files and directories."""

    return _artifact_sha256(Path(path).resolve())


def _candidate_id(
    kind: str,
    artifact_sha256: str,
    report: PromotionGateReport,
    metadata: Mapping[str, Any],
) -> str:
    identity = json.dumps(
        {
            "kind": kind,
            "artifact_sha256": artifact_sha256,
            "report": report.to_dict(),
            "metadata": dict(metadata),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(identity).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage and explicitly activate digest-bound SemOp artifacts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/promotion/semantic-model"),
        help="promotion store directory",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    stage = commands.add_parser("stage", help="stage a candidate without activating it")
    stage.add_argument("artifact", type=Path)
    stage.add_argument("--kind", choices=sorted(ArtifactPromotionStore.ALLOWED_KINDS), required=True)
    stage.add_argument("--report", type=Path, required=True)
    stage.add_argument("--metadata", type=Path)

    approve = commands.add_parser("approve", help="activate a passing staged candidate")
    approve.add_argument("candidate_id")
    approve.add_argument("--reviewer", required=True)
    approve.add_argument(
        "--confirm",
        action="store_true",
        help="explicitly attest that the sealed report was reviewed",
    )

    commands.add_parser("status", help="verify and print the active artifact")

    rollback = commands.add_parser("rollback", help="restore the previous active artifact")
    rollback.add_argument("--reviewer", required=True)
    rollback.add_argument("--reason", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    store = ArtifactPromotionStore(args.root)
    try:
        if args.command == "stage":
            artifact_path = (
                args.artifact.resolve()
                if args.artifact.is_absolute()
                else (store.root / args.artifact).resolve()
            )
            report_value = _read_json_object(args.report)
            if isinstance(report_value.get("report"), Mapping):
                report_value = dict(report_value["report"])
            report = PromotionGateReport(**report_value)
            metadata = (
                _read_json_object(args.metadata) if args.metadata is not None else {}
            )
            metadata.update(
                {
                    key: value
                    for key, value in _training_metadata(artifact_path).items()
                    if key not in metadata
                }
            )
            result: Mapping[str, Any] = store.stage(
                artifact_path,
                kind=args.kind,
                report=report,
                metadata=metadata,
            ).to_dict()
            exit_code = 0 if result["passed"] else 2
        elif args.command == "approve":
            if not args.confirm:
                raise ValueError("approval requires --confirm after human review")
            result = store.approve(
                args.candidate_id,
                reviewer=args.reviewer,
                attestation=PROMOTION_APPROVAL_ATTESTATION,
            )
            exit_code = 0
        elif args.command == "status":
            candidate, artifact = store.active_candidate()
            result = {
                "active": store.active(),
                "verified_candidate": candidate.to_dict(),
                "artifact_path": str(artifact),
            }
            exit_code = 0
        else:
            result = store.rollback(reviewer=args.reviewer, reason=args.reason)
            exit_code = 0
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"SemOp promotion error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return dict(value)


def _training_metadata(artifact: Path) -> dict[str, Any]:
    summary_path = artifact / "semop_training_summary.json"
    if not summary_path.is_file():
        return {}
    summary = _read_json_object(summary_path)
    mapping = {
        "base_model_id": summary.get("student_model_id"),
        "base_model_revision": summary.get("student_model_revision"),
        "training_corpus_sha256": summary.get("corpus_sha256"),
    }
    return {key: value for key, value in mapping.items() if value is not None}


if __name__ == "__main__":
    raise SystemExit(main())

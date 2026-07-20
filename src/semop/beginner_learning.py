from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from .kernel import (
    DomainKind,
    ExperienceReviewCorpusGrounder,
    RawExperienceGrounder,
    ReviewedExperienceRuleLearningLoop,
    ReviewedExperienceRuleLearningResult,
    TypedExperienceStore,
    UnifiedTypedReasoner,
    VerifiedRuleLearningLoop,
    VerifiedRuleLibrary,
    canonical_json,
)


@dataclass(frozen=True)
class BeginnerRuleCheckpoint:
    artifact_path: Path
    artifact_sha256: str
    library_sha256: str
    rule_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": self.artifact_sha256,
            "library_sha256": self.library_sha256,
            "rule_count": self.rule_count,
        }


@dataclass(frozen=True)
class LoadedBeginnerRuleLibrary:
    library: VerifiedRuleLibrary
    checkpoint: BeginnerRuleCheckpoint | None = None


@dataclass(frozen=True)
class BeginnerReviewedLearningRun:
    reviewed_records: int
    result: ReviewedExperienceRuleLearningResult
    checkpoint: BeginnerRuleCheckpoint | None = None

    @property
    def promoted(self) -> bool:
        return self.result.promoted


def load_beginner_rule_library(
    artifact_path: str | Path,
) -> LoadedBeginnerRuleLibrary:
    artifact = Path(artifact_path)
    manifest = _manifest_path(artifact)
    if not artifact.exists() and not manifest.exists():
        return LoadedBeginnerRuleLibrary(VerifiedRuleLibrary())
    if not artifact.is_file() or not manifest.is_file():
        raise ValueError(
            "beginner learned-rule artifact and checkpoint must both exist"
        )
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid beginner learned-rule checkpoint") from exc
    if not isinstance(value, dict) or value.get("format_version") != 1:
        raise ValueError("unsupported beginner learned-rule checkpoint")
    if value.get("artifact_name") != artifact.name:
        raise ValueError("beginner learned-rule artifact name mismatch")
    artifact_sha256 = _require_sha256(
        value.get("artifact_sha256"),
        "artifact",
    )
    library_sha256 = _require_sha256(
        value.get("library_sha256"),
        "library",
    )
    rule_count = value.get("rule_count")
    if type(rule_count) is not int or rule_count < 0:
        raise ValueError("beginner learned-rule count is invalid")
    library = VerifiedRuleLearningLoop.load_active(
        artifact,
        expected_sha256=artifact_sha256,
    )
    if library.artifact_sha256 != library_sha256:
        raise ValueError("beginner learned-rule library hash mismatch")
    if len(library.records) != rule_count:
        raise ValueError("beginner learned-rule count mismatch")
    checkpoint = BeginnerRuleCheckpoint(
        artifact_path=artifact.resolve(),
        artifact_sha256=artifact_sha256,
        library_sha256=library_sha256,
        rule_count=rule_count,
    )
    return LoadedBeginnerRuleLibrary(library, checkpoint)


def run_beginner_reviewed_learning(
    store: TypedExperienceStore,
    reasoner: UnifiedTypedReasoner,
    artifact_path: str | Path,
    *,
    incumbent_library: VerifiedRuleLibrary | None = None,
    required_domains: Sequence[DomainKind | str] = (
        DomainKind.LANGUAGE,
        DomainKind.MATH,
        DomainKind.VISION,
    ),
) -> BeginnerReviewedLearningRun:
    corpus = store.export_reviewed()
    grounder = ExperienceReviewCorpusGrounder(
        RawExperienceGrounder(reasoner, hard_negatives_per_example=0)
    )
    result = ReviewedExperienceRuleLearningLoop(grounder=grounder).run(
        corpus,
        namespace="beginner-reviewed-rule-learning",
        required_domains=required_domains,
        incumbent_library=incumbent_library,
    )
    checkpoint = _persist_promoted(result, Path(artifact_path))
    return BeginnerReviewedLearningRun(
        reviewed_records=len(corpus.records),
        result=result,
        checkpoint=checkpoint,
    )


def _persist_promoted(
    result: ReviewedExperienceRuleLearningResult,
    artifact_path: Path,
) -> BeginnerRuleCheckpoint | None:
    persisted = VerifiedRuleLearningLoop.persist_promoted(
        result.learning,
        artifact_path,
    )
    if persisted is None:
        return None
    checkpoint = BeginnerRuleCheckpoint(
        artifact_path=persisted.artifact_path,
        artifact_sha256=persisted.artifact_sha256,
        library_sha256=persisted.library_sha256,
        rule_count=persisted.rule_count,
    )
    manifest = _manifest_path(artifact_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(f".{manifest.name}.{uuid4().hex}.tmp")
    payload = {
        "format_version": 1,
        "artifact_name": artifact_path.name,
        "artifact_sha256": checkpoint.artifact_sha256,
        "library_sha256": checkpoint.library_sha256,
        "rule_count": checkpoint.rule_count,
    }
    try:
        temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        temporary.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint


def _manifest_path(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(".checkpoint.json")


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"beginner learned-rule {label} hash is invalid")
    return normalized

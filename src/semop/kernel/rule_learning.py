from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from time import process_time
from typing import Any
from uuid import uuid4

from .engine import OperatorKernel
from .model import SolveBudget, SolveResult
from .rule_discovery import (
    RuleActivationIssue,
    RuleDiscoveryBudget,
    RuleDiscoveryResult,
    RuleReviewEvidence,
    VerifiedRuleDiscovery,
    VerifiedRuleLibrary,
    rule_discovery_semantic_fingerprint,
    rule_discovery_task_digest,
)
from .self_learning import (
    LearningMetrics,
    LearningSplit,
    LearningTask,
    SemanticLabelAuthority,
    TaskEvaluation,
    summarize_task_evaluations,
    task_evaluation_from_result,
)


@dataclass(frozen=True)
class RuleLearningBudget:
    """Promotion gates for a jointly activated discovered-rule library."""

    discovery_budget: RuleDiscoveryBudget = field(
        default_factory=RuleDiscoveryBudget
    )
    joint_solve_budget: SolveBudget = field(default_factory=SolveBudget)
    required_labeled_outcome_accuracy: float = 1.0
    required_semantic_correctness: float = 1.0
    required_replay_integrity: float = 1.0
    max_false_positives: int = 0
    max_positive_regressions: int = 0
    min_new_goal_completions: int = 1
    max_joint_tasks: int = 256
    max_library_rules: int = 256
    require_every_new_rule_used: bool = True
    require_every_training_domain_improved: bool = True

    def __post_init__(self) -> None:
        rates = (
            self.required_labeled_outcome_accuracy,
            self.required_semantic_correctness,
            self.required_replay_integrity,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("rule-learning rate gates must be between zero and one")
        if self.max_false_positives < 0 or self.max_positive_regressions < 0:
            raise ValueError("rule-learning failure limits must not be negative")
        if self.min_new_goal_completions <= 0:
            raise ValueError("rule learning requires a positive completion gain")
        if self.max_joint_tasks <= 0 or self.max_library_rules <= 0:
            raise ValueError("rule-learning resource limits must be positive")


@dataclass(frozen=True)
class RuleLibraryActivationAudit:
    task_id: str
    domain: str
    activated_operator_names: tuple[str, ...]
    issues: tuple[RuleActivationIssue, ...]


@dataclass(frozen=True)
class RuleLibraryEvaluation:
    metrics: LearningMetrics
    tasks: tuple[TaskEvaluation, ...]
    activation_audit: tuple[RuleLibraryActivationAudit, ...]
    task_digests: tuple[tuple[str, str], ...]
    results: dict[str, SolveResult] = field(compare=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(legacy_aliases=False),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "expected_solved": task.expected_solved,
                    "success": task.success,
                    "verified": task.verified,
                    "false_positive": task.false_positive,
                    "expansions": task.expansions,
                    "proof_steps": task.proof_steps,
                    "cpu_seconds": task.cpu_seconds,
                    "halt_reason": task.halt_reason,
                }
                for task in self.tasks
            ],
            "activation_audit": [
                {
                    "task_id": item.task_id,
                    "domain": item.domain,
                    "activated_operator_names": list(
                        item.activated_operator_names
                    ),
                    "issues": [
                        {
                            "hypothesis_id": issue.hypothesis_id,
                            "reason": issue.reason,
                        }
                        for issue in item.issues
                    ],
                }
                for item in self.activation_audit
            ],
            "task_digests": dict(self.task_digests),
        }


@dataclass(frozen=True)
class RuleLibraryCheckpoint:
    artifact_path: Path
    artifact_sha256: str
    library_sha256: str
    rule_count: int


@dataclass(frozen=True)
class RulePromotionCertificate:
    incumbent_library_sha256: str
    candidate_library_sha256: str
    new_rule_ids: tuple[str, ...]
    used_new_rule_ids: tuple[str, ...]
    newly_solved_task_ids: tuple[str, ...]
    improved_domains: tuple[str, ...]
    joint_review_evidence: tuple[RuleReviewEvidence, ...]
    labeled_outcome_accuracy: float
    semantic_correctness: float
    replay_integrity: float
    false_positive_task_ids: tuple[str, ...] = ()
    positive_regression_task_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, digest in (
            ("incumbent", self.incumbent_library_sha256),
            ("candidate", self.candidate_library_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    f"rule promotion certificate has invalid {name} digest"
                )
        rates = (
            self.labeled_outcome_accuracy,
            self.semantic_correctness,
            self.replay_integrity,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("rule promotion certificate rates are invalid")
        if not self.new_rule_ids or not self.newly_solved_task_ids:
            raise ValueError("rule promotion certificate requires a capability gain")
        if not set(self.new_rule_ids).issubset(self.used_new_rule_ids):
            raise ValueError("rule promotion certificate contains unused new rules")
        if self.false_positive_task_ids or self.positive_regression_task_ids:
            raise ValueError("rule promotion certificate must be regression-free")
        evidence_ids = tuple(
            item.task_id for item in self.joint_review_evidence
        )
        if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(
                "rule promotion certificate requires unique joint review evidence"
            )
        if not any(item.expected_solved for item in self.joint_review_evidence):
            raise ValueError("rule promotion certificate requires joint positives")
        if not any(
            not item.expected_solved for item in self.joint_review_evidence
        ):
            raise ValueError("rule promotion certificate requires joint negatives")
        if not set(self.newly_solved_task_ids).issubset(evidence_ids):
            raise ValueError(
                "rule promotion certificate gain lacks joint review evidence"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RulePromotionCertificate":
        return cls(
            incumbent_library_sha256=str(value["incumbent_library_sha256"]),
            candidate_library_sha256=str(value["candidate_library_sha256"]),
            new_rule_ids=tuple(value["new_rule_ids"]),
            used_new_rule_ids=tuple(value["used_new_rule_ids"]),
            newly_solved_task_ids=tuple(value["newly_solved_task_ids"]),
            improved_domains=tuple(value["improved_domains"]),
            joint_review_evidence=tuple(
                RuleReviewEvidence(**item)
                for item in value["joint_review_evidence"]
            ),
            labeled_outcome_accuracy=float(
                value["labeled_outcome_accuracy"]
            ),
            semantic_correctness=float(value["semantic_correctness"]),
            replay_integrity=float(value["replay_integrity"]),
            false_positive_task_ids=tuple(
                value.get("false_positive_task_ids", ())
            ),
            positive_regression_task_ids=tuple(
                value.get("positive_regression_task_ids", ())
            ),
        )


@dataclass(frozen=True)
class RuleLearningResult:
    promoted: bool
    rejection_reasons: tuple[str, ...]
    discovery: RuleDiscoveryResult
    incumbent_library: VerifiedRuleLibrary = field(compare=False, repr=False)
    candidate_library: VerifiedRuleLibrary = field(compare=False, repr=False)
    active_library: VerifiedRuleLibrary = field(compare=False, repr=False)
    baseline: RuleLibraryEvaluation
    candidate: RuleLibraryEvaluation
    candidate_executed: bool
    newly_solved_task_ids: tuple[str, ...]
    positive_regression_task_ids: tuple[str, ...]
    false_positive_task_ids: tuple[str, ...]
    replay_failure_task_ids: tuple[str, ...]
    used_new_rule_ids: tuple[str, ...]
    improved_domains: tuple[str, ...]
    promotion_certificate: RulePromotionCertificate | None = None

    def __post_init__(self) -> None:
        if self.promoted:
            if not self.candidate_executed:
                raise ValueError("an unexecuted rule candidate cannot be promoted")
            if self.rejection_reasons or self.promotion_certificate is None:
                raise ValueError(
                    "promoted rule result requires a clean promotion certificate"
                )
            if self.active_library != self.candidate_library:
                raise ValueError("promoted rule result must activate the candidate")
            if (
                self.promotion_certificate.candidate_library_sha256
                != self.candidate_library.artifact_sha256
            ):
                raise ValueError("rule promotion certificate/library mismatch")
        else:
            if not self.rejection_reasons or self.promotion_certificate is not None:
                raise ValueError(
                    "rejected rule result requires reasons and no certificate"
                )
            if self.active_library != self.incumbent_library:
                raise ValueError("rejected rule result must preserve the incumbent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "rejection_reasons": list(self.rejection_reasons),
            "candidate_executed": self.candidate_executed,
            "discovery": self.discovery.to_dict(),
            "libraries": {
                "incumbent_rules": len(self.incumbent_library.records),
                "candidate_rules": len(self.candidate_library.records),
                "active_rules": len(self.active_library.records),
                "active_sha256": self.active_library.artifact_sha256,
            },
            "newly_solved_task_ids": list(self.newly_solved_task_ids),
            "positive_regression_task_ids": list(
                self.positive_regression_task_ids
            ),
            "false_positive_task_ids": list(self.false_positive_task_ids),
            "replay_failure_task_ids": list(self.replay_failure_task_ids),
            "used_new_rule_ids": list(self.used_new_rule_ids),
            "improved_domains": list(self.improved_domains),
            "promotion_certificate": (
                self.promotion_certificate.to_dict()
                if self.promotion_certificate is not None
                else None
            ),
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
        }


class VerifiedRuleLearningLoop:
    """Discover rules, then promote only a jointly replayed final library."""

    def __init__(self, budget: RuleLearningBudget | None = None) -> None:
        self.budget = budget or RuleLearningBudget()

    def run(
        self,
        training_tasks: Sequence[LearningTask],
        validation_tasks: Sequence[LearningTask],
        candidate_heldout_tasks: Sequence[LearningTask],
        joint_heldout_tasks: Sequence[LearningTask],
        *,
        incumbent_library: VerifiedRuleLibrary | None = None,
    ) -> RuleLearningResult:
        training = tuple(training_tasks)
        validation = tuple(validation_tasks)
        candidate_heldout = tuple(candidate_heldout_tasks)
        joint_heldout = tuple(joint_heldout_tasks)
        incumbent = incumbent_library or VerifiedRuleLibrary()
        if len(joint_heldout) > self.budget.max_joint_tasks:
            raise ValueError(
                "rule-learning joint task limit exceeded: "
                f"{len(joint_heldout)} > {self.budget.max_joint_tasks}"
            )
        if len(incumbent.records) > self.budget.max_library_rules:
            raise ValueError(
                "rule-learning incumbent rule limit exceeded: "
                f"{len(incumbent.records)} > {self.budget.max_library_rules}"
            )
        _validate_joint_splits(
            training,
            validation,
            candidate_heldout,
            joint_heldout,
        )

        discovery_training = _augment_tasks(training, incumbent)
        discovery_validation = _augment_tasks(validation, incumbent)
        discovery_heldout = _augment_tasks(candidate_heldout, incumbent)
        discovery = VerifiedRuleDiscovery(self.budget.discovery_budget).discover(
            discovery_training,
            discovery_validation,
            discovery_heldout,
        )
        candidate_library = _merge_libraries(incumbent, discovery.library)

        baseline = _evaluate_library(
            joint_heldout,
            incumbent,
            self.budget.joint_solve_budget,
        )
        candidate_rule_limit_exceeded = (
            len(candidate_library.records) > self.budget.max_library_rules
        )
        candidate = (
            baseline
            if candidate_rule_limit_exceeded
            else _evaluate_library(
                joint_heldout,
                candidate_library,
                self.budget.joint_solve_budget,
            )
        )
        baseline_by_id = {item.task_id: item for item in baseline.tasks}
        candidate_by_id = {item.task_id: item for item in candidate.tasks}

        newly_solved = tuple(
            sorted(
                task.task_id
                for task in joint_heldout
                if task.expected_solved
                and not (
                    baseline_by_id[task.task_id].success
                    and baseline_by_id[task.task_id].verified
                )
                and candidate_by_id[task.task_id].success
                and candidate_by_id[task.task_id].verified
            )
        )
        positive_regressions = tuple(
            sorted(
                task.task_id
                for task in joint_heldout
                if task.expected_solved
                and baseline_by_id[task.task_id].success
                and baseline_by_id[task.task_id].verified
                and not (
                    candidate_by_id[task.task_id].success
                    and candidate_by_id[task.task_id].verified
                )
            )
        )
        false_positives = tuple(
            sorted(
                task.task_id
                for task in joint_heldout
                if not task.expected_solved
                and candidate_by_id[task.task_id].success
            )
        )
        replay_failures = tuple(
            sorted(
                task.task_id
                for task in joint_heldout
                if candidate_by_id[task.task_id].success
                and not candidate_by_id[task.task_id].verified
            )
        )
        new_rule_ids = {
            record.hypothesis.hypothesis_id
            for record in discovery.library.records
        }
        used_new_rules = tuple(
            sorted(
                {
                    step.action.operator.name
                    for task in joint_heldout
                    if task.expected_solved
                    for step in candidate.results[task.task_id].proof
                    if step.action.operator.name in new_rule_ids
                }
            )
        )
        improved_domains = tuple(
            sorted(
                {
                    task.domain
                    for task in joint_heldout
                    if task.task_id in newly_solved
                }
            )
        )

        reasons = list(
            _promotion_rejections(
                discovery,
                baseline,
                candidate,
                newly_solved,
                positive_regressions,
                false_positives,
                replay_failures,
                used_new_rules,
                improved_domains,
                self.budget,
            )
        )
        if candidate_rule_limit_exceeded:
            reasons.insert(
                0,
                "candidate_joint_rule_limit_exceeded: "
                f"{len(candidate_library.records)} > "
                f"{self.budget.max_library_rules}",
            )
        resolved_reasons = tuple(dict.fromkeys(reasons))
        promoted = not resolved_reasons
        certificate = (
            _build_promotion_certificate(
                incumbent,
                candidate_library,
                discovery,
                joint_heldout,
                candidate,
                newly_solved,
                false_positives,
                positive_regressions,
                used_new_rules,
                improved_domains,
            )
            if promoted
            else None
        )
        return RuleLearningResult(
            promoted=promoted,
            rejection_reasons=resolved_reasons,
            discovery=discovery,
            incumbent_library=incumbent,
            candidate_library=candidate_library,
            active_library=candidate_library if promoted else incumbent,
            baseline=baseline,
            candidate=candidate,
            candidate_executed=not candidate_rule_limit_exceeded,
            newly_solved_task_ids=newly_solved,
            positive_regression_task_ids=positive_regressions,
            false_positive_task_ids=false_positives,
            replay_failure_task_ids=replay_failures,
            used_new_rule_ids=used_new_rules,
            improved_domains=improved_domains,
            promotion_certificate=certificate,
        )

    @staticmethod
    def persist_promoted(
        result: RuleLearningResult,
        path: str | Path,
    ) -> RuleLibraryCheckpoint | None:
        """Atomically replace an artifact only when the joint gate promoted it."""

        if not result.promoted:
            return None
        if result.promotion_certificate is None:
            raise ValueError("promoted rule result lacks a promotion certificate")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{uuid4().hex}.tmp"
        )
        artifact = _promoted_artifact(result)
        try:
            temporary.write_bytes(artifact)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return RuleLibraryCheckpoint(
            artifact_path=destination.resolve(),
            artifact_sha256=sha256(artifact).hexdigest(),
            library_sha256=result.active_library.artifact_sha256,
            rule_count=len(result.active_library.records),
        )

    @staticmethod
    def load_active(
        path: str | Path,
        *,
        expected_sha256: str,
    ) -> VerifiedRuleLibrary:
        artifact = Path(path).read_bytes()
        if sha256(artifact).hexdigest() != expected_sha256.lower():
            raise ValueError("promoted rule artifact hash mismatch")
        try:
            payload = json.loads(artifact.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid promoted rule artifact") from exc
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("unsupported promoted rule artifact format")
        certificate_value = payload.get("promotion_certificate")
        library_value = payload.get("library")
        if not isinstance(certificate_value, dict) or not isinstance(
            library_value, dict
        ):
            raise ValueError("promoted rule artifact is incomplete")
        certificate = RulePromotionCertificate.from_dict(certificate_value)
        library_artifact = json.dumps(
            library_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        library = VerifiedRuleLibrary.from_artifact(
            library_artifact,
            expected_sha256=certificate.candidate_library_sha256,
        )
        library_ids = {
            record.hypothesis.hypothesis_id for record in library.records
        }
        if not set(certificate.new_rule_ids).issubset(library_ids):
            raise ValueError("promoted rule artifact omits certified new rules")
        return library


def _build_promotion_certificate(
    incumbent: VerifiedRuleLibrary,
    candidate_library: VerifiedRuleLibrary,
    discovery: RuleDiscoveryResult,
    joint_heldout: tuple[LearningTask, ...],
    candidate: RuleLibraryEvaluation,
    newly_solved: tuple[str, ...],
    false_positives: tuple[str, ...],
    positive_regressions: tuple[str, ...],
    used_new_rules: tuple[str, ...],
    improved_domains: tuple[str, ...],
) -> RulePromotionCertificate:
    semantic_correctness = candidate.metrics.semantic_correctness
    if semantic_correctness is None:
        raise ValueError("promoted rule result lacks semantic correctness")
    review_evidence = tuple(
        RuleReviewEvidence(
            task_id=task.task_id,
            case_digest=task.label_evidence.case_digest,
            grounded_task_digest=rule_discovery_task_digest(task),
            domain=task.domain,
            expected_solved=task.expected_solved,
            reviewer=task.label_evidence.reviewer,
            reviewed_at=task.label_evidence.reviewed_at,
            attestation=task.label_evidence.attestation,
        )
        for task in sorted(joint_heldout, key=lambda item: item.task_id)
    )
    return RulePromotionCertificate(
        incumbent_library_sha256=incumbent.artifact_sha256,
        candidate_library_sha256=candidate_library.artifact_sha256,
        new_rule_ids=tuple(
            record.hypothesis.hypothesis_id
            for record in discovery.library.records
        ),
        used_new_rule_ids=used_new_rules,
        newly_solved_task_ids=newly_solved,
        improved_domains=improved_domains,
        joint_review_evidence=review_evidence,
        labeled_outcome_accuracy=candidate.metrics.labeled_outcome_accuracy,
        semantic_correctness=semantic_correctness,
        replay_integrity=candidate.metrics.primitive_replay_integrity,
        false_positive_task_ids=false_positives,
        positive_regression_task_ids=positive_regressions,
    )


def _promoted_artifact(result: RuleLearningResult) -> bytes:
    certificate = result.promotion_certificate
    if certificate is None:
        raise ValueError("promoted rule result lacks a promotion certificate")
    payload = {
        "format_version": 1,
        "library": json.loads(result.active_library.artifact.decode("utf-8")),
        "promotion_certificate": certificate.to_dict(),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _augment_tasks(
    tasks: tuple[LearningTask, ...],
    library: VerifiedRuleLibrary,
) -> tuple[LearningTask, ...]:
    if not library.records:
        return tasks
    return tuple(library.augment_task(task) for task in tasks)


def _merge_libraries(
    incumbent: VerifiedRuleLibrary,
    discovered: VerifiedRuleLibrary,
) -> VerifiedRuleLibrary:
    records = {
        record.hypothesis.hypothesis_id: record for record in incumbent.records
    }
    for record in discovered.records:
        identifier = record.hypothesis.hypothesis_id
        existing = records.get(identifier)
        if existing is not None and existing != record:
            raise ValueError(f"conflicting discovered rule record: {identifier}")
        records[identifier] = record
    return VerifiedRuleLibrary(tuple(records.values()))


def _evaluate_library(
    tasks: tuple[LearningTask, ...],
    library: VerifiedRuleLibrary,
    solve_budget: SolveBudget,
) -> RuleLibraryEvaluation:
    evaluations: list[TaskEvaluation] = []
    results: dict[str, SolveResult] = {}
    audits: list[RuleLibraryActivationAudit] = []
    digests: list[tuple[str, str]] = []
    for task in tasks:
        augmented = library.augment_task(task) if library.records else task
        activated = tuple(
            augmented.instance.metadata.get("activated_discovered_rules", ())
        )
        raw_issues = augmented.instance.metadata.get(
            "discovered_rule_activation_issues", ()
        )
        issues = tuple(
            RuleActivationIssue(
                hypothesis_id=str(item["hypothesis_id"]),
                reason=str(item["reason"]),
            )
            for item in raw_issues
        )
        started = process_time()
        result = OperatorKernel(augmented.instance.registry).solve(
            augmented.instance.state,
            augmented.instance.goals,
            budget=solve_budget,
        )
        elapsed = process_time() - started
        results[task.task_id] = result
        evaluations.append(task_evaluation_from_result(task, result, elapsed))
        audits.append(
            RuleLibraryActivationAudit(
                task_id=task.task_id,
                domain=task.domain,
                activated_operator_names=activated,
                issues=issues,
            )
        )
        digests.append((task.task_id, rule_discovery_task_digest(task)))
    resolved = tuple(evaluations)
    return RuleLibraryEvaluation(
        metrics=summarize_task_evaluations(resolved),
        tasks=resolved,
        activation_audit=tuple(audits),
        task_digests=tuple(sorted(digests)),
        results=results,
    )


def _promotion_rejections(
    discovery: RuleDiscoveryResult,
    baseline: RuleLibraryEvaluation,
    candidate: RuleLibraryEvaluation,
    newly_solved: tuple[str, ...],
    positive_regressions: tuple[str, ...],
    false_positives: tuple[str, ...],
    replay_failures: tuple[str, ...],
    used_new_rules: tuple[str, ...],
    improved_domains: tuple[str, ...],
    budget: RuleLearningBudget,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not discovery.library.records:
        reasons.append("no_heldout_retained_rule_candidates")
    incumbent_false_positives = tuple(
        item.task_id for item in baseline.tasks if item.false_positive
    )
    if incumbent_false_positives:
        reasons.append(
            "incumbent_joint_false_positive:"
            + ",".join(incumbent_false_positives)
        )
    if len(false_positives) > budget.max_false_positives:
        reasons.append(
            "candidate_joint_false_positive_limit_exceeded: "
            f"{len(false_positives)} > {budget.max_false_positives}:"
            + ",".join(false_positives)
        )
    if len(positive_regressions) > budget.max_positive_regressions:
        reasons.append(
            "candidate_joint_positive_regression_limit_exceeded: "
            f"{len(positive_regressions)} > "
            f"{budget.max_positive_regressions}:"
            + ",".join(positive_regressions)
        )
    if replay_failures:
        reasons.append(
            "candidate_joint_replay_failure:" + ",".join(replay_failures)
        )
    if (
        candidate.metrics.labeled_outcome_accuracy
        < budget.required_labeled_outcome_accuracy
    ):
        reasons.append(
            "candidate_joint_labeled_outcome_accuracy_below_gate: "
            f"{candidate.metrics.labeled_outcome_accuracy:.6f} < "
            f"{budget.required_labeled_outcome_accuracy:.6f}"
        )
    semantic_correctness = candidate.metrics.semantic_correctness
    if semantic_correctness is None:
        reasons.append("candidate_joint_semantic_correctness_unavailable")
    elif semantic_correctness < budget.required_semantic_correctness:
        reasons.append(
            "candidate_joint_semantic_correctness_below_gate: "
            f"{semantic_correctness:.6f} < "
            f"{budget.required_semantic_correctness:.6f}"
        )
    if (
        candidate.metrics.primitive_replay_integrity
        < budget.required_replay_integrity
    ):
        reasons.append(
            "candidate_joint_replay_integrity_below_gate: "
            f"{candidate.metrics.primitive_replay_integrity:.6f} < "
            f"{budget.required_replay_integrity:.6f}"
        )
    if len(newly_solved) < budget.min_new_goal_completions:
        reasons.append(
            "candidate_joint_new_completion_below_gate: "
            f"{len(newly_solved)} < {budget.min_new_goal_completions}"
        )
    new_rule_ids = {
        record.hypothesis.hypothesis_id for record in discovery.library.records
    }
    if budget.require_every_new_rule_used:
        unused = new_rule_ids.difference(used_new_rules)
        if unused:
            reasons.append(
                "candidate_joint_unused_new_rules:" + ",".join(sorted(unused))
            )
    if budget.require_every_training_domain_improved:
        required_domains = {
            domain
            for record in discovery.library.records
            for domain in record.hypothesis.training_domains
        }
        missing_domains = required_domains.difference(improved_domains)
        if missing_domains:
            reasons.append(
                "candidate_joint_unimproved_training_domains:"
                + ",".join(sorted(missing_domains))
            )
    required_domains = {
        domain
        for record in discovery.library.records
        for domain in record.hypothesis.training_domains
    }
    positive_domains = {
        item.domain for item in candidate.tasks if item.expected_solved
    }
    negative_domains = {
        item.domain for item in candidate.tasks if not item.expected_solved
    }
    missing_positive_domains = required_domains.difference(positive_domains)
    if missing_positive_domains:
        reasons.append(
            "candidate_joint_missing_positive_domains:"
            + ",".join(sorted(missing_positive_domains))
        )
    missing_negative_domains = required_domains.difference(negative_domains)
    if missing_negative_domains:
        reasons.append(
            "candidate_joint_missing_negative_domains:"
            + ",".join(sorted(missing_negative_domains))
        )
    return tuple(dict.fromkeys(reasons))


def _validate_joint_splits(
    training: tuple[LearningTask, ...],
    validation: tuple[LearningTask, ...],
    candidate_heldout: tuple[LearningTask, ...],
    joint_heldout: tuple[LearningTask, ...],
) -> None:
    if not training or not validation or not candidate_heldout or not joint_heldout:
        raise ValueError(
            "rule learning requires train, validation, candidate heldout, "
            "and joint heldout tasks"
        )
    if any(task.split is not LearningSplit.TRAIN for task in training):
        raise ValueError("rule-learning training tasks must use the train split")
    if any(
        task.split is not LearningSplit.HELDOUT
        for task in validation + candidate_heldout + joint_heldout
    ):
        raise ValueError("rule-learning evaluation tasks must use heldout split")
    if any(
        task.label_authority is not SemanticLabelAuthority.HUMAN_REVIEWED
        or not task.label_evidence.complete
        for task in joint_heldout
    ):
        raise ValueError(
            "rule-learning joint heldout requires digest-recorded human labels"
        )
    identifiers = tuple(
        task.task_id
        for task in training + validation + candidate_heldout + joint_heldout
    )
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("rule-learning task ids must be unique across all splits")
    prior_fingerprints = {
        rule_discovery_semantic_fingerprint(task)
        for task in training + validation + candidate_heldout
    }
    joint_fingerprints = tuple(
        rule_discovery_semantic_fingerprint(task) for task in joint_heldout
    )
    if len(joint_fingerprints) != len(set(joint_fingerprints)):
        raise ValueError("rule-learning joint heldout contains semantic duplicates")
    if prior_fingerprints.intersection(joint_fingerprints):
        raise ValueError("rule-learning final joint holdout semantic overlap")

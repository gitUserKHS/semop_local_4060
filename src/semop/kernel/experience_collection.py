from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from uuid import uuid4

from .contracts import DomainInstance
from .engine import ActionPolicy, RegistryPolicyProvider
from .experience import (
    GroundedLearningBatch,
    RawExperienceGrounder,
    grounded_instance_fingerprint,
)
from .experience_queue import (
    ExperienceCaptureResult,
    ExperienceProposalAuthority,
    ExperienceReviewCorpus,
    ExperienceSplitRole,
    ExperienceTrigger,
    TypedExperienceStore,
    observation_from_request,
)
from .model import FactStatus, SolveBudget
from .rule_discovery import VerifiedRuleLibrary
from .rule_learning import RuleLearningResult, VerifiedRuleLearningLoop
from .runtime import (
    DomainKind,
    TypedDomainRequest,
    UnifiedTypedReasoner,
    UnifiedTypedResult,
)
from .semantic_codec import semantic_request_digest
from .self_learning import LearningTask


@dataclass(frozen=True)
class ExperienceCollectorConfig:
    capture_successes: bool = False
    capture_matching_proposals: bool = True

    def __post_init__(self) -> None:
        if type(self.capture_successes) is not bool:
            raise TypeError("capture_successes must be boolean")
        if type(self.capture_matching_proposals) is not bool:
            raise TypeError("capture_matching_proposals must be boolean")


@dataclass(frozen=True)
class ExperienceCollectedRun:
    request_digest: str
    result: UnifiedTypedResult | None = field(compare=False, repr=False)
    capture: ExperienceCaptureResult | None = field(compare=False, repr=False)
    trigger: ExperienceTrigger | None = None
    error_type: str = ""
    error_message: str = ""

    @property
    def captured(self) -> bool:
        return self.capture is not None

    @property
    def succeeded(self) -> bool:
        return bool(self.result and self.result.success and self.result.verified)


class TypedExperienceCollector:
    """Run the production LMV boundary and append only noteworthy observations."""

    def __init__(
        self,
        store: TypedExperienceStore,
        *,
        reasoner: UnifiedTypedReasoner | None = None,
        config: ExperienceCollectorConfig | None = None,
    ) -> None:
        self.store = store
        self.reasoner = reasoner or UnifiedTypedReasoner()
        self.config = config or ExperienceCollectorConfig()

    def run(
        self,
        request: TypedDomainRequest,
        *,
        proposed_expected_solved: bool | None = None,
        proposal_authority: ExperienceProposalAuthority | str = (
            ExperienceProposalAuthority.UNKNOWN
        ),
        rationale: str = "",
        source: str = "typed-runtime",
        trigger: ExperienceTrigger | str | None = None,
        event_id: str | None = None,
        created_at: str | None = None,
        policy: ActionPolicy | RegistryPolicyProvider | None = None,
        budget: SolveBudget | None = None,
    ) -> ExperienceCollectedRun:
        request_digest = semantic_request_digest(request)
        authority = ExperienceProposalAuthority(proposal_authority)
        if proposed_expected_solved is not None and type(
            proposed_expected_solved
        ) is not bool:
            raise TypeError("experience proposed outcome must be boolean or null")
        if (
            proposed_expected_solved is None
            and authority is not ExperienceProposalAuthority.UNKNOWN
        ):
            raise ValueError(
                "experience proposal authority requires a proposed label"
            )
        if (
            proposed_expected_solved is not None
            and authority is ExperienceProposalAuthority.UNKNOWN
        ):
            raise ValueError("experience proposed labels require their authority")
        if proposed_expected_solved is not None and not rationale.strip():
            raise ValueError("experience proposed labels require a rationale")
        explicit_trigger = ExperienceTrigger(trigger) if trigger is not None else None
        resolved_event_id = event_id or f"experience-{uuid4().hex}"
        try:
            result = self.reasoner.run(request, policy=policy, budget=budget)
        except Exception as exc:
            resolved_trigger = explicit_trigger or ExperienceTrigger.RUNTIME_EXCEPTION
            observation = observation_from_request(
                request,
                event_id=resolved_event_id,
                observed_success=None,
                observed_verified=False,
                proposed_expected_solved=proposed_expected_solved,
                proposal_authority=authority,
                trigger=resolved_trigger,
                rationale=rationale,
                source=source,
                error_type=type(exc).__name__,
                error_message=str(exc),
                created_at=created_at,
            )
            capture = self.store.add_observation(observation)
            return ExperienceCollectedRun(
                request_digest=request_digest,
                result=None,
                capture=capture,
                trigger=resolved_trigger,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        typed = result.typed_result
        if typed is None:
            resolved_trigger = explicit_trigger or ExperienceTrigger.GROUNDING_FAILURE
            observation = observation_from_request(
                request,
                event_id=resolved_event_id,
                observed_success=None,
                observed_verified=False,
                proposed_expected_solved=proposed_expected_solved,
                proposal_authority=authority,
                trigger=resolved_trigger,
                rationale=rationale,
                source=source,
                error_type="NoTypedResult",
                error_message="; ".join(result.diagnostics),
                created_at=created_at,
            )
            capture = self.store.add_observation(observation)
            return ExperienceCollectedRun(
                request_digest=request_digest,
                result=result,
                capture=capture,
                trigger=resolved_trigger,
                error_type="NoTypedResult",
                error_message="; ".join(result.diagnostics),
            )

        if result.instance is None:
            raise ValueError("typed experience result lacks its grounded instance")
        grounding_concerns = _grounding_concerns(result.instance)
        resolved_trigger = explicit_trigger or _infer_trigger(
            typed.success,
            typed.verified,
            proposed_expected_solved,
            grounding_uncertainty=bool(grounding_concerns),
        )
        should_capture = (
            explicit_trigger is not None
            or not typed.success
            or not typed.verified
            or bool(grounding_concerns)
            or (
                proposed_expected_solved is not None
                and (
                    self.config.capture_matching_proposals
                    or proposed_expected_solved != typed.success
                )
            )
            or self.config.capture_successes
        )
        if not should_capture:
            return ExperienceCollectedRun(
                request_digest=request_digest,
                result=result,
                capture=None,
                trigger=None,
            )
        observation = observation_from_request(
            request,
            event_id=resolved_event_id,
            observed_success=typed.success,
            observed_verified=typed.verified,
            proposed_expected_solved=proposed_expected_solved,
            proposal_authority=authority,
            trigger=resolved_trigger,
            rationale=_append_grounding_concerns(rationale, grounding_concerns),
            source=source,
            grounded_fingerprint=grounded_instance_fingerprint(result.instance),
            proof_program=tuple(
                step.action.operator.name for step in typed.proof
            ),
            halt_reason=typed.halt_reason,
            expansions=typed.expansions,
            created_at=created_at,
        )
        return ExperienceCollectedRun(
            request_digest=request_digest,
            result=result,
            capture=self.store.add_observation(observation),
            trigger=resolved_trigger,
        )


@dataclass(frozen=True)
class ExperienceGroundedRuleCorpus:
    corpus: ExperienceReviewCorpus
    training: GroundedLearningBatch
    validation: GroundedLearningBatch
    candidate_heldout: GroundedLearningBatch
    joint_heldout: GroundedLearningBatch
    input_overlap: tuple[str, ...] = ()
    semantic_overlap: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return all(
            batch.complete
            for batch in (
                self.training,
                self.validation,
                self.candidate_heldout,
                self.joint_heldout,
            )
        )

    @property
    def leakage_free(self) -> bool:
        return not self.input_overlap and not self.semantic_overlap

    def require_ready(
        self,
    ) -> tuple[
        tuple[LearningTask, ...],
        tuple[LearningTask, ...],
        tuple[LearningTask, ...],
        tuple[LearningTask, ...],
    ]:
        batches = (
            self.training,
            self.validation,
            self.candidate_heldout,
            self.joint_heldout,
        )
        labels = tuple(role.value for role in ExperienceSplitRole)
        for label, batch in zip(labels, batches, strict=True):
            tasks = batch.require_complete()
            if not tasks:
                raise ValueError(f"reviewed experience {label} split is empty")
        if self.input_overlap:
            raise ValueError(
                "reviewed experience input overlap: "
                + ", ".join(self.input_overlap)
            )
        if self.semantic_overlap:
            raise ValueError(
                "reviewed experience semantic overlap: "
                + ", ".join(self.semantic_overlap)
            )
        return (
            self.training.tasks,
            self.validation.tasks,
            self.candidate_heldout.tasks,
            self.joint_heldout.tasks,
        )


class ExperienceReviewCorpusGrounder:
    def __init__(
        self,
        grounder: RawExperienceGrounder | None = None,
        *,
        max_examples_per_role: int = 1_000,
    ) -> None:
        self.grounder = grounder or RawExperienceGrounder(
            hard_negatives_per_example=0
        )
        if max_examples_per_role <= 0:
            raise ValueError("experience role example limit must be positive")
        self.max_examples_per_role = max_examples_per_role

    def ground(
        self,
        corpus: ExperienceReviewCorpus,
        *,
        namespace: str = "reviewed-experience",
        required_domains: Sequence[DomainKind | str] = (
            DomainKind.LANGUAGE,
            DomainKind.MATH,
            DomainKind.VISION,
        ),
    ) -> ExperienceGroundedRuleCorpus:
        normalized_namespace = namespace.strip()
        if not normalized_namespace:
            raise ValueError("reviewed experience namespace must not be empty")
        present_domains = {record.case.domain for record in corpus.records}
        missing_domains = tuple(
            DomainKind(domain).value
            for domain in required_domains
            if DomainKind(domain) not in present_domains
        )
        if missing_domains:
            raise ValueError(
                "reviewed experience corpus is missing domains: "
                + ", ".join(missing_domains)
            )
        batches: dict[ExperienceSplitRole, GroundedLearningBatch] = {}
        for role in ExperienceSplitRole:
            examples = corpus.raw_examples(role)
            if len(examples) > self.max_examples_per_role:
                raise ValueError(
                    f"reviewed experience {role.value} example limit exceeded"
                )
            batches[role] = self.grounder.ground(
                examples,
                namespace=f"{normalized_namespace}-{role.value}",
            )
        input_overlap = _cross_role_overlap(
            batches,
            attribute="input_fingerprints",
        )
        semantic_overlap = _cross_role_overlap(
            batches,
            attribute="semantic_fingerprints",
        )
        return ExperienceGroundedRuleCorpus(
            corpus=corpus,
            training=batches[ExperienceSplitRole.TRAIN],
            validation=batches[ExperienceSplitRole.VALIDATION],
            candidate_heldout=batches[ExperienceSplitRole.CANDIDATE_HELDOUT],
            joint_heldout=batches[ExperienceSplitRole.JOINT_HELDOUT],
            input_overlap=input_overlap,
            semantic_overlap=semantic_overlap,
        )


@dataclass(frozen=True)
class ReviewedExperienceRuleLearningResult:
    grounding: ExperienceGroundedRuleCorpus
    learning: RuleLearningResult = field(compare=False, repr=False)

    @property
    def promoted(self) -> bool:
        return self.learning.promoted


class ReviewedExperienceRuleLearningLoop:
    """Export approved online experience into the four-stage rule gate."""

    def __init__(
        self,
        *,
        grounder: ExperienceReviewCorpusGrounder | None = None,
        learning_loop: VerifiedRuleLearningLoop | None = None,
    ) -> None:
        self.grounder = grounder or ExperienceReviewCorpusGrounder()
        self.learning_loop = learning_loop or VerifiedRuleLearningLoop()

    def run(
        self,
        corpus: ExperienceReviewCorpus,
        *,
        namespace: str = "reviewed-experience-rule-learning",
        required_domains: Sequence[DomainKind | str] = (
            DomainKind.LANGUAGE,
            DomainKind.MATH,
            DomainKind.VISION,
        ),
        incumbent_library: VerifiedRuleLibrary | None = None,
    ) -> ReviewedExperienceRuleLearningResult:
        grounding = self.grounder.ground(
            corpus,
            namespace=namespace,
            required_domains=required_domains,
        )
        training, validation, candidate_heldout, joint_heldout = (
            grounding.require_ready()
        )
        learning = self.learning_loop.run(
            training,
            validation,
            candidate_heldout,
            joint_heldout,
            incumbent_library=incumbent_library,
        )
        return ReviewedExperienceRuleLearningResult(grounding, learning)


def _infer_trigger(
    success: bool,
    verified: bool,
    proposed_expected_solved: bool | None,
    *,
    grounding_uncertainty: bool = False,
) -> ExperienceTrigger:
    if proposed_expected_solved is not None and proposed_expected_solved != success:
        return ExperienceTrigger.EXPECTATION_MISMATCH
    if success and not verified:
        return ExperienceTrigger.REPLAY_FAILURE
    if not success:
        return ExperienceTrigger.UNSOLVED
    if grounding_uncertainty:
        return ExperienceTrigger.GROUNDING_UNCERTAINTY
    return ExperienceTrigger.SUCCESS_SAMPLE


def _grounding_concerns(instance: DomainInstance) -> tuple[str, ...]:
    concerns: list[str] = []
    unparsed_count = len(instance.metadata.get("unparsed_statements", ()))
    if unparsed_count:
        concerns.append(f"unparsed_statements={unparsed_count}")
    proposed_count = sum(
        record.fact is not None and record.fact.status is FactStatus.PROPOSED
        for record in instance.grounding_trace.records
    )
    if proposed_count:
        concerns.append(f"proposed_groundings={proposed_count}")
    return tuple(concerns)


def _append_grounding_concerns(
    rationale: str,
    concerns: Sequence[str],
) -> str:
    normalized = rationale.strip()
    if not concerns:
        return normalized
    grounding_note = "runtime grounding uncertainty: " + ", ".join(concerns)
    return f"{normalized} | {grounding_note}" if normalized else grounding_note


def _cross_role_overlap(
    batches: dict[ExperienceSplitRole, GroundedLearningBatch],
    *,
    attribute: str,
) -> tuple[str, ...]:
    overlap: list[str] = []
    for left, right in combinations(tuple(ExperienceSplitRole), 2):
        left_values = {
            fingerprint
            for _identifier, fingerprint in getattr(batches[left], attribute)
        }
        right_values = {
            fingerprint
            for _identifier, fingerprint in getattr(batches[right], attribute)
        }
        overlap.extend(
            f"{left.value}/{right.value}:{fingerprint}"
            for fingerprint in sorted(left_values.intersection(right_values))
        )
    return tuple(overlap)

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Sequence

from .adapters import MigrationMode
from .contracts import DomainInstance
from .engine import ActionPolicy
from .library import operator_schema_fingerprint
from .runtime import DomainKind, TypedDomainRequest, UnifiedTypedReasoner
from .self_learning import (
    LearningSplit,
    LearningTask,
    SemanticLabelAuthority,
    SemanticLabelEvidence,
    SelfLearningLoop,
    SelfLearningResult,
)
from .synthesis import with_hard_negative_distractors


@dataclass(frozen=True)
class RawLearningExample:
    """One labeled raw request before typed grounding and verifier execution."""

    example_id: str
    request: TypedDomainRequest = field(compare=False, repr=False)
    expected_solved: bool = True
    split: LearningSplit = LearningSplit.TRAIN
    source: str = "verifier"
    capability: str = ""
    structure_key: str = ""
    difficulty: int = 1
    label_authority: SemanticLabelAuthority = SemanticLabelAuthority.PROGRAMMATIC
    label_evidence: SemanticLabelEvidence = SemanticLabelEvidence()

    def __post_init__(self) -> None:
        example_id = self.example_id.strip()
        if not example_id:
            raise ValueError("raw learning example id must not be empty")
        if not isinstance(self.request, TypedDomainRequest):
            raise TypeError("raw learning example requires a TypedDomainRequest")
        if self.difficulty <= 0:
            raise ValueError("raw learning example difficulty must be positive")
        object.__setattr__(self, "example_id", example_id)
        object.__setattr__(self, "split", LearningSplit(self.split))
        object.__setattr__(
            self,
            "label_authority",
            SemanticLabelAuthority(self.label_authority),
        )
        if not isinstance(self.label_evidence, SemanticLabelEvidence):
            raise TypeError("raw learning label_evidence must be SemanticLabelEvidence")
        if self.label_authority is SemanticLabelAuthority.HUMAN_REVIEWED:
            if not self.label_evidence.complete:
                raise ValueError(
                    "human-reviewed raw examples require digest-bound label evidence"
                )
        elif self.label_evidence.complete:
            raise ValueError(
                "digest-bound label evidence requires HUMAN_REVIEWED authority"
            )


@dataclass(frozen=True)
class GroundingFailure:
    example_id: str
    domain: str
    error_type: str
    message: str


class RawGroundingError(ValueError):
    def __init__(self, failures: Sequence[GroundingFailure]) -> None:
        self.failures = tuple(failures)
        summary = "; ".join(
            f"{failure.example_id}: {failure.error_type}: {failure.message}"
            for failure in self.failures
        )
        super().__init__("raw learning grounding failed: " + summary)


@dataclass(frozen=True)
class GroundedLearningBatch:
    tasks: tuple[LearningTask, ...]
    failures: tuple[GroundingFailure, ...]
    input_fingerprints: tuple[tuple[str, str], ...]
    semantic_fingerprints: tuple[tuple[str, str], ...]

    @property
    def complete(self) -> bool:
        return not self.failures

    def require_complete(self) -> tuple[LearningTask, ...]:
        if self.failures:
            raise RawGroundingError(self.failures)
        return self.tasks


@dataclass(frozen=True)
class GroundedLearningSplit:
    training: GroundedLearningBatch
    heldout: GroundedLearningBatch
    input_overlap: tuple[str, ...] = ()
    semantic_overlap: tuple[str, ...] = ()

    @property
    def leakage_free(self) -> bool:
        return not self.input_overlap and not self.semantic_overlap

    def require_ready(self) -> tuple[tuple[LearningTask, ...], tuple[LearningTask, ...]]:
        training = self.training.require_complete()
        heldout = self.heldout.require_complete()
        if self.input_overlap:
            raise ValueError(
                "raw learning train/heldout input overlap: "
                + ", ".join(self.input_overlap)
            )
        if self.semantic_overlap:
            raise ValueError(
                "raw learning train/heldout semantic overlap: "
                + ", ".join(self.semantic_overlap)
            )
        return training, heldout


@dataclass(frozen=True)
class RawSelfLearningResult:
    """Audited grounding and verifier-gated learning from one raw run."""

    grounding: GroundedLearningSplit
    learning: SelfLearningResult = field(compare=False, repr=False)

    @property
    def promoted(self) -> bool:
        return self.learning.promoted


class RawExperienceGrounder:
    """Convert raw LMV requests into audited tasks without trusting model output."""

    def __init__(
        self,
        reasoner: UnifiedTypedReasoner | None = None,
        *,
        hard_negatives_per_example: int = 4,
    ) -> None:
        if not 0 <= hard_negatives_per_example <= 4:
            raise ValueError("raw experience hard negatives must be between zero and four")
        self.reasoner = reasoner or UnifiedTypedReasoner()
        self.hard_negatives_per_example = hard_negatives_per_example

    def ground(
        self,
        examples: Sequence[RawLearningExample],
        *,
        namespace: str = "raw-experience",
    ) -> GroundedLearningBatch:
        resolved = tuple(examples)
        normalized_namespace = namespace.strip()
        if not normalized_namespace:
            raise ValueError("raw experience namespace must not be empty")
        identifiers = [example.example_id for example in resolved]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("raw learning example ids must be unique")

        tasks: list[LearningTask] = []
        failures: list[GroundingFailure] = []
        input_fingerprints: list[tuple[str, str]] = []
        semantic_fingerprints: list[tuple[str, str]] = []
        for example in resolved:
            try:
                mode = MigrationMode(example.request.mode)
                if mode is MigrationMode.LEGACY:
                    raise ValueError("legacy mode cannot produce verifier training data")
                if isinstance(example.request.payload, DomainInstance):
                    raise ValueError(
                        "raw experience requires an adapter input, not a prebuilt "
                        "DomainInstance"
                    )
                domain = DomainKind(example.request.domain)
                input_fingerprint = _input_fingerprint(example.request)
                instance = self.reasoner.ground(example.request)
                if not instance.goals:
                    raise ValueError("grounded raw request has no explicit typed goals")
                semantic_fingerprint = _semantic_fingerprint(instance)
                if self.hard_negatives_per_example:
                    instance = with_hard_negative_distractors(
                        instance,
                        _distractor_index(normalized_namespace, example.example_id),
                        count=self.hard_negatives_per_example,
                    )
                instance = replace(
                    instance,
                    metadata={
                        **instance.metadata,
                        "experience_input_fingerprint": input_fingerprint,
                        "experience_semantic_fingerprint": semantic_fingerprint,
                        "experience_source_domain": domain.value,
                        "experience_hard_negatives": self.hard_negatives_per_example,
                    },
                )
                tasks.append(
                    LearningTask(
                        task_id=f"{normalized_namespace}:{example.example_id}",
                        instance=instance,
                        expected_solved=example.expected_solved,
                        split=example.split,
                        source=example.source,
                        domain=domain.value,
                        capability=example.capability,
                        structure_key=example.structure_key,
                        difficulty=example.difficulty,
                        label_authority=example.label_authority,
                        label_evidence=example.label_evidence,
                    )
                )
                input_fingerprints.append((example.example_id, input_fingerprint))
                semantic_fingerprints.append(
                    (example.example_id, semantic_fingerprint)
                )
            except Exception as exc:
                failures.append(
                    GroundingFailure(
                        example_id=example.example_id,
                        domain=str(example.request.domain),
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
        return GroundedLearningBatch(
            tasks=tuple(tasks),
            failures=tuple(failures),
            input_fingerprints=tuple(input_fingerprints),
            semantic_fingerprints=tuple(semantic_fingerprints),
        )

    def ground_split(
        self,
        training: Sequence[RawLearningExample],
        heldout: Sequence[RawLearningExample],
        *,
        namespace: str = "raw-experience",
    ) -> GroundedLearningSplit:
        normalized_namespace = namespace.strip()
        if not normalized_namespace:
            raise ValueError("raw experience namespace must not be empty")
        training_examples = tuple(training)
        heldout_examples = tuple(heldout)
        if not training_examples:
            raise ValueError("raw experience training split must not be empty")
        if not heldout_examples:
            raise ValueError("raw experience heldout split must not be empty")
        _require_split(training_examples, LearningSplit.TRAIN, "training")
        _require_split(heldout_examples, LearningSplit.HELDOUT, "heldout")
        all_identifiers = tuple(
            example.example_id for example in training_examples + heldout_examples
        )
        if len(all_identifiers) != len(set(all_identifiers)):
            raise ValueError("raw learning example ids must be unique across splits")
        training_batch = self.ground(
            training_examples,
            namespace=f"{normalized_namespace}-train",
        )
        heldout_batch = self.ground(
            heldout_examples,
            namespace=f"{normalized_namespace}-heldout",
        )
        input_overlap = _fingerprint_overlap(
            training_batch.input_fingerprints,
            heldout_batch.input_fingerprints,
        )
        semantic_overlap = _fingerprint_overlap(
            training_batch.semantic_fingerprints,
            heldout_batch.semantic_fingerprints,
        )
        return GroundedLearningSplit(
            training=training_batch,
            heldout=heldout_batch,
            input_overlap=input_overlap,
            semantic_overlap=semantic_overlap,
        )


class RawSelfLearningLoop:
    """Ground raw examples, audit the split, and run verifier-gated learning."""

    def __init__(
        self,
        learning_loop: SelfLearningLoop | None = None,
        *,
        grounder: RawExperienceGrounder | None = None,
    ) -> None:
        self.learning_loop = learning_loop or SelfLearningLoop()
        self.grounder = grounder or RawExperienceGrounder()

    def run(
        self,
        training: Sequence[RawLearningExample],
        heldout: Sequence[RawLearningExample],
        *,
        namespace: str = "raw-self-learning",
        incumbent_policy: ActionPolicy | None = None,
        resume: bool = False,
    ) -> RawSelfLearningResult:
        grounding = self.grounder.ground_split(
            training,
            heldout,
            namespace=namespace,
        )
        training_tasks, heldout_tasks = grounding.require_ready()
        learning = self.learning_loop.run(
            training_tasks,
            heldout_tasks,
            incumbent_policy=incumbent_policy,
            resume=resume,
        )
        return RawSelfLearningResult(grounding=grounding, learning=learning)


def _require_split(
    examples: Sequence[RawLearningExample],
    expected: LearningSplit,
    label: str,
) -> None:
    mismatches = tuple(
        example.example_id for example in examples if example.split is not expected
    )
    if mismatches:
        raise ValueError(
            f"raw experience {label} examples have the wrong split: "
            + ", ".join(mismatches)
        )


def _input_fingerprint(request: TypedDomainRequest) -> str:
    payload = request.payload
    image = getattr(payload, "image", None)
    digest = getattr(image, "digest", None)
    if callable(digest):
        material = (
            f"{DomainKind(request.domain).value}|{type(payload).__name__}|"
            f"{digest()}|{getattr(payload, 'goals', ())!r}|"
            f"{getattr(payload, 'query', '')!r}|"
            f"{getattr(payload, 'config', None)!r}|"
            f"{getattr(payload, 'count_selectors', ())!r}"
        )
    elif isinstance(payload, DomainInstance):
        material = "prebuilt|" + _semantic_fingerprint(payload)
    else:
        material = (
            f"{DomainKind(request.domain).value}|{type(payload).__name__}|"
            f"{payload!r}"
        )
    return sha256(material.encode("utf-8")).hexdigest()


def _semantic_fingerprint(instance: DomainInstance) -> str:
    facts = tuple(
        (
            fact.atom.canonical_key(),
            fact.status.value,
            fact.source,
            fact.confidence,
            fact.assertion_status.value,
            fact.evidence_status.value,
        )
        for fact in instance.state.facts
    )
    operators = tuple(
        operator_schema_fingerprint(operator)
        for operator in sorted(
            instance.registry.operators.values(),
            key=lambda item: item.name,
        )
    )
    material = repr(
        (
            facts,
            tuple(goal.atom.canonical_key() for goal in instance.goals),
            operators,
        )
    )
    return sha256(material.encode("utf-8")).hexdigest()


def _distractor_index(namespace: str, example_id: str) -> int:
    digest = sha256(f"{namespace}:{example_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _fingerprint_overlap(
    training: Sequence[tuple[str, str]],
    heldout: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    training_values = {fingerprint for _identifier, fingerprint in training}
    heldout_values = {fingerprint for _identifier, fingerprint in heldout}
    return tuple(sorted(training_values.intersection(heldout_values)))

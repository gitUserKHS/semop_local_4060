from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from itertools import combinations, permutations, product
import json
from math import factorial
from typing import Any

from .contracts import DomainInstance
from .engine import OperatorKernel
from .model import (
    Atom,
    OperatorFamily,
    OperatorSpec,
    Rule,
    SolveBudget,
    Variable,
)
from .registry import KernelRegistry
from .self_learning import (
    HUMAN_REVIEW_ATTESTATION,
    LearningSplit,
    LearningTask,
    SemanticLabelAuthority,
    SemanticLabelEvidence,
)


@dataclass(frozen=True)
class RuleDiscoveryBudget:
    """Bounds and evidence requirements for one-step typed Horn induction."""

    min_training_support: int = 3
    min_validation_positive: int = 1
    min_validation_negative: int = 1
    min_heldout_positive: int = 1
    min_heldout_negative: int = 1
    max_preconditions: int = 3
    max_variables: int = 8
    max_variable_permutations: int = 720
    max_candidates: int = 128
    solve_budget: SolveBudget = field(
        default_factory=lambda: SolveBudget(
            max_steps=32,
            max_expansions=20_000,
            timeout_seconds=10.0,
        )
    )

    def __post_init__(self) -> None:
        positive = (
            self.min_training_support,
            self.min_validation_positive,
            self.min_validation_negative,
            self.min_heldout_positive,
            self.min_heldout_negative,
            self.max_preconditions,
            self.max_variables,
            self.max_variable_permutations,
            self.max_candidates,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("rule-discovery limits must be positive")


@dataclass(frozen=True, order=True)
class RuleParameterTemplate:
    name: str
    type_name: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.type_name.strip():
            raise ValueError("rule-template parameter fields must not be empty")


@dataclass(frozen=True, order=True)
class RuleAtomTemplate:
    predicate: str
    arguments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.predicate.strip():
            raise ValueError("rule-template predicate must not be empty")
        if not self.arguments or any(not item.strip() for item in self.arguments):
            raise ValueError("rule-template arguments must not be empty")


@dataclass(frozen=True)
class TypedRuleHypothesis:
    hypothesis_id: str
    parameters: tuple[RuleParameterTemplate, ...]
    preconditions: tuple[RuleAtomTemplate, ...]
    effect: RuleAtomTemplate
    training_task_ids: tuple[str, ...]
    training_domains: tuple[str, ...]
    description_ko: str = "검증된 사례에서 귀납한 {operator} 규칙을 적용했다."

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("rule hypothesis id must not be empty")
        if not self.parameters or not self.preconditions:
            raise ValueError("rule hypothesis requires parameters and preconditions")
        parameter_names = tuple(item.name for item in self.parameters)
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("rule hypothesis parameter names must be unique")
        declared = set(parameter_names)
        used_preconditions = {
            name for atom in self.preconditions for name in atom.arguments
        }
        used_effect = set(self.effect.arguments)
        if not used_preconditions.union(used_effect).issubset(declared):
            raise ValueError("rule hypothesis references undeclared parameters")
        if used_preconditions.union(used_effect) != declared:
            raise ValueError("rule hypothesis contains unused parameters")
        if not used_effect.issubset(used_preconditions):
            raise ValueError("rule hypothesis effect variables must be premise-bound")
        if len(self.preconditions) != len(set(self.preconditions)):
            raise ValueError("rule hypothesis preconditions must be unique")
        if not self.training_task_ids:
            raise ValueError("rule hypothesis requires training support")
        if len(self.training_task_ids) != len(set(self.training_task_ids)):
            raise ValueError("rule hypothesis training task ids must be unique")
        if not self.training_domains or any(
            not domain.strip() for domain in self.training_domains
        ):
            raise ValueError("rule hypothesis requires non-empty training domains")
        if len(self.training_domains) != len(set(self.training_domains)):
            raise ValueError("rule hypothesis training domains must be unique")

    @property
    def signature(self) -> str:
        return _template_signature(
            self.parameters,
            self.preconditions,
            self.effect,
        )

    def instantiate(
        self,
        registry: KernelRegistry,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> OperatorSpec:
        return _instantiate_hypothesis(
            self,
            registry,
            retained=False,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TypedRuleHypothesis":
        return cls(
            hypothesis_id=str(value["hypothesis_id"]),
            parameters=tuple(
                RuleParameterTemplate(**item) for item in value["parameters"]
            ),
            preconditions=tuple(
                RuleAtomTemplate(
                    predicate=str(item["predicate"]),
                    arguments=tuple(item["arguments"]),
                )
                for item in value["preconditions"]
            ),
            effect=RuleAtomTemplate(
                predicate=str(value["effect"]["predicate"]),
                arguments=tuple(value["effect"]["arguments"]),
            ),
            training_task_ids=tuple(value["training_task_ids"]),
            training_domains=tuple(value["training_domains"]),
            description_ko=str(
                value.get(
                    "description_ko",
                    "검증된 사례에서 귀납한 {operator} 규칙을 적용했다.",
                )
            ),
        )


@dataclass(frozen=True, order=True)
class RuleReviewEvidence:
    task_id: str
    case_digest: str
    grounded_task_digest: str
    domain: str
    expected_solved: bool
    reviewer: str
    reviewed_at: str
    attestation: str = HUMAN_REVIEW_ATTESTATION

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("rule review evidence task id must not be empty")
        grounded_digest = self.grounded_task_digest.strip().lower()
        if len(grounded_digest) != 64 or any(
            character not in "0123456789abcdef" for character in grounded_digest
        ):
            raise ValueError(
                "rule review evidence requires a grounded-task SHA-256 digest"
            )
        domain = self.domain.strip()
        if not domain:
            raise ValueError("rule review evidence domain must not be empty")
        if type(self.expected_solved) is not bool:
            raise TypeError("rule review evidence expected_solved must be boolean")
        evidence = SemanticLabelEvidence(
            case_digest=self.case_digest,
            reviewer=self.reviewer,
            reviewed_at=self.reviewed_at,
            attestation=self.attestation,
        )
        if not evidence.complete:
            raise ValueError("rule review evidence must be complete")
        object.__setattr__(self, "case_digest", evidence.case_digest)
        object.__setattr__(self, "grounded_task_digest", grounded_digest)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "reviewer", evidence.reviewer)
        object.__setattr__(self, "reviewed_at", evidence.reviewed_at)
        object.__setattr__(self, "attestation", evidence.attestation)


@dataclass(frozen=True)
class RuleEvaluationSummary:
    compatible_task_ids: tuple[str, ...] = ()
    newly_solved_positive_task_ids: tuple[str, ...] = ()
    preserved_positive_task_ids: tuple[str, ...] = ()
    checked_negative_task_ids: tuple[str, ...] = ()
    false_positive_task_ids: tuple[str, ...] = ()
    regression_task_ids: tuple[str, ...] = ()
    replay_failure_task_ids: tuple[str, ...] = ()
    unreviewed_task_ids: tuple[str, ...] = ()
    unsolved_positive_task_ids: tuple[str, ...] = ()
    review_evidence: tuple[RuleReviewEvidence, ...] = ()

    @property
    def clean(self) -> bool:
        return not (
            self.false_positive_task_ids
            or self.regression_task_ids
            or self.replay_failure_task_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuleDiscoveryRecord:
    hypothesis: TypedRuleHypothesis
    accepted: bool
    rejection_reasons: tuple[str, ...]
    validation: RuleEvaluationSummary
    heldout: RuleEvaluationSummary

    def __post_init__(self) -> None:
        expected_id = "induced_rule_" + sha256(
            self.hypothesis.signature.encode("utf-8")
        ).hexdigest()[:16]
        if self.hypothesis.hypothesis_id != expected_id:
            raise ValueError("rule hypothesis id does not match its typed signature")
        if self.accepted:
            if self.rejection_reasons:
                raise ValueError("accepted rule record cannot contain rejections")
            if not self.validation.clean or not self.heldout.clean:
                raise ValueError("accepted rule record must have clean evaluations")
            for stage, summary in (
                ("validation", self.validation),
                ("heldout", self.heldout),
            ):
                if not summary.newly_solved_positive_task_ids:
                    raise ValueError(
                        f"accepted rule record needs {stage} positive support"
                    )
                if not summary.checked_negative_task_ids:
                    raise ValueError(
                        f"accepted rule record needs {stage} negative support"
                    )
                evidence_by_id = {
                    item.task_id: item for item in summary.review_evidence
                }
                if len(evidence_by_id) != len(summary.review_evidence):
                    raise ValueError(
                        f"accepted rule record has duplicate {stage} review evidence"
                    )
                required_ids = set(summary.newly_solved_positive_task_ids).union(
                    summary.checked_negative_task_ids
                )
                if required_ids != set(evidence_by_id):
                    raise ValueError(
                        f"accepted rule record lacks {stage} review evidence"
                    )
                if any(
                    not evidence_by_id[task_id].expected_solved
                    for task_id in summary.newly_solved_positive_task_ids
                ) or any(
                    evidence_by_id[task_id].expected_solved
                    for task_id in summary.checked_negative_task_ids
                ):
                    raise ValueError(
                        f"accepted rule record has inconsistent {stage} labels"
                    )
                positive_domains = {
                    evidence_by_id[task_id].domain
                    for task_id in summary.newly_solved_positive_task_ids
                }
                negative_domains = {
                    evidence_by_id[task_id].domain
                    for task_id in summary.checked_negative_task_ids
                }
                required_domains = set(self.hypothesis.training_domains)
                if not required_domains.issubset(positive_domains):
                    raise ValueError(
                        f"accepted rule record lacks {stage} positive domains"
                    )
                if not required_domains.issubset(negative_domains):
                    raise ValueError(
                        f"accepted rule record lacks {stage} negative domains"
                    )
        elif not self.rejection_reasons:
            raise ValueError("rejected rule record requires a rejection reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis.to_dict(),
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "validation": asdict(self.validation),
            "heldout": asdict(self.heldout),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuleDiscoveryRecord":
        accepted = value["accepted"]
        if type(accepted) is not bool:
            raise TypeError("rule discovery accepted flag must be boolean")
        return cls(
            hypothesis=TypedRuleHypothesis.from_dict(value["hypothesis"]),
            accepted=accepted,
            rejection_reasons=tuple(value["rejection_reasons"]),
            validation=_summary_from_dict(value["validation"]),
            heldout=_summary_from_dict(value["heldout"]),
        )


@dataclass(frozen=True)
class RuleActivationIssue:
    hypothesis_id: str
    reason: str


@dataclass(frozen=True)
class RuleActivationResult:
    activated_operator_names: tuple[str, ...]
    issues: tuple[RuleActivationIssue, ...]


@dataclass(frozen=True)
class VerifiedRuleLibrary:
    """Only heldout-falsified rule records may cross the execution boundary."""

    FORMAT_VERSION = 1

    records: tuple[RuleDiscoveryRecord, ...] = ()

    def __post_init__(self) -> None:
        if any(not record.accepted for record in self.records):
            raise ValueError("verified rule library cannot contain rejected records")
        identifiers = tuple(
            record.hypothesis.hypothesis_id for record in self.records
        )
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("verified rule library hypothesis ids must be unique")
        object.__setattr__(
            self,
            "records",
            tuple(sorted(self.records, key=lambda item: item.hypothesis.hypothesis_id)),
        )

    @property
    def artifact(self) -> bytes:
        payload = {
            "format_version": self.FORMAT_VERSION,
            "records": [record.to_dict() for record in self.records],
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @property
    def artifact_sha256(self) -> str:
        return sha256(self.artifact).hexdigest()

    @classmethod
    def from_artifact(
        cls,
        artifact: bytes,
        *,
        expected_sha256: str | None = None,
    ) -> "VerifiedRuleLibrary":
        if expected_sha256 is not None:
            actual = sha256(artifact).hexdigest()
            if actual != expected_sha256.lower():
                raise ValueError("verified rule library artifact hash mismatch")
        try:
            payload = json.loads(artifact.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid verified rule library artifact") from exc
        if not isinstance(payload, dict):
            raise ValueError("verified rule library artifact must be an object")
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported verified rule library format")
        return cls(
            tuple(
                RuleDiscoveryRecord.from_dict(item)
                for item in payload.get("records", ())
            )
        )

    def activate_into(self, registry: KernelRegistry) -> RuleActivationResult:
        activated: list[str] = []
        issues: list[RuleActivationIssue] = []
        for record in self.records:
            hypothesis = record.hypothesis
            if hypothesis.hypothesis_id in registry.operators:
                issues.append(
                    RuleActivationIssue(hypothesis.hypothesis_id, "operator_name_exists")
                )
                continue
            try:
                operator = _instantiate_hypothesis(
                    hypothesis,
                    registry,
                    retained=True,
                    metadata={
                        "validation_positive": str(
                            len(record.validation.newly_solved_positive_task_ids)
                        ),
                        "heldout_positive": str(
                            len(record.heldout.newly_solved_positive_task_ids)
                        ),
                        "validation_negative": str(
                            len(record.validation.checked_negative_task_ids)
                        ),
                        "heldout_negative": str(
                            len(record.heldout.checked_negative_task_ids)
                        ),
                        "verified_rule_library_sha256": self.artifact_sha256,
                    },
                )
                registry.register_operator(operator)
            except Exception as exc:
                issues.append(
                    RuleActivationIssue(
                        hypothesis.hypothesis_id,
                        f"incompatible_schema:{type(exc).__name__}:{exc}",
                    )
                )
                continue
            activated.append(hypothesis.hypothesis_id)
        return RuleActivationResult(tuple(activated), tuple(issues))

    def augment_instance(self, instance: DomainInstance) -> DomainInstance:
        registry = _clone_registry(instance.registry)
        activation = self.activate_into(registry)
        return replace(
            instance,
            registry=registry,
            metadata={
                **instance.metadata,
                "verified_rule_library_sha256": self.artifact_sha256,
                "activated_discovered_rules": activation.activated_operator_names,
                "discovered_rule_activation_issues": tuple(
                    asdict(issue) for issue in activation.issues
                ),
            },
        )

    def augment_task(self, task: LearningTask) -> LearningTask:
        return replace(task, instance=self.augment_instance(task.instance))


@dataclass(frozen=True)
class RuleDiscoveryResult:
    records: tuple[RuleDiscoveryRecord, ...]
    library: VerifiedRuleLibrary
    ignored_training_tasks: tuple[tuple[str, str], ...] = ()

    @property
    def accepted_records(self) -> tuple[RuleDiscoveryRecord, ...]:
        return tuple(record for record in self.records if record.accepted)

    @property
    def rejected_records(self) -> tuple[RuleDiscoveryRecord, ...]:
        return tuple(record for record in self.records if not record.accepted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_records": [record.to_dict() for record in self.records],
            "accepted_rules": len(self.accepted_records),
            "rejected_rules": len(self.rejected_records),
            "ignored_training_tasks": [
                list(item) for item in self.ignored_training_tasks
            ],
            "library": {
                "format_version": self.library.FORMAT_VERSION,
                "records": len(self.library.records),
                "artifact_bytes": len(self.library.artifact),
                "artifact_sha256": self.library.artifact_sha256,
            },
        }


class VerifiedRuleDiscovery:
    """Induce flat typed Horn rules and retain only human-falsified candidates."""

    def __init__(self, budget: RuleDiscoveryBudget | None = None) -> None:
        self.budget = budget or RuleDiscoveryBudget()

    def discover(
        self,
        training_tasks: Sequence[LearningTask],
        validation_tasks: Sequence[LearningTask],
        heldout_tasks: Sequence[LearningTask],
    ) -> RuleDiscoveryResult:
        training = tuple(training_tasks)
        validation = tuple(validation_tasks)
        heldout = tuple(heldout_tasks)
        _validate_rule_discovery_splits(training, validation, heldout)

        support: dict[
            str,
            tuple[
                tuple[RuleParameterTemplate, ...],
                tuple[RuleAtomTemplate, ...],
                RuleAtomTemplate,
                set[str],
                set[str],
            ],
        ] = {}
        ignored: list[tuple[str, str]] = []
        for task in training:
            if not task.expected_solved:
                ignored.append((task.task_id, "training_negative_not_a_proposal"))
                continue
            if len(task.instance.goals) != 1:
                ignored.append((task.task_id, "requires_single_goal"))
                continue
            baseline = _solve_task(task, task.instance.registry, self.budget.solve_budget)
            if baseline.success:
                ignored.append((task.task_id, "already_solved_by_registered_operators"))
                continue
            templates = _templates_from_task(task, self.budget)
            if not templates:
                ignored.append((task.task_id, "no_bounded_flat_rule_template"))
                continue
            existing = _existing_operator_signatures(task.instance.registry)
            for parameters, preconditions, effect in templates:
                signature = _template_signature(parameters, preconditions, effect)
                if signature in existing:
                    continue
                item = support.get(signature)
                if item is None:
                    item = (parameters, preconditions, effect, set(), set())
                    support[signature] = item
                item[3].add(task.task_id)
                item[4].add(task.domain)

        ranked = sorted(
            (
                (signature, item)
                for signature, item in support.items()
                if len(item[3]) >= self.budget.min_training_support
            ),
            key=lambda pair: (
                -len(pair[1][3]),
                len(pair[1][1]),
                pair[0],
            ),
        )[: self.budget.max_candidates]

        validation_baselines = _baseline_results(validation, self.budget.solve_budget)
        heldout_baselines = _baseline_results(heldout, self.budget.solve_budget)
        records: list[RuleDiscoveryRecord] = []
        for signature, item in ranked:
            parameters, preconditions, effect, task_ids, domains = item
            hypothesis = TypedRuleHypothesis(
                hypothesis_id="induced_rule_" + sha256(
                    signature.encode("utf-8")
                ).hexdigest()[:16],
                parameters=parameters,
                preconditions=preconditions,
                effect=effect,
                training_task_ids=tuple(sorted(task_ids)),
                training_domains=tuple(sorted(domains)),
            )
            validation_summary = _evaluate_hypothesis(
                hypothesis,
                validation,
                validation_baselines,
                self.budget.solve_budget,
            )
            reasons = list(
                _evaluation_rejections(
                    "validation",
                    validation_summary,
                    minimum_positive=self.budget.min_validation_positive,
                    minimum_negative=self.budget.min_validation_negative,
                )
            )
            heldout_summary = RuleEvaluationSummary()
            if not reasons:
                heldout_summary = _evaluate_hypothesis(
                    hypothesis,
                    heldout,
                    heldout_baselines,
                    self.budget.solve_budget,
                )
                reasons.extend(
                    _evaluation_rejections(
                        "heldout",
                        heldout_summary,
                        minimum_positive=self.budget.min_heldout_positive,
                        minimum_negative=self.budget.min_heldout_negative,
                    )
                )
            records.append(
                RuleDiscoveryRecord(
                    hypothesis=hypothesis,
                    accepted=not reasons,
                    rejection_reasons=tuple(reasons),
                    validation=validation_summary,
                    heldout=heldout_summary,
                )
            )

        ordered = tuple(
            sorted(records, key=lambda record: record.hypothesis.hypothesis_id)
        )
        return RuleDiscoveryResult(
            records=ordered,
            library=VerifiedRuleLibrary(
                tuple(record for record in ordered if record.accepted)
            ),
            ignored_training_tasks=tuple(sorted(ignored)),
        )


def _templates_from_task(
    task: LearningTask,
    budget: RuleDiscoveryBudget,
) -> tuple[
    tuple[
        tuple[RuleParameterTemplate, ...],
        tuple[RuleAtomTemplate, ...],
        RuleAtomTemplate,
    ],
    ...,
]:
    goal = task.instance.goals[0].atom
    facts = tuple(
        sorted(
            {fact.atom for fact in task.instance.state.eligible_facts},
            key=Atom.canonical_key,
        )
    )
    goal_keys = {argument.canonical_key() for argument in goal.arguments}
    templates: dict[
        str,
        tuple[
            tuple[RuleParameterTemplate, ...],
            tuple[RuleAtomTemplate, ...],
            RuleAtomTemplate,
        ],
    ] = {}
    for count in range(1, min(budget.max_preconditions, len(facts)) + 1):
        for selected in combinations(facts, count):
            premise_keys = {
                argument.canonical_key()
                for atom in selected
                for argument in atom.arguments
            }
            if not goal_keys.issubset(premise_keys):
                continue
            template = _canonicalize_atoms(
                selected,
                goal,
                max_variables=budget.max_variables,
                max_permutations=budget.max_variable_permutations,
            )
            if template is None:
                continue
            signature = _template_signature(*template)
            templates[signature] = template
    return tuple(templates[key] for key in sorted(templates))


def _canonicalize_atoms(
    preconditions: Sequence[Atom],
    effect: Atom,
    *,
    max_variables: int,
    max_permutations: int,
) -> tuple[
    tuple[RuleParameterTemplate, ...],
    tuple[RuleAtomTemplate, ...],
    RuleAtomTemplate,
] | None:
    term_variables: dict[tuple[Any, ...], str] = {}
    variable_types: dict[str, str] = {}

    def raw_atom(atom: Atom) -> tuple[str, tuple[str, ...]]:
        names: list[str] = []
        for term in atom.arguments:
            key = (term.type.name, term.canonical_key())
            name = term_variables.get(key)
            if name is None:
                name = f"raw_{len(term_variables)}"
                term_variables[key] = name
                variable_types[name] = term.type.name
            names.append(name)
        return atom.predicate.name, tuple(names)

    raw_preconditions = tuple(raw_atom(atom) for atom in preconditions)
    raw_effect = raw_atom(effect)
    if len(variable_types) > max_variables:
        return None
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, type_name in variable_types.items():
        grouped[type_name].append(name)
    permutation_count = 1
    for names in grouped.values():
        permutation_count *= factorial(len(names))
    if permutation_count > max_permutations:
        return None

    canonical_names: dict[str, tuple[str, ...]] = {}
    cursor = 0
    for type_name in sorted(grouped):
        count = len(grouped[type_name])
        canonical_names[type_name] = tuple(
            f"v{index}" for index in range(cursor, cursor + count)
        )
        cursor += count
    choices = []
    for type_name in sorted(grouped):
        raw_names = tuple(sorted(grouped[type_name]))
        target_names = canonical_names[type_name]
        choices.append(
            tuple(
                dict(zip(ordering, target_names, strict=True))
                for ordering in permutations(raw_names)
            )
        )

    best: tuple[
        tuple[tuple[str, tuple[str, ...]], ...],
        tuple[str, tuple[str, ...]],
        dict[str, str],
    ] | None = None
    for parts in product(*choices):
        mapping = {key: value for part in parts for key, value in part.items()}
        normalized_preconditions = tuple(
            sorted(
                (
                    predicate,
                    tuple(mapping[name] for name in arguments),
                )
                for predicate, arguments in raw_preconditions
            )
        )
        normalized_effect = (
            raw_effect[0],
            tuple(mapping[name] for name in raw_effect[1]),
        )
        key = (normalized_preconditions, normalized_effect)
        if best is None or key < best[:2]:
            best = (normalized_preconditions, normalized_effect, mapping)
    if best is None:
        return None
    normalized_preconditions, normalized_effect, mapping = best
    parameter_types = {
        mapping[raw_name]: type_name
        for raw_name, type_name in variable_types.items()
    }
    parameters = tuple(
        RuleParameterTemplate(name, parameter_types[name])
        for name in sorted(parameter_types, key=_variable_index)
    )
    return (
        parameters,
        tuple(
            RuleAtomTemplate(predicate, arguments)
            for predicate, arguments in normalized_preconditions
        ),
        RuleAtomTemplate(normalized_effect[0], normalized_effect[1]),
    )


def _evaluate_hypothesis(
    hypothesis: TypedRuleHypothesis,
    tasks: Sequence[LearningTask],
    baselines: Mapping[str, Any],
    solve_budget: SolveBudget,
) -> RuleEvaluationSummary:
    compatible: list[str] = []
    newly_solved: list[str] = []
    preserved: list[str] = []
    checked_negative: list[str] = []
    false_positive: list[str] = []
    regressions: list[str] = []
    replay_failures: list[str] = []
    unreviewed: list[str] = []
    unsolved_positive: list[str] = []
    review_evidence: dict[str, RuleReviewEvidence] = {}
    for task in tasks:
        registry = _clone_registry(task.instance.registry)
        try:
            operator = hypothesis.instantiate(registry)
            registry.register_operator(operator)
        except Exception:
            continue
        compatible.append(task.task_id)
        augmented = _solve_task(task, registry, solve_budget)
        baseline = baselines[task.task_id]
        is_human_gold = (
            task.label_authority is SemanticLabelAuthority.HUMAN_REVIEWED
            and task.label_evidence.complete
        )
        if augmented.success and not augmented.verified:
            replay_failures.append(task.task_id)
        if task.expected_solved:
            if baseline.success:
                if augmented.success and augmented.verified:
                    preserved.append(task.task_id)
                else:
                    regressions.append(task.task_id)
                continue
            used_candidate = any(
                step.action.operator.name == hypothesis.hypothesis_id
                for step in augmented.proof
            )
            if augmented.success and augmented.verified and used_candidate:
                if is_human_gold:
                    newly_solved.append(task.task_id)
                    review_evidence[task.task_id] = _review_reference(task)
                else:
                    unreviewed.append(task.task_id)
            else:
                unsolved_positive.append(task.task_id)
            continue
        if baseline.success:
            regressions.append(task.task_id)
            continue
        if augmented.success:
            false_positive.append(task.task_id)
        elif is_human_gold:
            checked_negative.append(task.task_id)
            review_evidence[task.task_id] = _review_reference(task)
        else:
            unreviewed.append(task.task_id)
    return RuleEvaluationSummary(
        compatible_task_ids=tuple(sorted(compatible)),
        newly_solved_positive_task_ids=tuple(sorted(newly_solved)),
        preserved_positive_task_ids=tuple(sorted(preserved)),
        checked_negative_task_ids=tuple(sorted(checked_negative)),
        false_positive_task_ids=tuple(sorted(false_positive)),
        regression_task_ids=tuple(sorted(regressions)),
        replay_failure_task_ids=tuple(sorted(replay_failures)),
        unreviewed_task_ids=tuple(sorted(unreviewed)),
        unsolved_positive_task_ids=tuple(sorted(unsolved_positive)),
        review_evidence=tuple(
            review_evidence[task_id] for task_id in sorted(review_evidence)
        ),
    )


def _evaluation_rejections(
    stage: str,
    summary: RuleEvaluationSummary,
    *,
    minimum_positive: int,
    minimum_negative: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if len(summary.newly_solved_positive_task_ids) < minimum_positive:
        reasons.append(
            f"{stage}_human_positive_support_below_gate: "
            f"{len(summary.newly_solved_positive_task_ids)} < {minimum_positive}"
        )
    if len(summary.checked_negative_task_ids) < minimum_negative:
        reasons.append(
            f"{stage}_human_negative_support_below_gate: "
            f"{len(summary.checked_negative_task_ids)} < {minimum_negative}"
        )
    if summary.false_positive_task_ids:
        reasons.append(
            f"{stage}_false_positive:"
            + ",".join(summary.false_positive_task_ids)
        )
    if summary.regression_task_ids:
        reasons.append(
            f"{stage}_regression:" + ",".join(summary.regression_task_ids)
        )
    if summary.replay_failure_task_ids:
        reasons.append(
            f"{stage}_replay_failure:"
            + ",".join(summary.replay_failure_task_ids)
        )
    return tuple(reasons)


def _existing_operator_signatures(registry: KernelRegistry) -> frozenset[str]:
    signatures: set[str] = set()
    for operator in registry.operators.values():
        if len(operator.effects) != 1 or operator.guards:
            continue
        if any(
            not isinstance(argument, Variable)
            for atom in operator.preconditions + operator.effects
            for argument in atom.arguments
        ):
            continue
        canonical = _canonicalize_atoms(
            operator.preconditions,
            operator.effects[0],
            max_variables=64,
            max_permutations=100_000,
        )
        if canonical is not None:
            signatures.add(_template_signature(*canonical))
    return frozenset(signatures)


def _template_signature(
    parameters: Sequence[RuleParameterTemplate],
    preconditions: Sequence[RuleAtomTemplate],
    effect: RuleAtomTemplate,
) -> str:
    return json.dumps(
        {
            "parameters": [asdict(item) for item in parameters],
            "preconditions": [asdict(item) for item in preconditions],
            "effect": asdict(effect),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _instantiate_hypothesis(
    hypothesis: TypedRuleHypothesis,
    registry: KernelRegistry,
    *,
    retained: bool,
    metadata: Mapping[str, str] | None = None,
) -> OperatorSpec:
    variables = {
        item.name: registry.variable(item.name, item.type_name)
        for item in hypothesis.parameters
    }
    preconditions = tuple(
        _instantiate_atom(registry, atom, variables)
        for atom in hypothesis.preconditions
    )
    effect = _instantiate_atom(registry, hypothesis.effect, variables)
    resolved_metadata = {
        "discovery_status": "retained" if retained else "candidate",
        "hypothesis_id": hypothesis.hypothesis_id,
        "training_support": str(len(hypothesis.training_task_ids)),
        "training_domains": ",".join(hypothesis.training_domains),
        **(metadata or {}),
    }
    tags = (
        ("discovered", "heldout_verified", "typed_horn")
        if retained
        else ("discovered_candidate", "typed_horn")
    )
    return OperatorSpec(
        Rule(
            name=hypothesis.hypothesis_id,
            parameters=tuple(
                variables[item.name] for item in hypothesis.parameters
            ),
            preconditions=preconditions,
            effects=(effect,),
            description_ko=hypothesis.description_ko,
        ),
        family=OperatorFamily.REASONING.value,
        tags=tags,
        metadata=tuple(sorted(resolved_metadata.items())),
    )


def _instantiate_atom(
    registry: KernelRegistry,
    template: RuleAtomTemplate,
    variables: Mapping[str, Variable],
) -> Atom:
    predicate = registry.predicates.get(template.predicate)
    if predicate is None:
        raise ValueError(f"missing predicate: {template.predicate}")
    if not predicate.verified:
        raise ValueError(f"unverified predicate: {template.predicate}")
    return registry.atom(
        predicate,
        *(variables[name] for name in template.arguments),
    )


def _review_reference(task: LearningTask) -> RuleReviewEvidence:
    evidence = task.label_evidence
    if (
        task.label_authority is not SemanticLabelAuthority.HUMAN_REVIEWED
        or not evidence.complete
    ):
        raise ValueError("rule evaluation task lacks human review evidence")
    return RuleReviewEvidence(
        task_id=task.task_id,
        case_digest=evidence.case_digest,
        grounded_task_digest=rule_discovery_task_digest(task),
        domain=task.domain,
        expected_solved=task.expected_solved,
        reviewer=evidence.reviewer,
        reviewed_at=evidence.reviewed_at,
        attestation=evidence.attestation,
    )


def _summary_from_dict(value: Mapping[str, Any]) -> RuleEvaluationSummary:
    if not isinstance(value, Mapping):
        raise TypeError("rule evaluation summary must be an object")
    fields = {
        "compatible_task_ids": tuple(value.get("compatible_task_ids", ())),
        "newly_solved_positive_task_ids": tuple(
            value.get("newly_solved_positive_task_ids", ())
        ),
        "preserved_positive_task_ids": tuple(
            value.get("preserved_positive_task_ids", ())
        ),
        "checked_negative_task_ids": tuple(
            value.get("checked_negative_task_ids", ())
        ),
        "false_positive_task_ids": tuple(
            value.get("false_positive_task_ids", ())
        ),
        "regression_task_ids": tuple(value.get("regression_task_ids", ())),
        "replay_failure_task_ids": tuple(
            value.get("replay_failure_task_ids", ())
        ),
        "unreviewed_task_ids": tuple(value.get("unreviewed_task_ids", ())),
        "unsolved_positive_task_ids": tuple(
            value.get("unsolved_positive_task_ids", ())
        ),
        "review_evidence": tuple(
            RuleReviewEvidence(**item)
            for item in value.get("review_evidence", ())
        ),
    }
    return RuleEvaluationSummary(**fields)


def _clone_registry(source: KernelRegistry) -> KernelRegistry:
    clone = KernelRegistry(source.types)
    clone.functions.update(source.functions)
    clone.predicates.update(source.predicates)
    clone.operators.update(source.operators)
    clone.guards.update(source.guards)
    return clone


def _solve_task(
    task: LearningTask,
    registry: KernelRegistry,
    budget: SolveBudget,
):
    return OperatorKernel(registry).solve(
        task.instance.state,
        task.instance.goals,
        budget=budget,
    )


def _baseline_results(
    tasks: Sequence[LearningTask],
    budget: SolveBudget,
) -> dict[str, Any]:
    return {
        task.task_id: _solve_task(task, task.instance.registry, budget)
        for task in tasks
    }


def _validate_rule_discovery_splits(
    training: tuple[LearningTask, ...],
    validation: tuple[LearningTask, ...],
    heldout: tuple[LearningTask, ...],
) -> None:
    if not training or not validation or not heldout:
        raise ValueError("rule discovery requires training, validation, and heldout")
    if any(task.split is not LearningSplit.TRAIN for task in training):
        raise ValueError("rule discovery training tasks must use the train split")
    if any(
        task.split is not LearningSplit.HELDOUT
        for task in validation + heldout
    ):
        raise ValueError("rule discovery validation and heldout tasks must be heldout")
    identifiers = tuple(task.task_id for task in training + validation + heldout)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("rule discovery task ids must be unique across splits")
    fingerprints = (
        {
            task.task_id: rule_discovery_semantic_fingerprint(task)
            for task in training
        },
        {
            task.task_id: rule_discovery_semantic_fingerprint(task)
            for task in validation
        },
        {
            task.task_id: rule_discovery_semantic_fingerprint(task)
            for task in heldout
        },
    )
    names = ("training", "validation", "heldout")
    for name, split_fingerprints in zip(names, fingerprints, strict=True):
        values = tuple(split_fingerprints.values())
        if len(values) != len(set(values)):
            raise ValueError(
                f"rule discovery {name} split contains duplicate semantic tasks"
            )
    for left, right in combinations(range(3), 2):
        overlap = set(fingerprints[left].values()).intersection(
            fingerprints[right].values()
        )
        if overlap:
            raise ValueError(
                f"rule discovery {names[left]}/{names[right]} semantic overlap"
            )


def rule_discovery_semantic_fingerprint(task: LearningTask) -> str:
    """Hash typed facts and goals while deliberately excluding the label."""

    payload = {
        "facts": [
            fact.atom.canonical_key() for fact in task.instance.state.eligible_facts
        ],
        "goals": [goal.atom.canonical_key() for goal in task.instance.goals],
    }
    return sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def rule_discovery_task_digest(task: LearningTask) -> str:
    """Bind one reviewed label to the exact typed evaluation task."""

    registry = task.instance.registry
    payload = {
        "domain": task.domain,
        "expected_solved": task.expected_solved,
        "types": registry.types.snapshot(),
        "functions": [
            (
                function.name,
                tuple(item.name for item in function.input_types),
                function.output_type.name,
                function.symmetry_groups,
                function.allow_repeated_arguments,
            )
            for function in sorted(
                registry.functions.values(), key=lambda item: item.name
            )
        ],
        "predicates": [
            (
                predicate.name,
                tuple(item.name for item in predicate.argument_types),
                predicate.symmetry_groups,
                predicate.verified,
                predicate.metadata,
            )
            for predicate in sorted(
                registry.predicates.values(), key=lambda item: item.name
            )
        ],
        "operators": [
            (
                operator.name,
                tuple(item.canonical_key() for item in operator.parameters),
                tuple(item.canonical_key() for item in operator.preconditions),
                tuple(item.canonical_key() for item in operator.effects),
                operator.guards,
                operator.cost,
                operator.family,
                operator.tags,
                operator.metadata,
            )
            for operator in sorted(
                registry.operators.values(), key=lambda item: item.name
            )
        ],
        "facts": [
            (
                fact.atom.canonical_key(),
                fact.status.value,
                fact.source,
                fact.confidence,
                fact.assertion_status.value,
                fact.evidence_status.value,
            )
            for fact in task.instance.state.facts
        ],
        "goals": [
            (goal.atom.canonical_key(), goal.label)
            for goal in task.instance.goals
        ],
    }
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _variable_index(name: str) -> int:
    return int(name[1:])

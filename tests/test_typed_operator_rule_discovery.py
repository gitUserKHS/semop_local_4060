from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (  # noqa: E402
    AssertionStatus,
    DomainInstance,
    EvidenceStatus,
    Fact,
    FactStatus,
    Goal,
    HUMAN_REVIEW_ATTESTATION,
    KernelRegistry,
    LearningSplit,
    LearningTask,
    OperatorKernel,
    RuleAtomTemplate,
    RuleDiscoveryBudget,
    RuleParameterTemplate,
    SemanticLabelAuthority,
    SemanticLabelEvidence,
    TypedRuleHypothesis,
    VerifiedRuleDiscovery,
    VerifiedRuleLibrary,
    WorldState,
    rule_discovery_task_digest,
)


DOMAINS = ("language", "math", "vision")


class VerifiedTypedRuleDiscoveryTests(unittest.TestCase):
    def test_discovers_three_rules_and_transfers_to_unseen_symbols(self) -> None:
        training, validation, heldout = _rule_splits()
        original_operator_counts = tuple(
            len(task.instance.registry.operators) for task in training
        )

        result = VerifiedRuleDiscovery().discover(training, validation, heldout)

        self.assertEqual(len(result.records), 4)
        self.assertEqual(len(result.accepted_records), 3)
        self.assertEqual(len(result.library.records), 3)
        report = result.to_dict()
        json.dumps(report)
        self.assertEqual(report["accepted_rules"], 3)
        self.assertEqual(
            report["library"]["artifact_sha256"],
            result.library.artifact_sha256,
        )
        self.assertEqual(
            {record.hypothesis.training_domains for record in result.records},
            {("language",), ("math",), ("vision",)},
        )
        self.assertTrue(
            all(
                len(record.hypothesis.training_task_ids) == 3
                for record in result.accepted_records
            )
        )
        self.assertTrue(
            all(
                len(record.validation.newly_solved_positive_task_ids) == 1
                and len(record.validation.checked_negative_task_ids) == 1
                and len(record.heldout.newly_solved_positive_task_ids) == 1
                and len(record.heldout.checked_negative_task_ids) == 1
                for record in result.accepted_records
            )
        )
        rejected_language = tuple(
            record
            for record in result.rejected_records
            if record.hypothesis.training_domains == ("language",)
        )
        self.assertEqual(len(rejected_language), 1)
        self.assertIn(
            "validation_false_positive:validation:language:negative",
            rejected_language[0].rejection_reasons,
        )
        self.assertEqual(
            tuple(len(task.instance.registry.operators) for task in training),
            original_operator_counts,
        )
        first_record = result.accepted_records[0]
        matching_training = next(
            task
            for task in training
            if task.domain == first_record.hypothesis.training_domains[0]
        )
        candidate = first_record.hypothesis.instantiate(
            matching_training.instance.registry
        )
        self.assertIn("discovered_candidate", candidate.tags)
        self.assertNotIn("heldout_verified", candidate.tags)
        self.assertEqual(dict(candidate.metadata)["discovery_status"], "candidate")
        tasks_by_id = {
            task.task_id: task for task in validation + heldout
        }
        for record in result.accepted_records:
            for evidence in (
                record.validation.review_evidence
                + record.heldout.review_evidence
            ):
                self.assertEqual(
                    evidence.grounded_task_digest,
                    rule_discovery_task_digest(tasks_by_id[evidence.task_id]),
                )
                self.assertEqual(
                    evidence.expected_solved,
                    tasks_by_id[evidence.task_id].expected_solved,
                )

        for task in heldout:
            augmented = result.library.augment_task(task)
            solved = OperatorKernel(augmented.instance.registry).solve(
                augmented.instance.state,
                augmented.instance.goals,
            )
            if task.expected_solved:
                self.assertTrue(solved.success and solved.verified, task.task_id)
                self.assertEqual(len(solved.proof), 1)
                self.assertIn("discovered", solved.proof[0].action.operator.tags)
                self.assertIn(
                    "heldout_verified", solved.proof[0].action.operator.tags
                )
                operator_metadata = dict(solved.proof[0].action.operator.metadata)
                self.assertEqual(operator_metadata["discovery_status"], "retained")
                self.assertEqual(
                    operator_metadata["verified_rule_library_sha256"],
                    result.library.artifact_sha256,
                )
                self.assertEqual(
                    len(augmented.instance.metadata["activated_discovered_rules"]),
                    1,
                )
            else:
                self.assertFalse(solved.success, task.task_id)

    def test_blocked_counterexample_rejects_overgeneral_language_rule(self) -> None:
        training, validation, heldout = _rule_splits(
            blocked_language_validation=True
        )

        result = VerifiedRuleDiscovery().discover(training, validation, heldout)
        language_records = tuple(
            record
            for record in result.records
            if record.hypothesis.training_domains == ("language",)
        )
        by_domain = {
            record.hypothesis.training_domains[0]: record
            for record in result.accepted_records
        }

        self.assertTrue(language_records)
        self.assertTrue(all(not record.accepted for record in language_records))
        self.assertTrue(
            any(
                reason.startswith("validation_false_positive:")
                for record in language_records
                for reason in record.rejection_reasons
            )
        )
        self.assertEqual(
            {
                task_id
                for record in language_records
                for task_id in record.validation.false_positive_task_ids
            },
            {"validation:language:negative"},
        )
        self.assertTrue(by_domain["math"].accepted)
        self.assertTrue(by_domain["vision"].accepted)
        self.assertEqual(len(result.library.records), 2)

    def test_unreviewed_validation_cannot_retain_a_rule(self) -> None:
        training, validation, heldout = _rule_splits()
        unreviewed = tuple(_remove_review(task) for task in validation)

        result = VerifiedRuleDiscovery().discover(training, unreviewed, heldout)

        self.assertEqual(result.library.records, ())
        self.assertTrue(result.records)
        self.assertTrue(
            all(
                any("validation_human_" in reason for reason in record.rejection_reasons)
                for record in result.records
            )
        )

    def test_same_semantic_instance_cannot_cross_split_with_flipped_label(self) -> None:
        training, validation, heldout = _rule_splits()
        duplicate = replace(
            training[0],
            task_id="validation:duplicate-with-flipped-label",
            split=LearningSplit.HELDOUT,
            expected_solved=False,
        )

        with self.assertRaisesRegex(ValueError, "semantic overlap"):
            VerifiedRuleDiscovery().discover(
                training,
                (duplicate,) + validation,
                heldout,
            )

    def test_exact_duplicates_within_a_split_do_not_inflate_support(self) -> None:
        training, validation, heldout = _rule_splits()
        duplicate = replace(training[0], task_id="train:duplicate")

        with self.assertRaisesRegex(ValueError, "duplicate semantic tasks"):
            VerifiedRuleDiscovery().discover(
                training + (duplicate,),
                validation,
                heldout,
            )

    def test_effect_variables_must_be_bound_by_preconditions(self) -> None:
        with self.assertRaisesRegex(ValueError, "premise-bound"):
            TypedRuleHypothesis(
                hypothesis_id="invalid",
                parameters=(
                    RuleParameterTemplate("v0", "Entity"),
                    RuleParameterTemplate("v1", "Entity"),
                ),
                preconditions=(RuleAtomTemplate("KNOWN", ("v0",)),),
                effect=RuleAtomTemplate("READY", ("v1",)),
                training_task_ids=("train:one",),
                training_domains=("language",),
            )

    def test_support_and_candidate_budgets_are_bounded(self) -> None:
        training, validation, heldout = _rule_splits()

        insufficient = VerifiedRuleDiscovery(
            RuleDiscoveryBudget(min_training_support=4)
        ).discover(training, validation, heldout)
        capped = VerifiedRuleDiscovery(
            RuleDiscoveryBudget(max_candidates=2)
        ).discover(training, validation, heldout)

        self.assertEqual(insufficient.records, ())
        self.assertEqual(insufficient.library.records, ())
        self.assertEqual(len(capped.records), 2)
        self.assertLessEqual(len(capped.library.records), 2)

    def test_retained_library_artifact_is_deterministic_and_fail_closed(self) -> None:
        training, validation, heldout = _rule_splits()
        first = VerifiedRuleDiscovery().discover(training, validation, heldout)
        second = VerifiedRuleDiscovery().discover(training, validation, heldout)

        self.assertEqual(first.library.artifact, second.library.artifact)
        restored = VerifiedRuleLibrary.from_artifact(first.library.artifact)
        self.assertEqual(restored, first.library)
        self.assertEqual(restored.artifact_sha256, first.library.artifact_sha256)

        payload = json.loads(first.library.artifact.decode("utf-8"))
        payload["records"][0]["accepted"] = False
        tampered = json.dumps(payload).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "rejected rule record"):
            VerifiedRuleLibrary.from_artifact(tampered)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            VerifiedRuleLibrary.from_artifact(
                tampered,
                expected_sha256=first.library.artifact_sha256,
            )

        missing_evidence = json.loads(first.library.artifact.decode("utf-8"))
        missing_evidence["records"][0]["validation"]["review_evidence"] = []
        with self.assertRaisesRegex(ValueError, "lacks validation review evidence"):
            VerifiedRuleLibrary.from_artifact(
                json.dumps(missing_evidence).encode("utf-8")
            )

        flipped_label = json.loads(first.library.artifact.decode("utf-8"))
        evidence = flipped_label["records"][0]["validation"][
            "review_evidence"
        ]
        positive_id = flipped_label["records"][0]["validation"][
            "newly_solved_positive_task_ids"
        ][0]
        next(item for item in evidence if item["task_id"] == positive_id)[
            "expected_solved"
        ] = False
        with self.assertRaisesRegex(ValueError, "inconsistent validation labels"):
            VerifiedRuleLibrary.from_artifact(
                json.dumps(flipped_label).encode("utf-8")
            )


def _rule_splits(
    *,
    blocked_language_validation: bool = False,
) -> tuple[
    tuple[LearningTask, ...],
    tuple[LearningTask, ...],
    tuple[LearningTask, ...],
]:
    training = tuple(
        _task(domain, f"train-{index}", LearningSplit.TRAIN, True)
        for domain in DOMAINS
        for index in range(3)
    )
    validation = tuple(
        item
        for domain in DOMAINS
        for item in (
            _task(domain, "validation-positive", LearningSplit.HELDOUT, True),
            _task(
                domain,
                "validation-negative",
                LearningSplit.HELDOUT,
                False,
                blocked=(blocked_language_validation and domain == "language"),
            ),
        )
    )
    heldout = tuple(
        item
        for domain in DOMAINS
        for item in (
            _task(domain, "heldout-positive", LearningSplit.HELDOUT, True),
            _task(domain, "heldout-negative", LearningSplit.HELDOUT, False),
        )
    )
    return training, validation, heldout


def _task(
    domain: str,
    token: str,
    split: LearningSplit,
    expected_solved: bool,
    *,
    blocked: bool = False,
) -> LearningTask:
    registry = _registry(domain)
    facts: tuple[Fact, ...]
    if domain == "language":
        goal_symbol = registry.symbol(f"goal-{token}", "ReasoningGoal")
        premise = registry.symbol(f"premise-{token}", "Premise")
        other = registry.symbol(f"other-{token}", "Premise")
        facts_list = [
            _fact(registry.atom("REQUIRES", goal_symbol, premise)),
            _fact(
                registry.atom(
                    "SATISFIED",
                    premise if expected_solved or blocked else other,
                )
            ),
        ]
        if blocked:
            facts_list.append(_fact(registry.atom("BLOCKED", premise)))
        facts = tuple(facts_list)
        goal = Goal(registry.atom("READY", goal_symbol))
    elif domain == "math":
        left = registry.symbol(f"left-{token}", "Number")
        right = registry.symbol(f"right-{token}", "Number")
        actual = registry.symbol(f"actual-{token}", "Number")
        requested = (
            actual
            if expected_solved
            else registry.symbol(f"wrong-{token}", "Number")
        )
        facts = (_fact(registry.atom("ADD_RESULT", left, right, actual)),)
        goal = Goal(registry.atom("EXACT_SUM", left, right, requested))
    else:
        left = registry.symbol(f"left-{token}", "Object")
        middle = registry.symbol(f"middle-{token}", "Object")
        right = registry.symbol(f"right-{token}", "Object")
        disconnected = registry.symbol(f"disconnected-{token}", "Object")
        facts = (
            _fact(registry.atom("LEFT_OF", left, middle)),
            _fact(
                registry.atom(
                    "LEFT_OF",
                    middle if expected_solved else disconnected,
                    right,
                )
            ),
        )
        goal = Goal(registry.atom("LEFT_OF", left, right))

    task_id = f"{split.value}:{domain}:{'positive' if expected_solved else 'negative'}"
    if token.startswith("train-"):
        task_id = f"train:{domain}:{token}"
    elif token.startswith("validation-"):
        task_id = f"validation:{domain}:{token.removeprefix('validation-')}"
    elif token.startswith("heldout-"):
        task_id = f"heldout:{domain}:{token.removeprefix('heldout-')}"
    reviewed = split is LearningSplit.HELDOUT
    return LearningTask(
        task_id=task_id,
        instance=DomainInstance(
            registry=registry,
            state=WorldState(facts),
            goals=(goal,),
            domain=domain,
            metadata={"fixture": "verified-rule-discovery", "token": token},
        ),
        expected_solved=expected_solved,
        split=split,
        source="reviewed" if reviewed else "verifier",
        domain=domain,
        capability=f"discover-{domain}-rule",
        structure_key=f"rule-discovery:{domain}",
        difficulty=2,
        label_authority=(
            SemanticLabelAuthority.HUMAN_REVIEWED
            if reviewed
            else SemanticLabelAuthority.CURATED_UNREVIEWED
        ),
        label_evidence=(
            _review_evidence(task_id) if reviewed else SemanticLabelEvidence()
        ),
    )


def _registry(domain: str) -> KernelRegistry:
    registry = KernelRegistry()
    entity = registry.types.register("Entity")
    if domain == "language":
        goal = registry.types.register("ReasoningGoal", entity)
        premise = registry.types.register("Premise", entity)
        registry.register_predicate("REQUIRES", (goal, premise))
        registry.register_predicate("SATISFIED", (premise,))
        registry.register_predicate("BLOCKED", (premise,))
        registry.register_predicate("READY", (goal,))
    elif domain == "math":
        number = registry.types.register("Number", entity)
        registry.register_predicate("ADD_RESULT", (number, number, number))
        registry.register_predicate("EXACT_SUM", (number, number, number))
    elif domain == "vision":
        object_type = registry.types.register("Object", entity)
        registry.register_predicate("LEFT_OF", (object_type, object_type))
    else:
        raise ValueError(domain)
    return registry


def _fact(atom) -> Fact:
    return Fact(
        atom,
        FactStatus.OBSERVED,
        source="rule-discovery-fixture",
        assertion_status=AssertionStatus.EXPLICIT,
        evidence_status=EvidenceStatus.ADAPTER_VERIFIED,
    )


def _review_evidence(task_id: str) -> SemanticLabelEvidence:
    return SemanticLabelEvidence(
        case_digest=sha256(task_id.encode("utf-8")).hexdigest(),
        reviewer="human:test-reviewer",
        reviewed_at="2026-07-18T12:00:00Z",
        attestation=HUMAN_REVIEW_ATTESTATION,
    )


def _remove_review(task: LearningTask) -> LearningTask:
    return replace(
        task,
        source="verifier",
        label_authority=SemanticLabelAuthority.CURATED_UNREVIEWED,
        label_evidence=SemanticLabelEvidence(),
    )


if __name__ == "__main__":
    unittest.main()

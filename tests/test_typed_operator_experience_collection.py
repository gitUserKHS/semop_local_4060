from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (  # noqa: E402
    AssertionStatus,
    DomainKind,
    DomainInstance,
    EvidenceStatus,
    ExperienceCollectorConfig,
    ExperiencePartitionConfig,
    ExperienceProposalAuthority,
    ExperienceQueueStatus,
    ExperienceReviewCorpusGrounder,
    ExperienceSplitRole,
    ExperienceTrigger,
    Fact,
    FactStatus,
    Goal,
    KernelRegistry,
    LanguageTextProblem,
    RasterImage,
    RasterVisionProblem,
    RawExperienceGrounder,
    ReviewedExperienceRuleLearningLoop,
    SemanticLabelAuthority,
    TypedDomainRequest,
    TypedExperienceCollector,
    TypedExperienceStore,
    UnifiedTypedReasoner,
    VisionCountGoal,
    VerifiedRuleLibrary,
    WorldState,
    semantic_request_digest,
)


WHITE = (255, 255, 255)
RED = (255, 0, 0)


class TypedExperienceCollectorTests(unittest.TestCase):
    def test_runtime_collects_only_noteworthy_language_math_and_vision_runs(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store = TypedExperienceStore(Path(directory) / "experience.db")
            collector = TypedExperienceCollector(store)

            ordinary_success = collector.run(
                TypedDomainRequest("math", "2 + 2 == 4"),
                event_id="ordinary-success",
            )
            language_failure = collector.run(
                TypedDomainRequest(
                    "language",
                    "Goal: deploy; Requires: tests",
                ),
                event_id="language-unsolved",
            )
            math_parse_failure = collector.run(
                TypedDomainRequest("math", "2 +"),
                event_id="math-grounding-failure",
            )
            vision_proposal = collector.run(
                _vision_request(1),
                proposed_expected_solved=True,
                proposal_authority=ExperienceProposalAuthority.FRONTIER_JUDGE,
                rationale="The external judge proposed that one object is present.",
                event_id="vision-judge-proposal",
            )

            stats = store.stats()
            vision_item = store.get_item(vision_proposal.request_digest)
            reviewed = store.export_reviewed()

        self.assertTrue(ordinary_success.succeeded)
        self.assertFalse(ordinary_success.captured)
        self.assertEqual(language_failure.trigger, ExperienceTrigger.UNSOLVED)
        self.assertTrue(language_failure.captured)
        self.assertEqual(
            math_parse_failure.trigger,
            ExperienceTrigger.RUNTIME_EXCEPTION,
        )
        self.assertEqual(math_parse_failure.error_type, "ArithmeticDslError")
        self.assertTrue(vision_proposal.succeeded)
        self.assertTrue(vision_proposal.captured)
        self.assertTrue(vision_proposal.capture.observation.proof_program)
        self.assertEqual(vision_item.status, ExperienceQueueStatus.PENDING)
        self.assertEqual(reviewed.records, ())
        self.assertEqual(
            dict(stats.by_domain),
            {"language": 1, "math": 1, "vision": 1},
        )

    def test_matching_proposals_can_be_ignored_without_losing_mismatches(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store = TypedExperienceStore(Path(directory) / "experience.db")
            collector = TypedExperienceCollector(
                store,
                config=ExperienceCollectorConfig(
                    capture_matching_proposals=False,
                ),
            )
            matching = collector.run(
                TypedDomainRequest("math", "3 + 4 == 7"),
                proposed_expected_solved=True,
                proposal_authority="programmatic",
                rationale="The generated equality should be solvable.",
                event_id="matching",
            )
            mismatch = collector.run(
                TypedDomainRequest("math", "3 + 4 == 8"),
                proposed_expected_solved=True,
                proposal_authority="programmatic",
                rationale="This intentionally incorrect proposal tests disagreement.",
                event_id="mismatch",
            )
            with self.assertRaisesRegex(ValueError, "require their authority"):
                collector.run(
                    TypedDomainRequest("math", "5 + 5 == 10"),
                    proposed_expected_solved=True,
                    rationale="This invalid proposal has no declared authority.",
                    event_id="missing-authority",
                )

        self.assertFalse(matching.captured)
        self.assertTrue(mismatch.captured)
        self.assertEqual(mismatch.trigger, ExperienceTrigger.EXPECTATION_MISMATCH)

    def test_runtime_augmenters_cannot_replace_facts_or_goals(self) -> None:
        request = TypedDomainRequest("math", "2 + 3 == 5")
        with self.assertRaisesRegex(ValueError, "copied registry"):
            UnifiedTypedReasoner(
                augmenters=(_IdentityAugmenter(),)
            ).run(request)
        with self.assertRaisesRegex(ValueError, "facts, goals, and domain"):
            UnifiedTypedReasoner(
                augmenters=(_GoalChangingAugmenter(),)
            ).run(request)
        with self.assertRaisesRegex(ValueError, "only append operators"):
            UnifiedTypedReasoner(
                augmenters=(_OperatorRemovingAugmenter(),)
            ).run(request)


class ReviewedExperienceGroundingTests(unittest.TestCase):
    def test_approved_lmv_cases_export_to_a_leakage_free_four_way_corpus(
        self,
    ) -> None:
        partition = ExperiencePartitionConfig(seed="four-way-lmv-test")
        role_domains = {
            ExperienceSplitRole.TRAIN: DomainKind.LANGUAGE,
            ExperienceSplitRole.VALIDATION: DomainKind.MATH,
            ExperienceSplitRole.CANDIDATE_HELDOUT: DomainKind.VISION,
            ExperienceSplitRole.JOINT_HELDOUT: DomainKind.LANGUAGE,
        }
        with TemporaryDirectory() as directory:
            store = TypedExperienceStore(
                Path(directory) / "experience.db",
                partition=partition,
            )
            collector = TypedExperienceCollector(store)
            for index, (role, domain) in enumerate(role_domains.items()):
                request = _request_for_role(
                    partition,
                    role,
                    domain,
                    seed=f"case-{index}",
                )
                run = collector.run(
                    request,
                    proposed_expected_solved=True,
                    proposal_authority="programmatic",
                    rationale="A generated positive awaits independent review.",
                    event_id=f"event-{role.value}",
                )
                self.assertTrue(run.succeeded)
                store.review(
                    run.request_digest,
                    expected_solved=True,
                    phenomenon=f"reviewed_{domain.value}_operator_case",
                    rationale="A human checked the exact raw input and outcome.",
                    reviewer="human:collection-test",
                    decision="approved",
                    reviewed_at="2026-07-18T12:00:00Z",
                )

            corpus = store.export_reviewed()
            grounding = ExperienceReviewCorpusGrounder().ground(
                corpus,
                namespace="collection-lmv",
            )
            split_tasks = grounding.require_ready()

        self.assertEqual(
            dict(corpus.role_counts),
            {role.value: 1 for role in ExperienceSplitRole},
        )
        self.assertTrue(grounding.complete)
        self.assertTrue(grounding.leakage_free)
        self.assertEqual([len(tasks) for tasks in split_tasks], [1, 1, 1, 1])
        self.assertTrue(
            all(
                task.label_authority is SemanticLabelAuthority.HUMAN_REVIEWED
                and task.label_evidence.complete
                for tasks in split_tasks
                for task in tasks
            )
        )

    def test_grounded_semantic_duplicates_across_roles_fail_closed(self) -> None:
        partition = ExperiencePartitionConfig(seed="semantic-overlap-test")
        text = "Goal: deploy; Requires: tests; Satisfied: tests"
        train = _language_context_for_role(
            partition,
            ExperienceSplitRole.TRAIN,
            text,
        )
        validation = _language_context_for_role(
            partition,
            ExperienceSplitRole.VALIDATION,
            text,
        )
        requests = (
            train,
            validation,
            _request_for_role(
                partition,
                ExperienceSplitRole.CANDIDATE_HELDOUT,
                DomainKind.VISION,
                seed="overlap-vision",
            ),
            _request_for_role(
                partition,
                ExperienceSplitRole.JOINT_HELDOUT,
                DomainKind.MATH,
                seed="overlap-math",
            ),
        )
        with TemporaryDirectory() as directory:
            store = TypedExperienceStore(
                Path(directory) / "experience.db",
                partition=partition,
            )
            collector = TypedExperienceCollector(store)
            for index, request in enumerate(requests):
                run = collector.run(
                    request,
                    proposed_expected_solved=True,
                    proposal_authority="programmatic",
                    rationale="Generated input for split overlap auditing.",
                    event_id=f"overlap-{index}",
                )
                store.review(
                    run.request_digest,
                    expected_solved=True,
                    phenomenon="cross_role_overlap_audit",
                    rationale="A human reviewed this exact expected outcome.",
                    reviewer="human:overlap-test",
                    decision="approved",
                    reviewed_at="2026-07-18T12:00:00Z",
                )

            grounding = ExperienceReviewCorpusGrounder().ground(
                store.export_reviewed(),
                namespace="overlap-audit",
            )

        self.assertEqual(grounding.input_overlap, ())
        self.assertTrue(grounding.semantic_overlap)
        with self.assertRaisesRegex(ValueError, "semantic overlap"):
            grounding.require_ready()


class ReviewedExperienceRuleLearningTests(unittest.TestCase):
    def test_reviewed_runtime_failures_learn_and_activate_a_verified_rule(
        self,
    ) -> None:
        partition = ExperiencePartitionConfig(seed="end-to-end-rule-learning")
        reasoner = UnifiedTypedReasoner(
            {DomainKind.LANGUAGE: _ChainLanguageAdapter()}
        )
        specifications = [
            (ExperienceSplitRole.TRAIN, True, f"train-{index}")
            for index in range(3)
        ]
        specifications.extend(
            (role, expected, f"{role.value}-{'positive' if expected else 'negative'}")
            for role in (
                ExperienceSplitRole.VALIDATION,
                ExperienceSplitRole.CANDIDATE_HELDOUT,
                ExperienceSplitRole.JOINT_HELDOUT,
            )
            for expected in (True, False)
        )

        with TemporaryDirectory() as directory:
            store = TypedExperienceStore(
                Path(directory) / "experience.db",
                partition=partition,
            )
            collector = TypedExperienceCollector(store, reasoner=reasoner)
            for role, expected, token in specifications:
                request = _chain_request_for_role(
                    partition,
                    role,
                    expected_solved=expected,
                    seed=token,
                )
                run = collector.run(
                    request,
                    proposed_expected_solved=expected,
                    proposal_authority="programmatic",
                    rationale="The curriculum proposes the reviewed chain outcome.",
                    event_id=f"rule-{token}",
                )
                self.assertFalse(run.succeeded)
                store.review(
                    run.request_digest,
                    expected_solved=expected,
                    phenomenon="start_implies_middle",
                    rationale="A human verified the exact premise and target symbols.",
                    reviewer="human:rule-learning-test",
                    decision="approved",
                    reviewed_at="2026-07-18T12:00:00Z",
                )

            reviewed_corpus = store.export_reviewed()
            learning = ReviewedExperienceRuleLearningLoop(
                grounder=ExperienceReviewCorpusGrounder(
                    RawExperienceGrounder(
                        reasoner,
                        hard_negatives_per_example=0,
                    )
                )
            ).run(
                reviewed_corpus,
                namespace="reviewed-chain",
                required_domains=(DomainKind.LANGUAGE,),
            )

        new_request = _chain_request(
            source="unseen-source",
            target="unseen-source",
        )
        baseline = reasoner.run(new_request)
        learned_reasoner = UnifiedTypedReasoner(
            {DomainKind.LANGUAGE: _ChainLanguageAdapter()},
            augmenters=(learning.learning.active_library,),
        )
        transferred = learned_reasoner.run(new_request)

        self.assertTrue(learning.promoted, learning.learning.rejection_reasons)
        self.assertEqual(len(learning.learning.active_library.records), 1)
        self.assertFalse(baseline.success)
        self.assertTrue(transferred.success and transferred.verified)
        self.assertEqual(len(transferred.typed_result.proof), 1)
        self.assertIn(
            "heldout_verified",
            transferred.typed_result.proof[0].action.operator.tags,
        )
        self.assertEqual(
            len(transferred.instance.metadata["activated_discovered_rules"]),
            1,
        )


def _request_for_role(
    partition: ExperiencePartitionConfig,
    role: ExperienceSplitRole,
    domain: DomainKind,
    *,
    seed: str,
) -> TypedDomainRequest:
    for index in range(10_000):
        if domain is DomainKind.LANGUAGE:
            token = f"{seed}-{index}"
            request = TypedDomainRequest(
                domain,
                f"Goal: {token}; Requires: check-{token}; "
                f"Satisfied: check-{token}",
            )
        elif domain is DomainKind.MATH:
            value = 100 + index
            request = TypedDomainRequest(domain, f"{value} + 1 == {value + 1}")
        else:
            request = _vision_request(index + 1)
        if partition.role_for(semantic_request_digest(request)) is role:
            return request
    raise AssertionError(
        f"could not generate a {domain.value} request for {role.value}"
    )


def _language_context_for_role(
    partition: ExperiencePartitionConfig,
    role: ExperienceSplitRole,
    text: str,
) -> TypedDomainRequest:
    for index in range(10_000):
        request = TypedDomainRequest(
            DomainKind.LANGUAGE,
            LanguageTextProblem(text, f"source-context-{role.value}-{index}"),
        )
        if partition.role_for(semantic_request_digest(request)) is role:
            return request
    raise AssertionError(f"could not generate a language request for {role.value}")


def _chain_request_for_role(
    partition: ExperiencePartitionConfig,
    role: ExperienceSplitRole,
    *,
    expected_solved: bool,
    seed: str,
) -> TypedDomainRequest:
    for index in range(10_000):
        source = f"{seed}-source-{index}"
        target = source if expected_solved else f"{seed}-other-{index}"
        request = _chain_request(source=source, target=target)
        if partition.role_for(semantic_request_digest(request)) is role:
            return request
    raise AssertionError(f"could not generate a chain request for {role.value}")


def _chain_request(*, source: str, target: str) -> TypedDomainRequest:
    return TypedDomainRequest(
        DomainKind.LANGUAGE,
        f"premise=START;goal=MIDDLE;source={source};target={target}",
    )


class _ChainLanguageAdapter:
    def adapt(self, payload: str | LanguageTextProblem) -> DomainInstance:
        text = payload.text if isinstance(payload, LanguageTextProblem) else payload
        fields = dict(
            field.split("=", 1)
            for field in text.split(";")
        )
        registry = KernelRegistry()
        entity = registry.types.register("Entity")
        for predicate in ("START", "MIDDLE"):
            registry.register_predicate(predicate, (entity,))
        source = registry.symbol(fields["source"], entity)
        target = registry.symbol(fields["target"], entity)
        fact = Fact(
            registry.atom(fields["premise"], source),
            FactStatus.OBSERVED,
            source="chain-language-adapter",
            assertion_status=AssertionStatus.EXPLICIT,
            evidence_status=EvidenceStatus.ADAPTER_VERIFIED,
        )
        return DomainInstance(
            registry=registry,
            state=WorldState((fact,)),
            goals=(Goal(registry.atom(fields["goal"], target)),),
            domain="language",
            metadata={"adapter": "chain-language-test"},
        )


class _IdentityAugmenter:
    def augment_instance(self, instance: DomainInstance) -> DomainInstance:
        return instance


class _GoalChangingAugmenter:
    def augment_instance(self, instance: DomainInstance) -> DomainInstance:
        augmented = VerifiedRuleLibrary().augment_instance(instance)
        return replace(augmented, goals=())


class _OperatorRemovingAugmenter:
    def augment_instance(self, instance: DomainInstance) -> DomainInstance:
        augmented = VerifiedRuleLibrary().augment_instance(instance)
        augmented.registry.operators.clear()
        return augmented


def _vision_request(index: int) -> TypedDomainRequest:
    width = 5 + index % 3
    rows = [
        [WHITE] * width,
        [WHITE, RED, RED, *([WHITE] * (width - 3))],
        [WHITE, RED, RED, *([WHITE] * (width - 3))],
        [WHITE] * width,
    ]
    return TypedDomainRequest(
        DomainKind.VISION,
        RasterVisionProblem(
            RasterImage.from_rows(rows, source=f"experience-{index}"),
            (VisionCountGoal("red", 1),),
        ),
    )


if __name__ == "__main__":
    unittest.main()

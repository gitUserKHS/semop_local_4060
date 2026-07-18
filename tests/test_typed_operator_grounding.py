from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    AssertionStatus,
    DomainInstance,
    EvidenceStatus,
    FactStatus,
    GroundingAuthority,
    GroundingBoundaryError,
    GroundingDecision,
    GroundingDisposition,
    GroundingLabel,
    GroundingTrace,
    KernelRegistry,
    LanguageTextAdapter,
    RasterImage,
    RasterVisionAdapter,
    RasterVisionProblem,
    VisionProblem,
    VisionRelationGoal,
    VisionWorldAdapter,
    WorldState,
    compose_domain_instances,
    decide_grounding,
    make_grounding_candidate,
    parse_arithmetic_expression,
    promote_grounding_proposal,
    review_grounding_proposal,
    stage_grounding_proposal,
)


WHITE = (255, 255, 255)
RED = (255, 0, 0)


def _candidate():
    registry = KernelRegistry()
    entity = registry.types.register("Entity")
    registry.register_predicate("READY", (entity,))
    item = registry.symbol("item", entity)
    candidate = make_grounding_candidate(
        domain="language",
        statement="item is ready",
        atom=registry.atom("READY", item),
        producer_id="tiny-grounder-v1",
        source="model_grounder",
        assertion_status=AssertionStatus.INFERRED,
        confidence=0.99,
    )
    return registry, candidate


class GroundingBoundaryTests(unittest.TestCase):
    def test_model_confidence_cannot_create_an_observed_fact(self) -> None:
        _, candidate = _candidate()

        with self.assertRaises(GroundingBoundaryError):
            decide_grounding(
                candidate,
                GroundingDecision(
                    GroundingDisposition.OBSERVED,
                    GroundingAuthority.MODEL_PROPOSAL,
                    "tiny-grounder-v1",
                    "the model was confident",
                    EvidenceStatus.UNVERIFIED,
                    0.99,
                ),
            )
        model_reject = decide_grounding(
            candidate,
            GroundingDecision(
                GroundingDisposition.REJECTED,
                GroundingAuthority.MODEL_PROPOSAL,
                "tiny-grounder-v1",
                "the model rejected its own candidate",
                EvidenceStatus.UNVERIFIED,
                0.99,
            ),
        )
        self.assertFalse(model_reject.hard_negative)
        self.assertEqual(GroundingTrace((model_reject,)).learning_examples, ())

    def test_rejected_verifier_result_becomes_a_hard_negative(self) -> None:
        _, candidate = _candidate()
        proposed = stage_grounding_proposal(
            candidate,
            authority=GroundingAuthority.MODEL_PROPOSAL,
            verifier_id="tiny-grounder-v1",
            rationale="small model proposed a typed concept",
            confidence=0.99,
        )
        rejected = promote_grounding_proposal(
            proposed,
            lambda _atom: False,
            verifier_id="exact-test-verifier",
        )
        trace = GroundingTrace((rejected,))

        self.assertIs(proposed.fact.status, FactStatus.PROPOSED)
        self.assertFalse(proposed.fact.proof_eligible)
        self.assertIsNone(rejected.fact)
        self.assertTrue(rejected.hard_negative)
        self.assertEqual(trace.hard_negatives, (rejected,))
        self.assertEqual(len(trace.learning_examples), 1)
        self.assertIs(trace.learning_examples[0].label, GroundingLabel.REJECT)
        self.assertEqual(
            rejected.decision.supersedes_record_digest,
            proposed.record_digest,
        )

    def test_human_review_explicitly_promotes_or_rejects_a_proposal(self) -> None:
        _, candidate = _candidate()
        proposed = stage_grounding_proposal(
            candidate,
            authority=GroundingAuthority.MODEL_PROPOSAL,
            verifier_id="tiny-grounder-v1",
            rationale="small model proposed a typed concept",
            confidence=0.8,
        )
        approved = review_grounding_proposal(
            proposed,
            approved=True,
            reviewer_id="human:dayoon",
            rationale="reviewer checked the source assertion",
            evidence_refs=("review:1",),
        )
        rejected = review_grounding_proposal(
            proposed,
            approved=False,
            reviewer_id="human:dayoon",
            rationale="reviewer found a relation-direction error",
        )
        revised = GroundingTrace((proposed,)).with_record(
            approved,
            replace_existing=True,
        )

        self.assertIs(approved.fact.status, FactStatus.OBSERVED)
        self.assertIs(
            approved.fact.evidence_status,
            EvidenceStatus.EXTERNAL_VERIFIED,
        )
        self.assertTrue(approved.fact.proof_eligible)
        self.assertTrue(rejected.hard_negative)
        self.assertIs(
            GroundingTrace((approved,)).learning_examples[0].label,
            GroundingLabel.ACCEPT,
        )
        self.assertEqual(revised.records, (approved,))
        with self.assertRaises(GroundingBoundaryError):
            review_grounding_proposal(
                proposed,
                approved=True,
                reviewer_id="automated-reviewer",
                rationale="not an attested human review",
            )
        with self.assertRaises(GroundingBoundaryError):
            decide_grounding(
                candidate,
                GroundingDecision(
                    GroundingDisposition.OBSERVED,
                    GroundingAuthority.HUMAN_REVIEW,
                    "automated-reviewer",
                    "forged human authority",
                    EvidenceStatus.EXTERNAL_VERIFIED,
                ),
            )

    def test_domain_instance_rejects_a_trace_fact_missing_from_state(self) -> None:
        registry, candidate = _candidate()
        proposed = stage_grounding_proposal(
            candidate,
            authority=GroundingAuthority.MODEL_PROPOSAL,
            verifier_id="tiny-grounder-v1",
            rationale="small model proposed a typed concept",
            confidence=0.8,
        )

        with self.assertRaises(GroundingBoundaryError):
            DomainInstance(
                registry,
                WorldState(),
                (),
                "language",
                grounding_trace=GroundingTrace((proposed,)),
            )


class CrossDomainGroundingTraceTests(unittest.TestCase):
    def test_language_math_and_vision_share_one_grounding_contract(self) -> None:
        language = LanguageTextAdapter().adapt(
            "Goal: deploy; Requires: tests; Satisfied: tests"
        )
        math = parse_arithmetic_expression("2 * (3 + 4)")
        image = RasterImage.from_rows(
            (
                (WHITE, WHITE, WHITE, WHITE),
                (WHITE, RED, RED, WHITE),
                (WHITE, RED, RED, WHITE),
                (WHITE, WHITE, WHITE, WHITE),
            )
        )
        vision = RasterVisionAdapter().adapt(RasterVisionProblem(image))

        self.assertTrue(language.grounding_trace.records)
        self.assertTrue(math.grounding_trace.records)
        self.assertTrue(vision.grounding_trace.records)
        self.assertTrue(
            language.grounding_trace.audit(language.state.facts).coverage_complete
        )
        self.assertTrue(math.grounding_trace.audit(math.state.facts).coverage_complete)
        self.assertTrue(
            vision.grounding_trace.audit(vision.state.facts).coverage_complete
        )
        self.assertEqual(
            {
                record.decision.authority
                for record in language.grounding_trace.records
            },
            {GroundingAuthority.EXPLICIT_INPUT},
        )
        self.assertEqual(language.grounding_trace.learning_examples, ())
        self.assertEqual(
            {
                record.decision.authority for record in math.grounding_trace.records
            },
            {GroundingAuthority.DETERMINISTIC_ADAPTER},
        )
        self.assertEqual(
            {
                record.decision.authority
                for record in vision.grounding_trace.records
            },
            {GroundingAuthority.DETERMINISTIC_ADAPTER},
        )

    def test_language_heuristics_and_unverified_vision_stay_proposed(self) -> None:
        language = LanguageTextAdapter().adapt(
            "책을 가방에 넣으려면 어떻게 해야 해?"
        )
        world = SimpleNamespace(
            query="Is A left of B?",
            entities=[
                SimpleNamespace(id="A", attributes={}),
                SimpleNamespace(id="B", attributes={}),
            ],
            relations=[
                SimpleNamespace(
                    source="A",
                    relation="LEFT_OF",
                    target="B",
                    confidence=0.999,
                    attributes={},
                )
            ],
        )
        vision = VisionWorldAdapter().adapt(
            VisionProblem(
                world,
                (VisionRelationGoal("LEFT_OF", "A", "B"),),
            )
        )

        self.assertTrue(
            all(
                record.decision.authority
                is GroundingAuthority.HEURISTIC_PROPOSAL
                for record in language.grounding_trace.records
            )
        )
        self.assertTrue(
            all(
                record.fact is not None
                and record.fact.status is FactStatus.PROPOSED
                for record in language.grounding_trace.records
            )
        )
        relation_record = next(
            record
            for record in vision.grounding_trace.records
            if record.candidate.atom.predicate.name == "LEFT_OF"
        )
        self.assertIs(
            relation_record.decision.authority,
            GroundingAuthority.IMPORTED_PROPOSAL,
        )
        self.assertIs(relation_record.fact.status, FactStatus.PROPOSED)

    def test_composition_preserves_fact_provenance_and_grounding_records(self) -> None:
        first = parse_arithmetic_expression("1 + 2")
        second = parse_arithmetic_expression("3 + 4")
        composed = compose_domain_instances((first, second)).instance
        audit = composed.grounding_trace.audit(composed.state.facts)

        self.assertTrue(audit.coverage_complete)
        self.assertEqual(
            len(composed.grounding_trace.records),
            len(first.grounding_trace.records)
            + len(second.grounding_trace.records),
        )
        self.assertTrue(
            all(
                fact.evidence_status is EvidenceStatus.ADAPTER_VERIFIED
                for fact in composed.state.facts
            )
        )
        self.assertEqual(
            len(
                {
                    record.candidate.candidate_id
                    for record in composed.grounding_trace.records
                }
            ),
            len(composed.grounding_trace.records),
        )


if __name__ == "__main__":
    unittest.main()

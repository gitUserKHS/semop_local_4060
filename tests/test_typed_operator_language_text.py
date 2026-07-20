from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    FactStatus,
    LanguageTextAdapter,
    LanguageTextParser,
    LanguageTextProblem,
    OperatorKernel,
    TypedDomainRequest,
    UnifiedTypedReasoner,
)
from semop.structures import StructuredMeaningGraph


def _solve(text: str):
    instance = LanguageTextAdapter().adapt(
        LanguageTextProblem(text, use_legacy_heuristics=False)
    )
    result = OperatorKernel(instance.registry).solve(instance.state, instance.goals)
    return instance, result


class LanguageTextParserTests(unittest.TestCase):
    def test_english_labels_produce_observed_typed_claims(self) -> None:
        parsed = LanguageTextParser().parse(
            LanguageTextProblem(
                "Goal: deploy\n"
                "Requires: tests, approval\n"
                "Satisfied: tests\n"
                "Satisfied: approval",
                use_legacy_heuristics=False,
            )
        )

        self.assertFalse(parsed.heuristic_used)
        self.assertFalse(parsed.unparsed_statements)
        self.assertTrue(all(claim.verified for claim in parsed.claims))
        self.assertIn(
            ("REQUIRES", ("deploy", "approval")),
            {(claim.relation, claim.arguments) for claim in parsed.claims},
        )

    def test_korean_wording_preserves_words_containing_conjunction_syllables(self) -> None:
        parsed = LanguageTextParser().parse(
            LanguageTextProblem(
                "배포하려면 테스트 통과와 승인이 필요하다. "
                "테스트 통과가 충족되었다. 승인이 충족되었다.",
                use_legacy_heuristics=False,
            )
        )
        claims = {(claim.relation, claim.arguments) for claim in parsed.claims}

        self.assertIn(("REQUIRES", ("배포", "테스트_통과")), claims)
        self.assertIn(("REQUIRES", ("배포", "승인")), claims)
        self.assertIn(("SATISFIED", ("테스트_통과",)), claims)
        self.assertNotIn(("REQUIRES", ("배포", "테스트_통")), claims)

    def test_korean_without_prohibition_extracts_only_explicit_requirements(self) -> None:
        parsed = LanguageTextParser().parse(
            LanguageTextProblem(
                "관리자 승인과 안전 확인 없이 랙 적재를 진행하지 않는다.",
                use_legacy_heuristics=False,
            )
        )
        claims = {(claim.relation, claim.arguments) for claim in parsed.claims}

        self.assertFalse(parsed.unparsed_statements)
        self.assertIn(("GOAL", ("랙_적재",)), claims)
        self.assertIn(("REQUIRES", ("랙_적재", "관리자_승인")), claims)
        self.assertIn(("REQUIRES", ("랙_적재", "안전_확인")), claims)
        self.assertFalse(
            any(
                relation in {"SATISFIED", "BLOCKED"}
                for relation, _arguments in claims
            )
        )

    def test_ambiguous_text_does_not_fabricate_observed_claims(self) -> None:
        parsed = LanguageTextParser().parse(
            LanguageTextProblem(
                "Deployment might happen sometime after testing",
                use_legacy_heuristics=False,
            )
        )

        self.assertEqual(parsed.claims, ())
        self.assertEqual(len(parsed.unparsed_statements), 1)


class LanguageTextReasoningTests(unittest.TestCase):
    def test_ready_goal_is_replayed_from_two_requirements(self) -> None:
        instance, result = _solve(
            "Goal: deploy; Requires: tests, approval; "
            "Satisfied: tests; Satisfied: approval"
        )

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(len(result.proof), 3)
        self.assertEqual(str(instance.goals[0].atom), "READY(deploy)")

    def test_korean_sentence_reaches_the_same_typed_program(self) -> None:
        instance, result = _solve(
            "배포하려면 테스트 통과와 승인이 필요하다. "
            "테스트 통과가 충족되었다. 승인이 충족되었다."
        )

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(len(result.proof), 3)
        self.assertEqual(str(instance.goals[0].atom), "READY(배포)")

    def test_korean_without_prohibition_does_not_invent_completion(self) -> None:
        instance, result = _solve(
            "관리자 승인과 안전 확인 없이 랙 적재를 진행하지 않는다."
        )

        self.assertFalse(result.success)
        self.assertFalse(result.verified)
        self.assertEqual(str(instance.goals[0].atom), "READY(랙_적재)")
        self.assertEqual(instance.metadata["explicit_claim_count"], 3)
        self.assertFalse(
            any(
                fact.atom.predicate.name in {"SATISFIED", "BLOCKED"}
                for fact in instance.state.facts
            )
        )

    def test_korean_without_prohibition_proves_ready_after_explicit_completion(self) -> None:
        instance, result = _solve(
            "관리자 승인과 안전 확인 없이 랙 적재를 진행하지 않는다. "
            "관리자 승인이 충족되었다. 안전 확인이 충족되었다."
        )

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(str(instance.goals[0].atom), "READY(랙_적재)")
        self.assertEqual(len(result.proof), 3)

    def test_negated_requirement_proves_not_ready(self) -> None:
        instance, result = _solve(
            "Goal: release; Requires: security review; "
            "Security review is not satisfied."
        )

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(len(result.proof), 2)
        self.assertEqual(str(instance.goals[0].atom), "NOT_READY(release)")

    def test_missing_status_leaves_goal_unproved(self) -> None:
        instance, result = _solve("Can we deploy? Deploy requires tests.")

        self.assertFalse(result.success)
        self.assertFalse(result.verified)
        self.assertEqual(len(result.proof), 0)
        self.assertEqual(str(instance.goals[0].atom), "READY(deploy)")

    def test_conflicting_statuses_are_not_usable_as_evidence(self) -> None:
        instance, result = _solve(
            "Goal: deploy; Requires: tests; "
            "Satisfied: tests; Blocked: tests"
        )
        statuses = {
            fact.status
            for fact in instance.state.facts
            if fact.atom.predicate.name in {"SATISFIED", "BLOCKED"}
        }

        self.assertEqual(instance.metadata["contradictions"], ("tests",))
        self.assertEqual(statuses, {FactStatus.CONTRADICTED})
        self.assertFalse(result.success)
        self.assertEqual(len(result.proof), 0)

    def test_legacy_heuristic_candidates_remain_proposed(self) -> None:
        instance = LanguageTextAdapter().adapt(
            "책을 가방에 넣으려면 어떻게 해야 해?"
        )

        self.assertTrue(instance.metadata["heuristic_used"])
        self.assertGreater(instance.metadata["proposed_claim_count"], 0)
        self.assertEqual(instance.goals, ())
        self.assertTrue(instance.state.facts)
        self.assertTrue(
            all(fact.status is FactStatus.PROPOSED for fact in instance.state.facts)
        )


class LanguageTextRuntimeTests(unittest.TestCase):
    def test_runtime_accepts_plain_text_without_mutating_it(self) -> None:
        result = UnifiedTypedReasoner().run(
            TypedDomainRequest(
                "language",
                "Goal: publish; Requires: tests; Satisfied: tests",
                "typed",
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertFalse(result.projection_applied)
        self.assertIn("immutable source payload", result.diagnostics[0])

    def test_existing_structured_graph_projection_still_works(self) -> None:
        graph = StructuredMeaningGraph(
            query="배포해도 될까?",
            intent="decision",
            hidden_goals=["deploy"],
            required_premises=["tests"],
            satisfied_premises=["tests"],
        )
        result = UnifiedTypedReasoner().run(
            TypedDomainRequest("language", graph, "typed")
        )

        self.assertTrue(result.success)
        self.assertTrue(result.projection_applied)
        self.assertIsNotNone(graph.operator_execution)


if __name__ == "__main__":
    unittest.main()

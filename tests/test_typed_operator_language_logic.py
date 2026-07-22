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
    LanguageLogicAdapter,
    LanguageLogicParser,
    LanguageLogicProblem,
    OperatorKernel,
    TypedDomainRequest,
    UnifiedTypedReasoner,
)


def _solve(text: str):
    instance = LanguageLogicAdapter().adapt(text)
    result = OperatorKernel(instance.registry).solve(instance.state, instance.goals)
    return instance, result


class LanguageLogicParserTests(unittest.TestCase):
    def test_problem_requires_non_empty_text(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a string"):
            LanguageLogicProblem(123)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            LanguageLogicProblem("   ")

    def test_english_and_korean_universal_claims_share_the_same_ir(self) -> None:
        english = LanguageLogicParser().parse(
            "Every programmer is a person. Ada is a programmer. "
            "Prove: Ada is a person."
        )
        korean = LanguageLogicParser().parse(
            "모든 개발자는 사람이다. 아다는 개발자이다. "
            "증명: 아다는 사람이다."
        )

        self.assertEqual(
            [claim.relation for claim in english.claims],
            ["INSTANCE_OF", "SUBCLASS_OF", "INSTANCE_OF"],
        )
        self.assertEqual(
            [claim.relation for claim in korean.claims],
            ["INSTANCE_OF", "SUBCLASS_OF", "INSTANCE_OF"],
        )
        self.assertTrue(english.claims[-1].is_goal)
        self.assertTrue(korean.claims[-1].is_goal)

    def test_ambiguous_sentence_is_preserved_without_becoming_a_fact(self) -> None:
        parsed = LanguageLogicParser().parse(
            LanguageLogicProblem(
                "Ada might eventually become mortal. Prove: Ada is mortal."
            )
        )

        self.assertEqual(len(parsed.unparsed_statements), 1)
        self.assertEqual(len(parsed.claims), 1)
        self.assertTrue(parsed.claims[0].is_goal)

    def test_english_and_korean_conjunction_rules_share_the_same_ir(self) -> None:
        english = LanguageLogicParser().parse(
            "If something is red and square, then it is a marker. "
            "Prove: tile is a marker."
        )
        korean = LanguageLogicParser().parse(
            "만약 어떤 것이 빨강이고 정사각형이면 그것은 표식이다. "
            "증명: 타일은 표식이다."
        )

        self.assertEqual(english.rules[0].antecedents, ("red", "square"))
        self.assertEqual(english.rules[0].consequent, "marker")
        self.assertEqual(korean.rules[0].antecedents, ("빨강", "정사각형"))
        self.assertEqual(korean.rules[0].consequent, "표식")


class LanguageLogicReasoningTests(unittest.TestCase):
    def test_two_hop_concept_inheritance_is_replayed(self) -> None:
        instance, result = _solve(
            "Every programmer is a person. Every person is mortal. "
            "Ada is a programmer. Prove: Ada is mortal."
        )

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(len(result.proof), 2)
        self.assertEqual(str(instance.goals[0].atom), "INSTANCE_OF(ada, mortal)")
        self.assertTrue(
            all("language" in step.action.operator.tags for step in result.proof)
        )

    def test_korean_two_hop_inheritance_uses_the_same_operators(self) -> None:
        _instance, result = _solve(
            "모든 개발자는 사람이다. 모든 사람은 생명체이다. "
            "아다는 개발자이다. 증명: 아다는 생명체이다."
        )

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(len(result.proof), 2)

    def test_conflicting_classification_is_not_proof_evidence(self) -> None:
        instance, result = _solve(
            "Ada is a programmer. Ada is not a programmer. "
            "Prove: Ada is a programmer."
        )
        statuses = {
            fact.status
            for fact in instance.state.facts
            if fact.atom.predicate.name in {"INSTANCE_OF", "NOT_INSTANCE_OF"}
        }

        self.assertEqual(instance.metadata["contradictions"], (("ada", "programmer"),))
        self.assertEqual(statuses, {FactStatus.CONTRADICTED})
        self.assertFalse(result.success)

    def test_negative_facts_do_not_enable_contraposition(self) -> None:
        _instance, result = _solve(
            "Every programmer is a person. Ada is not a person. "
            "Prove: Ada is not a programmer."
        )

        self.assertFalse(result.success)
        self.assertEqual(len(result.proof), 0)

    def test_explicit_negative_blocks_a_conflicting_derived_classification(self) -> None:
        _instance, result = _solve(
            "Every programmer is a person. Ada is a programmer. "
            "Ada is not a person. Prove: Ada is a person."
        )

        self.assertFalse(result.success)
        self.assertEqual(len(result.proof), 0)

    def test_explicit_negative_goal_can_be_observed_directly(self) -> None:
        _instance, result = _solve(
            "Ada is not a robot. Prove: Ada is not a robot."
        )

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(len(result.proof), 0)

    def test_plain_text_runtime_dispatches_to_logic_adapter(self) -> None:
        result = UnifiedTypedReasoner().run(
            TypedDomainRequest(
                "language",
                "Every programmer is a person. Ada is a programmer. "
                "Prove: Ada is a person.",
                "typed",
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.instance.metadata["input_kind"], "logic_text")
        self.assertFalse(result.projection_applied)

    def test_conjunction_rule_requires_every_antecedent(self) -> None:
        _instance, positive = _solve(
            "Rule: red & square -> marker. Tile is red. Tile is square. "
            "Prove: Tile is a marker."
        )
        _instance, missing = _solve(
            "Rule: red & square -> marker. Tile is red. "
            "Prove: Tile is a marker."
        )

        self.assertTrue(positive.success and positive.verified)
        self.assertEqual(
            positive.proof[-1].action.operator.name,
            "classify_conjunction_000",
        )
        self.assertFalse(missing.success)

    def test_conjunction_rules_chain_and_respect_explicit_negation(self) -> None:
        _instance, chained = _solve(
            "Rule: red & square -> marker. Rule: marker & visible -> target. "
            "Tile is red. Tile is square. Tile is visible. "
            "Prove: Tile is a target."
        )
        _instance, negated = _solve(
            "Rule: red & square -> marker. Tile is red. Tile is square. "
            "Tile is not a marker. Prove: Tile is a marker."
        )

        self.assertTrue(chained.success and chained.verified)
        self.assertEqual(len(chained.proof), 2)
        self.assertFalse(negated.success)


if __name__ == "__main__":
    unittest.main()

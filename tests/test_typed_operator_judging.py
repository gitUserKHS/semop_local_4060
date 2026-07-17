from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    Fact,
    FactStatus,
    Goal,
    JudgeBoundaryError,
    JudgeCandidate,
    JudgeDecision,
    JudgeVerdict,
    KernelRegistry,
    OperatorKernel,
    Rule,
    TraceCorpus,
    WorldState,
    promote_judged_fact,
    stage_judged_fact,
    teacher_review_metadata,
    teacher_review_to_solve_result,
    verify_judged_program,
)
from semop.kernel.domains import make_hidden_premise_instance


def _decision(
    verdict: JudgeVerdict = JudgeVerdict.SUPPORTS,
    confidence: float = 0.95,
) -> JudgeDecision:
    return JudgeDecision(
        verdict=verdict,
        confidence=confidence,
        model_id="frontier-teacher-test",
        prompt_fingerprint="sha256:test-prompt-v1",
        rationale="The proposed typed relation is supported by the supplied evidence.",
        evidence_refs=("evidence:1",),
    )


class FrontierJudgeBoundaryTests(unittest.TestCase):
    def test_supported_fact_stays_proposed_until_independent_verification(self) -> None:
        instance = make_hidden_premise_instance(
            "deploy",
            ["approval"],
            satisfied=["approval"],
        )
        candidate = JudgeCandidate(
            "candidate-1",
            "language",
            "The deployment is ready.",
            instance.goals[0].atom,
            ("document:approval",),
        )

        record = stage_judged_fact(candidate, _decision())

        self.assertIsNotNone(record.staged_fact)
        self.assertEqual(record.staged_fact.status, FactStatus.PROPOSED)
        staged_state = WorldState((record.staged_fact,))
        self.assertFalse(staged_state.contains(candidate.atom))
        self.assertTrue(staged_state.contains(candidate.atom, proof_eligible=False))

        promoted = promote_judged_fact(record, lambda atom: atom == candidate.atom)
        self.assertEqual(promoted.status, FactStatus.OBSERVED)
        self.assertTrue(WorldState((promoted,)).contains(candidate.atom))

    def test_rejection_or_weak_support_cannot_stage_a_positive_fact(self) -> None:
        instance = make_hidden_premise_instance(
            "deploy",
            ["approval"],
            satisfied=["approval"],
        )
        candidate = JudgeCandidate(
            "candidate-2",
            "language",
            "The deployment is ready.",
            instance.goals[0].atom,
        )

        rejected = stage_judged_fact(
            candidate,
            _decision(JudgeVerdict.REJECTS),
        )
        weak = stage_judged_fact(candidate, _decision(confidence=0.49))

        self.assertIsNone(rejected.staged_fact)
        self.assertIsNone(weak.staged_fact)
        with self.assertRaises(JudgeBoundaryError):
            promote_judged_fact(rejected, lambda _atom: True)

    def test_replay_accepts_an_irredundant_judge_proposed_program(self) -> None:
        instance = make_hidden_premise_instance(
            "deploy",
            ["approval"],
            satisfied=["approval"],
        )
        kernel = OperatorKernel(instance.registry)
        solved = kernel.solve(instance.state, instance.goals)
        candidate = JudgeCandidate(
            "program-1",
            "language",
            "Approval makes deployment ready.",
            instance.goals[0].atom,
        )

        review = verify_judged_program(
            kernel,
            instance.state,
            instance.goals,
            tuple(step.action for step in solved.proof),
            candidate,
            _decision(),
        )

        self.assertTrue(review.accepted)
        self.assertEqual(len(review.proof), len(solved.proof))
        self.assertTrue(review.final_state.contains(candidate.atom))

        verified_result = teacher_review_to_solve_result(
            kernel,
            instance.state,
            instance.goals,
            review,
        )
        metadata = teacher_review_metadata(review)
        corpus = TraceCorpus()
        trace = corpus.add_result(
            "judge-program-1",
            "language",
            verified_result,
            source="verifier",
            metadata=metadata,
        )

        self.assertTrue(verified_result.success)
        self.assertTrue(verified_result.verified)
        self.assertEqual(verified_result.halt_reason, "judge_program_replay_verified")
        self.assertEqual(len(trace.actions), len(review.proof))
        self.assertIn(
            ("judge_model_id", "frontier-teacher-test"),
            trace.metadata,
        )
        self.assertIn(
            ("verification", "typed_execution_and_proof_replay"),
            trace.metadata,
        )

        already_true = verify_judged_program(
            kernel,
            solved.final_state,
            instance.goals,
            tuple(step.action for step in solved.proof),
            candidate,
            _decision(),
        )
        self.assertFalse(already_true.accepted)
        self.assertIn("already satisfied", already_true.diagnostics[0])

    def test_redundant_teacher_step_is_rejected(self) -> None:
        registry = KernelRegistry()
        entity = registry.types.register("Entity")
        registry.register_predicate("START", (entity,))
        registry.register_predicate("SIDE", (entity,))
        registry.register_predicate("DONE", (entity,))
        item = registry.symbol("item", entity)
        start = registry.atom("START", item)
        registry.register_operator(
            Rule("derive_side", (), (start,), (registry.atom("SIDE", item),))
        )
        registry.register_operator(
            Rule("finish", (), (start,), (registry.atom("DONE", item),))
        )
        state = WorldState((Fact(start),))
        goal = Goal(registry.atom("DONE", item))
        kernel = OperatorKernel(registry)
        actions = {
            action.operator.name: action
            for action in kernel.enumerate_actions(state)
        }
        candidate = JudgeCandidate(
            "program-2",
            "language",
            "The item is done.",
            goal.atom,
        )

        review = verify_judged_program(
            kernel,
            state,
            (goal,),
            (actions["derive_side"], actions["finish"]),
            candidate,
            _decision(),
        )

        self.assertFalse(review.accepted)
        self.assertIn("redundant", review.diagnostics[0])

    def test_tampered_action_payload_and_wrong_goal_are_rejected(self) -> None:
        instance = make_hidden_premise_instance(
            "deploy",
            ["approval"],
            satisfied=["approval"],
        )
        kernel = OperatorKernel(instance.registry)
        solved = kernel.solve(instance.state, instance.goals)
        first = solved.proof[0].action
        tampered = replace(first, effects=first.preconditions)
        candidate = JudgeCandidate(
            "program-3",
            "language",
            "Approval makes deployment ready.",
            instance.goals[0].atom,
        )

        invalid = verify_judged_program(
            kernel,
            instance.state,
            instance.goals,
            (tampered,),
            candidate,
            _decision(),
        )
        wrong_goal_candidate = replace(candidate, atom=first.effects[0])
        wrong_goal = verify_judged_program(
            kernel,
            instance.state,
            instance.goals,
            tuple(step.action for step in solved.proof),
            wrong_goal_candidate,
            _decision(),
        )

        self.assertFalse(invalid.accepted)
        self.assertIn("payload mismatch", invalid.diagnostics[0])
        self.assertFalse(wrong_goal.accepted)
        self.assertIn("not a verifier goal", wrong_goal.diagnostics[0])

    def test_rejected_or_context_swapped_review_cannot_enter_training(self) -> None:
        instance = make_hidden_premise_instance(
            "deploy",
            ["approval"],
            satisfied=["approval"],
        )
        kernel = OperatorKernel(instance.registry)
        solved = kernel.solve(instance.state, instance.goals)
        candidate = JudgeCandidate(
            "program-4",
            "language",
            "Approval makes deployment ready.",
            instance.goals[0].atom,
        )
        accepted = verify_judged_program(
            kernel,
            instance.state,
            instance.goals,
            tuple(step.action for step in solved.proof),
            candidate,
            _decision(),
        )
        rejected = replace(accepted, accepted=False)
        swapped_state = WorldState()

        with self.assertRaises(JudgeBoundaryError):
            teacher_review_to_solve_result(
                kernel,
                instance.state,
                instance.goals,
                rejected,
            )
        with self.assertRaises(JudgeBoundaryError):
            teacher_review_metadata(rejected)
        with self.assertRaises(JudgeBoundaryError):
            teacher_review_to_solve_result(
                kernel,
                swapped_state,
                instance.goals,
                accepted,
            )

    def test_forged_accepted_review_is_replayed_before_conversion(self) -> None:
        instance = make_hidden_premise_instance(
            "deploy",
            ["approval"],
            satisfied=["approval"],
        )
        kernel = OperatorKernel(instance.registry)
        solved = kernel.solve(instance.state, instance.goals)
        candidate = JudgeCandidate(
            "program-5",
            "language",
            "Approval makes deployment ready.",
            instance.goals[0].atom,
        )
        accepted = verify_judged_program(
            kernel,
            instance.state,
            instance.goals,
            tuple(step.action for step in solved.proof),
            candidate,
            _decision(),
        )
        forged = replace(accepted, final_state=instance.state)

        with self.assertRaises(JudgeBoundaryError):
            teacher_review_to_solve_result(
                kernel,
                instance.state,
                instance.goals,
                forged,
            )


if __name__ == "__main__":
    unittest.main()

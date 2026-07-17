from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
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
    GoalDirectedPolicy,
    KernelRegistry,
    OperatorKernel,
    PolicyDecision,
    ProofStep,
    Rule,
    SolveBudget,
    TypeSystem,
    TypeValidationError,
    WorldState,
    parse_arithmetic_expression,
)
from semop.kernel.domains import parse_geometry_dsl


class TypedKernelTests(unittest.TestCase):
    def test_monotonic_search_does_not_enumerate_fact_subsets(self) -> None:
        problem = parse_arithmetic_expression(
            "((1 + 2) * (3 + 4) - (5 / 2)) / (6 - 5)"
        )

        result = OperatorKernel(problem.registry).solve(
            problem.state,
            problem.goals,
        )

        self.assertTrue(result.success)
        self.assertEqual(len(result.proof), 15)
        self.assertEqual(result.inference_rounds, 5)
        self.assertLessEqual(result.expansions, len(result.proof) + 2)

    def test_policy_can_skip_irrelevant_monotonic_actions(self) -> None:
        registry = KernelRegistry()
        entity = registry.types.register("Entity")
        registry.register_predicate("START", (entity,))
        registry.register_predicate("CANDIDATE", (entity,))
        registry.register_predicate("DONE", (entity,))
        item = registry.symbol("item", entity)
        for index in range(12):
            noise = registry.symbol(f"noise_{index:02d}", entity)
            registry.register_operator(
                Rule(
                    f"candidate_noise_{index:02d}",
                    (),
                    (registry.atom("START", item),),
                    (registry.atom("CANDIDATE", noise),),
                )
            )
        registry.register_operator(
            Rule(
                "zz_select_item",
                (),
                (registry.atom("START", item),),
                (registry.atom("CANDIDATE", item),),
            )
        )
        registry.register_operator(
            Rule(
                "finish_item",
                (),
                (registry.atom("CANDIDATE", item),),
                (registry.atom("DONE", item),),
            )
        )
        state = WorldState((Fact(registry.atom("START", item)),))
        goals = (Goal(registry.atom("DONE", item)),)
        budget = SolveBudget(
            beam_width=1,
            top_operators=2,
            max_expansions=100,
        )

        baseline = OperatorKernel(registry).solve(state, goals, budget=budget)
        guided = OperatorKernel(registry).solve(
            state,
            goals,
            policy=GoalDirectedPolicy(),
            budget=budget,
        )

        self.assertTrue(baseline.success and guided.success)
        self.assertEqual(len(baseline.proof), 2)
        self.assertLess(guided.expansions, 0.7 * baseline.expansions)

    def test_policy_receives_grounded_operator_frontier_without_changing_goal_check(self) -> None:
        registry = KernelRegistry()
        entity = registry.types.register("Entity")
        registry.register_predicate("START", (entity,))
        registry.register_predicate("CANDIDATE", (entity,))
        registry.register_predicate("DONE", (entity,))
        item = registry.symbol("item", entity)
        noise = registry.symbol("noise", entity)
        for name, target in (
            ("candidate_noise", noise),
            ("candidate_item", item),
        ):
            registry.register_operator(
                Rule(
                    name,
                    (),
                    (registry.atom("START", item),),
                    (registry.atom("CANDIDATE", target),),
                )
            )
        registry.register_operator(
            Rule(
                "finish_item",
                (),
                (registry.atom("CANDIDATE", item),),
                (registry.atom("DONE", item),),
            )
        )
        seen_goal_sets: list[set[str]] = []

        class FrontierPolicy:
            def score_actions(self, _state, policy_goals, actions):
                atoms = {str(goal.atom) for goal in policy_goals}
                seen_goal_sets.append(atoms)
                return PolicyDecision(
                    tuple(
                        100.0
                        if any(str(effect) in atoms for effect in action.effects)
                        else 0.0
                        for action in actions
                    ),
                    action_limit=1,
                )

        result = OperatorKernel(registry).solve(
            WorldState((Fact(registry.atom("START", item)),)),
            (Goal(registry.atom("DONE", item)),),
            policy=FrontierPolicy(),
        )

        self.assertTrue(result.success and result.verified)
        self.assertEqual(result.expansions, 2)
        self.assertIn("CANDIDATE(item)", seen_goal_sets[0])
        self.assertNotIn("CANDIDATE(noise)", seen_goal_sets[0])

    def test_goal_policy_prefers_verifier_goal_over_frontier(self) -> None:
        problem = parse_geometry_dsl(
            "point A, B, M\n"
            "assume midpoint(M, A, B)\n"
            "prove collinear(A, M, B)\n"
            "prove equal_length(segment(A, M), segment(M, B))\n"
        )
        actions = OperatorKernel(problem.registry).enumerate_actions(problem.state)
        policy_goals = (
            problem.goals[1],
            Goal(problem.goals[0].atom, label="operator_frontier"),
        )

        decision = GoalDirectedPolicy().score_actions(
            problem.state,
            policy_goals,
            actions,
        )
        scores = {
            action.operator.name: score
            for action, score in zip(actions, decision.action_scores, strict=True)
        }

        self.assertGreater(
            scores["midpoint_implies_equal_lengths"],
            scores["midpoint_implies_collinear"],
        )
        self.assertEqual(decision.action_limit, 1)

    def test_recursive_term_frontier_is_bounded_by_solve_budget(self) -> None:
        registry = KernelRegistry()
        entity = registry.types.register("Entity")
        wrap = registry.register_function("wrap", (entity,), entity)
        registry.register_predicate("READY", (entity,))
        item = registry.symbol("item", entity)
        candidate = registry.variable("candidate", entity)
        registry.register_operator(
            Rule(
                "unwrap_ready",
                (candidate,),
                (registry.atom("READY", registry.apply(wrap, candidate)),),
                (registry.atom("READY", candidate),),
            )
        )
        state = WorldState(
            (Fact(registry.atom("READY", registry.apply(wrap, item))),)
        )

        result = OperatorKernel(registry).solve(
            state,
            (Goal(registry.atom("READY", item)),),
            policy=GoalDirectedPolicy(),
            budget=SolveBudget(max_steps=4, max_facts=8, max_expansions=8),
        )

        self.assertTrue(result.success and result.verified)
        self.assertEqual(result.expansions, 1)

    def test_core_imports_without_site_packages(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(SRC)
        completed = subprocess.run(
            [sys.executable, "-S", "-c", "import semop.kernel"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_nominal_single_inheritance_is_assignable(self) -> None:
        types = TypeSystem()
        entity = types.register("Entity")
        point = types.register("Point", entity)
        special = types.register("SpecialPoint", point)

        self.assertTrue(types.is_assignable(special, entity))
        self.assertTrue(types.is_assignable(special, point))
        self.assertFalse(types.is_assignable(entity, point))

    def test_registry_canonicalizes_symmetric_atoms(self) -> None:
        registry = KernelRegistry()
        entity = registry.types.register("Entity")
        registry.register_predicate(
            "SAME", (entity, entity), symmetry_groups=((0, 1),)
        )
        a = registry.symbol("A", entity)
        b = registry.symbol("B", entity)

        self.assertEqual(
            registry.atom("SAME", a, b), registry.atom("SAME", b, a)
        )

    def test_effect_variables_must_be_bound_by_preconditions(self) -> None:
        registry = KernelRegistry()
        entity = registry.types.register("Entity")
        registry.register_predicate("KNOWN", (entity,))
        x = registry.variable("x", entity)

        with self.assertRaises(TypeValidationError):
            registry.register_operator(
                Rule(
                    "invent_fact",
                    (x,),
                    (),
                    (registry.atom("KNOWN", x),),
                )
            )

    def test_proposed_and_unverified_facts_are_not_proof_premises(self) -> None:
        registry = KernelRegistry()
        entity = registry.types.register("Entity")
        registry.register_predicate("SEEN", (entity,))
        registry.register_predicate("DONE", (entity,))
        unknown = registry.register_predicate("UNKNOWN", (entity,), verified=False)
        x = registry.variable("x", entity)
        registry.register_operator(
            Rule(
                "finish_seen",
                (x,),
                (registry.atom("SEEN", x),),
                (registry.atom("DONE", x),),
            )
        )
        a = registry.symbol("A", entity)
        proposed = WorldState(
            (Fact(registry.atom("SEEN", a), FactStatus.PROPOSED),)
        )
        unverified = WorldState(
            (Fact(registry.atom(unknown, a), FactStatus.OBSERVED),)
        )

        self.assertFalse(
            OperatorKernel(registry).solve(
                proposed, (Goal(registry.atom("DONE", a)),)
            ).success
        )
        self.assertEqual(unverified.eligible_facts, ())

    def test_success_is_replayed_and_tampering_is_rejected(self) -> None:
        registry, state, goal = _chain_problem()
        kernel = OperatorKernel(registry)
        result = kernel.solve(state, (goal,))

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(len(result.proof), 2)
        tampered_first = replace(result.proof[0], after_digest="bad")
        replay = kernel.replay(state, (goal,), (tampered_first, *result.proof[1:]))
        self.assertFalse(replay.verified)
        self.assertIn("digest mismatch", replay.diagnostics[0])

    def test_policy_halt_is_rejected_until_goal_is_verified(self) -> None:
        registry, state, goal = _chain_problem()

        class EarlyHalt:
            def score_actions(self, _state, _goals, actions):
                return PolicyDecision(tuple(0.0 for _ in actions), halt_probability=1.0)

        result = OperatorKernel(registry).solve(
            state,
            (goal,),
            policy=EarlyHalt(),
            budget=SolveBudget(max_expansions=50),
        )

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertTrue(
            any("halt was rejected" in diagnostic for diagnostic in result.diagnostics)
        )

    def test_policy_can_request_a_narrower_action_beam(self) -> None:
        registry = KernelRegistry()
        entity = registry.types.register("Entity")
        registry.register_predicate("START", (entity,))
        registry.register_predicate("DONE", (entity,))
        item = registry.symbol("item", entity)
        noise = registry.symbol("noise", entity)
        for name, target in (("finish_noise", noise), ("finish_item", item)):
            registry.register_operator(
                Rule(
                    name,
                    (),
                    (registry.atom("START", item),),
                    (registry.atom("DONE", target),),
                )
            )

        class GreedyPolicy:
            def score_actions(self, _state, _goals, actions):
                return PolicyDecision(
                    tuple(
                        10.0 if str(action.effects[0]) == "DONE(item)" else 0.0
                        for action in actions
                    ),
                    action_limit=1,
                )

        result = OperatorKernel(registry).solve(
            WorldState((Fact(registry.atom("START", item)),)),
            (Goal(registry.atom("DONE", item)),),
            policy=GreedyPolicy(),
            budget=SolveBudget(beam_width=4),
        )

        self.assertTrue(result.success and result.verified)
        self.assertEqual(result.expansions, 1)

    def test_policy_exception_uses_deterministic_fallback(self) -> None:
        registry, state, goal = _chain_problem()

        class BrokenPolicy:
            def score_actions(self, _state, _goals, _actions):
                raise RuntimeError("broken")

        result = OperatorKernel(registry).solve(
            state,
            (goal,),
            policy=BrokenPolicy(),
            budget=SolveBudget(max_expansions=50),
        )

        self.assertTrue(result.success)
        self.assertTrue(result.policy_used)
        self.assertTrue(any("policy failure" in item for item in result.diagnostics))

    def test_fact_limit_stops_term_generating_operator(self) -> None:
        registry = KernelRegistry()
        entity = registry.types.register("Entity")
        registry.register_function("wrap", (entity,), entity)
        registry.register_predicate("SEEN", (entity,))
        registry.register_predicate("DONE", (entity,))
        x = registry.variable("x", entity)
        registry.register_operator(
            Rule(
                "grow_term",
                (x,),
                (registry.atom("SEEN", x),),
                (registry.atom("SEEN", registry.apply("wrap", x)),),
            ),
            family="compose",
        )
        registry.register_operator(
            Rule(
                "seen_to_done",
                (x,),
                (registry.atom("SEEN", x),),
                (registry.atom("DONE", x),),
            ),
            family="verify",
        )
        a = registry.symbol("A", entity)
        unreachable = a
        for _ in range(8):
            unreachable = registry.apply("wrap", unreachable)
        result = OperatorKernel(registry).solve(
            WorldState((Fact(registry.atom("SEEN", a)),)),
            (Goal(registry.atom("DONE", unreachable)),),
            budget=SolveBudget(max_facts=4, max_expansions=100),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.halt_reason, "max_facts")
        self.assertLessEqual(len(result.final_state.facts), 4)

    def test_duplicate_effect_cycle_terminates_without_expansion(self) -> None:
        registry = KernelRegistry()
        entity = registry.types.register("Entity")
        registry.register_predicate("SEEN", (entity,))
        registry.register_predicate("DONE", (entity,))
        x = registry.variable("x", entity)
        registry.register_operator(
            Rule(
                "repeat_seen",
                (x,),
                (registry.atom("SEEN", x),),
                (registry.atom("SEEN", x),),
            )
        )
        a = registry.symbol("A", entity)
        result = OperatorKernel(registry).solve(
            WorldState((Fact(registry.atom("SEEN", a)),)),
            (Goal(registry.atom("DONE", a)),),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.halt_reason, "no_solution")
        self.assertEqual(result.expansions, 0)


def _chain_problem():
    registry = KernelRegistry()
    entity = registry.types.register("Entity")
    registry.register_predicate("START", (entity,))
    registry.register_predicate("MIDDLE", (entity,))
    registry.register_predicate("END", (entity,))
    x = registry.variable("x", entity)
    registry.register_operator(
        Rule(
            "start_to_middle",
            (x,),
            (registry.atom("START", x),),
            (registry.atom("MIDDLE", x),),
        )
    )
    registry.register_operator(
        Rule(
            "middle_to_end",
            (x,),
            (registry.atom("MIDDLE", x),),
            (registry.atom("END", x),),
        )
    )
    a = registry.symbol("A", entity)
    state = WorldState((Fact(registry.atom("START", a)),))
    return registry, state, Goal(registry.atom("END", a))


if __name__ == "__main__":
    unittest.main()

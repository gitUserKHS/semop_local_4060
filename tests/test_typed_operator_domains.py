from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import OperatorKernel, render_proof_ko
from semop.kernel.domains import (
    GeometryDslError,
    make_hidden_premise_instance,
    parse_geometry_dsl,
    parse_grid_problem,
)


class TypedDomainTests(unittest.TestCase):
    def test_geometry_midpoint_and_korean_proof(self) -> None:
        problem = parse_geometry_dsl(
            "point A, B, M\n"
            "let AM: Segment = segment(A, M)\n"
            "assume midpoint(M, A, B)\n"
            "prove collinear(A, M, B)\n"
            "prove equal_length(AM, segment(M, B))\n"
        )
        result = OperatorKernel(problem.registry).solve(problem.state, problem.goals)

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        rendered = render_proof_ko(result)
        self.assertIn("중점", rendered)
        self.assertIn("검증 완료", rendered)

    def test_geometry_transitivity_is_symmetry_aware(self) -> None:
        problem = parse_geometry_dsl(
            "segment Long, Shared, Short\n"
            "assume equal_length(Shared, Long)\n"
            "assume equal_length(Short, Shared)\n"
            "prove equal_length(Long, Short)\n"
        )

        result = OperatorKernel(problem.registry).solve(problem.state, problem.goals)
        self.assertTrue(result.success)
        self.assertEqual(result.proof[-1].action.operator.name, "equal_length_transitivity")

    def test_geometry_dsl_reports_locations_and_degenerate_terms(self) -> None:
        with self.assertRaises(GeometryDslError) as undeclared:
            parse_geometry_dsl("point A\nprove collinear(A, B, A)\n")
        self.assertEqual(undeclared.exception.location.line, 2)
        self.assertGreater(undeclared.exception.location.column, 1)

        with self.assertRaises(GeometryDslError) as degenerate:
            parse_geometry_dsl(
                "point A, B\nprove parallel(line(A, A), line(A, B))\n"
            )
        self.assertIn("degenerate", str(degenerate.exception))

    def test_hidden_premise_ready_blocked_and_unknown(self) -> None:
        ready = make_hidden_premise_instance(
            "deploy", ["tests", "approval"], satisfied=["tests", "approval"]
        )
        blocked = make_hidden_premise_instance(
            "deploy", ["approval"], blocked=["approval"], target="not_ready"
        )
        unknown = make_hidden_premise_instance("deploy", ["approval"])

        self.assertTrue(OperatorKernel(ready.registry).solve(ready.state, ready.goals).success)
        self.assertTrue(
            OperatorKernel(blocked.registry).solve(blocked.state, blocked.goals).success
        )
        self.assertFalse(
            OperatorKernel(unknown.registry).solve(unknown.state, unknown.goals).success
        )

    def test_grid_reachability_and_no_path(self) -> None:
        reachable = parse_grid_problem("S..\n##.\n..G")
        blocked = parse_grid_problem("S#G")

        solved = OperatorKernel(reachable.registry).solve(
            reachable.state, reachable.goals
        )
        failed = OperatorKernel(blocked.registry).solve(blocked.state, blocked.goals)
        self.assertTrue(solved.success)
        self.assertEqual(len(solved.proof), 4)
        self.assertFalse(failed.success)

    def test_proof_order_is_deterministic(self) -> None:
        problem = make_hidden_premise_instance(
            "deploy", ["b", "a"], satisfied=["a", "b"]
        )
        kernel = OperatorKernel(problem.registry)

        first = kernel.solve(problem.state, problem.goals)
        second = kernel.solve(problem.state, problem.goals)
        self.assertEqual(
            [step.action.canonical_key() for step in first.proof],
            [step.action.canonical_key() for step in second.proof],
        )


if __name__ == "__main__":
    unittest.main()

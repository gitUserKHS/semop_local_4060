from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import OperatorKernel, render_proof_ko
from semop.kernel.domains import (
    make_hidden_premise_instance,
    parse_geometry_dsl,
    parse_grid_problem,
)


PROBLEMS = (
    parse_geometry_dsl(
        "point A, B, M\n"
        "assume midpoint(M, A, B)\n"
        "prove equal_length(segment(A, M), segment(M, B))\n"
    ),
    make_hidden_premise_instance(
        "deploy",
        ["tests_pass", "approval"],
        satisfied=["tests_pass", "approval"],
    ),
    parse_grid_problem("S..\n##.\n..G"),
)


for problem in PROBLEMS:
    result = OperatorKernel(problem.registry).solve(problem.state, problem.goals)
    print(f"\n[{problem.domain}] success={result.success} verified={result.verified}")
    print(render_proof_ko(result))

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    JudgeCandidate,
    JudgeDecision,
    JudgeVerdict,
    OperatorKernel,
    TraceCorpus,
    stage_judged_fact,
    teacher_review_metadata,
    teacher_review_to_solve_result,
    verify_judged_program,
)
from semop.kernel.domains import make_hidden_premise_instance


def main() -> None:
    problem = make_hidden_premise_instance(
        "deploy",
        ["approval"],
        satisfied=["approval"],
    )
    kernel = OperatorKernel(problem.registry)

    # This verified baseline only stands in for typed actions proposed by an LLM.
    proposed_actions = tuple(
        step.action for step in kernel.solve(problem.state, problem.goals).proof
    )
    candidate = JudgeCandidate(
        candidate_id="demo-program-1",
        domain="language",
        statement="Approval makes deployment ready.",
        atom=problem.goals[0].atom,
        evidence=("sop:approval-required",),
    )
    decision = JudgeDecision(
        verdict=JudgeVerdict.SUPPORTS,
        confidence=0.94,
        model_id="frontier-llm-placeholder",
        prompt_fingerprint="sha256:demo-prompt-v1",
        rationale="The typed program matches the supplied prerequisite.",
        evidence_refs=("sop:approval-required",),
    )

    staged = stage_judged_fact(candidate, decision)
    print("judge fact status:", staged.staged_fact.status.value)

    review = verify_judged_program(
        kernel,
        problem.state,
        problem.goals,
        proposed_actions,
        candidate,
        decision,
    )
    result = teacher_review_to_solve_result(
        kernel,
        problem.state,
        problem.goals,
        review,
    )
    corpus = TraceCorpus()
    trace = corpus.add_result(
        "frontier-judge-demo",
        "language",
        result,
        source="verifier",
        metadata=teacher_review_metadata(review),
    )

    print("program accepted:", review.accepted)
    print("replay verified:", result.verified)
    print("training actions:", len(trace.actions))
    print("audit metadata:", dict(trace.metadata))


if __name__ == "__main__":
    main()

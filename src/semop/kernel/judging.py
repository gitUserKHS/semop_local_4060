from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from .engine import OperatorKernel
from .model import (
    Atom,
    Fact,
    FactStatus,
    Goal,
    GoalOutcome,
    GroundAction,
    KernelError,
    ProofStep,
    SolveResult,
    WorldState,
)


class JudgeBoundaryError(KernelError):
    """Raised when a semantic judge tries to cross a verifier boundary."""


class JudgeVerdict(str, Enum):
    SUPPORTS = "supports"
    REJECTS = "rejects"
    ABSTAINS = "abstains"


@dataclass(frozen=True)
class JudgeCandidate:
    candidate_id: str
    domain: str
    statement: str
    atom: Atom
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("judge candidate id cannot be empty")
        if not self.domain.strip():
            raise ValueError("judge candidate domain cannot be empty")
        if not self.statement.strip():
            raise ValueError("judge candidate statement cannot be empty")
        object.__setattr__(self, "evidence", tuple(self.evidence))


@dataclass(frozen=True)
class JudgeDecision:
    verdict: JudgeVerdict
    confidence: float
    model_id: str
    prompt_fingerprint: str
    rationale: str
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("judge confidence must be between 0 and 1")
        if not self.model_id.strip():
            raise ValueError("judge model id cannot be empty")
        if not self.prompt_fingerprint.strip():
            raise ValueError("judge prompt fingerprint cannot be empty")
        if not self.rationale.strip():
            raise ValueError("judge rationale cannot be empty")
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))


@runtime_checkable
class SemanticJudge(Protocol):
    """Provider-neutral boundary for a frontier model or human reviewer."""

    def judge(self, candidate: JudgeCandidate) -> JudgeDecision: ...


@dataclass(frozen=True)
class JudgedFactRecord:
    candidate: JudgeCandidate
    decision: JudgeDecision
    staged_fact: Fact | None
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class TeacherProgramReview:
    candidate: JudgeCandidate
    decision: JudgeDecision
    accepted: bool
    proof: tuple[ProofStep, ...]
    final_state: WorldState
    diagnostics: tuple[str, ...] = ()


def stage_judged_fact(
    candidate: JudgeCandidate,
    decision: JudgeDecision,
    *,
    min_confidence: float = 0.5,
) -> JudgedFactRecord:
    """Stage supported model output as proof-ineligible, never as observed truth."""

    _validate_confidence_threshold(min_confidence)
    if decision.verdict is not JudgeVerdict.SUPPORTS:
        return JudgedFactRecord(
            candidate,
            decision,
            None,
            (f"judge verdict was {decision.verdict.value}",),
        )
    if decision.confidence < min_confidence:
        return JudgedFactRecord(
            candidate,
            decision,
            None,
            (
                f"judge confidence {decision.confidence:.3f} was below "
                f"{min_confidence:.3f}",
            ),
        )
    source = (
        f"semantic_judge:{decision.model_id}:"
        f"{decision.prompt_fingerprint}"
    )
    return JudgedFactRecord(
        candidate,
        decision,
        Fact(
            candidate.atom,
            FactStatus.PROPOSED,
            source,
            decision.confidence,
        ),
    )


def promote_judged_fact(
    record: JudgedFactRecord,
    verifier: Callable[[Atom], bool],
    *,
    source: str = "deterministic_judge_verifier",
) -> Fact:
    """Promote only after an independent deterministic verifier accepts the atom."""

    if record.staged_fact is None:
        raise JudgeBoundaryError("judge decision did not stage a positive fact")
    if not record.candidate.atom.predicate.verified:
        raise JudgeBoundaryError("an unverified predicate cannot be promoted")
    try:
        verified = bool(verifier(record.candidate.atom))
    except Exception as exc:
        raise JudgeBoundaryError(
            f"deterministic fact verifier failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not verified:
        raise JudgeBoundaryError("deterministic fact verifier rejected the atom")
    return Fact(
        record.candidate.atom,
        FactStatus.OBSERVED,
        source,
        1.0,
    )


def verify_judged_program(
    kernel: OperatorKernel,
    initial_state: WorldState,
    goals: Sequence[Goal],
    actions: Sequence[GroundAction],
    candidate: JudgeCandidate,
    decision: JudgeDecision,
    *,
    max_steps: int = 6,
    min_confidence: float = 0.5,
    require_irredundant: bool = True,
) -> TeacherProgramReview:
    """Accept a judge-proposed program only after execution and proof replay."""

    _validate_confidence_threshold(min_confidence)
    if max_steps <= 0:
        raise ValueError("judge program max_steps must be positive")
    normalized_goals = tuple(goals)
    normalized_actions = tuple(actions)
    if decision.verdict is not JudgeVerdict.SUPPORTS:
        return _rejected_program(
            candidate,
            decision,
            initial_state,
            f"judge verdict was {decision.verdict.value}",
        )
    if decision.confidence < min_confidence:
        return _rejected_program(
            candidate,
            decision,
            initial_state,
            f"judge confidence was below {min_confidence:.3f}",
        )
    if candidate.atom not in {goal.atom for goal in normalized_goals}:
        return _rejected_program(
            candidate,
            decision,
            initial_state,
            "judge candidate atom is not a verifier goal",
        )
    if initial_state.contains(candidate.atom):
        return _rejected_program(
            candidate,
            decision,
            initial_state,
            "judge candidate goal was already satisfied before the program",
        )
    if not normalized_actions:
        return _rejected_program(
            candidate,
            decision,
            initial_state,
            "judge proposed an empty operator program",
        )
    if len(normalized_actions) > max_steps:
        return _rejected_program(
            candidate,
            decision,
            initial_state,
            f"judge program exceeded the {max_steps}-step limit",
        )

    materialized = _materialize_program(kernel, initial_state, normalized_actions)
    if materialized[0] is None:
        return _rejected_program(
            candidate,
            decision,
            materialized[1],
            *materialized[2],
        )
    proof, final_state, diagnostics = materialized
    replay = kernel.replay(initial_state, normalized_goals, proof)
    if not replay.verified:
        return _rejected_program(
            candidate,
            decision,
            replay.final_state,
            *(diagnostics + replay.diagnostics),
        )

    if require_irredundant and len(normalized_actions) > 1:
        for index in range(len(normalized_actions)):
            reduced = normalized_actions[:index] + normalized_actions[index + 1 :]
            reduced_materialized = _materialize_program(
                kernel,
                initial_state,
                reduced,
            )
            reduced_proof = reduced_materialized[0]
            if reduced_proof is None:
                continue
            reduced_replay = kernel.replay(
                initial_state,
                normalized_goals,
                reduced_proof,
            )
            if reduced_replay.verified:
                return _rejected_program(
                    candidate,
                    decision,
                    final_state,
                    f"operator step {index + 1} was redundant",
                )

    return TeacherProgramReview(
        candidate,
        decision,
        True,
        proof,
        final_state,
        diagnostics,
    )


def teacher_review_to_solve_result(
    kernel: OperatorKernel,
    initial_state: WorldState,
    goals: Sequence[Goal],
    review: TeacherProgramReview,
    *,
    max_steps: int = 6,
    min_confidence: float = 0.5,
    require_irredundant: bool = True,
) -> SolveResult:
    """Convert an accepted teacher program into replay-verified training data.

    ``TeacherProgramReview.accepted`` is only an audit field. This function does
    not trust it by itself: it reconstructs the proposed actions against the
    supplied registry and state, reruns guards, and replays the resulting proof.
    """

    normalized_goals = tuple(goals)
    if not normalized_goals:
        raise JudgeBoundaryError("teacher trace conversion requires at least one goal")
    if not review.accepted:
        raise JudgeBoundaryError(
            "a rejected teacher program cannot become training data"
        )
    if not review.proof:
        raise JudgeBoundaryError("an empty teacher proof cannot become training data")

    revalidated = verify_judged_program(
        kernel,
        initial_state,
        normalized_goals,
        tuple(step.action for step in review.proof),
        review.candidate,
        review.decision,
        max_steps=max_steps,
        min_confidence=min_confidence,
        require_irredundant=require_irredundant,
    )
    if not revalidated.accepted:
        detail = "; ".join(revalidated.diagnostics) or "verification failed"
        raise JudgeBoundaryError(
            f"teacher program failed conversion replay: {detail}"
        )
    if (
        revalidated.proof != review.proof
        or revalidated.final_state != review.final_state
    ):
        raise JudgeBoundaryError(
            "teacher review payload did not match the independently replayed program"
        )

    outcomes = tuple(
        GoalOutcome(
            goal=goal,
            proven=revalidated.final_state.contains(goal.atom),
            proof_step=next(
                (
                    step.index
                    for step in revalidated.proof
                    if goal.atom in step.effects
                ),
                None,
            ),
        )
        for goal in normalized_goals
    )
    if not all(outcome.proven for outcome in outcomes):
        raise JudgeBoundaryError("teacher program did not prove every verifier goal")

    return SolveResult(
        success=True,
        verified=True,
        goals=outcomes,
        initial_state=initial_state,
        final_state=revalidated.final_state,
        proof=revalidated.proof,
        expansions=len(revalidated.proof),
        elapsed_seconds=0.0,
        halt_reason="judge_program_replay_verified",
        policy_used=True,
        fallback_used=False,
        diagnostics=(
            "semantic judge proposed the program; typed execution and proof replay "
            "verified it",
            f"judge_model={review.decision.model_id}",
            f"judge_prompt_fingerprint={review.decision.prompt_fingerprint}",
        ),
        inference_rounds=len(revalidated.proof),
    )


def teacher_review_metadata(review: TeacherProgramReview) -> dict[str, str]:
    """Return stable audit metadata suitable for ``TraceCorpus.add_result``."""

    if not review.accepted:
        raise JudgeBoundaryError(
            "a rejected teacher program has no verified trace metadata"
        )
    return {
        "teacher_kind": "semantic_judge",
        "judge_candidate_id": review.candidate.candidate_id,
        "judge_domain": review.candidate.domain,
        "judge_model_id": review.decision.model_id,
        "judge_prompt_fingerprint": review.decision.prompt_fingerprint,
        "judge_verdict": review.decision.verdict.value,
        "judge_confidence": f"{review.decision.confidence:.6f}",
        "judge_evidence_refs": "|".join(review.decision.evidence_refs),
        "verification": "typed_execution_and_proof_replay",
    }


def _materialize_program(
    kernel: OperatorKernel,
    initial_state: WorldState,
    actions: tuple[GroundAction, ...],
) -> tuple[tuple[ProofStep, ...] | None, WorldState, tuple[str, ...]]:
    state = initial_state
    proof: list[ProofStep] = []
    for index, action in enumerate(actions, start=1):
        before_digest = state.digest()
        try:
            next_state = kernel.execute_action(state, action)
        except KernelError as exc:
            return None, state, (f"operator step {index} was rejected: {exc}",)
        after_digest = next_state.digest()
        if after_digest == before_digest:
            return None, state, (f"operator step {index} produced no new fact",)
        proof.append(
            ProofStep(
                index=index,
                action=action,
                premises=action.preconditions,
                effects=action.effects,
                before_digest=before_digest,
                after_digest=after_digest,
            )
        )
        state = next_state
    return tuple(proof), state, ()


def _rejected_program(
    candidate: JudgeCandidate,
    decision: JudgeDecision,
    state: WorldState,
    *diagnostics: str,
) -> TeacherProgramReview:
    return TeacherProgramReview(
        candidate,
        decision,
        False,
        (),
        state,
        tuple(item for item in diagnostics if item),
    )


def _validate_confidence_threshold(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError("minimum judge confidence must be between 0 and 1")


__all__ = [
    "JudgeBoundaryError",
    "JudgeCandidate",
    "JudgeDecision",
    "JudgeVerdict",
    "JudgedFactRecord",
    "SemanticJudge",
    "TeacherProgramReview",
    "promote_judged_fact",
    "stage_judged_fact",
    "teacher_review_metadata",
    "teacher_review_to_solve_result",
    "verify_judged_program",
]

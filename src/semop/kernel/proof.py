from __future__ import annotations

from .model import SolveResult


def render_proof_ko(result: SolveResult) -> str:
    """Render a replay-verified operator program as concise Korean proof steps."""

    lines: list[str] = []
    used_premises = {
        premise for proof_step in result.proof for premise in proof_step.premises
    }
    used_premises.update(
        outcome.goal.atom
        for outcome in result.goals
        if result.initial_state.contains(outcome.goal.atom)
    )
    eligible_initial = tuple(
        fact
        for fact in result.initial_state.eligible_facts
        if fact.atom in used_premises
    )
    for index, fact in enumerate(eligible_initial, start=1):
        origin = "관찰" if fact.status.value == "observed" else "가정"
        lines.append(f"{index}. [{origin}] {fact.atom}")
    offset = len(lines)
    for proof_step in result.proof:
        description = proof_step.action.operator.rule.description_ko
        bindings = {
            name: str(term) for name, term in proof_step.action.bindings
        }
        try:
            rendered = description.format(
                operator=proof_step.action.operator.name,
                **bindings,
            )
        except (KeyError, IndexError):
            rendered = f"{proof_step.action.operator.name} 정리를 적용했다."
        effects = ", ".join(str(effect) for effect in proof_step.effects)
        lines.append(f"{offset + proof_step.index}. [정리] {rendered} 따라서 {effects}.")
    conclusion = ", ".join(
        f"{outcome.goal}: {'증명됨' if outcome.proven else '미증명'}"
        for outcome in result.goals
    )
    status = "검증 완료" if result.verified else "검증 실패"
    lines.append(f"결론 ({status}): {conclusion}")
    return "\n".join(lines)

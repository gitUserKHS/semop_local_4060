from __future__ import annotations

from .model import EvidenceStatus, FactStatus, SolveResult


def render_proof_ko(result: SolveResult) -> str:
    """Render a replay-verified operator program as concise Korean proof steps."""

    lines: list[str] = []
    dependencies = result.dependencies.all_dependencies
    for index, fact in enumerate(dependencies, start=1):
        origin = {
            FactStatus.OBSERVED: "관찰",
            FactStatus.ASSUMED: "가정",
            FactStatus.DERIVED: "상위 도출",
        }[fact.status]
        evidence = {
            EvidenceStatus.EXTERNAL_VERIFIED: "외부 검증",
            EvidenceStatus.ADAPTER_VERIFIED: "어댑터 검증",
            EvidenceStatus.UNVERIFIED: "증거 미검증",
            EvidenceStatus.ASSUMED: "조건부",
            EvidenceStatus.DERIVED: "도출",
        }[fact.evidence_status]
        lines.append(f"{index}. [{origin}/{evidence}] {fact.atom}")
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
    if result.conditional:
        status += ", 가정 의존"
    elif result.unverified_dependencies:
        status += ", 의미 증거 미검증"
    lines.append(f"결론 ({status}): {conclusion}")
    return "\n".join(lines)

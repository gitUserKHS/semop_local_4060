from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


@dataclass
class UnifiedWorldSolverGuidance:
    primary_goal: str = ""
    hidden_constraints: list[str] = field(default_factory=list)
    priority_queries: list[str] = field(default_factory=list)
    priority_evidence: list[str] = field(default_factory=list)
    operator_focus: list[str] = field(default_factory=list)
    basis_focus: list[str] = field(default_factory=list)
    reflection_strategies: list[str] = field(default_factory=list)
    summary: str = ""

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class UnifiedWorldSolverGuidanceEngine:
    def build(
        self,
        *,
        plan: dict[str, Any] | None,
        reasoning: dict[str, Any] | None = None,
        context: str = "",
        domain: str = "",
        scenario: str = "",
    ) -> UnifiedWorldSolverGuidance:
        plan_payload = dict(plan or {})
        reasoning_payload = dict(reasoning or {})
        primary_goal = str(
            plan_payload.get("primary_goal")
            or reasoning_payload.get("primary_goal")
            or ""
        ).strip()
        hidden_constraints = self._unique_strings(
            list(plan_payload.get("hidden_constraints", []) or [])
            + list(reasoning_payload.get("blockers", []) or [])
            + list(reasoning_payload.get("prerequisites", []) or [])
        )
        operator_focus = self._unique_strings(
            self._labels(plan_payload.get("operator_algebra_targets", []))
            + self._labels(plan_payload.get("retained_operator_candidates", []), key="name")
        )
        basis_focus = self._unique_strings(
            self._flatten_related(plan_payload.get("operator_algebra_targets", []))
            + self._flatten_related(plan_payload.get("operator_functor_targets", []))
            + self._flatten_basis(plan_payload.get("retained_operator_candidates", []))
        )
        priority_evidence = self._unique_strings(
            list(reasoning_payload.get("evidence", []) or [])
            + hidden_constraints[:3]
        )
        priority_queries = self._priority_queries(
            plan_payload=plan_payload,
            primary_goal=primary_goal,
            hidden_constraints=hidden_constraints,
            operator_focus=operator_focus,
            priority_evidence=priority_evidence,
            context=context,
            domain=domain,
            scenario=scenario,
        )
        reflection_strategies = self._reflection_strategies(
            hidden_constraints=hidden_constraints,
            priority_evidence=priority_evidence,
            operator_focus=operator_focus,
            primary_goal=primary_goal,
        )
        summary = self._summary(
            primary_goal=primary_goal,
            hidden_constraints=hidden_constraints,
            operator_focus=operator_focus,
            reflection_strategies=reflection_strategies,
        )
        return UnifiedWorldSolverGuidance(
            primary_goal=primary_goal,
            hidden_constraints=hidden_constraints,
            priority_queries=priority_queries,
            priority_evidence=priority_evidence,
            operator_focus=operator_focus,
            basis_focus=basis_focus,
            reflection_strategies=reflection_strategies,
            summary=summary,
        )

    def case_priority(
        self,
        guidance: UnifiedWorldSolverGuidance | None,
        *,
        query: str,
        context: str = "",
        evidence_terms: Sequence[str] | None = None,
        claim_terms: Sequence[str] | None = None,
    ) -> float:
        if guidance is None:
            return 0.0
        text = " ".join(
            [
                str(query or ""),
                str(context or ""),
                " ".join(str(item) for item in (evidence_terms or []) if str(item).strip()),
                " ".join(str(item) for item in (claim_terms or []) if str(item).strip()),
            ]
        ).strip()
        signals = self._unique_strings(
            [guidance.primary_goal]
            + list(guidance.hidden_constraints)
            + list(guidance.priority_evidence)
            + list(guidance.operator_focus)
        )
        return self.text_priority(text, signals)

    def operator_priority(
        self,
        guidance: UnifiedWorldSolverGuidance | None,
        *,
        name: str,
        basis_signature: Sequence[str],
    ) -> float:
        if guidance is None:
            return 0.0
        focus_names = {self._normalize(item) for item in guidance.operator_focus}
        focus_basis = {self._normalize(item) for item in guidance.basis_focus}
        normalized_name = self._normalize(name)
        normalized_basis = {self._normalize(item) for item in basis_signature if self._normalize(item)}
        name_score = 1.0 if normalized_name and normalized_name in focus_names else 0.0
        basis_score = (
            len(normalized_basis & focus_basis) / float(len(focus_basis) or 1)
            if focus_basis
            else 0.0
        )
        return round((0.65 * name_score) + (0.35 * basis_score), 4)

    def text_priority(self, text: str, signals: Sequence[str]) -> float:
        normalized_text = self._normalize(text)
        if not normalized_text:
            return 0.0
        normalized_signals = [self._normalize(item) for item in signals if self._normalize(item)]
        if not normalized_signals:
            return 0.0
        matches = 0
        for signal in normalized_signals:
            if signal and signal in normalized_text:
                matches += 1
        return round(matches / float(len(normalized_signals)), 4)

    def _priority_queries(
        self,
        *,
        plan_payload: dict[str, Any],
        primary_goal: str,
        hidden_constraints: Sequence[str],
        operator_focus: Sequence[str],
        priority_evidence: Sequence[str],
        context: str,
        domain: str,
        scenario: str,
    ) -> list[str]:
        queries = [str(item) for item in (plan_payload.get("next_learning_queries", []) or []) if str(item).strip()]
        if hidden_constraints:
            queries.append(f"What exact evidence verifies the hidden constraint {hidden_constraints[0]} before action?")
        if operator_focus:
            queries.append(f"When should operator {operator_focus[0]} be preferred over a generic fallback in this environment?")
        if primary_goal:
            queries.append(f"Which action order preserves the primary goal {primary_goal} under the current blockers?")
        if priority_evidence:
            queries.append(f"Which grounded relation should be cited first because of {priority_evidence[0]}?")
        if context.strip():
            queries.append(
                "Which reusable operator from this context should become long-term memory next? "
                + f"Context: {self._snippet(context, 120)}"
            )
        elif domain or scenario:
            queries.append(
                f"What cross-domain operator pattern matters most for {domain or 'general'} / {scenario or 'qa'}?"
            )
        return self._unique_strings(queries)[:8]

    @staticmethod
    def _reflection_strategies(
        *,
        hidden_constraints: Sequence[str],
        priority_evidence: Sequence[str],
        operator_focus: Sequence[str],
        primary_goal: str,
    ) -> list[str]:
        strategies: list[str] = []
        if hidden_constraints:
            strategies.append("ground_hidden_constraints_before_final_claim")
        if priority_evidence:
            strategies.append("cite_local_evidence_before_conclusion")
        if operator_focus:
            strategies.append("preserve_operator_trace_during_refinement")
        if primary_goal:
            strategies.append("keep_primary_goal_visible")
        return UnifiedWorldSolverGuidanceEngine._unique_strings(strategies)

    @staticmethod
    def _summary(
        *,
        primary_goal: str,
        hidden_constraints: Sequence[str],
        operator_focus: Sequence[str],
        reflection_strategies: Sequence[str],
    ) -> str:
        bits: list[str] = []
        if primary_goal:
            bits.append(f"Primary goal stays fixed: {primary_goal}.")
        if hidden_constraints:
            bits.append("Ground hidden constraints first: " + " / ".join(hidden_constraints[:3]) + ".")
        if operator_focus:
            bits.append("Keep operator focus on " + ", ".join(operator_focus[:3]) + ".")
        if reflection_strategies:
            bits.append("Refinement strategies: " + ", ".join(reflection_strategies[:3]) + ".")
        return " ".join(bits).strip()

    @staticmethod
    def _labels(items: Sequence[Any], *, key: str = "label") -> list[str]:
        labels: list[str] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get(key) or "").strip()
            if label:
                labels.append(label)
        return labels

    @staticmethod
    def _flatten_related(items: Sequence[Any]) -> list[str]:
        values: list[str] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            for related in item.get("related_operators", []) or []:
                text = str(related or "").strip()
                if text:
                    values.append(text)
        return values

    @staticmethod
    def _flatten_basis(items: Sequence[Any]) -> list[str]:
        values: list[str] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            for related in item.get("basis_signature", []) or []:
                text = str(related or "").strip()
                if text:
                    values.append(text)
        return values

    @staticmethod
    def _snippet(text: str, limit: int) -> str:
        value = " ".join(str(text or "").split())
        if len(value) <= limit:
            return value
        return value[:limit].rsplit(" ", 1)[0] + "..."

    @staticmethod
    def _normalize(value: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else " " for ch in str(value or "")).strip()

    @staticmethod
    def _unique_strings(items: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            text = " ".join(str(item or "").split()).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(text)
        return ordered

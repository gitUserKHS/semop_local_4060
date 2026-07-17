from __future__ import annotations

from collections.abc import Sequence
import math

from .engine import ActionPolicy, PolicyDecision
from .library import MacroActivationResult, MdlMacroLibrary
from .model import Goal, GroundAction, WorldState
from .registry import KernelRegistry


class PrimitiveMacroPolicy:
    """Use retained programs as a prior over verifier-owned primitive actions."""

    def __init__(
        self,
        registry: KernelRegistry,
        library: MdlMacroLibrary,
        *,
        base_policy: ActionPolicy | None = None,
        macro_bonus: float = 10_000.0,
    ) -> None:
        if not math.isfinite(macro_bonus) or macro_bonus <= 0:
            raise ValueError("macro bonus must be finite and positive")
        self.activation: MacroActivationResult = library.activate(registry)
        self.base_policy = base_policy
        self.macro_bonus = float(macro_bonus)

    @property
    def active_program_count(self) -> int:
        return len(self.activation.programs)

    @property
    def rejected_program_count(self) -> int:
        return len(self.activation.rejected)

    def score_actions(
        self,
        state: WorldState,
        goals: Sequence[Goal],
        actions: Sequence[GroundAction],
    ) -> PolicyDecision:
        action_tuple = tuple(actions)
        base = self._base_decision(state, goals, action_tuple)
        goal_signatures = {
            _atom_type_signature(goal.atom.predicate.name, goal.atom.arguments)
            for goal in goals
        }
        relevant = tuple(
            program
            for program in self.activation.programs
            if goal_signatures.intersection(program.effect_signature)
        )
        macro_scores = [0.0] * len(action_tuple)
        for program in relevant:
            positions: dict[str, int] = {}
            for index, operator_name in enumerate(
                program.primitive_operator_names,
                start=1,
            ):
                positions[operator_name] = index
            length = len(program.primitive_operator_names)
            evidence_bonus = min(program.support, 1_000) / 10_000.0
            compression_bonus = min(program.reduction_ratio, 1.0) / 1_000.0
            for action_index, action in enumerate(action_tuple):
                position = positions.get(action.operator.name)
                if position is None:
                    continue
                score = (
                    self.macro_bonus
                    + position / length
                    + evidence_bonus
                    + compression_bonus
                )
                macro_scores[action_index] = max(macro_scores[action_index], score)

        combined = tuple(
            base_score + macro_score
            for base_score, macro_score in zip(
                base.action_scores,
                macro_scores,
                strict=True,
            )
        )
        macro_top = max(macro_scores, default=0.0)
        unique_macro_top = (
            macro_top > 0
            and sum(score == macro_top for score in macro_scores) == 1
        )
        return PolicyDecision(
            action_scores=combined,
            halt_probability=base.halt_probability,
            state_value=base.state_value,
            action_limit=1 if unique_macro_top else base.action_limit,
        )

    def _base_decision(
        self,
        state: WorldState,
        goals: Sequence[Goal],
        actions: tuple[GroundAction, ...],
    ) -> PolicyDecision:
        if self.base_policy is None:
            return PolicyDecision(tuple(0.0 for _action in actions))
        raw = self.base_policy.score_actions(state, goals, actions)
        if isinstance(raw, PolicyDecision):
            decision = raw
        else:
            decision = PolicyDecision(tuple(float(value) for value in raw))
        if len(decision.action_scores) != len(actions):
            raise ValueError("base policy score count does not match actions")
        if any(not math.isfinite(score) for score in decision.action_scores):
            raise ValueError("base policy returned a non-finite action score")
        return decision


def _atom_type_signature(predicate_name: str, arguments) -> str:
    return (
        f"{predicate_name}("
        + ",".join(argument.type.name for argument in arguments)
        + ")"
    )

from __future__ import annotations

from dataclasses import replace

import pytest

from semop.kernel import (
    KernelRegistry,
    TransitionExecutor,
    TransitionOperator,
    TransitionPlanner,
    TransitionRegistry,
    TransitionWorldState,
    TypeValidationError,
)


def _door_world():
    kernel = KernelRegistry()
    kernel.types.register("Entity")
    kernel.types.register("Agent", "Entity")
    kernel.types.register("Door", "Entity")
    kernel.types.register("Place", "Entity")
    kernel.register_predicate("AT", ("Agent", "Place"))
    kernel.register_predicate("CONNECTS", ("Door", "Place", "Place"))
    kernel.register_predicate("CLOSED", ("Door",))
    kernel.register_predicate("OPEN", ("Door",))

    robot = kernel.symbol("robot", "Agent")
    door = kernel.symbol("door", "Door")
    room_a = kernel.symbol("room_a", "Place")
    room_b = kernel.symbol("room_b", "Place")
    agent_var = kernel.variable("agent", "Agent")
    door_var = kernel.variable("door", "Door")
    source_var = kernel.variable("source", "Place")
    target_var = kernel.variable("target", "Place")

    transitions = TransitionRegistry(kernel)
    transitions.register(
        TransitionOperator(
            name="open_door",
            parameters=(door_var,),
            preconditions=(kernel.atom("CLOSED", door_var),),
            add_effects=(kernel.atom("OPEN", door_var),),
            delete_effects=(kernel.atom("CLOSED", door_var),),
            description_ko="닫힌 문을 열고 CLOSED 상태를 제거한다.",
        )
    )
    transitions.register(
        TransitionOperator(
            name="move_through_door",
            parameters=(agent_var, door_var, source_var, target_var),
            preconditions=(
                kernel.atom("AT", agent_var, source_var),
                kernel.atom("CONNECTS", door_var, source_var, target_var),
                kernel.atom("OPEN", door_var),
            ),
            add_effects=(kernel.atom("AT", agent_var, target_var),),
            delete_effects=(kernel.atom("AT", agent_var, source_var),),
            description_ko="열린 문을 통과해 위치 상태를 바꾼다.",
        )
    )
    initial = TransitionWorldState(
        frozenset(
            {
                kernel.atom("AT", robot, room_a),
                kernel.atom("CONNECTS", door, room_a, room_b),
                kernel.atom("CLOSED", door),
            }
        )
    )
    goal = kernel.atom("AT", robot, room_b)
    return kernel, transitions, initial, goal, door, robot, room_a


def test_transition_planner_applies_add_delete_effects_and_replays() -> None:
    kernel, transitions, initial, goal, door, robot, room_a = _door_world()

    result = TransitionPlanner(transitions).solve(initial, (goal,))

    assert result.success is True
    assert result.verified is True
    assert [step.action.operator.name for step in result.proof] == [
        "open_door",
        "move_through_door",
    ]
    assert kernel.atom("CLOSED", door) in initial.active
    assert kernel.atom("CLOSED", door) not in result.final_state.active
    assert kernel.atom("OPEN", door) in result.final_state.active
    assert kernel.atom("AT", robot, room_a) not in result.final_state.active
    assert goal in result.final_state.active
    assert result.final_state.time == 2


def test_transition_replay_rejects_tampered_digest() -> None:
    _, transitions, initial, goal, *_ = _door_world()
    result = TransitionPlanner(transitions).solve(initial, (goal,))
    tampered = (replace(result.proof[0], after_digest="0" * 64), *result.proof[1:])

    verified, _ = TransitionExecutor(transitions).replay(initial, tampered, (goal,))

    assert verified is False


def test_transition_effect_variables_must_be_bound_by_preconditions() -> None:
    kernel = KernelRegistry()
    kernel.types.register("Entity")
    kernel.register_predicate("KNOWN", ("Entity",))
    kernel.register_predicate("OPEN", ("Entity",))
    item = kernel.variable("item", "Entity")
    other = kernel.variable("other", "Entity")
    transitions = TransitionRegistry(kernel)

    with pytest.raises(TypeValidationError, match="not bound"):
        transitions.register(
            TransitionOperator(
                name="invent_state",
                parameters=(item, other),
                preconditions=(kernel.atom("KNOWN", item),),
                add_effects=(kernel.atom("OPEN", other),),
            )
        )

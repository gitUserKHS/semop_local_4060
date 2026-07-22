from __future__ import annotations

from ..model import (
    AssertionStatus,
    EvidenceStatus,
    Fact,
    FactStatus,
    Goal,
    Rule,
    WorldState,
)
from ..registry import KernelRegistry
from .base import DomainInstance


def create_grid_registry() -> KernelRegistry:
    registry = KernelRegistry()
    entity = registry.types.register("Entity")
    cell = registry.types.register("Cell", entity)
    registry.register_predicate(
        "ADJACENT", (cell, cell), symmetry_groups=((0, 1),)
    )
    registry.register_predicate("OPEN", (cell,))
    registry.register_predicate("REACHABLE", (cell,))
    registry.register_guard(
        "different_cells", lambda binding, _state: binding["x"] != binding["y"]
    )

    x = registry.variable("x", cell)
    y = registry.variable("y", cell)
    registry.register_operator(
        Rule(
            name="move_to_open_neighbor",
            parameters=(x, y),
            preconditions=(
                registry.atom("REACHABLE", x),
                registry.atom("ADJACENT", x, y),
                registry.atom("OPEN", y),
            ),
            effects=(registry.atom("REACHABLE", y),),
            guards=("different_cells",),
            description_ko="도달 가능한 칸 {x}에서 열린 이웃 칸 {y}로 이동했다",
        ),
        family="search",
        tags=("grid", "reachability", "transform"),
    )
    return registry


def parse_grid_problem(text: str) -> DomainInstance:
    rows = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not rows:
        raise ValueError("grid must contain at least one row")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("grid rows must have equal non-zero width")
    allowed = {"S", "G", ".", "#"}
    invalid = sorted({char for row in rows for char in row if char not in allowed})
    if invalid:
        raise ValueError(f"grid contains unsupported cells: {invalid}")
    starts = [
        (row, column)
        for row, values in enumerate(rows)
        for column, value in enumerate(values)
        if value == "S"
    ]
    goals = [
        (row, column)
        for row, values in enumerate(rows)
        for column, value in enumerate(values)
        if value == "G"
    ]
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError("grid must contain exactly one S and one G")

    registry = create_grid_registry()
    cells = {
        (row, column): registry.symbol(f"r{row}c{column}", "Cell")
        for row in range(len(rows))
        for column in range(width)
    }
    facts: list[Fact] = []
    for (row, column), cell in cells.items():
        if rows[row][column] != "#":
            facts.append(
                Fact(
                    registry.atom("OPEN", cell),
                    FactStatus.OBSERVED,
                    "grid",
                    assertion_status=AssertionStatus.EXPLICIT,
                    evidence_status=EvidenceStatus.ADAPTER_VERIFIED,
                )
            )
        for delta_row, delta_column in ((1, 0), (0, 1)):
            neighbor = (row + delta_row, column + delta_column)
            if neighbor in cells:
                facts.append(
                    Fact(
                        registry.atom("ADJACENT", cell, cells[neighbor]),
                        FactStatus.OBSERVED,
                        "grid",
                        assertion_status=AssertionStatus.EXPLICIT,
                        evidence_status=EvidenceStatus.ADAPTER_VERIFIED,
                    )
                )
    start_symbol = cells[starts[0]]
    goal_symbol = cells[goals[0]]
    facts.append(
        Fact(
            registry.atom("REACHABLE", start_symbol),
            FactStatus.ASSUMED,
            "grid_start",
            assertion_status=AssertionStatus.EXPLICIT,
            evidence_status=EvidenceStatus.ASSUMED,
        )
    )
    return DomainInstance(
        registry=registry,
        state=WorldState(tuple(facts)),
        goals=(Goal(registry.atom("REACHABLE", goal_symbol)),),
        domain="grid",
        metadata={
            "height": len(rows),
            "width": width,
            "start": starts[0],
            "goal": goals[0],
        },
    )

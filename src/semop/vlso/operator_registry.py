from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class OperatorType:
    name: str
    axis: str
    description: str


VLSO_OPERATOR_TYPES: List[OperatorType] = [
    OperatorType("OBJECT", "object", "Introduce a concrete entity in the world model."),
    OperatorType("PART_OF", "relation", "Attach a part to a larger object."),
    OperatorType("HAS", "relation", "Mark possession or containment."),
    OperatorType("IS_A", "relation", "Assign a taxonomic type."),
    OperatorType("CONNECTED", "topology", "Describe connectivity or adjacency."),
    OperatorType("INSIDE", "topology", "Describe containment in a spatial region."),
    OperatorType("PARALLEL", "geometry", "Describe geometric parallelism."),
    OperatorType("INTERSECTS", "geometry", "Describe geometric intersection."),
    OperatorType("ACTION", "action", "Introduce a task or manipulation action."),
    OperatorType("REQUIRES", "constraint", "Attach a prerequisite to an action or state."),
    OperatorType("BLOCKS", "constraint", "Mark a blocking condition or obstacle."),
    OperatorType("OPENABLE", "action", "Mark an object or part as openable."),
    OperatorType("TRANSFORM", "transformation", "Describe a state-changing transformation."),
    OperatorType("GOAL", "action", "Introduce a target or optimization goal."),
]

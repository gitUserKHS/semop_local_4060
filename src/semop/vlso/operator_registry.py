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
    OperatorType("PERPENDICULAR", "geometry", "Describe geometric perpendicularity."),
    OperatorType("INTERSECTS", "geometry", "Describe geometric intersection."),
    OperatorType("EQUAL_LENGTH", "geometry", "Describe equal segment lengths."),
    OperatorType("ACTION", "action", "Introduce a task or manipulation action."),
    OperatorType("REQUIRES", "constraint", "Attach a prerequisite to an action or state."),
    OperatorType("BLOCKS", "constraint", "Mark a blocking condition or obstacle."),
    OperatorType("OPENABLE", "action", "Mark an object or part as openable."),
    OperatorType("TRANSFORM", "transformation", "Describe a state-changing transformation."),
    OperatorType("GOAL", "action", "Introduce a target or optimization goal."),
    OperatorType("CONTAINER_BODY_OPERATOR", "structural", "Infer a dominant bounded region that behaves like a container body."),
    OperatorType("ACCESS_PORT_OPERATOR", "structural", "Infer a boundary-attached region that behaves like an access opening or entry port."),
    OperatorType("ACCESS_CONTROL_OPERATOR", "structural", "Infer a part that likely controls or mediates access to an interior region."),
    OperatorType("ATTACHED_GRASP_OPERATOR", "structural", "Infer an attached elongated part that can be used for grasping or carrying."),
    OperatorType("CONTROLLED_ACCESS_OPERATOR", "structural", "Infer that access depends on interacting with a control part on a bounded body."),
    OperatorType("MANIPULABLE_CONTAINER_OPERATOR", "structural", "Infer that a bounded body has both containment and manipulation affordances."),
]

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    GoalDirectedPolicy,
    TypedDomainRequest,
    UnifiedTypedReasoner,
    VisionProblem,
    VisionRelationGoal,
)
from semop.vlso.types import SharedWorldModel, VLSOEntity, VLSORelation


def main() -> None:
    language = (
        "배포하려면 테스트 통과와 승인이 필요하다. "
        "테스트 통과가 충족되었다. 승인이 충족되었다."
    )
    vision_world = SharedWorldModel(
        query="A는 C의 왼쪽인가?",
        entities=[
            VLSOEntity(name, name, "vision", "object", {"verified": True})
            for name in ("A", "B", "C")
        ],
        relations=[
            VLSORelation(
                "A",
                "LEFT_OF",
                "B",
                "vision",
                0.9,
                {"geometry_verified": True},
            ),
            VLSORelation(
                "B",
                "LEFT_OF",
                "C",
                "vision",
                0.9,
                {"geometry_verified": True},
            ),
        ],
    )
    vision = VisionProblem(
        vision_world,
        (VisionRelationGoal("LEFT_OF", "A", "C"),),
    )
    requests = (
        TypedDomainRequest("language", language, "shadow"),
        TypedDomainRequest("math", "(2 + 3) * 4", "shadow"),
        TypedDomainRequest("vision", vision, "shadow"),
    )
    for result in UnifiedTypedReasoner().run_many(
        requests, policy=GoalDirectedPolicy()
    ):
        print(f"\n[{result.domain.value}] success={result.success}")
        print(result.proof_ko)


if __name__ == "__main__":
    main()

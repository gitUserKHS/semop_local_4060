from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    RasterImage,
    RasterVisionProblem,
    TypedDomainRequest,
    UnifiedTypedReasoner,
    VisionRelationGoal,
)


WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def main() -> None:
    image = RasterImage.from_rows(
        (
            [WHITE] * 11,
            [WHITE, RED, RED, WHITE, GREEN, GREEN, WHITE, BLUE, BLUE, WHITE, WHITE],
            [WHITE, RED, RED, WHITE, GREEN, GREEN, WHITE, BLUE, BLUE, WHITE, WHITE],
            [WHITE] * 11,
        ),
        source="in_memory_demo",
    )
    problem = RasterVisionProblem(
        image=image,
        goals=(
            VisionRelationGoal(
                "LEFT_OF",
                "red",
                "blue",
                "빨간 물체는 파란 물체의 왼쪽에 있다",
            ),
        ),
        query="Is the red component left of the blue component?",
    )
    result = UnifiedTypedReasoner().run(
        TypedDomainRequest("vision", problem, "shadow")
    )

    print("detected:")
    for item in result.instance.metadata["detected_objects"]:
        print(f"  {item['id']}: bbox={item['bbox']} rgb={item['mean_rgb']}")
    print()
    print(result.proof_ko)


if __name__ == "__main__":
    main()

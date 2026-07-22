from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    GoalDirectedPolicy,
    RasterImage,
    RasterVisionProblem,
    TypedDomainRequest,
    UnifiedTypedReasoner,
    VisionAreaGoal,
    VisionCountGoal,
    VisionPropertyGoal,
)


WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def main() -> None:
    language = (
        "Every programmer is a person. Every person is mortal. "
        "Ada is a programmer. Prove: Ada is mortal."
    )
    math = "3*(x - 2) = x + 4"
    image = RasterImage.from_rows(
        (
            [WHITE] * 10,
            [WHITE, RED, RED, WHITE, BLUE, BLUE, BLUE, WHITE, WHITE, WHITE],
            [WHITE, RED, RED, WHITE, BLUE, BLUE, BLUE, WHITE, WHITE, WHITE],
            [WHITE] * 10,
        )
    )
    vision = RasterVisionProblem(
        image,
        (
            VisionPropertyGoal("SQUARE", "red"),
            VisionCountGoal("all", 2),
            VisionAreaGoal("blue", "red"),
        ),
    )
    requests = (
        TypedDomainRequest("language", language, "shadow"),
        TypedDomainRequest("math", math, "shadow"),
        TypedDomainRequest("vision", vision, "shadow"),
    )

    results = UnifiedTypedReasoner().run_many(
        requests,
        policy=GoalDirectedPolicy(),
    )
    for result in results:
        print(f"\n[{result.domain.value}] success={result.success}")
        print(result.proof_ko)


if __name__ == "__main__":
    main()

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
    SceneThresholdProblem,
    TypedDomainRequest,
    UnifiedTypedReasoner,
)


WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def demo_image() -> RasterImage:
    return RasterImage.from_rows(
        (
            [WHITE] * 11,
            [WHITE, RED, RED, WHITE, GREEN, GREEN, WHITE, BLUE, BLUE, WHITE, WHITE],
            [WHITE, RED, RED, WHITE, GREEN, GREEN, WHITE, BLUE, BLUE, WHITE, WHITE],
            [WHITE] * 11,
        )
    )


def main() -> None:
    questions = (
        (
            "English",
            "If the number of all objects is greater than 2, "
            "the scene is crowded. Prove: the scene is crowded.",
        ),
        (
            "Korean",
            "모든 물체의 개수가 2보다 크면 장면은 혼잡하다. "
            "증명: 장면은 혼잡하다.",
        ),
        (
            "Conjunction",
            "If the count of red objects is at least 1 and the count of blue "
            "objects is at least 1, the scene is colorful. "
            "Prove: the scene is colorful.",
        ),
    )
    reasoner = UnifiedTypedReasoner()
    policy = GoalDirectedPolicy()
    for label, text in questions:
        result = reasoner.run(
            TypedDomainRequest(
                "composed",
                SceneThresholdProblem(demo_image(), text),
                "typed",
            ),
            policy=policy,
        )
        proof = result.typed_result.proof if result.typed_result else ()
        operators = [step.action.operator.name for step in proof]
        print(f"\n[{label}] success={result.success}, verified={result.verified}")
        print("operators:", " -> ".join(operators))
        print(result.proof_ko)


if __name__ == "__main__":
    main()

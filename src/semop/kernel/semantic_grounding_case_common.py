from __future__ import annotations

from typing import Any, Mapping

from .runtime import DomainKind
from .semantic_benchmark import SemanticBenchmarkCase
from .self_learning import LearningSplit


PROGRAMMATIC_ORACLE_ID = "semop:controlled-semantic-oracle-v1"


def make_programmatic_semantic_case(
    *,
    case_id: str,
    domain: DomainKind,
    payload: Mapping[str, Any],
    expected: bool,
    phenomenon: str,
    split: LearningSplit,
    difficulty: int = 1,
    extra_tags: tuple[str, ...] = (),
) -> SemanticBenchmarkCase:
    return SemanticBenchmarkCase.from_mapping(
        {
            "case_id": case_id,
            "domain": domain.value,
            "payload": dict(payload),
            "expected_solved": expected,
            "phenomenon": phenomenon,
            "rationale": (
                "controlled generator independently specifies whether the exact "
                "typed goal is supported by the raw input"
            ),
            "split": split.value,
            "difficulty": difficulty,
            "author": PROGRAMMATIC_ORACLE_ID,
            "tags": [
                "programmatic_oracle",
                "controlled_semantics",
                "closed_world_support",
                *extra_tags,
            ],
        }
    )


__all__ = ["PROGRAMMATIC_ORACLE_ID", "make_programmatic_semantic_case"]

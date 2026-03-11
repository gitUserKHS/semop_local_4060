from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, List

from .contest_programmer import CompetitiveProgrammingReasoner
from .cp_dataset import CpDslExample, save_cp_dsl_examples


GEOMETRY_TERMS = ('triangle', 'polygon', 'point', 'segment', 'parallel', 'perpendicular', 'distance', 'area', 'circle', 'angle')


@dataclass
class CpGeometryEvalBuildSummary:
    input_path: str
    output_path: str
    input_examples: int
    kept_examples: int

    def model_dump(self) -> dict:
        return asdict(self)


class CpGeometryEvalBuilder:
    def __init__(self, reasoner: CompetitiveProgrammingReasoner | None = None) -> None:
        self.reasoner = reasoner or CompetitiveProgrammingReasoner()

    def build_from_normalized_jsonl(self, input_path: str | Path, output_path: str | Path, limit: int | None = None) -> CpGeometryEvalBuildSummary:
        rows = [json.loads(line) for line in Path(input_path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        examples: List[CpDslExample] = []
        seen: set[str] = set()
        for row in rows:
            statement = str(row.get('statement', '')).strip()
            if not statement or statement in seen:
                continue
            tags = [str(item).lower() for item in row.get('tags', [])]
            haystack = f"{statement} {row.get('solution', '')}".lower()
            if 'geometry' not in tags and not any(term in haystack for term in GEOMETRY_TERMS):
                continue
            result = self.reasoner.solve(statement)
            if result is None:
                continue
            examples.append(CpDslExample(
                statement=statement,
                goal_types=result.goal_types,
                domain_tags=result.domain_tags,
                logical_frames=result.logical_frames,
                dsl_operators=result.dsl_operators,
                target_algorithm=result.category,
                reasoning_sketch=result.approach,
                validation_ok=result.validation_report.get('overall_ok') if result.validation_report else None,
            ))
            seen.add(statement)
            if limit is not None and len(examples) >= limit:
                break
        save_cp_dsl_examples(output_path, examples)
        return CpGeometryEvalBuildSummary(
            input_path=str(input_path),
            output_path=str(output_path),
            input_examples=len(rows),
            kept_examples=len(examples),
        )

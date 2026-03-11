from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .contest_programmer import CompetitiveProgrammingReasoner
from .cp_dataset import CpDslExample, save_cp_dsl_examples


@dataclass
class CpGeometryTemplateSummary:
    output_path: str
    num_examples: int
    statements: list[str]

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class CpGeometryTemplateSplitSummary:
    train_output: str
    val_output: str
    train_examples: int
    val_examples: int

    def model_dump(self) -> dict:
        return asdict(self)


class CpGeometryTemplateGenerator:
    DEFAULT_STATEMENTS = [
        'Given coordinates of three points, compute the area of the triangle they form.',
        'Given two line segments, determine whether they intersect.',
        'Given three points A, B, C, determine whether they make a left turn or a right turn.',
        'Given a point and an axis-aligned rectangle, determine whether the point lies inside the rectangle.',
        'Given the vertices of a polygon, compute its perimeter.',
        'Given the vertices of a convex polygon, compute its signed area.',
        'Given four points of a quadrilateral, determine whether they form a parallelogram.',
        'Given two vectors, determine whether they are perpendicular.',
        'Given the endpoints of two segments, compute the shortest distance between them.',
        'Given a set of points, compute the area of their bounding rectangle.',
    ]

    def __init__(self, reasoner: CompetitiveProgrammingReasoner | None = None) -> None:
        self.reasoner = reasoner or CompetitiveProgrammingReasoner()

    def build_eval_set(self, output_path: str | Path, statements: Iterable[str] | None = None) -> CpGeometryTemplateSummary:
        rows: list[CpDslExample] = []
        selected = list(statements or self.DEFAULT_STATEMENTS)
        for statement in selected:
            rows.append(self._example_from_statement(statement))
        save_cp_dsl_examples(output_path, rows)
        return CpGeometryTemplateSummary(
            output_path=str(output_path),
            num_examples=len(rows),
            statements=selected,
        )

    def build_train_val_split(self, train_output: str | Path, val_output: str | Path, train_ratio: float = 0.8, statements: Iterable[str] | None = None) -> CpGeometryTemplateSplitSummary:
        selected = list(statements or self.DEFAULT_STATEMENTS)
        cutoff = max(1, min(len(selected) - 1, int(round(len(selected) * train_ratio))))
        train_rows = [self._example_from_statement(statement) for statement in selected[:cutoff]]
        val_rows = [self._example_from_statement(statement) for statement in selected[cutoff:]]
        save_cp_dsl_examples(train_output, train_rows)
        save_cp_dsl_examples(val_output, val_rows)
        return CpGeometryTemplateSplitSummary(
            train_output=str(train_output),
            val_output=str(val_output),
            train_examples=len(train_rows),
            val_examples=len(val_rows),
        )

    def _example_from_statement(self, statement: str) -> CpDslExample:
        result = self.reasoner.solve(statement)
        return CpDslExample(
            statement=statement,
            goal_types=result.goal_types,
            domain_tags=result.domain_tags,
            logical_frames=result.logical_frames,
            dsl_operators=result.dsl_operators,
            target_algorithm=result.category,
            reasoning_sketch=result.approach,
            validation_ok=result.validation_report.get('overall_ok') if result.validation_report else None,
        )

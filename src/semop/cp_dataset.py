from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, List


@dataclass
class CpDslExample:
    statement: str
    goal_types: List[str]
    domain_tags: List[str]
    logical_frames: List[str]
    dsl_operators: List[str]
    target_algorithm: str
    reasoning_sketch: str
    validation_ok: bool | None = None

    def model_dump(self) -> dict:
        return asdict(self)


def save_cp_dsl_examples(path: str | Path, examples: Iterable[CpDslExample]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.model_dump(), ensure_ascii=False) + "\n")


def load_cp_dsl_examples(path: str | Path) -> List[CpDslExample]:
    records: List[CpDslExample] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(CpDslExample(**json.loads(line)))
    return records

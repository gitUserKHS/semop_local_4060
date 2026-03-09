from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class VisualConceptTarget:
    subject_id: str = ""
    positive_labels: List[str] = field(default_factory=list)
    negative_labels: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class VisualConceptExample:
    image_path: str
    targets: List[VisualConceptTarget] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class VisualConceptDataset:
    def load_jsonl(self, path: str | Path) -> List[VisualConceptExample]:
        examples: List[VisualConceptExample] = []
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                targets = []
                for target in row.get("targets", []):
                    targets.append(
                        VisualConceptTarget(
                            subject_id=str(target.get("subject_id", "")),
                            positive_labels=[str(item) for item in target.get("positive_labels", [])],
                            negative_labels=[str(item) for item in target.get("negative_labels", [])],
                            notes=str(target.get("notes", "")),
                        )
                    )
                if not targets and (row.get("positive_labels") or row.get("negative_labels")):
                    targets.append(
                        VisualConceptTarget(
                            subject_id=str(row.get("subject_id", "")),
                            positive_labels=[str(item) for item in row.get("positive_labels", [])],
                            negative_labels=[str(item) for item in row.get("negative_labels", [])],
                            notes=str(row.get("notes", "")),
                        )
                    )
                examples.append(
                    VisualConceptExample(
                        image_path=str(row["image_path"]),
                        targets=targets,
                        metadata=dict(row.get("metadata", {})),
                    )
                )
        return examples

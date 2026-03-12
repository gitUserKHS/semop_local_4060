from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .distillation import TeacherTraceRecord, TeacherTraceExporter


@dataclass
class OperatorCurriculumPhase:
    name: str
    tasks: list[str]
    description: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorLearningBundleSummary:
    train_trace_jsonl: str
    val_trace_jsonl: str
    train_sft_jsonl: str
    val_sft_jsonl: str
    curriculum_plan_path: str
    task_counts: dict[str, int]
    phase_counts: dict[str, int]
    num_train: int
    num_val: int

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class OperatorCurriculumBuilder:
    DEFAULT_PHASES = [
        OperatorCurriculumPhase(
            name='foundation',
            tasks=['hidden_premise', 'cp_structuring', 'cp_hidden_constraints'],
            description='Hidden goals, constraints, and problem-structuring traces.',
        ),
        OperatorCurriculumPhase(
            name='grounding',
            tasks=['vlso_grounded_qa', 'vlso_real_image'],
            description='Grounded visual and cross-modal operator traces.',
        ),
        OperatorCurriculumPhase(
            name='operatorization',
            tasks=['operator_proposal', 'operator_self_evolution'],
            description='Higher-operator proposal, decomposition, and evolution traces.',
        ),
    ]

    def __init__(self, phases: list[OperatorCurriculumPhase] | None = None) -> None:
        self.phases = phases or list(self.DEFAULT_PHASES)
        self._task_to_phase = {task: phase.name for phase in self.phases for task in phase.tasks}

    def build_bundle(
        self,
        teacher_trace_path: str | Path,
        workspace: str | Path,
        val_ratio: float = 0.15,
        max_per_task: int = 0,
    ) -> OperatorLearningBundleSummary:
        rows = self.load_traces(teacher_trace_path)
        by_task: dict[str, list[TeacherTraceRecord]] = {}
        for row in rows:
            by_task.setdefault(row.task, []).append(row)
        selected: list[TeacherTraceRecord] = []
        task_counts: dict[str, int] = {}
        phase_counts: dict[str, int] = {}
        for task, task_rows in by_task.items():
            if max_per_task > 0:
                task_rows = task_rows[:max_per_task]
            selected.extend(task_rows)
            task_counts[task] = len(task_rows)
            phase = self._task_to_phase.get(task, 'unassigned')
            phase_counts[phase] = phase_counts.get(phase, 0) + len(task_rows)
        selected.sort(key=lambda row: (self._task_to_phase.get(row.task, 'zzz'), row.task, row.input_text))
        train_rows: list[TeacherTraceRecord] = []
        val_rows: list[TeacherTraceRecord] = []
        per_task_selected: dict[str, list[TeacherTraceRecord]] = {}
        for row in selected:
            per_task_selected.setdefault(row.task, []).append(row)
        for task_rows in per_task_selected.values():
            split = int(round(len(task_rows) * (1.0 - val_ratio)))
            split = min(max(split, 1), len(task_rows))
            if split == len(task_rows) and len(task_rows) > 1:
                split -= 1
            train_rows.extend(task_rows[:split])
            val_rows.extend(task_rows[split:])
        workspace_path = Path(workspace)
        workspace_path.mkdir(parents=True, exist_ok=True)
        train_trace = workspace_path / 'operator_train_traces.jsonl'
        val_trace = workspace_path / 'operator_val_traces.jsonl'
        train_sft = workspace_path / 'operator_train_sft.jsonl'
        val_sft = workspace_path / 'operator_val_sft.jsonl'
        curriculum_plan = workspace_path / 'operator_curriculum_plan.json'
        TeacherTraceExporter.save_jsonl(train_trace, train_rows)
        TeacherTraceExporter.save_jsonl(val_trace, val_rows)
        TeacherTraceExporter.save_jsonl(train_sft, TeacherTraceExporter.to_sft_records(train_rows))
        TeacherTraceExporter.save_jsonl(val_sft, TeacherTraceExporter.to_sft_records(val_rows))
        curriculum_plan.write_text(json.dumps({
            'phases': [phase.model_dump() for phase in self.phases],
            'task_counts': task_counts,
            'phase_counts': phase_counts,
            'num_train': len(train_rows),
            'num_val': len(val_rows),
            'source': str(teacher_trace_path),
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        return OperatorLearningBundleSummary(
            train_trace_jsonl=str(train_trace),
            val_trace_jsonl=str(val_trace),
            train_sft_jsonl=str(train_sft),
            val_sft_jsonl=str(val_sft),
            curriculum_plan_path=str(curriculum_plan),
            task_counts=task_counts,
            phase_counts=phase_counts,
            num_train=len(train_rows),
            num_val=len(val_rows),
        )

    @staticmethod
    def load_traces(path: str | Path) -> list[TeacherTraceRecord]:
        rows: list[TeacherTraceRecord] = []
        with Path(path).open('r', encoding='utf-8-sig') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(TeacherTraceRecord(**json.loads(line)))
        return rows

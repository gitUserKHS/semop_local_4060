from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .cp_parser_eval import CpLearnedParser, CpParserEvaluator
from .cp_training import CpParserTrainConfig, CpParserTrainingScaffold, CpTrainingBundleBuilder
from .cp_dataset import load_cp_dsl_examples


@dataclass
class CpLoraExperimentConfig:
    workspace: str
    model_name_or_path: str
    dataset_paths: list[str]
    episode_store_path: str | None = None
    eval_dataset_paths: list[str] | None = None
    train_ratio: float = 0.9
    allow_failed_episodes: bool = False
    execute_train: bool = False
    dry_run_train: bool = True
    local_files_only: bool = False
    use_lora: bool = True
    max_steps: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    max_length: int = 512
    warmup_steps: int = 10
    seed: int = 42
    resume_from_checkpoint: str | None = None
    save_steps: int = 25
    save_total_limit: int = 2

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CpLoraExperimentSummary:
    config: dict[str, Any]
    train_jsonl: str
    val_jsonl: str
    train_sft_jsonl: str
    val_sft_jsonl: str
    heuristic_eval: dict[str, Any]
    training_summary: dict[str, Any] | None = None
    compare_eval: dict[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class CpLoraExperimentRunner:
    def __init__(self, bundle_builder: CpTrainingBundleBuilder | None = None, scaffold: CpParserTrainingScaffold | None = None, evaluator: CpParserEvaluator | None = None) -> None:
        self.bundle_builder = bundle_builder or CpTrainingBundleBuilder()
        self.scaffold = scaffold or CpParserTrainingScaffold()
        self.evaluator = evaluator or CpParserEvaluator()

    def run(self, config: CpLoraExperimentConfig) -> CpLoraExperimentSummary:
        workspace = Path(config.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        bundle = self.bundle_builder.build(
            dataset_paths=config.dataset_paths,
            episode_store_path=config.episode_store_path,
            allow_failed_episodes=config.allow_failed_episodes,
            train_ratio=config.train_ratio,
        )
        train_jsonl = workspace / 'train.jsonl'
        val_jsonl = workspace / 'val.jsonl'
        train_sft = workspace / 'train_sft.jsonl'
        val_sft = workspace / 'val_sft.jsonl'
        self.bundle_builder.save_bundle(
            bundle,
            train_output=train_jsonl,
            val_output=val_jsonl,
            train_sft_output=train_sft,
            val_sft_output=val_sft,
        )
        val_examples = load_cp_dsl_examples(val_jsonl)
        eval_examples = list(val_examples)
        if config.eval_dataset_paths:
            eval_examples = []
            for dataset_path in config.eval_dataset_paths:
                eval_examples.extend(load_cp_dsl_examples(dataset_path))
        heuristic_eval = self.evaluator.evaluate_examples(eval_examples, predictor='heuristic').model_dump()
        training_summary = None
        compare_eval = None
        if config.execute_train:
            training_output = workspace / 'training_run'
            train_config = CpParserTrainConfig(
                model_name_or_path=config.model_name_or_path,
                output_dir=str(training_output),
                train_jsonl=str(train_jsonl),
                max_steps=config.max_steps,
                batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                max_length=config.max_length,
                warmup_steps=config.warmup_steps,
                seed=config.seed,
                dry_run=config.dry_run_train,
                local_files_only=config.local_files_only,
                allow_failed_episodes=config.allow_failed_episodes,
                use_lora=config.use_lora,
                resume_from_checkpoint=config.resume_from_checkpoint,
                save_steps=config.save_steps,
                save_total_limit=config.save_total_limit,
            )
            training_summary = self.scaffold.run(train_config)
            model_dir = training_summary.get('final_model_dir')
            if model_dir and Path(model_dir).exists():
                compare_eval = self.evaluator.compare_examples(
                    eval_examples,
                    model=CpLearnedParser(model_dir, local_files_only=config.local_files_only),
                ).model_dump()
        summary = CpLoraExperimentSummary(
            config=config.model_dump(),
            train_jsonl=str(train_jsonl),
            val_jsonl=str(val_jsonl),
            train_sft_jsonl=str(train_sft),
            val_sft_jsonl=str(val_sft),
            heuristic_eval=heuristic_eval,
            training_summary=training_summary,
            compare_eval=compare_eval,
        )
        (workspace / 'experiment_summary.json').write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

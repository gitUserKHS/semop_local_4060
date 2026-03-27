from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
from typing import Iterable, List

from .contest_programmer import CompetitiveProgrammingReasoner
from .cp_dataset import CpDslExample, load_cp_dsl_examples, save_cp_dsl_examples
from .cp_episode_store import CpEpisodeStore
from .hardware_profiles import detect_local_hardware


@dataclass
class CpSftRecord:
    prompt: str
    completion: str

    def model_dump(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion}


@dataclass
class CpParserTrainConfig:
    model_name_or_path: str
    output_dir: str
    train_jsonl: str | None = None
    episode_store_path: str | None = None
    max_steps: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    max_length: int = 512
    warmup_steps: int = 10
    seed: int = 42
    dry_run: bool = False
    local_files_only: bool = False
    allow_failed_episodes: bool = False
    use_lora: bool = False
    use_qlora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] | None = None
    resume_from_checkpoint: str | None = None
    save_steps: int = 25
    save_total_limit: int = 2
    hardware_profile: str = 'auto'
    auto_configure_for_local_gpu: bool = True

    def model_dump(self) -> dict:
        return asdict(self)


class CpDslDatasetBuilder:
    def __init__(self, reasoner: CompetitiveProgrammingReasoner | None = None) -> None:
        self.reasoner = reasoner or CompetitiveProgrammingReasoner()

    def build_from_statements(self, statements: Iterable[str]) -> List[CpDslExample]:
        examples: List[CpDslExample] = []
        for statement in statements:
            text = statement.strip()
            if not text:
                continue
            result = self.reasoner.solve(text)
            if result is None:
                continue
            validation_report = result.validation_report or {}
            examples.append(
                CpDslExample(
                    statement=text,
                    goal_types=result.goal_types,
                    domain_tags=result.domain_tags,
                    logical_frames=result.logical_frames,
                    dsl_operators=result.dsl_operators,
                    target_algorithm=result.category,
                    reasoning_sketch=result.approach,
                    validation_ok=validation_report.get("overall_ok"),
                )
            )
        return examples

    @staticmethod
    def save(path: str | Path, examples: Iterable[CpDslExample]) -> None:
        save_cp_dsl_examples(path, examples)


class CpTrainingPlanner:
    @staticmethod
    def build_sft_records(examples: Iterable[CpDslExample]) -> List[CpSftRecord]:
        records: List[CpSftRecord] = []
        for example in examples:
            prompt = (
                "Read the contest problem and output structured reasoning fields: "
                "goal_types, domain_tags, logical_frames, dsl_operators, target_algorithm, reasoning_sketch.\n\n"
                f"Problem:\n{example.statement}"
            )
            completion_payload = {
                "goal_types": example.goal_types,
                "domain_tags": example.domain_tags,
                "logical_frames": example.logical_frames,
                "dsl_operators": example.dsl_operators,
                "target_algorithm": example.target_algorithm,
                "reasoning_sketch": example.reasoning_sketch,
            }
            records.append(CpSftRecord(prompt=prompt, completion=json.dumps(completion_payload, ensure_ascii=False)))
        return records

    @staticmethod
    def save_sft_records(path: str | Path, records: Iterable[CpSftRecord]) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")

    @staticmethod
    def summarize_examples(examples: Iterable[CpDslExample]) -> dict:
        examples = list(examples)
        algorithms: dict[str, int] = {}
        frames: dict[str, int] = {}
        operators: dict[str, int] = {}
        for example in examples:
            algorithms[example.target_algorithm] = algorithms.get(example.target_algorithm, 0) + 1
            for frame in example.logical_frames:
                frames[frame] = frames.get(frame, 0) + 1
            for operator in example.dsl_operators:
                operators[operator] = operators.get(operator, 0) + 1
        return {
            "num_examples": len(examples),
            "algorithm_counts": algorithms,
            "frame_counts": frames,
            "operator_counts": operators,
        }


def load_cp_sft_records(path: str | Path) -> List[CpSftRecord]:
    records: List[CpSftRecord] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(CpSftRecord(**payload))
    return records


class CpParserTrainingScaffold:
    @staticmethod
    def _resolve_local_model_source(model_name_or_path: str) -> str:
        source_path = Path(model_name_or_path)
        if source_path.exists():
            return str(source_path)
        if "/" not in model_name_or_path:
            return model_name_or_path
        owner, name = model_name_or_path.split("/", 1)
        snapshot_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{owner}--{name}" / "snapshots"
        if not snapshot_root.exists():
            return model_name_or_path
        snapshots = [candidate for candidate in snapshot_root.iterdir() if candidate.is_dir()]
        if not snapshots:
            return model_name_or_path
        snapshots.sort(key=lambda candidate: candidate.stat().st_mtime, reverse=True)
        return str(snapshots[0])

    @staticmethod
    def find_latest_checkpoint(output_dir: str | Path) -> str | None:
        base = Path(output_dir)
        if not base.exists():
            return None
        checkpoints = []
        for child in base.iterdir():
            if not child.is_dir() or not child.name.startswith("checkpoint-"):
                continue
            try:
                step = int(child.name.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            checkpoints.append((step, child))
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda item: item[0])
        return str(checkpoints[-1][1])

    def build_examples_from_episode_store(
        self,
        episode_store_path: str | Path,
        allow_failed_episodes: bool = False,
    ) -> List[CpDslExample]:
        store = CpEpisodeStore(episode_store_path)
        examples: List[CpDslExample] = []
        for record in store.iter_records():
            validation_ok = bool(record.validation_report.get("overall_ok"))
            if not allow_failed_episodes and not (record.compile_ok and validation_ok):
                continue
            examples.append(
                CpDslExample(
                    statement=record.statement,
                    goal_types=record.goal_types,
                    domain_tags=record.domain_tags,
                    logical_frames=record.logical_frames,
                    dsl_operators=record.dsl_operators,
                    target_algorithm=record.category,
                    reasoning_sketch=record.approach,
                    validation_ok=validation_ok,
                )
            )
        return examples

    def prepare_records(
        self,
        train_jsonl: str | None = None,
        episode_store_path: str | None = None,
        allow_failed_episodes: bool = False,
    ) -> List[CpSftRecord]:
        if train_jsonl:
            examples = load_cp_dsl_examples(train_jsonl)
        elif episode_store_path:
            examples = self.build_examples_from_episode_store(
                episode_store_path,
                allow_failed_episodes=allow_failed_episodes,
            )
        else:
            raise ValueError("Either train_jsonl or episode_store_path must be provided.")
        return CpTrainingPlanner.build_sft_records(examples)

    @staticmethod
    def summarize_records(records: Iterable[CpSftRecord]) -> dict:
        records = list(records)
        lengths = [len((record.prompt + "\n" + record.completion).split()) for record in records]
        return {
            "num_records": len(records),
            "avg_word_length": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
            "max_word_length": max(lengths) if lengths else 0,
        }

    @staticmethod
    def _device_summary(preferred_profile: str = 'auto') -> dict:
        profile = detect_local_hardware(preferred_profile)
        summary = profile.model_dump()
        summary['torch_available'] = True
        return summary

    @staticmethod
    def _apply_hardware_profile(config: CpParserTrainConfig):
        profile = detect_local_hardware(config.hardware_profile)
        if not config.auto_configure_for_local_gpu:
            return config, profile
        effective = replace(config)
        if profile.detected_profile in {'rtx_4060_8gb', 'cuda_low_vram'}:
            effective.batch_size = 1
            effective.gradient_accumulation_steps = max(int(effective.gradient_accumulation_steps), int(profile.recommended_gradient_accumulation))
            effective.max_length = min(int(effective.max_length), 512)
            effective.use_lora = True
            effective.use_qlora = bool(profile.recommended_use_qlora or effective.use_qlora)
            effective.save_total_limit = min(max(1, int(effective.save_total_limit)), 2)
        elif profile.cuda_available:
            effective.use_lora = bool(effective.use_lora or profile.recommended_use_lora)
        return effective, profile

    def run(self, config: CpParserTrainConfig) -> dict:
        effective_config, hardware_profile = self._apply_hardware_profile(config)
        records = self.prepare_records(
            train_jsonl=effective_config.train_jsonl,
            episode_store_path=effective_config.episode_store_path,
            allow_failed_episodes=effective_config.allow_failed_episodes,
        )
        output_dir = Path(effective_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_path = output_dir / "train_sft.jsonl"
        CpTrainingPlanner.save_sft_records(train_path, records)

        summary = {
            "config": config.model_dump(),
            "effective_config": effective_config.model_dump(),
            "record_summary": self.summarize_records(records),
            "device_summary": self._device_summary(effective_config.hardware_profile),
            "hardware_profile": hardware_profile.model_dump(),
            "operator_algebra_mode": hardware_profile.operator_algebra_mode,
            "train_sft_path": str(train_path),
            "lora_enabled": effective_config.use_lora,
            "resume_from_checkpoint": effective_config.resume_from_checkpoint,
        }
        if effective_config.dry_run:
            plan_path = output_dir / "training_plan.json"
            plan_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["mode"] = "dry_run"
            summary["plan_path"] = str(plan_path)
            return summary

        try:
            import torch
            from torch.utils.data import Dataset
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
        except Exception as exc:
            raise RuntimeError(
                "transformers/torch training dependencies are not available. Run with --dry-run or install the required packages."
            ) from exc

        class SftDataset(Dataset):
            def __init__(self, items: List[CpSftRecord], tokenizer: object, max_length: int) -> None:
                self.items = items
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self) -> int:
                return len(self.items)

            def __getitem__(self, index: int) -> dict:
                record = self.items[index]
                text = f"{record.prompt}\n\n### Answer\n{record.completion}"
                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"][0]
                attention_mask = encoded["attention_mask"][0]
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids.clone(),
                }

        model_source = self._resolve_local_model_source(effective_config.model_name_or_path) if effective_config.local_files_only else effective_config.model_name_or_path
        summary["resolved_model_source"] = model_source
        if effective_config.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=effective_config.local_files_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_kwargs = {'local_files_only': effective_config.local_files_only, 'low_cpu_mem_usage': True}
        if effective_config.use_qlora and hardware_profile.bitsandbytes_available and hardware_profile.cuda_available:
            model_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
            model_kwargs['device_map'] = 'auto'
        elif hardware_profile.cuda_available:
            model_kwargs['torch_dtype'] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            **model_kwargs,
        )
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        if effective_config.use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except Exception as exc:
                raise RuntimeError(
                    "LoRA was requested but peft is not available. Install `peft` or run without --use-lora."
                ) from exc
            target_modules = effective_config.lora_target_modules or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=effective_config.lora_rank,
                lora_alpha=effective_config.lora_alpha,
                lora_dropout=effective_config.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            summary["lora_target_modules"] = target_modules
        dataset = SftDataset(records, tokenizer, effective_config.max_length)
        training_kwargs = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": effective_config.batch_size,
            "gradient_accumulation_steps": effective_config.gradient_accumulation_steps,
            "learning_rate": effective_config.learning_rate,
            "max_steps": effective_config.max_steps,
            "warmup_steps": effective_config.warmup_steps,
            "logging_steps": max(1, min(10, effective_config.max_steps)),
            "save_steps": max(1, effective_config.save_steps),
            "save_total_limit": max(1, effective_config.save_total_limit),
            "report_to": [],
            "fp16": bool(hardware_profile.cuda_available),
            "bf16": False,
            "seed": effective_config.seed,
            "remove_unused_columns": False,
            "optim": 'paged_adamw_8bit' if effective_config.use_qlora and hardware_profile.bitsandbytes_available else 'adamw_torch',
        }
        try:
            args = TrainingArguments(**training_kwargs, overwrite_output_dir=False)
        except TypeError:
            args = TrainingArguments(**training_kwargs)
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train(resume_from_checkpoint=effective_config.resume_from_checkpoint or None)
        trainer.save_model(str(output_dir / "final_model"))
        tokenizer.save_pretrained(str(output_dir / "final_model"))
        latest_checkpoint = self.find_latest_checkpoint(output_dir)
        checkpoint_dirs = []
        for path_value in output_dir.glob("checkpoint-*"):
            if not path_value.is_dir():
                continue
            try:
                _ = int(path_value.name.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            checkpoint_dirs.append(str(path_value))
        checkpoint_dirs.sort(key=lambda value: int(Path(value).name.split("-", 1)[1]))
        summary["mode"] = "train"
        summary["final_model_dir"] = str(output_dir / "final_model")
        summary["latest_checkpoint"] = latest_checkpoint
        summary["checkpoint_dirs"] = checkpoint_dirs
        return summary


@dataclass
class CpTrainingBundle:
    train_examples: List[CpDslExample]
    val_examples: List[CpDslExample]


class CpTrainingBundleBuilder:
    def __init__(self, scaffold: CpParserTrainingScaffold | None = None) -> None:
        self.scaffold = scaffold or CpParserTrainingScaffold()

    def build(
        self,
        dataset_paths: Iterable[str | Path] = (),
        episode_store_path: str | Path | None = None,
        allow_failed_episodes: bool = False,
        train_ratio: float = 0.9,
    ) -> CpTrainingBundle:
        examples: List[CpDslExample] = []
        for dataset_path in dataset_paths:
            examples.extend(load_cp_dsl_examples(dataset_path))
        if episode_store_path:
            examples.extend(
                self.scaffold.build_examples_from_episode_store(
                    episode_store_path,
                    allow_failed_episodes=allow_failed_episodes,
                )
            )
        deduped: dict[str, CpDslExample] = {}
        for example in examples:
            if example.statement not in deduped:
                deduped[example.statement] = example
        ordered = list(deduped.values())
        ordered.sort(key=lambda item: item.statement)
        if not ordered:
            return CpTrainingBundle(train_examples=[], val_examples=[])
        split_index = int(len(ordered) * train_ratio)
        split_index = min(max(split_index, 1), len(ordered))
        if split_index == len(ordered) and len(ordered) > 1:
            split_index -= 1
        return CpTrainingBundle(
            train_examples=ordered[:split_index],
            val_examples=ordered[split_index:],
        )

    def save_bundle(
        self,
        bundle: CpTrainingBundle,
        train_output: str | Path,
        val_output: str | Path,
        train_sft_output: str | Path | None = None,
        val_sft_output: str | Path | None = None,
    ) -> None:
        save_cp_dsl_examples(train_output, bundle.train_examples)
        save_cp_dsl_examples(val_output, bundle.val_examples)
        if train_sft_output is not None:
            CpTrainingPlanner.save_sft_records(train_sft_output, CpTrainingPlanner.build_sft_records(bundle.train_examples))
        if val_sft_output is not None:
            CpTrainingPlanner.save_sft_records(val_sft_output, CpTrainingPlanner.build_sft_records(bundle.val_examples))

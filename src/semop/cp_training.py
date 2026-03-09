from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, List

from .contest_programmer import CompetitiveProgrammingReasoner
from .cp_dataset import CpDslExample, load_cp_dsl_examples, save_cp_dsl_examples
from .cp_episode_store import CpEpisodeStore


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
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] | None = None

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
    def _device_summary() -> dict:
        try:
            import torch
        except Exception:
            return {"torch_available": False, "device": "cpu"}
        cuda_available = bool(torch.cuda.is_available())
        summary = {
            "torch_available": True,
            "device": "cuda" if cuda_available else "cpu",
            "cuda_available": cuda_available,
        }
        if cuda_available:
            summary["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                props = torch.cuda.get_device_properties(0)
                summary["vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
            except Exception:
                pass
        return summary

    def run(self, config: CpParserTrainConfig) -> dict:
        records = self.prepare_records(
            train_jsonl=config.train_jsonl,
            episode_store_path=config.episode_store_path,
            allow_failed_episodes=config.allow_failed_episodes,
        )
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_path = output_dir / "train_sft.jsonl"
        CpTrainingPlanner.save_sft_records(train_path, records)

        summary = {
            "config": config.model_dump(),
            "record_summary": self.summarize_records(records),
            "device_summary": self._device_summary(),
            "train_sft_path": str(train_path),
            "lora_enabled": config.use_lora,
        }
        if config.dry_run:
            plan_path = output_dir / "training_plan.json"
            plan_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["mode"] = "dry_run"
            summary["plan_path"] = str(plan_path)
            return summary

        try:
            import torch
            from torch.utils.data import Dataset
            from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
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

        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, local_files_only=config.local_files_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            local_files_only=config.local_files_only,
        )
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        if config.use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except Exception as exc:
                raise RuntimeError(
                    "LoRA was requested but peft is not available. Install `peft` or run without --use-lora."
                ) from exc
            target_modules = config.lora_target_modules or [
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
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            summary["lora_target_modules"] = target_modules
        dataset = SftDataset(records, tokenizer, config.max_length)
        use_cuda = bool(torch.cuda.is_available())
        args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            max_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
            logging_steps=max(1, min(10, config.max_steps)),
            save_steps=max(1, config.max_steps),
            save_total_limit=1,
            report_to=[],
            fp16=use_cuda,
            bf16=False,
            seed=config.seed,
            remove_unused_columns=False,
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train()
        trainer.save_model(str(output_dir / "final_model"))
        tokenizer.save_pretrained(str(output_dir / "final_model"))
        summary["mode"] = "train"
        summary["final_model_dir"] = str(output_dir / "final_model")
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

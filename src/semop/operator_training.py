from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

from .distillation import DistillationSftRecord


@dataclass
class OperatorTrainConfig:
    model_name_or_path: str
    output_dir: str
    train_jsonl: str
    max_steps: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    max_length: int = 768
    warmup_steps: int = 10
    seed: int = 42
    dry_run: bool = False
    local_files_only: bool = False
    use_lora: bool = True
    use_qlora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    resume_from_checkpoint: str | None = None
    save_steps: int = 25
    save_total_limit: int = 2

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class OperatorTrainingScaffold:
    @staticmethod
    def _config_path(output_dir: str | Path) -> Path:
        return Path(output_dir) / 'operator_training_run_config.json'

    @staticmethod
    def _resume_signature(config: OperatorTrainConfig, model_source: str) -> dict[str, Any]:
        return {
            'model_source': model_source,
            'train_jsonl': str(config.train_jsonl),
            'use_lora': bool(config.use_lora),
            'use_qlora': bool(config.use_qlora),
            'lora_rank': int(config.lora_rank),
            'lora_alpha': int(config.lora_alpha),
            'lora_dropout': float(config.lora_dropout),
            'batch_size': int(config.batch_size),
            'gradient_accumulation_steps': int(config.gradient_accumulation_steps),
        }

    @classmethod
    def _load_saved_signature(cls, output_dir: str | Path) -> dict[str, Any] | None:
        config_path = cls._config_path(output_dir)
        if not config_path.exists():
            return None
        try:
            payload = json.loads(config_path.read_text(encoding='utf-8'))
        except Exception:
            return None
        signature = payload.get('resume_signature')
        return signature if isinstance(signature, dict) else None

    @classmethod
    def resolve_resume_checkpoint(
        cls,
        output_dir: str | Path,
        requested_checkpoint: str | None,
        config: OperatorTrainConfig,
        model_source: str,
    ) -> tuple[str | None, dict[str, Any]]:
        info: dict[str, Any] = {}
        effective_resume = requested_checkpoint
        if not effective_resume:
            return None, info
        saved_signature = cls._load_saved_signature(output_dir)
        current_signature = cls._resume_signature(config, model_source)
        if saved_signature is not None and saved_signature != current_signature:
            info['resume_skipped_reason'] = 'checkpoint_config_mismatch'
            info['resume_skipped_signature'] = {'saved': saved_signature, 'current': current_signature}
            effective_resume = None
        return effective_resume, info

    @classmethod
    def _save_run_metadata(cls, output_dir: str | Path, config: OperatorTrainConfig, model_source: str) -> None:
        config_path = cls._config_path(output_dir)
        payload = {
            'config': config.model_dump(),
            'resume_signature': cls._resume_signature(config, model_source),
        }
        config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    @staticmethod
    def find_latest_checkpoint(output_dir: str | Path) -> str | None:
        base = Path(output_dir)
        if not base.exists():
            return None
        checkpoints: list[tuple[int, Path]] = []
        for child in base.iterdir():
            if not child.is_dir() or not child.name.startswith('checkpoint-'):
                continue
            try:
                step = int(child.name.split('-', 1)[1])
            except (IndexError, ValueError):
                continue
            checkpoints.append((step, child))
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda item: item[0])
        return str(checkpoints[-1][1])

    @staticmethod
    def load_sft_records(path: str | Path) -> list[DistillationSftRecord]:
        rows: list[DistillationSftRecord] = []
        with Path(path).open('r', encoding='utf-8-sig') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(DistillationSftRecord(**json.loads(line)))
        return rows

    @staticmethod
    def _resolve_local_model_source(model_name_or_path: str) -> str:
        source_path = Path(model_name_or_path)
        if source_path.exists():
            return str(source_path)
        if '/' not in model_name_or_path:
            return model_name_or_path
        owner, name = model_name_or_path.split('/', 1)
        cache_root = Path.home() / '.cache' / 'huggingface' / 'hub' / f'models--{owner}--{name}' / 'snapshots'
        if not cache_root.exists():
            return model_name_or_path
        snapshots = [item for item in cache_root.iterdir() if item.is_dir()]
        if not snapshots:
            return model_name_or_path
        snapshots.sort(key=lambda item: item.stat().st_mtime)
        return str(snapshots[-1])

    @staticmethod
    def _device_summary() -> dict[str, Any]:
        try:
            import torch
        except Exception:
            return {'torch_available': False, 'device': 'cpu'}
        cuda_available = bool(torch.cuda.is_available())
        summary: dict[str, Any] = {'torch_available': True, 'device': 'cuda' if cuda_available else 'cpu', 'cuda_available': cuda_available}
        if cuda_available:
            summary['gpu_name'] = torch.cuda.get_device_name(0)
            try:
                props = torch.cuda.get_device_properties(0)
                summary['vram_gb'] = round(props.total_memory / (1024 ** 3), 2)
            except Exception:
                pass
        return summary

    def run(self, config: OperatorTrainConfig) -> dict[str, Any]:
        records = self.load_sft_records(config.train_jsonl)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {
            'config': config.model_dump(),
            'num_records': len(records),
            'device_summary': self._device_summary(),
            'train_jsonl': config.train_jsonl,
            'resume_from_checkpoint': config.resume_from_checkpoint,
        }
        if config.dry_run:
            plan_path = output_dir / 'operator_training_plan.json'
            plan_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
            self._save_run_metadata(output_dir, config, config.model_name_or_path)
            summary['mode'] = 'dry_run'
            summary['plan_path'] = str(plan_path)
            return summary
        try:
            import torch
            from torch.utils.data import Dataset
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
        except Exception as exc:
            raise RuntimeError('transformers/torch training dependencies are not available.') from exc

        class SftDataset(Dataset):
            def __init__(self, items: list[DistillationSftRecord], tokenizer: object, max_length: int) -> None:
                self.items = items
                self.tokenizer = tokenizer
                self.max_length = max_length
            def __len__(self) -> int:
                return len(self.items)
            def __getitem__(self, index: int) -> dict[str, Any]:
                record = self.items[index]
                text = f"{record.prompt}\n\n### Answer\n{record.completion}"
                encoded = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
                input_ids = encoded['input_ids'][0]
                attention_mask = encoded['attention_mask'][0]
                return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids.clone()}

        model_source = self._resolve_local_model_source(config.model_name_or_path) if config.local_files_only else config.model_name_or_path
        summary['resolved_model_source'] = model_source
        effective_resume, resume_info = self.resolve_resume_checkpoint(output_dir, config.resume_from_checkpoint, config, model_source)
        summary.update(resume_info)
        if config.local_files_only:
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=config.local_files_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_kwargs: dict[str, Any] = {'local_files_only': config.local_files_only}
        if config.use_qlora:
            try:
                import torch
                if torch.cuda.is_available():
                    model_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                    model_kwargs['device_map'] = 'auto'
            except Exception:
                pass
        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        if hasattr(model, 'config'):
            model.config.use_cache = False
        if config.use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except Exception as exc:
                raise RuntimeError('LoRA/QLoRA requested but peft is not available.') from exc
            target_modules = config.lora_target_modules or ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
            lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=target_modules, bias='none')
            model = get_peft_model(model, lora_config)
            summary['lora_target_modules'] = target_modules
        dataset = SftDataset(records, tokenizer, config.max_length)
        use_cuda = bool(torch.cuda.is_available())
        training_kwargs = {
            'output_dir': str(output_dir),
            'per_device_train_batch_size': config.batch_size,
            'gradient_accumulation_steps': config.gradient_accumulation_steps,
            'learning_rate': config.learning_rate,
            'max_steps': config.max_steps,
            'warmup_steps': config.warmup_steps,
            'logging_steps': max(1, min(10, config.max_steps)),
            'save_steps': max(1, config.save_steps),
            'save_total_limit': max(1, config.save_total_limit),
            'report_to': [],
            'fp16': False,
            'bf16': False,
            'seed': config.seed,
            'remove_unused_columns': False,
        }
        try:
            args = TrainingArguments(**training_kwargs, overwrite_output_dir=False)
        except TypeError:
            args = TrainingArguments(**training_kwargs)
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        try:
            trainer.train(resume_from_checkpoint=effective_resume or None)
        except Exception as exc:
            message = str(exc)
            if effective_resume and 'parameter group' in message and 'optimizer' in message:
                summary['resume_retry_reason'] = 'optimizer_state_mismatch'
                summary['resume_retry_checkpoint'] = effective_resume
                trainer.train(resume_from_checkpoint=None)
            else:
                raise
        self._save_run_metadata(output_dir, config, model_source)
        trainer.save_model(str(output_dir / 'final_model'))
        tokenizer.save_pretrained(str(output_dir / 'final_model'))
        latest_checkpoint = self.find_latest_checkpoint(output_dir)
        checkpoint_dirs = sorted([str(path) for path in output_dir.glob('checkpoint-*') if path.is_dir()], key=lambda value: int(Path(value).name.split('-', 1)[1]) if Path(value).name.split('-', 1)[1].isdigit() else -1)
        summary['mode'] = 'train'
        summary['final_model_dir'] = str(output_dir / 'final_model')
        summary['latest_checkpoint'] = latest_checkpoint
        summary['checkpoint_dirs'] = checkpoint_dirs
        return summary

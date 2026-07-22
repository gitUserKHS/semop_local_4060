from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .semantic_distillation import (
    SemanticDistillationConfig,
    SemanticDistillationCorpus,
    build_training_messages,
)
from .semantic_models import MODEL_SPECS, _pretrained_reference


def train_semantic_student(
    corpus_path: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 2,
    max_records: int = 0,
    device: str = "auto",
    seed: int = 4060,
) -> dict[str, Any]:
    """Train a candidate-only Qwen3.5-0.8B LoRA from reviewed train records."""

    if epochs <= 0 or max_records < 0:
        raise ValueError("epochs must be positive and max_records cannot be negative")
    corpus_file = Path(corpus_path)
    output = Path(output_path)
    if output.exists():
        raise ValueError("semantic student output already exists")
    corpus = SemanticDistillationCorpus.load_jsonl(corpus_file)
    corpus.require_split("train")
    positives = corpus.positives
    if max_records:
        positives = positives[:max_records]
    if not positives:
        raise ValueError("distillation corpus contains no positive records")
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "install requirements-distill.txt before semantic student training"
        ) from exc

    config = SemanticDistillationConfig(epochs=epochs)
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available() and device != "cpu"
    if device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA was requested but is unavailable")
    if use_cuda and config.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the configured GPU does not support BF16 training")
    active_device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.bfloat16 if use_cuda and config.bf16 else torch.float32

    model_reference = _pretrained_reference(
        config.student_model_id,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_reference,
        local_files_only=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_reference,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(active_device)
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)
    update_count = 0
    losses: list[float] = []

    def autocast_context():
        if use_cuda and config.bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    for _ in range(config.epochs):
        for index, record in enumerate(positives, start=1):
            batch = _encode_record(processor, record, config.max_sequence_length)
            batch = {name: value.to(active_device) for name, value in batch.items()}
            with autocast_context():
                output = model(**batch)
                loss = output.loss / config.gradient_accumulation_steps
            loss.backward()
            losses.append(float(output.loss.detach().cpu()))
            if index % config.gradient_accumulation_steps == 0 or index == len(positives):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1

    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True)
    processor.save_pretrained(output)
    corpus_digest = sha256(corpus_file.read_bytes()).hexdigest()
    summary = {
        "schema_version": "semop.semantic-student-training.v1",
        "candidate_only": True,
        "student_model_id": config.student_model_id,
        "student_model_revision": MODEL_SPECS[config.student_model_id].revision,
        "teacher_model_id": config.teacher_model_id,
        "corpus_sha256": corpus_digest,
        "positive_records": len(positives),
        "hard_negatives_reserved_for_ranking": len(corpus.records) - len(corpus.positives),
        "training_updates": update_count,
        "mean_loss": sum(losses) / len(losses),
        "device": str(active_device),
        "bf16": use_cuda and config.bf16,
        "bitsandbytes_used": False,
        "config": asdict(config),
    }
    (output / "semop_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _encode_record(processor: Any, record: Any, max_length: int) -> dict[str, Any]:
    import torch

    messages = list(build_training_messages(record))
    chat_template = getattr(processor, "chat_template", None)
    if not chat_template:
        chat_template = getattr(
            getattr(processor, "tokenizer", None),
            "chat_template",
            None,
        )
    if not chat_template:
        raise RuntimeError("student model does not provide a chat template")
    prompt_text = processor.apply_chat_template(
        messages[:-1],
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        messages,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    prompt_length = min(len(prompt_ids), int(labels.shape[-1]))
    labels[:, :prompt_length] = -100
    if not bool((labels != -100).any()):
        raise ValueError("training completion was truncated entirely")
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels.to(dtype=torch.long),
    }

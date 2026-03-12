# Operator Learning Plan

## Goal

Train a small local student model to internalize operator-centered reasoning across:
- hidden-premise recovery
- CP problem structuring
- hidden CP constraints
- VLSO grounded visual reasoning
- operator proposal and self-evolution traces

The target is not raw answer imitation. The target is reusable intermediate reasoning structure.

## Training Philosophy

Use a teacher-student-verifier setup.

Teacher:
- SemOp structured pipelines
- hidden-premise engine
- CP parser
- VLSO grounded QA
- operator proposal / self-evolution loops

Student:
- small local model trained with LoRA or QLoRA
- predicts structured JSON outputs from prompts

Verifier:
- existing SemOp evaluators
- held-out premise, CP, VLSO, and transfer benchmarks

## Curriculum

### Phase 1. Foundation
Tasks:
- hidden_premise
- cp_structuring
- cp_hidden_constraints

### Phase 2. Grounding
Tasks:
- vlso_grounded_qa
- vlso_real_image

### Phase 3. Operatorization
Tasks:
- operator_proposal
- operator_self_evolution

## Commands

### 1. Export teacher traces

```bash
.\.venv312\Scripts\python.exe tools\eval\export_teacher_traces.py ^
  --hidden-premises examples\hidden_premise_eval.jsonl ^
  --cp-input examples\cp_parser_eval.jsonl ^
  --cp-hidden-input examples\cp_hidden_constraint_eval.jsonl ^
  --vlso-input examples\vlso_eval.jsonl ^
  --vlso-real-image-input examples\vlso_real_image_eval_gold.jsonl ^
  --operator-transfer-input examples\operator_transfer_eval.jsonl ^
  --examples-root examples ^
  --output data\teacher_traces.jsonl ^
  --sft-output data\teacher_traces_sft.jsonl
```

### 2. Build operator-learning bundle

```bash
.\.venv312\Scripts\python.exe tools\eval\build_operator_learning_bundle.py ^
  --teacher-traces data\teacher_traces.jsonl ^
  --workspace data\operator_learning_bundle
```

### 3. Dry-run generic operator training

```bash
.\.venv312\Scripts\python.exe tools\eval\run_operator_training.py ^
  --workspace data\operator_learning_bundle ^
  --model Qwen/Qwen2.5-0.5B-Instruct ^
  --dry-run ^
  --use-lora
```

### 4. Real LoRA/QLoRA training

```bash
.\.venv312\Scripts\python.exe tools\eval\run_operator_training.py ^
  --workspace data\operator_learning_bundle ^
  --model E:\path\to\local_text_model ^
  --use-lora ^
  --use-qlora ^
  --max-steps 200 ^
  --save-steps 25 ^
  --save-total-limit 3
```

### 5. Resume from checkpoint

```bash
.\.venv312\Scripts\python.exe tools\eval\run_operator_training.py ^
  --workspace data\operator_learning_bundle ^
  --model E:\path\to\local_text_model ^
  --use-lora ^
  --use-qlora ^
  --resume-from-checkpoint data\operator_learning_bundle\training_run\checkpoint-100
```

## RTX 4060 Guidance

Recommended starting point:
- model size: 0.5B to 1.5B
- use_lora: on
- use_qlora: on only when `bitsandbytes` is available
- batch size: 1
- grad accumulation: 8 to 16
- max length: 768

Use dry-run first, then a short 100-200 step experiment before longer runs.

## Practical note

If LoRA fails immediately with `peft is not available`, install `peft` in the Python environment you use to launch training. If you use `local_files_only`, SemOp now resolves local Hugging Face snapshot directories first to avoid online metadata lookups.

# CP Training Manual

## Goal

Use this guide when you want to train the small CP parser yourself. The heavy training run is intentionally separate from the main application flow.

The target model is not a full solver. It is a parser that maps:
- statement -> goal types
- statement -> domain tags
- statement -> logical frames
- statement -> DSL operators
- statement -> target algorithm family
- statement -> reasoning sketch

That learned parser is then combined with:
- CP knowledge base
- episodic memory from real incidents
- validator
- repair loop

## Recommended Hardware

Known working local setup:
- Python 3.12
- `.venv312`
- RTX 4060 8GB
- CUDA-enabled PyTorch

Practical starting point:
- model size: 0.5B to 1.5B
- LoRA: enabled
- batch size: 1
- gradient accumulation: 8 to 16
- max length: 512
- LoRA rank: 8 or 16

## 1. Prepare Data

### Build the CP corpus

```bash
.\.venv312\Scripts\python.exe tools/cp/prepare_cp_corpus.py ^
  --inputs examples\cp_corpus_inputs examples\cp_statement_seeds.jsonl ^
  --output examples\cp_statement_corpus_expanded.jsonl
```

### Build the labeled DSL dataset

```bash
.\.venv312\Scripts\python.exe tools/cp/build_cp_dsl_dataset.py ^
  --inputs examples\cp_corpus_inputs examples\cp_statement_seeds.jsonl ^
  --output examples\cp_dsl_expanded_dataset.jsonl ^
  --corpus-output examples\cp_statement_corpus_expanded.jsonl ^
  --sft-output examples\cp_dsl_expanded_sft.jsonl
```

### Ingest real incidents

```bash
.\.venv312\Scripts\python.exe tools/cp/ingest_cp_episodes.py ^
  --inputs examples\cp_incident_cases.jsonl ^
  --store data\cp_episodes.db
```

### Merge into train and validation bundles

```bash
.\.venv312\Scripts\python.exe tools/cp/build_cp_training_bundle.py ^
  --inputs examples\cp_dsl_expanded_dataset.jsonl ^
  --episode-store data\cp_episodes.db ^
  --train-output data\cp_train.jsonl ^
  --val-output data\cp_val.jsonl ^
  --train-sft-output data\cp_train_sft.jsonl ^
  --val-sft-output data\cp_val_sft.jsonl
```

## 2. Dry-Run First

```bash
.\.venv312\Scripts\python.exe tools/cp/train_cp_parser.py ^
  --model Qwen/Qwen2.5-0.5B-Instruct ^
  --train-jsonl data\cp_train.jsonl ^
  --output-dir data\cp_parser_dry_run ^
  --dry-run ^
  --use-lora ^
  --batch-size 1 ^
  --grad-accum 8 ^
  --max-length 512 ^
  --lora-rank 8 ^
  --lora-alpha 16
```

Check:
- `data\cp_parser_dry_run	rain_sft.jsonl`
- `data\cp_parser_dry_run	raining_plan.json`

## 3. Actual LoRA Training

```bash
.\.venv312\Scripts\python.exe tools/cp/train_cp_parser.py ^
  --model Qwen/Qwen2.5-0.5B-Instruct ^
  --train-jsonl data\cp_train.jsonl ^
  --output-dir data\cp_parser_lora_4060 ^
  --use-lora ^
  --batch-size 1 ^
  --grad-accum 8 ^
  --max-steps 300 ^
  --lr 2e-5 ^
  --max-length 512 ^
  --warmup-steps 20 ^
  --lora-rank 8 ^
  --lora-alpha 16 ^
  --lora-dropout 0.05
```

Or train from episodes directly:

```bash
.\.venv312\Scripts\python.exe tools/cp/train_cp_parser.py ^
  --model Qwen/Qwen2.5-0.5B-Instruct ^
  --episode-store data\cp_episodes.db ^
  --output-dir data\cp_parser_from_episodes ^
  --use-lora ^
  --batch-size 1 ^
  --grad-accum 8 ^
  --max-steps 300
```

## 4. Evaluate

```bash
.\.venv312\Scripts\python.exe tools/eval/evaluate_cp_parser.py ^
  --input data\cp_val.jsonl ^
  --mode heuristic
```

Track at least:
- algorithm exact match
- logical frame overlap
- DSL operator overlap
- reasoning sketch usefulness by manual spot-check

## 5. Data Quality Rules

Good data:
- accepted editorial summaries
- wrong-answer cases with the hidden edge case noted
- time-limit cases with the failed first idea recorded
- concise statement rewrites

Bad data:
- code without the statement
- verdict without failure reason
- mislabeled algorithm families
- raw very long editorials copied without summarization


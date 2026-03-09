# Data Layout

This document explains what data to download or prepare, where to put it, and what each folder is used for.

## 1. Customer Documents

Recommended location:
- `data\customer_docs\`

Put these file types there:
- `.md`
- `.txt`

Recommended contents:
- SOP documents
- exception handling guides
- onboarding manuals
- quality gate procedures
- safety procedures

Example:
- `data\customer_docs\warehouse_sop_01.md`
- `data\customer_docs\shipping_quality_rules.txt`

### Convert customer docs into eval stubs

```bash
.\.venv312\Scripts\python.exe tools/ops/build_customer_eval_from_docs.py ^
  --inputs data\customer_docs ^
  --output data\customer_eval.jsonl ^
  --max-cases 50
```

Output:
- `data\customer_eval.jsonl`

## 2. Customer Eval JSONL

Recommended location:
- `data\customer_eval.jsonl`

This is the main labeled benchmark file you should maintain during a PoC.

If you do not have customer data yet, start from:
- `examples\customer_ops_eval_template.jsonl`
- `examples\ops_labeled_eval_ko.jsonl`

## 3. Customer Baseline Config

Recommended location:
- `data\customer_baseline_config.json`

Starter example:
- `examples\customer_baseline_config.json`

Use this when you want the baseline to reflect what the customer already does.
For example:
- boost chunks that mention approval
- prefer chunks that mention hold or rescan
- require that at least one chunk includes a domain-critical keyword

### Run benchmark with this config

```bash
.\.venv312\Scripts\python.exe tools/eval/compare_ops_baseline.py ^
  --input data\customer_eval.jsonl ^
  --baseline configurable_keyword ^
  --baseline-config data\customer_baseline_config.json
```

## 4. Review Queue Database

Recommended location:
- `data\ops_review_queue.db`

This SQLite database stores:
- risky answers that were queued for review
- KPI snapshot at queue time
- audit items
- review status
- resolution note

Statuses:
- `pending`
- `approved`
- `rejected`
- `needs_followup`

## 5. Learned Feedback Rules

Recommended location:
- `data\feedback_rules.json`

These rules are generated from reviewed queue items and fed back into the copilot.

### Generate rules

```bash
.\.venv312\Scripts\python.exe tools/ops/learn_feedback_rules.py ^
  --review-queue data\ops_review_queue.db ^
  --output data\feedback_rules.json
```

### Use rules

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --feedback-rules data\feedback_rules.json ^
  --query "The aisle is blocked and approval is still missing. What should I do?" ^
  --context-file data\customer_docs\warehouse_sop_01.md
```

## 6. Public Dataset Storage

If you use the research pipeline with public datasets, these are the main locations.

### Downloaded raw files
- `data\starter_downloads_312\`
- `data\reasoning_core_downloads_312\`
- `data\all_public_downloads_312\`

### Normalized JSONL files
- `data\starter_normalized_312\`
- `data\reasoning_core_normalized_312\`
- `data\all_public_normalized_312\`

### SQLite memory stores
- `data\semop_starter_312.db`
- `data\semop_reasoning_core_312.db`
- `data\semop_all_public_312.db`

You normally do not place files in these folders manually. The ingest scripts fill them.

## 7. Recommended Minimal PoC Layout

A practical customer PoC can start with this structure:

- `data\customer_docs\`
- `data\customer_eval.jsonl`
- `data\customer_baseline_config.json`
- `data\ops_review_queue.db`
- `data\feedback_rules.json`

That is the minimum operational data layout for this repository.

## 8. Competitive Programming Knowledge, Episodes, And Corpora

Recommended files and folders:
- `data\knowledge\cp_knowledge.json`
- `data\cp_episodes.db`
- `examples\cp_corpus_inputs\`
- `examples\cp_statement_seeds.jsonl`
- `examples\cp_statement_corpus_expanded.jsonl`
- `examples\cp_dsl_expanded_dataset.jsonl`
- `examples\cp_dsl_expanded_sft.jsonl`
- `data\cp_train.jsonl`
- `data\cp_val.jsonl`
- `data\cp_train_sft.jsonl`
- `data\cp_val_sft.jsonl`

What each item is for:
- `cp_knowledge.json`: CP DSL, logical frames, algorithm metadata, compiler profile, and hardware notes
- `cp_episodes.db`: successful and failed contest attempts, validation outcomes, repair traces, and generated code
- `cp_corpus_inputs\`: mixed raw statement sources in `.txt`, `.md`, `.jsonl`, `.json`, `.csv`, `.html`, or `.zip`
- `cp_statement_corpus_expanded.jsonl`: normalized and deduplicated statement corpus
- `cp_dsl_expanded_dataset.jsonl`: labeled statement -> DSL / frame / algorithm training set
- `cp_dsl_expanded_sft.jsonl`: prompt/completion records for small-model parser training
- `cp_train.jsonl` / `cp_val.jsonl`: merged train/validation DSL bundles
- `cp_train_sft.jsonl` / `cp_val_sft.jsonl`: prompt/completion train/validation bundles

You can rebuild the corpus and datasets with:
```bash
.\.venv312\Scripts\python.exe tools/cp/prepare_cp_corpus.py ^
  --inputs examples\cp_corpus_inputs examples\cp_statement_seeds.jsonl ^
  --output examples\cp_statement_corpus_expanded.jsonl

.\.venv312\Scripts\python.exe tools/cp/build_cp_dsl_dataset.py ^
  --inputs examples\cp_corpus_inputs examples\cp_statement_seeds.jsonl ^
  --output examples\cp_dsl_expanded_dataset.jsonl ^
  --corpus-output examples\cp_statement_corpus_expanded.jsonl ^
  --sft-output examples\cp_dsl_expanded_sft.jsonl
```

You can record episodic memory during solving with:
```bash
.\.venv312\Scripts\python.exe solve_contest.py ^
  --query "There are many range sum queries on an array and no updates. Output the sum from l to r each time." ^
  --episode-store data\cp_episodes.db
```

You can prepare a small parser training run with:
```bash
.\.venv312\Scripts\python.exe tools/cp/train_cp_parser.py ^
  --model Qwen/Qwen2.5-0.5B-Instruct ^
  --episode-store data\cp_episodes.db ^
  --output-dir data\cp_parser_from_episodes ^
  --dry-run
```

## 9. Temporary Compile And Validation Files

Temporary C++ build and validation files are written under:
- `data\tmp\cpp_syntax_checks\`
- `data\tmp\cpp_validation\`

These files are transient local artifacts and should not be committed.




## 10. CP Incident Archives And GUI

Recommended files:
- `examples\cp_incident_cases.jsonl`
- `data\cp_episodes.db`

Recommended incident fields:
- `problem_id`
- `statement`
- `editorial_summary`
- `outcome`
- `failure_kind`
- `code`

```bash
.\.venv312\Scripts\python.exe tools/cp/ingest_cp_episodes.py ^
  --inputs examples\cp_incident_cases.jsonl ^
  --store data\cp_episodes.db
```

```bash
.\.venv312\Scripts\python.exe cp_copilot_gui.py ^
  --episode-store data\cp_episodes.db
```


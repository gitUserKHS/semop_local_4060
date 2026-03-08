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
.\.venv312\Scripts\python.exe build_customer_eval_from_docs.py ^
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
.\.venv312\Scripts\python.exe compare_ops_baseline.py ^
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
.\.venv312\Scripts\python.exe learn_feedback_rules.py ^
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

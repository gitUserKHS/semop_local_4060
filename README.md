# SemOp Local 4060

SemOp Local 4060 is a local prototype for structured reasoning and domain copilot experiments.

The current product direction is not a general chatbot. It is a Korean warehouse and field-operations SOP copilot that:
- reads SOPs, manuals, and exception-handling documents
- reconstructs hidden prerequisites, blockers, and alternatives
- reduces unsafe or non-executable answers
- leaves an audit trace and operational KPI summary
- compares itself against configurable baselines such as plain RAG

## What You Can Run Today

- `app.py`
  - research-oriented structured reasoning CLI
- `ops_copilot.py`
  - product-style warehouse/operations copilot CLI
- `ops_copilot_gui.py`
  - local browser GUI for testing queries, baselines, and review queue items
- `compare_ops_baseline.py`
  - SemOp vs baseline benchmark on labeled cases
- `evaluate_ops_kpis.py`
  - KPI averages on operations case sets
- `learn_feedback_rules.py`
  - converts resolved review items into reusable feedback rules

## Recommended Environment

This repository has already been tested with a Python 3.12 virtual environment and GPU PyTorch.

Known working environment:
- Python `3.12`
- virtual environment: `.venv312`
- torch `2.10.0+cu128`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 4060`

Run commands with:

```bash
.\.venv312\Scripts\python.exe <script>.py ...
```

## Quick Start

### 1. Run the operations copilot on a single SOP question

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --query "지게차로 팔레트를 랙에 올리려는데 통로가 막혀 있고 아직 승인도 안 났습니다. 어떻게 해야 하나요?" ^
  --context-file examples\customer_sop_sample.md
```

### 2. Open the GUI

```bash
.\.venv312\Scripts\python.exe ops_copilot_gui.py ^
  --review-queue data\ops_review_queue.db ^
  --baseline-config examples\customer_baseline_config.json
```

Then open `http://127.0.0.1:8765`.

### 3. Compare SemOp against a baseline

```bash
.\.venv312\Scripts\python.exe compare_ops_baseline.py ^
  --input examples\ops_labeled_eval_ko.jsonl ^
  --baseline configurable_keyword ^
  --baseline-config examples\customer_baseline_config.json
```

### 4. Learn feedback rules from reviewed items

```bash
.\.venv312\Scripts\python.exe learn_feedback_rules.py ^
  --review-queue data\ops_review_queue.db ^
  --output data\feedback_rules.json
```

### 5. Re-run the copilot with learned rules

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --feedback-rules data\feedback_rules.json ^
  --query "The aisle is blocked and approval is still missing. What should I do?" ^
  --context-file examples\customer_sop_sample.md
```

## Where To Put Data

There are two main data paths.

### A. Public reasoning datasets

You do not need to download these manually if you use the curated manifest scripts.

1. Generate a manifest:

```bash
.\.venv312\Scripts\python.exe write_curated_manifest.py --preset reasoning_core --output data\reasoning_core_manifest.json
```

2. Ingest it:

```bash
.\.venv312\Scripts\python.exe ingest_public_manifest.py ^
  --manifest data\reasoning_core_manifest.json ^
  --download-root data\reasoning_core_downloads_312 ^
  --normalized-root data\reasoning_core_normalized_312 ^
  --store data\semop_reasoning_core_312.db ^
  --mode heuristic ^
  --overwrite
```

What goes where:
- raw downloaded files: `data\reasoning_core_downloads_312\`
- normalized JSONL files: `data\reasoning_core_normalized_312\`
- final SQLite memory DB: `data\semop_reasoning_core_312.db`

### B. Customer SOP or manual documents

Put customer `.md` or `.txt` files in a folder such as:
- `data\customer_docs\`

Then build evaluation stubs from those documents:

```bash
.\.venv312\Scripts\python.exe build_customer_eval_from_docs.py ^
  --inputs data\customer_docs ^
  --output data\customer_eval.jsonl ^
  --max-cases 50
```

If you do not have customer data yet, start from:
- `examples\customer_sop_sample.md`
- `examples\customer_ops_eval_template.jsonl`
- `examples\ops_labeled_eval_ko.jsonl`

## Important Config Files

- baseline config JSON:
  - example: `examples\customer_baseline_config.json`
  - purpose: define a customer-style baseline retriever
- learned feedback rules JSON:
  - example output: `data\feedback_rules.json`
  - purpose: apply supervisor-reviewed corrections back into the copilot
- review queue SQLite:
  - example: `data\ops_review_queue.db`
  - purpose: triage risky answers and record resolution notes

## Main Docs

- usage manual: `docs/usage_manual.md`
- data layout and download guide: `docs/data_layout.md`
- docs index: `docs/index.md`
- GUI and evaluation workflow: `docs/gui_and_eval_workflow.md`
- feedback loop workflow: `docs/feedback_loop_workflow.md`
- customer eval schema: `docs/customer_eval_schema.md`
- architecture and internals: `docs/architecture_and_features.md`
- product framing review: `docs/productization_review.md`

## Validation Status

Latest verified commands:
- `python -m unittest discover -s tests -v`
- `.\.venv312\Scripts\python.exe -m unittest discover -s tests -v`
- `.\.venv312\Scripts\python.exe compare_ops_baseline.py --input examples\ops_labeled_eval_ko.jsonl --baseline configurable_keyword --baseline-config examples\customer_baseline_config.json`

At the time of the last validation:
- all tests passed: `38`
- SemOp on labeled ops eval achieved:
  - `avg_relation_recall = 1.0`
  - `avg_answer_term_recall = 0.89`
  - `forbidden_phrase_hit_rate = 0.0`

## Current Limits

- customer evaluation sets are still small unless you add real customer documents
- KPI scores are useful for PoC work, but should still be calibrated against human labels
- review-derived feedback rules are simple and heuristic, not full training updates
- GUI is for internal demos and PoCs, not hardened production deployment

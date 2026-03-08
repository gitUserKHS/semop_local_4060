# Usage Manual

## 1. Decide Which Flow You Need

Use one of these flows.

### Research flow
Use this if you want to inspect the structured meaning graph directly.

```bash
.\.venv312\Scripts\python.exe app.py --mode heuristic --query "세차장이 멀고 길이 막히는데 어떻게 가야 할까요?"
```

### Product copilot flow
Use this if you want warehouse or operations answers with KPI and audit trace.

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --query "지게차로 팔레트를 랙에 올리려는데 통로가 막혀 있고 아직 승인도 안 났습니다. 어떻게 해야 하나요?" ^
  --context-file examples\customer_sop_sample.md
```

## 2. Prepare Data

### Option A: Use included example data

Files already included:
- `examples\customer_sop_sample.md`
- `examples\ops_sop_cases_ko.jsonl`
- `examples\ops_labeled_eval_ko.jsonl`
- `examples\customer_ops_eval_template.jsonl`
- `examples\customer_baseline_config.json`

These are enough to test the full workflow.

### Option B: Use your own customer SOP documents

1. Put `.md` or `.txt` files into a folder, for example:
- `data\customer_docs\`

2. Generate evaluation case stubs from those docs:

```bash
.\.venv312\Scripts\python.exe build_customer_eval_from_docs.py ^
  --inputs data\customer_docs ^
  --output data\customer_eval.jsonl ^
  --max-cases 50
```

3. Manually review and edit `data\customer_eval.jsonl`.

You should especially review these fields:
- `expected_relations`
- `expected_answer_terms`
- `forbidden_phrases`
- `expected_clarification`

## 3. Run a Baseline Comparison

### Default lexical baseline

```bash
.\.venv312\Scripts\python.exe compare_ops_baseline.py ^
  --input examples\ops_labeled_eval_ko.jsonl ^
  --baseline lexical_rag
```

### First-chunk baseline

```bash
.\.venv312\Scripts\python.exe compare_ops_baseline.py ^
  --input examples\ops_labeled_eval_ko.jsonl ^
  --baseline first_chunk
```

### Customer-config baseline

```bash
.\.venv312\Scripts\python.exe compare_ops_baseline.py ^
  --input data\customer_eval.jsonl ^
  --baseline configurable_keyword ^
  --baseline-config examples\customer_baseline_config.json
```

If you want a customer-specific baseline, copy and edit:
- `examples\customer_baseline_config.json`

## 4. Use the GUI

Start the GUI:

```bash
.\.venv312\Scripts\python.exe ops_copilot_gui.py ^
  --review-queue data\ops_review_queue.db ^
  --baseline-config examples\customer_baseline_config.json
```

Then open:
- `http://127.0.0.1:8765`

The GUI lets you:
- choose example cases
- paste your own SOP context
- run SemOp Copilot
- compare against a baseline
- inspect review queue items
- approve, reject, or mark items as follow-up

## 5. Resolve Review Items And Learn Rules

### Step 1: Run the copilot with a review queue enabled

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --review-queue data\ops_review_queue.db ^
  --query "The aisle is blocked and approval is still missing. What should I do?" ^
  --context-file examples\customer_sop_sample.md
```

### Step 2: Open the GUI and resolve pending items

Set the status to one of:
- `approved`
- `rejected`
- `needs_followup`
- `pending`

Add a resolution note describing what should change.

### Step 3: Learn feedback rules from reviewed items

```bash
.\.venv312\Scripts\python.exe learn_feedback_rules.py ^
  --review-queue data\ops_review_queue.db ^
  --output data\feedback_rules.json
```

### Step 4: Re-run the copilot with the learned rules

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --feedback-rules data\feedback_rules.json ^
  --query "The aisle is blocked and approval is still missing. What should I do?" ^
  --context-file examples\customer_sop_sample.md
```

## 6. Optional Public Dataset Pipeline

If you want to test the broader research pipeline, use curated public datasets.

### Generate a curated manifest

```bash
.\.venv312\Scripts\python.exe write_curated_manifest.py --preset reasoning_core --output data\reasoning_core_manifest.json
```

### Download, normalize, and ingest

```bash
.\.venv312\Scripts\python.exe ingest_public_manifest.py ^
  --manifest data\reasoning_core_manifest.json ^
  --download-root data\reasoning_core_downloads_312 ^
  --normalized-root data\reasoning_core_normalized_312 ^
  --store data\semop_reasoning_core_312.db ^
  --mode heuristic ^
  --overwrite
```

## 7. What To Edit First In A Real PoC

If you start with a real customer, edit these first:
- `data\customer_eval.jsonl`
- `examples\customer_baseline_config.json` or your own baseline config copy
- `data\feedback_rules.json` after the first supervisor review cycle

That is the minimum set needed to turn this repo from demo mode into a customer-specific PoC.

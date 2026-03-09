# Feedback Loop Workflow

## Goal

Turn supervisor review outcomes into reusable rules that the copilot applies on later runs.

## Inputs

- review queue database:
  - `data\ops_review_queue.db`
- resolved review items with status:
  - `approved`
  - `needs_followup`
- optional resolution notes written by the reviewer

## Output

- learned feedback rule file:
  - `data\feedback_rules.json`

## What A Feedback Rule Can Do

A learned rule can:
- add unsafe phrases into `invalid_advice`
- insert recommended actions into the answer alternatives
- warn when important terms are missing from the plan
- leave an explicit feedback note in warnings and audit trace

## End-To-End Flow

### 1. Run the copilot with a review queue

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --review-queue data\ops_review_queue.db ^
  --query "The aisle is blocked and approval is still missing. What should I do?" ^
  --context-file examples\customer_sop_sample.md
```

### 2. Resolve review items in the GUI

```bash
.\.venv312\Scripts\python.exe ops_copilot_gui.py ^
  --review-queue data\ops_review_queue.db ^
  --baseline-config examples\customer_baseline_config.json
```

Open `http://127.0.0.1:8765`, select a queued item, and set:
- `approved`
- `rejected`
- `needs_followup`
- `pending`

Also write a short resolution note.

### 3. Learn rules from reviewed items

```bash
.\.venv312\Scripts\python.exe tools/ops/learn_feedback_rules.py ^
  --review-queue data\ops_review_queue.db ^
  --output data\feedback_rules.json
```

### 4. Re-run the copilot with learned rules attached

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --feedback-rules data\feedback_rules.json ^
  --query "The aisle is blocked and approval is still missing. What should I do?" ^
  --context-file examples\customer_sop_sample.md
```

## How Matching Works

Rules are matched by:
- `domain`
- `scenario`
- `trigger_terms`

So if you want a rule to apply later, the future run must use the same or compatible scenario tag.

## Customer Document Bootstrap

If a customer gives you a folder of `.txt` or `.md` files, build eval stubs first.

```bash
.\.venv312\Scripts\python.exe tools/ops/build_customer_eval_from_docs.py ^
  --inputs data\customer_docs ^
  --output data\customer_eval.jsonl ^
  --max-cases 50
```

Then benchmark against the customer baseline.

```bash
.\.venv312\Scripts\python.exe tools/eval/compare_ops_baseline.py ^
  --input data\customer_eval.jsonl ^
  --baseline configurable_keyword ^
  --baseline-config data\customer_baseline_config.json
```

## Practical Recommendation

Do not learn rules from every reviewed item.
Start with:
- repeated approval failures
- repeated hold/rescan misses
- repeated unsafe direct-execution advice
- repeated clarification misses

Those four categories usually give the highest value first.


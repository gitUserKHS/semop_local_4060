# GUI And Evaluation Workflow

## Goal

Use the local GUI as a single test console for:
- SemOp Copilot answers
- baseline comparison
- KPI inspection
- review queue triage
- review status updates

## Start The GUI

```bash
.\.venv312\Scripts\python.exe ops_copilot_gui.py ^
  --review-queue data\ops_review_queue.db ^
  --baseline-config examples\customer_baseline_config.json ^
  --feedback-rules data\feedback_rules.json
```

Open:
- `http://127.0.0.1:8765`

## What The GUI Shows

### Copilot form
- domain selector
- scenario field
- query input
- SOP context input
- example selector
- baseline selector
- optional baseline comparison toggle

### SemOp result panel
- answer text
- KPI summary
- audit trace
- review queue status if the answer was queued

### Baseline panel
- baseline answer text
- retrieved chunks

### Review dashboard
- queue counts by status
- pending items
- recent items
- selected item detail JSON
- review status update form

## Typical GUI Session

1. Pick an example or paste your own SOP context.
2. Run SemOp.
3. Enable baseline comparison if needed.
4. Inspect KPI and audit trace.
5. If the case lands in review, open it from the review dashboard.
6. Set status and add a resolution note.
7. Later, export learned feedback rules from the queue.

## KPI CLI

If you want only KPI averages for a case set:

```bash
.\.venv312\Scripts\python.exe tools/eval/evaluate_ops_kpis.py --input examples\ops_sop_cases_ko.jsonl
```

## Baseline Benchmark CLI

### Lexical baseline

```bash
.\.venv312\Scripts\python.exe tools/eval/compare_ops_baseline.py --input examples\ops_labeled_eval_ko.jsonl --baseline lexical_rag
```

### Customer-config baseline

```bash
.\.venv312\Scripts\python.exe tools/eval/compare_ops_baseline.py ^
  --input data\customer_eval.jsonl ^
  --baseline configurable_keyword ^
  --baseline-config data\customer_baseline_config.json
```

## Current Benchmark Meaning

The benchmark is product-facing, not academic.

It focuses on:
- relation recovery
- answer-term coverage
- forbidden phrase hits
- clarification alignment

This is enough to show whether the structured copilot understands operational context better than a baseline retriever.

## Current Limits

- benchmark labels still need manual review in real customer PoCs
- baseline results are only as realistic as the baseline config you provide
- GUI is for local PoC and internal testing, not production deployment


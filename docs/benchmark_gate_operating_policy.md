# Benchmark Gate Operating Policy

This document describes the operating rules wired into the continuous training loop.

## Review Promotion Rules

Approved review items are promoted into retraining traces only when they pass the promotion policy.

Promotable reasons by default:
- `approved_training_trace`
- `grounding_review`
- `clarification_need_rate`
- `low_plan_executability`
- `low_audit_usefulness`
- `manual_review_note`
- `compiler_validity_gap`
- `repair_failure`
- `context_misread_review`
- `relation_recovery_gap`

Blocked by default:
- `invalid_advice_rate`

Override:
- `approved_training_trace` can override blocked reasons when a corrected answer is present.

Additional constraints:
- review status must be `approved`
- corrected `answer_text` must be present
- `manual_review_note` requires a non-empty `resolution_note`

Implementation:
- `src/semop/operating_policies.py`
- `src/semop/continuous_learning.py`
- `src/semop/unified_benchmark.py`
- `src/semop/review_queue.py`

## Severity Weights

Review items carry an explicit severity level, and the effective weight is now policy-resolved by `domain/scenario`.

Global default weights:
- `low`: `0.75`
- `medium`: `1.0`
- `high`: `1.35`
- `critical`: `1.7`

Domain and scenario overrides:
- `warehouse_onboarding`: `low=0.8`, `high=1.45`, `critical=1.85`
- `warehouse_onboarding/onboarding`: `critical=1.95`
- `warehouse_exception`: `low=0.9`, `medium=1.1`, `high=1.6`, `critical=1.95`
- `warehouse_exception/exception_response`: `high=1.75`, `critical=2.25`
- `warehouse_exception/quality_gate`: `high=1.7`, `critical=2.05`

Operational effect:
- promoted reviews contribute policy-resolved `training_weight` into `continuous_learning_sft.jsonl`
- `review_promotion_manifest.json` records `severity` and the resolved `training_weight`
- review-derived benchmark cases keep `severity` and `case_weight`
- slice balancing keeps the highest-weight cases first when a slice is over capacity

## Domain Thresholds

### `general`
- unseen transfer: `0.05`
- analogy usefulness: `0.10`
- compiler validity: `0.60`
- grounded explanation fidelity: `0.55`
- repair success rate: `0.55`
- regression tolerance: `0.03`

### `warehouse_onboarding`
- unseen transfer: `0.08`
- analogy usefulness: `0.12`
- compiler validity: `0.68`
- grounded explanation fidelity: `0.74`
- repair success rate: `0.62`
- regression tolerance: `0.025`

### `warehouse_exception`
- unseen transfer: `0.08`
- analogy usefulness: `0.10`
- compiler validity: `0.74`
- grounded explanation fidelity: `0.68`
- repair success rate: `0.74`
- regression tolerance: `0.02`

## Slice Minimums And Slice Baseline Regression

The gate enforces scenario-aware minimums on top of overall domain thresholds.

Default high-risk slices:
- `warehouse_exception/exception_response`: compiler validity `0.80`, grounded explanation fidelity `0.75`, repair success rate `0.78`
- `warehouse_exception/quality_gate`: compiler validity `0.78`, grounded explanation fidelity `0.72`, repair success rate `0.76`
- `warehouse_onboarding/onboarding`: compiler validity `0.70`, grounded explanation fidelity `0.78`, repair success rate `0.64`

Operational effect:
- a candidate can pass overall averages and still be rejected if a high-risk slice falls below its own minimum
- a candidate can also be rejected when a slice regresses against the persisted baseline beyond tolerance, even if the overall average is stable
- `benchmark_gate.json` records `slice_metrics`, `slice_blocking_reasons`, and `baseline_slice_metrics`

## Persistent Benchmark Corpus

When `BenchmarkGatedContinuousTrainer` receives `benchmark_corpus_path`, approved promoted reviews are converted into benchmark cases and merged into a persistent corpus.

Operational effect:
- newly approved reviews immediately affect the current gate
- the same benchmark cases are reused on later runs even if the review queue has been rotated or cleared
- duplicate benchmark cases are deduplicated by stable content signature
- each `domain/scenario` slice is capped by policy, not by one global constant
- current defaults: `general/qa=4`, `warehouse_onboarding/onboarding=5`, `warehouse_exception/exception_response=6`, `warehouse_exception/quality_gate=5`
- when a slice exceeds its cap, the corpus keeps the highest-weight cases first and reports `trimmed_case_count`, `slice_balance_limit`, and `slice_balance_limits`

## Runtime Outputs

Each gated training run now writes:
- `benchmark_gate.json`
- `accepted_benchmark_summary.json` when accepted
- `promoted_review_benchmark_cases.json`
- `benchmark_corpus.json` or another configured persistent benchmark corpus path
- `continuous_learning_bundle/review_promotion_manifest.json`

These files make it explicit why a run was accepted or rejected, which approved reviews were promoted or filtered, how severe those promoted reviews were, and which slice-level checks blocked a candidate.

# Premise-First Refactor Status

## Purpose

This document records the current refactor direction that puts hidden-premise recovery ahead of later answer generation.

## What Changed

- `HiddenPremiseExplorer` now follows a three-stage flow:
  - candidate retrieval
  - premise proposal
  - premise validation
- SQLite memory is now explicitly split into:
  - `premise_memories`
  - `operator_memories`
- The core pipeline now:
  - enriches with hidden premises
  - injects premise/operator memory hints
  - stores runtime premise/operator evidence back into SQLite
- CP parsing is now more parser-first:
  - `CompetitiveProgrammingReasoner.parse_problem(...)`
  - heuristic CP evaluation uses problem structure directly instead of full solver output
- VLSO language alignment now carries:
  - required premises
  - premise candidates
  - premise validations
  - goal-preservation checks
- Common evaluation now has a shared snapshot runner:
  - `src/semop/common_eval.py`
  - `tools/eval/evaluate_semop_stack.py`

## Why This Matters

The project goal is not just to answer. It is to recover hidden goals, prerequisites, and operator structure before giving advice or generating code. This refactor makes that architecture explicit in both runtime and evaluation.

## Current Limits

- premise retrieval now blends lexical, commonsense script, premise memory, script memory, operator memory, and a lightweight learned compatibility scorer
- premise proposal still includes domain heuristics for car-wash and bag/book cases
- CP learned parser remains optional; heuristic parser-first evaluation is the default strong baseline
- VLSO still depends on the quality of visual structure extraction upstream

## Current Validation Snapshot

- `python -m unittest discover -s tests -v` passes locally
- total passing tests: `176`
- `tools/eval/evaluate_hidden_premises.py --input examples/hidden_premise_eval.jsonl --mode heuristic` currently reports:
  - `critical_premise_recall = 0.9808`
  - `hidden_goal_recall = 1.0`
  - `goal_preservation_accuracy = 0.9615`
  - `unsupported_premise_precision = 0.9038`
  - `clarification_accuracy = 0.9`
- `requirement_state_accuracy = 0.8462`
- `clarification_score_mae = 0.2015`
- `operator_supported_premise_recall = 1.0`
- `tools/eval/evaluate_semop_stack.py --hidden-premises examples/hidden_premise_eval.jsonl --cp-input examples/cp_parser_eval.jsonl` is available as a shared stack snapshot runner for premise, CP, and VLSO evaluation

- access-family hidden goals now explicitly include box/pouch/suitcase/bin in addition to drawer/cabinet/bottle/jar
- script compatibility scoring now blends lexical overlap with concept-family compatibility and harder sibling negatives

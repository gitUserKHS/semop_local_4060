# Hidden Premise Explorer

## Why this layer exists

Some queries are misleading if the system reasons only over their surface wording.

Example:
- `???? ??? ?? ??, ?????`

A surface parse can recover:
- destination: `car_wash`
- obstacle: `traffic`
- candidate action: `walk`

But the real question usually depends on hidden goals and assumptions:
- the true goal is often `clean_car_goal`
- a normal car wash usually requires `vehicle_present`
- therefore `walk_without_car` may satisfy arrival but fail the real goal

## Current pipeline position

The hidden-premise layer now sits between surface graph extraction and later operator/memory reasoning:

`query -> surface graph -> hidden premise explorer -> logical grammar priors -> memory priors -> symbolic/verifier layers`

## What it produces

The explorer enriches `StructuredMeaningGraph` with:
- `hidden_goals`
- `hidden_assumptions`
- `required_premises`
- `optional_interpretations`
- `goal_preservation_checks`
- `clarification_needed`

## Current implemented patterns

### Car wash / service-place reasoning
- recover `clean_car_goal` as the default hidden goal
- add `vehicle_present` as a required premise
- detect booking/contact/cancellation language and switch to a conditional interpretation
- mark `walk_without_car` as `risk_high` for the washing goal and `conditionally_valid` for booking/contact goals

### Bag / containment reasoning
- recover `store_book_in_bag_goal`
- recover `open_access` and `available_space`
- mark direct insertion without access as a goal-preservation risk

## Evaluation approach

The starter evaluator uses three metrics:
- `critical_premise_recall`
- `hidden_goal_recall`
- `goal_preservation_accuracy`

This is intentionally different from plain answer accuracy. The target is not only to answer well, but to recover the hidden assumptions that justify the answer.

## Main files

- `src/semop/premise_explorer.py`
- `src/semop/premise_eval.py`
- `src/semop/pipeline.py`
- `src/semop/structures.py`
- `src/semop/response_synthesizer.py`

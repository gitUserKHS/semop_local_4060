# Operator Algebra And Functor Hypotheses

## Goal

Move from isolated operator labels to reusable operator structure.

The project now treats some higher-order operators as compositions of simpler operators and records tentative category-style mappings between domains.

## Why This Matters

If one operator can be represented as a composition of simpler operators, the system can:
- generalize from small data
- explain why a complex action or scene interpretation is valid
- align text reasoning and visual reasoning through shared structure

## Current Implementation

Primary module:
- `src/semop/operator_algebra.py`

Graph fields:
- `operator_decompositions`
- `functor_hypotheses`

Pipeline integration:
- `src/semop/pipeline.py`

VLSO projection:
- `src/semop/vlso/language_parser.py`
- `src/semop/vlso/aligner.py`
- `src/semop/vlso/qa.py`

## Example Decompositions

Visual side:
- `CONTAINER_ACCESS_OPERATOR = CONTAINER_BODY_OPERATOR + ACCESS_PORT_OPERATOR`
- `CONTROLLED_ACCESS_OPERATOR = ACCESS_PORT_OPERATOR + ACCESS_CONTROL_OPERATOR`
- `MANIPULABLE_CONTAINER_OPERATOR = CONTAINER_BODY_OPERATOR + ATTACHED_GRASP_OPERATOR`

Language side:
- `SERVICE_GOAL_OPERATOR = TYPICAL_FOR + REQUIRES`
- `GOAL_PRESERVATION_OPERATOR = hidden_goal + REQUIRES + BLOCKED_BY`
- `CONTAINMENT_GOAL_OPERATOR = TYPICAL_FOR + REQUIRES + CONTAINS`

## Current Functor Hypotheses

- `ServiceGoalToConstraintFunctor`
  - maps service scripts into hidden-goal / requirement logic
- `VisualStructureToActionFunctor`
  - maps access ports, control parts, and grasp parts into action preconditions
- `GeometryToCpFrameFunctor`
  - maps visual geometry relations into computational-geometry reasoning frames

These are hypotheses, not formal proofs. They are stored because they provide reusable alignment priors across modules.

## Hidden Premise Alignment

The hidden-premise layer now feeds into the shared operator world model.

Example:
- query: car wash + walk without car
- hidden goal: `clean_car_goal`
- required premise: `vehicle_present`
- higher operator: `GOAL_PRESERVATION_OPERATOR`

This allows VLSO and text reasoning to share the same logic about access, prerequisites, and goal failure.

## Evaluation

Relevant tests currently verify:
- hidden-goal decomposition is induced for service questions
- VLSO language parsing projects hidden premises into the shared world model
- VLSO alignment uses hidden-goal checks to add cross-modal warnings or access-first steps

## Near-Term Next Work

1. Learn operator decompositions from more domains instead of only using seeded rules.
2. Score functor hypotheses by downstream verifier success.
3. Use operator decomposition recovery as an explicit benchmark beside label recovery.

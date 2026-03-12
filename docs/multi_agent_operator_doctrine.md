# Multi-Agent Operator Doctrine

## Mandatory Shared Goal

All agents that touch this repository should optimize for one common objective:

> Convert text, images, operational situations, and programming problems into a shared basis-operator representation, then perform reasoning, verification, and transfer through operator algebra rather than isolated domain heuristics.

This is the project-level doctrine.

## Engineering Rules

1. Do not add domain-specific logic unless it can be expressed in terms of basis operators, operator composition, or verifier logic.
2. Model-side components may propose operators, but only the verifier and transfer bench may retain them.
3. Hidden-premise recovery, VLSO structural grounding, and CP structuring must all be explainable in the same operator language.
4. Benchmarks should prioritize operator recovery, decomposition, premise support, transfer, and compiler alignment over surface phrasing.

## Basis Operator Set

The current shared basis operator families are:
- goal: `HIDDEN_GOAL`, `TYPICAL_FOR`
- constraint: `REQUIRES`, `BLOCKED_BY`, `ALTERNATIVE`
- structure: `CONTAINS`, `PART_OF`, `STRUCTURAL_PART_OF`, `CONTAINER_BODY_OPERATOR`, `ACCESS_PORT_OPERATOR`, `ACCESS_CONTROL_OPERATOR`, `ATTACHED_GRASP_OPERATOR`
- geometry: `PARALLEL`, `PERPENDICULAR`, `EQUAL_LENGTH`, `SYMMETRIC_STRUCTURE`, `AXIS_ALIGNED_STRUCTURE`
- topology: `CLOSED_BOUNDARY_STRUCTURE`
- computation: `RANGE_QUERY`, `FEASIBILITY_CHECK`, `STATE_TRANSITION`, `CONNECTIVITY`, `OPTIMIZE`

These are implemented in `src/semop/basis_operators.py`.

## Compiler Rule

Reasoning should compile into explicit instructions.
The compiler must:
- declare recovered basis operators
- bind premise assertions and relation bindings
- define higher operators only through normalized basis signatures
- emit alignment statistics so we can measure whether a reasoning path really uses the shared basis set

This is implemented in `src/semop/operator_runtime.py`.

## Benchmark Rule

Common evaluation should track not only answer correctness but also:
- operator-supported premise recall
- compiler alignment score
- basis-operator coverage or non-empty rate
- operator transfer across domains

This is wired through `src/semop/common_eval.py`, `src/semop/premise_eval.py`, and `src/semop/progress_report.py`.

## What Success Looks Like

A future strong version of SemOp should be able to:
- recover hidden premises from natural language
- recover structural operators from images
- map both into the same world model
- compose higher operators from basis operators
- verify those compositions through a deterministic runtime
- transfer retained operators across domains

If a new feature does not strengthen one of those goals, it is likely not part of the core operator-intelligence mission.


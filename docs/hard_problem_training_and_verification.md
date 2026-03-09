# Hard Problem Solving, Verification, And Training

## Goal

This document explains the current strategy for harder problems in SemOp.
The target is not only to answer a question once.
It is to:
1. structure the problem
2. generate candidate solution paths
3. verify those paths
4. feed the result back into future reasoning

This is the practical route toward harder tasks such as:
- advanced domain exception handling
- math word problems
- olympiad proof search
- competitive programming reduction

## Core Idea

Hard problems usually fail for one of two reasons:
- the system chose the wrong structure for the problem
- the system had no verification loop and accepted a weak answer too early

The current implementation therefore uses three layers:

### 1. Structure first
The pipeline converts the query into a structured graph and applies:
- relation extraction
- logical grammar priors
- operator induction
- memory priors
- symbolic reasoning

### 2. Candidate generation
The system generates one or more candidate answers or strategies from:
- symbolic arithmetic
- olympiad proof search
- competitive programming templates
- structured fallback strategies when no direct solver is available

### 3. Verification and learning
The hard-problem engine scores the candidate using explicit checks and then updates logical-pattern weights based on success or failure.

## Current Implementation

### Relation extraction priors from corpus grammar
Relevant files:
- `src/semop/logical_grammar.py`
- `src/semop/pipeline.py`

The system now uses corpus-induced grammar patterns not only for warnings and plan priors, but also for early relation injection.

Examples:
- `has` can induce `HAS`
- `requires` can induce `REQUIRES`
- `if` can induce `CONDITION_ON`
- `before` can induce `BEFORE`
- `can` can induce `AFFORDS`

This matters because hard problems depend heavily on correct early structure.
If the graph is wrong, later search is usually wrong too.

### Hard problem engine
Relevant files:
- `src/semop/hard_problem_engine.py`
- `solve_hard_problem.py`

This module runs:
1. the full structured pipeline
2. symbolic candidate collection
3. contest-style candidate generation when relevant
4. verification checks
5. pattern-weight updates

Outputs include:
- chosen answer
- verification score
- individual checks
- matched logical patterns
- whether the problem is currently treated as solved

## Verification Strategy

The current verification layer is lightweight but explicit.

### Arithmetic
Checks:
- whether equations were produced
- whether the answer is equation-backed

Why it helps:
- arithmetic should be verifiable, not only plausible

### Olympiad proof search
Checks:
- whether a proof outline exists
- whether an operator trace exists
- whether the solver says the proof is complete or still incomplete

Why it helps:
- proof problems should expose search state, not hide uncertainty

### Document grounding
Checks:
- whether evidence spans were extracted

Why it helps:
- evidence-backed answers are easier to audit and safer to trust

### Competitive programming
Checks:
- whether the problem was classified into a concrete algorithm family
- whether the generated solution is better than a generic fallback

Why it helps:
- algorithm-family classification is the first useful correctness bottleneck in contest tasks

### Structural reasoning fallback
Checks:
- whether the graph has a plan
- whether operator candidates exist
- whether obvious invalid advice is absent

Why it helps:
- even when direct solving fails, a strong structured state is still a useful intermediate result

## Training Strategy

### Current training loop
Relevant file:
- `src/semop/hard_problem_engine.py`

`PatternOutcomeTrainer` stores weights for matched logical grammar patterns in:
- `data/logical_pattern_weights.json`

For each pattern key such as:
- `PRECONDITION_FRAME::requires`
- `CONDITIONAL_FRAME::if`
- `TEMPORAL_ORDER_FRAME::before`

it records:
- success count
- failure count
- current weight

The pipeline can read these weights and use them to scale future logical grammar priors.

That means the system now has a minimal closed loop:
1. match logical patterns
2. solve problem
3. verify result
4. update weights
5. use those weights in future relation and operator priors

## Why This Is A Good Direction For Difficult Problems

### For olympiad problems
The strongest path is:
- grammar frame detection
- proof-state construction
- meta-operator search
- verification of proof trace
- weight updates from solved vs unsolved attempts

### For competitive programming
The strongest path is:
- problem statement to grammar frame
- frame to algorithm family
- algorithm family to code template
- sample-based or structural verification
- weight updates from accepted vs failed attempts

### For domain reasoning
The strongest path is:
- connector and concept binding
- relation extraction
- plan executability check
- audit trace review
- human feedback to reweight patterns

## Current Limits

This is still an early training loop.
Important gaps remain:
- verification is still heuristic for many hard tasks
- pattern weighting is not yet tied to benchmark accuracy over time
- contest verification does not yet compile and run against examples
- olympiad verification does not yet use theorem checking or formal proof checking
- relation priors still depend on simple mention-order matching

## Best Next Steps

1. Add sample-based verification for competitive programming.
2. Add proof-state scoring from known olympiad lemmas and meta-operator traces.
3. Add solved/failed benchmark batches and update weights from aggregate metrics, not single cases only.
4. Replace connector string matching with embedding-based connector and frame clustering.
5. Learn which relation direction is correct from outcome data, not only heuristics.

## Commands

### Run a hard-problem analysis

```bash
.\.venv312\Scripts\python.exe solve_hard_problem.py ^
  --query "If the bag has no open access, check the zipper before inserting the book." ^
  --memory-store data\semop_memory.db ^
  --memory-source grammar_demo
```

### Run and update pattern weights automatically

```bash
.\.venv312\Scripts\python.exe solve_hard_problem.py ^
  --query "Prove that 1+2+...+n = n(n+1)/2 for all positive integers n." ^
  --memory-store data\semop_memory.db ^
  --memory-source grammar_demo ^
  --learn ^
  --success auto
```

### Inspect the learned weights

Open:
- `data\logical_pattern_weights.json`

Those weights are then reused by the pipeline on future runs.


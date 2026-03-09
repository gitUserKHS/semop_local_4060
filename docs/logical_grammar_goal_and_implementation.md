# Logical Grammar Goal And Implementation

## Ultimate Goal

The long-term goal of this project is not only to answer questions.
It is to discover and reuse the reasoning grammar behind questions.

That means the system should learn patterns such as:
- `A has B`
- `A is B`
- `A requires B`
- `if A, then do B`
- `before A, do B`
- `A can do B`

and then bind those logical connectors to concept slots, relation types, and operator families.

In practical terms, the target behavior is:
1. read many questions and documents
2. discover recurring logical word + concept combinations
3. induce reusable grammar frames and operators
4. reuse those frames as priors on new problems
5. solve the problem with better structure, less hallucination, and stronger auditability

This is the bridge between:
- domain reasoning
- math word problems
- olympiad proof search
- competitive programming problem reduction

## Why This Matters

A raw LLM can often produce text that sounds plausible.
That is not enough for deeper reasoning tasks.

The harder step is deciding:
- what the entities are
- which relation is being asserted
- whether the sentence is a state, condition, capability, precondition, or temporal order
- which operator family should activate next

This project now starts to treat that step itself as a learnable object.

## Current Implementation

### 1. Corpus-scale logical grammar induction

Implemented in:
- `src/semop/logical_grammar.py`

This module scans many structured graphs and queries and induces patterns such as:
- `PRECONDITION_FRAME`
- `CONDITIONAL_FRAME`
- `TEMPORAL_ORDER_FRAME`
- `POSSESSION_FRAME`
- `STATE_ASSERTION_FRAME`
- `CAPABILITY_FRAME`
- `CONTAINMENT_FRAME`
- `CAUSAL_FRAME`
- `ALTERNATIVE_FRAME`

Each learned pattern stores:
- the connector text
- support count
- relation hints
- concept slots
- example frames
- example queries
- induced operator name
- grammar template
- confidence

### 2. Corpus learner now exports logical patterns

Implemented in:
- `src/semop/corpus_learning.py`

`CorpusReasoningLearner` now returns both:
- `learned_families`
- `logical_patterns`

So a training run does not only discover operator families.
It also discovers reusable logical grammar frames.

### 3. Pipeline uses logical patterns as inference priors

Implemented in:
- `src/semop/pipeline.py`
- `src/semop/corpus_store.py`

The pipeline can now:
1. load the latest corpus learning summary from SQLite memory
2. read the learned `logical_patterns`
3. match connectors in a new query
4. inject grammar hypotheses
5. inject operator candidates based on those grammar priors
6. add plan priors such as:
   - check prerequisites first
   - branch on condition first
   - preserve temporal order
   - verify capability/affordance

This means a new query is no longer parsed only from local heuristics.
It can also inherit structural priors from the corpus.

### 4. Olympiad and contest extensions still sit on top of this layer

Relevant files:
- `src/semop/olympiad_reasoner.py`
- `src/semop/contest_programmer.py`
- `solve_olympiad.py`
- `solve_contest.py`

These solvers still need more work, but they now fit the larger direction:
first identify the grammar frame and problem structure, then choose a reasoning operator or algorithm family.

## What This Enables

### Domain reasoning
Queries like warehouse SOPs or traffic exceptions can learn from patterns such as:
- `requires`
- `blocked by`
- `before`
- `if`

### Math word problems
Questions can be structured by patterns such as:
- state assertion
- part-whole
- capability
- temporal order
- conditional branching

### Olympiad proof search
The system can move toward:
- contradiction frames
- induction frames
- extremal frames
- invariant frames
- modular arithmetic frames

### Competitive programming
The system can move toward:
- problem statement to algorithm-family frame
- frame to data structure/operator selection
- frame to code-template generation

## Current Limits

The system is still early.
Important limitations remain:
- connector matching is still surface-based, not embedding-based
- concept slot extraction is still lightweight
- grammar priors are injected heuristically, not learned end-to-end
- the system does not yet refine grammar frames from solved vs failed outcomes
- olympiad and contest coverage are still narrow compared with real top-level performance

## Recommended Next Steps

1. Use solved vs failed cases to reweight logical patterns.
2. Connect logical patterns to relation extraction directly, not only to plan priors.
3. Add embedding-based clustering for connectors and paraphrases.
4. Learn proof/algorithm meta-operators from success traces.
5. Build domain-specific grammar packs for operations, math, olympiad, and competitive programming.

## How To Run

### Train logical grammar patterns from a corpus

```bash
.\.venv312\Scripts\python.exe tools/corpus/train_corpus.py ^
  --input examples\reasoning_corpus_ko.jsonl ^
  --mode heuristic ^
  --output data\reasoning_learning.json ^
  --store data\semop_memory.db ^
  --source grammar_demo
```

The output JSON now includes `logical_patterns`.

### Use those patterns during inference

```bash
.\.venv312\Scripts\python.exe app.py ^
  --mode heuristic ^
  --query "If the bag has no open access, check the zipper before inserting the book."
```

If the memory store contains a recent learning run for the selected source, the pipeline can inject:
- logical grammar warnings
- grammar hypotheses
- frame-style operator candidates
- plan priors

### Try the specialized math/programming flows

```bash
.\.venv312\Scripts\python.exe solve_olympiad.py --query "Prove that 1+2+...+n = n(n+1)/2 for all positive integers n."
.\.venv312\Scripts\python.exe solve_contest.py --query "Given an array and many range sum queries, output the sum from l to r for each query."
```


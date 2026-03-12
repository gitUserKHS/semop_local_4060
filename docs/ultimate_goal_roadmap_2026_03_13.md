# Ultimate Goal Roadmap (2026-03-13)

## Goal

Build a general operator-algebra intelligence system that can:
- parse text, documents, and images into a shared typed operator graph
- recall several structurally similar cases at once
- reason with reusable operators and learned compositions
- verify, repair, and reject weak answers through compiler-like checks
- improve from retained traces instead of relearning from scratch

## Current State

Implemented:
- unified parser priors for text/document/vision entry points
- multimodal graph fusion into the same `StructuredMeaningPipeline`
- analogy retrieval with learned ranking and verifier/planning hooks
- retained higher-order operator algebra
- typed operator compiler with legality and functor checks
- repair loop with learned action ranking
- retained repair-program memory with counterexample-conditioned repair synthesis
- unified benchmark artifacts for parser, analogy, retained algebra, repair policy, and retained repair programs

## Remaining Work To Reach The Architecture Limit

### 1. Learned parser dominance

Still needed:
- shift more of `heuristic_extractors.py` and `premise_explorer.py` into learned parsing
- train direct `sentence/document/image -> operator graph` supervision rather than only adding priors
- add parser confidence calibration and fallback thresholds

Concrete implementation path:
1. export gold operator graphs from runtime traces and reviewed cases
2. train a graph-slot predictor over hidden goals, premises, edges, and decompositions
3. route low-confidence cases through the heuristic+verifier fallback only

### 2. End-to-end multimodal grounding

Still needed:
- make visual evidence nodes participate in the same `GROUNDED_BY` chain as document evidence
- train cross-modal operator alignment instead of relying mainly on parser-side fusion
- benchmark real-image operator grounding with the same compiler/repair loop

Concrete implementation path:
1. add visual evidence nodes and grounding edges to the main graph
2. store verified vision-language alignment traces in memory
3. train retained multimodal functors from successful grounded traces

### 3. Learned operator retention beyond decomposition frequency

Still needed:
- retain operators and functors based on downstream transfer value, not only repeated verification
- learn when to reject a retained operator in a mismatched domain
- attach operator retirement and replacement logic

Concrete implementation path:
1. log retained-operator activation outcomes at runtime
2. train utility predictors from transfer success and compiler validity deltas
3. demote retained operators that repeatedly hurt composition score

### 4. Typed repair program search

Still needed:
- synthesize multi-step repair programs over typed compiler failures, not only select stored sequences
- learn when a repair should be rejected instead of applied
- feed successful repairs back into parser and operator training

Concrete implementation path:
1. treat `repair_applied:*` traces as repair programs with typed preconditions
2. train a repair-program proposer over compiler findings and graph state
3. add verifier gates for unsafe or low-utility repairs

### 5. Grounded explanation fidelity

Still needed:
- require every major answer claim to map to operator trace, evidence node, or verified memory record
- score explanation completeness against recovered graph state
- expose unsupported claims as compiler findings

Concrete implementation path:
1. emit claim-to-evidence alignments in response synthesis
2. add explanation compiler passes for unsupported claims
3. train explanation repair from failed grounded answers

### 6. Unified continuous learning loop

Still needed:
- merge parser learning, operator retention, analogy policy, repair policy, and repair-program retention into one online curriculum
- add review queues for failed transfer, failed grounding, and failed repair cases
- automate artifact refresh and benchmark gating

Concrete implementation path:
1. promote runtime traces into reviewable training bundles
2. retrain artifacts on accepted traces only
3. gate deployment on the unified benchmark scoreboard

## What Was Added In This Iteration

Implemented now:
- `src/semop/retained_repair_programs.py`
- graph supervision export plus calibrated parser-first graph-slot prediction in `src/semop/graph_supervision.py`, `src/semop/unified_parser.py`, and `src/semop/pipeline.py`
- retained repair-program training from successful repair traces
- counterexample-conditioned repair synthesis in `src/semop/operator_repair.py`
- explicit multimodal grounding edges and grounding-fidelity compiler checks in `src/semop/pipeline.py` and `src/semop/operator_runtime.py`
- unified artifact export for retained repair programs in `src/semop/unified_benchmark.py`

- retained multimodal alignment memory in `src/semop/multimodal_alignment_memory.py`

- retained-operator activation tracking and retirement in `src/semop/retained_operator_algebra.py`

- continuous-learning bundle export in `src/semop/continuous_learning.py`

## Engineering Rule

The next implementation should only be accepted if it improves at least one of:
- parser dominance
- multimodal grounding
- operator retention quality
- repair synthesis quality
- grounded explanation fidelity
- unified continuous learning

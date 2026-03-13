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

## Progress Estimate

This is a system-design progress estimate, not a claim that the research problem is solved.

- architecture coverage: about `80-85%`
  - shared graph, analogy, compiler, repair, grounding, benchmark, and continuous-learning loops exist in code
- policy and operationalization: about `85-90%`
  - review promotion, benchmark gate, slice baseline checks, severity weighting, and persistent corpora are wired in
- learned generalization: about `45-60%`
  - parser-first and retained memories exist, but broad data coverage and repeated retraining are still the main bottleneck
- end-goal readiness overall: about `60-70%`
  - the scaffold is strong; the main remaining gap is not missing subsystems but scaling supervised traces, multimodal alignment, and benchmarked transfer

What remains largest:
1. make learned parsing dominate heuristics on a larger share of inputs
2. improve real multimodal grounding quality on diverse visual evidence
3. close the loop from retained operator usage outcomes back into parser/operator retraining
4. enforce claim-level grounded explanation checking, not only graph-level grounding
5. run the continuous-learning loop repeatedly on larger reviewed corpora and prove benchmark gains across unseen slices
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

Progress now in code:
- `src/semop/operator_repair.py` synthesizes typed multi-step proposals for hidden-goal prerequisite repair, document grounding repair, visual grounding repair, claim-grounding repair, and functor repair.
- `src/semop/retained_repair_programs.py` now learns multi-step repair compositions from successful repair traces and keeps utility-weighted sequence records.
- `src/semop/repair_utility.py` now learns expected repair utility from post-repair benchmark-like deltas and lets runtime reject low-value repair actions or programs.
- `src/semop/unified_benchmark.py` now reinjects promoted review graphs and repair-trace graphs into parser, retained operator, repair-program, and repair-utility training.

Still needed:
- scale repair utility learning with larger reviewed corpora and later benchmark outcomes, not only current local delta signals
- add slice-specific utility gates so high-risk scenarios can reject repairs that look acceptable on global averages
- retire or demote repair programs that repeatedly fail continuous-learning benchmark gates over time

Concrete implementation path:
1. treat `repair_applied:*` and `repair_rejected:*` traces as repair-program and utility supervision with typed preconditions
2. accumulate slice-aware post-repair outcome deltas from benchmark runs into the repair-utility artifact
3. use continuous gate outcomes to retire low-utility repair programs and reinforce high-transfer ones

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




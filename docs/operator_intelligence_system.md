# Operator Intelligence System

## One-Line Definition

SemOp Local 4060 aims to become a logic-operator-based intelligence system that converts language, images, operational documents, and algorithmic problems into explicit world models, then reasons with reusable operators, memory, and verifiers instead of relying on direct text generation alone.

## Why This Is The Right Abstraction

The project is no longer just a warehouse copilot, a CP helper, or a visual demo. The common pattern across all active modules is:

1. Parse the input into structure.
2. Bind entities and relations into operator-friendly frames.
3. Retrieve reusable memories and prototypes.
4. Run a constrained reasoning path.
5. Verify, repair, or reject weak answers.

That is the core intelligence loop.

## Four Axes Of The System

### 1. Operator Learning

Goal:
- learn reusable frames, relation patterns, and operator families that survive beyond a single domain

Current implementation:
- logical word to frame induction in `src/semop/logical_grammar.py`
- hidden-premise extraction and goal-preservation modeling in `src/semop/premise_explorer.py` and `src/semop/premise_eval.py`
- operator algebra and functor-hypothesis induction in `src/semop/operator_algebra.py`
- family and hierarchy induction in `src/semop/emergent_operators.py`, `src/semop/operator_hierarchy.py`, and `src/semop/operator_registry.py`
- structural-first visual operator induction in `src/semop/vlso/structural_operators.py`
- visual operator prototype learning in `src/semop/vlso/operator_learning.py`
- few-shot and pseudo-label visual concept learning in `src/semop/vlso/concept_learning.py` and `src/semop/vlso/self_training.py`, now treated as secondary adaptation layers rather than the primary source of meaning

Target capability:
- the system should learn that `has`, `requires`, `before`, `inside`, `opening`, `access`, `reachable`, `connected`, `range_query`, or `transition` are not isolated tokens but reusable operator patterns

### 2. World Model

Goal:
- build explicit, typed, inspectable models of the current task or scene

Current implementation:
- structured meaning graphs for text in `src/semop/pipeline.py`
- CP DSL and hidden-structure parsing in `src/semop/contest_programmer.py`
- shared language-vision operator space in `src/semop/vlso/language_parser.py`, `src/semop/vlso/visual_parser.py`, and `src/semop/vlso/aligner.py`

Target capability:
- the same hidden structure should be visible whether the input is a question, an SOP, a contest statement, or an image

### 3. Memory

Goal:
- store reusable experiences at several compression levels

Current implementation:
- corpus/query/operator memory in `src/semop/corpus_store.py` and `src/semop/memory_retrieval.py`
- multi-analogy structural memory selection in `src/semop/memory_analogies.py`
- learned analogy-policy scoring in `src/semop/analogy_policy.py`
- CP episodic memory in `src/semop/cp_episode_store.py`
- VLSO concept memory, embedding memory, and hybrid memory in `src/semop/vlso/concept_memory.py`, `src/semop/vlso/embedding_store.py`, and `src/semop/vlso/hybrid_memory.py`

Target capability:
- a small reasoning core should query a large external memory instead of storing everything inside model weights
- the system should be able to recall several structurally similar cases at once, not only a single nearest neighbor
- memory retrieval should be re-ranked by hidden goals, premises, and operator structure rather than surface wording alone

### 4. Verifier

Goal:
- prevent confident nonsense by making outputs fail visible checks

Current implementation:
- operations KPI and audit checks in `src/semop/ops_kpi.py`
- CP compile/sample/random/counterexample loop in `src/semop/cp_validation.py` and `src/semop/cp_repair.py`
- hard-problem verification in `src/semop/hard_problem_engine.py`
- grounded visual QA in `src/semop/vlso/qa.py`
- analogy-conditioned premise and runtime verification in `src/semop/pipeline.py` and `src/semop/operator_runtime.py`
- operator composition compiler findings in `src/semop/operator_runtime.py`

Target capability:
- every answer path should be auditable, rejectable, and repairable

## What “Smarter AI” Means Here

The target is not merely a bigger model. The target is a system that:
- reconstructs hidden prerequisites instead of skipping them
- notices missing access, missing openings, and blocked transitions
- retrieves similar structural failures and avoids repeating them
- explains its answer through evidence and operator traces
- improves from a small amount of labeled data by learning more general operators

## Practical Interpretation

Today the project already has:
- a shared operator philosophy across language, CP, ops, and VLSO
- a structural-first VLSO path that tries to derive container/access/grasp operators from geometry and topology before semantic labels are applied
- local memory and prototype stores
- several verifier loops
- small-data learning paths

It does not yet have:
- a single learned end-to-end operator parser that dominates all heuristic paths
- broad benchmarked generalization across arbitrary images and arbitrary tasks
- a unified training/evaluation harness across all four axes

## Immediate Engineering Rule

New features should be justified by at least one of these questions:
- does this improve operator learning?
- does this improve world-model quality?
- does this improve reusable memory?
- does this improve verification?

If not, it is probably not part of the long-term intelligence core.

## Shared Multi-Agent Rule

All future work should follow the doctrine in `docs/multi_agent_operator_doctrine.md`: every domain must reduce to shared basis operators, higher operators must be normalized through composition, and retention is decided by verifier and transfer benches rather than ad hoc naming.




## 2026-03-13 Integration Update
- learned unified parser priors now live in `src/semop/unified_parser.py` and are applied inside `src/semop/pipeline.py` before induction.
- multimodal graph fusion now happens in `src/semop/pipeline.py`: `visual_input` is parsed through `src/semop/vlso/visual_parser.py`, merged into the main graph, and then sent through the same premise/analogy/compiler/context loop as language.
- document/PDF-text grounding is now structural rather than only textual: `source_context` is chunked into evidence nodes in `src/semop/pipeline.py`, and `src/semop/symbolic_reasoners.py` links grounded answers back to evidence nodes with `GROUNDED_BY` edges.
- retained higher-order operator algebra is trained in `src/semop/retained_operator_algebra.py` and injected back into runtime as reusable retained operators/functors.
- the compiler in `src/semop/operator_runtime.py` now performs typed verification, functor applicability checks, operator-composition legality checks, and emits counterexample-style repair hints.
- unified training/evaluation harness now exists in `src/semop/unified_benchmark.py`, combining artifact training with a single scoreboard over transfer, analogy, compiler, grounding, and repair metrics.
- `src/semop/operator_repair.py` now closes the loop after compiler findings: it can inject missing goal-preservation decompositions, bind missing prerequisite edges, reattach document context nodes, and recompile the graph.
- retained repair-program memory now lives in `src/semop/retained_repair_programs.py`: successful compiler-guided repairs are retained as reusable programs, and `src/semop/operator_repair.py` can synthesize action sequences from retained traces plus live counterexample hints.
- `src/semop/pipeline.py` now attaches explicit `question -> GROUNDED_BY -> evidence` edges for both document chunks and visual evidence nodes, so grounding is represented in the shared graph instead of staying implicit.
- `src/semop/operator_runtime.py` now runs a grounding-fidelity compiler pass that warns when document or visual reasoning lacks explicit evidence links.
- `src/semop/response_synthesizer.py` now surfaces grounded evidence lines directly in the compiled execution summary.
- `src/semop/graph_supervision.py` now exports runtime graphs into supervision JSONL, so parser learning can train on reviewed operator graphs instead of only token priors.
- `src/semop/unified_parser.py` now predicts hidden goals, required premises, and decompositions with a calibrated confidence score, and `src/semop/pipeline.py` can activate a parser-first bootstrap route when that confidence is high enough.
- `src/semop/multimodal_alignment_memory.py` now retains visual-language alignment traces and can project hidden goals, required premises, and functors back into new visual queries.
- `src/semop/retained_operator_algebra.py` now tracks activation success and can retire low-utility retained operators instead of reusing them forever.
- `src/semop/continuous_learning.py` now exports runtime traces, graph supervision, and SFT rows into a continuous-learning bundle for later retraining.
- `src/semop/operator_repair.py` now synthesizes typed multi-step repair proposals, applies reject gates for unsafe actions such as unsupported-claim trimming, and records `repair_rejected:*` decisions for downstream learning.
- `src/semop/continuous_learning.py` now exports applied and rejected repair-program traces so successful repairs feed back into later parser and operator training.


- `src/semop/repair_utility.py` now learns expected repair utility from post-repair benchmark-like deltas and lets runtime reject low-value repair actions or programs.
- `src/semop/unified_benchmark.py` now reinjects promoted review graphs and repair-trace graphs into unified parser, retained operator, repair-program, and repair-utility training.

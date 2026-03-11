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
- CP episodic memory in `src/semop/cp_episode_store.py`
- VLSO concept memory, embedding memory, and hybrid memory in `src/semop/vlso/concept_memory.py`, `src/semop/vlso/embedding_store.py`, and `src/semop/vlso/hybrid_memory.py`

Target capability:
- a small reasoning core should query a large external memory instead of storing everything inside model weights

### 4. Verifier

Goal:
- prevent confident nonsense by making outputs fail visible checks

Current implementation:
- operations KPI and audit checks in `src/semop/ops_kpi.py`
- CP compile/sample/random/counterexample loop in `src/semop/cp_validation.py` and `src/semop/cp_repair.py`
- hard-problem verification in `src/semop/hard_problem_engine.py`
- grounded visual QA in `src/semop/vlso/qa.py`

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

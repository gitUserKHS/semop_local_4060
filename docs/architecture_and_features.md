# SemOp Architecture And Current Feature Status

## Current Project State

SemOp Local 4060 is no longer just one pipeline. It currently contains four active tracks that share a structured-reasoning philosophy:
- structured meaning graph reasoning for language-heavy tasks
- operations copilot and audit workflows
- competitive-programming reasoning, validation, and repair
- VLSO, a shared operator space for language and visual inputs

As of the latest local validation in this workspace:
- `python -m unittest discover -s tests -v` passes
- total passing tests: `99`

## Capability Snapshot

### 1. Core structured reasoning

Current status:
- query -> graph extraction via heuristic or local LLM path
- operator induction, grammar hypothesis generation, and memory-prior injection
- logical word to concept-frame pattern learning for connectors such as `has`, `if`, `before`, `requires`, and `can`
- response synthesis into readable structured explanations

Primary modules:
- `src/semop/pipeline.py`
- `src/semop/logical_grammar.py`
- `src/semop/emergent_operators.py`
- `src/semop/operator_hierarchy.py`
- `src/semop/operator_registry.py`
- `src/semop/response_synthesizer.py`

### 2. Operations copilot

Current status:
- warehouse and SOP-oriented copilot CLI and GUI
- KPI scoring, audit trace, baseline comparison, and review queue
- feedback-rule learning from reviewed items

Primary modules:
- `src/semop/domain_copilot.py`
- `src/semop/ops_kpi.py`
- `src/semop/review_queue.py`
- `ops_copilot.py`
- `ops_copilot_gui.py`
- `tools/ops/learn_feedback_rules.py`
- `tools/eval/compare_ops_baseline.py`

### 3. Symbolic math and hard-problem reasoning

Current status:
- arithmetic parsing and direct numeric solving
- line-item, ratio, unit-rate, and basic equation handling
- document evidence recovery for text and simple table-like structures
- hard-problem engine with verification checks and logical-pattern weight updates
- olympiad-style proof search remains prototype-level, not production-grade

Primary modules:
- `src/semop/symbolic_arithmetic.py`
- `src/semop/symbolic_document.py`
- `src/semop/symbolic_reasoners.py`
- `src/semop/hard_problem_engine.py`
- `src/semop/olympiad_reasoner.py`

### 4. Competitive programming

Current status:
- contest statement -> hidden structure -> CP DSL / frame / algorithm family
- C++17 code generation with conservative syntax for local MinGW GCC 6.3.0
- validator with sample checks, brute-force/random checks for supported tasks, counterexample reporting, and failure typing
- repair loop with compile, WA, TLE, and fallback rewrite behavior
- episodic memory for solved and failed incidents
- dry-run and LoRA-ready parser training scaffolds for RTX 4060 8GB
- beginner-friendly CP GUI

Primary modules:
- `src/semop/contest_programmer.py`
- `src/semop/cp_validation.py`
- `src/semop/cp_repair.py`
- `src/semop/cp_episode_store.py`
- `src/semop/cp_training.py`
- `solve_contest.py`
- `cp_copilot_gui.py`
- `tools/cp/*`

### 5. VLSO: visual-language shared operator space

Current status:
- language parser into shared operator graph
- detector payload adapters for detections, segments, annotations, and instances
- raw-image parsing with mask cleanup, border-frame suppression, geometry hints, and weak object/part/affordance inference
- deep-first vision backbone path with DINOv2/OpenCLIP adapter slots and structural fallback
- visual embedding store for scene retrieval
- few-shot visual concept memory and prototype compression
- candidate-label builder, prototype trainer, and next-label recommender for sample-efficient learning
- grounded visual QA in structured or local-LLM answer modes
- manifest-driven visual data collection planner for public APIs

Primary modules:
- `src/semop/vlso/vision_backbones.py`
- `src/semop/vlso/image_preprocess.py`
- `src/semop/vlso/image_parser.py`
- `src/semop/vlso/detector_adapters.py`
- `src/semop/vlso/object_reasoner.py`
- `src/semop/vlso/geometry_reasoner.py`
- `src/semop/vlso/embedding_store.py`
- `src/semop/vlso/concept_memory.py`
- `src/semop/vlso/concept_learning.py`
- `src/semop/vlso/qa.py`
- `src/semop/vlso/data_collection.py`
- `vlso_demo.py`
- `tools/vlso/*`

## Architecture By Layer

### Extraction and parsing layer

Purpose:
- convert text, CP statements, SOP context, detector output, or raw images into structured intermediate representations

Representative components:
- `heuristic_extractors.py`
- `llm_client.py`
- `contest_programmer.py`
- `vlso/language_parser.py`
- `vlso/visual_parser.py`
- `vlso/image_parser.py`

### Memory layer

Purpose:
- store reusable reasoning artifacts, episodes, visual exemplars, and embeddings

Representative components:
- `corpus_store.py`
- `memory_retrieval.py`
- `cp_episode_store.py`
- `vlso/embedding_store.py`
- `vlso/concept_memory.py`

### Operator and grammar learning layer

Purpose:
- lift repeated patterns into reusable families, hierarchies, and grammar priors

Representative components:
- `logical_grammar.py`
- `emergent_operators.py`
- `operator_hierarchy.py`
- `operator_registry.py`
- `corpus_learning.py`

### Verification layer

Purpose:
- reject brittle or invalid outputs and attach evidence for repair or audit

Representative components:
- `validation.py`
- `hard_problem_engine.py`
- `cp_validation.py`
- `cp_repair.py`
- `ops_kpi.py`
- `review_queue.py`

### UX and CLI layer

Purpose:
- expose the system as runnable demos and PoC tools

Representative entry points:
- `app.py`
- `ops_copilot.py`
- `ops_copilot_gui.py`
- `solve_contest.py`
- `cp_copilot_gui.py`
- `vlso_demo.py`

## Current Strengths

- strong emphasis on explicit intermediate structure instead of pure text generation
- good local-first support on RTX 4060 8GB with small-model and symbolic-heavy flows
- competitive-programming path has real validation and repair, not just code generation
- VLSO path now supports raw images, detector outputs, concept memory, and grounded QA
- documentation and tooling already cover practical few-shot data loops

## Current Limits

### General reasoning
- still a prototype and not a fully learned, end-to-end general reasoner
- many graph extraction paths remain heuristic-heavy

### CP
- validator coverage is strong for implemented templates, but still narrow compared with the full contest landscape
- learned parser path exists as a scaffold; final quality still depends on actual training data and fine-tuning

### VLSO
- deep backbones improve retrieval and priors, but robust raw-image operator extraction is still not equivalent to a production detector/segmenter stack
- few-shot concept memory is efficient, but it does not replace broader labeled coverage
- general visual QA accuracy across arbitrary internet images is not solved yet

## Recommended Next Steps

1. Grow VLSO multi-object labels with 10-50 diverse scenes and keep using the prototype recommender.
2. Add actual detector or segmentation model outputs into the VLSO loop for harder scenes.
3. Train the CP learned parser and compare it against the heuristic parser on a larger held-out set.
4. Keep `README.md`, `docs/index.md`, and this file aligned whenever new tools are added.

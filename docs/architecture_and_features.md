# SemOp Architecture And Current Feature Status

## Current Project State

SemOp Local 4060 is no longer just one pipeline. It currently contains four active tracks that share a structured-reasoning philosophy:
- structured meaning graph reasoning for language-heavy tasks
- operations copilot and audit workflows
- competitive-programming reasoning, validation, and repair
- VLSO, a shared operator space for language and visual inputs

As of the latest local validation in this workspace:
- `python -m unittest discover -s tests -v` passes
- total passing tests: `129`

## Intelligence Axes

The project is now best understood through four persistent axes:
- `operator learning`: induce reusable logical, structural, and visual operators
- `world model`: build explicit typed representations instead of relying on direct answers
- `memory`: retain reusable exemplars, episodes, and compressed prototypes
- `verifier`: reject brittle outputs through checks, repair, and grounded evidence

The code-level map for these axes is captured in `src/semop/intelligence_map.py`, and the design intent is documented in `docs/operator_intelligence_system.md`, `docs/operator_intelligence_roadmap.md`, and `docs/operator_intelligence_execution_steps.md`.

## Capability Snapshot

### 1. Core structured reasoning

Current status:
- hidden-premise exploration and goal-preservation checks for queries with implicit goals or missing assumptions
- operator algebra layer that decomposes higher operators into simpler basis operators
- category-inspired functor hypotheses that align service goals, visual structure, and geometry frames
- operator induction, grammar hypothesis generation, and memory-prior injection
- logical word to concept-frame pattern learning for connectors such as `has`, `if`, `before`, `requires`, and `can`
- response synthesis into readable structured explanations

Primary modules:
- `src/semop/pipeline.py`
- `src/semop/premise_explorer.py`
- `src/semop/premise_eval.py`
- `src/semop/operator_algebra.py`
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
- geometry-heavy contest statements can be lifted into geometric-configuration frames and routed to a dedicated computational-geometry template
- C++17 code generation with conservative syntax for local MinGW GCC 6.3.0
- validator with sample checks, brute-force/random checks for supported tasks, counterexample reporting, failure typing, and verifier-aware reranking over multiple candidate solutions
- repair loop with compile, WA, TLE, and fallback rewrite behavior
- episodic memory for solved and failed incidents
- dry-run and LoRA-ready parser training scaffolds for RTX 4060 8GB, plus heuristic-vs-model comparison summaries
- manifest-driven labeled CP dataset download and normalization
- geometry-focused CP corpus bootstrap from URLs or Hugging Face datasets
- geometry-only CP parser eval-set builder from normalized labeled corpus rows
- built-in geometry template generator for starter held-out parser eval assets
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
- segmentation-aware structural grounding that preserves `part_of`, `part_of_confidence`, `structural_role`, and `segmentation_confidence` into downstream structural operators
- raw-image parsing with mask cleanup, border-frame suppression, geometry hints, and structural-first operator induction
- polygon-edge reasoning that derives `PARALLEL`, `PERPENDICULAR`, and `EQUAL_LENGTH` relations plus shape hypotheses such as triangle, rectangle, square, parallelogram, and quadrilateral
- deep-first vision backbone path with DINOv2/OpenCLIP adapter slots and structural fallback
- visual embedding store for scene retrieval
- structural-first visual operator induction that derives container, access-port, access-control, and grasp operators from geometry/topology primitives before semantic labels are considered
- few-shot visual concept memory and prototype compression as a secondary adaptation layer
- candidate-label builder, prototype trainer, next-label recommender, pseudo-label self-training loop, and review-priority scoring for sample-efficient active learning
- grounded visual QA in structured or local-LLM answer modes, plus before/after approved-store impact evaluation
- explicit operator decomposition recovery and functor-hypothesis recovery benchmark
- manifest-driven visual data collection planner for public APIs
- one-command geometry/access visual bootstrap for dry-run planning, record collection, approval, and staged downloads
- end-to-end geometry reasoning pipeline for candidates, pseudo labels, concept store, operator store, and grounded QA eval
- synthetic geometry scene generation for PNG + detector-style JSON bootstrap without external downloads
- whitelist-based download staging and Open Images bbox-to-detector JSONL ingest

Primary modules:
- `src/semop/vlso/vision_backbones.py`
- `src/semop/vlso/image_preprocess.py`
- `src/semop/vlso/image_parser.py`
- `src/semop/vlso/detector_adapters.py`
- `src/semop/vlso/object_reasoner.py`
- `src/semop/vlso/structural_operators.py`
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
- `premise_explorer.py`
- `premise_eval.py`
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
- `operator_algebra.py`
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
- VLSO path now supports raw images, detector outputs, structural operator induction from geometry/topology primitives, concept memory, grounded QA, concept-store comparison, cluster-level self-training review, and few-shot visual operator prototypes that learn reusable relation patterns from small image sets
- VLSO operator families now generalize beyond bag-only scenes toward drawer, door, bottle, box, and tool-like access structures through broader operator derivation rules
- starter held-out eval assets now exist for VLSO grounded QA, VLSO geometry QA, general CP parser benchmarking, and geometry-only CP parser benchmarking
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
- structural induction is now the primary VLSO reasoning path, but the raw-image parser and topology extraction are still too weak to guarantee fully general scene understanding
- few-shot concept memory is efficient, but it now acts as adaptation rather than the first source of semantics
- general visual QA accuracy across arbitrary internet images is not solved yet

## Recommended Next Steps

1. Grow VLSO multi-object labels with 10-50 diverse scenes and keep using the prototype recommender.
2. Add actual detector or segmentation model outputs into the VLSO loop for harder scenes.
3. Train the CP learned parser and compare it against the heuristic parser on a larger held-out set.
4. Keep `README.md`, `docs/index.md`, and this file aligned whenever new tools are added.


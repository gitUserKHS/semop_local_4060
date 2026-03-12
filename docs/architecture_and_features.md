# SemOp Architecture And Current Feature Status

## Current Project State

SemOp Local 4060 is no longer just one pipeline. It currently contains four active tracks that share a structured-reasoning philosophy:
- structured meaning graph reasoning for language-heavy tasks
- operations copilot and audit workflows
- competitive-programming reasoning, validation, and repair
- VLSO, a shared operator space for language and visual inputs

As of the latest local validation in this workspace:
- `python -m unittest discover -s tests -v` passes
- total passing tests: `177`

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
- three-stage hidden-premise engine with candidate retrieval, premise proposal, and premise validation
- hidden-premise exploration and goal-preservation checks for queries with implicit goals or missing assumptions
- SQLite-backed premise/operator memory for runtime reuse and later retrieval
- operator algebra layer that decomposes higher operators into simpler basis operators
- operator compiler/executor runtime that turns premise and operator-algebra outputs into a deterministic instruction stream and execution report
- category-inspired functor hypotheses that align service goals, visual structure, and geometry frames
- operator self-evolution loop that mines repeated decompositions into evolved operator proposals, scores utility, and retains promising higher operators
- operator proposal engine that lets model-side summarizers propose higher operators from repeated decomposition patterns and visual geometry signatures before the verifier decides retention
- visual-signal ablation bench for testing whether symmetry, closure, and axis-alignment signals actually improve retained operator quality
- proposal comparator that can contrast heuristic proposals with LLM-backed operator naming and summarization
- operator transfer benchmark for checking whether evolved operators survive held-out domains
- operator induction, grammar hypothesis generation, and memory-prior injection
- logical word to concept-frame pattern learning for connectors such as `has`, `if`, `before`, `requires`, and `can`
- response synthesis into readable structured explanations
- `tools/eval/evaluate_semop_stack.py` can now run a shared snapshot across hidden-premise, CP parser, and VLSO grounded evaluation, including operator-level premise support

Primary modules:
- `src/semop/pipeline.py`
- `src/semop/premise_explorer.py`
- `src/semop/premise_eval.py`
- `src/semop/operator_algebra.py`
- `src/semop/operator_evolution.py`
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
- geometry primitive backbone now derives symmetry, axis-alignment, closure, right-angle, and parallel-edge signatures before semantic labeling
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

Recent progress note:
- hidden-premise retrieval is now hybrid: rule + commonsense script + premise/script/operator memory + trainable script-compatibility scorer
- review-approved VLSO clusters can now seed premise/script/operator memory, not just concept stores
- operator support can now be evaluated at the premise level, not only by final answer correctness


- strong emphasis on explicit intermediate structure instead of pure text generation
- good local-first support on RTX 4060 8GB with small-model and symbolic-heavy flows
- competitive-programming path has real validation and repair, not just code generation
- VLSO path now supports raw images, detector outputs, structural operator induction from geometry/topology primitives, concept memory, grounded QA, concept-store comparison, cluster-level self-training review, and few-shot visual operator prototypes that learn reusable relation patterns from small image sets
- VLSO operator families now generalize beyond bag-only scenes toward drawer, door, bottle, box, and tool-like access structures through broader operator derivation rules
- starter held-out eval assets now exist for VLSO grounded QA, VLSO geometry QA, general CP parser benchmarking, and geometry-only CP parser benchmarking
- a real-image VLSO benchmark builder and reviewed-gold finalizer now exist for turning downloaded public images into a harder held-out QA set
- the CP LoRA experiment harness now supports separate held-out evaluation inputs so trained parsers can be compared against heuristic baselines on a different split
- documentation and tooling already cover practical few-shot data loops

## Current Limits

Operator self-evolution status:
- evolved operators are now mined, merged, retained, transfer-tested, and can be persisted across repeated iterations through the SQLite-backed self-evolution loop
- this is still a starter self-evolution loop, not a fully autonomous lifelong operator invention system
- the main bottleneck is still cross-domain transfer on larger held-out sets


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
- a starter real-image benchmark now exists at `examples/vlso_real_image_eval.jsonl`, and the current low grounded-QA score confirms that arbitrary internet-image grounding is still the main bottleneck
- general visual QA accuracy across arbitrary internet images is not solved yet

## Recommended Next Steps

1. Grow VLSO multi-object labels with 10-50 diverse scenes and keep using the prototype recommender.
2. Add actual detector or segmentation model outputs into the VLSO loop for harder scenes.
3. Train the CP learned parser and compare it against the heuristic parser on a larger held-out set.
4. Keep `README.md`, `docs/index.md`, and this file aligned whenever new tools are added.

## Progress Tracking

SemOp now has a research-facing progress estimator driven by actual evaluator outputs instead of manual percentages.
Use `tools/eval/evaluate_operator_intelligence_progress.py` to combine hidden-premise, CP parser, and VLSO grounded QA metrics into six axes:
- operator architecture
- premise reasoning
- shared world model
- CP structuring
- raw visual reasoning
- general operator transfer

The long-range completion plan is documented in `docs/operator_intelligence_to_100_plan.md`.

- access-family hidden goals now explicitly include box/pouch/suitcase/bin in addition to drawer/cabinet/bottle/jar, and premise support operators are injected from hidden-goal structure
- script compatibility scoring now blends lexical overlap with concept-family compatibility and harder sibling negatives

- the common evaluator now supports `cp_hidden_constraints` and `vlso_real_image` snapshots, so parser-first hidden-constraint evaluation and reviewed real-image VLSO evaluation can be tracked alongside the core starter benches
- a trainable compatibility experiment path now exists to compare hidden-premise metrics before and after a learned script-compatibility scorer is loaded


## Distillation and QLoRA

The project now includes a common teacher-trace export layer via `src/semop/distillation.py` and `tools/eval/export_teacher_traces.py`. Hidden-premise reasoning, CP structuring, and VLSO grounded QA can be exported into a unified JSONL/SFT format for QLoRA experiments.

The distillation exporter also covers `operator_proposal` and `operator_self_evolution` tasks so that operator invention loops can be distilled, not only hidden-premise and VLSO/CP outputs. A generic operator-learning curriculum builder and student-training scaffold now turn those teacher traces into train/val bundles and LoRA/QLoRA-ready SFT runs.

## Shared Doctrine

The repository now treats `docs/multi_agent_operator_doctrine.md` as the top-level engineering rule for all future agents and subprojects: basis operators first, composition and compiler alignment second, task-specific heuristics last.

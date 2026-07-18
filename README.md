# SemOp Local 4060

## 처음 실행하는 사람을 위한 30초 시작

코드나 typed DSL을 몰라도 된다. Windows에서는 저장소 루트의
`start_semop.bat`를 더블클릭하면 로컬 브라우저 화면이 열린다.

명령어로 실행하려면 다음 두 줄이면 된다.

```powershell
$env:PYTHONPATH = "src"
python -m semop.beginner_web
```

화면에서 `언어 조건`, `수학식`, `색상 비전` 중 하나를 고르고 예제 버튼을 누른 뒤
`검증하기`를 누르면 된다. 외부 런타임 라이브러리나 LLM API는 필요하지 않다.

현재 쉬운 화면의 범위는 통제된 목표·필요조건 문장, 정확한 계산식·일차방정식,
작은 색상 격자다. 일반 자유 대화나 자연 사진 이해로 오해하지 않도록 각 결과에
검증 범위와 현실 증거 여부를 함께 표시한다. 자세한 그림 설명은
[한국어 첫걸음 가이드](docs/beginner_guide_ko.md)에 있다.

SemOp Local 4060 is a local prototype for structured reasoning and domain copilot experiments.

The current product direction is not a general chatbot. It is a Korean warehouse and field-operations SOP copilot that:
- reads SOPs, manuals, and exception-handling documents
- reconstructs hidden prerequisites, blockers, and alternatives
- reduces unsafe or non-executable answers
- leaves an audit trace and operational KPI summary
- compares itself against configurable baselines such as plain RAG

At the research level, the longer-term target is broader: learn how logical words such as `has`, `is`, `requires`, `if`, `before`, and `can` bind to concept frames, then reuse those learned grammar priors during reasoning across domain QA, math, olympiad proof search, competitive programming, and VLSO world-model reasoning.

The architectural target is a logical-operator-based intelligence system organized around four axes: operator learning, world-model construction, reusable memory, and verifier loops.
The central hypothesis is combinatorial: a small shared controller should learn to assemble many typed operator programs, while language, mathematics, and vision enter through domain adapters and every claimed result remains executor-verifiable.
The resource doctrine is CPU-first and sample-efficient: the symbolic core should run offline on an ordinary PC, while small local models and RTX 4060-class GPUs remain optional parsing, perception, and training accelerators.
A verifier-first typed operator core now runs beside the legacy runtime. It turns domain inputs into immutable typed facts, uses goal-relevant monotonic agenda chaining instead of enumerating fact subsets, slices the first supporting operator DAG, and replays every successful proof before returning it. The pipeline defaults to `shadow`: legacy output remains user-facing while typed proof, timing, allocation, and expansion measurements are written to the audit trace.
A shared typed grounding boundary now records language, math, and vision inputs as candidate, authority decision, fact, and immutable trace. Neural and heuristic producers can only propose; deterministic verifiers and explicit human reviews create the accept/reject examples used by later grounding-policy learning.
A dependency-free sparse grounding head now learns `ACCEPT/REJECT/ABSTAIN` from those independently verified examples. It shares typed and sensor-contract features across language, math, and vision, forces abstention on unseen sensor contracts, replays old labels during online updates, and promotes a new generation only after an untouched validation gate. Even an accepted prediction remains `PROPOSED` until a separate verifier or human promotes it.
A candidate-level semantic bridge now replaces the artificial sensor signal with controlled raw requirement text, exact expressions, and RGB rasters passed through the production adapters. Exact candidate reviews bind the case, typed atom, label, and human attestation to one digest. In the fixed repeated-template run, 20 labels per domain reach 91.7% completion with 344 parameters and zero false accepts; this is not a structural holdout result. A newer structural ablation reaches 60.0% completion at 20 labels and 72.5% at 100 with the full profile. Removing surface text changes neither result, while removing only the shared target-relative support margin already causes vision false accepts and fail-closed rollback. Human semantic gold remains at zero and is reported as `not_evaluated`.
A dependency-free raster adapter now adds a narrow real-pixel path: it detects small color components, verifies exact bounding-box or touching relations, keeps centroid-only guesses as `proposed`, and sends the resulting facts through the same operator proof replay.
A new hidden-premise layer now sits between surface parsing and later reasoning so the system can recover implicit goals and prerequisites before giving advice.
The current refactor direction is premise-first: candidate retrieval, premise proposal, and premise validation now precede later answer selection, CP code generation, and cross-modal alignment.
An operator-algebra layer now also records how higher operators decompose into simpler basis operators and stores category-style functor hypotheses for cross-modal alignment.
The current premise-first refactor is implemented end-to-end: hidden-premise candidate retrieval, proposal, validation, SQLite premise/operator memory, parser-first CP evaluation, and a shared evaluator snapshot are all wired into the codebase.
An operator self-evolution loop is now also present: repeated higher-operator decompositions can be mined into evolved operator proposals, utility-scored, retained, and tested on a small cross-domain transfer benchmark.
A new operator-proposal engine now sits in front of that loop: repeated decomposition and geometry/topology patterns are summarized into model-proposed higher operators, then normalized, merged, and passed to the verifier and transfer bench instead of being accepted directly.

## What You Can Run Today

### Typed Operator Core v1

```powershell
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -p "test_typed_operator_*.py" -v
python tools/eval/evaluate_low_resource_transfer.py
python tools/eval/evaluate_low_resource_transfer.py --suite language-math-vision
python tools/eval/evaluate_low_resource_transfer.py --suite composed-v4
python tools/eval/evaluate_typed_self_learning.py
python tools/eval/evaluate_semantic_flow_self_learning.py
python tools/eval/evaluate_active_macro_learning.py
python tools/eval/evaluate_hierarchical_self_learning.py
python tools/eval/evaluate_raw_grounded_self_learning.py
python tools/eval/evaluate_semantic_benchmark.py
python tools/eval/evaluate_semantic_benchmark.py --gate-semantic-correctness 1.0 --gate-min-gold 3 --gate-min-gold-per-domain 1 --gate-domains language,math,vision
python tools/eval/evaluate_grounding_self_learning.py --checkpoint-root artifacts/grounding_self_learning --output artifacts/grounding_self_learning/report.json
python tools/eval/evaluate_semantic_grounding_learning.py --checkpoint-root artifacts/semantic_grounding_learning --output artifacts/semantic_grounding_learning/report.json
python tools/eval/evaluate_semantic_grounding_ablation.py --output artifacts/semantic_grounding_ablation_v1.json
python tools/eval/evaluate_semantic_grounding_curriculum.py --require-pass --output artifacts/semantic_grounding_curriculum.json
python tools/eval/review_semantic_grounding.py --case-id language-ready-two-requirements
python tools/eval/review_typed_experience.py --db artifacts/experience/typed-experience.db stats
python tools/eval/review_typed_experience.py --db artifacts/experience/typed-experience.db list --status pending
python tools/eval/run_lodo_controller_experiment.py --output-dir artifacts/lodo_debug
python examples/typed_multidomain_demo.py
python examples/typed_compositional_v2_demo.py
python examples/typed_cross_domain_scene_demo.py
python examples/typed_frontier_judge_demo.py
python examples/typed_self_learning_demo.py --output artifacts/self_learning_run_01 --examples-per-structure 3
python examples/typed_self_discovery_demo.py --output artifacts/self_discovery_run_01 --examples-per-structure 1
python examples/typed_raw_self_learning_demo.py
python examples/typed_raster_vision_demo.py
```

`typed_multidomain_demo.py` sends a Korean premise sentence, an exact arithmetic
expression, and a verified vision scene through the same runtime and policy.
`typed_compositional_v2_demo.py` exercises Horn-style language inheritance, exact
linear equations, and raster shape/count/area operators without external runtime
dependencies.
`typed_cross_domain_scene_demo.py` runs one replayable vision -> exact comparison ->
language classification program instead of solving the three domains independently.
`typed_frontier_judge_demo.py` shows the frontier-LLM boundary: judge output remains
proposed until typed execution and proof replay admit the program to the trace corpus.
`typed_self_learning_demo.py` closes that loop: an active curriculum balances six
real language/math/vision adapter structures, replay-verified traces train a tiny
sparse action policy, and three entirely held-out capability compositions plus
negative controls gate promotion before an atomic hash-checked checkpoint is written.
`typed_self_discovery_demo.py` expands the pool without an LLM: verifier-backed
2/3-domain composition, bounded verified suffix scaffolds, and support-ablation
counterfactuals create new tasks before the same held-out promotion gate runs.
The typed kernel also exposes `VerifiedRuleDiscovery`, a separate bounded path that
anti-unifies repeated unsolved tasks into executable flat Horn-rule candidates.
Only rules that newly solve digest-recorded, human-reviewed validation and untouched
held-out positives while preserving matched negatives enter a hash-checked library;
activation remains explicit and every use is proof-replayed.
`VerifiedRuleLearningLoop` then combines those rules with the incumbent library and
uses a fourth final joint holdout to catch unsafe rule interactions. Promotion writes
an atomic hash-checked envelope containing the library and exact joint-review
certificate; rejection leaves the incumbent artifact untouched.
`TypedExperienceCollector` now connects this rule path to real language, math, and
raster-vision executions. Noteworthy runs enter an append-audited SQLite queue;
user or frontier-judge labels remain proposals until a separate digest-bound human
review. Approved cases are assigned by a precommitted four-way hash partition,
grounded again, checked for cross-split semantic leakage, and only then enter joint
rule promotion. A promoted library is activated explicitly through
`UnifiedTypedReasoner(augmenters=(library,))` and every derived result is replayed.
`evaluate_semantic_flow_self_learning.py` trains only on short verified
vision-to-math-to-language flows, then gates promotion on new phrasing, larger images,
reused measurements, deeper operator programs, and sound negative controls.
`evaluate_active_macro_learning.py` learns repeated primitive programs from verified
language, math, and raster-vision traces, validates them on a separate split, and uses
only promoted programs as search priors on held-out groundings. The kernel still
executes and replays every primitive step; a macro cannot inject a fact or bypass a
guard.
`evaluate_hierarchical_self_learning.py` trains and promotes a tiny shared-family
controller on language only, verifies zero-shot family transfer to math and vision,
and independently promotes procedural memory. The same evaluator supports sparse,
29K diagnostic recurrent, and full 5.84M recurrent profiles. It then evaluates their
combination on a final joint holdout that neither component used for selection.
Registry-specific macro activation and controller inference are packed into one
portable, hash-checked brain artifact.
`evaluate_raw_grounded_self_learning.py` removes the prebuilt-IR assumption from the
learning boundary. It grounds raw requirement text, exact math strings, and RGB pixel
problems through the production adapters, rejects failed or overlapping splits, and
trains on language only before evaluating untouched math and vision transfer.
`evaluate_semantic_benchmark.py` runs digest-bound language, math, and raster-vision
near misses through the same adapters. Seed labels remain `curated_unreviewed`; only
an exact, separately attested review can contribute to `semantic_correctness`.

Optional controller training:

```powershell
python -m pip install -r requirements-train.txt
python tools/train/train_tiny_controller.py --output artifacts/tiny_debug.npz --examples-per-domain 1 --epochs 1 --debug-small
```

- `src/semop/kernel/`: immutable typed IR, operators, forward search, proof replay, adapters, traces, and verifier-gated MDL macro activation
- `src/semop/kernel/experience.py`: audited raw input grounding, split fingerprints, bounded hard negatives, and the end-to-end raw self-learning API
- `src/semop/kernel/experience_queue.py`: append-audited LMV execution queue, digest-bound reviews, and deterministic four-way partitioning
- `src/semop/kernel/experience_collection.py`: production failure collection, reviewed-corpus grounding, leakage audit, and joint rule-learning bridge
- `src/semop/kernel/grounding.py`: shared LMV candidate authority, proposal promotion, hard-negative lineage, and verified learning examples
- `src/semop/kernel/semantic_grounding*.py`: exact candidate review/learning bridge, input-only operator features, and controlled semantic case generation
- `src/semop/tiny_controller/grounding_*.py`: anonymized LMV features, sparse selective policy, verified replay, online promotion, and hash-checked artifacts
- `src/semop/tiny_controller/`: 5.84M-parameter default policy architecture; NumPy inference and isolated PyTorch training
- `docs/typed_operator_core.md`: execution contract and extension workflow
- `docs/trust_provenance_and_metrics.md`: assertion/evidence/logical provenance, conditional proofs, honest metric names, and CI gates
- `docs/typed_grounding_boundary.md`: shared language/math/vision grounding trace, authority rules, review promotion, and research basis
- `docs/sparse_grounding_self_learning.md`: shared accept/reject/abstain policy, continual replay, risk-coverage gates, evaluation, and honest limits
- `docs/semantic_grounding_self_learning.md`: exact candidate reviews and real-adapter 0/5/20/100 semantic grounding evaluation
- `docs/semantic_data_acquisition.md`: pinned public sources, audited downloads, and generated/model-proposed data trust rules
- `docs/operator_boundary_curriculum.md`: verified synthetic/public/model data authority, typed decision features, feature-novel selection, and honest low-shot results
- `docs/lmv_semantic_benchmark.md`: digest-bound review workflow and authority-separated three-domain semantic evaluation
- `docs/language_math_vision_typed_runtime.md`: direct three-domain API, trust boundary, and current limits
- `docs/language_text_adapter.md`: high-precision Korean/English claims, proposed fallback, and contradiction handling
- `docs/typed_compositional_extensions.md`: v2 language logic, exact equations, raster quantification, and controller scoring contract
- `docs/composed_operator_runtime.md`: v4 registry composition, conjunctive scene conditions, operator frontier, and verified vision-math-language programs
- `docs/typed_dataflow.md`: reusable numeric measurement-to-condition-to-conclusion compiler and semantic-flow holdout
- `docs/frontier_llm_judge.md`: safe frontier-LLM teacher/judge roles and mandatory verifier/replay boundary
- `docs/verifier_gated_self_learning.md`: active three-domain curriculum, structural holdout promotion, rollback, and checkpoints
- `docs/raw_grounded_self_learning.md`: raw language/math/pixel grounding, leakage audit, failure handling, and measured self-learning transfer
- `docs/active_macro_learning.md`: primitive-expanded procedural memory, schema pinning, promotion gates, and rollback
- `docs/hierarchical_operator_brain.md`: shared controller plus procedural memory, independent split gates, and portable brain artifacts
- `docs/self_discovered_curriculum.md`: verifier-backed task composition, failure signals, counterfactual generation, lineage, and bounded self-discovery
- `docs/verified_rule_discovery.md`: bounded typed Horn induction, joint-library promotion, review provenance, held-out falsification, and atomic rollback
- `docs/online_verified_self_learning.md`: persistent LMV runtime experience, independent review, four-way split, rule promotion, and explicit activation
- `docs/raster_vision.md`: dependency-free raster input, pixel trust boundary, and learned-detector extension point
- `docs/tiny_controller.md`: architecture, losses, data limits, and artifact format
- `docs/low_resource_transfer_evaluation.md`: three-domain A/B benchmark and promotion gates
- `tools/eval/evaluate_typed_self_learning.py`: machine-readable structural-transfer, active-selection, promotion, and resource gates
- `tools/eval/evaluate_typed_task_discovery.py`: machine-readable task novelty, replay, depth extrapolation, and self-discovery transfer gates
- `tools/eval/evaluate_semantic_flow_self_learning.py`: machine-readable cross-domain semantic-flow transfer and resource gates
- `tools/eval/evaluate_active_macro_learning.py`: machine-readable three-domain macro induction, primitive replay, and resource gates
- `tools/eval/evaluate_hierarchical_self_learning.py`: machine-readable controller/macro/joint ablation and family-transfer gates
- `tools/eval/evaluate_raw_grounded_self_learning.py`: language-only raw training followed by untouched raw math/pixel transfer gates
- `tools/eval/evaluate_semantic_grounding_learning.py`: controlled raw LMV candidate learning, rollback, untouched test, and human-review audit
- `tools/eval/evaluate_semantic_grounding_ablation.py`: surface/margin shortcut audit against unseen compositions and raster structures
- `tools/eval/evaluate_semantic_grounding_curriculum.py`: prefix-vs-feature-novel low-resource A/B with fail-closed development extrapolation gates
- `tools/eval/review_semantic_grounding.py`: inspect and attest one exact typed candidate label
- `tools/eval/review_typed_experience.py`: inspect, attest, review, and export persistent typed runtime experience
- `docs/lodo_controller_experiment.md`: leakage-controlled language/math/vision holdout training and evaluation

No trained controller artifact is committed yet. An earlier full 5.84M synthetic
leave-one-domain-out snapshot passed the expansion gate, and the v3 frontier-aware
contract passes a fresh 29K training/export diagnostic. The hierarchical split now
also trains the current 5.84M controller from three language traces and verifies its
math/vision transfer plus macro composition. The broader frontier-aware 5.84M LODO
rerun and verified human-reviewed 20/100-shot gates remain unevaluated. The same full
controller now also trains from three raw language examples through the public
grounding boundary and reduces untouched raw math and pixel expansions from 6 to 2
in each domain. This is a controlled structural-transfer result, not evidence of
open-domain understanding. With the digest-bound LMV benchmark and verified typed
online reviewed-learning and sparse grounding self-learning milestones, local
validation now passes 306 typed-operator tests plus 24 subtests and all 673
repository tests; `shadow`
remains the default.

- `app.py`
  - research-oriented structured reasoning CLI
- `ops_copilot.py`
  - product-style warehouse/operations copilot CLI
- `semop_easy_gui.py`
  - one-page beginner GUI for ops, CP, image QA, concept-store comparison, cluster review, approved-cluster retraining, VLSO impact evaluation, CP parser comparison, labeled dataset download, and VLSO image download staging
- `ops_copilot_gui.py`
  - local browser GUI for testing queries, baselines, and review queue items
- `solve_olympiad.py`
  - symbolic proof-search CLI for olympiad-style math questions
- `solve_contest.py`
  - competitive programming approach + C++17 template generator
- `cp_copilot_gui.py`
  - beginner-friendly local browser GUI for CP analysis, incident ingest, and episode memory
- `solve_hard_problem.py`
  - structured hard-problem solving, verification, and pattern-weight learning
- `vlso_demo.py`
  - Vision-Language Semantic Operators demo for shared language and visual operator reasoning
- `tools/vlso/index_vlso_visual_memory.py`
  - index structured visual observations or image-backed visual memories into a local VLSO embedding store
- `tools/vlso/index_visual_concepts.py`
  - index few-shot visual concept exemplars into a local VLSO concept-memory store
- `tools/vlso/build_visual_concept_candidates.py`
  - build multi-object concept-label candidates for manual few-shot labeling
- `tools/vlso/train_visual_concepts.py`
  - compress labeled concept examples into prototype memory for sample-efficient VLSO learning
- `tools/vlso/train_visual_operators.py`
  - learn higher-level visual operator prototypes such as container-body, opening-control, and attached-grasp from a few labeled images
- `src/semop/operator_proposal.py`
  - propose higher operators from repeated decomposition patterns and geometry/topology signatures before self-evolution validation
- `tools/vlso/run_geometry_reasoning_pipeline.py`
  - end-to-end geometry visual pipeline for candidates, pseudo labels, concept store, operator store, and grounded QA eval and structural operator recovery eval
- `tools/vlso/generate_geometry_dataset.py`
  - generate synthetic geometry images, paired detector-style JSON payloads, and a starter geometry QA eval set
  - retrain approved VLSO clusters into fresh concept/operator stores
- `tools/vlso/recommend_visual_labels.py`
  - rank the next most informative targets to label based on novelty and uncertainty
- `tools/eval/compare_ops_baseline.py`
  - SemOp vs baseline benchmark on labeled cases
- `tools/eval/evaluate_ops_kpis.py`
  - KPI averages on operations case sets
- `tools/eval/evaluate_vlso_grounded_qa.py`
  - grounded VLSO QA evaluator on image-or-observation jsonl cases
- `tools/eval/evaluate_vlso_review_impact.py`
  - compare grounded QA before and after approved-cluster retraining
- `tools/eval/evaluate_cp_parser.py`
  - evaluate heuristic, learned, or side-by-side CP parsers on DSL/frame labels
- `tools/eval/evaluate_hidden_premises.py`
  - evaluate hidden-goal recovery, critical premise recall, unsupported-premise precision, and goal-preservation checks
- `tools/eval/evaluate_semop_stack.py`
  - run a common evaluation snapshot across hidden premises, CP parser structure, and VLSO grounded QA
- `tools/eval/evaluate_operator_intelligence_progress.py`
  - convert the common evaluation snapshot into progress estimates for operator architecture, premise reasoning, world-model quality, and cross-domain transfer
- `tools/eval/evaluate_operator_algebra.py`
  - evaluate operator decomposition recovery and functor-hypothesis recovery
- `tools/eval/evaluate_operator_transfer.py`
  - evaluate evolved operators on a starter cross-domain transfer benchmark
- `tools/eval/evaluate_visual_signal_operator_impact.py`
  - compare retained operators before and after ablating symmetry, closure, and axis-alignment geometry signals
- `tools/eval/compare_operator_proposals.py`
  - compare heuristic and LLM-backed operator proposal engines on the same graph set
- `docs/operator_algebra_and_functors.md`
  - explain operator decomposition and functor-hypothesis alignment
- `tools/ops/learn_feedback_rules.py`
  - converts resolved review items into reusable feedback rules

## Recommended Environment

This repository has already been tested with a Python 3.12 virtual environment and GPU PyTorch.

Known working environment:
- Python `3.12`
- virtual environment: `.venv312`
- torch `2.10.0+cu128`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 4060`

Run commands with the script path shown in the docs, for example:

```bash
.\.venv312\Scripts\python.exe tools\cp\train_cp_parser.py ...
```

Current code layout is documented in:
- `docs/project_structure.md`
- `docs/pdf_direction_and_plan_2026_03_10.md`
- `docs/operator_intelligence_system.md`
- `docs/operator_intelligence_roadmap.md` and `docs/operator_intelligence_execution_steps.md` and `docs/jepa_relevance_and_integration.md`
- `docs/operator_intelligence_to_100_plan.md`

Refresh it after structural changes with:

```bash
.\.venv312\Scripts\python.exe tools\maintenance\update_code_structure_docs.py
```

## Quick Start

### 1. Open the easiest GUI

```bash
.\.venv312\Scripts\python.exe semop_easy_gui.py
```

Then open `http://127.0.0.1:8770`.

From the easy GUI you can click through:
- download labeled CP datasets
- build a VLSO image-collection plan
- generate an object-family manifest for bag, box, drawer, door, bottle, tool, cabinet, suitcase, jar, bin, and pouch
- run a family-target batch collector that writes manifest, records, approved, and download-manifest files automatically
- preview public-image cards and approve downloads
- run downloaded-image learning
- label downloaded images with a review queue
- retrain concept and operator stores from approved labels only
- generate synthetic geometry images
- generate CP geometry eval/train starter sets
- run CP LoRA training or resume the latest checkpoint from the easy GUI

### 2. Run the operations copilot on a single SOP question

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --query "The aisle is blocked and approval is still missing. What should I do?" ^
  --context-file examples\customer_sop_sample.md
```

### 3. Open the full operations GUI

```bash
.\.venv312\Scripts\python.exe ops_copilot_gui.py ^
  --review-queue data\ops_review_queue.db ^
  --baseline-config examples\customer_baseline_config.json
```

Then open `http://127.0.0.1:8765`.

### 4. Try the olympiad proof-search prototype

```bash
.\.venv312\Scripts\python.exe solve_olympiad.py ^
  --query "Prove that the sum of two odd integers is even."
```

### 4B. Try the VLSO prototype with structured observations

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --query "How do I put a book into a bag?" ^
  --visual-json examples\vlso\bag_closed_observation.json
```

### 4C. Try the VLSO prototype in deep mode on a raw image

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "What shapes are visible here?" ^
  --image-path data\scene.png ^
  --format json
```

If a local DINOv2 checkpoint exists under `models\vision\dinov2\...`, it is auto-resolved.
If not, SemOp falls back to `token_geometry_v1` and keeps the raw-image shape parser active.
This is the intended current architecture: `deep-first + structural fallback`, not `deep-only`.

You can also load learned affordance weights and request a grounded answer directly:

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "How can I access the bag opening?" ^
  --image-path data\vlso_samples\backpack_public_domain.jpg ^
  --affordance-weights data\vlso_samples\trained_affordance_weights.json ^
  --answer-mode structured ^
  --format json
```

For local Qwen-style answer generation on top of the grounded world model, switch `--answer-mode llm` and optionally override `--answer-model-id`.

You can also ground new images through a segmentation-style detector payload and few-shot concept memory:

```bash
python tools\vlso\index_visual_concepts.py ^
  --labels examples\vlso_visual_concepts_template.jsonl ^
  --store data\vlso_visual_prototypes.db
```

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "What objects are visible here?" ^
  --image-path data\vlso_samples\backpack_public_domain.jpg ^
  --concept-store data\vlso_visual_prototypes.db ^
  --affordance-weights data\vlso_samples\trained_affordance_weights.json ^
  --answer-mode structured ^
  --format json
```

Segmentation-style detector JSON is accepted through `--detector-json`, including `annotations`, `segments`, and `instances` payloads.
Open Images box annotations can be converted into this detector JSONL shape with `tools\vlso\ingest_open_images_annotations.py`.

For sample-efficient learning, build candidate rows, train compact concept prototypes, and label only the most novel or uncertain targets next. The full loop is documented in `docs/vlso_data_collection_guide.md`. Public API and dataset options are summarized in `docs/data_collection_api_research.md`.
The raw-image path now adds mask refinement, dominant border-frame suppression, and lightweight object/part/affordance inference before graph construction.
Small public tuning samples are stored under `data\vlso_samples\` and listed in `data\vlso_samples\SOURCES.md`.
A collection guide for growing this set to 10-20 images is in `docs\vlso_data_collection_guide.md`.
Label-candidate generation and weight re-estimation CLIs are `tools\vlso\build_affordance_label_candidates.py` and `tools\vlso\train_affordance_classifier.py`. Unsupervised or weakly supervised prototype bootstrapping is available through `tools\vlso\self_train_visual_concepts.py`. You can compare a hand-labeled concept store against a pseudo-labeled store with `vlso_demo.py --compare-concept-store ...`, and review cluster summaries in `semop_easy_gui.py`. Manifest-driven API collection planning is available through `tools\vlso\collect_visual_data.py`, and whitelist/download staging is available through `tools\vlso\prepare_visual_downloads.py`.

### 3D. Try the VLSO prototype with detector output or an explicit local backbone

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "What geometric structure is visible here?" ^
  --detector-json examples\vlso\detector_output_example.json
```

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "What geometric structure is visible here?" ^
  --image-path path\to\scene.png ^
  --vision-backbone dinov2_adapter ^
  --vision-model-path models\vision\dinov2\dinov2-small
```

### 4. Generate a contest-programming approach, validate it, and store the episode

```bash
.\.venv312\Scripts\python.exe solve_contest.py ^
  --query "Given a weighted graph with N cities and M roads, answer the shortest path from city 1 to all cities." ^
  --episode-store data\cp_episodes.db
```

Competitive-programming knowledge is stored in:
- `data\knowledge\cp_knowledge.json`

Solved contest episodes are stored in:
- `data\cp_episodes.db`

Those episodes are later reused as episodic retrieval and reranking priors for new contest statements.
You can also download and normalize labeled CP datasets with `tools\cp\download_cp_labeled_datasets.py`.

Geometry data bootstrap shortcuts:

```bash
.\.venv312\Scripts\python.exe tools\vlso\bootstrap_geometry_visual_data.py ^
  --workspace data\vlso_geometry_bootstrap
```

```bash
.\.venv312\Scripts\python.exe tools\cp\bootstrap_geometry_corpus.py ^
  --manifest examples\cp_geometry_labeled_manifest.json ^
  --download-root data\cp_geometry_downloads ^
  --output data\cp_geometry_labeled.jsonl
```

The VLSO bootstrap writes a geometry/access preset manifest automatically and can later be re-run with `--execute-collect` or `--execute-downloads`.
The CP bootstrap can consume normal URLs or Hugging Face datasets through the manifest. You can also generate synthetic geometry scenes and eval assets locally with `tools\vlso\generate_geometry_dataset.py`.

Full geometry reasoning bootstrap:

```bash
.\.venv312\Scripts\python.exe tools\vlso\run_geometry_reasoning_pipeline.py ^
  --inputs data\vlso_samples ^
  --workspace data\vlso_geometry_pipeline ^
  --eval-mode heuristic ^
  --answer-mode structured
```

Build a geometry-only CP parser eval set from normalized labeled corpus rows:

```bash
.\.venv312\Scripts\python.exe tools\cp\build_geometry_parser_eval.py ^
  --input data\cp_geometry_labeled.jsonl ^
  --output examples\cp_geometry_parser_eval.jsonl
```

Generate a starter geometry-only CP eval set directly from built-in templates:

```bash
.\.venv312\Scripts\python.exe tools\cp\generate_geometry_eval_templates.py ^
  --output examples\cp_geometry_parser_eval.jsonl
```

### 4B. Open the CP GUI

```bash
.\.venv312\Scripts\python.exe cp_copilot_gui.py ^
  --episode-store data\cp_episodes.db
```

Then open `http://127.0.0.1:8787`.

### 4C. Ingest real WA/TLE/editorial incidents into episodic memory

```bash
.\.venv312\Scripts\python.exe tools/cp/ingest_cp_episodes.py ^
  --inputs examples\cp_incident_cases.jsonl ^
  --store data\cp_episodes.db
```

### 5. Prepare a CP corpus, then build DSL and SFT datasets

```bash
.\.venv312\Scripts\python.exe tools/cp/prepare_cp_corpus.py ^
  --inputs examples\cp_corpus_inputs examples\cp_statement_seeds.jsonl ^
  --output examples\cp_statement_corpus_expanded.jsonl

.\.venv312\Scripts\python.exe tools/cp/build_cp_dsl_dataset.py ^
  --inputs examples\cp_corpus_inputs examples\cp_statement_seeds.jsonl ^
  --output examples\cp_dsl_expanded_dataset.jsonl ^
  --corpus-output examples\cp_statement_corpus_expanded.jsonl ^
  --sft-output examples\cp_dsl_expanded_sft.jsonl
```

### 6. Build a train/val bundle, then dry-run or train a small CP parser with optional LoRA

```bash
.\.venv312\Scripts\python.exe tools/cp/build_cp_training_bundle.py ^
  --inputs examples\cp_dsl_expanded_dataset.jsonl ^
  --episode-store data\cp_episodes.db ^
  --train-output data\cp_train.jsonl ^
  --val-output data\cp_val.jsonl ^
  --train-sft-output data\cp_train_sft.jsonl ^
  --val-sft-output data\cp_val_sft.jsonl
```

```bash
.\.venv312\Scripts\python.exe tools/cp/train_cp_parser.py ^
  --model Qwen/Qwen2.5-0.5B-Instruct ^
  --train-jsonl data\cp_train.jsonl ^
  --output-dir data\cp_parser_dry_run ^
  --dry-run ^
  --use-lora
```

You can score the heuristic parser or a learned parser with:

```bash
.\.venv312\Scripts\python.exe tools/eval/evaluate_cp_parser.py ^
  --input data\cp_val.jsonl ^
  --mode heuristic
```

Or compare heuristic vs learned parser side by side:

```bash
.\.venv312\Scripts\python.exe tools/eval/evaluate_cp_parser.py ^
  --input examples\cp_parser_eval.jsonl ^
  --mode compare ^
  --model path\to\your_cp_parser_model
```

Compare VLSO grounded QA before and after approved-cluster retraining:

```bash
.\.venv312\Scripts\python.exe tools/eval/evaluate_vlso_review_impact.py ^
  --input examples\vlso_eval.jsonl ^
  --primary-concept-store data\vlso_visual_prototypes.db ^
  --primary-operator-store data\vlso_visual_operators.db ^
  --compare-concept-store data\vlso_geometry_pipeline_gui\approved_review_concepts.db ^
  --compare-operator-store data\vlso_geometry_pipeline_gui\approved_review_operators.db
```

Build a starter real-image VLSO eval set from downloaded family-batch images:

```bash
.\.venv312\Scripts\python.exe tools\vlso\build_real_image_eval.py ^
  --records data\vlso_family_batch\family_records.jsonl ^
  --manifest data\vlso_family_batch\family_download_manifest.jsonl ^
  --candidate-output data\vlso_family_batch\real_image_eval_candidates.jsonl ^
  --seed-output examples\vlso_real_image_eval.jsonl ^
  --limit 24
```

Run grounded QA on that starter real-image benchmark:

```bash
.\.venv312\Scripts\python.exe tools\eval\evaluate_vlso_grounded_qa.py ^
  --input examples\vlso_real_image_eval.jsonl ^
  --mode deep ^
  --answer-mode structured
```

This seed benchmark is intentionally difficult. The current run is a gap-finding benchmark and shows that arbitrary internet-image grounding is still weak.

Run a one-command CP LoRA experiment workspace build and dry-run training plan:

```bash
.\.venv312\Scripts\python.exe tools\cp\run_cp_lora_experiment.py ^
  --workspace tests\cp_lora_experiment_smoke ^
  --model local-test-model ^
  --inputs examples\cp_parser_eval.jsonl ^
  --execute-train ^
  --dry-run-train ^
  --local-files-only ^
  --max-steps 4
```

If you replace `local-test-model` with a real local base model, the same command can build train/val bundles, emit SFT files, run LoRA training, and compare learned vs heuristic parsing.

To save checkpoints during training and resume later, add `--save-steps`, `--save-total-limit`, and `--resume-from-checkpoint`:

```bash
.\.venv312\Scripts\python.exe tools\cp\run_cp_lora_experiment.py ^
  --workspace data\cp_lora_run ^
  --model path\to\your_local_base_model ^
  --inputs examples\cp_parser_eval.jsonl examples\cp_hidden_constraint_eval.jsonl ^
  --eval-inputs examples\cp_geometry_parser_eval.jsonl ^
  --execute-train ^
  --max-steps 200 ^
  --save-steps 25 ^
  --save-total-limit 3 ^
  --resume-from-checkpoint data\cp_lora_run\training_run\checkpoint-100
```

If you already reviewed `data\vlso_family_batch\real_image_eval_candidates.jsonl`, finalize the approved rows into a gold real-image eval set with:

```bash
.\.venv312\Scripts\python.exe tools\vlso\finalize_real_image_eval.py ^
  --candidates data\vlso_family_batch\real_image_eval_candidates.jsonl ^
  --output examples\vlso_real_image_eval_gold.jsonl
```

To run the CP LoRA experiment against an extra held-out evaluation set, add `--eval-inputs`:

```bash
.\.venv312\Scripts\python.exe tools\cp\run_cp_lora_experiment.py ^
  --workspace data\cp_lora_run ^
  --model path\to\your_local_base_model ^
  --inputs examples\cp_parser_eval.jsonl ^
  --eval-inputs examples\cp_geometry_parser_eval.jsonl ^
  --execute-train ^
  --max-steps 200
```

### 7. Run a hard-problem analysis

```bash
.\.venv312\Scripts\python.exe solve_hard_problem.py ^
  --query "If the bag has no open access, check the zipper before inserting the book." ^
  --memory-store data\semop_memory.db ^
  --memory-source grammar_demo
```

### 8. Compare SemOp against a baseline

```bash
.\.venv312\Scripts\python.exe tools/eval/compare_ops_baseline.py ^
  --input examples\ops_labeled_eval_ko.jsonl ^
  --baseline configurable_keyword ^
  --baseline-config examples\customer_baseline_config.json
```

### 9. Learn feedback rules from reviewed items

```bash
.\.venv312\Scripts\python.exe tools/ops/learn_feedback_rules.py ^
  --review-queue data\ops_review_queue.db ^
  --output data\feedback_rules.json
```

### 10. Re-run the copilot with learned rules

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --feedback-rules data\feedback_rules.json ^
  --query "The aisle is blocked and approval is still missing. What should I do?" ^
  --context-file examples\customer_sop_sample.md
```

## Where To Put Data

There are two main data paths.

### A. Public reasoning datasets

You do not need to download these manually if you use the curated manifest scripts.

1. Generate a manifest:

```bash
.\.venv312\Scripts\python.exe tools/corpus/write_curated_manifest.py --preset reasoning_core --output data\reasoning_core_manifest.json
```

2. Ingest it:

```bash
.\.venv312\Scripts\python.exe tools/corpus/ingest_public_manifest.py ^
  --manifest data\reasoning_core_manifest.json ^
  --download-root data\reasoning_core_downloads_312 ^
  --normalized-root data\reasoning_core_normalized_312 ^
  --store data\semop_reasoning_core_312.db ^
  --mode heuristic ^
  --overwrite
```

What goes where:
- raw downloaded files: `data\reasoning_core_downloads_312\`
- normalized JSONL files: `data\reasoning_core_normalized_312\`
- final SQLite memory DB: `data\semop_reasoning_core_312.db`

### B. Competitive-programming knowledge, episode memory, and local compiler setup

The CP pipeline uses:
- `data\knowledge\cp_knowledge.json`
- `data\cp_episodes.db`

These files store:
- official online source URLs
- logical problem frames
- CP DSL operators
- algorithm/data-structure triggers
- complexity metadata
- memory schema layers
- compiler and hardware notes
- successful and failed validation episodes when `solve_contest.py --episode-store ...` is used

The current local compiler observation is conservative:
- `g++ = MinGW.org GCC 6.3.0-1`
- generated code avoids fragile syntax and is syntax-checked locally

### C. Customer SOP or manual documents

Put customer `.md` or `.txt` files in a folder such as:
- `data\customer_docs\`

Then build evaluation stubs from those documents:

```bash
.\.venv312\Scripts\python.exe tools/ops/build_customer_eval_from_docs.py ^
  --inputs data\customer_docs ^
  --output data\customer_eval.jsonl ^
  --max-cases 50
```

If you do not have customer data yet, start from:
- `examples\customer_sop_sample.md`
- `examples\customer_ops_eval_template.jsonl`
- `examples\ops_labeled_eval_ko.jsonl`

## Important Config Files

- baseline config JSON:
  - example: `examples\customer_baseline_config.json`
  - purpose: define a customer-style baseline retriever
- learned feedback rules JSON:
  - example output: `data\feedback_rules.json`
  - purpose: apply supervisor-reviewed corrections back into the copilot
- review queue SQLite:
  - example: `data\ops_review_queue.db`
  - purpose: triage risky answers and record resolution notes

## Main Docs

- logical grammar goal and implementation: `docs/logical_grammar_goal_and_implementation.md`
- hard-problem training and verification: `docs/hard_problem_training_and_verification.md`
- competitive-programming focus and implementation: `docs/cp_focus_research_and_implementation.md`
- usage manual: `docs/usage_manual.md`
- data layout and download guide: `docs/data_layout.md`
- docs index: `docs/index.md`
- GUI and evaluation workflow: `docs/gui_and_eval_workflow.md`
- feedback loop workflow: `docs/feedback_loop_workflow.md`
- customer eval schema: `docs/customer_eval_schema.md`
- architecture and internals: `docs/architecture_and_features.md`
- data-collection API research: `docs/data_collection_api_research.md`
- product framing review: `docs/productization_review.md`

## Validation Status

Latest verified commands:
- `python -m unittest discover -s tests -v`
- `.\.venv312\Scripts\python.exe -m unittest discover -s tests -v`
- `python solve_contest.py --query "Given a weighted graph with N cities and M roads, answer the shortest path from city 1 to all cities."`
- `.\.venv312\Scripts\python.exe tools/eval/compare_ops_baseline.py --input examples\ops_labeled_eval_ko.jsonl --baseline configurable_keyword --baseline-config examples\customer_baseline_config.json`

The old 129-test snapshot referred to the pre-typed product baseline. Current typed
milestone counts and timings are recorded near the typed-core overview above and are
independently exercised by `.github/workflows/typed-core.yml`.

At the time of the legacy ops validation:
- SemOp on labeled ops eval achieved:
  - `avg_relation_recall = 1.0`
  - `avg_answer_term_recall = 0.89`
  - `forbidden_phrase_hit_rate = 0.0`

## Current Limits

- customer evaluation sets are still small unless you add real customer documents
- KPI scores are useful for PoC work, but should still be calibrated against human labels
- review-derived feedback rules are simple and heuristic, not full training updates
- GUI is for internal demos and PoCs, not hardened production deployment


## CP Learning Docs

For the detailed CP-specific workflow, see:
- `docs/cp_focus_research_and_implementation.md`
- `docs/cp_gui_manual.md`
- `docs/cp_training_manual.md`

VLSO examples:
- `examples/vlso_eval.jsonl`
  - starter grounded QA eval and structural operator recovery eval set for VLSO
- `examples/vlso_geometry_eval.jsonl`
  - starter geometry-grounded QA eval and structural operator recovery eval set for VLSO
- `examples/cp_parser_eval.jsonl`
  - starter held-out statement-to-DSL eval set for CP parser comparison
- `examples/cp_geometry_parser_eval.jsonl`
  - geometry-only CP parser eval set generated from normalized labeled corpus rows
- `examples/vlso/bag_closed_observation.json`
- `examples/vlso/geometry_scene.json`
- `examples/vlso/detector_output_example.json`

Recent geometry upgrades:
- VLSO now derives polygon edge entities and grounded geometry relations such as `PARALLEL`, `PERPENDICULAR`, and `EQUAL_LENGTH`
- VLSO can infer shape hypotheses including `triangle`, `right_triangle`, `isosceles_triangle`, `rectangle`, `square`, `parallelogram`, and `quadrilateral`
- CP parsing now recognizes geometry-heavy statements and routes them into `computational_geometry_analysis`
- CP code generation now includes a geometry template with `Point`, `cross`, `dot`, and orientation-style predicates

## New evaluation paths

- Real-image VLSO reviewed gold set: `examples/vlso_real_image_eval_gold.jsonl`
- CP hidden-constraint parser bench: `examples/cp_hidden_constraint_eval.jsonl`
- Script compatibility before/after experiment: `tools/eval/run_premise_compatibility_experiment.py`

Example commands:
```bash
python tools\vlso\finalize_real_image_eval.py --candidates data\vlso_family_batch\real_image_eval_candidates.jsonl --output examples\vlso_real_image_eval_gold.jsonl --auto-approve-limit 8
python tools\eval\evaluate_semop_stack.py --hidden-premises examples\hidden_premise_eval.jsonl --cp-input examples\cp_parser_eval.jsonl --cp-hidden-input examples\cp_hidden_constraint_eval.jsonl --vlso-input examples\vlso_eval.jsonl --vlso-real-image-input examples\vlso_real_image_eval_gold.jsonl
python tools\eval\run_premise_compatibility_experiment.py --input examples\hidden_premise_eval.jsonl --memory-store data\semop_memory.db --output-model data\script_compatibility_model.json
```


## QLoRA and distillation

Teacher traces for hidden-premise reasoning, CP structuring, and VLSO grounded QA can be exported with `tools/eval/export_teacher_traces.py`. The roadmap is documented in `docs/qlora_distillation_roadmap.md`.

The same exporter also supports operator proposal and self-evolution traces via `--operator-transfer-input examples\operator_transfer_eval.jsonl`.

You can then build a generic operator-learning curriculum and dry-run a small student model:

```bash
.\.venv312\Scripts\python.exe tools\eval\build_operator_learning_bundle.py ^
  --teacher-traces data\teacher_traces.jsonl ^
  --workspace data\operator_learning_bundle
```

```bash
.\.venv312\Scripts\python.exe tools\eval\run_operator_training.py ^
  --workspace data\operator_learning_bundle ^
  --model Qwen/Qwen2.5-0.5B-Instruct ^
  --dry-run ^
  --use-lora
```

The full workflow is documented in `docs/operator_learning_plan.md`.

To reduce black-box reasoning, SemOp now also compiles hidden-premise and operator-algebra outputs into a deterministic operator program before final response synthesis. This runtime records satisfied facts, missing facts, and goal-risk decisions instead of leaving the whole reasoning path inside the model.

## Overall Understanding Benchmark

Run the broad understanding check across hidden-premise reasoning, CP structuring, CP hidden constraints, starter VLSO, and reviewed real-image VLSO:

```bash
.\.venv312\Scripts\python.exe tools\eval\evaluate_understanding.py
```

In the easy GUI, use `5. Geometry starter tools -> Evaluation shortcuts -> Run overall understanding benchmark`.

## Shared Agent Doctrine

All future agent work in this repository should follow `docs/multi_agent_operator_doctrine.md`: recover shared basis operators first, compose higher operators through algebra, and only retain operators that survive verifier and transfer checks.

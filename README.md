# SemOp Local 4060

SemOp Local 4060 is a local prototype for structured reasoning and domain copilot experiments.

The current product direction is not a general chatbot. It is a Korean warehouse and field-operations SOP copilot that:
- reads SOPs, manuals, and exception-handling documents
- reconstructs hidden prerequisites, blockers, and alternatives
- reduces unsafe or non-executable answers
- leaves an audit trace and operational KPI summary
- compares itself against configurable baselines such as plain RAG

At the research level, the longer-term target is broader: learn how logical words such as `has`, `is`, `requires`, `if`, `before`, and `can` bind to concept frames, then reuse those learned grammar priors during reasoning across domain QA, math, olympiad proof search, competitive programming, and VLSO world-model reasoning.

## What You Can Run Today

- `app.py`
  - research-oriented structured reasoning CLI
- `ops_copilot.py`
  - product-style warehouse/operations copilot CLI
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
- `tools/vlso/recommend_visual_labels.py`
  - rank the next most informative targets to label based on novelty and uncertainty
- `tools/eval/compare_ops_baseline.py`
  - SemOp vs baseline benchmark on labeled cases
- `tools/eval/evaluate_ops_kpis.py`
  - KPI averages on operations case sets
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

Refresh it after structural changes with:

```bash
.\.venv312\Scripts\python.exe tools\maintenance\update_code_structure_docs.py
```

## Quick Start

### 1. Run the operations copilot on a single SOP question

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --query "The aisle is blocked and approval is still missing. What should I do?" ^
  --context-file examples\customer_sop_sample.md
```

### 2. Open the GUI

```bash
.\.venv312\Scripts\python.exe ops_copilot_gui.py ^
  --review-queue data\ops_review_queue.db ^
  --baseline-config examples\customer_baseline_config.json
```

Then open `http://127.0.0.1:8765`.

### 3. Try the olympiad proof-search prototype

```bash
.\.venv312\Scripts\python.exe solve_olympiad.py ^
  --query "Prove that the sum of two odd integers is even."
```

### 3B. Try the VLSO prototype with structured observations

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --query "How do I put a book into a bag?" ^
  --visual-json examples\vlso\bag_closed_observation.json
```

### 3C. Try the VLSO prototype in deep mode on a raw image

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
Label-candidate generation and weight re-estimation CLIs are `tools\vlso\build_affordance_label_candidates.py` and `tools\vlso\train_affordance_classifier.py`. Manifest-driven API collection planning is available through `tools\vlso\collect_visual_data.py`, and whitelist/download staging is available through `tools\vlso\prepare_visual_downloads.py`.

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

At the time of the last validation:
- all tests passed: `99`
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
- `examples/vlso/bag_closed_observation.json`
- `examples/vlso/geometry_scene.json`
- `examples/vlso/detector_output_example.json`





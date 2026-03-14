# Usage Manual

Code layout reference:
- `docs/project_structure.md`
- refresh with `.\.venv312\Scripts\python.exe tools\maintenance\update_code_structure_docs.py`


## 0. Fastest Start

If you just want to test the project without learning commands first, run:

```bash
.\.venv312\Scripts\python.exe semop_studio_gui.py
```

Then open:
- `http://127.0.0.1:8780`

`semop_studio_gui.py` is the friendlier default surface for:
- one-click beginner setup, training, and testing
- context reasoning
- vision-grounded reasoning
- unified artifact training
- benchmark-gated learning
- compact result inspection

If you are starting from scratch, use the `One-click setup + train + test` button in `One-click mode` first. It seeds built-in hidden-premise, transfer, and starter vision examples, writes approved grounded review traces, trains the unified artifact bundle, then runs the benchmark gate and beginner smoke tests in one pass.

Long actions now run in the background. Watch the `Live jobs` card in the GUI while they run; the page refreshes automatically every few seconds until the active job finishes.

You can cancel a queued or running background job from the same card, retry a finished job, and reload a saved result later. Each job now keeps a small step log, and failed or blocked jobs show suggested recovery buttons such as `Guided starter loop` or `One-click setup`. Completed, failed, and cancelled jobs are saved automatically to `studio_job_history.json` under the current output directory.

Finished jobs also show up in the `Completion alerts` card. That card keeps unread notifications in `studio_notifications.json`, so you can come back later and still see which run finished, failed, or was cancelled without reading the full log again.

If you want to inspect how two runs differ, use the `Artifact compare` card. It can compare two files such as `benchmark_gate.json` and `accepted_benchmark_summary.json`, or compare two whole output directories and list changed, left-only, and right-only artifacts.


If you already have a store and just want to re-check the current bundle, use `One-click test current bundle`.

If the benchmark gate is blocked with `0.0` on analogy, grounding, or repair, use the GUI button `Guided starter loop` once. It seeds starter graphs, writes approved review traces, and reruns training plus the gate automatically.

If you need the older all-in-one power-user lab, you can still run:

```bash
.\.venv312\Scripts\python.exe semop_easy_gui.py
```

Then open:
- `http://127.0.0.1:8770`

That advanced GUI still lets you:
- test warehouse and SOP reasoning
- test competitive-programming analysis
- ask image-grounded questions on `data\scene.png` or your own image
- compare a primary VLSO concept store against a pseudo-labeled store
- review self-training clusters and save approve/reject decisions
- download or normalize labeled CP datasets
- build a VLSO public-image collection plan
- generate an object-family manifest for bag, box, drawer, door, bottle, tool, cabinet, suitcase, jar, bin, and pouch
- run a family-target batch collector that writes manifest, records, approved, and download-manifest files automatically
- preview public-image cards and approve downloads
- run learning on downloaded images
- edit downloaded-image labels with review status and notes
- retrain concept and operator stores from approved labels only
- download or stage labeled CP data with one click
- generate synthetic geometry images and eval files
- run one-click VLSO geometry self-training and operator learning
- retrain approved clusters into new concept/operator stores
- compare approved VLSO stores on grounded QA eval sets
- compare CP heuristic parser against a learned parser model
- run or resume CP LoRA training from the easy GUI with checkpoint saving
- generate CP geometry eval and train starter files
- compare hand-labeled and pseudo-labeled concept stores
- approve or reject self-training clusters from the easy GUI

## Progress Snapshot

If you want a benchmark-backed estimate of how close the project is to the operator-intelligence target, run:

```bash
.\.venv312\Scripts\python.exe tools\eval\evaluate_operator_intelligence_progress.py ^
  --hidden-premises examples\hidden_premise_eval.jsonl ^
  --cp-input examples\cp_parser_eval.jsonl ^
  --vlso-input examples\vlso_eval.jsonl
```

This produces:
- operator architecture
- premise reasoning
- shared world model
- CP structuring
- raw visual reasoning
- general operator transfer

and two conservative overall estimates:
- research architecture overall
- robust general intelligence overall

Use these numbers for trend tracking. They are starter-benchmark estimates, not production-grade capability claims.

## 0A. Evaluate operator self-evolution and transfer

Run the operator algebra benchmark first:

```bash
.\.venv312\Scripts\python.exe tools\eval\evaluate_operator_algebra.py ^
  --input examples\operator_algebra_eval.jsonl ^
  --mode heuristic
```

Then run the starter cross-domain transfer benchmark:

```bash
.\.venv312\Scripts\python.exe tools\eval\evaluate_operator_transfer.py ^
  --input examples\operator_transfer_eval.jsonl ^
  --mode heuristic
```

This second command measures whether retained evolved operators still help on held-out domains instead of only on the domain that first produced them.

To measure whether geometry-first signals really matter for retention, run:

```bash
.\.venv312\Scripts\python.exe tools\eval\evaluate_visual_signal_operator_impact.py ^
  --input examples\operator_transfer_eval.jsonl ^
  --mode heuristic
```

If you have a local LLM model id for operator summarization, compare heuristic and LLM proposal engines with:

```bash
.\.venv312\Scripts\python.exe tools\eval\compare_operator_proposals.py ^
  --input examples\operator_transfer_eval.jsonl ^
  --mode heuristic ^
  --llm-model-id Qwen/Qwen2.5-3B-Instruct
```
Internally, retained operators are now preceded by a proposal phase: repeated basis patterns and geometry signatures are summarized into candidate higher operators, then normalized and scored before retention.

To run the full iterative self-evolution loop and persist retained operators back into SQLite memory, use:

```bash
.\.venv312\Scripts\python.exe tools\eval\run_operator_self_evolution.py ^
  --input examples\operator_transfer_eval.jsonl ^
  --memory-store data\semop_memory.db ^
  --source operator_self_evolution ^
  --iterations 3
```

## 0B. Build a starter real-image VLSO benchmark

If you already used the GUI family-batch collector and have downloaded files under `data\vlso_family_batch`, build a starter real-image eval set with:

```bash
.\.venv312\Scripts\python.exe tools\vlso\build_real_image_eval.py ^
  --records data\vlso_family_batch\family_records.jsonl ^
  --manifest data\vlso_family_batch\family_download_manifest.jsonl ^
  --candidate-output data\vlso_family_batch\real_image_eval_candidates.jsonl ^
  --seed-output examples\vlso_real_image_eval.jsonl ^
  --limit 24
```

Then score the current VLSO stack on those real images:

```bash
.\.venv312\Scripts\python.exe tools\eval\evaluate_vlso_grounded_qa.py ^
  --input examples\vlso_real_image_eval.jsonl ^
  --mode deep ^
  --answer-mode structured
```

Use this as a gap-finding benchmark, not a final scorecard. The current starter run is intentionally harsh and exposes weak raw internet-image grounding.

## 0C. Run the CP LoRA experiment harness

This creates train/val bundles, SFT exports, a heuristic baseline snapshot, and an optional LoRA training plan in one workspace:

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

If you replace `local-test-model` with a real local base model path, the same command can run a real LoRA experiment.

To save checkpoints and resume later, add these flags:

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

The easy GUI exposes the same flow under `5. Geometry starter tools -> CP LoRA train/resume`.

If you manually review `data\vlso_family_batch\real_image_eval_candidates.jsonl`, convert the approved rows into a reusable gold eval file with:

```bash
.\.venv312\Scripts\python.exe tools\vlso\finalize_real_image_eval.py ^
  --candidates data\vlso_family_batch\real_image_eval_candidates.jsonl ^
  --output examples\vlso_real_image_eval_gold.jsonl
```

If you want the CP LoRA experiment to compare against an extra held-out set after training, pass `--eval-inputs`:

```bash
.\.venv312\Scripts\python.exe tools\cp\run_cp_lora_experiment.py ^
  --workspace data\cp_lora_run ^
  --model path\to\your_local_base_model ^
  --inputs examples\cp_parser_eval.jsonl ^
  --eval-inputs examples\cp_geometry_parser_eval.jsonl ^
  --execute-train ^
  --max-steps 200
```

## 1. Decide Which Flow You Need

Use one of these flows.

### Research flow
Use this if you want to inspect the structured meaning graph directly.

```bash
.\.venv312\Scripts\python.exe app.py --mode heuristic --query "?몄감?μ씠 硫怨?湲몄씠 留됲엳?붾뜲 ?대뼸寃?媛???좉퉴??"
```

### Product copilot flow
Use this if you want warehouse or operations answers with KPI and audit trace.

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --query "吏寃뚯감濡??붾젅?몃? ?숈뿉 ?щ━?ㅻ뒗???듬줈媛 留됲? ?덇퀬 ?꾩쭅 ?뱀씤?????ъ뒿?덈떎. ?대뼸寃??댁빞 ?섎굹??" ^
  --context-file examples\customer_sop_sample.md
```

## 2. Prepare Data

### Option A: Use included example data

Files already included:
- `examples\customer_sop_sample.md`
- `examples\ops_sop_cases_ko.jsonl`
- `examples\ops_labeled_eval_ko.jsonl`
- `examples\customer_ops_eval_template.jsonl`
- `examples\customer_baseline_config.json`

These are enough to test the full workflow.

### Option B: Use your own customer SOP documents

1. Put `.md` or `.txt` files into a folder, for example:
- `data\customer_docs\`

2. Generate evaluation case stubs from those docs:

```bash
.\.venv312\Scripts\python.exe tools/ops/build_customer_eval_from_docs.py ^
  --inputs data\customer_docs ^
  --output data\customer_eval.jsonl ^
  --max-cases 50
```

3. Manually review and edit `data\customer_eval.jsonl`.

You should especially review these fields:
- `expected_relations`
- `expected_answer_terms`
- `forbidden_phrases`
- `expected_clarification`

## 3. Run a Baseline Comparison

### Default lexical baseline

```bash
.\.venv312\Scripts\python.exe tools/eval/compare_ops_baseline.py ^
  --input examples\ops_labeled_eval_ko.jsonl ^
  --baseline lexical_rag
```

### First-chunk baseline

```bash
.\.venv312\Scripts\python.exe tools/eval/compare_ops_baseline.py ^
  --input examples\ops_labeled_eval_ko.jsonl ^
  --baseline first_chunk
```

### Customer-config baseline

```bash
.\.venv312\Scripts\python.exe tools/eval/compare_ops_baseline.py ^
  --input data\customer_eval.jsonl ^
  --baseline configurable_keyword ^
  --baseline-config examples\customer_baseline_config.json
```

If you want a customer-specific baseline, copy and edit:
- `examples\customer_baseline_config.json`

## 4. Use the GUI

### Easiest all-in-one GUI

```bash
.\.venv312\Scripts\python.exe semop_easy_gui.py
```

Then open:
- `http://127.0.0.1:8770`

### Full operations GUI

```bash
.\.venv312\Scripts\python.exe ops_copilot_gui.py ^
  --review-queue data\ops_review_queue.db ^
  --baseline-config examples\customer_baseline_config.json
```

Then open:
- `http://127.0.0.1:8765`

The GUI lets you:
- choose example cases
- paste your own SOP context
- run SemOp Copilot
- compare against a baseline
- inspect review queue items
- approve, reject, or mark items as follow-up
- export teacher traces and run generic operator-student training

## 5. Resolve Review Items And Learn Rules

### Step 1: Run the copilot with a review queue enabled

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --review-queue data\ops_review_queue.db ^
  --query "The aisle is blocked and approval is still missing. What should I do?" ^
  --context-file examples\customer_sop_sample.md
```

### Step 2: Open the GUI and resolve pending items

Set the status to one of:
- `approved`
- `rejected`
- `needs_followup`
- `pending`

Add a resolution note describing what should change.

### Step 3: Learn feedback rules from reviewed items

```bash
.\.venv312\Scripts\python.exe tools/ops/learn_feedback_rules.py ^
  --review-queue data\ops_review_queue.db ^
  --output data\feedback_rules.json
```

### Step 4: Re-run the copilot with the learned rules

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --feedback-rules data\feedback_rules.json ^
  --query "The aisle is blocked and approval is still missing. What should I do?" ^
  --context-file examples\customer_sop_sample.md
```

## 6. Optional Public Dataset Pipeline

If you want to test the broader research pipeline, use curated public datasets.

### Generate a curated manifest

```bash
.\.venv312\Scripts\python.exe tools/corpus/write_curated_manifest.py --preset reasoning_core --output data\reasoning_core_manifest.json
```

### Download, normalize, and ingest

```bash
.\.venv312\Scripts\python.exe tools/corpus/ingest_public_manifest.py ^
  --manifest data\reasoning_core_manifest.json ^
  --download-root data\reasoning_core_downloads_312 ^
  --normalized-root data\reasoning_core_normalized_312 ^
  --store data\semop_reasoning_core_312.db ^
  --mode heuristic ^
  --overwrite
```

## 7. What To Edit First In A Real PoC

If you start with a real customer, edit these first:
- `data\customer_eval.jsonl`
- `examples\customer_baseline_config.json` or your own baseline config copy
- `data\feedback_rules.json` after the first supervisor review cycle

That is the minimum set needed to turn this repo from demo mode into a customer-specific PoC.

## 8. Train Corpus-Level Logical Grammar Priors

If you want the system to learn reusable logical connectors such as `has`, `if`, `before`, or `requires`, run a corpus learning pass and store the result.

```bash
.\.venv312\Scripts\python.exe tools/corpus/train_corpus.py ^
  --input examples\reasoning_corpus_ko.jsonl ^
  --mode heuristic ^
  --output data\reasoning_learning.json ^
  --store data\semop_memory.db ^
  --source grammar_demo
```

The output JSON now contains `logical_patterns` in addition to operator families.
When you later run the pipeline with the same memory store, those patterns can be used as inference priors.

## 9. Use Learned Grammar Priors During Inference

```bash
.\.venv312\Scripts\python.exe app.py --mode heuristic --query "If the bag has no open access, check the zipper before inserting the book."
```

If a matching learning run exists in the memory store, the graph may include:
- logical grammar prior warnings
- grammar hypotheses from corpus learning
- frame-style operator candidates
- extra plan priors for condition, order, capability, or prerequisites

## 10. Competitive Programming Flow

Use this when you want the system to read a contest problem, recover hidden structure, choose an algorithm family, and emit C++17 code with a syntax check.

### Knowledge file

The CP engine reads:
- `data\knowledge\cp_knowledge.json`

This file already stores:
- official online source URLs
- logical problem frames
- CP DSL operators
- algorithm triggers and hidden concepts
- complexity metadata
- memory schema layers
- compiler and hardware notes

### Generate a solution sketch, validate it, and store the episode

```bash
.\.venv312\Scripts\python.exe solve_contest.py ^
  --query "There are many range sum queries on an array and no updates. Output the sum from l to r each time." ^
  --episode-store data\cp_episodes.db
```

What you get:
- algorithm family
- time and memory complexity
- hidden concepts
- matched logical frames
- goal types and domain tags
- matched DSL operators
- semantic / structural / episodic / procedural memory projection
- reasoning steps
- compile-check status
- generated C++17 code
- optional episodic memory record in `data\cp_episodes.db`
- episode-driven reranking priors for later similar statements


### Download labeled public CP datasets into normalized JSONL

```bash
python tools\cp\download_cp_labeled_datasets.py ^
  --manifest examples\cp_labeled_manifest.json ^
  --download-root data\cp_labeled_downloads ^
  --output data\cp_labeled_normalized.jsonl
```

This is the easiest way to bootstrap statement + solution + tag style public datasets before converting them into SemOp-specific labels.

### Prepare an expanded CP corpus, then build the labeled DSL and SFT datasets

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

This produces:
- a normalized CP statement corpus from mixed `.txt`, `.md`, `.jsonl`, `.json`, `.csv`, `.html`, and `.zip` inputs
- labeled statement -> DSL / frame / algorithm records
- prompt/completion records for future LoRA or small-model parser training

Current built-in execution validators cover:
- `prefix_sum_range_query`
- `fenwick_tree`
- `dijkstra_shortest_path`
- `segment_tree`
- `lazy_segment_tree`
- `dsu_connectivity`
- `grid_bfs`
- `knapsack_dp`
- `binary_search_answer`

The CLI now reports:
- sample check result
- random/brute-force check result
- overall validation status
- repair-loop status when a compile or validation fix was attempted

### Build a merged train/val bundle

```bash
.\.venv312\Scripts\python.exe tools/cp/build_cp_training_bundle.py ^
  --inputs examples\cp_dsl_expanded_dataset.jsonl ^
  --episode-store data\cp_episodes.db ^
  --train-output data\cp_train.jsonl ^
  --val-output data\cp_val.jsonl ^
  --train-sft-output data\cp_train_sft.jsonl ^
  --val-sft-output data\cp_val_sft.jsonl
```

### Train a small parser model for statement -> DSL / frame / sketch

Use a dry run first. This prepares the training file and GPU-aware plan without loading the model.

```bash
.\.venv312\Scripts\python.exe tools/cp/train_cp_parser.py ^
  --model Qwen/Qwen2.5-0.5B-Instruct ^
  --train-jsonl examples\cp_dsl_expanded_dataset.jsonl ^
  --output-dir data\cp_parser_dry_run ^
  --dry-run ^
  --use-lora
```

To train from stored contest episodes instead, point the trainer at the SQLite store:

```bash
.\.venv312\Scripts\python.exe tools/cp/train_cp_parser.py ^
  --model Qwen/Qwen2.5-0.5B-Instruct ^
  --episode-store data\cp_episodes.db ^
  --output-dir data\cp_parser_from_episodes ^
  --dry-run
```

On RTX 4060 8GB, keep the starting point small:
- prefer `0.5B` to `1.5B` parser models first
- start with `--dry-run` and local files if the model is already cached
- turn on `--use-lora` before attempting full-model training
- use the generated `training_plan.json` before attempting a real run

### Evaluate the heuristic parser or a learned parser

```bash
.\.venv312\Scripts\python.exe tools/eval/evaluate_cp_parser.py ^
  --input data\cp_val.jsonl ^
  --mode heuristic
```

For a learned parser:

```bash
.\.venv312\Scripts\python.exe tools/eval/evaluate_cp_parser.py ^
  --input data\cp_val.jsonl ^
  --mode model ^
  --model <local-model-or-adapter> ^
  --local-files-only
```

For a direct heuristic-vs-model comparison:

```bash
.\.venv312\Scripts\python.exe tools/eval/evaluate_cp_parser.py ^
  --input examples\cp_parser_eval.jsonl ^
  --mode compare ^
  --model <local-model-or-adapter>
```

For VLSO approved-store impact evaluation:

```bash
.\.venv312\Scripts\python.exe tools/eval/evaluate_vlso_review_impact.py ^
  --input examples\vlso_eval.jsonl ^
  --primary-concept-store data\vlso_visual_prototypes.db ^
  --primary-operator-store data\vlso_visual_operators.db ^
  --compare-concept-store data\vlso_geometry_pipeline_gui\approved_review_concepts.db ^
  --compare-operator-store data\vlso_geometry_pipeline_gui\approved_review_operators.db
```

### Current compiler constraint

Local validation was done with:
- `g++ = MinGW.org GCC 6.3.0-1`

So generated code intentionally uses a conservative C++17 subset.

### Recommended use on RTX 4060 8GB

For CP, keep the pipeline mostly CPU-first. Use the GPU only for lightweight model assistance if you add one later.

## 11. Run The Hard-Problem Solver

Use this when the problem is harder than a normal QA case and you want:
- structured reasoning output
- verification checks
- optional logical-pattern weight updates

```bash
.\.venv312\Scripts\python.exe solve_hard_problem.py ^
  --query "If the bag has no open access, check the zipper before inserting the book." ^
  --memory-store data\semop_memory.db ^
  --memory-source grammar_demo
```

To update pattern weights automatically after the run:

```bash
.\.venv312\Scripts\python.exe solve_hard_problem.py ^
  --query "Prove that 1+2+...+n = n(n+1)/2 for all positive integers n." ^
  --memory-store data\semop_memory.db ^
  --memory-source grammar_demo ^
  --learn ^
  --success auto
```

This writes weights to:
- `data\logical_pattern_weights.json`



## 11. CP Incident Memory, GUI, And Training

### Ingest real WA/TLE/editorial cases into the episode store

```bash
.\.venv312\Scripts\python.exe tools/cp/ingest_cp_episodes.py ^
  --inputs examples\cp_incident_cases.jsonl ^
  --store data\cp_episodes.db
```

### Open the beginner-friendly CP GUI

```bash
.\.venv312\Scripts\python.exe cp_copilot_gui.py ^
  --episode-store data\cp_episodes.db
```

Then open:
- `http://127.0.0.1:8787`

### Train later with the detailed guide

See:
- `docs\cp_training_manual.md`
- `docs\cp_gui_manual.md`


## 12. Vision-Language Semantic Operators (VLSO)

Use this when you want language, detector output, and raw images to land in the same operator space.

Default recommendation:
- use `--mode deep`
- keep the structural parser active as the fallback that converts raw images or detector output into operator graphs
- place local vision checkpoints under `models\vision\dinov2\...` or `models\vision\openclip\...`

### Structured observation input

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "How do I put a book into a bag?" ^
  --visual-json examples\vlso\bag_closed_observation.json
```

### Detector output input

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --query "What geometric structure is visible here?" ^
  --detector-json examples\vlso\detector_output_example.json
```

### Raw image input with the deep-first parser

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "What shapes are visible here?" ^
  --image-path data\scene.png
```

### Raw image input with an explicit local vision backbone

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "What shapes are visible here?" ^
  --image-path data\scene.png ^
  --vision-backbone dinov2_adapter ^
  --vision-model-path models\vision\dinov2\dinov2-small
```

Notes:
- if `--image-path` is given without detector JSON, SemOp runs a local raw-image parser first and converts connected components into VLSO objects and shape hints
- if a local DINOv2 or OpenCLIP checkpoint is available, the same image path is also embedded through that backbone for visual-memory retrieval
- recommended local folders are `models\vision\dinov2\...` and `models\vision\openclip\...`
- if the local checkpoint is missing or cannot load, SemOp falls back to `token_geometry_v1`
- `--affordance-weights` loads learned weak-classifier weights from a JSON file such as `data\vlso_samples\trained_affordance_weights.json`
- `--answer-mode structured` returns a deterministic grounded answer from the operator graph
- `--answer-mode llm` uses the grounded world model as context for a local text model, and `--answer-model-id` overrides the default Qwen model id

### Grounded VLSO QA with learned affordance weights

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "How can I access the bag opening?" ^
  --image-path data\vlso_samples\backpack_public_domain.jpg ^
  --affordance-weights data\vlso_samples\trained_affordance_weights.json ^
  --answer-mode structured ^
  --format json
```

### Grounded VLSO QA with a local Qwen-style answer model

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "What objects are visible here?" ^
  --image-path data\scene.png ^
  --answer-mode llm ^
  --answer-model-id Qwen/Qwen2.5-3B-Instruct
```


### Convert Open Images boxes into detector JSONL

```bash
python tools\vlso\ingest_open_images_annotations.py ^
  --boxes data\open_images\train-annotations-bbox.csv ^
  --labels data\open_images\class-descriptions-boxable.csv ^
  --output data\open_images_vlso.jsonl ^
  --limit-images 100
```

The output rows can be fed into the existing `--detector-json` path after selecting or splitting per image.

### Segmentation-style detector payloads

`--detector-json` now accepts these payload families:
- `detections` or `predictions` with `bbox`, `polygon`, `affordances`, and `state`
- `annotations` plus `categories` in a COCO-like shape
- `segments` with polygon-like segmentation data
- `instances` with `boxes`, `labels`, `scores`, and `polygons` or `masks`


### Plan automatic public-image collection with a dry-run manifest

```bash
python tools\vlso\collect_visual_data.py ^
  --manifest examples\vlso_collection_manifest.json ^
  --output data\vlso_collection_plan.jsonl ^
  --summary-output data\vlso_collection_plan_summary.json
```

Use this to stage API-backed collection before downloading anything.
Provider tradeoffs are documented in `docs\data_collection_api_research.md`.

Then review and build a download manifest:

```bash
python tools\vlso\prepare_visual_downloads.py ^
  --records data\vlso_collection_records.jsonl ^
  --approved-output data\vlso_collection_approved.jsonl ^
  --manifest-output data\vlso_download_manifest.jsonl ^
  --download-root data\vlso_downloads ^
  --allow-providers wikimedia_commons openverse ^
  --allow-licenses cc0 by by-sa
```

### Build a few-shot visual concept memory

This is the recommended low-data workflow: label a few diverse objects, compress them into prototypes, then ask the recommender which targets are most worth labeling next.

### Train compact prototypes for sample-efficient learning

```bash
python tools\vlso\train_visual_concepts.py ^
  --labels examples\vlso_visual_concepts_template.jsonl ^
  --store data\vlso_visual_prototypes.db ^
  --summary-output data\vlso_visual_prototypes_summary.json
```

### Recommend the next targets to label

```bash
python tools\vlso\recommend_visual_labels.py ^
  --candidates data\vlso_samples\concept_candidates.jsonl ^
  --concept-store data\vlso_visual_prototypes.db ^
  --weights data\vlso_samples\trained_affordance_weights.json ^
  --limit 10
```


1. Generate candidate rows for manual labeling:

```bash
python tools\vlso\build_visual_concept_candidates.py ^
  --inputs data\vlso_samples\backpack_public_domain.jpg data\vlso_samples\military_backpack_cc0.jpg ^
  --output data\vlso_samples\concept_candidates.jsonl
```

2. Fill labels using the same schema as `examples\vlso_visual_concepts_template.jsonl`.

3. Index them into the few-shot concept store:

```bash
python tools\vlso\index_visual_concepts.py ^
  --labels examples\vlso_visual_concepts_template.jsonl ^
  --store data\vlso_visual_prototypes.db
```

4. Use that concept store during QA:

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
- the current robust path is `deep-first + structural fallback`, not `deep-only`
- the raw-image parser now performs mask cleanup, dominant border-frame suppression, and lightweight container/part/affordance inference before building the operator graph
- public sample images for local tuning are stored under `data\vlso_samples\` with source notes in `data\vlso_samples\SOURCES.md`
- expand this set with the workflow in `docs\vlso_data_collection_guide.md` and optional labels in `examples\vlso_affordance_labels_template.jsonl`
- build editable candidate labels with `tools\vlso\build_affordance_label_candidates.py` and retrain weights with `tools\vlso\train_affordance_classifier.py`

### Index structured visual observations or raw images into the embedding store

```bash
.\.venv312\Scripts\python.exe tools\vlso\index_vlso_visual_memory.py ^
  --inputs examples\vlso\bag_closed_observation.json examples\vlso\geometry_scene.json ^
  --store data\vlso_visual_memory.db
```

```bash
.\.venv312\Scripts\python.exe tools\vlso\index_vlso_visual_memory.py ^
  --image-inputs path\to\scene1.png path\to\scene2.png ^
  --store data\vlso_visual_memory.db ^
  --model-id openclip_adapter ^
  --model-path path\to\local\openclip_checkpoint
```




## Architecture References

Use these docs when you want the high-level system direction instead of command-by-command usage.

- `operator_intelligence_system.md`: the long-term intelligence-core definition
- `operator_intelligence_roadmap.md`: the staged roadmap for operator learning, world models, memory, and verifiers


## Quick Eval Commands

```bash
.\.venv312\Scripts\python.exe tools\eval\evaluate_vlso_grounded_qa.py --input examples\vlso_eval.jsonl --mode heuristic --answer-mode structured
```

```bash
.\.venv312\Scripts\python.exe tools\eval\evaluate_vlso_grounded_qa.py --input examples\vlso_geometry_eval.jsonl --mode heuristic --answer-mode structured
```

```bash
.\.venv312\Scripts\python.exe tools\eval\evaluate_cp_parser.py --input examples\cp_parser_eval.jsonl --mode heuristic
```

```bash
.\.venv312\Scripts\python.exe tools\eval\evaluate_cp_parser.py --input examples\cp_geometry_parser_eval.jsonl --mode heuristic
```


## Geometry Data Bootstrap

### VLSO geometry and access images

```bash
.\.venv312\Scripts\python.exe tools\vlso\bootstrap_geometry_visual_data.py ^
  --workspace data\vlso_geometry_bootstrap
```

Add `--execute-collect` to call the public APIs and add `--execute-downloads` to fetch approved media files.
The preset manifest is also available at `examples\vlso_geometry_collection_manifest.json`.

### Generate synthetic geometry scenes locally

```bash
.\.venv312\Scripts\python.exe tools\vlso\generate_geometry_dataset.py ^
  --output-dir examples\vlso\generated_geometry ^
  --eval-output examples\vlso_geometry_eval.jsonl
```

This creates paired PNG and JSON scene assets for parallel/perpendicular, square, and triangle cases, then writes a ready-to-run VLSO eval set.

### CP geometry labeled corpus

```bash
.\.venv312\Scripts\python.exe tools\cp\bootstrap_geometry_corpus.py ^
  --manifest examples\cp_geometry_labeled_manifest.json ^
  --download-root data\cp_geometry_downloads ^
  --output data\cp_geometry_labeled.jsonl
```

`examples\cp_geometry_labeled_manifest.json` supports both normal URLs and Hugging Face datasets. Geometry rows are filtered by tags and text terms before normalization.

## Operator Algebra Benchmark

```bash
.\.venv312\Scripts\python.exe tools\eval\evaluate_operator_algebra.py ^
  --input examples\operator_algebra_eval.jsonl ^
  --mode heuristic
```

This measures:
- operator decomposition recovery
- functor-hypothesis recovery

## Segmentation-Aware Structural Grounding

Detector payloads can now carry explicit fields such as:
- `part_of`
- `part_of_confidence`
- `structural_role`
- `segmentation_confidence`

Useful `structural_role` values include:
- `opening`
- `access_control`
- `handle`
- `grasp`
- `strap`

These are converted into stronger structural bindings such as `ACCESS_PORT_OPERATOR`, `ACCESS_CONTROL_OPERATOR`, and `ATTACHED_GRASP_OPERATOR`.

## Added starter benches

- `examples/vlso_real_image_eval_gold.jsonl`: reviewed or auto-selected real-image VLSO benchmark
- `examples/cp_hidden_constraint_eval.jsonl`: CP parser benchmark focused on hidden efficiency and structural constraints

## Added experiment tool

Use `tools/eval/run_premise_compatibility_experiment.py` to train a script-compatibility scorer from SQLite memory and compare hidden-premise metrics before and after loading the trained model.


## Teacher Trace Export

```bash
.\.venv312\Scripts\python.exe tools\eval\export_teacher_traces.py ^
  --hidden-premises examples\hidden_premise_eval.jsonl ^
  --cp-input examples\cp_parser_eval.jsonl ^
  --vlso-input examples\vlso_eval.jsonl ^
  --examples-root examples ^
  --output data\teacher_traces.jsonl ^
  --sft-output data\teacher_traces_sft.jsonl
```

This exports a common distillation dataset for QLoRA/SFT experiments.

Add `--operator-transfer-input examples\operator_transfer_eval.jsonl` if you want operator-proposal and self-evolution traces in the exported teacher dataset.

You can also export harder traces with `--cp-hidden-input examples\cp_hidden_constraint_eval.jsonl` and `--vlso-real-image-input examples\vlso_real_image_eval_gold.jsonl`.

## 0D. Build an operator-learning curriculum and student model

Build the curriculum bundle from teacher traces:

```bash
.\.venv312\Scripts\python.exe tools\eval\build_operator_learning_bundle.py ^
  --teacher-traces data\teacher_traces.jsonl ^
  --workspace data\operator_learning_bundle
```

Dry-run a generic operator student model:

```bash
.\.venv312\Scripts\python.exe tools\eval\run_operator_training.py ^
  --workspace data\operator_learning_bundle ^
  --model Qwen/Qwen2.5-0.5B-Instruct ^
  --dry-run ^
  --use-lora
```

Resume from checkpoint if needed:

```bash
.\.venv312\Scripts\python.exe tools\eval\run_operator_training.py ^
  --workspace data\operator_learning_bundle ^
  --model E:\path\to\local_text_model ^
  --use-lora ^
  --use-qlora ^
  --resume-from-checkpoint data\operator_learning_bundle\training_run\checkpoint-100
```

See `docs/operator_learning_plan.md` for the full training roadmap.

SemOp now also runs an operator compiler/executor after premise recovery and operator algebra. This means hidden premises, goal-preservation checks, and operator decompositions are compiled into a deterministic support trace before the final answer is synthesized.

## Overall Understanding Benchmark

Run the broad understanding check across hidden-premise reasoning, CP structuring, CP hidden constraints, starter VLSO, and reviewed real-image VLSO:

```bash
.\.venv312\Scripts\python.exe tools\eval\evaluate_understanding.py
```

In the easy GUI, use `5. Geometry starter tools -> Evaluation shortcuts -> Run overall understanding benchmark`.


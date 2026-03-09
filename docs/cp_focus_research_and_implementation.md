# Competitive Programming Focus

## Goal

The CP direction is now explicit:
- read the problem statement
- recover hidden algorithmic structure
- translate the statement into a CP DSL
- choose candidate algorithm families
- verify feasibility and implementation risk
- generate conservative C++17 code
- compile-check locally before trusting the output

This is not just code generation. It is a structured reasoning engine for contest problems.

## Review Of The Proposed Direction

The proposal is correct on the important points.

What matches this repository well:
- a small local reasoning core should not memorize every solution
- problem solving should go through an intermediate structured representation
- memory must be split into semantic, structural, episodic, and procedural layers
- hard problems need candidate generation plus verification, not one-shot code generation
- RTX 4060 8GB is realistic if the system stays CPU-first and uses the GPU only as an auxiliary component

What should remain realistic:
- this is a strong CP copilot direction, not yet a guaranteed fully automatic Ruby solver
- the immediate target is reliable decomposition, algorithm-family ranking, implementation planning, and verification hooks

## CP DSL 1st Dictionary

The first CP DSL dictionary is now stored in:
- `data/knowledge/cp_knowledge.json`

The current dictionary contains at least 30 operators, including:
- `HAS`
- `IS_A`
- `CONSTRAINT`
- `COUNT`
- `OPTIMIZE`
- `DECIDE`
- `CONSTRUCT`
- `QUERY_LOOP`
- `STATE`
- `TRANSITION`
- `DP_TRANSITION`
- `INVARIANT`
- `MONOTONE`
- `CASE_SPLIT`
- `REDUCE_TO`
- `PRECOMPUTE`
- `PREFIX`
- `AGGREGATE`
- `RANGE_QUERY`
- `ARRAY_INDEX`
- `SORT_BY`
- `OFFLINE_PROCESS`
- `STATE_GRAPH`
- `SHORTEST_PATH`
- `RELAX_EDGE`
- `NEIGHBOR_EXPAND`
- `GRID_TO_GRAPH`
- `CONNECTIVITY`
- `MERGE_SETS`
- `SUBTREE`
- `TRAVERSE`
- `FEASIBILITY_CHECK`
- `BINARY_SEARCH`

These are not only documentation labels. The reasoner matches them from phrases, goals, domains, and logical frames.

## Memory Schema

The implemented CP memory schema is also stored in:
- `data/knowledge/cp_knowledge.json`

It has four layers.

### Semantic memory
- concepts
- algorithm families
- invariants
- complexity signals

### Structural memory
- goal types
- domain tags
- logical frames
- DSL operators
- constraints

### Episodic memory
- retrieval keys
- candidate algorithms
- selected algorithm
- similar successful episodes
- similar failed episodes
- reranking priors from past validation outcomes

### Procedural memory
- coding rules
- implementation checklist
- validation plan

## Current Reasoning Flow

The current `solve_contest.py` flow is:

1. Normalize the statement.
2. Extract explicit constraints and cue phrases.
3. Infer goal types such as `count`, `optimize`, or `query`.
4. Infer domain tags such as `graph`, `array`, or `dp`.
5. Match logical frames.
6. Match CP DSL operators.
7. Build the four-layer memory projection.
8. Rank candidate algorithms.
9. Generate C++17 code.
10. Run syntax-only compile checking.
11. For supported families, run full compile + sample validation + random/brute-force validation.

## Logical Frames Already Implemented

Examples already wired into the knowledge base:
- `single_source_shortest_path`
- `offline_range_aggregation`
- `dynamic_range_aggregation`
- `dynamic_connectivity`
- `grid_shortest_walk`
- `monotone_answer_search`
- `capacity_value_tradeoff`

## Why This Matters On Hard Problems

The system now tries to solve contest problems through this path:
- phrase -> goal/domain/frame
- frame -> DSL operators
- DSL operators -> hidden concepts
- hidden concepts -> candidate algorithm family
- algorithm family -> implementation plan and syntax-checked code

That makes the project much closer to a reasoning copilot than to a plain code generator.

## RTX 4060 8GB Feasibility

This design is realistic on RTX 4060 8GB because the main pipeline is still lightweight.

Recommended hardware split:
- CPU-first: parsing, DSL matching, frame matching, algorithm ranking, complexity analysis, code generation, local compile check
- GPU-optional: embedding, reranking, or future small-model parser assistance
- avoid for now: heavy fine-tuning or large fully autonomous search models

The current local compiler observation is:
- `g++ = MinGW.org GCC 6.3.0-1`

So generated code intentionally stays in a conservative C++17 subset.

## What Is Still Missing

The current system still needs:
- richer constraint parsing
- larger real CP corpora and editorials in episodic memory
- better parser supervision beyond heuristic labels
- stronger repair beyond a small deterministic rule set

## Next Recommended Steps

1. Grow the recorded episode store with real problem attempts and editorials.
2. Expand validators for monotonicity, nonnegative edges, subtree size, and state-space feasibility.
3. Replace more heuristic labels with trained parser outputs.
4. Strengthen code repair beyond the current deterministic fixes.
5. Expand the algorithm inventory and procedural checklists.

## Validation And Training Data

- sample-based execution validation for supported families
- random / brute-force validation for `prefix_sum_range_query`, `fenwick_tree`, `segment_tree`, `lazy_segment_tree`, `dijkstra_shortest_path`, `dsu_connectivity`, `grid_bfs`, `binary_search_answer`, and `knapsack_dp`
- mixed-format corpus normalization from raw contest statement folders, HTML exports, markdown packs, and zip bundles
- labeled dataset builder from raw contest statements
- prompt/completion SFT export for future small-model training

Main files:
- `src/semop/cp_validation.py`
- `src/semop/cp_dataset.py`
- `src/semop/cp_episode_store.py`
- `src/semop/cp_training.py`
- `tools/cp/prepare_cp_corpus.py`
- `tools/cp/build_cp_dsl_dataset.py`
- `tools/cp/train_cp_parser.py`
- `src/semop/cp_corpus.py`
- `src/semop/cp_repair.py`

The parser training path now supports optional LoRA adapters for small local models, which is the intended route on RTX 4060 8GB.
- `examples/cp_statement_seeds.jsonl`
- `examples/cp_dsl_seed_dataset.jsonl`
- `examples/cp_dsl_seed_sft.jsonl`
- `examples/cp_corpus_inputs\`
- `examples/cp_statement_corpus_expanded.jsonl`
- `examples/cp_dsl_expanded_dataset.jsonl`
- `examples/cp_dsl_expanded_sft.jsonl`




## Episode Growth, Validator Expansion, And Repair Upgrade

### 1. Larger episodic memory from real incidents

New incident ingest flow:
- local JSONL / JSON / CSV archives
- editorials and verdict metadata
- WA / TLE / RE failure kinds
- optional failed code

Main files:
- `src/semop/cp_episode_ingest.py`
- `tools/cp/ingest_cp_episodes.py`
- `examples/cp_incident_cases.jsonl`

Stored extra metadata in `data/cp_episodes.db` now includes:
- `problem_id`
- `editorial_summary`
- `outcome_label`
- `failure_kind`
- `source_kind`

### 2. Validator expansion

Important report fields:
- `checker_kind`
- `failure_type`
- `counterexample_input`
- `expected_output`
- `actual_output`
- `brute_force_cases_run`

This supports:
- category-aware custom checker behavior
- brute-force-backed randomized validation
- explicit counterexample capture for repair

### 3. Cause-specific repair

The repair loop now reacts to:
- compile errors
- runtime errors
- time limits
- output mismatches

Current repair remains conservative, but it is now failure-class aware rather than only string cleanup.


# SemOp Code Structure And Features

## Goal

SemOp Local 4060 is a research-oriented local prototype for structured reasoning. The core idea is:
- turn a query into a structured meaning graph (SMG)
- induce reusable operator families and grammar hypotheses
- retrieve similar past graphs from corpus memory
- apply memory priors during inference
- add symbolic reasoning for domains where pure graph heuristics are weak

The long-term target from the PDF is still larger than the current system: unsupervised operator discovery at corpus scale with stronger transfer and self-learning loops.

## Top-Level Flow

1. `app.py`
   - CLI entrypoint
   - runs the pipeline
   - prints either text or JSON

2. `src/semop/pipeline.py`
   - orchestrates the full reasoning pass
   - stages:
     - `_prepare_graph`: heuristic or LLM extraction
     - `_attach_memory_hints`: attach retrieved hints from similar past graphs
     - `EmergentOperatorInducer.induce(...)`: induce operator candidates and grammar hypotheses
     - `_apply_memory_priors`: promote or inject operator families from memory
     - `_finalize_graph`: apply symbolic solver and sort operators

3. `src/semop/response_synthesizer.py`
   - converts graph output into human-readable reasoning text
   - includes summary, reasoning, symbolic insights, plan, alternatives, operators, grammar, cautions

## Core Data Model

`src/semop/structures.py`
- `Node`: typed concept/entity node
- `Edge`: typed relation with confidence and provenance
- `PlanStep`: executable reasoning step
- `OperatorCandidate`: induced operator/family metadata
- `SymbolicResult`: symbolic answer, evidence, equations, confidence, source
- `StructuredMeaningGraph`: the central intermediate representation

The graph is the main contract between extraction, induction, memory retrieval, validation, symbolic reasoning, and response generation.

## Extraction Layer

### Heuristic extraction
`src/semop/heuristic_extractors.py`
- bag/book access case
- car wash under traffic constraint case
- generic fallback extraction

### LLM extraction
`src/semop/llm_client.py`
- local Transformers integration
- JSON repair loop for malformed structured outputs

## Knowledge And Validation

`src/semop/commonsense_kb.py`
- concept normalization
- default relations and scripts

`src/semop/semantic_operators.py`
- relation application helpers

`src/semop/validation.py`
- graph cleanup and consistency checks

## Operator Learning

`src/semop/emergent_operators.py`
- induces operator candidates from graph structure
- generates grammar hypotheses

`src/semop/corpus_learning.py`
- aggregates patterns over multiple queries
- learns operator families, support, purity, transfer summaries

`src/semop/operator_hierarchy.py`
- learns corpus-scale operator hierarchy
- `L1`: micro operators grouped by concrete operator signature
- `L2`: family operators grouped by reusable operator family
- `L3`: abstract operators clustered from family phrases and signatures
- extracts composition patterns such as `PRECONDITION -> ALTERNATIVE_SEARCH`
- outputs support, purity, confidence, examples, types, and child/parent links

`src/semop/operator_registry.py`
- turns hierarchy summaries into a typed operator registry
- maps `L2` families to `L3` abstract parents
- attaches type contracts back onto induced operators during inference
- enables abstract-parent promotion and sibling-family injection from memory

## Memory Layer

`src/semop/corpus_store.py`
- SQLite storage for train/test examples and learning summaries

`src/semop/memory_retrieval.py`
- lexical or embedding-backed similarity search
- supports local embedding cache via `SEMOP_EMBED_MODEL_PATH`

`src/semop/memory_prior_eval.py`
- evaluates baseline vs memory-prior reasoning behavior

## Symbolic Layer

### Arithmetic
`src/semop/symbolic_arithmetic.py`
- expression parser with nested arithmetic, unary minus, ceil/floor
- line-item split problems
- monthly sales multiplier pattern
- ratio-share problems
- unit-rate problems
- fraction-weighted total problems
- round-up pack counting problems

### Document evidence
`src/semop/symbolic_document.py`
- heading-aware block construction
- table row parsing for pipe/tab/multi-space formats
- question segment detection
- overlap-based evidence ranking
- number overlap bonus
- finance keyword bonus for FinanceBench-style questions
- returns up to two supporting blocks or rows

### Symbolic orchestrator
`src/semop/symbolic_reasoners.py`
- routes to arithmetic first
- falls back to document grounding when arithmetic does not apply
- injects symbolic operators and plan steps into the graph

## Public Corpus Pipeline

`prepare_public_dataset.py`
- normalize local public datasets into a common JSONL format

`build_corpus.py`
- deduplicate and augment seed queries

`ingest_corpus.py`
- run the pipeline over prepared examples and store them in SQLite

`download_public_dataset.py`
- download or copy remote datasets

`ingest_public_manifest.py`
- batch ingest datasets from a manifest

`write_curated_manifest.py`
- generate curated preset manifests

`retrain_memory.py`
- relearn family summaries from stored memory

`evaluate_transfer.py`
- held-out transfer evaluation

`evaluate_memory_prior.py`
- compare baseline vs memory-prior inference

`refresh_memory_store.py`
- rerun the current pipeline over stored queries so stale graphs can be replaced with newer diversity-aware operators

`refresh_and_rebuild_hierarchy.py`
- refresh every source in a SQLite corpus and rebuild per-source/global hierarchies in one pass

`build_operator_hierarchy.py`
- learn an L1-L3 operator hierarchy from graphs already stored in SQLite
- optionally persist the hierarchy summary and nodes back into the store
- export a JSON summary for inspection or downstream analysis

## Curated Dataset Scope

Current presets include:
- GSM8K
- GSM-IC
- OfficeQA
- BIG-Bench Mistake logical deduction
- FinanceBench

This gives the project broad reasoning variety, but symbolic coverage is still sparse compared with full dataset diversity.

## Current Refactoring State

Recent refactor changes:
- split symbolic logic into arithmetic and document modules
- reduced `pipeline.py` into clearer phase boundaries
- made `SymbolicResult` explicit in the graph schema
- kept symbolic integration behind one orchestrator class

This makes it easier to extend domain-specific solvers without pushing more special-case logic into the main pipeline.

## Current Limitations

- operator self-learning now includes corpus-scale L1-L3 hierarchy extraction, but execution still depends on a lightweight registry rather than a full self-improving training loop
- memory prior raises operator confidence more than it raises true reasoning accuracy today
- arithmetic coverage is still pattern-based rather than parser-complete
- document grounding is sentence-level and does not yet perform table parsing or multi-hop evidence chaining
- OfficeQA and FinanceBench would still benefit from task-specific parsers for tables, headings, and report structure

## Suggested Next Refactors

1. Move memory prior logic from `pipeline.py` into its own module.
2. Add a symbolic registry so new domain solvers can be plugged in declaratively.
3. Split heuristic extractors by domain instead of keeping all heuristics in one file.
4. Add task-level evaluation scripts for symbolic accuracy, not just operator reuse/confidence.
5. Add hierarchy-level metrics such as abstraction stability, composition reuse, and OOD family attachment rate.


## PDF Alignment Notes

The PDF emphasizes semantic operators, typed intermediate graphs, operator composition, abstraction layers, and auditability. The latest symbolic refactor moves a small part of the system closer to that direction:
- arithmetic reasoning now uses an explicit expression parser instead of direct hard-coded arithmetic only
- document reasoning now treats headings and table-like rows as typed evidence blocks instead of plain sentence bags
- both modules preserve symbolic traces inside `SymbolicResult`, which is closer to operator-signature and audit-oriented reasoning than raw answer generation

This still does not complete the PDF vision of corpus-scale unsupervised operator formation, but it reduces two immediate gaps in domain reasoning quality.

The new hierarchy learner is the first direct step toward that target because it turns many stored graphs into a reusable abstraction stack instead of stopping at per-query operator induction.

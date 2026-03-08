# Corpus-Scale Operator Hierarchy Plan

## Target

Build a corpus-scale unsupervised operator hierarchy that moves from per-query induced operators to reusable abstraction layers aligned with the PDF:
- L1: concrete operator signatures
- L2: reusable operator families
- L3: abstract operator clusters
- composition patterns across corpora

## Phase 1: Stable hierarchy extraction

Status: completed

Goals:
- learn L1-L3 nodes from stored structured meaning graphs
- persist hierarchy runs and node payloads in SQLite
- expose a CLI to rebuild hierarchy from any corpus source and split
- verify support, purity, confidence, type flow, and composition signatures

Implemented:
- `src/semop/operator_hierarchy.py`
- `build_operator_hierarchy.py`
- SQLite persistence in `src/semop/corpus_store.py`

## Phase 2: Typed operator registry

Status: in progress

Goals:
- convert hierarchy nodes into executable registry entries
- attach input/output type contracts to `L3 -> L2 -> L1`
- log operator composition traces during inference
- allow memory prior to promote not only families but abstract operator parents

## Phase 3: Self-improving corpus loop

Status: planned

Goals:
- periodically rebuild hierarchy from accumulated corpora
- compare successive hierarchy runs for stability and drift
- promote stable high-purity abstractions into the registry
- flag ambiguous or low-purity clusters for review or isolation

## Phase 4: Evaluation

Status: planned

Metrics:
- family purity
- abstraction stability across runs
- composition reuse rate
- OOD family attachment rate
- downstream symbolic accuracy impact
- invalid-advice reduction

## Immediate execution checklist

1. Keep ingesting larger public corpora into SQLite memory.
2. Rebuild L1-L3 hierarchy per source and per mixed corpus.
3. Inspect recurring composition chains.
4. Turn high-support L3 nodes into registry seeds.
5. Refresh stored corpora so new diversity-aware induction replaces stale graphs.
6. Connect registry-guided execution back into the pipeline.

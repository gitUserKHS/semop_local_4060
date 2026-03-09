# PDF Alignment Notes

## What The PDF Emphasizes

From the extracted PDF, the strongest requirements are:
- move beyond new tokens into reusable semantic operators
- treat operator abstraction as a research object
- build a typed structured meaning graph with affordance, script, part-whole, and constraint information
- compose operators with type constraints
- preserve provenance and operator signatures for auditability
- evaluate purity, transfer, ambiguity, executability, and invalid-advice behavior
- use lightweight local experiments with external memory and evaluation loops

## What Was Already Present

Before this change, the codebase already had:
- SMG-style intermediate graphs
- relation provenance
- memory retrieval and memory prior
- operator induction and grammar hypotheses
- corpus learning and transfer evaluation
- symbolic hooks for arithmetic and document QA

## What This Change Adds In Direct Response To The PDF

### 0. Corpus-scale operator hierarchy learning
The PDF places heavy emphasis on abstraction layers for semantic operators rather than one-off per-query reasoning artifacts.

The new hierarchy learner moves the codebase in that direction because it:
- learns `L1` micro operators from concrete induced operator signatures
- aggregates them into `L2` reusable operator families
- clusters those families into `L3` abstract operator groups
- records composition patterns across stored corpora
- stores hierarchy summaries back into SQLite for repeated analysis

Relevant files:
- `src/semop/operator_hierarchy.py`
- `tools/corpus/build_operator_hierarchy.py`
- `src/semop/corpus_store.py`


### 1. Expression-parser arithmetic
The PDF repeatedly frames operator composition as typed semantic transformation rather than raw token compression.

The new arithmetic layer is now closer to that direction because it:
- builds arithmetic expressions explicitly
- parses nested expressions instead of hard-coded final arithmetic only
- supports `ceil(...)` for discrete planning-style cases
- leaves equation traces inside `SymbolicResult`

Relevant file:
- `src/semop/symbolic_arithmetic.py`

### 2. Header/table-aware document grounding
The PDF stresses structured intermediate representations, affordance-aware reasoning, and auditability.

The document layer is now closer to that direction because it:
- converts raw text into typed evidence blocks
- distinguishes headings, table rows, and sentence blocks
- scores rows using token overlap, numeric overlap, finance-specific hints, and heading overlap
- returns evidence blocks that are easier to audit than plain sentence bag matching

Relevant file:
- `src/semop/symbolic_document.py`

## Current Gap Against The Full PDF Vision

The project still does not fully satisfy the larger research target in the PDF:
- operator abstraction is still induced heuristically, not learned at full corpus scale from a persistent self-improving loop
- operator composition is not yet treated as a general typed grammar parser across all domains
- symbolic execution is still domain-fragmented rather than unified under one operator registry
- auditability exists, but family-level purity and transfer are still stronger than actual downstream correctness

## Immediate Next Step Suggested By The PDF

The next high-value step is:
- move from domain-specific symbolic modules to a typed operator registry with composition logging and family-level execution traces
- use the new hierarchy output as the registry seed, so `L3 -> L2 -> L1` becomes executable rather than only descriptive

That would connect the current symbolic layer more directly to the PDF's ideas of operator hierarchy, composition grammar, and human-auditable reasoning.


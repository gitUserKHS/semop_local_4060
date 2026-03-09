# VLSO Implementation Plan

## Goal

Build a shared semantic-operator layer so that language inputs and visual observations are both converted into the same world-model graph before reasoning.

The target is not a raw end-to-end vision model yet. The first implementation stage is an operator-space prototype.

## Core Definition

VLSO means:
- language -> semantic operator graph
- vision -> semantic operator graph
- both graphs -> shared world model
- shared world model -> reasoning, planning, and validation

## Phase 1 Scope

### 1. Shared operator space

Define a first operator dictionary across these axes:
- object
- relation
- topology
- geometry
- action
- constraint
- transformation

### 2. Language parser

Reuse the existing structured-meaning pipeline and convert its graph into VLSO entities, relations, and operators.

### 3. Visual parser

For now, accept:
- structured observation JSON
- simple textual scene descriptions

This keeps the prototype stable on a local machine and avoids pretending that raw-pixel parsing is already solved.

### 4. Cross-modal aligner

Merge language and visual graphs into one shared world model and run a first set of inference rules.

### 5. Demo and tests

Ship a runnable CLI and regression tests so the layer is concrete, not only theoretical.

## Current Implementation Files

- `src/semop/vlso/types.py`
- `src/semop/vlso/operator_registry.py`
- `src/semop/vlso/language_parser.py`
- `src/semop/vlso/visual_parser.py`
- `src/semop/vlso/aligner.py`
- `src/semop/vlso/reasoner.py`
- `vlso_demo.py`

## Planned Next Steps

### Phase 2
- accept detector output from a real vision model
- align objects and parts across frames
- add geometry and topology rules beyond simple heuristics

### Phase 3
- connect VLSO world models to robot or simulator execution
- use the same operator space for math, geometry, and embodied planning
- learn visual operator induction from corpora

## Code Cleanup Strategy

Because this repository has grown across multiple tracks, new multimodal work is isolated under `src/semop/vlso/` instead of being mixed into the existing CP, ops, and symbolic files.

That gives a cleaner top-level split:
- `src/semop/vlso/`: multimodal operator-space prototype
- `src/semop/contest_*`, `cp_*`: competitive-programming stack
- `src/semop/domain_*`, `ops_*`: operations copilot stack
- `src/semop/symbolic_*`, `olympiad_*`: symbolic math and proof stack


### 6. Visual embedding memory and geometry/topology extraction

The next implementation stage now includes:
- `src/semop/vlso/vision_backbones.py`
- `src/semop/vlso/embedding_store.py`
- `src/semop/vlso/geometry_topology.py`
- `tools/vlso/index_vlso_visual_memory.py`

Current design:
- default visual embeddings come from a deterministic local `token_geometry_v1` backbone
- external backbones such as DINOv2 or OpenCLIP are represented as adapter slots
- visual observations can be indexed in a SQLite embedding DB and searched by cosine similarity
- geometry and topology are extracted from bounding boxes and line segments before cross-modal alignment


### 7. Detector adapters and shape reasoning

The VLSO stack now also includes:
- `src/semop/vlso/detector_adapters.py`
- `src/semop/vlso/geometry_reasoner.py`

This means the current path is:
- detector output -> visual observation
- visual observation -> geometry/topology extraction
- geometry/topology -> shape hypotheses and logical constraints
- shape hypotheses -> shared world model -> cross-modal reasoning


### 8. Raw image parsing and local backbones

The VLSO stack now also includes:
- `src/semop/vlso/image_parser.py`
- `src/semop/vlso/vision_backbones.py`
- `vlso_demo.py`

Current path:
- raw image path -> connected-component parser -> shape candidates -> geometry/topology extraction -> shared world model
- local DINOv2 or OpenCLIP checkpoint -> image embedding -> visual memory retrieval
- if the local checkpoint is unavailable -> fallback to `token_geometry_v1`

This means VLSO is no longer limited to hand-authored observation JSON. It can now bootstrap from raw images on a local machine, while still keeping detector-output and local-backbone hooks separate and auditable.

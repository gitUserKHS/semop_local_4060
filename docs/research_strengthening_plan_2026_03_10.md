# Research-Backed Strengthening Plan (2026-03-10)

## Purpose

This note summarizes practical research directions for strengthening three SemOp tracks at once:
- competitive-programming reasoning
- visual reasoning / VLSO
- operator-based logical reasoning

The goal is not to chase a single giant model. The stronger route for this codebase is:
- better world models
- stronger verifier loops
- richer memory
- learned operator induction from small or weakly labeled data

## What Was Implemented In This Step

### Review-aware retraining

Approved self-training clusters can now be exported back into reusable stores.

Code:
- `src/semop/vlso/review_retrain.py`
- `tools/vlso/retrain_from_cluster_reviews.py`
- `semop_easy_gui.py`

Behavior:
- read cluster summary JSON
- read human review decisions
- export approved labels to JSONL
- build an approved concept store
- rebuild an approved operator store
- let the easy GUI point vision QA directly at the newly retrained stores
- prioritize cluster review using low-margin and mixed-label ambiguity scoring
- rerank top CP candidates with verifier results plus structural-alignment scoring instead of trusting a single heuristic pick

## Research Directions

### 1. Competitive programming

Most useful pattern:
- generate multiple structured solution sketches
- validate with execution
- repair with counterexamples
- rerank by verifier success rather than surface plausibility

Why this matches SemOp:
- the repo already has DSL extraction, validator, repair loop, and episodic memory
- the main gap is stronger learned parsing and richer verifier-guided search

Recommended research themes:
- verifier-guided search for code generation
- execution-based ranking on contest tasks
- larger held-out parser benchmarks for `statement -> frame -> algorithm -> DSL`

Practical next steps for this repo:
- train the CP parser with LoRA on the generated geometry/general bundles
- add multiple candidate sketches per problem instead of one heuristic path
- log repair traces as reusable episodic patterns

Useful sources:
- AlphaCode paper: https://arxiv.org/abs/2203.07814
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- ReAct: https://arxiv.org/abs/2210.03629

### 2. Visual reasoning / VLSO

Most useful pattern:
- stronger perception backbone
- object/part segmentation
- geometry/topology extraction
- operator induction over compact prototypes
- active review on uncertain clusters

Why this matches SemOp:
- the repo already has raw-image parsing, detector adapters, geometry extraction, concept memory, operator memory, and self-training
- the main gap is stronger perception and better cluster quality on real scenes

Recommended research themes:
- self-supervised visual representations for few-shot transfer
- segmentation-assisted operator grounding
- active learning on cluster summaries instead of dense manual labeling

Practical next steps for this repo:
- use DINOv2 features as the default geometry/object prototype backbone when available
- connect segmentation payloads more aggressively into operator learning
- prioritize cluster review on low-margin or mixed-label clusters
- expand evaluation beyond bags to box, drawer, door, bottle, tool

Useful sources:
- DINOv2 official repo: https://github.com/facebookresearch/dinov2
- SAM 2 official repo: https://github.com/facebookresearch/sam2
- CLEVR dataset/project: https://cs.stanford.edu/people/jcjohns/clevr/
- GQA dataset/project: https://cs.stanford.edu/people/dorarad/gqa/index.html

### 3. Logical/operator learning

Most useful pattern:
- induce compact operator families
- store prototypes instead of raw examples only
- run search over explicit states and operators
- let human review correct operator clusters, not only final answers

Why this matches SemOp:
- this codebase is already built around typed operators, memory, and verification
- the most leverage comes from better operator induction and reuse across domains

Recommended research themes:
- neuro-symbolic reasoning with explicit intermediate states
- process supervision instead of answer-only supervision
- retrieval over operator families and prior episodes

Practical next steps for this repo:
- attach operator-family priors to CP parser output ranking
- use approved VLSO clusters to expand cross-domain operator families
- build a benchmark that tests whether the same operator family transfers across ops, CP, and vision tasks

Useful sources:
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- ReAct: https://arxiv.org/abs/2210.03629

## Recommended Priority For SemOp

### Immediate
- use GUI/CLI review-aware retraining to clean VLSO operator stores
- run CP LoRA parser training on the geometry and general bundles
- expand evaluation sets and stop relying only on starter examples

### Next
- multi-candidate CP search with verifier reranking
- segmentation-assisted VLSO operator extraction
- transfer benchmark for shared operator families

### Later
- cross-modal world model training where language and vision both map into the same operator space with stronger learned parsers

## Commands

### Retrain from approved VLSO clusters

```bash
python tools\vlso\retrain_from_cluster_reviews.py ^
  --summary data\vlso_geometry_pipeline_gui\geometry_cluster_summary.json ^
  --reviews data\vlso_geometry_pipeline_gui\geometry_cluster_reviews.json
```

### Easy GUI

```bash
.\.venv312\Scripts\python.exe semop_easy_gui.py
```

Then open `http://127.0.0.1:8770` and use:
- `Run geometry self-training`
- `Retrain from approved clusters`


## Structural-First Visual Update (2026-03-11)

See also `docs/jepa_relevance_and_integration.md` for the JEPA-specific interpretation and integration choice.

The current VLSO direction now prioritizes `geometry/topology -> structural operator -> optional semantic prior` instead of `label -> reasoning`.

Implementation consequences:
- raw-image parsing now exposes hole structure, fill ratios, and perimeter-derived compactness before weak labels are applied
- detector and segmentation payloads preserve `part_of`, relation metadata, and mask-derived hole/fill features for structural grounding
- evaluation now tracks `operator_recall` and `operator_binding_recall`, not just entity/relation overlap

Research judgement:
- DINOv2-style self-supervised visual features remain useful for retrieval and prototype memory, but they are not treated as the primary explanation layer
- segmentation-centric systems such as SAM 2 are best treated as region proposal/grounding backbones that feed structural operators rather than as end-task solvers
- the right benchmark target for this repository is `operator recovery under visual variation`, because that better matches the project goal of reusable logical operators


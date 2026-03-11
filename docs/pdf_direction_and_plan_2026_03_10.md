# PDF-Driven Direction And Implementation Plan (2026-03-10)

## Source documents
- `vlso_research_report_2026_03_10.pdf`
- `VLSO_제안서_개정본.pdf`
- `2603.03269v1.pdf`

## Core direction extracted from the PDFs

### 1. Operator-centric shared semantic space
The two VLSO documents converge on the same thesis: the real novelty is not just multimodality, but a shared operator space where language and vision are both converted into typed structure before reasoning.

Implementation decision:
- keep VLSO centered on typed entities, relations, operators, constraints, and audit traces
- keep induction and execution separate
- prefer reusable operators over raw end-to-end label generation

### 2. Few-shot structural learning, not only class learning
The VLSO documents repeatedly push toward learning structure with a small number of examples. That means the system should not only memorize `bag` or `zipper`, but also reusable relation patterns such as container-body, opening-control, and attached-grasp.

Implementation decision:
- add few-shot visual operator prototypes
- let a small labeled set teach relation patterns that can generalize across different objects

### 3. Hybrid memory is the right scaling mechanism
`2603.03269v1.pdf` (LoGeR) argues for a hybrid memory design: lossless local memory plus compressed global memory. Although LoGeR is for long-context geometric reconstruction, that design transfers well to VLSO.

Implementation decision:
- local memory: concept exemplars and nearest-neighbor visual matches
- global memory: compressed visual operator prototypes
- fused reasoning: combine both when assigning priors to new image regions

## What was implemented from this direction
- `src/semop/vlso/operator_learning.py`
  - few-shot visual operator prototype memory and trainer
- `src/semop/vlso/hybrid_memory.py`
  - local exemplar + global operator fusion
- `src/semop/vlso/object_reasoner.py`
  - hybrid memory now injects fused label priors
- `tools/vlso/train_visual_operators.py`
  - CLI for building operator memory from a small labeled image set
- `vlso_demo.py`
  - can now load `--operator-store`

## Resulting reasoning stack
1. Parse the image into candidate regions and geometry/topology features.
2. Use weak learned affordance rules for a first-pass prior.
3. Query local concept memory for exemplar-level similarity.
4. Query global operator memory for abstract structural similarity.
5. Fuse both sources into label and affordance priors.
6. Build the shared world model and answer from grounded structure.

## Why this is more fundamental
This moves the project away from only patching answers or labels. It teaches the system how a region behaves in a scene graph:
- container body
- access opening control
- attached grasp part
- container access pattern

That is closer to the PDFs than a plain object classifier.

## Next recommended step
Grow the operator vocabulary beyond bag-like scenes:
- drawer / cabinet / door / handle / bottle / tool
- add negative operator prototypes for border fragments, reflections, and decorative patterns
- build a small evaluation set where the target is operator recovery, not just object naming

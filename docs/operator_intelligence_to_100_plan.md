# Toward 100 Percent Operator Intelligence

## Aim

The project target is not a bigger chatbot. The target is a reusable operator-centered intelligence stack where:
- visual input is converted into geometry, topology, and structural operators
- language input is converted into hidden goals, hidden premises, and operator constraints
- both modalities meet in a shared world model
- answers are filtered through verifier-backed goal preservation

## Research grounding

This plan follows five research directions that fit the current codebase.

1. JEPA-style latent prediction
   Source: I-JEPA and V-JEPA.
   Use: predict structural operator priors from context instead of reconstructing pixels.

2. self-supervised visual representation
   Source: DINOv2.
   Use: strong features as priors, but not as the final meaning layer.

3. segmentation-first grounding
   Source: SAM 2.
   Use: better region and part grounding before operator induction.

4. geometry and topology primitives
   Source: OpenCV contour and line extraction.
   Use: edges, corners, parallel structure, holes, and containment as the first layer of meaning.

5. false-assumption and atomic-assumption evaluation
   Use: hidden-premise extraction should be evaluated as premise recovery plus goal-preservation quality, not only final answer quality.

## What 100 percent means in this repository

A practical 100 percent is not AGI. It means the project has closed the core operator-intelligence loop:
- operator-first architecture is stable
- hidden-premise reasoning is strong across daily QA, VLSO, and CP problem parsing
- the shared world model is cross-modal rather than siloed
- raw visual reasoning works on real images with verifier-backed operator recovery
- operator transfer across domains is measured, not assumed
- memory, review, and retraining loops improve the core metrics instead of adding noise

## Current bottlenecks

- hidden-premise retrieval is improved but still partly heuristic
- raw-image perception still needs stronger segmentation-aware grounding
- CP learned parser has scaffold and evaluation, but not the final trained path
- the real-image VLSO benchmark is still small
- operator algebra exists, but transfer and decomposition are not yet benchmarked at real scale

## The six practical stages to 100 percent

### Stage 1. Premise-first engine stabilization
Success target:
- critical premise recall >= 0.98
- hidden goal recall >= 0.97
- unsupported premise precision >= 0.92
- clarification accuracy >= 0.9 on daily/visual/CP pairs

### Stage 2. Structural-first VLSO perception
Success target:
- operator recall >= 0.9 on real-image held-out cases
- operator binding recall >= 0.85
- operator-premise support >= 0.85

### Stage 3. Cross-modal operator alignment
Success target:
- text-side premise recovery helps visual QA
- visual structure helps text-side plan validation
- common world-model fields are reused across VLSO, ops, and CP

### Stage 4. CP parser-first intelligence
Success target:
- learned parser beats heuristic parser on held-out DSL/frame/operator labels
- premise augmentation improves frame/operator extraction on hidden-constraint cases

### Stage 5. Memory and review as learning loops
Success target:
- review-approved cases improve benchmarked metrics, not only demos
- memory layers show measurable precision/recall gains

### Stage 6. Operator transfer benchmark
Success target:
- operators learned in one domain help another domain
- decomposition recovery is stable
- operator-premise support tracks real transfer

## Immediate next implementation order

1. trainable script-compatibility scorer with lightweight learned weights
2. real-image VLSO benchmark growth and segmentation-aware grounding
3. CP parser LoRA training and compare-mode evaluation
4. operator-transfer benchmark across modalities
5. progress snapshots tied to real evaluator outputs

## Progress accounting

The repository should track at least these axes:
- operator architecture
- premise reasoning
- shared world model
- CP structuring
- raw visual reasoning
- general operator transfer

These are estimated automatically by `tools/eval/evaluate_operator_intelligence_progress.py` from the common evaluator snapshot.

## Primary references

- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA code: https://github.com/facebookresearch/jepa
- DINOv2: https://arxiv.org/abs/2304.07193
- SAM 2: https://arxiv.org/abs/2408.00714
- OpenCV contours: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
- OpenCV Hough lines: https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html

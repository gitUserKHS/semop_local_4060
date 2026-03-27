# World-Model Math Research Notes (2026-03-22)

## Goal

Use world-model ideas to improve geometry and high-difficulty math solving in this repository without pretending we already have a giant fully trained theorem prover.

## Primary Sources Checked

- I-JEPA paper: https://arxiv.org/abs/2301.08243
- V-JEPA official code and paper link: https://github.com/facebookresearch/jepa
- World Models (Ha and Schmidhuber): https://doi.org/10.5281/zenodo.1207631
- Large Concept Models official code: https://github.com/facebookresearch/large_concept_model
- Google DeepMind on IMO gold-medal Deep Think, with discussion of AlphaProof and AlphaGeometry 2: https://deepmind.google/en/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/

## What Matters For This Repository

Inference from the sources above:

- JEPA is most useful here as a design rule, not as a promise to train a huge new model immediately.
  - predict latent structure from partial context
  - avoid treating raw surface reconstruction as the main objective
  - prefer representation-space forecasting of what matters next

- World Models and Large Concept Models matter because they separate state-building from action or next-token generation.
  - build a compact state first
  - plan or predict in that state
  - only then decode an answer

- AlphaGeometry and AlphaProof matter because they show that difficult math improves when learned priors are paired with symbolic search and explicit verification.
  - neural prior only is not enough
  - symbolic search only is brittle
  - the hybrid wins because the verifier can reject attractive nonsense

## Implementation Choice In This Repo

The new `src/semop/world_model_math.py` follows that hybrid reading.

1. Build a latent math world model from the text problem and optional diagram.
2. Predict a `strategy prior` before choosing an answer.
3. Ask the solver ensemble for candidates.
4. Rank candidates by verification and world-model alignment.
5. Return beginner-friendly next actions when proof confidence is not strong enough.

## Why This Is The Right Level Today

This repository already has:
- `StructuredMeaningPipeline`
- `HardProblemEngine`
- `OlympiadReasoner`
- VLSO geometry parsing and JEPA-inspired predictive priors

So the highest-leverage step is not training a giant new model first. It is wiring these pieces into a better world-model loop for geometry and hard math, then making that usable from a beginner-facing GUI.

## Current Limits

This is still not a full formal theorem prover or a guaranteed research-math engine.

The current implementation is best understood as:
- a stronger math/geometry orchestration layer
- a world-model-first strategy prior
- a diagram-aware hybrid verifier
- a beginner-facing interface for experimentation

## Immediate Next Step

If this path proves useful, the next research-grade upgrade is:
- train a real latent strategy predictor from approved proof traces and geometry diagrams
- collect diagram-plus-proof supervision for geometry problems
- add formal proof export or theorem-checker integration for the highest-confidence branch only

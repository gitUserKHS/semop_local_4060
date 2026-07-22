# Ultimate AGI Goal

Last updated: 2026-07-16

This document records the user's explicitly stated long-term north star so future developers do not optimize only for narrow demos.

## Ultimate Target

Build a broadly capable AGI system that can move toward real-world human-level capability, including:
- autonomous driving
- robot control and safe embodied action
- mathematical problem solving and proof generation
- general prompt understanding, hidden-context reasoning, and logical response generation
- multimodal image and video understanding
- creativity, intuition, and novel concept fusion

## Engineering Interpretation

This repo should treat the following as one shared doctrine rather than separate toy features:
- a small shared controller that composes typed operators instead of directly emitting answers
- operator algebra
- shared world models
- multimodal grounding
- planning and control priors
- memory and analogy
- compiler-style verification
- self-improvement and reviewed-corpus growth

## Product Direction

The user does not want only a research stack. The project should move toward commercializable deployment by improving:
- readiness tracking
- safety gates
- grounded explanation fidelity
- repeatable data collection and training loops
- beginner-friendly CPU-only operation on ordinary PCs, with RTX 4060 8GB class hardware as an optional acceleration tier
- useful adaptation from small reviewed datasets rather than dependence on massive private corpora

## Immediate Build Priorities

1. Strengthen prompt understanding and hidden-context reasoning.
2. Prove low-resource transfer of shared typed operator families across language, mathematics, and algorithmic environments.
3. Strengthen multimodal image and video situation understanding through proposed facts and independent verifiers.
4. Strengthen math, planning, and embodied-world-model scaffolds.
5. Add concept-fusion creativity that remains auditable.
6. Track ultimate AGI readiness and commercialization blockers explicitly.
7. Improve capability per label and per compute budget before increasing model or dataset scale.

## Anti-Goals

Avoid optimizing for these unless they directly strengthen the priorities above:
- disconnected demos that do not feed the shared world-model stack
- larger raw generation without grounding or verification
- improvements that require much larger datasets or hardware without an ablation proving the added value
- specialist UX that fragments the beginner workflow
- metrics that look good but do not improve real deployment readiness

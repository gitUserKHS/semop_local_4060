# Recursive Self-Evolution Research Notes

Last updated: 2026-03-27

This note records the research ideas used to build the repo's recursive self-improvement loop.
The implementation target is not unrestricted AGI. It is a practical, RTX 4060-friendly recursive improvement loop over environment-specific reasoning programs.

## Primary Sources

- [AlphaEvolve (Google DeepMind)](https://deepmind.google/discover/blog/alphaevolve-a-coding-agent-for-scientific-and-algorithmic-discovery/)
- [FunSearch (Nature)](https://www.nature.com/articles/s41586-023-06924-6)
- [Voyager project](https://voyager.minedojo.org/)
- [Reflexion (arXiv)](https://arxiv.org/abs/2303.11366)
- [Self-Refine project](https://selfrefine.info/)

## What These Papers Suggest

1. AlphaEvolve and FunSearch:
- Keep a population or database of candidate programs.
- Use an automated evaluator as the selection rule.
- Preserve strong candidates, mutate or recombine them, and keep searching.
- Diversity matters, because greedy search collapses too early.

2. Voyager:
- Lifelong embodied progress works better when the system stores reusable skills instead of re-solving from scratch each time.
- Curriculum and skill reuse matter as much as raw generation quality.

3. Reflexion and Self-Refine:
- Failed attempts should leave structured reflection memory.
- The next attempt should be changed by that reflection, not merely rerun.

## Why The Repo Uses A Different Implementation

This repo targets RTX 4060 8GB and operator-algebra-first reasoning.
So the implementation does not attempt expensive foundation-model recursive weight training.
Instead it evolves local improvement programs made of:
- focus tags
- seed query sets
- self-evolution pressure
- grounded correction loops
- action rehearsal synthesis

## Adaptation In This Repo

The recursive runner therefore does this:
- build a diverse seed population of local improvement programs
- run each candidate through the environment-specific adaptive learning loop
- score candidates with automated metrics such as local intelligence, grounded reasoning, self-reflection, embodied planning, and ready-axis count
- mutate weak candidates based on weak axes
- recombine strong parents
- keep a diversity-aware archive
- deploy the best evolved local bundle back into the canonical environment

## Current Product Interpretation

This is not yet a free-ended self-rewriting AGI.
It is a practical recursive self-improvement scaffold for:
- hidden-context reasoning
- grounded answer correction
- multimodal environment understanding
- reusable local action rehearsal
- environment-specific self-improvement

## Implementation Targets

Look at these files for the active implementation:
- `src/semop/recursive_self_evolution.py`
- `src/semop/adaptive_environment_learning.py`
- `semop_studio_gui.py`

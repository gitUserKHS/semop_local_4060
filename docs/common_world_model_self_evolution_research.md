# Common World Model + Self-Evolution Research Notes

## Goal

This project is aiming for a single operator-first world-model stack that can support text reasoning, image/video grounding, math, competitive programming, and future robotics-like control under RTX 4060 8GB constraints.

The practical doctrine is:

1. Convert every domain result into one shared `entity-relation-state-event` world model.
2. Run reasoning over that shared state instead of over isolated domain payloads.
3. Let self-improvement loops mutate prompts, operators, memories, and programs against verifier signals.
4. Store integrated world traces so later training uses the same latent structure that runtime reasoning uses.

## Why this direction is defensible

### World-model lineage

- [World Models](https://arxiv.org/abs/1803.10122) (Ha and Schmidhuber, submitted March 27, 2018) showed that a compact agent can learn from compressed spatial-temporal world representations and can even train inside a dreamed environment before transferring back to the real environment.
- [Mastering Atari, Go, chess and shogi by planning with a learned model](https://www.nature.com/articles/s41586-020-03051-4) (MuZero, published December 23, 2020) is the strongest classical argument that planning over a learned model is better than acting from raw observations alone.
- [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (DreamerV3, submitted January 10, 2023; revised April 17, 2024) argues that one algorithm can cover 150+ tasks when it keeps environment modeling and imagination inside the control loop.
- [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) (LeCun, version 0.9.2 dated June 27, 2022) is the clearest high-level case for hierarchical predictive world models, intrinsic motivation, and planning as core AGI ingredients.

### Self-improvement and reflective learning

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) (submitted March 20, 2023; revised October 10, 2023) shows that textual reflection memory can improve future trials without full weight updates.
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) (submitted March 30, 2023; revised May 25, 2023) is the cleanest lightweight loop for generator -> critique -> refine using the same model.
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) (submitted May 25, 2023; revised October 19, 2023) is important because it stores reusable executable skills instead of solving every step from scratch.
- [Self-Discover: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/abs/2402.03620) (submitted February 6, 2024) supports the idea that the model should choose and compose reasoning structures on its own instead of relying on one static prompt shape.
- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) (submitted March 14, 2024; revised March 18, 2024) supports latent internal reasoning that improves later outputs even when the final answer stays concise.
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) (submitted January 18, 2024; revised March 28, 2025) is relevant for evaluator bootstrapping because it explicitly trains a model to improve both output quality and its own reward quality.

### Program evolution and evaluator-driven search

- [Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6) (FunSearch, published December 14, 2023; Nature 2024 issue) is the best evidence that LLM creativity becomes far more reliable when proposals are filtered by automated evaluators and kept as a diverse population of programs.
- [AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (Google DeepMind blog, May 14, 2025) extends that doctrine from single functions to larger code-level algorithm improvement with automated scoring.
- [Dyna-Think: Synergizing Reasoning, Acting, and World Model Simulation in AI Agents](https://arxiv.org/abs/2506.00320) (submitted May 31, 2025; revised October 10, 2025) directly argues that action quality tracks world-model quality and that explicit world simulation should be part of the training loop.
- [WebEvolver: Enhancing Web Agent Self-Improvement with Coevolving World Model](https://arxiv.org/abs/2504.21024) (submitted April 23, 2025; revised August 21, 2025) is especially aligned with this repository because it coevolves the agent and the world model instead of treating the world model as a frozen side module.

## Translation into repo architecture

### 1. One shared world projection layer

Every major domain should project into the same `SharedWorldModel` shape.

Required projections:

- ops graph -> entities, relations, plan events, operator trace
- image world -> grounded entities, relations, affordances, attention, agent intent
- video world -> temporal entities, state changes, appearance/disappearance events
- math world -> strategy priors, checks, candidate answers, verification events
- cp world -> problem goals, constraints, chosen algorithm, compile/validation events
- self-evolution world -> weaknesses, strategies, reflections, repair gains, approved traces

### 2. One shared reasoning summary

The runtime and training stack should both depend on the same derived summary fields:

- `primary_goal`
- `active_domains`
- `blockers`
- `prerequisites`
- `alternatives`
- `next_steps`
- `evidence`
- `operator_trace`
- `warnings`

This keeps the user-facing answer, the evaluator, and the distillation trace aligned.

### 3. Self-evolution should mutate structures, not just text

The self-improvement loop should be able to change:

- which operators are invoked
- which evidence gets promoted
- which hidden premises are treated as required
- which world-model relations are trusted or rejected
- which plan order is rehearsed
- which program survives in the population

### 4. Evaluators should stay cheap and local

Under RTX 4060 8GB constraints, the main loop should prefer:

- verifier-backed symbolic scoring
- compact distillation data with structured traces
- lightweight route/question understanding models
- reusable operator libraries and memory
- selective retraining only on promoted traces

This is more compatible with local deployment than always-on heavyweight generation.

## What was wired in this repo

The current implementation direction now includes:

- `src/semop/unified_world_model.py`: shared projection and reasoning engine
- `src/semop/unified_responder.py`: integrated payloads attached to `ops`, `vision`, `video`, `visual_3d`, `math`, and `cp`
- `src/semop/grounding_self_evolution.py`: self-evolution now persists an integrated world snapshot and integrated reasoning summary
- `src/semop/adaptive_environment_learning.py`: environment brain, self-evolution, merged graphs, and temporal scene are fused into one integrated world and one integrated reasoning summary
- `src/semop/recursive_self_evolution.py`: program selection now considers unified world-model integration quality in addition to prior metrics
- `src/semop/continuous_learning.py`: exported training traces now include integrated world traces and integrated reasoning summaries

## Next experiments that matter

1. Make operator induction write directly into the shared world model instead of only post-hoc projection.
2. Score self-evolution candidates partly by contradiction reduction inside the integrated world graph.
3. Add cross-modal memory retrieval where a text prompt can pull prior visual and video world snapshots with matching blockers or affordances.
4. Add world-model delta learning so the system explicitly learns what changed between two runs, not only the latest merged state.
5. Use the integrated traces as the default distillation target for small local models that handle routing, question understanding, and answer narration.

## Practical conclusion

The literature does not support a pure text-only imitation path for the target system. The stronger pattern is consistent across papers: world modeling, evaluator-backed search, reflection memory, reusable skills, and structured self-improvement loops outperform one-shot generation when long-horizon reasoning or open-ended improvement matters.

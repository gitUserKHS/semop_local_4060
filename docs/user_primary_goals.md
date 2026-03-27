# User Primary Goals

Last updated: 2026-03-27

This document records the user's explicitly stated top-level goals so future agents can align implementation decisions with them.

## Core Goals

1. Prompt understanding and logical reasoning
   - When the user gives a prompt, the system should infer hidden context, hidden intent, unstated constraints, and relevant assumptions.
   - The system should respond with a logical, coherent, and context-aware answer.
   - The system should not stay at shallow keyword matching or template-style replies.

2. Multimodal situation understanding
   - The system should recognize and interpret visual images and video.
   - It should understand the situation shown, describe what is happening, and connect visual evidence to reasoning.
   - The multimodal path should be part of the same reasoning stack, not a disconnected side demo.

3. Long-horizon AGI direction
   - The long-term target is a generally capable intelligence system that can eventually perform human-level tasks such as autonomous driving, robot control, mathematical problem solving, and broad everyday reasoning.
   - These domains should converge through a shared world model, operator algebra, memory, and verification stack instead of becoming disconnected specialist silos.

4. Creativity and intuition
   - Creativity should be approached as structured concept fusion, reusable world-model abstraction, and operator recombination rather than unconstrained text generation alone.
   - Intuition should emerge from learned priors, analogical memory, and latent world-model planning that still remain auditable.

5. Productization and deployment
   - The project should move toward a commercializable stage with repeatable readiness measurement, grounded verification, safety gates, and beginner-friendly operation.
   - Research progress is valuable only when it strengthens deployable reasoning quality and trustworthy multimodal behavior.

## What This Means For Development

- Prefer unified prompt-first UX so a beginner can type one request and let the system route it.
- Improve hidden-premise recovery, contextual inference, grounding, and explanation quality before chasing raw model size.
- Keep operator algebra and compiler-style verification central to the reasoning process.
- Treat image and video understanding as first-class inputs to the same reasoning and context stack.
- Evaluate progress using both language reasoning and multimodal scene-understanding tasks.
- Add explicit readiness tracking for autonomous-driving-style embodied planning, robot-control-style action safety, mathematical proof ability, and concept-fusion creativity.
- Favor implementations that move the stack toward commercial readiness, not just research demos.

## Priority Signals For Future Agents

The following kinds of work are highly aligned with the user's stated goals:
- Better hidden context and intent inference from prompts.
- Better grounded explanation fidelity.
- Better multimodal reasoning over images and video.
- Better unified chat-style workflow for beginners.
- Better RTX 4060 8GB efficiency using symbolic-first and operator-algebra-first reasoning.
- Better embodied-world-model planning for driving and robotics readiness.
- Better concept fusion and creative hypothesis generation grounded in operator and world-model structure.
- Better commercialization readiness tracking and deployment-safe product loops.

The following kinds of work are lower priority unless they directly support the above goals:
- Narrow specialist GUI knobs that make the main workflow more fragmented.
- Larger raw generation behavior without better grounding or reasoning quality.
- Isolated demos that do not improve the main unified reasoning path.

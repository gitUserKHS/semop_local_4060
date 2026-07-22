# Project Agent Notes

## User-Declared Primary Goals
These goals should guide future changes across the whole repo.
If there is a tradeoff between a narrow demo feature and these goals, prefer these goals.

1. When the user gives a prompt, the system should infer hidden context, hidden intent, and relevant constraints correctly, then produce a logical answer.
2. The system should support multimodal understanding of visual images and video so it can recognize a situation, understand what is happening, and describe it clearly.
3. The ultimate long-term goal is a broadly capable AGI that can eventually handle human-level domains such as autonomous driving, robot control, mathematical problem solving, and general reasoning from one shared world-model stack.
4. Creativity and intuition should be treated as first-class engineering targets. Prefer implementations that combine world models, operator algebra, memory, and concept fusion instead of treating creativity as pure free-form text generation.
5. Productization matters. Prefer changes that move the system toward deployable safety, verification, multimodal grounding, and repeatable readiness tracking rather than isolated research demos.
6. Sample and hardware efficiency are first-class goals. The symbolic/operator core should remain useful with little labeled data and on an ordinary CPU-only PC; local models and RTX 4060-class GPUs are optional accelerators, not mandatory foundations.

## Practical Interpretation
- Keep the main product surface unified and beginner-friendly. Prefer one-box prompt workflows over separate specialist entry points when possible.
- Favor improvements that increase hidden-premise recovery, grounded reasoning, contextual inference, multimodal transfer, and verification quality.
- Preserve a CPU-first operator core that runs offline on ordinary PCs. Keep the RTX 4060 8GB strategy as an optional acceleration and parameter-efficient training tier, rather than shifting the system toward heavyweight generation-first behavior.
- Prefer few-shot operator induction, compositional transfer, synthetic curriculum data, external memory, and verifier feedback over collecting large labeled datasets by default.
- Report data count and hardware/resource budgets alongside quality metrics so improvements are not bought invisibly with more labels, memory, or model size.
- Existing module boundaries and APIs may be replaced when they obstruct the shared typed-operator kernel or low-resource target. Preserve behavior through benchmark baselines, compatibility adapters, and staged migration rather than preserving legacy structure for its own sake.
- When collecting data or adding benchmarks, include both text/context reasoning and multimodal scene-understanding cases.
- Treat autonomous driving, robotics, math, and creativity as future capability surfaces that should converge into the same world-model and operator-verification doctrine.
- When there is a choice, build reusable readiness trackers, planning scaffolds, and product-style safety gates rather than one-off vertical hacks.

## Source of Truth
For the expanded version of these goals, see `docs/user_primary_goals.md`, `docs/user_ultimate_agi_goal.md`, and `docs/low_resource_operator_intelligence.md`.

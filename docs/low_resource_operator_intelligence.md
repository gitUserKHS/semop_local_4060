# Low-Resource Operator Intelligence Doctrine

## North Star

SemOp should gain capability by discovering, composing, transferring, and verifying reusable operators, not by assuming that every improvement requires a larger model or a larger labeled dataset.

This is a cross-cutting engineering constraint for the operator-learning, world-model, memory, verifier, and product layers. It is a target and evaluation policy, not a claim that every current module already meets the budget.

## Execution Tiers

### Tier 0: Ordinary PC Core

- CPU-only and offline after installation
- reference target: 4 or more CPU cores and at most 16 GB RAM
- no mandatory model download, cloud API, CUDA, or training framework
- typed operators, symbolic reasoning, compact retrieval, proof/audit traces, and deterministic verifiers remain available

### Tier 1: Small Neural Assist

- optional CPU, iGPU, or small local-model assistance for parsing, ranking, and perception
- neural output is treated as a candidate or prior and must pass the same operator and verifier gates
- the system falls back to Tier 0 when the model or checkpoint is unavailable

### Tier 2: Consumer GPU Learning

- optional RTX 4060 8GB-class acceleration
- prefer LoRA/QLoRA, short context, small batches, cached features, and compact student models
- training artifacts must feed reusable operators, memories, or calibrated priors back into the Tier 0/1 runtime

## Data-Efficiency Protocol

When a feature learns from labeled examples, report results at `0`, `5`, `20`, and `100` labels whenever the task shape permits it. Each report should include:

- held-out in-domain quality
- cross-domain or compositional transfer quality
- unsupported-claim or verifier failure rate
- number of labels and synthetic examples used
- model/checkpoint size, peak memory, and wall-clock latency

Synthetic and pseudo-labeled data may build a curriculum, but the final gate must contain held-out human-reviewed or rule-verifiable cases. A higher score bought only by more labels is not evidence of a better operator abstraction.

## Runtime Order

The default decision order is:

1. Parse into explicit entities, types, relations, goals, and premises.
2. Retrieve compact structural memories and applicable basis operators.
3. Compose and execute deterministic operator programs.
4. Verify types, grounding, constraints, and counterexamples.
5. Use a neural model only for unresolved parsing, ranking, perception, or proposal work.
6. Re-run verification before accepting or retaining the result.

This order keeps the system useful without a model while allowing neural components to improve difficult cases.

## Definition Of Done

A new intelligence feature is not complete until it states:

- the lowest execution tier it supports
- behavior when optional models or GPUs are absent
- labeled and synthetic data counts
- CPU latency and peak-memory measurement method
- verifier and transfer results against a simpler baseline
- which reusable operator, memory, or world-model capability the feature adds

Features that cannot yet meet a budget must expose that limitation explicitly and must not silently turn optional heavyweight dependencies into core requirements.

## Refactor And Migration Policy

The current architecture is not immutable. String-only operator representations, duplicated registries, or domain-specific pipelines may be replaced when a shared typed kernel gives better transfer, verification, or resource use.

A structural replacement must still be staged:

1. Freeze current behavior and resource measurements as a baseline.
2. Introduce the new typed interface with adapters for one representative domain.
3. Run fast CPU-core tests plus held-out transfer and verifier gates.
4. Migrate additional domains only when the shared representation reduces duplication or improves measured quality.
5. Remove the legacy path after its callers and benchmarks have moved.

Backward compatibility is a migration tool, not a permanent architectural constraint.

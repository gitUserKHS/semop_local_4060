# VLSO Structural-First Rearchitecture

## Why The Previous Direction Was Insufficient

The previous VLSO stack leaned too much on:
- weak semantic labels
- few-shot concept matching
- pseudo-label refinement

That is useful as adaptation, but it is not the right primary intelligence path for a system that is supposed to generalize across arbitrary visual materials.

If the system must reason generally, then it should first recover:
- geometry
- topology
- part-whole structure
- bounded regions
- access paths
- control parts
- grasp parts

Only after those structures are recovered should semantic labels be used as optional priors.

## New Structural-First Rule

The VLSO visual path now follows this order:
1. raw image or detector payload -> regions, polygons, boxes, edges
2. geometry/topology features -> structural operator induction
3. structural operators -> world-model bindings
4. optional semantic labels, few-shot memory, pseudo labels
5. grounded QA over the world model

In short:
- structure first
- labels second
- QA last

## New Core Module

Primary new module:
- `src/semop/vlso/structural_operators.py`

It induces reusable bindings such as:
- `CONTAINER_BODY_OPERATOR`
- `ACCESS_PORT_OPERATOR`
- `ACCESS_CONTROL_OPERATOR`
- `ATTACHED_GRASP_OPERATOR`
- `CONTROLLED_ACCESS_OPERATOR`
- `MANIPULABLE_CONTAINER_OPERATOR`

These are derived from geometry/topology signatures, not from class labels alone.

## Integration Changes

### Object reasoning
`src/semop/vlso/object_reasoner.py`
- structural operator induction now runs before weak label interpretation
- inferred affordances and constraints can come directly from structure
- structural bindings are written into observation metadata

### Visual parsing
`src/semop/vlso/visual_parser.py`
- structural operator bindings are promoted into the shared world model
- bindings are preserved as `world.metadata['structural_operator_bindings']`

### Alignment
`src/semop/vlso/aligner.py`
- visual metadata now survives merge into the final world model
- structural bindings are no longer dropped during alignment

### QA
`src/semop/vlso/qa.py`
- opening/access questions now prefer structural access paths
- inventory questions can summarize structural containers and access parts
- weak label names are no longer the only basis for answers

## Design Consequence

Few-shot and pseudo-label learning are still useful, but their role is narrower:
- adapt
- refine
- disambiguate
- accelerate review

They are not the primary source of meaning anymore.

## Current Benefit

The system can now answer in a way that is closer to the intended philosophy, for example:
- `Structural access path: use side_handle to reach opening region opening_band on container.`

This is better than merely saying:
- `strap-like part`
- `zipper-like part`

because it explains the scene through reusable operator structure.

## Remaining Work

The structural-first path is now in place, but several upgrades are still needed:
- stronger raw-image region extraction
- better detector/segmenter grounding
- richer topological operators for articulated objects
- more operator families for tools, doors, drawers, bottles, and multi-part mechanisms
- benchmark sets that explicitly score structural operator recovery rather than only label overlap

## Engineering Rule Going Forward

Any new VLSO improvement should answer this question first:
- does it improve geometry/topology -> operator induction?

If not, it is probably not part of the long-term intelligence core.

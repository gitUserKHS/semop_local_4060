# Operator Intelligence Execution Steps

## Goal

Build an operator-centric intelligence system where visual reasoning and text reasoning converge through a shared structural world model.

The target is not a label-matching model. The target is:
- geometric/topological primitive extraction
- structural operator induction
- cross-modal operator alignment
- verifier-backed reasoning

## Step 1. Geometry Primitive Backbone

Purpose:
- extract reusable visual primitives before semantic labels

Concrete outputs:
- edges
- corner counts
- right-angle structure
- parallel-edge structure
- equal-length structure
- hole structure
- compactness and fill ratios

Current implementation:
- `src/semop/vlso/geometry_backbones.py`
- integrated into `src/semop/vlso/visual_parser.py`
- features forwarded into `src/semop/vlso/affordance_features.py`

Success criterion:
- structural operators can be induced even when semantic labels are weak or missing

## Step 2. Structural Operator Induction

Purpose:
- map primitives into reusable visual operators

Key operators:
- `CONTAINER_BODY_OPERATOR`
- `ACCESS_PORT_OPERATOR`
- `ACCESS_CONTROL_OPERATOR`
- `ATTACHED_GRASP_OPERATOR`
- `CONTROLLED_ACCESS_OPERATOR`
- `MANIPULABLE_CONTAINER_OPERATOR`

Current implementation:
- `src/semop/vlso/structural_operators.py`
- `src/semop/vlso/object_reasoner.py`

Success criterion:
- the system explains access/manipulation in operator terms, not just object labels

## Step 3A. Hidden Premise And Operator Algebra

Purpose:
- recover implicit goals and represent higher operators as compositions of simpler operators

Current implementation:
- `src/semop/premise_explorer.py`
- `src/semop/premise_eval.py`
- `src/semop/operator_algebra.py`

Success criterion:
- hidden goals and required premises become explicit graph structure and reusable operator decompositions

## Step 3B. Cross-Modal Operator Alignment

Purpose:
- let text and vision meet in the same operator space

Examples:
- text: `REQUIRES(open_access, insert)`
- vision: `ACCESS_PORT_OPERATOR`, `ACCESS_CONTROL_OPERATOR`
- aligned world model: `open first, then insert`

Current implementation:
- `src/semop/vlso/language_parser.py`
- `src/semop/vlso/aligner.py`
- `src/semop/vlso/qa.py`

Success criterion:
- visual reasoning can directly improve text reasoning and vice versa

## Step 4. Operator-Centric Memory

Purpose:
- store reusable structures instead of only labels

Memory layers:
- local exemplars
- compressed concept prototypes
- compressed operator prototypes
- episodic failures and repairs

Current implementation:
- `src/semop/vlso/concept_memory.py`
- `src/semop/vlso/operator_learning.py`
- `src/semop/vlso/hybrid_memory.py`
- `src/semop/cp_episode_store.py`

Success criterion:
- few examples expand operator reuse across object families and problem families

## Step 5. Verifier-Centered Intelligence

Purpose:
- prevent confident nonsense

Verifier targets:
- VLSO operator recovery
- grounded QA evidence
- CP compile/run/sample/random validation
- consistency checks across the shared world model

Current implementation:
- `src/semop/vlso/eval.py`
- `src/semop/cp_validation.py`
- `src/semop/cp_repair.py`

Success criterion:
- high-confidence outputs must be grounded in recovered operators or executable checks

## Step 6. Learned Parsers As Secondary Boosters

JEPA-style latent prediction belongs here as a structural booster rather than the primary source of meaning.

Purpose:
- use learned models to improve extraction quality without making them the sole source of truth

Examples:
- DINOv2/OpenCLIP for feature priors
- CP learned parser beside heuristic parser
- segmentation/detector backbones for better region grounding

Success criterion:
- learned modules improve operator recovery, not replace the structural pipeline

## Immediate Next Work

1. strengthen raw-image primitive extraction further with line, corner, and symmetry cues
2. connect actual segmentation and detector outputs more strongly into structural operators
3. expand operator-recovery eval from synthetic to real image sets
4. train the CP parser and compare it against heuristic+verifier mode

## Engineering Rule

Any new feature should improve at least one of these:
- primitive extraction quality
- operator induction quality
- cross-modal world-model quality
- verifier quality

If it does not, it is probably not part of the core intelligence path.

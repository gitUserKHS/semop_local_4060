# Hidden Premise Engine Next Steps

Completed in the current refactor:
- concept/script-compatible retrieval now reranks premise, script, and operator memory support
- clarification policy now uses a calibrated score instead of a single hard rule
- hidden premise benchmark now includes visual-language minimal pairs and CP counterfactual cases

Current next steps:
- learned script compatibility scoring is now blended into premise reranking
- VLSO approved review clusters can now seed SemOp premise/script/operator memories
- common evaluation now exposes operator-level premise support metrics

## Current state

The hidden-premise engine is now organized as:
- candidate retrieval
- premise proposal
- premise validation
- goal-preservation checking
- SQLite premise/operator memory reuse
- parser-first CP and VLSO alignment hooks

Current measured snapshot:
- test suite: `176` passing tests
- hidden premise benchmark: `20` cases
- hidden premise metrics are now tracked with satisfied/missing requirement state as well

## What is still weak

- retrieval now blends lexical overlap with concept-family and script/operator compatibility, but it is still largely symbolic and memory-driven
- unsupported-premise precision improved, but still needs harder real-world negative cases
- clarification decisions are still heuristic and under-specified
- CP hidden-constraint discovery is still lightweight compared with the parser itself
- VLSO hidden-premise transfer exists, but visual-side hidden-goal benchmarks are still small

## Next implementation steps

### Step 1. Premise-memory retrieval tuning
- add script-memory ranking beyond lexical overlap
- bias retrieved premise candidates by concept-conditioned script/operator compatibility
- compare premise recall with and without memory layers

### Step 2. Validation and clarification calibration
- add more requirement-state templates for satisfied vs missing prerequisites
- tighten unsupported-premise precision on small-input counterfactuals
- improve clarification behavior for optional-goal branches

### Step 3. Hidden-premise benchmark expansion
- extend the evaluation set with more visual-language minimum pairs
- add CP cases where the hidden premise is complexity, invariants, or implementation feasibility
- evaluate requirement_state_accuracy and clarification_score_mae next to goal preservation

### Step 4. CP structurer integration
- feed hidden-premise outputs into CP frame ranking before algorithm selection
- recover hidden requirements such as monotonicity, offline processing, or low-complexity obligations
- evaluate parser accuracy with and without premise augmentation

### Step 5. VLSO structural hidden-premise alignment
- add visual cases where opening, handle, control part, or occupancy changes the hidden goal
- score hidden-premise recovery and operator binding recovery together
- connect detector/segmentation structural hints directly into premise validation

## Recommended order

1. stronger premise retrieval
2. clarification/contradiction validation
3. benchmark expansion
4. CP parser integration
5. VLSO hidden-premise benchmark growth


## Updated next steps

1. replace the lightweight memory-learned scorer with a trained script-compatibility model
2. expand VLSO review-approved seeding from access/container cases into broader visual scripts
3. add operator-premise support metrics to VLSO grounded QA and full-stack comparison reports

- access-family hidden goals now explicitly include box/pouch/suitcase/bin in addition to drawer/cabinet/bottle/jar
- script compatibility scoring now blends lexical overlap with concept-family compatibility and harder sibling negatives

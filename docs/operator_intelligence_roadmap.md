# Operator Intelligence Roadmap

## Purpose

This roadmap reframes SemOp Local 4060 around four long-term axes:
- operator learning
- world model quality
- memory quality
- verifier quality

The goal is not feature sprawl. The goal is measurable generalization of reusable logical operators.

## 0-3 Months

### Operator Learning
- expand visual operator families beyond bag-like scenes into `box`, `drawer`, `door`, `bottle`, and `tool`
- expand CP DSL labels and learn a small parser that predicts frame plus algorithm family
- keep collecting logical connector examples for `has`, `requires`, `if`, `before`, `inside`, `contains`, and `reachable`

### World Model
- reduce heuristic noise in VLSO raw-image parsing
- improve object-part-opening separation
- tighten CP statement-to-DSL projection for ambiguous problem statements

### Memory
- enlarge VLSO prototype stores with diverse few-shot scenes
- enlarge CP episodic memory with accepted/editorial/failure cases
- keep local, compact stores first instead of large uncontrolled archives

### Verifier
- widen CP validator coverage
- build a small VLSO QA eval set with grounded expected relations
- expand ops evaluation with more forbidden-phrase and executability cases

## 3-6 Months

### Operator Learning
- train a learned CP parser that can run beside the heuristic parser
- add pseudo-label acceptance and cluster review for visual operators at scale
- connect operator learning outputs back into runtime ranking and pruning

### World Model
- make VLSO world models more object-centric and less polygon-fragment-centric
- unify relation typing across text graphs, CP DSL, and visual graphs
- standardize operator traces and reasoning traces

### Memory
- add stronger retrieval/reranking for CP episodes
- add stronger local/global memory fusion for VLSO scenes
- start measuring which memory layers improve accuracy and which only add noise

### Verifier
- add more counterexample synthesis in CP
- add approval-based cluster review flows for VLSO
- add regression eval suites for all major entry points

## 6-12 Months

### Operator Learning
- move from operator labels toward operator algebra
- learn compositional transitions such as `container access -> opening control -> reachable interior`
- learn cross-domain abstractions that survive beyond one modality

### World Model
- build richer multimodal world models with geometry, topology, and action affordances
- bring CP and visual reasoning closer by reusing shared notions like transition, invariant, containment, accessibility, and hierarchy

### Memory
- add stronger compressed prototype memories instead of only exemplar memories
- measure transfer from one object family to another and from one problem family to another

### Verifier
- require grounded evidence for high-confidence answers
- route low-confidence answers into review flows by default
- compare learned paths against hand-built heuristic baselines with fixed held-out sets

## Priority Order Right Now

1. VLSO operator-family expansion with small but diverse image sets
2. CP learned parser training and held-out evaluation
3. VLSO grounded QA eval set and CP parser eval set
4. stronger memory-quality measurement
5. architecture cleanup only when it improves one of the four axes

## RTX 4060 8GB Guidance

Recommended constraints:
- prefer small local models and LoRA over large end-to-end fine-tunes
- keep heavy lifting in explicit memory and verifiers
- use self-supervised embeddings, clustering, and prototypes for vision first
- use symbolic and verification-heavy paths where possible

This keeps the project aligned with its core thesis:

> a smaller reasoning core can become much stronger when it uses structured world models, explicit operator learning, large reusable memory, and hard verification loops.

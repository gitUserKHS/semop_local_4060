# Tiny Controller v1

## Contract

The controller is a search policy, not an answer generator. It ranks operator schemas
and typed arguments, estimates halt/value, and returns scores. The typed executor is
the only component allowed to derive facts.

CPU inference lives in `semop.tiny_controller` and requires NumPy only. PyTorch is
isolated in `semop.tiny_controller.training` and is needed only to train or export a
model.

## Default Architecture

| Component | Default |
| --- | ---: |
| Hidden dimension | 192 |
| Shared relation message blocks | 2 |
| Internal recurrent passes | 4 |
| Confident greedy margin | 0.1 logit |
| Name-free token buckets | 24,576 |
| Relation buckets | 2,048 |
| Operator buckets | 2,048 |
| Parameter count | 5,837,578 |
| Float32 uncompressed size | about 22.3 MiB |
| Hard cap | 15,000,000 parameters / 64 MiB artifact |

Symbols are anonymized by type and relational role. The model receives type,
predicate, fact status, verifier-goal/frontier role, operator schema, argument type,
and proof-state features. Each operator embedding pools its domain-neutral family, semantic tags, typed
precondition/effect signatures, cost, and schema name. Raw point, object, and variable
names are not model features. Training-only `hard_negative` and `synthetic` tags are
explicitly removed before encoding. The current NumPy artifact format version is `5`;
v2 zero-head and v3 unscaled-head artifacts are migrated while loading, and v4 weight
arrays remain loadable under the bounded v5 scoring contract.

Each recurrent pass performs relation-aware message aggregation over term nodes.
The final state vector feeds:

- an operator schema score
- a typed argument pointer score
- an action/state compatibility score
- an eight-value domain-neutral action/goal structure score
- a halt head
- a state-value head

Action CE and NumPy inference use the same joint score. Latent action compatibility
and the argument pointer are bounded with `tanh` before the learned structural score
is added. This prevents an out-of-domain hash/message logit from growing without
bound and drowning out the separately learned typed goal-overlap head; no fixed
feature value is declared correct.

When the top action exceeds the runner-up by the configured score margin, the policy
requests a one-action beam for that round. Smaller gaps keep the kernel's default
beam of four. This is a confidence-based compute decision, not a success decision;
the selected action still needs typed execution and proof replay, and monotonic search
can continue after a wrong ranking.

## Inference

```python
from semop.kernel import OperatorKernel
from semop.tiny_controller import NumpyTinyController

policy = NumpyTinyController.load("artifacts/tiny_controller.npz")
result = OperatorKernel(problem.registry).solve(
    problem.state,
    problem.goals,
    policy=policy,
)
```

If scoring raises an exception or guided search does not solve within its reserved
budget, the kernel continues with deterministic search. A controller's halt score is
never accepted as proof.

## Training Objective

`controller_loss` uses the fixed v1 weights:

```text
action CE
+ argument CE
+ 0.5 * halt BCE
+ 0.5 * value MSE
+ 0.2 * recursive consistency MSE
```

Training examples are built from replay-verified traces. A later action in the same
verified proof is treated as an alternative valid ordering, never as a negative.
`build_decision_training_cases` calls `OperatorKernel.policy_goals`, so training and
NumPy inference see the same bounded grounded frontier relations. A frontier is only
an action-ranking hint; it is never a proof fact or halt condition.
For each positive action, `build_decision_training_cases` retains at most four
explicitly certified `hard_negative` actions. The built-in synthetic curriculum adds
four wrong ground bindings with the same goal predicate and argument types per trace.
The label tags are hidden, so the controller must use effect/goal structure rather
than provenance to rank them.

The `TraceCorpus` enforces:

- reviewed splits of `0/5/20/100` examples per domain
- at most 5,000 synthetic traces per domain
- synthetic operator depth at most 6
- at most four hard negatives per positive
- rejection of failed or non-replayed traces

A frontier LLM may propose a typed program, but its output cannot enter this corpus
directly. `verify_judged_program` must execute and replay it first, and
`teacher_review_to_solve_result` independently repeats that verification before
conversion. See `frontier_llm_judge.md`.

Run an end-to-end verifier-generated training smoke test with a deliberately small
debug architecture:

```powershell
python -m pip install -r requirements-train.txt
python tools/train/train_tiny_controller.py `
  --output artifacts/tiny_debug.npz `
  --examples-per-domain 1 `
  --epochs 1 `
  --debug-small
```

Remove `--debug-small` to train the default 5.84M architecture. The command writes
the NumPy artifact, a JSONL verified-trace corpus, and a JSON training summary. The
default `language-math-vision` curriculum contributes an equal number of hidden-
premise language, exact arithmetic, and verified spatial-relation traces. Use
`--curriculum operator-v1` for the compatibility geometry/hidden-premise/grid set.
Use `--curriculum language-math-vision-composed` to add replayable 4/5-step
vision-count -> exact-comparison -> conjunctive-language traces as a fourth domain.
These curricula are synthetic and record `reviewed_examples: 0`; none may be
reported as a 5/20/100-shot human-reviewed run. The summary records the curriculum,
trained domain list, and verified trace count per domain.

```powershell
python tools/train/train_tiny_controller.py `
  --output artifacts/tiny_composed_debug.npz `
  --curriculum language-math-vision-composed `
  --examples-per-domain 1 --epochs 1 --debug-small
```

Use repeated `--domain` flags for a real holdout artifact:

```powershell
python tools/train/train_tiny_controller.py `
  --output artifacts/without_vision.npz `
  --domain language --domain math `
  --examples-per-domain 20 --epochs 5
```

`tools/eval/run_lodo_controller_experiment.py` automates all three holdouts and checks
the adjacent metadata before reporting a domain as unseen.

## Macro Library

`MdlMacroLibrary.induce` examines verified sub-programs of length 2 through 6. It
anonymizes ground terms into typed slots and retains a macro only when all conditions
hold:

1. support from at least three distinct traces
2. valid typed bindings and a continuous proof-state chain
3. at least 10% description-length reduction
4. a caller-supplied held-out no-regression validator passes

Macros remain abbreviations over primitive verified programs. They do not bypass the
executor or replay verifier.

## Current Status

The architecture, NumPy artifact format, PyTorch mirror, loss, trace extraction, and
fallback path are implemented. A leakage-controlled full-model smoke trained on two
synthetic traces per holdout and transferred goal-binding selection to all three
unseen domains. Language and math passed the 30% held-out median reduction gate;
vision's dedicated distractor fell from 13 to 1 while its median remained 1 because
most vision positives already require zero or one action. No production artifact is
committed, and human-reviewed 20/100-shot transfer remains unevaluated.

A 29,834-parameter composed-only diagnostic trained on 20 synthetic traces for five
epochs reduced `composed-v4` expansion from 47 to 40 with soundness 100% and zero
false positives. The non-learned goal-directed policy reaches 34, so this is a
training/export compatibility result, not evidence that the neural controller has
surpassed symbolic guidance.

## Research Lineage

- [DreamCoder](https://arxiv.org/abs/2006.08381) motivates typed program induction,
  wake/sleep data generation, and library compression; v1 keeps only bounded verified
  trace synthesis and MDL macro retention.
- [Tiny Recursive Model](https://arxiv.org/abs/2510.04871) and
  [HRM](https://arxiv.org/abs/2506.21734) motivate repeated computation with shared
  small networks. They are puzzle-focused preprints, so recursion count, consistency,
  and halt behavior remain ablation targets rather than assumed truths.
- [Generalist Neural Algorithmic Learner](https://arxiv.org/abs/2209.11142) motivates
  sharing a graph processor across algorithms and domains.
- [Faithful Compositional Networks](https://arxiv.org/abs/2005.00724) supports keeping
  explanation fidelity separate from neural intermediate activations; SemOp uses
  executable proof replay instead.
- V-JEPA-style latent prediction remains a later perception/planning direction. Its
  large video pretraining regime is intentionally outside this ordinary-PC v1.

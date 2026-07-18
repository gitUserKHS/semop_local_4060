# Low-Resource Transfer Evaluation

## Run It

```powershell
python tools/eval/evaluate_low_resource_transfer.py --run-fast-tests
```

Direct language/math/vision adapter suite:

```powershell
python tools/eval/evaluate_low_resource_transfer.py `
  --suite language-math-vision `
  --run-fast-tests
```

Cross-domain vision -> math -> language suite:

```powershell
python tools/eval/evaluate_low_resource_transfer.py --suite composed-v4
```

Use a NumPy policy artifact:

```powershell
python tools/eval/evaluate_low_resource_transfer.py `
  --model artifacts/tiny_controller.npz `
  --shot-model 20=artifacts/tiny_20.npz `
  --shot-model 100=artifacts/tiny_100.npz `
  --output artifacts/eval/low_resource_transfer.json
```

The JSON report contains per-case measurements, domain aggregates, unguided/guided
A/B results, shot-curve availability, leave-one-domain-out zero-shot slices, model
size, artifact size, RSS, and gate decisions.

## Split Policy

Cases are separated by structural change rather than random names or numbers:

- geometry: renamed points, new relation chains, and operator composition
- hidden premise: new wording identifiers, `REQUIRES/BLOCKED_BY/GOAL` composition,
  and missing-premise negative controls
- grid: new path shapes, obstacle layouts, larger grids, and no-path controls

The optional `language-math-vision` suite uses the same measurement code:

- language: graph 입력과 실제 한국어·영어 wording, multiple requirements, blockers,
  Horn inheritance, contradiction, and missing-premise controls
- math: exact numeric representation, unseen operator nesting, one-variable equations,
  deeper AST, and wrong-answer controls
- vision: verified spatial composition, inverse canonicalization, unverified confidence,
  unknown-relation controls, raw-pixel composition, shape/count/area goals, and
  centroid-only uncertainty

All three domains also include typed argument distractors whose predicate schemas
remain goal-relevant. Static predicate pruning cannot remove them; a policy must pick
the correct ground binding.

Every domain includes an expected-unsolved control. A reported success on one of
these cases counts as a false positive and breaks expected-outcome accuracy. The
default synthetic controls now preserve the original target and remove one proof
dependency before the verifier confirms that the near miss is unsolved.

## Gates

The report evaluates these promotion conditions:

- primitive replay integrity is exactly 100%
- guided replay-verified goal completion is no more than 1 percentage point below unguided search
- median expansions fall at least 30% in at least two of three domains
- 20-shot verified solve rate reaches at least 90% of 100-shot rate
- model at most 15M parameters and artifact at most 64 MiB
- controller adds at most 512 MiB peak RSS
- small-case p95 at most 2 seconds and large-case p95 at most 10 seconds

Missing 20-shot or 100-shot artifacts produce `null`, not a fabricated pass. In that
case `overall_status` is `not_evaluated`.
An artifact is accepted for a shot curve only when its adjacent `.summary.json`
declares the same `reviewed_examples_per_domain`; a filename or CLI label alone is
not treated as evidence.

The JSON report keeps `proof_soundness` and `verified_solve_rate` as compatibility
aliases. They mean `primitive_replay_integrity` and
`replay_verified_goal_completion`; neither is a human semantic-accuracy score.
`semantic_correctness` remains `null` until human-reviewed gold tasks are supplied.

Likewise, a domain slice is labeled `verified_domain_held_out` only when the adjacent
training summary exists and its `trained_domains` omits that domain. An all-domain
artifact is reported as `domain_seen_during_training`, not as leave-one-out transfer.

## Current Baseline

Both symbolic suites currently solve every expected-solvable case and reject every
negative control with replay soundness 100%. Agenda forward chaining and grounded
operator frontiers reduced the 36-case language/math/vision suite from 156 executed
actions to 81. It proves all 25 positive cases and rejects all 11 negative controls.
Language median expansion falls 33.3%, math falls 37.5%, and the dedicated vision
chain falls 92.3%, so the balanced
symbolic two-domain expansion gate passes. The 20/100-shot gate is still unavailable,
so `overall_status` remains `not_evaluated` and `shadow` remains the default.

The separate eight-case `composed-v4` suite executes real vision -> exact comparison
-> language classification proofs. It verifies five positive programs in 3 to 5
steps, rejects a false threshold, an explicit negation, and a false conjunct, and
reduces total expansion from 47 to 34 with zero false positives.

The full 5.84M LODO smoke used 20 synthetic traces per available training domain and
five epochs. It preserved 100% reported soundness and solve rate while passing held-out
median reduction in language and math. The fresh Horn distractor improved only from
15 to 13 expansions, so this is structural typed-action transfer rather than evidence
of broad language understanding. See `docs/lodo_controller_experiment.md`; synthetic
counts are not reviewed 20/100-shot evidence.

The 2026-07-17 v4 typed snapshot completed 134 fast tests in 13.532 test seconds and
501 repository tests plus 20 subtests in 283.27 seconds. On that machine the
goal-directed baseline
measured small-case p95 at about 0.006 seconds and large-case p95 at about 0.016 seconds. These numbers are a local
regression reference, not a cross-machine performance claim.

## Fast Core Suite

```powershell
python -m unittest discover -s tests -p "test_typed_operator_*.py" -v
```

This suite covers the typed kernel, symbolic baselines, direct language/math/vision
adapters, migration runtime, controller runtime/training split, trace and macro
constraints, and the benchmark schema. It is
the 30-second CPU gate; the broader legacy suite remains a separate regression gate.

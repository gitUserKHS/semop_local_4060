# Language/Math/Vision LODO Controller Experiment

## Purpose

This experiment asks whether one small action-ranking controller can reuse an
operator-selection principle in a domain whose traces were completely withheld.
For each run, two domains provide synthetic replay-verified traces and the third is
evaluated only after NumPy export.

```text
train(language, math) -> test vision
train(language, vision) -> test math
train(math, vision) -> test language
```

The evaluator accepts a slice as `verified_domain_held_out` only when the artifact's
adjacent summary declares `trained_domains` and omits the evaluated domain.

## Leakage Controls

- `hard_negative` and `synthetic` provenance tags are not controller features.
- Training distractors use the same goal predicate and argument types with the wrong
  ground binding.
- The policy receives eight domain-neutral structural values: exact goal effect,
  predicate match, type match, argument overlap, novel-effect ratio, binding count,
  precondition count, and effect count.
- Verifier goals and grounded operator frontiers have distinct relation roles, and
  verifier-generated decision cases use the same `policy_goals` contract as inference.
- Structural values have a learned linear head; no fixed score declares an action
  correct.
- Every selected action is still executed and replayed by the typed kernel.

## Run

Fast 29K diagnostic run:

```powershell
python tools/eval/run_lodo_controller_experiment.py `
  --output-dir artifacts/lodo_debug `
  --examples-per-domain 20 `
  --epochs 5
```

Default 5.84M controller:

```powershell
python tools/eval/run_lodo_controller_experiment.py `
  --output-dir artifacts/lodo_full `
  --examples-per-domain 20 `
  --epochs 5 `
  --full-model
```

Use repeated `--held-out language|math|vision` to run selected slices. The output
directory contains three `.npz` artifacts, verified trace corpora, adjacent training
summaries, and `lodo_report.json`. Each run now evaluates both the held-out slice of
`language-math-vision` and the complete `composed-v4` suite automatically.

## 2026-07-17 Full-Model Smoke Result

This local smoke used the 5,837,578-parameter model, 20 synthetic traces from each
available training domain, five epochs, and seed 0. Each holdout trained on 40 traces.
These are replay-verified generated traces, not human-reviewed 20-shot examples. It is
a pipeline and structural-transfer result, not a data-scale claim.

| Held out | Trained on | Held-out median expansion | Dedicated binding case | Soundness |
| --- | --- | ---: | ---: | ---: |
| language | math, vision | `3 -> 2` (`33.3%`) | old binding `14 -> 2`; fresh Horn `15 -> 13` | `100%` |
| math | language, vision | `6.5 -> 4` (`38.5%`) | `15 -> 3` (`80.0%`) | `100%` |
| vision | language, math | `1 -> 1` | `13 -> 1` (`92.3%`) | `100%` |

All held-out solve rates were 100%. Compressed artifacts were about 21.64 MB and the
largest held-out p95 CPU time was about 0.083 seconds on the development PC. The LODO
transfer gate passed because two domains reduced median expansion by at least 30%,
but the fresh Horn composition moved only modestly. The result supports bounded typed
action ranking; it does not show broad semantic transfer.

## 2026-07-17 Frontier Diagnostic And v4 Replay

After adding cross-domain operator frontiers, the 29,834-parameter diagnostic model
was retrained for all three holdouts with the same 20 synthetic traces per available
domain and five epochs. This validates the updated training -> NumPy export -> replay
pipeline; it is not the 5.84M promotion run.

| Held out | Held-out median expansion | Solve rate | Soundness |
| --- | ---: | ---: | ---: |
| language | `3 -> 2` (`33.3%`) | `100%` | `100%` |
| math | `8 -> 5` (`37.5%`) | `100%` | `100%` |
| vision | `1 -> 1` | `100%` | `100%` |

Each of the three artifacts was also replayed on the eight-case `composed-v4` suite.
They kept 100% reported soundness, zero false positives, and reduced total expansion
from 47 to 40. That suite contains five positive and three negative controlled cases,
including two-condition count conjunctions and measurement reuse. It confirms limited
pipeline compatibility rather than broad cross-domain generalization. The full 5.84M
frontier-aware LODO run has not yet been repeated; the preceding full-model table is
the earlier structural snapshot.

## What This Does Not Prove

The benchmark deliberately isolates typed goal-binding selection. It does not prove
unrestricted free-form language understanding, novel mathematical theorem discovery, semantic
recognition in natural photographs, or robust transfer under real distribution
shift. `promotion_ready` remains false because verified human-reviewed 20/100-shot
artifacts have not been evaluated. `shadow` therefore remains the default runtime.

# Repair gate audit — 7 October 2026

The invariant geometry is working. The remaining problem includes correspondence
precision, candidate selection, and confidence, not just the update budget.
No experiment in this bundle passes the contact gate.

## Evidence from the supplied bundle

All values below use the bundle's rows, not estimates from plots. Validation
aggregates 144 observations (48 patterns × 3 seeds); training has 418 patterns.
"Geometric" means whole and every part Chamfer <= 0.01. "Accepted" additionally
requires solver status `ok`. The existing metric intentionally reports geometry
separately from confidence; this audit preserves that definition.

| Geometry / view supervision | Fixed geometric | Fixed accepted | Training geometric | Validation geometric | Rotation / samples / samples+poses |
|---|---:|---:|---:|---:|---:|
| existing / existing | 0/16 | 0/16 | 0/418 | 0/144 | 0% / 0% / 0% |
| existing / contrastive | 0/16 | 0/16 | 9/418 (2.2%) | 5/144 (3.5%) | 0% / 0% / 0% |
| revised / existing | 11/16 | 6/16 | 54/418 (12.9%) | 26/144 (18.1%) | 68.8% / 66.7% / 75.0% |
| revised / contrastive | 9/16 | 4/16 | 80/418 (19.1%) | 36/144 (25.0%) | 56.2% / 58.3% / 62.5% |

The geometry change is essential in these four experiments. Contrastive
supervision improves training and validation geometry, but reduces fixed-set
geometry and robustness relative to revised/existing. This is one training seed;
it is not yet evidence of a stable causal tradeoff across seeds.

The 87.5% three-piece result is **7/8 on the fixed overfit subset**. It does not
generalize: three-piece training is **21/168 (12.5%)**, and validation is **1/54
(1.9%)** across observations. Two-piece training is 59/250 and validation 35/90.

## The extra blocker: confidence

For revised/contrastive, 12/16 fixed solves are low-confidence. Only four are
geometrically successful with `ok` status. All 80 geometric training successes
and all 36 geometric validation successes are low-confidence; neither evaluation
has an `ok` solve. This is material even though the training/robustness gate checks
currently use geometric success rates.

The fixed gate requires **16/16 geometric successes and zero low-confidence
solves**. Each robustness condition requires at least **44/48** geometric
successes. Training requires at least **335/418** geometric successes. More
overfit updates alone cannot establish these other requirements.

Confidence is `matchability * exp(-contact_rms / huber_delta)`, with delta=0.02
and a minimum confidence of 0.1. Even perfect matchability cannot pass when
contact RMS exceeds `0.02 * log(10) = 0.04605`. Geometrically successful training
examples have RMS around 0.07–0.16 because scoring still includes inconsistent
selected correspondences. Robust fitting can find a good pose while the full
support used for confidence remains poor. Improve that support; do not lower
the acceptance threshold or silently redefine confidence using favorable inliers.

Mean top-match distance on true-contact pairs is 0.0896 on training and 0.0860
on validation. These are descriptive means across pairs, including mistakes,
not Chamfer errors or point-weighted means. They support investigating precision.
Dustbin means separate matchable/unmatched rows: training 0.1505/0.8638,
validation 0.1723/0.8582, overfit 0.0672/0.9256.

## What the seven fixed geometric failures show

| Pattern prefix / suffix | Max part Chamfer | Candidate already geometrically valid? | Interpretation |
|---|---:|---|---|
| 6f4749e8 / 005 | 0.010217 | Yes | selection/refinement can lose a valid candidate |
| b8767b71 / 004 | 0.011847 | Yes | selection/refinement can lose a valid candidate |
| 970027a2 / 006 | 0.012852 | No | candidate precision/coverage |
| d3eea69d / 006 | 0.051623, ~180° | Yes | wrong candidate wins |
| 77a2242b / 000 | 0.212185, ~180° | No | candidate generation/matching |
| 6f4749e8 / 004 | 0.096390, ~179° | No | candidate generation/matching |
| ac131ac1 / 002 | 0.250991, ~180° | No | candidate generation/matching |

Three of the seven failures already have a valid **unrefined** candidate.
Candidate coverage is 11/16 fixed, 154/418 training, and 60/144 validation.
Coverage is diagnostic, not an upper bound after refinement: refinement can
also rescue initially invalid candidates. Three of the four flips have `smooth`
cut-family metadata; only one is `planar_control`. Planar symmetry is a plausible
mechanism, not a demonstrated explanation for all four. Distances and cosines
in the encoder are also invariant to reflection; investigate whether directional
surface context is missing before prescribing a chirality feature.

Chamfer uses normalized Euclidean distances, not millimeters. Physical margins
need the object's scale. Whole-object Chamfer can look good despite a misplaced
part, so keep the per-part criterion.

## CPU experiments actually run

Replayed 36 saved tensors at samples_poses/4101 with four solver variants: **144
solves**. Baseline reproduced geometry, status and confidence (within 1e-6) for
every saved observation. Canonical coordinates were used only for metrics;
filtering and solving used local XYZ and predicted probabilities.

| Variant | Overfit subset geometric / accepted | Training subset geometric / accepted | Validation subset geometric / accepted |
|---|---:|---:|---:|
| baseline | 7/12 / 3/12 | 1/12 / 0/12 | 3/12 / 0/12 |
| no refinement | 8/12 / 3/12 | 0/12 / 0/12 | 3/12 / 0/12 |
| Huber delta 0.005 | 8/12 / 0/12 | 1/12 / 0/12 | 3/12 / 0/12 |
| mutual maxima only | 6/12 / 4/12 | 0/12 / 0/12 | 4/12 / 0/12 |

These are selected tensor subsets, not random samples or the full gate. No
solver change merits promotion from this evidence. Decreasing Huber delta also
makes confidence exponentially stricter, so it is not a clean refinement-only
ablation in the current implementation.

Reproduce after extracting the supplied archive into this directory:

```bash
python scripts/analyze_reassembly_repair.py \
  --results analysis/repair_gate_audit \
  --output analysis/repair_gate_audit/audit.json
```

## Next GPU experiments

Four fresh-run configurations are in `configs/repair_gate_experiments`. All use
20k overfit / 30k contact updates. Compare each intervention with `long_control`
at matched budgets and at intermediate 10k/20k history points:

1. **long_control**: determine how much comes from more training alone.
2. **localization_x2**: double only localization KL weight (1 → 2), to test
   whether tighter learned correspondences improve both geometry and RMS.
3. **view_weight_002**: lower only contrastive weight (0.1 → 0.02), to test the
   observed generalization/overfit tradeoff without discarding resampled views.
4. **existing_view**: longer-budget revised/existing control, since it solved
   more fixed cases and was more robust at 10k.

The supplied runs actually used **10k overfit**, while the current local default
already says **20k**. Late revised/contrastive overfit recall goes 66.7% → 71.3%
from 8k to 10k; geometry fluctuates 50% → 43.8% → 37.5% → 56.2% → 56.2%.
Longer training is justified as an experiment, not a guarantee of convergence.
At the observed throughput, 50k total updates are roughly 20 hours per
configuration plus preflight/evaluation; this is an extrapolation, not measured
runtime for the new configurations.

Run a single variant without repeating the two failed legacy architectures:

```bash
ROOT="$HOME/reassembly_v2"
python scripts/run_reassembly_repair_experiment.py \
  --config configs/repair_gate_experiments/long_control.yaml \
  --managed-root "$ROOT" \
  --manifest "$ROOT/bottles498/prepared/manifest.json" \
  --output "$ROOT/repair_gate_experiments/long_control" \
  --field-diagnostic-report "$ROOT/field_diagnostics/repair_v3/summary.json" \
  --device cuda:0
```

Use the actual VM paths if different. Change both config and output for each
experiment. The runner uses the existing preflight, query-cache, training,
contact-gate and validation code and exits 2 on gate failure. It does not train
scaffolds or touch final test data. Preflight may adjust batch/accumulation while
preserving effective batch. `--resume` is for an unchanged configuration;
changing a completed comparison budget through the original comparisons phase
does not silently extend its runs. Explicit budget-only train resume exists,
but requires new matching preflight evidence and a fresh gate evaluation.

Choose the next implementation from these results. If localization improves
recall but RMS remains high, test a supervised penalty on distant/wrong-interface
real-match probability mass, alongside the existing dustbin objective. If valid
candidates keep losing, test an XYZ-derived surface-normal opposition or overlap
score separately; validate normal orientation on concave fragments. Increasing
candidate count alone cannot fix incorrect scoring. A learned scaffold cannot
rescue this contact gate because scaffold training is deliberately downstream.

No checkpoints or prepared manifest/assets are in this checkout or archive, and
local PyTorch is CPU-only. The four GPU training experiments are prepared but
**not run**. Original pipeline code, confidence thresholds and gate criteria
were preserved. The archive does not contain the pre-fix run, so the reported
before/after causal improvement cannot be independently verified here.
`log(K)` balances the dustbin against K zero-logit candidates in the illustrative
case; actual logits are normalized cosine/learned-temperature plus fracture
gating, so this is not a universal exact calibration.

## Verification

All four configs passed the repair config validator; the focused runner's CLI
loaded successfully. Baseline replay agreed on all 36 saved observations.
The model/evaluation/experiment unit suites ran 37 tests: 36 passed and one test
failed in two supervision subcases. The existing legacy-equivalence test expects
the repair matcher's dustbin to equal the legacy value 1.0, but the current repair
model deliberately initializes it to log(24)=3.178 in that fixture. This mismatch
predates this audit; no production model code was changed. That test's control
contract needs an explicit update to reflect the dustbin intervention.

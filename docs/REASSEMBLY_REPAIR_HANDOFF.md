# Fragment assembly repair: experiment handoff

This document transfers the context for the diagnosis-driven repair of the
fragment-assembly pipeline. Use it with the repository checkout and the compact
VM result bundles. It records measured evidence, not assumptions about which
repair will work.

## Objective

The system receives two or three complete XYZ point-cloud fragments of one
broken object. It must return rigid transforms that assemble the original input
points in the reference fragment's coordinate frame:

`x_aligned = x @ R.T + t`

The research hypothesis is that a coarse whole-object prior can improve a
contact-based assembly after contact candidates have been generated. It is not
assumed that the prior can replace fracture matching.

The prepared ShapeNet-bottle dataset is fixed for these experiments:

- 73 accepted source meshes and 546 complementary-fracture patterns.
- 418 train patterns from 56 sources; 48 validation patterns from 6 sources;
  69 test patterns from 11 sources; 11 held-out cut-family patterns.
- Dataset fingerprint:
  `b7f2b84a20964276c894300a7ece4d8f97f232903bace0f14131535594462043`.
- Inputs use 1,024 points per fragment. New runs start from fresh
  initialization; legacy checkpoints are diagnostic artifacts only.

## Why the repair experiment exists

The original v2 pipeline was reproducible, but its learned correspondence
stage did not solve assembly:

| Measured result | Meaning |
|---|---|
| Pilot contact-only and predicted-scaffold models: 0/418 training and 0/48 validation assemblies | Failure already existed on training geometry. |
| Fixed overfit inputs: 16/16 successful | The code path can fit one fixed observation set. |
| Overfit with new rotations, samples, or both: 0/48 each | The overfit check did not establish pose or sampling robustness. |
| Oracle correspondences on the originally selected points: 12/12 successful | Point selection and rigid candidate solving contain usable geometry. |
| Oracle segmentation with learned matches: 0/12 successful | Segmentation alone is not the main repair. |

The diagnostic measurements supported correspondence discrimination as the
first bottleneck: descriptor effective rank was about 1.85/128, transformed
view similarity was about 0.997, and true-contact top-one recall was 1.33%,
near uniform ranking (about 1.41%). Lowering confidence thresholds admitted
wrong poses instead of recovering correct assemblies.

The original scaffold was also unsuitable for refinement near the surface. Its
overall SDF error was 0.0254 versus 0.0533 for a zero field, but its
near-surface error was 0.0346 versus 0.0178 for zero. The old 32-cubed grid had
about 0.145 normalized spacing, larger than the 0.1 truncation distance.

## Repair architecture being tested

The repair implementation is `fragment-assembly-repair-v3` under
`reassembly/repair`.

1. **Stage 1 — contact learning.** It retains the 1,024-point input,
   256-to-128-to-64 hierarchy, 128 channels, and 256 matching points. The
   revised variant replaces raw rotated XYZ in the contact path with local
   distance/angle encodings and adds two four-head, 128-channel contextual
   attention blocks. Coordinates remain available for neighborhoods, field
   queries, and pose estimation, but cross-fragment attention never treats
   unaligned coordinates as physical proximity.
2. **View supervision.** The revised supervision uses independently sampled,
   independently posed views, multiple geometric positives, separated
   negatives, and an ambiguity exclusion. It supervises the exact contextual
   embedding consumed by correspondence logits.
3. **Stage 2 — scaffold.** Only after Stage 1 passes its gate, it freezes the
   contact model and trains a reference-frame continuous field with a scaffold
   adapter. Each object gets 2,048 queries: 512 exterior surface zero-distance,
   512 near-interior, 512 near-exterior, and 512 surrounding points.
4. **Stage 3 — numerical assembly.** Weighted Kabsch, non-collinearity checks,
   candidate limits, spanning-tree enumeration, and explicit failure statuses
   are retained. The continuous field scores and refines already generated
   contact candidates using transformed predicted exterior points. It does not
   create candidates and it does not use a visualization grid for refinement.

## Completed phase: focused field diagnostic

Command already run on the VM:

```bash
bash scripts/run_reassembly_repair.sh field
```

The supplied `field_diagnostic_bundle.tar.gz` reports a complete run: 144/144
jobs, with 137 executed jobs and 7 conditional skips. It performed no optimizer
updates. It used the old predicted-scaffold checkpoint, 12 deterministic
validation patterns from 6 sources, 1,024-point inputs, FP32 field queries, and
FP64 numerical refinement. It is a measurement of the old field and grid
representation, not v3 training.

### Field results

| Measurement | Result | Interpretation |
|---|---:|---|
| GT-grid exterior-surface error, 32 / 64 / 128 cubed | 0.01791 / 0.004766 / 0.001224 | The old 32-cubed grid was too coarse for surface refinement. |
| Conditional GT-grid error, 256 cubed | 0.0003345 | Grid convergence is achieved only at much higher resolution. |
| Predicted continuous exterior error | 0.033171 | Learned field error remains after removing grid interpolation. |
| Predicted grid versus its continuous field, 32 / 64 / 128 cubed | 0.010456 / 0.003416 / 0.001069 | Gridding adds error, but does not explain the whole learned-field error. |
| Continuous predicted field-only refinement from exact poses | Chamfer 0.000378 to 0.040153; 12/12 to 0/12 successes | The old predicted field actively harms exact assemblies when used alone. |
| GT continuous field with oracle exterior points, 5 degree perturbed starts | 12/12 successful | Accurate field information can guide local refinement under a diagnostic oracle control. |
| Same GT diagnostic at 15 degrees | 3/12 successful | A field is a local guide, not a substitute for contact candidate generation. |
| Predicted conditioning versus GT conditioning, fixed uncertainty | mean probability change 1.47e-8; no directional argmax changes | The old matcher was effectively insensitive to the geometric content of scaffold conditioning. |

The field diagnostic also found that predicted-field uncertainty was regionally
miscalibrated: near-surface MAE 0.03462 with 57.8% of absolute errors within
one predicted sigma; surrounding-space MAE 0.01625 with 76.82% within sigma.

**Conclusion:** use a continuous field after contact candidate alignment, train
surface-relevant queries, and retain grids only for inspection. The completed
field diagnostic does not demonstrate that a learned v3 scaffold helps yet.

## Phase commands and dependencies

Run from the repository checkout on the VM. Set these once per shell, adapting
the checkout path only if necessary:

```bash
export REASSEMBLY_ROOT=/home/kpandey/reassembly_v2
export REPAIR_DEVICE=cuda:0
export REPAIR_WORK="$REASSEMBLY_ROOT/repair_v3"
export FIELD_DIAGNOSTICS="$REASSEMBLY_ROOT/field_diagnostics/repair_v3"
```

Run phases in this order:

```bash
# Already complete. Do not rerun merely to start the next phase.
bash scripts/run_reassembly_repair.sh field

# Current decision phase: four seed-42 contact-learning comparisons.
bash scripts/run_reassembly_repair.sh comparisons

# Only if comparisons writes selection.json with status "selected".
bash scripts/run_reassembly_repair.sh replicate

# Only if the selected contact configuration passes for seeds 42, 43, and 44.
bash scripts/run_reassembly_repair.sh scaffold

# Only if scaffold validation writes acceptance.json with passed: true.
bash scripts/run_reassembly_repair.sh final-test
```

`replicate`, `scaffold`, and `final-test` must not run in parallel with
`comparisons`: each consumes a decision artifact created by the previous phase.
The workflow stops at a failed gate rather than changing confidence thresholds
or treating a completed training job as success.

### What each remaining phase measures

| Phase | Work | Required output / next decision |
|---|---|---|
| `comparisons` | Four matched, fresh seed-42 contact experiments: existing geometry × existing views; existing × resampled contrastive; revised geometry × existing views; revised × resampled contrastive. Each uses a 16-pattern overfit run and 10,000 contact updates. | `selection.json` ranks only variants that pass the contact gate, then uses source-macro assembly success, assembly success, matching top-one recall, and a deterministic name tiebreak. If status is `contact_gate_failed`, stop and analyze evidence. |
| `replicate` | Repeats the selected variant and unchanged `existing__existing` control at seeds 43 and 44. | `replication.json` and per-seed contact gate reports. All selected-model seeds must pass before field training. |
| `scaffold` | Trains one Stage-2 field per selected-contact seed, then evaluates identical contact candidates under contact-only, predicted-field, GT-field, and perturbed-field conditions. | `acceptance.json`. It assesses field contribution without giving the field credit for missing candidates. |
| `final-test` | Locks the validation-selected configuration and evaluates test and held-out cut-family splits. | `final_test.json`; it is blocked unless `acceptance.json` says `passed: true`. |

## Gates and acceptance criteria

The Stage-1 contact gate requires all of the following before scaffold training:

- 16/16 fixed original-input overfit fits, with no low-confidence failures.
- At least 90% success separately for rotation, resampling, and combined
  rotation-plus-resampling controls across their three recorded seeds.
- At least 80% assembly success on training geometry with fresh deterministic
  observations.

Final validation keeps the original 0.01 normalized per-part and whole-assembly
Chamfer criteria. It requires at least 39/48 successes for **each** of three
evaluation pose/sample seeds. Predicted-field guidance must either improve
success or reduce Chamfer without reducing success, and that direction must
repeat in at least two of the three fresh training seeds. The validation split
has only six source objects, so report source-macro statistics and uncertainty;
do not describe 48 fracture patterns as 48 independent objects.

## Runtime, resource, and artifact handling

The active VM has a V100 32 GB and about 545 GB free root storage. The workflow
keeps a 20 GiB reserved-GPU cap, 40 GiB managed-artifact cap, and 50 GiB free
space reserve. The completed field run took about 35.25 recorded job-hours,
mostly exact 256-cubed ground-truth mesh-grid construction (about 30.25 hours),
not model training. Comparisons, replication, and scaffold phases are training
experiments and should be run in a persistent `tmux` or `screen` session.

Every phase writes a compact review archive:

```bash
bash scripts/run_reassembly_repair.sh bundle
```

Return this file to the reviewing chat after the current phase:

```text
$REPAIR_WORK/repair_results.tar.gz
```

The VM logs are under `$REASSEMBLY_ROOT/repair_logs`. The compact bundle carries
the structured evidence: resolved configuration, provenance, training curves,
gate reports, per-example evaluation records, matching/segmentation/candidate
statistics, field metrics, and bounded qualitative artifacts. Checkpoint
weights are deliberately not included.

## Current handoff status

- **Field diagnostic:** complete and analyzed; its result bundle is already
  available to this project context.
- **Comparisons:** this is the current phase. Its results must be inspected
  before running `replicate`. If it has finished, provide
  `$REPAIR_WORK/repair_results.tar.gz` (and `selection.json` if a quick status
  check is needed) to the new chat.
- **Replication, scaffold, and final test:** not authorized by the experiment
  evidence until their preceding gates pass.

## Suggested first message in the new chat

> Read `docs/REASSEMBLY_REPAIR_HANDOFF.md` in this repository. The focused
> field diagnostic is complete and I am providing the latest
> `repair_results.tar.gz` from the VM. Analyze the completed phase
> evidence-first, inspect `selection.json` and all contact gate reports, and
> tell me whether the next gated phase may run. Do not change architecture,
> losses, thresholds, or checkpoints until the evidence is reviewed.


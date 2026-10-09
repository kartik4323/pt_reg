# Contact orientation implementation and GPU commands

The latest executed notebook shows 15/16 fixed geometric successes but only
7/16 confidence-accepted successes, 198/418 training successes, and 51/144
validation successes. Validation candidate coverage is 89/144. At tolerance
0.03 the training/validation counts are 275/418 and 71/144: relaxing geometry
does not remove the orientation or confidence problems.

This implementation preserves the contact-only gate. No learned shape prior
is used before that gate; optional contact orientation is a soft constraint
derived only from complete fragment XYZ. Normal sign estimated from the
fragment centroid is a proxy, particularly on concave fragments. Unreliable
estimates are ignored rather than used to reject assemblies.

## What changed

- Local 32-neighbor PCA estimates normals. Reliability requires smallest/middle
  eigenvalue ratio < 0.2, middle/largest > 0.1 and absolute radial cosine > 0.2.
- The solver scores whether mating normals face opposite directions, weighted
  by original correspondence probabilities. The score coefficient is 0.0004.
- Eight legacy and eight normal-constrained proposals share the existing
  16-proposal cap. Normal offsets of 0.02 preserve correspondence mass. Four
  distinct candidates are retained; insufficient normal support uses the
  original candidate generator.
- Refinement is reverted if it worsens the combined score. Confidence still
  uses the original contact RMS and matchability formula. Geometry tolerance
  0.01, confidence 0.1, minimum pair mass 0.05 and correspondence weight 0.001
  are unchanged.
- Existing configurations remain disabled by default. New configuration files
  enable orientation explicitly. Solver overrides do not alter checkpoint
  weights and are labeled diagnostic. They cannot certify scaffold-training
  lineage or be used as official acceptance evidence.
- The notebook now distinguishes candidate loss, rotation-flip indicators,
  geometric success and accepted success, and aligns initial prediction/GT
  cameras. A rotation over 170 degrees on a failed reconstruction is an
  indicator, not proof of a symmetry failure.

## Local validation actually completed

Replayed all 36 saved observations in the older archive, with baseline and
orientation enabled (72 solves). All baseline outcomes/status/confidence agree
with the previous replay. No geometric regressions in these selected subsets:

| Saved subset | Baseline geometry | Orientation geometry | Accepted baseline → orientation |
|---|---:|---:|---:|
| Overfit | 7/12 | 8/12 | 3/12 → 3/12 |
| Training | 1/12 | 2/12 | 0/12 → 0/12 |
| Validation | 3/12 | 5/12 | 0/12 → 0/12 |

See `orientation_replay_summary.json` for counts by fragment count. These are
selected samples from the older 10k run, not the latest 30k run or gate passage.
No confidence improvement is claimed. The test suite covers rigid and
permutation invariance, curved and degenerate surfaces, opposing normals,
mass preservation, fallback, cache identity, refinement rollback, padding,
three-fragment composition, diagnostic overrides/resume, and training/evaluation
contracts. Final verification ran 77 tests: 76 passed and one CUDA-only check
was skipped on this CPU machine. Notebook code was executed against 722 archived
observations; matching initial cameras and bounds were verified programmatically
and the static assembly preview was inspected.

## First: evaluate orientation on your existing checkpoints

No retraining is needed for this step. On the GPU machine, from the repository:

```bash
git pull origin main
ROOT="$HOME/reassembly_v2"
SOURCE="$ROOT/repair_gate_experiments/long_control_retry"
python scripts/run_reassembly_repair_ablation.py \
  --experiment "$SOURCE" \
  --managed-root "$ROOT" \
  --manifest "$ROOT/bottles498/prepared/manifest.json" \
  --output "$ROOT/repair_gate_experiments/long_control_orientation_eval" \
  --solver-config configs/repair_gate_experiments/orientation_solver.yaml \
  --device cuda:0
```

This evaluates fixed-16, all robustness modes, full training and validation,
always completing validation even if the gate fails. Exit 2 means the gate
checks failed; results are still saved. Use a fresh output directory. After an
interruption add `--resume` with identical inputs and unchanged implementation.
The source run and checkpoints are never overwritten.

For a baseline re-evaluation with the current implementation, use the same
command with a different output directory and omit `--solver-config`. This
checks disabled behavior against the original saved evidence.

Open `repair_experiment_analysis.ipynb`, set:

```python
RUN_DIR = Path.home() / "reassembly_v2/repair_gate_experiments/long_control_orientation_eval"
BASELINE_DIR = Path.home() / "reassembly_v2/repair_gate_experiments/long_control_retry"
```

Run all cells to compare metrics and inspect saved assemblies. The ablation
has no new training history because it does not train. The source experiment
retains the learning curves. Compare the same modes/seeds and inspect confidence
alongside geometry, especially three-fragment validation and candidate loss.

## Then: matched-budget precision experiments

The main experiment runner creates two fresh models at 20k overfit / 30k contact
updates. Run localization-only first, with unchanged orientation behavior:

```bash
python scripts/run_reassembly_repair_experiment.py \
  --config configs/repair_gate_experiments/localization_x2.yaml \
  --managed-root "$ROOT" \
  --manifest "$ROOT/bottles498/prepared/manifest.json" \
  --output "$ROOT/repair_gate_experiments/localization_x2" \
  --field-diagnostic-report "$ROOT/field_diagnostics/repair_v3/summary.json" \
  --device cuda:0
```

The default loss weight is 1; this configuration doubles only localization
weight. Dustbin, segmentation, view-loss weight and training budgets are unchanged.

Evaluate the combination on those same localization checkpoints using
`run_reassembly_repair_ablation.py` with `--experiment` pointing to
`localization_x2`, orientation_solver.yaml, and a fresh output such as
`localization_x2_orientation_eval`. This isolates solver effects without another
training run.

For a fully configured fresh experiment whose unchanged gate can certify its
own training lineage, use `orientation_only.yaml` or
`orientation_localization_x2.yaml` with the main experiment runner and a matching
fresh output. Do not promote a configuration merely because selected examples
improve: require all original gate checks, then the existing replication/scaffold
acceptance stages. Do not resume the old training runs after code changes.

## Export and replay without a GPU

Export the completed source experiment so the latest correspondences can be
replayed locally. This includes reports, histories, per-observation rows, bounded
sample assemblies and tensors; it excludes checkpoints and prepared source assets.

```bash
python scripts/export_reassembly_repair_diagnostics.py \
  --experiment "$SOURCE" \
  --output "$ROOT/long_control_retry_diagnostics.tar.gz"
```

Use a new filename if it exists. The archive includes a SHA256 manifest and
refuses export if source files change during packaging. After extraction into
a fresh directory:

```bash
python scripts/replay_reassembly_repair_orientation.py \
  --experiment /path/to/extracted/diagnostics \
  --solver-config configs/repair_gate_experiments/orientation_solver.yaml \
  --output /path/to/new/orientation_replay.json
```

Replay uses local XYZ and predicted probabilities only for solving, canonical
coordinates only for metrics. It covers the selected samples_poses/4101 tensors,
not all evaluation observations. The full latest-checkpoint GPU evaluation and
fresh localization training have **not** been run on this CPU-only workstation.

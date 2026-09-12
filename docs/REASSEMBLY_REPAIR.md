# Repair workflow: contacts first, spatial shape guidance second

This implements the diagnosis-driven experiment plan. It does not claim that the new model has met the learning targets: those require the VM runs below. All new learned runs start fresh; existing v2 checkpoints are read only by the focused diagnostic. The existing `python -m reassembly` workflow remains available for exact replay.

## VM setup and inputs

Run from the repository checkout using the existing Python environment with the CUDA build of PyTorch. The reference environment uses Python3.10, torch2.11.0+cu126, a V10032GB, and the versions in `requirements.reassembly-repair.txt`. If that environment is already installed, no reinstall is necessary.

```bash
cd /home/kpandey/satellite/pt_reg  # adjust to your checkout
git pull --ff-only
conda activate ptreg-v2           # use your existing environment name
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"

export REASSEMBLY_ROOT=/home/kpandey/reassembly_v2
export REPAIR_DEVICE=cuda:0
export REPAIR_WORK="$REASSEMBLY_ROOT/repair_v3"
export FIELD_DIAGNOSTICS="$REASSEMBLY_ROOT/field_diagnostics/repair_v3"
# Optional: explicitly select the original downloaded bundle or VM diagnostic directory.
# export DIAGNOSTIC_BUNDLE=/absolute/path/diagnostic_bundle.tar.gz
```

The default prepared manifest is `$REASSEMBLY_ROOT/bottles498/prepared/manifest.json`; override `REPAIR_MANIFEST` if its location differs. The configured dataset fingerprint is `b7f2b84a20964276c894300a7ece4d8f97f232903bace0f14131535594462043`. A different local dataset is rejected. Do not regenerate the73 accepted sources or546 patterns for these comparisons.

The scripts check resolved paths and storage before execution. Keep outputs on the root filesystem with adequate space. Managed storage stays below40GiB with at least50GiB free; training must use less than20GiB reserved GPU memory. No command has a wall-clock deadline by default.

## Run the phases

Run each phase separately. Use a persistent terminal/tmux session for long VM work. Rerunning the same command resumes explicitly selected checkpoints and completed workflow steps; it never searches historical checkpoints to initialize a fresh experiment.

```bash
# 1. Focused continuous-field/grid/refinement diagnostic; no optimizer updates.
bash scripts/run_reassembly_repair.sh field

# 2. Four matched geometry×supervision experiments, seed42.
bash scripts/run_reassembly_repair.sh comparisons

# 3. Repeat the selected configuration and unchanged control at seeds43 and44.
bash scripts/run_reassembly_repair.sh replicate

# 4. Train fields on the gated contact models; paired validation on identical candidates.
bash scripts/run_reassembly_repair.sh scaffold

# 5. Only after validation acceptance, run the locked test/cut-family evaluations.
bash scripts/run_reassembly_repair.sh final-test

# Rebuild a compact review bundle at any time, including after a failed gate.
bash scripts/run_reassembly_repair.sh bundle
```

An exit2 at a learning gate means its measurements did not meet the requirements. Inspect `selection.json`, `gate/contact_gate.json`, or `acceptance.json`. Do not continue by changing success or confidence thresholds. Execution errors instead carry a traceback/error or `failure.json`.

The comparisons use10,000 updates each for contact learning and a separate fresh16-pattern overfit run per configuration. The fixed set receives fresh samples and poses during training. All four configurations use identical sample streams, point resolution, optimizer settings, effective batch8, and absolute curriculum boundaries at600/1300 updates. The first unchanged architecture/objective condition is the longer-training control. Preflight falls back from batch2/accumulation4 to batch1/accumulation8 only after measuring memory at full geometry resolution.

Validation evaluates all48 patterns. Contact checkpoints are selected lexicographically by assembly success, correspondence recall, then matching objective. Field checkpoints use near-surface error, then overall SDF error. Only best/latest weights are retained.

## What changed

- Revised geometry uses distances and angles within each fragment, retaining coordinates separately. Two self/cross-attention blocks compare contextual features without treating different fragment frames as aligned.
- Independently sampled/posed view supervision uses multiple geometric positives, distant negatives, and an ignored ambiguous neighborhood. It reaches the actual contextual embeddings consumed by the matcher.
- Existing contact localization, unmatched states, segmentation balancing, point budgets, Kabsch, candidate enumeration, confidence thresholds, and rigid refinement constraints remain.
- Stage1 trains contacts and geometry. Stage2 freezes both and trains a field adapter plus field. Stage3 is candidate assembly and numerical refinement, with no mandatory matcher-only optimizer phase.
- Field queries contain equal surface-zero, near-inside, near-outside, and surrounding groups. Missing signed groups receive a separate source-hash-bound query cache; original prepared assets are unchanged.
- The prior scores candidate-transformed predicted exterior points. Refinement evaluates the continuous field and query gradients; coarse grids are visualization/diagnostic artifacts.

Before field training, the contact gate requires16/16 original-input fits, at least90% success separately on rotation/resampling/combined controls across three seeds, and at least80% assembly success across training geometry with fresh deterministic observations. Overfit and main experiment checkpoints have distinct purposes and cannot initialize one another.

Final validation requires39/48 successes for each of the three evaluation seeds, using the unchanged0.01 per-part and whole-assembly Chamfer criteria. Predicted-field benefit must repeat in at least two of three independently trained models. Reports use source-level grouping/bootstrap; six validation sources limit statistical conclusions. Test and cut-holdout share source identities previously inspected during diagnosis.

## Focused diagnostic details

```bash
python -m diagnostics.reassembly_field \
  --managed-root "$REASSEMBLY_ROOT" \
  --output "$FIELD_DIAGNOSTICS" \
  --device cuda:0
```

The followup uses the original12 validation patterns and old VM weights. It measures continuous predicted/GT fields, fresh32/64/128 grids with both truncation orders, conditional256 convergence, exact and perturbed poses, field-only and oracle-contact-anchored refinement, uncertainty held fixed in matcher-content interventions, and regional calibration. It saves per-example jobs incrementally and verifies hashes before/after. Completed diagnostic execution is a prerequisite for training, not a statement that a scaffold is useful.

## Outputs to return

- `$FIELD_DIAGNOSTICS/field_diagnostic_bundle.tar.gz` (see the diagnostic completion message for the exact filename).
- `$REPAIR_WORK/repair_results.tar.gz` after each experiment phase or failed learning gate.
- Logs under `$REASSEMBLY_ROOT/repair_logs` remain on the VM; the bundles include structured evidence rather than checkpoint weights.

Each training run saves `run.json`, resolved configuration, optimizer/scaler/random state, explicit lineage, data/config/code hashes, `updates.jsonl`, `history.jsonl`, `numerics.jsonl` when needed, and `training_report.json`. Evaluations save `examples.jsonl`, probability/segmentation/candidate measurements, source/difficulty summaries, field metrics, bounded original/target/assembly arrays, and learning plots. Training completion never counts as assembly acceptance.

## XYZ inference

After a selected field run, use its explicit stage2 checkpoint. Inputs are2–3 NumPy arrays `[N,3]` in consistent units; no labels, normals, meshes, or target object are accepted.

```bash
python -m reassembly.repair infer \
  --managed-root "$REASSEMBLY_ROOT" \
  --output "$REASSEMBLY_ROOT/inference/object01" \
  --checkpoint /absolute/path/to/scaffold/best.pt \
  --fragment /absolute/path/fragment_a.npy \
  --fragment /absolute/path/fragment_b.npy \
  --device cuda:0
```

`result.json` records status, confidence, reference index, and matrices. `assembly.npz` contains rigidly transformed original-resolution points and the matrices. The convention is `x_aligned = x @ R.T + t`; preprocessing is composed out and the reference transform is identity. Failure returns no substituted identity assembly. Use `--condition contact_only` for explicit contact-only inference from a stage1 or stage2 checkpoint.

Add `--save-scaffold` to predicted-field inference to save a coarse field NPZ and SDF/uncertainty slice visualization. Its coordinate-to-output mapping is recorded in `result.json`; the grid does not participate in refinement.

## Local verification

```bash
python -m unittest discover -s tests -p 'test_reassembly*.py' -v
python -m reassembly.repair --help
```

Synthetic correctness/backward tests do not certify GPU memory, rotation/sample generalization on the VM dataset, or the80% learning target. Those results remain measured outputs of the commands above.

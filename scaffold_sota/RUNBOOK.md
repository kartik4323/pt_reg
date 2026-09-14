# Bottle pilot: VM runbook

Read [README.md](README.md) for implementation scope and [PLAN.md](PLAN.md) for the scientific design. Commands run from the repository on the Linux VM. Replace the manifest/checkpoint placeholders with actual immutable artifacts. Native CUDA compatibility and model learning remain to be measured; this repository does not contain trained SOTA weights or a completed v3 export.

## 1. Own a separate runtime directory

The orchestration interpreter needs Python 3.8+ and PyYAML. Data/prior operations also need Torch, NumPy, SciPy and trimesh. Model training runs in separately created native environments. Existing pinned checkouts under `sota_repro/models/<model>/upstream` must be present and clean.

```bash
cd ~/Kartik_23CS30026/pt_reg
export STUDY_ROOT="$HOME/Kartik_23CS30026/scaffold_sota"
export MANIFEST="/absolute/path/to/bottle/manifest.json"
export PYTHONDONTWRITEBYTECODE=1
python -m scaffold_sota init --run-root "$STUDY_ROOT"
python -m scaffold_sota check-data --run-root "$STUDY_ROOT" --manifest "$MANIFEST"
```

The run root must be outside the repository and `/data`. The runner enforces a 75-GiB study cap, a 50-GiB free-space floor, PyTorch reserved memory below 20 GiB and 2 GiB free GPU headroom. Other GPU compute jobs cause a stop; they are never terminated. Environments, caches, temporary files, logs and native source copies all belong to this root. Its marker and lock prevent accidental ownership of another experiment.

## 2. Install the pilot recipients independently

```bash
# Inspect the planned commands before executing installation.
python -m scaffold_sota setup --run-root "$STUDY_ROOT" --model jigsaw
python -m scaffold_sota setup --run-root "$STUDY_ROOT" --model jigsaw --execute
python -m scaffold_sota setup --run-root "$STUDY_ROOT" --model ccs --execute
python -m scaffold_sota setup --run-root "$STUDY_ROOT" --model garf --execute
```

Setup copies pinned upstream code, installs into `envs/<model>`, and logs each dependency step under `environments/<model>`. A successful installation is labelled `installed_unverified`; it does not certify legacy CUDA extension execution. Failures point to their step logs. `setup --source PATH` accepts another clean source checkout at the pinned revision. Matrix jobs use the default pinned checkout locations.

## 3. Start native baselines before v3 arrives

```bash
python -m scaffold_sota matrix --run-root "$STUDY_ROOT" --manifest "$MANIFEST" \
  --config scaffold_sota/configs/pilot.json --models jigsaw ccs garf --seeds 42 \
  --output "$STUDY_ROOT/matrices/pilot_pending.json"
python -m scaffold_sota run-matrix --matrix "$STUDY_ROOT/matrices/pilot_pending.json"
```

The matrix contains GARF feature pretraining, each native baseline and B0–B4. Native/B0/B1 can run now; v3-dependent B2–B4 stay `awaiting_v3`. Execution is sequential, using each model's environment. The matrix seals code, config, input manifest, native revisions and prior artifacts. Do not modify these during a comparison.

Training starts with a real largest-piece forward/backward and prediction-contract check. Nonfinite gradients or malformed poses stop the run. An untrained matcher can return a declared no-solve result and still learn. Standalone `preflight` requires valid returned poses, and neither preflight is an accuracy gate. GARF's generated config enables scaled FP16 and records native stage AdamW hyperparameters. Other optimizer/schedule deviations are explicit matrix metadata.

The initial 10,000 updates are a pilot budget, not established convergence. Batch size is one object with eight gradient-accumulation steps; this does not reproduce native batch-normalization statistics of batch size eight. Assess fixed-fit, new rotations/resampling and validation predictions before testing the hypothesis. Change budgets using training/validation evidence, start new run identities and match changes across conditions.

An optional fixed-fit diagnostic for Jigsaw uses its own run ID:

```bash
"$STUDY_ROOT/envs/jigsaw/bin/python" -m scaffold_sota train \
  --run-root "$STUDY_ROOT" --manifest "$MANIFEST" --model jigsaw \
  --config scaffold_sota/configs/fixed_fit.json --run-id jigsaw_fixed_fit
```

This fits the first 16 fixed training observations. Evaluate the training split with `--limit 16 --observation-seed 42`, then seed 4101 for new observations. The checkpoint/result records subset and fixed training; it is not the full-data baseline.

B0 receives matched extra native training. B1 adds an equal-size null branch, B2 pooled global scaffold, B3 spatial geometry with constant uncertainty and B4 spatial geometry with predicted uncertainty. Each seed's arms branch from that seed's selected native `best.pt`. Branch gates start at zero; B1–B4 parameter counts match. Native frozen extractors and prior weights remain frozen.

## 4. Import v3 and the historical v2 control

```bash
python -m scaffold_sota export-prior --run-root "$STUDY_ROOT" \
  --checkpoint /absolute/path/to/completed/v3.pt --training-manifest "$MANIFEST" \
  --output "$STUDY_ROOT/priors/v3.pt"
python -m scaffold_sota export-prior --run-root "$STUDY_ROOT" \
  --checkpoint /absolute/path/to/historical/v2.pt --training-manifest "$MANIFEST" \
  --output "$STUDY_ROOT/priors/v2.pt"
python -m scaffold_sota matrix --run-root "$STUDY_ROOT" --manifest "$MANIFEST" \
  --config scaffold_sota/configs/pilot.json --models jigsaw ccs garf --seeds 42 \
  --prior-v3 "$STUDY_ROOT/priors/v3.pt" --prior-v2 "$STUDY_ROOT/priors/v2.pt" \
  --output "$STUDY_ROOT/matrices/pilot_v3.json"
python -m scaffold_sota run-matrix --matrix "$STUDY_ROOT/matrices/pilot_v3.json" --resume
```

The supplied training manifest must be the one actually used for the prior; exports verify its fingerprint and record training-source identities. Export strips the original matcher/optimizer and never runs Stage 3. v2 is never silently substituted for v3. Existing completed native jobs are reusable only with matching lineage. Static token caches are evaluation-only: training generates fresh deterministic observations at every optimizer step and queries the frozen prior on demand, by default on CPU.

Resume an interrupted job using `run-matrix --resume` or the identical original `train` command with `--resume`. Resume checks code, config, data, prior, seed, feature checkpoint and native parent. `status --run-root ...` lists study runs. `unlock` removes only a same-host lock whose owner process has exited.

## 5. Seal predictions and compare

Evaluation uses the checkpoint's data settings. Keep split, point budget, observation seed and full observation identities matched across conditions. Native prediction receives observed points, mask, anchor and the declared scaffold only. GT labels and source mesh paths are not passed into native prediction.

```bash
"$STUDY_ROOT/envs/jigsaw/bin/python" -m scaffold_sota evaluate \
  --run-root "$STUDY_ROOT" --manifest "$MANIFEST" \
  --checkpoint "$STUDY_ROOT/runs/jigsaw_assembly_native_seed42/best.pt" \
  --split val --observation-seed 4101 --output "$STUDY_ROOT/evaluations/jigsaw_native_val4101.jsonl"
"$STUDY_ROOT/envs/jigsaw/bin/python" -m scaffold_sota evaluate \
  --run-root "$STUDY_ROOT" --manifest "$MANIFEST" \
  --checkpoint "$STUDY_ROOT/runs/jigsaw_assembly_B4_seed42/best.pt" \
  --prior "$STUDY_ROOT/priors/v3.pt" --split val --observation-seed 4101 \
  --output "$STUDY_ROOT/evaluations/jigsaw_B4_val4101.jsonl"
python -m scaffold_sota frozen-compare --run-root "$STUDY_ROOT" --manifest "$MANIFEST" \
  --predictions "$STUDY_ROOT/evaluations/jigsaw_native_val4101.jsonl" \
  --prior-v3 "$STUDY_ROOT/priors/v3.pt" --prior-v2 "$STUDY_ROOT/priors/v2.pt" \
  --output "$STUDY_ROOT/evaluations/jigsaw_frozen_val4101" --mode refine
```

Frozen comparison replays identical sealed starts. A0 preserves native output, A1 applies the common refinement without a field, A2/A3 use v3 constant/predicted uncertainty, A4 uses v2, A5 is a labelled GT-field diagnostic, A6 another training source and A7 a source-balanced training generic. Predicted exterior masks, not GT fracture labels, select field-fitting points. Missing priors/estimators and failed interventions are explicitly skipped and stay in every condition's count; a skipped intervention is not evidence of no benefit.

For generative ranking, export `evaluate --candidates 8`, then call `frozen-compare --mode rank` on the same file. The first draw stays the native operational output; GT candidate recall is diagnostic only. Duplicate and failed draws are counted; fewer than two distinct valid poses makes ranking ineligible. DiffAssemble's native default sampler can return duplicates. Ranking and refinement are separate comparisons.

Evaluate B0/B1/B2/B3 with the same command pattern, then compare complete files:

```bash
python -m scaffold_sota compare --run-root "$STUDY_ROOT" \
  --baseline "$STUDY_ROOT/evaluations/jigsaw_B0_val4101.jsonl" \
  --treatment "$STUDY_ROOT/evaluations/jigsaw_B4_val4101.jsonl" \
  --output "$STUDY_ROOT/evaluations/jigsaw_B4_vs_B0.json"
```

Success uses the mean symmetric **unsquared** Chamfer threshold 0.01 in normalized units. Every failed/missing solve stays in the denominator. Reports include source-macro success/accuracy, valid-solve distance and pose errors, failure/harm and a paired source-object bootstrap. Keep ordinary test and held-out-cut tables separate. Confirm selected contrasts with training seeds `42 43 44` and observation seeds `4101 4102 4103`; independent recipient seeds each need their own baseline/prerequisites. A one-seed pilot or six validation objects cannot establish broad generality. Broader categories/fragment counts need a separate validated reader/protocol; this reader accepts complete two/three-piece bottles only.

## 6. Content controls and prior probes

`evaluate --prior-control NAME` changes input content while recipient weights stay fixed. Names: `constant_uncertainty`, `shuffled_uncertainty`, `null`, `global`, `wrong`, `generic`, `distance_noise`, `distance_bias`, `shuffled_distance`, `sign_flip`. `ground_truth` additionally requires `--diagnostic-only`. Reports identify the substitution and donor/artifact. Wrong/generic controls use training donors only. Always select fresh output paths.

```bash
python -m scaffold_sota cache-prior --run-root "$STUDY_ROOT" --manifest "$MANIFEST" \
  --prior "$STUDY_ROOT/priors/v3.pt" --split val --observation-seed 4101 \
  --output "$STUDY_ROOT/caches/v3_val4101"
python -m scaffold_sota prior-diagnostics --run-root "$STUDY_ROOT" --manifest "$MANIFEST" \
  --prior "$STUDY_ROOT/priors/v3.pt" --kind field --diagnostic-only \
  --output "$STUDY_ROOT/diagnostics/v3_field"
python -m scaffold_sota prior-diagnostics --run-root "$STUDY_ROOT" --manifest "$MANIFEST" \
  --prior "$STUDY_ROOT/priors/v3.pt" --kind pose --diagnostic-only \
  --output "$STUDY_ROOT/diagnostics/v3_pose"
```

Caches are streamed, compressed and bound to exact input hashes; they contain no GT meshes or poses. Pass a cache directory as evaluation `--prior` with the same point budget and observation seed. Wrong/generic substitutions require a live frozen artifact to query training donors. All content controls preserve the same 512 fixed multiscale Sobol query locations. Pose probes start at exact GT or 5-degree/0.02 and 15-degree/0.05 perturbations with anchor fixed. They test whether a prior moves a correct assembly away; diagnostic outcomes never select operational poses.

## 7. Local checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest scaffold_sota/tests -q
python -m scaffold_sota --help
```

Tests check frozen priors, content controls, cache binding, coordinate algebra, leakage, failure denominators, source-paired statistics, native adapter boundaries, exact resume and ownership. Tiny native spies exist only in tests; production adapters instantiate pinned native models. CPU tests do not certify CUDA feasibility or improved assembly accuracy.

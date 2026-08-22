# PartNet / GPAT GPU runbook

Run this protocol only on the Linux/CUDA VM.  This repository intentionally
does not download or store PartNet, its metadata, or generated sample clouds.
The raw archive remains under `$DATA_ROOT` and run bundles exclude it.

## 1. Set up isolated environments

```bash
export OUR_ROOT=/path/to/fragment_compact/pt_reg
export GPAT_ROOT=/path/to/gpat                 # untouched official clone
export DATA_ROOT=/mnt/partnet_data
export RUN_ROOT=/mnt/partnet_runs

cd "$OUR_ROOT"
conda env create -f env.yml -n fragment-assembly
conda activate fragment-assembly
pip install -r requirements.txt
# Recommended for robust FPFH + ICP gauge registration:
pip install open3d

git clone https://github.com/real-stanford/gpat "$GPAT_ROOT"
cd "$GPAT_ROOT"
conda env create -f environment.yml -n gpat
conda activate gpat
pip install -e .
cd utils/chamfer && python setup.py install
cd ../pointops && python setup.py install
```

Do not modify the official GPAT clone.  The `gpat_reference` suite records its
Git revision and complete command output separately from this project.

## 2. Obtain and prepare PartNet on the VM

Request PartNet v0 access with the ShapeNet account required by the PartNet
portal, then place the downloaded archive and metadata only here:

```bash
mkdir -p "$DATA_ROOT/partnet_raw" "$DATA_ROOT/partnet_meta"
ln -s "$DATA_ROOT/partnet_raw" "$GPAT_ROOT/dataset/partnet_raw"
ln -s "$DATA_ROOT/partnet_meta" "$GPAT_ROOT/dataset/partnet_dataset"
cd "$GPAT_ROOT"
conda activate gpat
python dataset/preprocess.py
```

The official preprocessor creates `$GPAT_ROOT/dataset/partnet/{train,val,test}`.
Its exact parts, target, poses, segmentation and equivalence labels are the
interchange source for both methods.  GPAT's `--ratio=1` non-exact protocol is
kept for its own reference run only; this pipeline consumes exact parts until
the diagnosis is complete.

Prepare a portable *processed* pilot without copying raw data into the repo:

```bash
cd "$OUR_ROOT"
conda activate fragment-assembly
python scripts/prepare_partnet_gpat.py \
  --raw-root "$DATA_ROOT/partnet_raw" \
  --metadata-root "$DATA_ROOT/partnet_meta" \
  --gpat-root "$GPAT_ROOT/dataset/partnet" \
  --output-root "$DATA_ROOT/partnet-pilot-v1" \
  --manifest partnet-pilot-v1
```

This deterministically selects 48 seen-category train objects (16 each:
chair/lamp/faucet), 12 official validation objects (4 each), 12 seen test
objects (4 each), and 12 unseen tests (6 tables, 6 displays).  It rejects
objects with more than 20 parts and writes every source ID, source hash and
preprocessing hash into `manifest.json`.

For a full run, change only `--output-root` and `--manifest full`:

```bash
python scripts/prepare_partnet_gpat.py \
  --raw-root "$DATA_ROOT/partnet_raw" \
  --metadata-root "$DATA_ROOT/partnet_meta" \
  --gpat-root "$GPAT_ROOT/dataset/partnet" \
  --output-root "$DATA_ROOT/partnet-full" \
  --manifest full
```

## 3. Run the suites

Use the pilot config first.  It runs the 20-part CUDA preflight before Stage 3;
if reserved memory is over 22 GiB it changes only the Stage-3 target feature
count / geometry cloud cap from `512/2048` to `256/1024`, and records that
choice in `stage3_preflight.json` and the resolved config.

```bash
cd "$OUR_ROOT"
conda activate fragment-assembly

python scripts/run_experiment.py --suite pilot \
  --config configs/partnet_gpat_pilot.yaml \
  --data-root "$DATA_ROOT/partnet-pilot-v1" --run-root "$RUN_ROOT"
```

Reproduce GPAT before interpreting any comparison.  Commands are deliberately
explicit rather than hidden in this repository; first run the documented exact
and non-exact (`--ratio=1`) cases, then canonical and `--rand` target cases.

```bash
cd "$OUR_ROOT"
python scripts/run_experiment.py --suite gpat_reference \
  --run-root "$RUN_ROOT" --official-gpat-root "$GPAT_ROOT" \
  --official-checkpoint "$GPAT_ROOT/logs/pretrained/gpat.pth" \
  --official-command "python learning/assembler.py --eval --cat=Chair --exp=reference --cuda=0"
```

The exact official flags/checkpoint must be included in the command above. Run
one immutable reference directory per category/protocol combination.

Run all Stage-1 controls on the full processed data.  The suite creates the
four named conditions with seeds 41, 42 and 43: `scratch`, `pretrained_e2e`,
`pretrained_schedule`, and `no_compat_graph`.

```bash
python scripts/run_experiment.py --suite stage1_ablation \
  --config configs/partnet_gpat_full.yaml --seeds 41 42 43 \
  --data-root "$DATA_ROOT/partnet-full" --run-root "$RUN_ROOT"
```

Select the Stage-2 winner by mean validation aligned Chamfer (and retain the
best `scratch` seed as the end-to-end control):

```bash
python scripts/select_stage2_best.py --run-root "$RUN_ROOT"
```

Then train the six Stage-3 target-fidelity runs by supplying the two named
paths printed in `stage2_selection.json`. Each resulting Stage-3 checkpoint is
evaluated with GT, raw reconstruction, and oracle-registered reconstruction
targets.

```bash
python scripts/run_experiment.py --suite target_fidelity \
  --config configs/partnet_gpat_full.yaml --data-root "$DATA_ROOT/partnet-full" \
  --run-root "$RUN_ROOT" \
  --stage2-checkpoint best=/path/to/best/stage2_assembly.pt \
  --stage2-checkpoint scratch=/path/to/scratch/stage2_assembly.pt
```

Finally, evaluate the GT-target robustness curves.  The suite runs Gaussian
jitter, dropout, local holes, outliers and a global-SE(3) zero-degradation
control; every level uses five fixed seeds and reports measured post-alignment
Chamfer plus confidence intervals.

```bash
python scripts/run_experiment.py --suite target_noise \
  --config configs/partnet_gpat_full.yaml --data-root "$DATA_ROOT/partnet-full" \
  --run-root "$RUN_ROOT" \
  --stage2-checkpoint best=/path/to/best/stage2_assembly.pt \
  --stage3-checkpoint best=/path/to/target_fidelity/stage3_pose_gt_target.pt
```

For a controlled restart of exactly one named run, add `--resume-dir
"$RUN_ROOT/<timestamp>_<suite>_seedN"` and select one seed/condition (or one
checkpoint pair).  Existing completed stage checkpoints are reused and history
files append rather than being truncated.

## 4. Transfer a result, not the dataset

Each invocation creates `runs/<timestamp>_<suite>_<seed>/` with resolved config,
command, Git diff/revision, package locks, GPU profile, copied manifest, logs,
metric histories, checkpoints, predictions, registration diagnostics, fixed
qualitative IDs, and `RESULTS.md`.

```bash
python scripts/package_run.py --run-dir "$RUN_ROOT/<timestamp>_<suite>_<seed>"
```

Copy only the produced `run_bundle.tar.zst` into this workspace.  It contains no
raw PartNet archive or raw PartNet-derived dataset folder; it is sufficient for
post-run analysis and reproducibility checks.

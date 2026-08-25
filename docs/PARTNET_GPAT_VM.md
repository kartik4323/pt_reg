# PartNet / GPAT GPU runbook

Run this protocol only on the Linux/CUDA VM.  This repository intentionally
does not download or store PartNet, its metadata, or generated sample clouds.
The raw archive remains under `$DATA_ROOT` and run bundles exclude it.

## 1. Set up isolated environments

```bash
export OUR_ROOT="$HOME/Kartik_23CS30026/pt_reg"
export GPAT_ROOT="$HOME/Kartik_23CS30026/gpat"  # untouched official clone
export DATA_ROOT="$HOME/Kartik_23CS30026/partnet_data"
export RUN_ROOT="$HOME/Kartik_23CS30026/partnet_runs"

cd "$OUR_ROOT"
# You are already using this environment.
conda activate kartik_pt_env
pip install -r requirements.txt
# Recommended for robust FPFH + ICP gauge registration:
pip install open3d

git clone https://github.com/real-stanford/gpat "$GPAT_ROOT"
cd "$GPAT_ROOT"
conda env create -f environment.yml -n gpat
conda activate gpat
# The current official repository is not an installable Python package.
# Keep its root on the import path instead.
export PYTHONPATH="$GPAT_ROOT:${PYTHONPATH:-}"
# Prefer the cuBLAS shipped by the Conda environment over a system CUDA copy.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# PyTorch3D has no remaining pre-built wheel for Python 3.6 / Torch 1.10 / CUDA
# 11.1.  Build the pinned release from source.  GPAT itself uses its transforms.
conda install -y -c conda-forge ninja
pip install --no-build-isolation --no-cache-dir fvcore
pip install --no-build-isolation --no-cache-dir \
  "git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2"
# GPAT's unpinned environment can otherwise resolve a recent trimesh that
# requires numpy.typing (absent from the required NumPy 1.19 / Python 3.6).
pip install --no-cache-dir "trimesh==3.9.29"
# Keep TensorBoard and its flags dependency compatible with Python 3.6.
pip install --no-cache-dir "tensorboard==2.6.0" "absl-py==0.15.0"
# v0.6.2's datatypes module has a Python-3.7-only typing guard, although its
# stated requirements include Python 3.6.  This changes only the fallback
# typing helpers; it does not alter the GPAT checkout or PyTorch3D algorithms.
python - <<'PY'
from pathlib import Path
import sys

path = Path(sys.prefix) / "lib" / "python3.6" / "site-packages" / "pytorch3d" / "common" / "datatypes.py"
old = 'else:\n    raise ImportError("This module requires Python 3.7+")\n'
new = '''else:
    def get_origin(cls):
        return getattr(cls, "__origin__", None)

    def get_args(cls):
        return getattr(cls, "__args__", None)
'''
text = path.read_text()
if text.count(old) != 1:
    raise RuntimeError(f"Unexpected PyTorch3D datatypes.py contents: {path}")
path.write_text(text.replace(old, new))
PY
python -c "import torch, pytorch3d; from pytorch3d import transforms; print('PyTorch3D OK:', torch.__version__, torch.version.cuda, pytorch3d.__version__)"
cd utils/chamfer && python setup.py install
cd ../pointops && python setup.py install
```

Do not modify the official GPAT clone.  The `gpat_reference` suite records its
Git revision and complete command output separately from this project.

## 2. Obtain and prepare PartNet on the VM

Request PartNet v0 access with the ShapeNet account required by the PartNet
portal.  The official GPAT preprocessor iterates the public splits, so it
needs the complete PartNet v0 *annotation* archive.  Download only the
`data_v0_chunk` archive and its ten split volumes: GPAT does not need the
semantic- or instance-segmentation HDF5 archives.

```bash
pip install -U huggingface_hub
hf auth login
hf download --repo-type dataset ShapeNet/PartNet-archive \
  --include "data_v0_chunk.*" \
  --local-dir "$DATA_ROOT/PartNet-archive"

# Reassemble the split zip and extract it only on the VM.  The result must
# contain $DATA_ROOT/partnet_extracted/data_v0/<numeric-id>/result_after_merging.json.
zip -s 0 "$DATA_ROOT/PartNet-archive/data_v0_chunk.zip" \
  --out "$DATA_ROOT/data_v0.zip"
mkdir -p "$DATA_ROOT/partnet_extracted"
unzip -q "$DATA_ROOT/data_v0.zip" -d "$DATA_ROOT/partnet_extracted"
find "$DATA_ROOT/partnet_extracted" -maxdepth 3 \
  -name result_after_merging.json -print -quit
```

The existing `$DATA_ROOT/partnet_raw` directory is deliberately replaced only
when it is confirmed empty.  This keeps GPAT's expected numeric-ID layout
without moving or duplicating the raw data:

```bash
mkdir -p "$DATA_ROOT/partnet_raw"
[ -z "$(find "$DATA_ROOT/partnet_raw" -mindepth 1 -print -quit)" ] \
  || { echo "partnet_raw is not empty; inspect it before changing it"; exit 1; }
rmdir "$DATA_ROOT/partnet_raw"
ln -s "$DATA_ROOT/partnet_extracted/data_v0" "$DATA_ROOT/partnet_raw"
# This provides stats/after_merging_label_ids/*-hier.txt required by GPAT.
git clone https://github.com/daerduoCarey/partnet_dataset.git "$DATA_ROOT/partnet_meta"
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
conda activate kartik_pt_env
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
conda activate kartik_pt_env

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

# DiffAssemble: legacy-environment recovery and training

This compatibility profile targets the official source at
`256df64d4e5465e690b5b84e6024c60e21c3f3c8`, Linux x86_64, and CUDA 11.3.
It creates **`sota-diffassemble-repro-v1`**, leaving the old `sota-diffassemble`
environment, other models, and retained meshes untouched. The official clone is
unchanged; a guarded import-only patch is applied to disposable native copies.
Do not run the old upstream Conda YAML on top of this profile.

## Why the previous environment repeatedly failed

The upstream YAML pins Torch but leaves Python, PyG, Lightning's supporting
libraries, and several runtime dependencies unspecified. Resolving it today is
not the same as reconstructing the authors' environment. The old suite's smoke
command only ran `--help`; it never tested a training step.

| Failure or compatibility boundary | Profile choice |
| --- | --- |
| `iJIT_NotifyEvent` with the legacy Torch binary | MKL 2024.0.0 |
| Newer glibc rejects `libtorch_cpu.so`'s executable-stack flag | Conditional `patchelf` header repair, original library backup and before/after hashes |
| Removed `pkg_resources` | Setuptools 65.6.3 |
| Lightning imports private TorchMetrics `_compare_version` | Lightning 1.7.7 + TorchMetrics 0.9.2 |
| New pip rejects Lightning's old dependency metadata | pip 24.0 |
| Missing PyTorch3D / mismatched compiled operators | PyTorch3D 0.7.2, exact Python 3.10 / Torch 1.12.1 / CUDA 11.3 build |
| PyG extensions must match Torch's binary ABI | PyG 2.2.0 + `pt112cu113` wheels for scatter/sparse/cluster/spline |
| TorchVision's old JPEG dependency conflicts with newer Conda Pillow builds | TorchVision 0.13.1 + Pillow 9.2.0 (includes `Image.Resampling`) |
| Missing transitive imports | Explicit NetworkX, YACS, TensorBoard, and full native import-chain dependencies |
| Official backbone package imports absent `backbone_vist.py` | Recorded patch exports only the released `Eff_GAT_3d`; no 3D model/loss changes |
| Legacy scientific stack | NumPy 1.23.5 + SciPy 1.9.3 |
| Native split paths omit the subset directory | Adapter writes `everyday/Category/object`, separately from other model views |
| Native logger targets the authors' W&B account | Offline W&B logging; no account or upload required |

The recipe pins compatibility-sensitive packages and the audited pip transitive
versions in `constraints.resolved.txt`; installation also records
the complete resolved Conda package URLs, pip versions, and recipe hashes in the
setup run. These old packages are for isolated research reproduction, not an
internet-facing service.

The executable-stack repair is triggered only by that exact Torch import error.
It patches a private copy before replacing the environment-local library, so
Conda's hardlinked package cache and other environments are not modified. No
global loader/security settings are changed. `runtime-repair.json` records
whether a repair was necessary and where the original library is backed up.

`patches/3d-backbone-imports.patch` changes only the backbone package's eager
imports. Its preimage must match the pinned source; modified files are not
silently overwritten. `native-patches.json` records its hash and target hashes
for each setup/run. The patcher refuses to modify an `upstream/` or Git checkout.

## Continue after materializing Breaking Bad

First transfer this updated `sota_repro/` code to your Linux checkout. These are
local workspace changes until you commit/push and pull them, or copy them over.
Do not discard your Linux-side edits to update the checkout.

Run from the project root in your existing launcher environment:

```bash
cd /home/kpandey/satellite/pt_reg
conda activate breaking-bad

# Keep your EXISTING SOTA_DATA_ROOT: it contains the materialized corpus.
: "${SOTA_DATA_ROOT:?Set SOTA_DATA_ROOT to the existing retained dataset}"
test -f "$SOTA_DATA_ROOT/common_v1_manifest.json" || exit 1
export SOTA_RUN_ROOT="$(pwd)/runs/assembly_common_v1"
export SOTA_SCRATCH_ROOT="$(pwd)/scratch/assembly_repro/native_views"
mkdir -p "$SOTA_RUN_ROOT" "$SOTA_SCRATCH_ROOT"

# Confirms you have the revised registry, before installing anything.
python -c "from sota_repro.registry import load_registry; print(load_registry()['diffassemble'].data['environment']['name'])"
# Must print: sota-diffassemble-repro-v1

python -m sota_repro setup --model diffassemble --run-root "$SOTA_RUN_ROOT"
```

Setup creates the fresh environment, installs the complete profile, runs
`pip check`, checks imports independently, exercises CPU implementations of the
compiled libraries, constructs the native model/optimizer, and checks synthetic
mesh loading and offline point-cloud logging. It does not
require a GPU allocation or download data. Network access to Conda, PyPI and the
official PyG wheel host is required. Allow several GiB for packages and caches.
If setup is interrupted, rerun that same setup command: it updates the new
environment rather than failing with `prefix already exists`.

Then, on a GPU-allocated node:

```bash
nvidia-smi
python -m sota_repro smoke --model diffassemble \
  --data-root "$SOTA_DATA_ROOT" --run-root "$SOTA_RUN_ROOT" --gpu 0
```

The new smoke test checks all imports, package consistency, the native CLI,
model construction, CUDA operators (including PyTorch3D backward), and one real
native training batch followed by validation, checkpoint save/reload, and a
native test batch (including mesh export). For the batch diagnostic only,
it selects one smallest retained train/validation pattern and uses 20 diffusion
steps. This is not a score reproduction. Full training keeps the native
300-step schedule, 500 epochs, 1,000 points/part, batch size 1, and native model
and loss. Both paths run outside `upstream/`.
Do not use the generated `smoke-only.ckpt` for reported results.

Only after smoke finishes successfully:

```bash
python -m sota_repro train --model diffassemble \
  --data-root "$SOTA_DATA_ROOT" --run-root "$SOTA_RUN_ROOT" --gpu 0
```

`--gpu 0` means the first visible GPU. If your scheduler sets
`CUDA_VISIBLE_DEVICES`, its assigned device list is preserved and indexed
logically. The profile ships prebuilt CUDA 11.3 binaries: no `nvcc` compilation
is required. A GPU/driver unsupported by these legacy binaries cannot be fixed
by installing more Python packages; the CUDA smoke checks expose that boundary.

## Checkpoints, resume, and evaluation

Native W&B files, visualizations and Lightning checkpoints stay under the
training run's `native_source/`. Find the native checkpoints with:

```bash
find "$SOTA_RUN_ROOT" -type f -name '*.ckpt' -print
```

Choose the actual checkpoint, then:

```bash
export CHECKPOINT="/absolute/path/to/the/selected/checkpoint.ckpt"
test -f "$CHECKPOINT" || exit 1

python -m sota_repro train --model diffassemble \
  --data-root "$SOTA_DATA_ROOT" --run-root "$SOTA_RUN_ROOT" \
  --gpu 0 --resume "$CHECKPOINT"

# Alternatively, evaluate without resuming training:
python -m sota_repro test --model diffassemble \
  --data-root "$SOTA_DATA_ROOT" --run-root "$SOTA_RUN_ROOT" \
  --gpu 0 --checkpoint "$CHECKPOINT"
```

This native entry evaluates **everyday validation**. Artifact zero-shot
evaluation is not wired up and is rejected rather than silently reporting
everyday metrics as artifact results. Native evaluation does not yet export the
suite's common prediction JSONL. These commands are not a claim of end-to-end
paper-score reproduction.

## Diagnostics and verification scope

The terminal now shows native errors directly, alongside the exact
`native.stdout.log` path. Setup and smoke also save `doctor.json`, aggregating
individual failures so the next issue is not hidden by the first import error.
Setup records `conda-explicit.txt`, `pip-freeze.txt`, and `recipe-hashes.json`.
The run manifest records the post-setup environment and each execution's source
identity, seed and input checkpoint hashes.

Do not delete or download the data again for an environment repair. Do not run
the seven-model setup loop to fix DiffAssemble. If a check fails, retain the
whole `doctor.json` and `native.stdout.log`; no repeated blind package upgrades.

Validation completed: **33 Linux CPU checks passed**, including the native CLI,
model/optimizer construction, synthetic mesh loader, offline logger, and
Torch/PyG/PyTorch3D operators. See [the recorded CPU audit](validation/cpu-audit.json).
The suite also passes **15 regression tests** for adapters, logging, setup
retry, GPU selection and guarded source/runtime patches.

The CUDA batch test must still run on your GPU host; the development machine
has Intel graphics, not a CUDA GPU. Neither training on your retained dataset
nor full-paper results have been verified here. A passing import/CPU check
alone is not GPU validation.

Compatibility references: [official DiffAssemble environment](https://github.com/IIT-PAVIS/DiffAssemble/blob/256df64d4e5465e690b5b84e6024c60e21c3f3c8/singularity/build/conda_env.yaml),
[PyTorch3D 0.7.2 installation](https://github.com/facebookresearch/pytorch3d/blob/v0.7.2/INSTALL.md),
[official PyG Torch 1.12/CUDA 11.3 wheels](https://data.pyg.org/whl/torch-1.12.0+cu113.html),
[patchelf executable-stack documentation](https://github.com/NixOS/patchelf/blob/master/patchelf.1).

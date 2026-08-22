# Compatibility-Driven 3D Fragment Assembly

This repository is now focused on the two-stage pipeline:

1. Stage 1 learns binary fit/non-fit fragment compatibility with BCE + InfoNCE.
2. Stage 2 reconstructs the full object with a compatibility graph, residual
   GNN, refinement transformer, and cross-attention point decoder.
3. Stage 3 learns fragment-to-object SE(3) poses with a pose-sensitive branch,
   cross-attention to the reconstructed object, and optional ICP only as a
   baseline/refinement.

The default encoder in the provided configs is `se3_invariant`: learned scalar
fragment embeddings and tokens are invariant to global rotation/translation,
while centered `token_xyz` vectors are rotation-equivariant. The older raw-XYZ
PointNet path is still available as `type: "token_pointnet"` for ablations.
An optional `type: "se3_transformer"` backend is available for experiments with
`e3nn` and `torch-geometric`; those dependencies are intentionally not required
for the default lightweight install.

## PartNet / GPAT benchmark

The research benchmark uses PartNet and the official GPAT interchange format,
not the legacy ShapeNet generator.  It is designed for a Linux/CUDA VM and
never downloads PartNet into this workspace.  See
[the VM runbook](docs/PARTNET_GPAT_VM.md) for the pilot, official GPAT
reproduction, Stage-1 ablation matrix, Stage-3 gauge-aware study, noise suite,
and portable result-bundle commands.

## Quick Mock Run

```powershell
python scripts\run_two_stage_pipeline.py --config configs\two_stage_mock.yaml --mode smoke --device cpu
```

For a longer mock run:

```powershell
python scripts\run_two_stage_pipeline.py --config configs\two_stage_mock.yaml --mode all --device cpu
```

Outputs are written to the directory configured by `output.dir`.

### Generated run artifacts

Every training/eval run writes self-describing files into `output.dir` so a run
can be diagnosed after the fact (useful when training on a remote GPU box and
copying the directory back):

- `run_manifest_{stage}.json` — resolved config + environment (torch/CUDA/device,
  host, timestamp) that produced the run.
- `{stage}_history.jsonl` — one JSON line per epoch: all loss/metric values, the
  learning rate(s), per-epoch wall-time (and, for Stage 3, the loss weights).
- `{stage}_train.log` — plaintext mirror of the per-epoch console summary lines.
- `stage3_eval_{reconstruction,ground_truth}.json` — pose-error **distributions**
  (mean / p50 / p90 / p95 / max), not just means. Named by target source so a
  ground-truth eval no longer clobbers the reconstruction eval.
- `oracle_alignment_report.json` — output of the pre-flight label check (below).
- `RESULTS.md` — human-readable roll-up of whatever artifacts are present.

Copy the whole `output.dir` back for analysis; `RESULTS.md` + the `*_history.jsonl`
files are the fastest way to see what happened.

## Use The Original ShapeNet Dataset

The training code expects point clouds in this layout:

```text
shape_processed/
  metadata.json
  02691156/
    <object_id>.npy
  03001627/
    <object_id>.npy
```

`metadata.json` must map each synset to the available object IDs:

```json
{
  "02691156": ["object_a", "object_b"],
  "03001627": ["object_c"]
}
```

### 1. Get And Prepare ShapeNet From Scratch

ShapeNetCore is gated. First request/accept access on the dataset host, then
use a token from that account. With Hugging Face access approved:

```powershell
$env:HF_TOKEN="hf_your_token_here"   # do NOT commit a real token

python scripts\get_shapenet_data.py `
  --source huggingface `
  --output-root .\shape_processed `
  --categories 02691156 02828884 02933112 02958343 03001627 03211117 03636649 03691459 04256520 04379243 `
  --num-points 4096
```

The command downloads the gated repository, extracts any archives it finds, and
creates the `.npy` point-cloud dataset plus `metadata.json`.

If the repository layout or access flow downloads more than you want, add
`--hf-all-files` to mirror the full repository. Without it, the script tries to
download category-matching files only.

#### Low-disk machines: `--per-category`

By default the script extracts **all** category archives before converting any
of them, so it needs every category's extracted meshes on disk at once (far more
than the downloaded zips). On a space-constrained box this fails with
`No space left on device` partway through. Add `--per-category` to stream one
synset at a time — extract → convert to `.npy` → delete that synset's extracted
meshes before the next — which caps peak disk at roughly a single category. Add
`--delete-archives` to also delete each source archive once its category is
converted. The mode rebuilds a correct cumulative `metadata.json` at the end and
is resumable: synsets already present in `--output-root` are skipped.

If you already downloaded the zips (they live under `--download-root`, default
`./downloads/shapenet_hf`), add `--skip-download` so it reuses them:

```powershell
python scripts\get_shapenet_data.py `
  --source huggingface `
  --skip-download `
  --per-category `
  --delete-archives `
  --output-root .\shape_processed `
  --categories 02691156 02828884 02933112 02958343 03001627 03211117 03636649 03691459 04256520 04379243 `
  --num-points 4096
```

If you already have ShapeNet zip/tar files, skip Hugging Face and prepare from
local archives:

```powershell
python scripts\get_shapenet_data.py `
  --source archive `
  --archive-root C:\path\to\ShapeNetArchives `
  --extract-root .\downloads\ShapeNetCore `
  --output-root .\shape_processed `
  --categories 02691156 02828884 02933112 02958343 03001627 03211117 03636649 03691459 04256520 04379243 `
  --num-points 4096
```

Use `--max-objects-per-category 100` for a small pilot run.

If you already have extracted meshes, you can use the converter directly:

```powershell
python scripts\prepare_shapenet_points.py `
  --input-root C:\path\to\ShapeNetCore `
  --output-root .\shape_processed `
  --categories 02691156 02828884 02933112 02958343 03001627 03211117 03636649 03691459 04256520 04379243 `
  --num-points 4096
```

### 2. Edit The YAML

Use `configs/two_stage_shapenet.yaml`. As shipped it trains on 10 categories
(`02691156 02828884 02933112 02958343 03001627 03211117 03636649 03691459
04256520 04379243`). Edit the `data` block to point at your data and, if you
want a smaller/different set, trim the `categories` list — e.g.:

```yaml
data:
  shapenet_root: "./shape_processed"
  categories:
    - "02691156"
    - "03001627"
    - "04379243"
```

If your real data is somewhere else, set `data.shapenet_root` to that folder.
If you use different ShapeNet categories, replace the synset IDs in both the
preparation command and the YAML file.

### 3. Run Training

```powershell
python scripts\run_two_stage_pipeline.py --config configs\two_stage_shapenet.yaml --mode stage1 --device cuda

python scripts\run_two_stage_pipeline.py `
  --config configs\two_stage_shapenet.yaml `
  --mode stage2 `
  --stage1-checkpoint outputs_two_stage_shapenet\stage1_pretrained.pt `
  --device cuda

python scripts\run_two_stage_pipeline.py `
  --config configs\two_stage_shapenet.yaml `
  --mode stage3-train `
  --stage2-checkpoint outputs_two_stage_shapenet\stage2_assembly.pt `
  --device cuda

# Note: `--mode stage3` runs stage 3 evaluation only; use `--mode stage3-train` to train and save the stage 3 model.
```

To run everything in sequence:

```powershell
python scripts\run_two_stage_pipeline.py --config configs\two_stage_shapenet.yaml --mode all --device cuda
```

Use `--device cpu` if CUDA is not available.

Before training Stage 3, validate the synthetic pose labels and transform
convention with the oracle alignment check:

```powershell
python scripts\oracle_stage3_alignment.py `
  --config configs\two_stage_shapenet.yaml `
  --split train `
  --num-samples 64 `
  --device cuda
```

The oracle fragment MSE and pose errors should be near zero, and the oracle
union Chamfer should be clearly better than random/model output. If not, fix
the transform labels before training.

### 4. Run Inference

After Stage 2 training, reconstruct from a dataset sample:

```powershell
python scripts\infer_assembly.py `
  --config configs\two_stage_shapenet.yaml `
  --checkpoint outputs_two_stage_shapenet\stage2_assembly.pt `
  --split test `
  --index 0 `
  --device cuda `
  --output-dir outputs\inference
```

Or reconstruct from explicit fragment `.npy` files:

```powershell
python scripts\infer_assembly.py `
  --config configs\two_stage_shapenet.yaml `
  --stage2-checkpoint outputs_two_stage_shapenet\stage2_assembly.pt `
  --fragments fragment_a.npy fragment_b.npy fragment_c.npy `
  --device cuda `
  --output-dir outputs\inference
```

To test the whole pipeline from one complete object point cloud, let the script
cut the object into fragments, reconstruct it, run Stage 3 assembly, and compare
against the original object:

```powershell
python scripts\infer_assembly.py `
  --config configs\two_stage_shapenet.yaml `
  --stage1-checkpoint outputs_two_stage_shapenet\stage1_pretrained.pt `
  --stage2-checkpoint outputs_two_stage_shapenet\stage2_assembly.pt `
  --stage3-checkpoint outputs_two_stage_shapenet\stage3_pose.pt `
  --object-point-cloud shape_processed\02691156\<object_id>.npy `
  --num-fragments 5 `
  --cut-strategy irregular `
  --pose-mode auto `
  --device cuda `
  --output-dir outputs\inference
```

The script writes `.npy` and `.ply` reconstructions plus compatibility scores
and metadata. In object-point-cloud mode it also writes aligned fragments,
aligned union point clouds, predicted rotations/translations, the generated
input fragments, the ground-truth object point cloud, and comparison metrics.

Visualize an inference folder:

```powershell
python scripts\visualize_inference.py `
  --dir outputs\inference `
  --prefix <output_prefix> `
  --show-target `
  --show-aligned `
  --show-fragments
```

Use `--list` to show available prefixes. The visualizer uses Open3D when
installed and falls back to matplotlib.

The default (multi-view) mode needs `<prefix>_target.npy`,
`<prefix>_reconstruction.npy`, and `<prefix>_aligned_union.npy` to all exist,
so run inference with `--save-inputs` and a Stage 3 pose (object-point-cloud
mode does both automatically). For a plain dataset reconstruction without those
files, pass `--legacy-scene` instead.

## Troubleshooting

If PowerShell prints `No pyvenv.cfg file` before the script starts, your
terminal is pointing at a broken local virtual environment. Reset the terminal
Python path or use the Python launcher directly:

```powershell
deactivate
py -3.10 -m pip install -r requirements.txt
py -3.10 scripts\get_shapenet_data.py --help
```

If a stale `py312_env` folder remains in this project, close any terminal or
process using it, then delete that folder. The project does not require
`py312_env`; use `env.yml`, `requirements.txt`, or any clean Python 3.10+
environment.

## Important Code Switches

- Mock data is controlled by `mock_data.enabled`. Keep it `false` for real
  ShapeNet training.
- Reconstruction resolution is controlled by
  `model.assembly.output_points` and `data.num_points_per_object`.
- Per-fragment input resolution is controlled by
  `data.num_points_per_fragment`.
- Fragment count is controlled by `fragment.min_fragments` and
  `fragment.max_fragments`; keep `max_fragments < 10`.
- Stage 1 pair sampling is controlled by `pairs.positive_ratio`,
  `pairs.easy_negative_ratio`, `pairs.hard_negative_ratio`, and
  `pairs.perturbed_positive_ratio`.
- To try the optional graph SE(3) backend, install `e3nn` and
  `torch-geometric`, then set `model.encoder.type: "se3_transformer"`.
- If memory is tight, reduce `stage2.batch_size`,
  `model.assembly.output_points`, or `data.num_points_per_fragment`.

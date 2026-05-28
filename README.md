# Compatibility-Driven 3D Fragment Assembly

This repository is now focused on the two-stage pipeline:

1. Stage 1 learns binary fit/non-fit fragment compatibility with BCE + InfoNCE.
2. Stage 2 reconstructs the full object with a compatibility graph, residual
   GNN, refinement transformer, and cross-attention point decoder.
3. Stage 3 estimates post-hoc SE(3) fragment poses with ICP.

The default encoder in the provided configs is `se3_invariant`: learned scalar
fragment embeddings and tokens are invariant to global rotation/translation,
while centered `token_xyz` vectors are rotation-equivariant. The older raw-XYZ
PointNet path is still available as `type: "token_pointnet"` for ablations.
An optional `type: "se3_transformer"` backend is available for experiments with
`e3nn` and `torch-geometric`; those dependencies are intentionally not required
for the default lightweight install.

## Quick Mock Run

```powershell
python scripts\run_two_stage_pipeline.py --config configs\two_stage_mock.yaml --mode smoke --device cpu
```

For a longer mock run:

```powershell
python scripts\run_two_stage_pipeline.py --config configs\two_stage_mock.yaml --mode all --device cpu
```

Outputs are written to the directory configured by `output.dir`.

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
$env:HF_TOKEN="hf_your_token_here"

python scripts\get_shapenet_data.py `
  --source huggingface `
  --output-root .\shape_processed `
  --categories 02691156 03001627 04379243 `
  --num-points 4096
```

The command downloads the gated repository, extracts any archives it finds, and
creates the `.npy` point-cloud dataset plus `metadata.json`.

If the repository layout or access flow downloads more than you want, add
`--hf-all-files` to mirror the full repository. Without it, the script tries to
download category-matching files only.

If you already have ShapeNet zip/tar files, skip Hugging Face and prepare from
local archives:

```powershell
python scripts\get_shapenet_data.py `
  --source archive `
  --archive-root C:\path\to\ShapeNetArchives `
  --extract-root .\downloads\ShapeNetCore `
  --output-root .\shape_processed `
  --categories 02691156 03001627 04379243 `
  --num-points 4096
```

Use `--max-objects-per-category 100` for a small pilot run.

If you already have extracted meshes, you can use the converter directly:

```powershell
python scripts\prepare_shapenet_points.py `
  --input-root C:\path\to\ShapeNetCore `
  --output-root .\shape_processed `
  --categories 02691156 03001627 04379243 `
  --num-points 4096
```

### 2. Edit The YAML

Use `configs/two_stage_shapenet.yaml` and change:

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
  --mode stage3 `
  --stage2-checkpoint outputs_two_stage_shapenet\stage2_assembly.pt `
  --device cuda
```

To run everything in sequence:

```powershell
python scripts\run_two_stage_pipeline.py --config configs\two_stage_shapenet.yaml --mode all --device cuda
```

Use `--device cpu` if CUDA is not available.

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

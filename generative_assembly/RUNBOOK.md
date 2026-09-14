# Run E0–E7 and retain reusable pipeline artifacts

Run commands from the `pt_reg` repository root. This study lives in `generative_assembly/`; keep its environments, prepared data, run roots and exports separate from existing repair/scaffold-transfer work.

| Stage | Question tested | Main retained outputs |
|---|---|---|
| E0 | Are coordinates, rendering and geometry proposals usable? | Observed points, pose candidates, RGB/depth/masks, camera, checks |
| E1 | Can a pretrained editor complete fragment renders while preserving evidence? | Every image hypothesis, selection scores, silhouette metrics, human-review sheet |
| E2 | Does image-to-3D preserve useful geometry? | Meshes, sampled point clouds, alignment transforms, shape/exterior errors |
| E3 | Could a correct complete shape help assembly at all? | Oracle/wrong/distorted-template controls, global and perturbed-start poses |
| E4 | Does generated guidance improve geometric assembly? | Raw-image, generated, geometry, compute and privileged-control results |
| E5 | Do multiple hypotheses and confidence gating help? | Nested hypothesis budgets, accepted/rejected guidance, paired comparisons |
| E6 | Do inferred assemblies teach a useful assembler? | Teacher labels, four learner controls, checkpoints, learning curves, student metrics |
| E7 | Does the method tolerate changed observations? | Fresh corrupted-input generations, poses and comparisons by corruption level |

Experiment completion and scientific success are distinct. Run the low-cost checks first, inspect development gates, and only then spend the full generation/training budget.

## What is implemented and what must be measured

All eight experiment stages have execution paths. Core generation uses SD 1.5 inpainting + depth ControlNet and Qwen-Image-Edit-2509; reconstruction uses the pinned official InstantMesh script in an isolated environment. The small assembly learner is a study-specific PointNet baseline, not a pretrained SOTA assembler. Optional SDXL and SD 1.5 without depth use the same image worker. Another 3D model can be connected through an explicit argv adapter.

CPU tests cover orchestration, mathematical coordinate contracts, reference isolation, checkpoint creation/inference, failures, resume checks and exports. This machine currently has CPU-only PyTorch; real diffusion/InstantMesh inference, CUDA memory feasibility, and learned assembly quality have not been measured here. Smoke runs contain deliberately non-generative substitutes and cannot be exported as a real pipeline.

Implementation qualifications: exterior classification uses a fixed curvature heuristic; interpenetration is an oriented-surface proxy, not exact solid intersection; reference silhouettes use point splats; camera preservation needs the saved image-review sheet. E7 erosion removes likely fracture-region points as a proxy, not a physical material-loss simulation. E6 implements the first self-training round. Further training rounds and physical erosion should be separate follow-up studies if the first round works. These limitations are recorded with results rather than hidden in successful job status.

## 1. Verify the software on CPU

Use an environment with Python 3.10+ and PyTorch. The core requirements do not install the large generative models.

```bash
python -m pip install -r generative_assembly/requirements.txt
python -m unittest discover -s generative_assembly/tests -v
python -m generative_assembly demo --out /absolute/path/ga-smoke-data --sources 8
bash generative_assembly/run_all.sh generative_assembly/configs/smoke.json /absolute/path/ga-smoke-data/dataset.json /absolute/path/ga-smoke-run smoke
```

The tests use temporary directories. The explicit smoke command retains inspectable artifacts. Its report must say `SMOKE_NOT_RESEARCH_EVIDENCE`, even if any geometric scores happen to be good.

On Windows, use the PowerShell launcher below. `run_all.sh` is the Linux/Bash equivalent; all individual CLI commands also work in PowerShell. Use quoted absolute Windows paths.

```powershell
python -m generative_assembly demo --out "$PWD/ga-smoke-data" --sources 8
./generative_assembly/run_all.ps1 -Config ./generative_assembly/configs/smoke.json -Dataset "$PWD/ga-smoke-data/dataset.json" -Root "$PWD/ga-smoke-run" -Phase smoke
```

For real experiments use the same launcher with the locked config and replace `-Phase smoke` with `dev`, then `train`, `test`, `robustness`, and optionally `real`, reviewing the scientific gates between phases. GPU model installation is documented for Linux; the Windows launcher has only been validated with CPU smoke backends.

## 2. Prepare real fragment data

See [DATA_PREPARATION.md](DATA_PREPARATION.md) for ready-to-run commands that reuse the existing bottle dataset, its verified Windows location, server discovery, and preparation from an existing archive. Reuse that original prepared directory for comparisons with the other pipeline.

### Existing prepared v2 data

Use the original *prepared dataset manifest*, not a results/diagnostic bundle:

```bash
python -m generative_assembly import-v2 --input /absolute/path/original-prepared/manifest.json --out /absolute/path/ga-data
```

This copies observed point reservoirs into separately posed fragment clouds. It strips fracture labels and complete objects from the public input NPZ files and stores references under `evaluator_only/`. It retains source-level splits, converting val → dev and cut_holdout → test. Existing bottle data is labelled bottle; it does not become a three-category benchmark. A test set that has already guided previous work must be treated as development evidence; create a new source holdout for a fresh confirmatory claim.

### Breaking Bad meshes or your scans

Create a JSON import specification. Paths below are relative to that JSON file. A source_id must identify the original object, not a fracture pattern. All patterns from one source must use one split. Include train, dev and test pools before starting; real is an optional fourth split. Supply a source_hash when near-identical meshes may have multiple names.

```json
{
  "cases": [
    {
      "id": "vase001-break01",
      "source_id": "vase001",
      "category": "vase",
      "split": "dev",
      "canonical_fragments": true,
      "reservoir_points": 4096,
      "fragments": ["vase001/piece_0.obj", "vase001/piece_1.obj"],
      "complete_mesh": "vase001/complete.obj"
    },
    {
      "id": "scan002",
      "source_id": "scan002",
      "category": "unknown",
      "split": "real",
      "canonical_fragments": false,
      "fragments": ["scan002/a.ply", "scan002/b.npy"]
    }
  ]
}
```

For Breaking Bad, `canonical_fragments:true` means the supplied meshes are already in their original common assembly frame. The importer independently rotates/translates them and saves those transformations only for evaluation. For actual unposed scans use false. With false, an optional `world_transforms` .npy file can supply one 4×4 input-to-common-reference transform per piece for evaluation. Never set canonical_fragments true on genuinely unassembled scans.

Complete meshes and reference poses are optional for real inference. Without them, the report records `unscored_no_reference`; input-only diagnostics cannot certify true assembly accuracy. A point-only .npy file contains an N×3 array. Meshes are surface-sampled, retaining supplied geometric normals. No target cloud is invented by taking a convex hull of the fragments.

```bash
python -m generative_assembly import-spec --input /absolute/path/import-spec.json --out /absolute/path/ga-data
```

The public dataset contract is `dataset.json` plus `inputs/*.npz`. NPZ keys are `points_0`, `points_1`, etc., and optional `normals_i`. Other keys, including targets and fracture labels, are rejected by the public loader. Preserve the matching evaluator_only directory for later scoring, but model workers and the training loader do not consume it.

## 3. Set up the GPU environments

Use a separate Linux CUDA environment for image generation and study orchestration, and a second one for InstantMesh's upstream dependency stack. Do not install into an environment running another experiment. Install a PyTorch CUDA build compatible with the host driver before these requirements.

```bash
python3.10 -m venv /absolute/path/ga-env
source /absolute/path/ga-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r generative_assembly/requirements-images.txt
```

Clone [InstantMesh](https://github.com/TencentARC/InstantMesh) into a separate tools directory and follow its official installation instructions in `/absolute/path/instantmesh-env`. Its compiled dependencies and old diffusion stack should stay separate from the current Qwen/Diffusers environment. Install this study's core NumPy/SciPy/Pillow/trimesh dependencies there as needed; workers locate this package through PYTHONPATH. The reconstruction worker invokes the upstream run.py with `--no_rembg` so it does not silently resize/recenter the observed foreground.

```bash
git clone https://github.com/TencentARC/InstantMesh.git /absolute/path/InstantMesh
```

The upstream source must be clean and its HEAD will be locked before the study begins. The worker pins checkpoint revisions for `TencentARC/InstantMesh` and `sudo-ai/zero123plus-v1.2`. Save the environment lock from both environments before starting. Avoid preexisting untracked local checkpoints in the InstantMesh checkout unless their hashes are recorded with the experiment.

Qwen's full checkpoint is large. A 24-GB GPU alone is not a promise it will fit: CPU offload still needs substantial host RAM. The worker uses BF16 for Qwen and the configured precision for SD. Start with one case and inspect actual memory. For a first smaller run set `image_models` to `["sd15_depth"]`; use a different study root/config when adding Qwen. A quantized Qwen backend must be a separately declared command/adapter condition, not relabelled as the full checkpoint.

## 4. Configure and lock the models

Copy `generative_assembly/configs/pilot.json` to a study-owned configuration outside the source tree. Edit:

- `reconstruction.repo`: absolute InstantMesh checkout path.
- `reconstruction.python`: absolute Python executable in its environment.
- `images.python`: null uses the current Python; otherwise use the dedicated image environment.
- `training.device`, available CPU/GPU resources, generation models and budgets.
- `primary_model`, `primary_input` and `primary_policy`: fixed main pipeline choices, not chosen on the final test.

```bash
python -m generative_assembly lock-models --config /absolute/path/ga-pilot.json --out /absolute/path/ga-locked.json
python -m generative_assembly doctor --config /absolute/path/ga-locked.json --dataset /absolute/path/ga-data/dataset.json --root /absolute/path/ga-run
```

Run lock-models before doctor/run, since the resolved config participates in run identity. The command resolves exact Hugging Face revisions and the local InstantMesh commit. It does not start image generation. First inference downloads model weights to the configured Hugging Face cache. Record `HF_HOME` if you choose a dedicated cache. `pip freeze` from each environment should be saved with your run, e.g. `ga-env.freeze.txt` and `instantmesh-env.freeze.txt`.

The executable default uses 512 observed points/fragment for the bounded baseline, with 2,048 evaluation points. Increase observed points before creating a new run identity for the higher-density study. Four reconstructed hypotheses are retained for E5 by default: twelve objects × two inputs × two image models × four seeds gives 192 edited images and **228 reconstructions** including 24 raw and 12 true-image controls. Setting `reconstruct_all_for_E5:false` reduces this to 132 reconstructions, but some K=4 conditions then have incomplete budgets and cannot support a matched-budget claim.

## 5. Run development experiments

For a single-case environment smoke with real pretrained models:

```bash
python -m generative_assembly run --config /absolute/path/ga-locked.json --dataset /absolute/path/ga-data/dataset.json --root /absolute/path/ga-run --split dev --limit 1 --stages E0 E1 E2
```

This checks actual model installation and resource use. It is not the synthetic smoke substitute. Inspect `worker.log`, `backend.json`, image.png, mesh.obj and alignment.json.

Then run the whole development suite:

```bash
bash generative_assembly/run_all.sh /absolute/path/ga-locked.json /absolute/path/ga-data/dataset.json /absolute/path/ga-run dev
```

The equivalent stage order is E0, E1, E2, E3, E4, E5, then evaluate. You can run E3 after E0 to test the oracle shape-to-pose mechanism before paying for generation. Each stage can be invoked separately:

```bash
python -m generative_assembly run --config /absolute/path/ga-locked.json --dataset /absolute/path/ga-data/dataset.json --root /absolute/path/ga-run --split dev --stages E3
python -m generative_assembly evaluate --config /absolute/path/ga-locked.json --dataset /absolute/path/ga-data/dataset.json --root /absolute/path/ga-run --split dev
```

Inspect `evaluation/dev/REPORT.md`, summary.json, metrics.jsonl and image_review.json. Complete the camera/exterior/single-object review using saved images. The report does not fabricate those judgements. In E3, a template that helps only with privileged starts is not proof of global assembly. E4 B5 is capped by time and candidate batches; `compute_matched:false` means it is not a genuine matched-wall-time control and cannot exclude extra-compute explanations.

Scientific gates are review decisions; a completed stage is not automatically a pass. Do not run self-training just because all files exist. If development results lead to changing the method, use a new run root with the modified config/code. Report any use of development ground truth to tune the method.

## 6. Generate pseudo-labels and run E6

Once development inference is useful, run:

```bash
bash generative_assembly/run_all.sh /absolute/path/ga-locked.json /absolute/path/ga-data/dataset.json /absolute/path/ga-run train
```

This generates training-fragment hypotheses, creates all/filtered pseudo-labels, trains geometry-only, all-label, filtered-label and count-matched random-label models, and runs those students on development objects. Oracle image/template branches are disabled on the training split. Three seeds are configured by default. Each job saves optimizer/RNG state for step-level resume.

The teacher is frozen. The filtered and random-count controls have equal eligible-label counts; the all-label control retains all available teacher outputs. If the confidence gate accepts zero cases, filtered training fails explicitly. That is evidence the pseudo-label mechanism is not ready. Do not lower thresholds after seeing final-test results or turn an empty training pool into a successful job.

Compare the student against both the geometry-only learner and frozen teacher; source-paired comparisons against B0 alone are supplementary. Training loss is not an acceptance metric. No second pseudo-label training round is launched automatically.

## 7. Freeze, then run the locked synthetic test

After all method and learning choices are final:

```bash
bash generative_assembly/run_all.sh /absolute/path/ga-locked.json /absolute/path/ga-data/dataset.json /absolute/path/ga-run test
```

This freezes method/checkpoint IDs, executes E0–E6 on test, evaluates, and verifies artifact hashes. The CLI refuses test inference before freeze and refuses further training after freeze. The main E4 gate requires the configured success improvement, a positive source-paired confidence interval and bounded damage to initially successful cases. Inconclusive results remain inconclusive.

For an inference-only study where E6 was deliberately not justified, invoke `freeze`, then `run --split test --stages E0 E1 E2 E3 E4 E5`, and evaluate separately. Do not describe that result as a completed self-training experiment.

## 8. Run robustness and real-scan transfer

```bash
bash generative_assembly/run_all.sh /absolute/path/ga-locked.json /absolute/path/ga-data/dataset.json /absolute/path/ga-run robustness
bash generative_assembly/run_all.sh /absolute/path/ga-locked.json /absolute/path/ga-data/dataset.json /absolute/path/ga-run real
```

E7 creates independently keyed observations and **reruns generation/reconstruction** for each corruption. It never reuses a clean input's generated template as if it were conditioned on the corrupted scan. This multiplies model cost substantially; prespecify reduced factors/repeats in a separate run if needed. Normalization is held fixed from the original observed set to isolate the corruption factor. Rotation predictions are mapped back before evaluation. Missing-piece trials retain the reference fragment and require at least two remaining pieces. Different initial fragment counts are supplied by the dataset, not invented from a two-piece example.

The real phase requires split=real records. Without reference poses it saves useful predictions and confidence/contact diagnostics, while marking accuracy unscored. Unseen-category claims require a genuinely excluded category in the task-specific training/development pools.

## 9. Resume, inspect failures and preserve results

Rerun the same command to reuse completed jobs. Outputs are checked by SHA-256 before reuse. A missing/modified artifact fails verification instead of silently changing the experiment. Failed jobs require an explicit retry:

```bash
python -m generative_assembly status --config /absolute/path/ga-locked.json --dataset /absolute/path/ga-data/dataset.json --root /absolute/path/ga-run
python -m generative_assembly run --config /absolute/path/ga-locked.json --dataset /absolute/path/ga-data/dataset.json --root /absolute/path/ga-run --split dev --stages E1 E2 --retry-failed
python -m generative_assembly train --config /absolute/path/ga-locked.json --dataset /absolute/path/ga-data/dataset.json --root /absolute/path/ga-run --retry-failed
```

The default stops on failed jobs. `run --allow-failures` continues a benchmark while retaining explicit failures in result denominators. It does not repair dependencies or produce fake outputs. Always inspect status.json afterwards.

After a process dies, a running.lock may remain. Inspect its PID, host and timestamp. Only after confirming the process is dead:

```bash
python -m generative_assembly unlock --lock /absolute/path/ga-run/jobs/JOB_ID/running.lock --confirmed-process-dead
```

Do not run concurrent mutations against the same study root. Model workers are sequential by default, suitable for a shared single GPU. This launcher does not stop other users' GPU jobs or manage their environments.

## 10. Export material for the final pipeline

```bash
python -m generative_assembly verify --config /absolute/path/ga-locked.json --dataset /absolute/path/ga-data/dataset.json --root /absolute/path/ga-run
python -m generative_assembly export --config /absolute/path/ga-locked.json --dataset /absolute/path/ga-data/dataset.json --root /absolute/path/ga-run --kind pipeline --out /absolute/path/ga-pipeline-assets
python -m generative_assembly export --config /absolute/path/ga-locked.json --dataset /absolute/path/ga-data/dataset.json --root /absolute/path/ga-run --kind research --out /absolute/path/ga-research-evidence
```

Use a new export directory each time. The pipeline bundle excludes oracle and smoke jobs and evaluator reference files. It includes the exact code, locked model references, observed inputs, generated assets, selected poses, accepted/rejected pseudo-label records, trained student checkpoints and `pipeline_recipe.json`. `SHA256SUMS.json` verifies portability. The research bundle additionally preserves oracle controls and evaluator data.

Read [ARTIFACTS.md](ARTIFACTS.md) for the schema. These exports provide everything needed to select and integrate the successful components into a later final pipeline, while retaining the failures needed to design fallback behavior. Export does not automatically mark a research model production-approved.

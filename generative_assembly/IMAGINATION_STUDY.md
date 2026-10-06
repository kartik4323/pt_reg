# Controlled imagination study v2

Deploy the updated `generative_assembly/` package together with `run_imagination_study.sh`
and `imagination_state_analysis.ipynb` on the Linux inference host. Running only the
new script against the old package will not work. The original executed notebook
is preserved. No training or model installation is performed by the runner.

## First run

From the repository's `pt_reg` directory:

```bash
LIMIT=3 STAGES="E0 E1" bash run_imagination_study.sh
```

The runner prints a new timestamped `RUN_GROUP`. Open
`imagination_state_analysis.ipynb`, set that path, and Run All. Inspect raw and
cleaned images, foreground validity, growth, protected-region retention, clipping,
and missing-region precision/recall. The notebook embeds images rather than linking
to server-only paths. It exports every metric row to `analysis/metrics.csv` and JSON.

Once E1 outputs are worth reconstructing, resume the exact group:

```bash
RUN_GROUP=/home/kpandey/generative_assembly/runs/imagination-v2-TIMESTAMP \
LIMIT=3 STAGES="E2" bash run_imagination_study.sh
```

Or run E0/E1/E2 in one invocation with `LIMIT=3 bash run_imagination_study.sh`.
Defaults produce 192 edited images across four profiles and three cases, plus raw
and complete-image controls; up to 228 reconstructions before invalid-image skips.
Use `LIMIT=1` for an initial resource check. `LIMIT=10` runs all ten development cases.

## Controlled profiles

All default profiles use one fixed, padded orthographic camera, the point-splat
renderer, both F and A inputs, seeds 11/23/37/51, both SD15 and SD15-depth, and an
explicit bottle category. A is the best geometry-proposal assembly, not GT assembly.
The fixed renderer avoids silent Poisson fallback during the main comparisons.

| Profile | Prompt | Protection | Whitening |
| --- | --- | --- | --- |
| legacy_cleanup | Existing short prompt | Dilated entire silhouette | 140 |
| clean | Existing short prompt | Dilated entire silhouette | Disabled |
| completion | Explicit broken-object completion | Dilated entire silhouette | Disabled |
| exterior | Explicit broken-object completion | Estimated exterior only | Disabled |

`legacy_cleanup` isolates the historical whitening rule under the new fixed camera
and renderer. It is not an exact reproduction of an older run's unknown config.
Within a profile, SD15-depth versus SD15 isolates stock depth conditioning.
Unknown depth remains zero in the stock depth ablation; no claim is made that it
represents measured background. No-protection is an optional diagnostic:

```bash
PROFILES="exterior unprotected" LIMIT=3 bash run_imagination_study.sh
```

Optional instruction editor and renderer comparisons:

```bash
PROFILES="exterior qwen" LIMIT=1 bash run_imagination_study.sh
PROFILES="qwen qwen_depth" LIMIT=1 bash run_imagination_study.sh
PROFILES="exterior surface" LIMIT=1 bash run_imagination_study.sh
```

Qwen uses the existing Qwen-Image-Edit-2509 adapter: image-only by default, image
plus depth in `qwen_depth`. It is instruction editing, not mask-conditioned
inpainting. The image environment must already support its pipeline and the GPU
must support the configured BF16/offload path; start with one case. `surface`
uses the existing Poisson-to-dense-splats renderer, which may extrapolate geometry;
it is not a triangle-rasterized ground-truth mesh render.

`PROFILES="exterior gemini"` enables the existing API adapter. Supply credentials
through the environment and an account with image-generation quota. Gemini's API
does not use the SD seed, so four requests are independent repeats, not reproducible
seeded generations. API failures remain recorded. Model downloads/revision locking
occur through the standard `lock-models` command.

## E2 and three priors

Every valid E1 image is reconstructed for diagnosis. Unchanged fragments remain
in E2 as controls, but generated hypotheses must have at least 5% foreground growth
relative to input and held-out observed error <= 0.08 to enter prior selection.
These input-only heuristics are declared engineering thresholds, not calibrated
confidence. Sample growth can also come from haze, so inspect images and geometry.

Rejected templates have only `raw` in `shape.npz`; neither the notebook nor the
standard evaluator scores them as aligned. The shape gate uses PCA thickness to
avoid rejecting a long bottle merely because it is slender. Up to three candidates
per model/input are selected by observed held-out error, then symmetric shape
distance >= 0.025. The thresholds are in normalized fragment-diameter units.

`priors/CASE.json` lists selected job IDs, shortfalls, and filtering reasons.
`priors/CASE/MODEL__INPUT.npz` contains `normalized_1..3` and
`original_anchor_frame_1..3` point clouds. The latter restores input scale and anchor
translation; it does not rotate the object into a GT world frame. Fewer than three
valid distinct hypotheses is reported honestly.

## Assembly and interpretation

```bash
RUN_GROUP=/path/to/existing/group LIMIT=3 STAGES="E4" bash run_imagination_study.sh
```

E4 evaluates each selected hypothesis separately, alongside no-prior, wrong-shape,
true-shape and true-image controls. Missing selected priors are reported as
not-applicable rather than replaced. The pipeline still uses its existing soft
surface-distance prior; it is not a solid inside/outside containment test.

Evaluate completion quality and rejection rates before ranking Chamfer. Compare
case-balanced results and raw controls; never select per-instance seeds using GT.
The oracle measures the entire render/reconstruction/input-only-alignment bridge,
not a universal InstantMesh error floor. Silhouette scores are sampled point-splat
proxies; metric distances are normalized units, not centimetres.

`RUN_GROUP`, `DATASET`, `BASE_CONFIG`, `IMAGES_PYTHON`, `INSTANTMESH_PYTHON`, `GA_ROOT`,
`PROFILES`, `STAGES`, and `LIMIT` can be overridden through the environment.
Use `RETRY_FAILED=1` only after fixing an environment/API failure. Resume with the
same code, config, dataset and case limit; otherwise use a new group. Existing runs
are never deleted.

Return the executed notebook, `analysis/metrics.csv`, `analysis/metrics.json`, and
profile logs. Preserve each run's `study.json`, `experiment.json`, result/backend/
alignment records, and selected point clouds for follow-up diagnosis.

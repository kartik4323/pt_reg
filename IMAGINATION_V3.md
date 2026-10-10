# Bottle-completion repair: bounded v3 experiment cycle

This is a repaired experiment pipeline, **not yet a validated final model**. Real SD15,
SDXL, Qwen and InstantMesh inference must run on the VM. Local smoke substitutes are
branded, cannot be promoted, and unchanged fragments do not count as completions.

## What changed

- E0: deterministic public-input surfels (up to 8,192 points per fragment), bounded
  tangent support, dense grey shading, original-point provenance, 35% input-derived
  framing, correct camera-up rotations, separate support/exterior/edit masks. Assembly
  sampling stays independent. Poisson is diagnostic and fails explicitly.
- E1: short bottle completion/instruction-edit prompts, configurable negatives,
  SD15 512 versus SDXL/Qwen 1024, locked CPU `isnet-general-use` segmentation. Raw,
  alpha, RGBA and white-composited images are retained. No brightness stripping in v3.
  Matte gates enforce 1–85% occupancy, 95% protected coverage, <=2% border contact,
  and >=95% dominant foreground. Colour retention is not geometric preservation.
- E2: explicitly prepared RGBA at 85% foreground fill, independent reconstruction
  preprocessing transform, pinned upstream/checkpoint/custom-pipeline provenance,
  continuous bounded similarity fitting, independent observed fit/holdout points,
  PCA dimensions, thickness and component diagnostics. Largest mesh component must
  have >=90% of surface area in v3. The old slab ratio threshold remains 6.0; no
  promotion threshold was raised to admit the repeated oracle ratio 6.2.
- Workers: cooperative per-GPU locks, immediately checked free memory (24 GiB for
  E2), 30-second checks for up to 30 minutes, host memory and competing process
  diagnostics, failed-attempt archives, explicit OOM/capability/resource/budget failures,
  peak memory even on failed workers, persistent budget accounting and deadlines.
- E4: existing per-prior comparisons plus `B2__deploy`, which compares all selected
  hypotheses with geometry-only assembly using the same input-only objective. Open
  surfaces use distance/penetration proxies; enclosure is measured only for watertight
  meshes. Candidate-bank ceiling and true-geometry registration are evaluator-only.
- Evaluation: missing-surface distance/precision/recall/F-score; rank-1 input-selected
  hypothesis, never best-GT selection; failure-inclusive source-balanced coverage,
  source-bootstrap assembly statistics, blinded review, freeze/test exposure guards,
  optional full-input noise/dropout stress, portable ZIP collection.

Depth background values represent **unobserved**, not known empty space. ControlNet
is soft conditioning, not a hard prohibition against growing geometry.

## Dataset requirement (check before a large run)

The old three variants of one source are useful preliminary diagnostics only. Full
validation requires **10 independent development sources and 20 untouched test
sources**, with no source or identical geometry crossing splits. Use a source-disjoint
`schema_version=1` public dataset. `evaluator_only/index.json` contains separate complete
geometry and pose references; these files never enter public generation/selection.

Choose/reserve the test sources before screening and never inspect their results to
tune the method. Software can enforce this cycle's exposure guard, not erase prior
human inspection or certify the provenance of a dataset. Do not relabel inspected
variants as new independent test sources.

```bash
cd /home/kpandey/satellite/pt_reg
git pull --ff-only
export GA_ROOT=/home/kpandey/generative_assembly
# Change this if the historical file lacks the required independent sources.
export DATASET="$GA_ROOT/bottles498-inputs-v1/dataset_2parts.json"
export BASE_CONFIG="$GA_ROOT/configs/sd15-v100-locked.json"
"$GA_ROOT/envs/images/bin/python" -c 'from generative_assembly.data import inventory; import os; d=inventory(os.environ["DATASET"]); print({s:len({c["source_id"] for c in d["cases"] if c["split"]==s}) for s in ("dev","test")})'
```

For an existing public import specification, create a **new** dataset using
`python -m generative_assembly import-spec --input /absolute/spec.json --out /new/dataset`.
Its source-level splits must be reserved explicitly; importing a previously
source-leaking prepared manifest will fail rather than conceal leakage.

## VM setup and phase commands

Use one visible 32 GB GPU. Never kill somebody else's process. Ensure Python 3.11,
adequate disk, and the existing clean, working InstantMesh checkout/environment from
the locked base configuration. The setup script creates isolated image/matte envs;
it does not alter the historical environments or install/patch InstantMesh.

```bash
export CUDA_VISIBLE_DEVICES=0  # choose your assigned physical device
bash setup_imagination_v3.sh
export RUN_GROUP="$GA_ROOT/runs/imagination-v3-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_GROUP/environment-locks"
cp "$GA_ROOT/environment-locks/images-v3.txt" "$GA_ROOT/environment-locks/matte-v3.txt" "$RUN_GROUP/environment-locks/"
"$GA_ROOT/envs/instantmesh/bin/python" -m pip freeze > "$RUN_GROUP/environment-locks/instantmesh.txt"
PHASE=pilot bash run_imagination_study.sh
```

Pilot uses one development source, F/A, seed 11 and raw/oracle controls for each
model. Oracle controls include both the original input-derived camera and a separate,
evaluator-only full-object framing control, so a clipped oracle image is not mistaken
for an intrinsic reconstruction error floor. Complete-geometry registration is also
strictly evaluator-only and never enters the public prior bank.
Qwen uses its Plus editing interface and an explicitly recorded sequential
CPU-offload configuration. BF16 support and >=64 GiB available host RAM are required;
unsupported hardware is recorded and skipped, never replaced with another model or
dtype. A V100 will fail this BF16 capability check. Offload speed/memory still need
pilot measurement. Downloaded checkpoints are locked by revision and hash.

InstantMesh's Zero123++ adapter locks the authors' `sudo-ai/zero123plus-pipeline`
repository separately from the weights, downloads `pipeline.py` at that commit,
and passes its local file path to Diffusers. The generic community mirror is a
dataset, not a model repository, and does not contain this pipeline. A repository
commit is not a Diffusers community-version folder. An older lock missing the
authors' pipeline must be regenerated in a new run identity, not edited in place.

If the initial pilot stopped in `lock-models` with the community-mirror 404,
update the code and create a new v3 `RUN_GROUP`; retain the failed directory for
diagnostics. Setup environments and matte weights can be reused. That initial
locking failure occurred before inference and did not consume GPU experiment time.

After reviewing the pilot images, masks, resource failures and runtime:

```bash
PHASE=representation bash run_imagination_study.sh
# Inspect the notebook BEFORE choosing the representation. The default is frame35.
# Other choices: legacy_splat, dense, frame50, exterior35, depth35.
export REPRESENTATION=frame35
PHASE=models bash run_imagination_study.sh
export FINALISTS="models_sd15 models_sdxl"  # at most two; choose from development results
PHASE=reconstruction bash run_imagination_study.sh
PHASE=validate bash run_imagination_study.sh
```

Representation comparisons change one factor: splat→surfel→35%→50%; exterior35 and
depth35 each compare against frame35. All use matched seeds 11/23/37/51 and F/A on
three independent development sources. Models use the chosen representation and
matched sources/seeds, with resolution differences explicitly recorded. Reconstruction
adds fixed seed-11 raw-RGB/RGBA pairs; those diagnostic arms never enter prior selection.
Validation uses 10 independent development sources. Primary completion evaluation is
F because a GT-assembled union of every fragment may have no missing region; A remains
an input-only assembly/rendering comparison.

Before freezing, open each validation profile's **blinded_review.html** before looking
at evaluator scores/GT overlays, then fill its `review.json` (bottle identity, full outline, same view,
surface quality). Null means pending. Do not use GT overlays to approve a hypothesis.

```bash
PHASE=freeze bash run_imagination_study.sh
PHASE=test bash run_imagination_study.sh
# Optional, only if the persistent budget admits the phase:
PHASE=stress bash run_imagination_study.sh
# Exports an approved bundle ONLY if every predefined test gate passes:
PHASE=promote bash run_imagination_study.sh
```

Freeze chooses by development assembly gain, then development missing-distance gain,
with deterministic name tie-breaking. Freeze alone is NOT validation/promotion: an
inadequate development finalist may be frozen as a measured candidate, but cannot be
declared working. Test uses exactly the frozen configuration on 20 untouched sources.
Once test starts, development/refreeze are blocked. Test may only resume unchanged.
Stress perturbs the full public point clouds (noise .005 anchor-diameter units and
25% dropout) and re-runs rendering/generation/reconstruction/assembly in separate
frozen diagnostic runs; stress outcomes never select the winner.

## Resume, failures and the 24-hour cap

Keep the **same exported RUN_GROUP** across all phases/resumes. Do not regenerate its
timestamp, upgrade code/envs, change configs, edit model locks, or delete the budget
ledger in the middle of a cycle. Changed code/config/dataset requires a new run identity;
it does not grant another experiment budget or make exposed test sources untouched.

```bash
RETRY_FAILED=1 PHASE=reconstruction bash run_imagination_study.sh
# Example: retry only E2 after fixing resource availability, retaining E0/E1:
RETRY_FAILED=1 STAGES=E2 PHASE=validate bash run_imagination_study.sh
RETRY_FAILED=1 PHASE=test bash run_imagination_study.sh
```

`PROFILES` optionally filters the named profiles of pilot/representation/models/validate;
`STAGES` limits explicitly requested work. Without PHASE, the historical v2 runner
retains RUN_GROUP/PROFILES/STAGES/RETRY_FAILED behavior. Never reuse a v2 result root
for changed v3 configs. Missing E0 is a dependency error, not something E2 can repair.

`budget.json` reserves four hours for screening and twenty for validation/test/stress.
It conservatively charges worker wall time while admitted to this runner's single GPU
(including loading), not CUDA kernel time. Matte CPU time and waiting for another
owner's GPU are excluded. Pilot timings drive phase estimates; each actual worker has
a remaining-budget deadline. A phase that cannot fit is refused at its boundary; an
unexpected overrun can stop mid-phase, preserving resumable jobs. Insufficient dataset
or budget yields **preliminary results**, not relaxed gates. Hard-killed parents may
leave reservation/lock records: inspect their PID/host and reconcile only confirmed
dead workers; never clear a live reservation or another user's GPU process.

Failure files are under `$RUN_GROUP/<profile>/jobs/<job_id>/worker.log` and `error.txt`.
Retries move the prior attempt into that job's `attempts/<timestamp>/`. Review
`gpu_preflight.json`, `backend.json`, `alignment.json`, `phase_admission.json` and
`<profile>-<phase>.log`; insufficient GPU memory is `resource_unavailable`, not a
model-quality failure.

## Notebook and one results ZIP

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$GA_ROOT/envs/images-v3/bin/python" -m generative_assembly.imagination_report --root "$RUN_GROUP" --dataset "$DATASET"
"$GA_ROOT/envs/images-v3/bin/python" -m jupyter nbconvert --to notebook --execute \
  imagination_state_analysis.ipynb --output imagination_state_analysis_executed.ipynb \
  --output-dir "$RUN_GROUP" --ExecutePreprocessor.kernel_name=imagination-v3 \
  --ExecutePreprocessor.timeout=1800
"$GA_ROOT/envs/images-v3/bin/python" -m generative_assembly.study_cycle collect \
  --root "$RUN_GROUP" --out "${RUN_GROUP}-results.zip"
```

The notebook embeds image bytes and shows the stage funnel, renderer provenance,
raw/alpha/white gallery, missing versus observed metrics, rejected template diagnostics,
three-prior coverage, assembly comparisons, resources and budget. `PROFILE` can be
set via `export PROFILE=validation_sd15` (or the selected finalist) for the optional 3D plot. Portable metrics remain available
without a relocated dataset; evaluator recomputation requires relocated references.
The ZIP includes metrics, the saved executed notebook, images, selected mesh/point
cloud priors, rejected meshes, locks/configs/reviews and failure diagnostics. It
excludes checkpoint weights and does not include every accepted intermediate mesh.

## Acceptance and promotion

Required test gates: >=80% of eligible cases with three genuinely distinct priors
(and blinded approval); >=10% source-balanced missing-distance improvement over
matched raw reconstruction with no missing F-score reduction; >=5 percentage-point
assembly-success gain, positive lower source-bootstrap CI, <=5% damage to previously
successful geometry-only cases; public, non-smoke hypotheses; sufficient independent
sources/references. Failed/rejected/abstained cases remain in coverage denominators.
At least 80% of cases must have valid matched shape comparisons; unpaired failures
are reported explicitly, not assigned invented distances. Completion scores for the
remaining failures are unavailable, not zero-error successes.

If gates fail, retain `promotion.json`, the best measured development candidate and
the isolated bottleneck. The approved export is a research-validation bundle, not a
general production certification. It contains no evaluator references as inference
inputs and does not package downloaded model weights.

## Local verification

```bash
python -m unittest discover -s generative_assembly/tests -v
python -m generative_assembly.cpu_diagnostics --out outputs_imagination_v3_local
bash -n run_imagination_study.sh
bash -n setup_imagination_v3.sh
```

CPU fixtures test rendering, masks, transforms, matte gates, native-resolution mocked
adapters, RGBA/preprocessing/hash contracts, GPU admission, budgets, failure archives,
alignment/holdout separation, duplicate rejection, original-coordinate exports,
ground-truth-free safe abstention, source balancing, promotion, stress inputs and
notebook execution. Real historical-gallery segmentation was not run locally because
the pinned ONNX checkpoint/rembg environment is unavailable; no substitute was used.

Implementation references: [rembg models](https://github.com/danielgatis/rembg#models),
[SDXL resolution](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl),
[Qwen editing interface](https://huggingface.co/Qwen/Qwen-Image-Edit-2509),
[InstantMesh preprocessing](https://github.com/TencentARC/InstantMesh/blob/main/run.py).
The adapter verifies the VM's locked `run.py` contract rather than assuming the current
upstream checkout matches it.

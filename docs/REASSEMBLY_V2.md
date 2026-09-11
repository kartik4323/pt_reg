# Reassembly v2 runbook

The experiment predicts a coarse shape scaffold from complete 2–3 fragment sets and uses it to guide rigid assembly. XYZ arrays are the only inference inputs. Meshes, fracture labels, corresponding interfaces, intact targets and poses are training/evaluation supervision only.

## Installation and locations

Run commands from the repository root on the A5000 server, in a fresh environment with an appropriate CUDA-enabled PyTorch installation. `requirements.txt` adds Trimesh, Manifold3D, Rtree and SciPy; the encoder and numerical solver need no compiled PointNet++ extension. No pretrained model is downloaded.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_reassembly*.py' -v
```

The default managed directory is `~/reassembly_v2`. Resolve it and check that its filesystem has sufficient space. The CLI checks resolved locations, caps managed acquisitions/preparation/runs at 40 GiB, and preserves 50 GiB free. `/data` is never a default. Keep all experiment outputs in this managed directory so the cap includes earlier runs. `--managed-root` selects another filesystem; moving the directory also requires explicitly updating manifest/checkpoint paths supplied to commands.

This is a bounded pilot workflow. Each learned stage is limited to 2,000 updates, checkpoint retention is best/latest, and each evaluation retains at most eight qualitative arrays. It never launches full training automatically.

## Acquire and prepare

```bash
python -m reassembly acquire --config configs/reassembly_v2.yaml --dry-run
python -m reassembly acquire --config configs/reassembly_v2.yaml
python -m reassembly prepare --config configs/reassembly_v2.yaml \
  --source ~/reassembly_v2/sources/02876657.zip
```

The metadata check verifies access and archive size before streaming only the bottle archive. If access is unavailable, obtain ShapeNet authorization through the dataset provider and authenticate locally, or pass an already obtained category archive to `prepare`. Never put tokens in configs or reports. No complete dataset download or archive extraction is required.

Preparation examines up to 100 candidate meshes. If fewer than 30 source objects yield accepted patterns, it writes rejection/yield reports and exits with code 2. Inspect `prepared/preparation_report.json` and reassess before learning on held-out objects. `--source` also accepts a bottle mesh directory. See [data details](REASSEMBLY_DATA.md) for exact acceptance and provenance behavior.

Source objects are assigned to 80/10/10 partitions before cuts. The test objects also receive a held-out cut family. Each pattern contains every piece from its fracture. Planar/symmetric controls remain labeled controls; they can be ambiguous. Hard patterns bias cuts toward smaller contacts with stronger curvature and noise; contact fraction is recorded, because geometry determines the actual footprint.

## Preflight on the actual GPU

```bash
python -m reassembly preflight --config configs/reassembly_v2.yaml \
  --manifest ~/reassembly_v2/prepared/manifest.json --device cuda
```

Preflight verifies source/pattern hashes, runs deterministic geometry/solver checks, then performs actual forward, losses, backward and optimizer steps for all three learned stages. It uses prepared three-piece patterns to exercise the largest input case and includes chunked 32³ field inference. Passing CUDA preflight requires peak reserved memory strictly below 20 GiB. If batch 2 exceeds that limit, it retries batch 1 with increased accumulation and unchanged geometry resolution. A second failure stops the pilot.

Always use `preflight/config.resolved.json` afterward, since it records the measured batch/accumulation choice. Training checks the dataset fingerprint, allocation configuration, correctness results and GPU model. Reprofile after changing geometry resolution, architecture, loss settings or device. The `--device cpu` mode exercises correctness but cannot satisfy a CUDA training gate. Without `--manifest`, preflight uses an explicitly labeled allocation fixture; that report cannot authorize dataset training.

## Fixed 16-pattern check

The following Bash variables are conveniences for explicit paths:

```bash
ROOT="$HOME/reassembly_v2"
CFG="$ROOT/preflight/config.resolved.json"
DATA="$ROOT/prepared/manifest.json"
PF="$ROOT/preflight/preflight.json"

python -m reassembly train --config "$CFG" --manifest "$DATA" --device cuda \
  --preflight-report "$PF" --overfit --stage 1 --run-dir "$ROOT/overfit/s1"
python -m reassembly train --config "$CFG" --manifest "$DATA" --device cuda \
  --preflight-report "$PF" --overfit --stage 2 --run-dir "$ROOT/overfit/s2" \
  --initialize-from "$ROOT/overfit/s1/best.pt"
python -m reassembly train --config "$CFG" --manifest "$DATA" --device cuda \
  --preflight-report "$PF" --overfit --stage 3 --run-dir "$ROOT/overfit/s3" \
  --initialize-from "$ROOT/overfit/s2/best.pt"
python -m reassembly evaluate --config "$CFG" --manifest "$DATA" --device cuda \
  --overfit --checkpoint "$ROOT/overfit/s3/best.pt" --output "$ROOT/overfit/eval"
```

Fixed mode uses the same 16 train patterns, points, poses and noise at every update and evaluation. The acceptance flag requires all 16 assemblies to pass the geometric threshold with `ok` solver status. A low reconstruction loss alone does not pass it. Failure is a research result: inspect segmentation/matching diagnostics, SDF error and contact residuals; do not substitute a fabricated passing report. Overfit checkpoints are explicitly rejected when initializing held-out pilot stages.

## Fresh held-out pilots and ablations

After `overfit/eval/evaluation.json` has `fixed_fit_passed: true`, run both conditions from fresh stage 1 with identical budgets and seed. Stage 2 retains the geometry objectives during encoder adaptation. Stage 3 freezes encoder/scaffold and learns only the contact matcher. The contact-only condition shares the scaffold-training budget for a controlled comparison but disables all scaffold use in matching/solving.

```bash
for CONDITION in predicted contact_only; do
  RUN="$ROOT/pilot/$CONDITION"
  python -m reassembly train --config "$CFG" --manifest "$DATA" --device cuda \
    --preflight-report "$PF" --overfit-report "$ROOT/overfit/eval/evaluation.json" \
    --condition "$CONDITION" --stage 1 --run-dir "$RUN/s1"
  python -m reassembly train --config "$CFG" --manifest "$DATA" --device cuda \
    --preflight-report "$PF" --condition "$CONDITION" --stage 2 --run-dir "$RUN/s2" \
    --initialize-from "$RUN/s1/best.pt"
  python -m reassembly train --config "$CFG" --manifest "$DATA" --device cuda \
    --preflight-report "$PF" --condition "$CONDITION" --stage 3 --run-dir "$RUN/s3" \
    --initialize-from "$RUN/s2/best.pt"
  python -m reassembly evaluate --config "$CFG" --manifest "$DATA" --device cuda \
    --condition "$CONDITION" --checkpoint "$RUN/s3/best.pt" \
    --output "$ROOT/eval/$CONDITION"
done

for CONDITION in gt perturbed; do
  python -m reassembly evaluate --config "$CFG" --manifest "$DATA" --device cuda \
    --condition "$CONDITION" --checkpoint "$ROOT/pilot/predicted/s3/best.pt" \
    --output "$ROOT/eval/$CONDITION"
done
```

The GT condition queries the preserved intact source mesh in the observed reference frame. It does not use GT pose correspondences for assembly. Both GT and perturbed fields replace matcher conditioning and numerical scaffold guidance. Perturbation spatially shifts the field and offsets distance while keeping its uncertainty, deliberately testing misleading guidance. Contact-only evaluation requires a contact-only trained checkpoint.

Repeat evaluations with `--split cut_holdout` and distinct output directories for cut-family generalization. `--split val` and `--limit N` support bounded diagnostics; a subset result should be reported as such. Validation within training runs periodically and selects best checkpoints; evaluation never retrains.

Resume only an explicitly chosen `latest.pt` or `best.pt` in the same run directory, with the same stage/purpose/condition and remaining update budget:

```bash
python -m reassembly train --config "$CFG" --manifest "$DATA" --device cuda \
  --preflight-report "$PF" --stage 3 --run-dir "$ROOT/pilot/predicted/s3" \
  --resume "$ROOT/pilot/predicted/s3/latest.pt"
```

Checkpoints store v2 architecture/configuration, dataset fingerprint, optimizer/scaler/RNG states, run identity and all preceding stage budgets. Legacy and external checkpoints are rejected. There is no checkpoint search or implicit resume. Use fresh directories for new runs/evaluations/inferences.

## Inference and transforms

```bash
python -m reassembly infer --config "$CFG" --device cuda \
  --checkpoint "$ROOT/pilot/predicted/s3/best.pt" \
  --fragment fragment_a.npy --fragment fragment_b.npy --fragment fragment_c.npy \
  --output "$ROOT/inference/example_001" --save-scaffold
```

Supply two or three finite, variable-length `N×3` XYZ arrays in consistent units. Each needs at least three points and nonzero extent; very small or degenerate observations may explicitly fail. Inference samples model inputs while retaining every original point for export. No mesh, normals, labels or whole target is accepted.

The reference is the fragment with greatest RMS radius, with index tie-breaking. Every fragment is centered separately; the shared scale is the sum of bounding radii. For normalized solver transforms `(R_i, t_i)`, export composes:

```text
x_aligned = x_original @ R_i.T + t_export_i
t_export_i = reference_centroid + shared_scale * t_i - R_i @ fragment_centroid_i
```

`result.json` contains proper rotations, translations, 4×4 column-homogeneous matrices, reference index, confidence, status and diagnostics. The reference transform is exactly identity. `assembly.npz` contains matrices and aligned original-resolution fragment arrays. Optional `scaffold.npz`/`scaffold.png` contain the field and its slices; `scaffold_frame` in the JSON defines conversion back to output units. The final object is exclusively rigidly transformed input points.

If no valid connected assembly exists, transforms are null and no assembly file is written. Weak valid support is marked `low_confidence`. The CLI exits with code 2 for either case; inspect the status before consuming poses. Planar non-collinear correspondences remain valid; collinear/insufficient support is rejected.

## Report and decision

```bash
python -m reassembly report --config "$CFG" \
  --input "$ROOT/prepared/preparation_report.json" --input "$PF" \
  --input "$ROOT/overfit/eval/evaluation.json" \
  --input "$ROOT/pilot/predicted/s1/training_report.json" \
  --input "$ROOT/pilot/predicted/s2/training_report.json" \
  --input "$ROOT/pilot/predicted/s3/training_report.json" \
  --input "$ROOT/eval/contact_only/evaluation.json" \
  --input "$ROOT/eval/predicted/evaluation.json" \
  --input "$ROOT/eval/gt/evaluation.json" --input "$ROOT/eval/perturbed/evaluation.json"
```

Reports include preparation yield/storage, per-stage throughput and GPU memory, bounded linear cost projections, per-part and whole-assembly Chamfer, geometric success, failure/low-confidence rate, contact residuals, pose diagnostics and difficulty/piece-count groups. Chamfer is symmetric mean **Euclidean** nearest-neighbor distance, in the shared normalized scale, averaged over both directions. Distances exclude failed solves; failure rate always counts them. Rotation error is diagnostic because symmetry can make a unique rotation unidentifiable.

Acceptance binds reports to the same dataset/configuration and all three stage budgets, and requires the GT/perturbed controls. The current conservative benefit rule requires higher geometric success without worse Chamfer or failure rate. Reconstruction, matching and pre/post-refinement diagnostics remain separate. Improved SDF reconstruction alone is insufficient. No full-training command runs as a side effect of reporting.

## Architecture and literature

The shared encoder uses three PointNet++-style local aggregation levels with 256/128/64 groups, 128 channels and 1,024 XYZ points per fragment. The 64 coarse tokens condition the scaffold. In architecture `coarse-scaffold-reassembly-v2.1-local-contacts`, the matcher separately gathers 256 full-resolution features: 128 globally sampled points and up to 128 additional spatially sampled predicted fracture points, with a global fallback. Ground-truth labels never select inference points. Geometry objectives are surface segmentation, multi-positive local mating matches and pointwise transformed-view consistency. Matching includes both matched/dustbin mass supervision (radius 0.05) and a distance-weighted conditional distribution loss (sigma 0.01); membership in the broad radius alone is insufficient for accurate assembly. No whole-fragment compatibility attraction or fragment dropout is used. Augmentation teaches rotation robustness; no exact SE(3) invariance claim is made.

Contact confidence uses retained matched-versus-dustbin probability at supported points, so the exterior area cannot dilute a small valid interface. Individual correspondence probabilities still weight Kabsch and refinement. Assembly confidence is the best spanning tree's weakest contact confidence, discounted by residual error; it is a diagnostic score, not a calibrated probability of assembly success. The 0.01 geometric threshold and requirement that all 16 overfit examples have `status: ok` remain unchanged. Pre-v2.1 checkpoints are rejected; reuse prepared data but repeat preflight and train fresh stages. See [the overfit failure analysis](REASSEMBLY_OVERFIT_REVIEW.md).

The implicit TSDF receives grouped global context and query-to-reference-token geometry, with two attention blocks and 2,048 training queries. It never treats independently posed fragment coordinates as aligned. Unweighted distance supervision remains present alongside bounded uncertainty calibration. The field is evaluated at 32³ in chunks.

Matching preserves dustbin probability and confidence. The solver creates at most four distinct valid poses per pair from local weighted Kabsch fits, enumerates all possible spanning trees for three pieces, and refines the best four candidates for up to five damped Gauss–Newton iterations. Contact and exterior-field residuals are normalized separately, with scaffold coefficient at most 0.25; uncertainty and confidences are detached. There is no proximity-based collision penalty or inference-time pose optimization through the network.

[Neural Shape Mating](https://neural-shape-mating.github.io/) motivates complementary cuts and shape reconstruction; [Jigsaw](https://jiaxin-lu.github.io/Jigsaw/) motivates segmentation, contact matching and robust alignment. [Jigsaw++](https://arxiv.org/html/2410.11816v2) motivates complete-shape guidance, but its assembly experiment in Section 6.2 obtains original-surface correspondences through GT geometry. This implementation must estimate contacts from input XYZ and predicted features. These are design inspirations, not reproductions or imported checkpoints.

Breaking Bad remains a later transfer experiment. The read-only inventory helper selects only whole fracture directories already containing 2–3 pieces. It never truncates a larger set; intact-target/provenance validation for a learning adapter remains separate.

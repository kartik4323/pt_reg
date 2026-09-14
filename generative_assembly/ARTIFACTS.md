# Artifact contract and reuse

Every study has a frozen identity: resolved configuration, public dataset SHA-256, and SHA-256 of this package's Python source. Changing any of these requires a new run root. Successful jobs are reused only after checking each output hash. A job ID includes stage, case/source, split, arm, oracle flag, input hash, extra factors and parent IDs. Failed jobs remain failures until `--retry-failed` is explicitly used; their exceptions and worker logs are retained.

```text
study.json                         configuration, code/data identity and host
doctor.json                        installed capabilities and available hardware
frozen_method.json                 pre-test model/selection/checkpoint choices
jobs/<job_id>/result.json          condition, parents, status, duration, artifact hashes
jobs/<job_id>/request.json         exact model request and checkpoint settings
jobs/<job_id>/worker.log           complete child-model stdout/stderr
jobs/<job_id>/error.txt            exception when a job fails
jobs/<E0-id>/observed.npz           normalized observed points, sample IDs, centers, scale
jobs/<E0-id>/candidates.npz         immutable common pose bank and geometry scores
jobs/<E0-id>/camera.json            shared F/A camera, extent and visibility selection
jobs/<E0-id>/{F,A}/                image.png, control.png, mask.png, render.npz
jobs/<E1-id>/image.png              every image hypothesis, including failed-quality samples
jobs/<E2-id>/mesh.obj               actual reconstructed surface
jobs/<E2-id>/shape.npz              raw surface samples and input-aligned template
jobs/<E2-id>/alignment.json         similarity transform and held-out observed residual
jobs/<pose-job>/poses.npz           normalized and original-coordinate matrices
jobs/<pose-job>/prediction.json    template ID, selected candidate, confidence diagnostics
jobs/<E6_TRAIN-id>/last.pt          model, optimizer, step, RNG, pool and lineage
jobs/<E6_TRAIN-id>/learning_curve.jsonl
pseudo_labels/train.json           teacher poses, acceptance, source IDs and oracle=false
results.jsonl                      machine-readable complete index, including failures
status.json                        completed/failed counts by stage
evaluation/<split>/metrics.jsonl   per-case evidence; oracle flags and failure denominators
evaluation/<split>/summary.json    source-macro metrics and paired confidence intervals
evaluation/train/pseudo_label_quality.json  evaluator-only teacher accuracy vs acceptance
evaluation/<split>/REPORT.md       readable summary
evaluation/<split>/image_review.json  blank human-review fields; never fabricated labels
```

`poses.npz['original'][i]` is a proper 4×4 matrix mapping original input fragment i to the original reference fragment's coordinate frame. For row-array points, `assembled = points @ T[:3,:3].T + T[:3,3]`. The reference fragment remains identity. `normalized` uses independently centered fragments and one shared input-only scale, with metadata in prediction.json. Fragment identity is preserved. Generated surfaces never replace real fragments in this output.

`template_job` links a pose/pseudo-label to the exact generated image → reconstructed mesh → aligned surface lineage. `backend.json` captures the model revision, relevant library version and measured worker GPU peak where available. Image/3D model settings and random seeds are in request.json. Never describe heuristic confidence as calibrated probability.

The pipeline bundle includes non-oracle complete jobs, public input data, pseudo-labels, trained checkpoints, source code and a recipe. The research bundle additionally includes oracle jobs, failed-job logs and evaluator references. Both include evaluation summaries as evidence; inference must not consume those summaries as per-instance features. The pipeline recipe keeps `production_approved=false`: generating an export is not scientific approval.

`summary.json` contains targeted comparisons for completion versus raw-image reconstruction, the wrong/true-shape controls, compute control, gated versus always-on priors, and filtered students versus geometry/all/random-count learners and teacher. Robustness comparisons are separated by corruption factor and level. Bootstrap units are original sources; fracture patterns and perturbation repeats are not independent objects. E1/E2 scalar means have their own scored denominators, while generation failures remain listed separately. Re-evaluation preserves completed manual review fields and rejects changed reference-index hashes.

Pretrained weights are referenced by immutable revision, rather than duplicated into every bundle. Student weights are copied. Existing absolute paths inside historical requests document the original run; the recipe names portable job-relative locations and runtime bindings to update on a new host. Check SHA256SUMS.json after transfer. A JSON reader can recover all proposed/final poses without importing PyTorch; only student checkpoints require it.

Preserve the entire run root and original prepared dataset while experiments are active. Do not delete unsuccessful image samples: they are needed to diagnose selection, rejection, and final pipeline failure handling. For long-term archival, export a research bundle and compress that directory with the platform's archive tool.

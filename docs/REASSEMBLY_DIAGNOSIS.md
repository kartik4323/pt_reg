# Read-only diagnosis of the recorded bottles498 pilot

Run this from the VM repository, with the existing `ptreg-v2` environment active. No downloads, training, checkpoint migration, or optimizer updates are performed. The diagnostic code lives entirely in `diagnostics/reassembly_v2`; production modules are imported and their file hashes are checked afterward.

```bash
git pull --ff-only && python -m diagnostics.reassembly_v2 \
  --managed-root /home/kpandey/reassembly_v2 \
  --output "/home/kpandey/reassembly_v2/diagnostics/$(date -u +%Y%m%dT%H%M%SZ)" \
  --device cuda:0 --hours 4
```

The managed root must be the existing VM data root containing `bottles498`. This diagnosis is bound to fingerprint `b7f2b84a20964276c894300a7ece4d8f97f232903bace0f14131535594462043`. It rejects the different locally prepared dataset. If the root or recorded checkpoint paths differ, return the resulting error for artifact reconciliation; do not copy in unrelated weights or rebuild data.

The output argument is optional: omitting it creates a timestamped directory below `<managed-root>/diagnostics`. The command exits with code **0** for completed coverage and **2** for partial coverage, a deadline, a replay mismatch, missing artifacts, or another error. A partial run is evidence to inspect, not a failed training run to restart. Computation stops at the deadline; final hash verification and packaging may finish afterward.

Return **`diagnostic_bundle.tar.gz`**, whose full path is printed at exit. It contains no checkpoints or mesh datasets. If interrupted during final packaging, the incremental JSON files remain in the output directory.

## Artifact requirements and execution order

The runner requires best/latest checkpoints, `config.resolved.json`, `run.json`, `training_report.json`, and `history.jsonl` for all stages of `overfit`, `pilot/predicted`, and `pilot/contact_only`. It also requires `prepared/manifest.json`, `preflight/preflight.json`, `overfit/eval/evaluation.json`, and the eight reports at `eval/{test,cut_holdout}/{contact_only,predicted,gt,perturbed}/evaluation.json`.

1. Verify prepared-file hashes and source splits; inventory artifact hashes, configurations, checkpoint steps/lineage, GPU and runtime settings. Run existing geometry/solver correctness checks and the diagnostic input/target contracts. Profile actual inference, GT field construction, and copied-model backward calculations.
2. Replay the 16 overfit cases and eight held-out evaluations. Check every stored per-example scalar/status, plus aggregate metrics, with exact counts/statuses and `atol=1e-5, rtol=1e-4` for floating-point values. A mismatch blocks downstream interpretation. Replay uses each report's recorded configuration and verifies its explicitly recorded checkpoint path, step and lineage.
3. Evaluate both final pilot models on all **418 training and 48 validation patterns**, with deterministic samples. Inspect best/latest Stage 1, 2 and 3 representations on a deterministic 12-pattern validation subset.
4. Run the overfit and validation pose/sample controls with seeds **4101, 4102, 4103**, then oracle, matching-conditioning, fixed-refinement and validation-only threshold interventions.
5. Probe predicted/GT fields, known and perturbed GT poses, and individual objective gradients on copied checkpoint models. Compile the report and verify checkpoint/production/input hashes again.

The experiment manifest lists every job and exact pattern identity before the main measurement loop. Subset selection rotates source objects and available difficulty/piece-count groups without consulting model performance. Repeated cuts of one source are reported as repeated cuts, not independent source objects.

## Interpreting the controls

All controls retain the production success threshold, geometry resolution, non-collinearity tests and proper-rotation requirement. The weight/mass sweep is diagnostic only: it does not change saved configurations or count as production acceptance. Shuffled correspondence weights are evaluated alongside every threshold combination.

- **Oracle selection:** use GT fracture membership to choose points; retain learned features and gating. It still uses the production selector's broad-coverage allocation, so it is not a guarantee of full interface coverage.
- **Oracle gating:** retain the original selected indices and descriptors; replace fracture probabilities in both the directional log-priors and final probability products. The combined control changes both selection and gating.
- **Oracle correspondence distributions:** use independently sampled points, matching interface IDs and the production distance-weighted multi-positive target. Keep learned fracture gates, including matchability gating. Run once on original selected points and once on oracle-selected points. Failure can therefore still arise from selection, gating, geometry or solver requirements.
- **Fixed-refinement controls:** obtain up to four starting assemblies using learned matches and contact-only scoring, then replay those *same starts and matches* with no field, the predicted field and the GT field. Store every start and result. The headline diagnostic metric uses the first fixed start, avoiding a changed candidate-ranking policy between these comparisons. If no starts exist, explicitly record that refinement was not exercised.
- **Matcher-conditioning controls:** hold encoded inputs fixed and compare actual directional probability matrices and final weights under predicted, GT, perturbed and disabled conditioning. Hold the solver field disabled in this comparison. The complete matrix values and differences are saved for the bounded validation subset.
- **Pose/field controls:** start at GT poses and at fixed 5-degree/0.02 and 15-degree/0.05 normalized-unit perturbations, keeping the reference fixed. Field-only refinement is evaluated with both predicted and oracle exterior weights. These controls test field direction when initialization is available; they are not learned assembly results.
- **Gradient controls:** evaluate raw and currently weighted objective norms and gradient cosines on deep-copied models, with no optimizer. These measurements describe the current checkpoint, not past optimization dynamics.

The report separates **Observed**, **Supported intervention effects**, and **Unresolved**. Each intervention includes affected pattern IDs, unchanged/worsened cases and limits of interpretation. A recovered pose count by itself is not evidence of geometrically correct assembly. Null values mean undefined or unavailable measurements; zero-valued production failure sentinels are not reused as measurements of the raw correspondence tensors.

## Outputs and limits

| Output | Contents |
|---|---|
| `inventory.json` | Input hashes, checkpoint/configuration lineage, prepared-data verification, code revision and runtime provenance |
| `experiments.json` | Ordered jobs, checkpoints, cohort identities, fixed subset, seeds and interventions |
| `correctness.json`, `adapter_identity.json`, `profile.json` | Implementation checks and measured resource/timing profile |
| `results/*.json` | Incremental per-example/per-pair measurements, filter exits, candidate/refinement traces, intervention metrics and gradients |
| `tensors/*.npz`, `visuals/*` | Bounded subset features/probabilities, field grids, SDF slices and coarse zero-crossing views |
| `summary.json`, `REPORT.md` | Source-level results, first-failure counts, paired effects, hash verification and explicit unrun jobs |
| `worker.log`, `progress.json` | Execution trace and active job |
| `diagnostic_bundle.tar.gz` | Compact package to return for evidence review |

Batch size is **1** and inference remains FP32 as in production evaluation. Field queries remain chunked. A CUDA allocator limit and peak-reserved check enforce **less than 20 GiB**. Managed artifacts remain within **40 GiB**, with **50 GiB free space** preserved. The runner reserves up to **2 GiB** of headroom for diagnostics including the archive, and stops raw output at **900 MiB**. Thus startup can require roughly 52 GiB free before any output exists. The code never defaults to `/data`.

If a deadline ends a run after artifact verification, the same code and unchanged inputs can continue completed diagnostic jobs with `--resume --output <that-exact-directory>`. This is an explicit additional invocation with its own requested deadline, not an automatic extension. Resume rejects changed code/input hashes and previously failed or mismatched jobs. After an adapter fix or artifact correction, use a fresh output directory.

No architecture, objective, solver threshold or training change should be selected from this bundle until its reproducibility checks and controlled comparisons have been reviewed.

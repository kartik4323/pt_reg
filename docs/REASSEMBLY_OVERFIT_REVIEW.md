# Fixed-example assembly failure: September 11, 2026

The supplied `logs/` came from commit `e308fe6`. All three learned stages completed 2,000 optimizer updates; the subsequent quality gate correctly stopped the workflow. The latest evaluation had 5/16 geometric successes, 3/16 missing solutions, and 13/16 low-confidence solutions. Eight returned solutions were geometrically inaccurate and the five geometrically correct solutions were also low-confidence. This is an assembly-quality failure, not another AMP exception or a storage failure.

## Training assessment

| Stage | Latest fixed-example validation | Assessment |
|---|---|---|
| 1: geometry | Total 0.32391; segmentation 0.11103; consistency 0.03667 | Clear learning. Four early AMP overflows recovered; none after update 86. |
| 2: scaffold and geometry | SDF L1 0.0068676; uncertainty MAE 0.0053392; segmentation 0.03588 | Reconstruction and segmentation improved. One early overflow recovered. |
| 3: matching | Matching loss 0.00003212 | The old objective was fitted very well, but this did not imply precise correspondences or successful assembly. |

Stage 2's negative total training loss is expected from its log-scale calibration term; it does not make the unweighted SDF loss negative. Its progress-line `val`/best-checkpoint score is SDF L1, whereas the report also includes the composite validation `loss`. Training and validation here use the same 16 fixed patterns, so none of these numbers demonstrates held-out generalization. The 0.00647 mean whole-object Chamfer excluded failed solves and hid individual-part errors; acceptance correctly also checks every part. The uploaded console logs omit detailed sample arrays, so the exact trained prediction for each failure cannot be replayed from these files alone.

## Confirmed representation and objective problems

1. **Sparse contact support.** The old matcher reused the encoder's 64 coarse context tokens. Small fracture surfaces could have fewer than three supported sampled points. The solver correctly refuses those rigid fits. Lowering the loss cannot recover points that never enter the matcher.
2. **An imprecise optimum.** The old mass loss allowed probability anywhere within 0.05 normalized units on the correct interface. It could approach zero without preferring a 0.002-distance match over a 0.045-distance match. The assembly criterion is 0.01. Independent interface samples require multiple valid matches, but they also require localization within those matches.
3. **Confidence diluted by unrelated surface area.** Selected contact weight was divided by all 64 points, including correctly unmatched exterior points. Even five exact unit-weight correspondences could score only 5/64, below the 0.1 status threshold before any residual discount. Multiplication of two correspondence distributions also confounds the existence of a contact with how its mass is divided among nearby independently sampled points.

## Changes

The hierarchy and XYZ input resolution remain 256→128→64 and 1,024 points. The matcher now has its own bounded 256-point selection from propagated point features, mixing global FPS with FPS on predicted fracture points. It uses no labels at inference. A conditional KL localization term favors a Gaussian distribution over nearby points on the correct interface while preserving the existing matched/dustbin mass objective. All three training stages use the corrected objective; mass and localization are logged separately.

Pose fitting retains the actual bidirectional correspondence weights. Confidence separately uses retained matched-versus-dustbin mass over supported points and the weakest edge of the best spanning tree, with residual discounting. Weak dustbin mass, insufficient support, and collinear fits still fail or remain low-confidence. No threshold was relaxed, no GT information was added to inference, and no identity fallback was introduced.

Evaluation logs now include every sample's reason, confidence, per-part metrics, match support/candidate counts, and refinement diagnostics. `evaluation.json` also includes a mutually exclusive `failure_breakdown`. The architecture identifier changed to `coarse-scaffold-reassembly-v2.1-local-contacts` so previous weights cannot silently resume under different matching semantics. CUDA preflight must be repeated before a fresh three-stage overfit.

## Local evidence and limits

`python -m reassembly.contact_audit` is an explicitly GT-supervised diagnostic, not model inference or an acceptance gate. It evaluates ideal correspondence probabilities on independently sampled prepared surfaces with the current solver and no scaffold. On the local fixed 16 patterns:

| Ideal input to solver | Geometric successes | Missing solutions |
|---|---:|---:|
| Legacy 64-token hierarchy, uniform allowed matches | 5/16 | 3/16 |
| 256 global samples, uniform allowed matches | 14/16 | 0/16 |
| 256 global/contact samples, distance-weighted matches | 16/16 | 0/16 |

The final condition also returned `ok` for all 16, with mean whole Chamfer 0.001049 and mean moving-part rotation error 0.327 degrees. This diagnoses a representation/objective ceiling and shows the revised interface can support accurate poses. It does **not** establish that fresh learned weights will reach this ceiling or that a predicted scaffold will improve results.

The local prepared fingerprint is `9e6a9677a1688b579ffe1ab205fddd0e4fa0815ea48332b9c9bfc6417d29b0d1`; the uploaded run uses `b7f2b84a20964276c894300a7ece4d8f97f232903bace0f14131535594462043`. Both describe 73 sources/546 patterns, but the local audit is not an exact replay of the VM's checkpoints or binary dataset. The trained model and detailed VM artifacts were not available locally. GPU memory and throughput for the revised matcher must be measured on the VM; local CPU correctness is not a CUDA benchmark.

Validation: all 88 reassembly tests passed, including small-interface coverage, permutation consistency, distance-aware supervision, dustbin confidence, weak bridges, and rejection of old architecture checkpoints. Actual full-resolution CPU preflight passed all three stages with batch size 2 and four accumulation steps. A fresh one-update-per-stage real-data smoke run exercised checkpoint handoffs and the complete evaluation/reporting path; its untrained poses correctly failed the gate. All 21 Bash blocks in the server guide passed syntax checks.

A separate full-resolution CPU float16 preflight using actual CPU GradScaler also completed all three optimizer steps with finite gradients, scale 65,536, and no retries. It used test-only autocast/scaler overrides and does not substitute for CUDA preflight on the VM.

The next required evidence is a fresh fixed-example run with the unchanged assembly gate. If it fails, the new per-sample log records will distinguish unsupported contacts, inaccurate poses, low confidence, and refinement effects. Full/held-out training must remain blocked until that gate passes. Scaffold benefit still requires the planned contact-only/predicted/GT/perturbed comparison.

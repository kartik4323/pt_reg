# Pilot performance assessment

The experiment completed correctly, but learned assembly did not generalize. Both pilot reports correctly set `advance_eligible: false`. No longer run should be justified by execution success alone.

## Outcomes

| Condition | Test successes | Unseen-cut successes |
|---|---:|---:|
| Contact only | 0/69 | 0/11 |
| Predicted scaffold | 0/69 | 0/11 |
| GT scaffold control | 0/69 | 0/11 |
| Perturbed scaffold control | 0/69 | 0/11 |

Every failed example returned `no_valid_connected_assembly`. There were no pair pose candidates, no assembly hypotheses, and no refinement steps. These are missing solutions, not slightly inaccurate poses or low-confidence solutions. Null Chamfer/rotation metrics mean no returned poses; they are not zero errors.

The test split contains 27 easy, 22 intermediate, and 20 hard patterns; 49 two-piece and 20 three-piece patterns. All failed. The unseen-cut split contains 11 intermediate three-piece patterns. Both splits use the same 11 held-out source objects, so these are 80 patterns, not 80 independent objects.

## Positive signals

- The fixed 16-example model assembled 16/16 with no low-confidence results: mean whole Chamfer 0.001291 and moving-part rotation error 0.393 degrees. The architecture and solver can fit those fixed samples/poses.
- All six pilot training stages completed 2,000 updates with stable numerical handling. Early AMP scale reductions recovered; neither Stage 3 run needed an overflow retry.
- Resource constraints are comfortable on the recorded **Tesla V100-PCIE-32GB**: preflight reserved 0.404 GiB, maximum training reserved 0.363 GiB, approximately 1.87 hours for six training stages, and approximately 301 MiB managed storage at training completion. These are PyTorch reserved-memory measurements, not total GPU process usage or A5000 benchmarks.
- Scaffold validation SDF L1 improved from 0.04560 at update 100 to a best 0.02652. This is a reconstruction-learning signal, though not yet proof of useful shape completion or pose guidance.
- Reported budgets/configurations are matched. Identical Stage 1/2 metrics across conditions are expected: both run the same objectives with the same initialization seed; the contact-only protocol spends the same Stage 2 training budget and disables scaffold use at Stage 3/inference.

## Where performance fails

### 1. Geometry features are not ready for a frozen encoder

Stage 1 ends with train/validation total losses 4.348/4.838 and segmentation 0.867/0.867. Stage 2 reduces segmentation to 0.694/0.697. Test segmentation averages 0.701. For the implemented balanced binary cross entropy, a constant 50/50 predictor has loss ln(2) = 0.6931. The reported segmentation losses therefore do not demonstrate useful separation of fracture and exterior points. This is a loss comparison, not a measurement of classification accuracy; precision/recall, IoU, and probability histograms are absent.

Stage 1 transformed-view consistency is 0.00186 on validation. Low consistency error alone is not evidence of useful rotational robustness: nearly constant descriptors would also satisfy it. Descriptor collapse is a hypothesis, not a confirmed diagnosis without descriptor variance or similarity measurements.

The pilot also has substantial training loss; calling this simply memorization of the full training set would be wrong. The fixed overfit run used the same 16 point samples and base poses, while the pilot introduces fresh samples and independent full rotations. The missing diagnostic is performance on those same 16 source patterns under new samples/poses, followed by performance on the pilot training patterns.

### 2. The matcher never provides enough accepted support

For predicted-scaffold test evaluation, all 109 pair records report insufficient support. In 85/109, the **entire** correspondence matrix has less than 0.05 total weight, already below the solver's required selected mass. All 33 unseen-cut pair records also fail; 23/33 have total mass below 0.05. Some pairs are legitimately non-contacting, so not every pair should pass, but a connected set of valid contacts is missing for every object.

Weights multiply both directional correspondence probabilities and both predicted fracture probabilities. Weak fracture predictions or diffuse matches can therefore strongly suppress pose-fitting weights. The solver additionally requires at least three selected correspondences, individual weights >=0.001, and selected mass >=0.05. The current zero `selected_correspondences` diagnostic is a failure sentinel; it does not distinguish zero thresholded entries from a few entries or insufficient selected mass. The reports do not contain full weight matrices, dustbin probabilities, maxima, or entropy, so they cannot identify which factor dominates.

Stage 3 freezes encoder/scaffold and changes only the matcher. Validation matching loss moves from 3.48279 to best 3.43543, about 1.4% improvement. Its final validation components are mass 2.78747 and localization 0.65082. Extending Stage 3 alone would leave the weak segmentation/encoder fixed and is not supported by this near-flat curve.

### 3. Scaffold quality and usefulness are separate questions

Predicted SDF L1 is 0.02441 on test and 0.02694 on unseen cuts, versus 0.00768 for the separate fixed-example overfit model. These different data/pose regimes make this a generalization diagnostic, not a controlled model comparison. The SDF metric mixes near-surface and surrounding-space queries; source reconstruction visuals, a simple field baseline, and separate query-region/sign metrics are needed before calling the scaffold geometrically good.

Calibration is mixed: for test queries with predicted uncertainty 0.02–1, mean predicted sigma is 0.02644 versus mean error 0.02616. The lower-uncertainty test bin predicts 0.01609 versus error 0.01860. On unseen cuts those pairs are 0.02433/0.02818 and 0.01749/0.02246, suggesting overconfidence on this split. Good average agreement in one bin does not establish per-query uncertainty quality; test uncertainty MAE is 0.02139.

GT scaffolds do not rescue this run because the matcher still creates no valid candidates and refinement never starts. This does not show that complete-shape priors cannot help. The GT control supplies a field, not GT correspondences or poses, and the solver still samples it onto the coarse grid. Likewise, failure under a perturbed field cannot demonstrate robustness to misleading priors when the unperturbed pipeline already fails completely.

The actual solver's pair-mass diagnostics change by only about 0.093% on average when replacing the predicted field with GT on test (maximum 0.203%). This is limited evidence of weak conditioning effects on total contact mass, not proof that every individual match is unchanged. The evaluation's `predicted_model_matching_loss` is deliberately recomputed with the predicted field for all scaffold controls, so its identical values across GT/predicted/perturbed are not evidence that overrides were ignored.

## Recommended next experiment

1. Diagnose on validation data using the existing pilot checkpoints: fracture precision/recall and probability histograms, descriptor variance and cross-point similarity, correspondence entropy/dustbin mass, raw maxima, and counts/mass before and after each filter. Measure assembly on pilot training patterns too.
2. Separate pose/sample robustness from source generalization: fixed 16 patterns with fixed poses, new rotations, new point samples, then both. A pass on the first case does not establish the others.
3. Run explicit diagnostic interventions: oracle segmentation with learned matches; oracle contacts with the predicted field; and a validation-only support-threshold sweep. Keep oracle results separate from XYZ-only performance. A threshold sweep may diagnose filtering, but accepting arbitrary weak matches is not a solution.
4. Require useful geometry and candidate-recall validation before freezing the encoder for Stage 3. Use the results above to decide between changed geometry supervision, a pose-augmentation curriculum, or a longer/adaptive geometry stage. Do not infer a specific repair from the aggregate losses alone.
5. Once the contact baseline produces valid assemblies, repeat the matched scaffold comparison. Until then, reconstruction improvements cannot establish the central architecture's value.

No training or inference code was changed for this assessment. `assess.py` produces the adjacent machine-readable summary and training curves from the uploaded reports/logs; the summary records hashes of the input reports.

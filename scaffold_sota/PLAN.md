# Independent scaffold-transfer study across assembly models

Date: 2026-09-14. Status: proposed experiment, based on source inspection and existing diagnostics. No new training or model comparisons have been run for this plan.

The user selected a bottle pilot followed by a broader benchmark if useful, a new v3 scaffold when ready with old v2 as a control, and a single NVIDIA RTX A5000 (24,564 MiB). No SOTA assembly checkpoints are currently trained. There is no time limit. The supplied filesystem snapshot reports about 161G available on `/` and 30G on `/data` (99% used). Use the root filesystem for proposed experiment storage after verifying the resolved path's mount; do not allocate this study on `/data`.

## Separation from the three-stage project

This is an independent research experiment named `scaffold_sota`, with its own implementation, configurations, training jobs, checkpoints, run manifests, evaluation and reports. It is not a repair, replacement Stage 3, or continuation of the three-stage pipeline. The objective is to evaluate scaffold transfer into existing SOTA architectures, not to make the original pipeline succeed.

Authoritative code/planning area: `pt_reg/scaffold_sota/`. Proposed external run root: `~/Kartik_23CS30026/scaffold_sota/`, to be used after verifying that its resolved path resides on `/`. No outputs or study-specific configuration changes go into `reassembly/`, `configs/reassembly_*`, `diagnostics/reassembly_*`, the original pipeline's run roots, or existing SOTA reproduction runs. Existing original reports/checkpoints remain historical inputs rather than combined study results.

Allowed reuse is explicit and read-only: source-object manifests and immutable data assets; pinned SOTA source/checkpoint readers; pure geometry, normalization, sampling and metric utilities; and exported Stage 2 fields or frozen field/checkpoint snapshots with their coordinate metadata and hashes. Experiment-owned adapters mediate these dependencies. Never invoke the original repair workflow as this study's launcher, use its acceptance gates as this study's prerequisites, change its training schedule, or overwrite its models/results. Shared utilities that need changed behavior get study-local wrappers or versioned copies here.

The upstream Stage 2 generator is an artifact supplier. A v3 export is the chosen main input when available; v2 exports are controls. Record the supplied artifact and its validity independently of whether the original contact/pose pipeline succeeds. Study-specific field quality tests are separate experiments, not reruns that alter the original diagnostic state. Native SOTA preparation/training can proceed without a v3 export.

Any future out-of-fold prior training, more-fragment support or broader-dataset adaptation is owned by this study, implemented in study-local prior modules with separate configurations and checkpoints. It may reuse a pinned prior architecture, but must not modify or resume the original three-stage project. Freeze these study-local priors during recipient comparisons. References to v2/v3 below identify artifact provenance, not workflow dependencies.

Share GPU/disk capacity only through resource scheduling. A busy GPU delays this experiment; it does not authorize stopping or changing another experiment. Findings and performance tables stay separate even when both projects use the same underlying geometry.

## Research question and recommendation

Does an object-specific scaffold predicted from unassembled fragments improve the rigid poses produced by existing assembly architectures? Does retaining spatial geometry help more than a global shape descriptor, and does predicted uncertainty reduce harm when that geometry is wrong?

Start with **Jigsaw, CCS, and GARF**. They represent correspondence matching, direct part-pose regression, and generative pose prediction. Build and validate each native baseline first. Run inexpensive frozen-output controls on all three; make **learned additional-input conditioning the main experiment**, because it can affect pose generation rather than only repair existing poses. Extend to PMTR, PuzzleFusion++, DiffAssemble, and CMNet after the first comparisons. Treat GPAT as a separate target-quality study on semantic parts.

These are architectural priorities, not predictions of measured gains. In particular, a missing pose candidate cannot be rescued by reranking, and an accurate silhouette does not uniquely determine fragment poses. A bottle can leave axial rotation ambiguous.

The scaffold is computed from the same fragments, so it does not supply a new sensor observation. Its possible value comes from learned shape structure, auxiliary supervision, and how that structure is exposed to the solver. Account for its training data and compute, and test whether explicit geometry matters beyond extra model capacity or encoder features.

## Evidence that changes the starting assumption

The completed field diagnostic concerns `pilot/predicted/s3/best.pt`, an old v2 checkpoint containing the Stage 2 field; it does not measure a trained v3 field. On 12 validation patterns from six sources, continuous predicted-field-only refinement worsened all exact-pose starts and all 5-degree starts. With learned exterior weights, mean whole Chamfer rose from 0.000378 to 0.040153 at exact starts even as the objective decreased. Oracle exterior masks did not remove the harm. Accurate fields and oracle contacts were separate diagnostic controls.

Consequently, preserve v2 as a measured adverse control. Do not call v3 pose-useful until it is measured. A failed direct-gradient test does not exclude learned conditioning or candidate ranking; test those mechanisms independently. The current repair matcher deliberately does not receive scaffold content: v3 uses its field for spatial scoring/refinement after contact candidates exist.

Read-only historical evidence: [field diagnostic analysis](../../artifact_work/field_analysis.md), [field measurements](../../artifact_work/field_metrics.json), [original repair workflow](../docs/REASSEMBLY_REPAIR.md), [original repair model](../reassembly/repair/model.py). These sources motivate controls but are not this study's experiment outputs.

## Architecture audit and per-model interventions

All eight implementations are present in the pinned suite. None except GPAT has a native complete-target input. A new learned branch must be trained. Reuse pinned `upstream` sources read-only; store patches/adapters under `scaffold_sota/` and create disposable source copies in this study's run root. Preserve existing reproduction configurations and runs.

| Model | Inspected architecture | Recommended additional-input experiment | Frozen-output experiment | Priority and feasibility |
|---|---|---|---|---|
| Jigsaw | Fragment point features, within/across-part attention, fracture classifier, affinity/Sinkhorn/Hungarian, RANSAC and global alignment | Residual cross-attention from point features to scaffold tokens after existing attention and before classification/affinity. Tests whether global shape helps fracture selection and correspondence formation. | Export poses and predicted contacts, then common rigid refinement. Reranking requires explicitly exposing/generating a matched candidate bank. | First pilot; medium engineering work. Useful matching representative. |
| CCS | Part encoder, shared-memory transformer, direct rotation/translation head | Add scaffold-token cross-attention around the part correlator. Keep real-fragment output slots and the existing internal memory mechanism. | Common pose refinement. Default geometric config is not a native multi-sample pose generator. | First pilot; clearest part-level learned-input integration. |
| GARF | Fracture-aware point features, superpoint tokens, transformer flow/diffusion pose prediction | Cross-attend superpoint/denoising tokens to a static scaffold bank. Preserve fragment membership, time conditioning, and current pose encoding. | Equal-budget generated candidates and contact-preserving refinement. Guidance during flow integration is a separate later arm. | First pilot; strongest generative representative, but requires backbone pretraining before flow training and actual A5000 smoke tests. |
| PMTR | Pairwise KPConv features, proxy-matching transformer, coarse-to-fine matching, registration, global pose graph | Add scaffold context at superpoint features before coarse matching; optionally extend to fine descriptors in a separate ablation. Every pair retains full-object scaffold identity. | Retain internal pairwise hypotheses and assemble them under the same graph policy, or refine final poses. Pair hypotheses are not already whole-object candidates. | Second wave; medium/high integration burden. |
| PuzzleFusion++ | Fragment VQ-VAE, pose diffusion denoiser, verifier and iterative agglomeration | Add scaffold attention to denoising tokens; later test verifier conditioning separately. Keep the input reference fixed as fragments merge. | Rerank equal-budget samples or refine final output. Freeze verifier/merge rules when testing denoiser changes. | Second wave; multi-stage training and preprocessing. A denoiser-only pilot must be named as that variant, not full PuzzleFusion++. |
| DiffAssemble | Fragment encodings, graph-based pose diffusion | First a pooled scaffold condition; then scaffold tokens/virtual context nodes with outputs masked to real fragments. Match original pose parameterization. | Equal-budget diffusion samples and a separate final-pose wrapper. | Second wave; comparatively simple conditioning idea, but legacy environment, native export and CUDA training need verification. |
| CMNet | SO(3)-equivariant backbone, invariant features, shape/complementary-occupancy matching, weighted rigid fit and sparse graph | Start with an invariant pooled scaffold descriptor conditioning invariant feature channels. Only then test a pose-aware spatial branch. Do not concatenate arbitrary scaffold XYZ into the equivariant backbone. | Final pose-graph refinement. The inspected registration does not expose a ready-made bank of final poses. | Later; strongest invariance and pairwise plumbing constraints. |
| GPAT | Target and part encoders, target segmentation by part, segment-to-part pose fitting | Replace its native target by the predicted scaffold; fine-tune for imperfect targets only in a separately labelled experiment. | Native fitting with clean/predicted/degraded/generic targets, with optional optimizer held constant. | Separate PartNet track; not a no-target fracture baseline. Current bottle Stage 2 does not establish PartNet generalization. |

Candidate benefit by model is a hypothesis: matching models could gain better correspondences or graph consistency; direct pose regressors could gain global layout; diffusion/flow models could gain less ambiguous pose trajectories. Exact fracture geometry remains essential in all three cases.

## Native baseline prerequisites

1. Record each pinned revision, model configuration, initialization, data manifest, and training stages. Train on the same eligible source split. Public backbone initialization, if later used, must be declared and identical across that model's conditions; it is not assumed available here.
2. Implement study-local bottle-to-native input adapters. The existing SOTA launchers target Breaking Bad/PartNet and do not automatically accept the shared frozen bottle corpus. Preserve full fragment sets, contact labels for training where needed, and each loader's scale/pose conventions; write any native views only inside this study's run root.
3. Validate a small fixed training set, then the same geometries under fresh rotations and samples, then all validation sources. A completed training job or lower loss is insufficient. Report native failures before adding scaffold inputs.
4. For GARF, train its fracture-segmentation feature extractor and explicitly pass the resulting checkpoint to flow training through a study-owned launcher. The current registry's bare flow command does not do this; the implementation freezes the extractor even if no checkpoint is supplied. Do not train a pose model on a frozen random encoder, or change the shared registry as part of this experiment.
5. For PuzzleFusion++, budget VQ-VAE, denoiser, verifier and required matching/verifier data. Start the full-model comparison only when those stages run correctly. Its full inference dataset can silently skip cases lacking matching files; reconcile every requested ID and count missing outputs as pipeline failures. For DiffAssemble, add a prediction export path rather than treating the generic JSON converter as an existing native exporter, and patch its actual selected `spatial_diffusion_3d_test_double_diffusion.py` implementation.
6. Preserve native reference results separately from our common protocol, especially when native evaluation uses an oracle coordinate convention or best-of-many selection.

## Scaffold interface and frame contract

Consume one frozen v3 export/checkpoint snapshot and one frozen v2 export/checkpoint snapshot as distinct prior sources, with hashes and training-source manifests. Study-local inference adapters may load a private frozen model copy when continuous queries are required; they never write into the source checkpoint directory. Reuse the same predictions across recipient models for the same raw input. Freeze all prior weights during the main study; recipient-specific joint fine-tuning would be a different experiment.

The current field predicts a truncated signed distance and bounded log error scale at query coordinates. It is anchored to an input-selected reference fragment, not a free canonical object frame. `normalize_fragments` centers fragments independently, uses the sum of fragment radii as a shared scale, and chooses the largest-RMS-radius fragment as reference. It currently validates exactly two or three fragments.

Store: input/sample hashes, source/pattern IDs, checkpoint hash, fragment ordering, reference index, per-fragment centers, shared scale, query bounds, truncation, query coordinates, predicted distances, log scales and representation version. No target points, GT transforms, exterior labels or fracture correspondences belong in the production prior record.

For learned inputs, start with 512 spatial field tokens carrying query position, distance, log scale and validity; use the same token count and projection dimension within a model's ablations. Use a fixed multiscale uniform/stratified query bank defined from input reference metadata, with both reference-neighborhood and wider-volume coverage. Freeze this input-only bank across prior conditions. Never choose main-study queries from GT surfaces or the predicted field's near-zero set: those query positions would themselves encode shape and leak the original prior into a wrong-field or null-content control. Validate coverage/representation accuracy on a small subset before locking the token budget, increasing it equally across branches if needed. The query bank can be streamed or cached without dense 256-cubed volumes.

Predicted near-surface adaptive sampling can be a later representation ablation. In that case distinguish fixed-support value swaps from replacement of the entire geometry record; the former does not remove shape information encoded in query support. The pooled B2 branch supplies no per-token query coordinates or localized positional embeddings to the recipient, only repeated pooled context plus common input reference metadata.

Do not query the reference-frame field directly at coordinates from an unposed moving fragment. Before poses exist, use fragment-local shape features to attend to scaffold geometry, with an explicit reference indicator; do not invent a cross-frame Euclidean positional bias. Once a pose estimate exists, transform points into the reference frame before spatial queries. If the scaffold encoder is rotation invariant, provide an appropriate reference-frame pathway for pose prediction separately.

Convert native transforms back to original input units, then re-express them relative to the predicted transform of the chosen input anchor. This changes the global coordinate convention using predictions and input preprocessing only. Apply field gradients and rigid updates in that common frame. Keep output as `x_aligned = x @ R.T + t`; no piece deformation, shape completion, scaling, or generated surface substitution is allowed in the scored assembly.

GARF and PuzzleFusion++ scale fragments separately and expose per-fragment scale; compose that normalization out before field queries. Their pose tensors use translation then `wxyz`, whereas DiffAssemble uses `wxyz` then translation, with an optional 6D rotation representation. Test conventions numerically rather than inferring from array shape. GARF/PuzzleFusion++ native loaders and samplers use reference poses; an input-derived identity anchor must replace any unavailable test-time target pose. GARF's native global scale also uses axis-aligned fragment extent in assembled coordinates: retain that only in a labelled native-benchmark reproduction, and use shared input-derived scale for operational comparisons.

For wrong-object or generic controls, use a fixed training-only template bank and an input-only alignment procedure. A deliberately mismatched field tests robustness; a training-derived generic field aligned to the observed reference tests category-shape bias. Report these separately so bad alignment is not mistaken for object specificity.

## Comparison A: frozen baselines, separate ranking and refinement

Cache each native model's outputs once per observation and sampling seed. All interventions use these identical starts. The primary comparison uses one native output (`K=1`); an optional `K=8` study gives all selectors the same eight candidates and includes native selection over that same pool. For deterministic models, generating extra candidates is a declared new sampling policy, not a native feature. Oracle best-of-K is diagnostic only.

Ranking is not applicable at K=1 because it cannot change the assembly. At K=8, compare the scaffold selector against native/no-scaffold selection from that identical new bank, and retain the original K=1 baseline as a separate reference. Count candidate generation cost in both K=8 conditions.

Run the following columns separately for **ranking without pose changes** and **rigid refinement** where supported. Do not combine changed candidate generation, selection and refinement into one unexplained improvement.

| ID | Condition | Purpose |
|---|---|---|
| A0 | Native output, no wrapper | Reference performance |
| A1 | Same wrapper/iteration budget, no scaffold | Extra optimization/selection control |
| A2 | v3 predicted field, constant uncertainty | Geometry benefit |
| A3 | Same v3 field, predicted uncertainty | Uncertainty benefit at fixed geometry |
| A4 | v2 field with constant uncertainty, otherwise identical to A2 | Historical adverse-prior control; replay its own uncertainty separately |
| A5 | GT field at the same representation budget, constant uncertainty | Accurate-geometry diagnostic, not a guaranteed upper bound |
| A6 | Another source's predicted field, matched query/scale policy | Object-specificity/incorrect-prior stress test |
| A7 | Fixed training-derived generic scaffold | Generic-shape control |

For A3 additionally replay spatially shuffled uncertainty with distance values unchanged and comparable total weight. When swapping distance content, hold query positions and uncertainty fixed. Separate improved geometry from merely increasing the scaffold's effective penalty strength.

The common refinement objective combines predicted contact consistency where available, a pose trust region, and a robust field term on predicted original-surface points. Contact-free models get the declared common input-only contact estimator, if used, in A1 as well as scaffold arms; its cost and errors are counted. Do not attract internal fracture surfaces to the intact-object zero level. Preserve modeled cavities. A nonpenetration term requires a reliable geometric representation; do not label a point-cloud heuristic as an exact collision measure.

Measure exact-start and 5-degree/0.02, 15-degree/0.05 perturbation controls with the anchor fixed. These GT-initialized diagnostic probes are isolated from practical inference. If the predicted objective moves correct poses away, mark direct refinement as unsupported until the weighting/acceptance design changes. Do not infer that a lower field objective certifies an improved assembly.

## Comparison B: scaffold as a learned additional input

For each selected model, train one native baseline from scratch. Branch all following runs from the same selected native checkpoint and use equal additional updates, effective batch size, input streams, optimizer policy and validation-selection budget. Freeze Stage 2. Preserve native frozen components, including GARF's pretrained feature extractor and PuzzleFusion++'s VQ-VAE. Match which native pose/matching parameters are updated across B0–B4; train the new branch in B1–B4. Do not silently compare adapter-only tuning to full-network tuning or unfreeze a native fixed encoder. Start the scaffold residual gate at zero so initialization reproduces the baseline.

| ID | Matched training condition | Main contrast |
|---|---|---|
| B0 | Native architecture, additional training only | Controls additional optimization budget |
| B1 | Same scaffold branch receiving learned null tokens | Controls additional recipient capacity and branch compute |
| B2 | Pooled predicted v3 scaffold descriptor, broadcast to the same token slots, without per-token positions | Global context versus access to localized spatial tokens |
| B3 | Spatial v3 geometry with constant uncertainty channel | Spatial geometry versus pooled context |
| B4 | Identical spatial geometry with predicted log scale | Contribution of uncertainty |

Retain the original native checkpoint as a further reported reference. B0 has fewer parameters than the scaffold branches; B1 is the capacity control. Match B1–B4 token lengths and projections where architecture permits and report actual parameter count/FLOPs rather than claiming perfect compute equivalence.

Supply identical reference indicators, masks, input normalization and any anchor features to B1–B4. Use the same input-derived frame preprocessing for B0; if exposing a reference cue requires an additional native branch, document and control it. An extra anchor cue must not be mistaken for a benefit from scaffold geometry.

For the winning architectures, add an encoder-feature control: supply frozen Stage 2 encoder features through a matched context branch without decoded field geometry. This tests whether the improvement is attributable to explicit scaffold information rather than the auxiliary encoder alone. If robust uncertainty is the central claim, add a training arm with the same scheduled corruptions in B3/B4, then compare on withheld corruption types/severities.

Evaluation-only content swaps on the same trained B4 recipient: v3, v2, GT distances with fixed sigma, wrong-source, generic, shifted/rotated and spatially incomplete scaffold, constant sigma, and spatially shuffled sigma. These measure sensitivity/distribution shift. Do not label GT substitution a trained upper bound or its failure proof that accurate priors cannot help. A recipient trained with GT scaffold is an optional separately trained oracle diagnostic.

If claiming that v3 is a better learned-input prior than v2, additionally train a recipient with v2 under the identical B4 schedule; test-only substitution into a v3-trained model confounds prior quality with distribution shift. The frozen A2/A4 contrast already isolates the two fields under the specified common refinement policy.

Keep training scaffolds predicted rather than replacing them with perfect targets. Prefer source-level out-of-fold prior predictions for adapter training: the entire supervised prior pipeline must exclude the held-out fold, not just its decoder. If that cost is deferred during the pilot, explicitly report in-sample prior-training predictions and measure the train/validation prior-quality gap. Before making the broader claim, use cross-fitting or a disjoint prior-training/adapter-training split to rule out this mismatch. Every held-out evaluation prior is generated without training on that object's other cuts.

Any required cross-fitting runs use study-owned prior implementations/configurations and independent outputs, not the original three-stage trainer or its checkpoints as mutable resume state.

Freeze the final token sampling, field preprocessing, corruption policy, losses and checkpoint selection before final testing. Candidate scoring based on GT errors must never choose the delivered sample.

## GPAT: separate target-quality matrix

GPAT already consumes a target shape. Do not create a supposed native no-scaffold GPAT baseline by deleting its required target. Compare: native clean target; predicted target; matched-budget coarsened/noisy clean target; wrong-object target; training-derived generic target. Distinguish frozen native GPAT substitution from matched fine-tuning on predicted targets.

Replace every target-dependent representation, including optional `target_100k.npy` and its nearest-neighbor map, not only `target.npy`. Otherwise the pose-fitting stage retains the intact target. Keep target point budget, target-to-part fitting and optional CMA refinement equal. Report semantic part equivalence handling and do not combine these results numerically with fractured-bottle assembly. Train/extend a study-owned prior for this semantic setting before claiming benefit there; leave the original Stage 2 implementation unchanged.

GPAT requires actual surface point clouds, not signed-distance query tokens. Extract and sample each predicted implicit surface at the same target budget as the controls, then rebuild all optional dense-target data from that surface inside this study's storage. Record invalid/empty surfaces as failures. For predicted-target training, regenerate target-point assignment supervision using training-only geometry and an explicit unmatched/ignore policy for unsupported generated points; never reuse clean-target point labels on different sampled geometry.

## Datasets, splits and evaluation

Pilot: current prepared bottle corpus, fingerprint `b7f2b84a20964276c894300a7ece4d8f97f232903bace0f14131535594462043`. It contains 73 accepted source objects and 546 fracture patterns: 418 training, 48 validation, 69 test and 11 cut-holdout patterns. The 48 validation patterns cover six sources; test and cut-holdout share 11 sources. These are procedural complementary cuts and complete two/three-piece sets, not the full Breaking Bad problem.

The historical test/cut-holdout have already been inspected in earlier diagnoses. Preserve their identities for continuity but call this pilot exploratory. A fresh source-object holdout is needed for the confirmatory broader study. Never split different cuts of one source between prior/recipient training and test. Hash/deduplicate underlying objects when combining ShapeNet-derived corpora or pretrained resources.

Broader study: first use whole two/three-fragment Breaking Bad sets on disjoint source objects; train a study-owned prior on the appropriate training distribution. Extend that isolated provider's normalization, masks, training and query coverage before moving to 4–20 fragments; leave the original Stage 2 implementation unchanged. Do not drop pieces from large sets to satisfy the old interface. Missing fragments and real scans are separate distribution/task extensions requiring appropriate targets and evaluation; bottle success does not establish them. Keep the suite's everyday/artifact and PartNet protocols distinct.

The existing two evaluators use incompatible Chamfer definitions: `reassembly/evaluation.py` averages two unsquared Euclidean nearest-neighbor means; `sota_repro/evaluation.py` sums two squared-distance means. The numeric threshold 0.01 therefore does not mean the same thing. For the bottle pilot use the existing reassembly definition and scale, with success requiring whole-object and every part's Chamfer <=0.01. Also report native metrics under their original named definitions. Do not pool the two Chamfer columns.

Score every model's exported poses on one fixed common evaluation point set per original fragment, with the same target samples and shared normalization. This point set is independent of each native loader's internal sampling budget. Record input observation sampling separately, and keep it identical within a model's conditions. Native sampling differences must not change the common metric's evaluation geometry.

Primary outcomes: source-macro assembly success difference and the fraction of initially successful assemblies made unsuccessful. Count missing/invalid pose outputs as failures in the full denominator. Report their status separately; do not fabricate identity poses or silently drop failed examples. The common adapter currently requires nonempty part arrays, so an explicit failure-capable record contract is required before scoring this study.

Secondary outcomes: moving-part rotation/translation errors, part accuracy, whole/per-part Chamfer, fracture-contact alignment, candidate recall at K, ranking quality, accepted/rejected updates, collision estimate where valid, prior error/calibration, wall time and peak GPU/disk use. Report continuous pose metrics on returned outputs together with failure rate. Respect geometric symmetries and legitimate semantic equivalences without allowing arbitrary fragment reassignment. Do not let a plausible whole-object surface hide swapped or misplaced fragments.

Use three independent recipient-training seeds for confirmation and three held-out observation seeds for rotations/resampling. For each recipient-training seed, train its native baseline (including required learned prerequisites) and branch the compared conditions from that seed's checkpoint. Three adapter runs from one common native checkpoint measure adapter variability only and must be labelled accordingly. Keep the same frozen Stage 2 prior across these paired recipient contrasts; this estimates benefit conditional on that prior. Repeat with independently trained priors before claiming robustness to prior-training randomness. A first one-seed smoke/pilot selects feasible integrations, not final claims. Bootstrap over source objects with all their patterns retained; report per-source results and paired uncertainty intervals. Six validation sources provide weak statistical precision regardless of how many cuts or seeds are evaluated. Adjust confirmatory multi-model comparisons or predeclare primary contrasts and label the rest exploratory.

## Native evaluation hazards to remove before guidance

- Jigsaw's estimator can use a GT pivot quaternion/translation to align its outputs. Intercept before this operation, set `align_pivot=False`, and use an input-derived anchor frame. A GT global gauge convention may remain an explicitly labelled final scoring operation; it is not a legal scaffold input.
- CCS's generic min-of-N test code selects with GT losses when multiple samples are enabled. The pinned geometric default has `noise_dim=0` and one sample. Do not raise K and report oracle selection as deployable inference.
- PMTR computes GT correspondence bookkeeping within its forward path. Its training branch can replace predicted coarse matches; prediction export must use evaluation mode and not consume supervision.
- GPAT's sparse and dense clean targets can both influence its output. Replace and trace all target paths in predicted-target conditions.
- Audit diffusion/flow wrappers for GT best-of-many aggregation, pivot conventions, supplied target normalization and output completeness. Store raw samples and select using only input-available quantities. The final evaluator may read GT after predictions are sealed.

## Execution order and decision rules

| Phase | Work | Decision/output |
|---|---|---|
| 0 | Confirm disk capacity; inventory data/prior lineage; implement and verify native adapters, frame conversion and failure records | Immutable experiment manifest; no learning interpretation before contract checks pass |
| 1 | Train Jigsaw, CCS, GARF native bottle baselines sequentially, including GARF backbone stage; assess fixed-fit, rotation/resampling and validation | At least one functioning matching and one functioning regression/generative baseline for a meaningful cross-family test |
| 2 | Import a frozen v3 snapshot when available; run study-owned v3/v2 field and pose probes; cache observations and prior inputs here | Separate evidence for ranking, gradients and representation quality; no original-pipeline gate or diagnostic-state changes |
| 3 | Run A0–A7 on frozen pilot models; run B0–B4 for the feasible first models using one training seed | Identify which integration works and whether extra training/capacity/global shape explains it |
| 4 | Repeat the selected primary B4-vs-B0/B1 and B4-vs-B3 contrasts with three training seeds; add encoder-feature and robustness controls | A reproducible effect in at least two distinct model families supports expansion; one-family benefit supports a narrower claim |
| 5 | Add remaining suite models; extend/train a study-local copy of the prior architecture and evaluate fresh Breaking Bad source holdout | Confirm generalization independently of the original three-stage project; include zero/negative results |
| 6 | Optional GPAT target-quality study and contemporary-baseline extension | Separate semantic and broader-literature results |

Choose practical minimum improvement and acceptable harm on validation before confirmatory testing. Use effect sizes, counts, and source-level intervals, not only a significance flag. Record baseline ceiling effects. Do not continue broad expensive training solely because an oracle scaffold helps, and do not discard failed model families from the reported matrix. No predetermined performance threshold can certify novelty.

If GT geometry helps A5 but v3 does not help A2/A3, investigate prior quality or trust weighting. If neither helps direct refinement but B3/B4 helps, prefer learned conditioning. If B1 matches B4, extra capacity may explain the gain; if B2 matches B3, spatial detail has not shown benefit; if B3 matches B4, uncertainty has not shown benefit. If encoder features match the field, weaken the explicit-geometry claim. If nothing beats matched controls, report the negative result and revise the hypothesis before expanding.

## RTX A5000 and storage plan

Use one GPU job at a time. Account for other work already using the A5000 and wait for capacity; this study does not own or interrupt the original pipeline's jobs. The A5000 is an Ampere GPU compatible in principle with FlashAttention-2's architecture requirement; this does not certify GARF's full dependency stack or memory use. GARF's PTv3 encoder may still enable flash attention independently of its denoiser switch. Validate the actual pinned extensions and a real forward/backward/evaluation pass on this GPU.

Initial operational target: PyTorch peak reserved memory below 20 GiB, while also monitoring total device usage/non-PyTorch allocations and leaving device headroom. With 24,564 MiB total, a 20-GiB PyTorch limit alone is not a guarantee that a job fits. Preflight the largest allowed fragment count and every required training stage. If it does not fit, reduce microbatch first and preserve effective batch with gradient accumulation; then enable supported activation checkpointing, chunk field/attention work, or offload prior caching. Do not silently reduce geometry resolution for only the scaffold condition. Match any required reduction across comparison arms and record it.

Start training microbatch at one, increase only after measurement, and use a matched effective batch target such as eight where the native objective permits. Keep rigid fitting/rotation geometry numerically stable in FP32/FP64 as appropriate; use AMP only where supported. Train GARF pretraining and flow separately; train PuzzleFusion++ stages separately. Separate environments for legacy dependencies, with resumable optimizers/RNG states and atomic checkpoints.

The supplied root filesystem has about 161G available; `/data` has only 30G available and is 99% used. Proposed workspace: `~/Kartik_23CS30026/scaffold_sota`, subject to checking its resolved path/mount with `findmnt -T`/`df`. Keep checkpoints, environments and scratch on the root filesystem; do not default any native loader to `/data`.

Preserve a hard free-space floor of 50 GiB across **all** concurrent work on that filesystem. Budget this independent study for up to 75 GiB of additional growth and leave a separate 15-GiB allowance for other work, giving a 90-GiB filesystem-growth forecast and roughly 71 GiB remaining from the supplied snapshot. The allowance for other work does not authorize this study to run or manage it. These are proposed ceilings to validate against measured footprints, not promised dataset/environment sizes:

| Additional storage category | Initial ceiling |
|---|---:|
| Other work: separate allowance, outside this study's ownership | 15 GiB |
| Retained SOTA corpus (pilot reuses bottles; broader corpus later) | 20 GiB |
| Active environments, compiled extensions and package/build caches | 22 GiB |
| One native preprocessing/cache/scratch stage | 12 GiB |
| SOTA weights, active resumable state, pose/prior caches | 18 GiB |
| Logs, manifests, compact reports and visualizations | 3 GiB |
| Independent study subtotal | 75 GiB |
| Filesystem growth forecast including other work | 90 GiB |

Current used bytes are already reflected in the 161G snapshot; do not count them again as new growth. Inventory existing directories before reservation. The original repair workflow manages its own cap and artifacts independently; this study's accounting does not modify them. Study-local prior training/cache costs must fit inside this study's 75-GiB budget, not the allowance for other work. Preflight archive expansion, checkpoint writes and environment builds before starting, and monitor free space during runs. If actual footprints exceed a category ceiling, revise the measured allocation within the total and reserve, or pause for capacity; never erase unrelated/user data or silently shrink one condition's training data. Stage environments sequentially if retaining every legacy environment would exceed the envelope.

Use one disposable native cache at a time, compact float32 prior-token records, and best/latest checkpoints rather than every epoch. Never delete a checkpoint required by a dependent stage or an active/resumable run. Keep exact selected priors and baseline outputs for paired replay. Avoid repeating the old exhaustive 256-cubed GT-grid sweep: it dominated diagnostic runtime and is unnecessary for this intervention study. Measure cached-representation fidelity with a small validation-only comparison instead.

No reliable hour estimate is available before throughput is measured. For scheduling, record preprocessing time, seconds/update, validation cost and peak storage per stage; total GPU work is the sum of native prerequisite training plus each condition's updates and evaluations. B0–B4 across three models is 15 adaptation runs per seed, 45 for three seeds, plus three independently trained native baselines per model and their prerequisites, before additional diagnostics. Phase 3 screens first; Phase 4 replicates selected contrasts rather than blindly executing the complete factorial. No time limit does not remove the need for resumability or learning-based stopping.

## Novelty and related work

The broad idea is established. [Jigsaw++](https://arxiv.org/html/2410.11816v2) generates complete-shape priors, tests transfer across initial assembly methods, and describes rigid refinement. Its fracture-reassembly experiment constructs surface-to-prior matches using ground-truth point positions and then perturbs those matches. A credible distinction is practical guidance without those oracle matches, together with measured spatial/uncertainty effects. No official Jigsaw++ release was found in the reviewed author page/repository inventory on this date; treat it as a close conceptual comparator and clearly label any independent reimplementation.

[GPAT](https://proceedings.mlr.press/v229/li23a/li23a.pdf) already conditions on target point clouds. Earlier [template-guided fracture assembly](https://openaccess.thecvf.com/content_iccv_2015/papers/Zhang_3D_Fragment_Reassembly_ICCV_2015_paper.pdf) combines template evidence and fracture matching. Neither adding a template penalty nor combining global shape and local contacts alone is a new contribution.

The pinned eight-model suite is not an exhaustive September 2026 SOTA list. [Assembler](https://arxiv.org/abs/2506.17074) predicts assembled anchors and rigidly fits parts; [TORA](https://arxiv.org/abs/2604.04050) transfers topology representations from a frozen teacher during training and has [official code](https://github.com/NahyukLEE/tora). [SARe](https://arxiv.org/html/2603.21611v1) is relevant to flow-based assembly and selective refinement. Add at least one appropriate recent released baseline during the broader phase if making a current-SOTA claim; first audit its data/modality and compute requirements. [FragmentDiff's official repository](https://github.com/xuqunce/FragDiff) was still README-only in this audit, so do not count it as a ready executable baseline.

Proposed claim, conditional on results: **An input-only predicted whole-object field improves rigid assembly across matching and generative architectures, with spatial conditioning and calibrated error estimates reducing failures under imperfect priors.** Do not claim “first” or universal improvement from this planning audit. If only bottles, one architecture, pooled context, or oracle matches benefit, state that narrower result.

## Code evidence for implementation planning

These repository-relative locations were inspected as read-only references; proposed branches above are not implemented yet. They are not authorized edit locations for this study. All study-specific implementations and source patches belong under `scaffold_sota/`.

| Component | Source entry points |
|---|---|
| Stage 2 output/context | `reassembly/model.py:142` (`ShapeField`); `reassembly/repair/model.py:164` (field adapter), `:211` (scaffold-independent matcher) |
| Frames and field sampling | `reassembly/geometry.py:52`, `:148`; `reassembly/repair/fields.py:50`, `:132` |
| Two/three-piece and transfer limitation | `reassembly/geometry.py:60`; `reassembly/prepare.py:54`; `reassembly/repair/assembly.py:79` |
| Metrics/export | `reassembly/evaluation.py:22`; `sota_repro/evaluation.py:104`; `sota_repro/adapters.py:17` |
| Jigsaw | `sota_repro/models/jigsaw/upstream/model/jigsaw/joint_seg_align_model.py:145`, `:163`, `:248`; `utils/estimate_transform.py:137`, `:145` within that upstream tree |
| CCS | `sota_repro/models/ccs/upstream/multi_part_assembly/models/wx_transformer/network.py:103`, `:116`; `models/modules/base_model.py:360` under `multi_part_assembly` |
| PMTR | `sota_repro/models/pmtr/upstream/model/pmtr.py:100`, `:122`, `:141`; `model/local_global_registration.py:314`; `test_mpa.py:145` |
| CMNet | `sota_repro/models/cmnet/upstream/model/cmnet.py:149`, `:167`, `:184`; `model/local_global_registration.py:282`; `multi_part_assembly.py:128` |
| GARF training prerequisites | `sota_repro/models/garf/upstream/README.md:175`; `sota_repro/models.lock.yaml:106` |
| GARF conditioning and frozen extractor | `sota_repro/models/garf/upstream/assembly/models/denoiser/modules/denoiser_transformer.py:325`, `:342`, `:370`; `assembly/models/denoiser/denoiser_base.py:60` in that upstream tree |
| PuzzleFusion++ conditioning and full inference | `sota_repro/models/puzzlefusion_pp/upstream/puzzlefusion_plusplus/denoiser/model/modules/denoiser_transformer.py:117`, `:191`; `puzzlefusion_plusplus/auto_aggl.py:201`, `:268`, `:315` in that upstream tree |
| DiffAssemble selected path and feature mixing | `sota_repro/models/diffassemble/upstream/puzzle_diff/train_3d.py:19`; `puzzle_diff/model/backbones/efficient_gat_3d.py:185`, `:206` in that upstream tree |
| GPAT target use | `sota_repro/models/gpat/upstream/learning/gpat/gpat.py:60`; `learning/assembler.py:89`, `:110`, `:123`, `:133` in that upstream tree |
| Versioned model/data contracts | `sota_repro/models.lock.yaml`; `sota_repro/docs/MODELS.md`; `sota_repro/docs/DATA.md`; `sota_repro/README.md` |

No further user decision is needed to establish this plan. A valid exported v3 snapshot, successful native SOTA training and measured runtime/storage preflight remain future execution dependencies; none is assumed achieved. Completion or success of the original three-stage pose pipeline is not a prerequisite. Native SOTA preparation/training can proceed independently before the v3 artifact is available, with sequential GPU scheduling.

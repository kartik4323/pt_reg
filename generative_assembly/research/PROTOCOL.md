# Experiments for generative-prior-assisted fragment assembly

Date: 2026-09-14. Status: experimental design; no new model inference, training, or performance experiments have been run. The accompanying JSON is a planning manifest, not an implemented launcher configuration.

## Research question

Can a pretrained model complete an image rendered from broken fragments into an object-specific whole-object image, and can its reconstructed 3D geometry improve rigid assembly without task-specific complete-shape or pose targets?

The main outcome is correct placement of the measured fragments. A plausible generated object, a lower training loss, or a better-looking rendering does not establish success. Predict one rigid transform per fragment; generated geometry is an auxiliary hypothesis and is reported separately from the assembled input points.

## Scope and supervision contract

- Pieces are already grouped by original object; relative poses are unknown; scans share physical scale. No grouping across mixed objects is assumed.
- Start with all pieces present, two or three asymmetric pieces, sufficiently scanned fracture interfaces and little material loss. Five-piece cases, missing pieces, noise and real scans are later tests.
- The main inference path receives fragment XYZ and optional normals estimated from those points. It receives no complete object, true pose, true fracture/exterior labels, true silhouette, true contact graph, or reference photograph of the intact object.
- Foundation models stay frozen in E0–E5. A model pretrained on complete objects is an external prior, not evidence that the full system learned only from fragments. Record its training-data disclosures and known overlap; unknown overlap remains unknown.
- E6 trains an assembler only from fragment sets and inferred poses. Its initialization must not silently import a target-domain supervised pose model.
- Ground truth is available to the evaluator and explicitly marked oracle arms. Those arms are diagnostics, excluded from the deployable pipeline and pseudo-label cache.
- If complete shapes or pose labels are used to choose prompts, tune thresholds, calibrate confidence, or select checkpoints on development data, report that development supervision. The strict track fixes these choices from fragment-only criteria or a separate engineering set; evaluator labels diagnose performance but do not select per-instance outputs or tune the model. A claim of zero labeled data anywhere would be inappropriate for a study tuned against labeled validation scores.
- Category names are excluded from the main generic-prompt track. A category-informed track adds only a declared category label. Do not use a VLM's inferred category as ground truth; log that separately if explored.

## What existing work supplies

The current [GenPC repository](https://github.com/GenPC-Team/GenPC) supplies a close implementation reference for depth prompting, image generation, image-to-3D generation and alignment. Its current README recommends Qwen-Image-Edit and TRELLIS.2, and lists ControlNet and InstantMesh as alternatives. This differs from simply reproducing the original CVPR 2025 stack. Pin the code revision and audit the data/evaluation path before adapting it; do not run a demonstration script as an unattended benchmark or inherit ground-truth-dependent alignment.

The existing local [field diagnostic](../../artifact_work/field_analysis.md) motivates correct-start drift and contact-only controls: that historical learned field could reduce its objective while worsening true geometry. Those measurements concern a different model, and are not results for the new generative pipeline. This protocol is independent of the existing repair and scaffold-transfer experiments; no changes to their models or runs are needed.

## Pretrained model shortlist

| Role | First candidate | Why test it | Qualification |
|---|---|---|---|
| Image completion, principal candidate | [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) | Released image editor supporting multiple input images and structural image conditions; can receive a neutral fragment render and its depth/edge representation. | Editing capability does not establish fracture completion or camera/geometry preservation. Benchmark those. |
| Image completion, modular first smoke test | [SD 1.5 inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) plus [ControlNet depth](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth) | A compatible masked-inpainting/depth-control pipeline gives a reproducible contrast to instruction editing at 512 × 512. | The missing object's extent is unknown: ordinary hole filling is insufficient. The inpainting repository is a mirror of the original weights. |
| Structural-control ablation | Same SD 1.5 pipeline without depth ControlNet | Tests whether depth conditioning adds value beyond appearance and an inpaint mask. | Stock depth control has no missing-depth validity channel: zero-valued unknown pixels are not measured background. Test their treatment. |
| Optional higher-resolution image control | [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) | Tests a dedicated inpainting checkpoint trained at 1024 × 1024. | Record the larger resolution and compute as changed factors. Do not attach the SD 1.5 depth checkpoint to SDXL. |
| Image-to-3D, fixed first bridge | [InstantMesh](https://github.com/TencentARC/InstantMesh) | Released image-to-mesh inference gives an explicit surface from which to sample the target point cloud. | Hold it fixed while comparing image generators. A complete input image still leaves hidden geometry ambiguous. |
| Image-to-3D, later replication | [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | A second bridge tests whether conclusions depend on InstantMesh. | Use only after a resource smoke test. Changing both image and 3D models in one comparison confounds attribution. |

Use official checkpoint identifiers and immutable revisions. Record precision, quantization, offloading, scheduler, guidance, steps, resolution and seed. Qwen's published checkpoint is large; fitting its full pipeline on a nominal 24-GB GPU is not assumed. Quantized/distilled variants are separate model conditions, not identical replicas. Run image and 3D stages in separate processes where needed, cache outputs, and measure actual VRAM and latency before estimating the study budget.

## Dataset, splits and repeat structure

Proposed budgets, subject to a source-identity inventory:

| Pool | Original objects | Purpose |
|---|---:|---|
| Development | 24: eight bottles, eight bowls, eight mugs | Protocol diagnostics and small screens; label any tuning from reference scores. |
| Unlabeled training | 120: forty per category | E6 only; fragment sets exposed to learning, reference geometry sealed for auditing. |
| Locked synthetic evaluation | 60: twenty per category | Paired evaluation after configuration freeze. |
| Real-scan transfer | At least 15 distinct objects if available | External validity; report uncertainty and selection constraints. |

Use [Breaking Bad](https://breaking-bad-dataset.github.io/) for synthetic fractures or a separately documented physical fracture simulation. Include two-, three- and five-piece variants where valid, ideally one fixed pattern of each count per source. Do not synthesize incompatible patterns merely to meet a count. A bottle-only first screen is acceptable, but must be reported as such; smooth rotationally symmetric bottles are insufficient by themselves to establish pose recovery.

The first small screen uses twelve development objects, one two-piece pattern each. Later expand to all eligible development patterns. Sample size is a pilot budget, not a power guarantee. Estimate source-level uncertainty before funding a larger confirmatory run.

Keep every fracture pattern, view, resampling and pose augmentation of an original object in the same split. Detect duplicate meshes and near-duplicates across pools. Previously inspected local test objects are development evidence, not a newly blind final set. If the required independent sources are unavailable, record the reduced scope explicitly.

All iterative stage gates are assessed on development sources. Freeze E4, E5 and E6 choices before jointly scoring them on the locked synthetic evaluation pool. Do not inspect final E4 results, redesign E5 or E6, and then reuse that same pool as an untouched test. A disappointing final result is reported; subsequent redesign requires a new final set.

Use fixed generation seeds 11, 23, 37 and 51. Use three independently randomized input-pose/resampling repeats for full evaluation; average them within source. E6 uses three training seeds. These repeats are not additional independent objects.

## Input preparation and rendering

1. Retain immutable full-resolution fragment clouds. A point budget such as 2,048 observed points per fragment is an initial engineering setting; use fixed seeds and retain a separate evaluation sample.
2. Center each fragment independently, retain its center, and use one input-only shared scale: the maximum fragment diameter, estimated by a fixed rotation-invariant procedure. Do not infer scale from the correctly assembled bounding box or normalize every piece to a separate unit size.
3. Fix a reference piece using an input-only rule such as largest estimated surface area, breaking ties by stable fragment ID. All predicted transforms are expressed relative to it and exported back to the original input coordinates.
4. Render six fixed local-camera views per selected input at 512 × 512, with saved cameras, depth, silhouette and neutral shaded RGB. Estimate normals consistently. Maintain a fixed padded canvas based on observed fragments; do not crop using the intact object or its full silhouette. Record truncation if the hypothesized whole exceeds the canvas.
5. Select the first camera from the reference fragment using an input-only visibility score. For each screen case, F and A share exactly the same reference piece, camera, canvas and normalization; the rough assembly is expressed in that reference frame. This pairs their contextual-input comparison and allows one true-image oracle reconstruction per case. Camera selection must not inspect the intact render. Each alternative view creates a new generation input and counts toward the hypothesis budget.
6. Construct two principal inputs: F, the largest informative fragment; and A, the top rough assembly from a frozen geometry-only candidate proposer. Preserve separate fragment identity for scoring. Multi-image input of up to three separately rendered pieces is a later arm with an explicitly larger information budget, not an unlabelled replacement for F.
7. For inpainting, protect only the estimated visible original-exterior region. Treat the missing region as unknown, including canvas outside the fragment silhouette. Do not lock the white background as if the original object could not extend into it. Fracture faces are editable because they should become internal.
8. Exterior classification is itself uncertain. The main mask must come from an input-only fixed heuristic or declared frozen model. Include no-protection and true-exterior-mask oracle diagnostics. True exterior labels never enter a main generator input. Report protected area and rejected/uncertain area to expose an all-zero-mask shortcut.

Example fixed generic instruction, to be adapted only once for the chosen model API:

> The image shows a broken piece or rough arrangement of pieces from one rigid object. Create one plausible intact object containing the visible original exterior geometry. Retain the reference camera, the scale and position of surviving exterior features. Continue the missing body beyond the fragment boundary. Remove exposed break faces where they become internal. Use a neutral gray material and plain background. Show one object, with no labels or additional objects.

The category-informed version adds exactly one sentence, such as "The object category is a mug." Prompts are requests, not guarantees of geometric preservation.

## Experiment sequence

Run E0 first. E1 and E3 can then proceed independently. E2 consumes E1 images. E4 integrates the components only after the bridge and pose-use diagnostics are understood. E5 tests trust, E6 tests learning, and E7 tests scope. A failed early gate is a reason to diagnose that component before scaling, not to relabel failed outputs.

### E0 — Coordinate, rendering and resource integrity

**Question:** Can the pipeline preserve measured geometry and coordinate conventions before any generation?

Back-project known rendered depth with its camera and compare it to the corresponding visible input points. Apply known rigid transformations and verify export/inversion. Confirm all original fragments survive I/O with their relative sizes unchanged. In a clean evaluator, audit that inference can execute with all reference files unavailable. Run one image and one image-to-3D job, recording elapsed time, peak process/GPU memory and cache size.

**Controls:** no image editing; direct depth round-trip; identity transforms; one deliberate wrong camera/scale to verify the check detects it. Correspondences for numerical unit checks are synthetic diagnostic data, not assembly labels.

**Pass:** transform round-trip error below 1e-5 in normalized coordinates; correct depth back-projection within the measured rasterization tolerance; no reference-file accesses; explicit failures instead of silent empty outputs; resource use within the actual available machine budget. These are implementation gates, not evidence of learned completion.

### E1 — Does image completion recover the right object geometry?

**Question:** Does a pretrained image editor add an object-specific continuation while retaining the measured exterior?

On the twelve-object screen, compare raw fragment/rough-assembly render, SD 1.5 inpainting with depth control and Qwen editing. Use both F and A inputs. Generate four samples per input/model. This is 12 × 2 × 2 × 4 = 192 edited images, plus 24 raw controls. Begin with generic prompts; category prompts, inpainting without depth and higher-resolution SDXL are follow-ups, not a full initial factorial grid.

**Evaluator-only references:** render the intact source from the same reference camera; identify the original exterior visible in both fragment and intact views. Keep break faces and regions that become occluded outside the strict preservation score. Mark cases without comparable visible exterior as not assessable for that metric, while retaining them in pipeline denominators.

**Measurements:** raw-output exterior contour/keypoint displacement, intact silhouette IoU, missing-region silhouette precision/recall, multiple-object rate, cropped-output rate, changed-view rate, and generation failure rate. Segment neutral objects with a frozen background/mask method. Include human geometric inspection of every screen output, blinded to model identity where possible. CLIP similarity or visual appeal is optional secondary evidence.

Compute exterior drift in pixels at the original fixed camera/canvas. Never realign the generated image to the reference for the main metric. If compositing or mask restoration is used, report both pre- and post-composite scores; copying observed pixels trivially cannot prove the generator preserved geometry. RGB alone supplies no metric-depth score: assess that after E2.

**Diagnostic gate:** at least 90% single-object usable outputs; median exterior contour displacement no more than 2% of image width on comparable regions; median intact-silhouette IoU improves by at least 0.10 over raw input. These are proposed engineering thresholds, not literature standards. Small-sample failure triggers diagnosis; the final decision is downstream pose utility, not silhouette alone.

### E2 — Does useful image completion survive image-to-3D reconstruction?

**Question:** Is the image-to-3D bridge introducing the error that prevents assembly?

Fix InstantMesh, its camera/crop handling and sampling budget. Reconstruct: (a) raw input render, (b) the two hypotheses selected from each E1 input/model by the frozen input-only score, and (c) the true complete image, an oracle diagnostic. For the screen this is at most 24 raw + 96 edited + 12 true-image reconstructions = 132. Also retain the true mesh as a geometry oracle; a true image is not a true 3D shape.

The selector may use visible-exterior drift, output validity, silhouette retention and agreement with held-out observed patches; it cannot use true whole-object IoU or reference 3D error. If no trustworthy selector is available, use the first two seeded outputs, clearly labelled. On a smaller subset reconstruct all four hypotheses to separate sampling quality from selection quality. Report true best-of-four only as an oracle ceiling.

Sample the reconstructed mesh surface; do not substitute Gaussian centers for a verified surface. Estimate one template scale and pose from observed exterior evidence within bounded alignment candidates. Never deform or separately rescale the real fragments. The main alignment cannot use intact-object ICP. Any reference-aligned shape metric is a separately labelled oracle-aligned diagnostic.

**Measurements:** full-shape Chamfer and F-score, original-exterior residual, missing-region surface error, thickness/cavity failure where evaluable, scale error, camera consistency, and correspondence residual on observed patches withheld from alignment. A mesh can be watertight while being geometrically wrong.

**Diagnostic gate:** at least 90% valid nonempty surfaces; positive median improvement in full-shape error over raw-image-to-3D, accompanied by no material increase in held-out exterior error. Use a provisional 10% relative median Chamfer improvement to justify expansion. If true-image reconstruction is already poor, change the 3D bridge before blaming fragment-image completion. If only reference-aligned scores improve, solve the template alignment problem before E4.

### E3 — Can even a perfect complete shape help the pose solver?

**Question:** Is the complete-shape-to-assembly mechanism useful independently of image generation?

Freeze a geometry-only proposer and optimizer. Compare no template, true complete mesh, an unrelated category-matched mesh, and a deliberately distorted true mesh. Wrong templates are selected from an allowed non-test pool, excluding the current source and duplicates, and aligned using the same observed-input procedure. The true-shape arm is labelled privileged, including its category/identity information.

Evaluate two distinct regimes:

- **Local basin diagnostic:** exact poses and poses perturbed by 5°/0.02 and 15°/0.05 in shared normalized units. GT is used only to create these diagnostic starts. It does not supply contacts, surface labels or fragment-to-template correspondences to the principal oracle-template comparison. Separate true-exterior/true-contact controls can localize failures, but their added privileges must be explicit.
- **Unknown-pose regime:** start from a cached bank of 32 proposals made solely from the unposed fragment set. Every template condition receives exactly that bank; no GT-near candidate is injected.

For complete closed-fragment geometry, use opposed fracture normals and non-interpenetration constraints. For open/incomplete surface scans, a reliable signed volume may not exist; mark volumetric collision unmeasurable and report a surface proxy separately. Avoid attractive all-to-all contact losses that pull every piece together.

**Measurements:** source-macro assembly success, moving-part errors, candidate recall@32, selection regret against the bank's evaluator-only best, exact-start drift and initially-correct-to-incorrect transitions. If recall is low, the proposer is the bottleneck; a ranker cannot select a missing solution.

**Go/no-go:** a true complete template should improve ambiguous/perturbed cases over the same geometry-only solver. Failure means stage 4 needs redesign even with perfect generation. If it helps only when true correspondences are provided, correspondence inference remains unsolved. No-template and true-template exact-start cases should both remain geometrically successful.

### E4 — Does generated guidance improve assembly end to end?

**Question:** Do selected generated shapes improve actual rigid poses from unknown inputs?

Use cached input clouds, generation outputs, 32 pose candidates and identical refinement budgets. The primary arms are:

| Arm | Shape guidance | Purpose |
|---|---|---|
| B0 | None | Geometry-only baseline |
| B1 | Raw render → InstantMesh | Tests whether image completion adds value beyond image-to-3D alone |
| B2 | Selected generated complete image → InstantMesh | Main proposed bridge |
| B3 | Wrong but category-matched complete shape | Tests object-specific information |
| B4 | True complete shape | Privileged performance headroom |
| B5 | None, with additional geometric search under matched wall-time budget | Tests whether extra compute explains the gain |
| B6 | True complete image → InstantMesh | Privileged control separating image-to-3D error from image-completion error |

First test reranking only; then test contact-preserving rigid refinement as a separately named intervention. A matched candidate bank makes the ranking comparison clean; the wall-time control addresses total pipeline cost. Report F and A generation inputs separately. The A condition uses one frozen rough assembly, not a fresh GT-picked render for each final candidate.

**Final confirmation criterion:** B2 improves source-macro assembly success over B0 by at least five percentage points, the paired source-bootstrap 95% interval for the difference excludes zero, and at most 5% of baseline-successful cases become failures. Also report B2–B1 and B2–B3: B2 beating B0 alone does not show that image completion or object-specificity caused the gain. Failure of B2–B5 means benefit may be explainable by compute allocation. Compare B4 → B6 → B2 to localize losses in reconstruction and completion. An interval crossing zero is inconclusive, not proof of no effect; enlarge the study only with a prespecified extension. Advancement to E5/E6 uses development evidence, not this final-test result.

### E5 — Does uncertainty filtering reject harmful hypotheses?

**Question:** Can fragment-only evidence identify misleading completions?

Compare unconditional template use, constant weak weight, and a proposed input-only gate at image budgets K = 1, 2 and 4. Use nested fixed seed sets. Match the number of reconstructed hypotheses and pose refinements when comparing gating mechanisms. Larger K is explicitly more compute.

Potential trust features: independent fracture-contact residual, held-out exterior compatibility, change in overlap proxy, consistency under rerendering/camera changes, and disagreement between generated hypotheses. Agreement among samples is a heuristic, not calibrated probability or evidence of truth. All masks and candidate-contact choices are fixed or generated from inputs; an optimizer may not erase inconvenient points by zeroing their exterior/contact weights.

Stress with wrong-category shapes, plausible same-category distractors, wrong thickness and altered aspect ratios. Construct these controls without adapting them to the test answer. Evaluate rejection rate, coverage versus assembly error, damage prevented relative to unconditional use, and retained benefit on correct hypotheses. Report confidence calibration only if confidence has a defined probabilistic meaning; reference-label calibration would be a separate supervised-development arm.

**Gate:** reduce the baseline-successful-to-failure rate by at least 50% relative to unconditional guidance, without a statistically supported loss of assembly success versus B0. Expose the reject-all solution by reporting acceptance coverage and requiring the E4 improvement gate to remain satisfied. Do not optimize a gate on final-test labels.

### E6 — Does self-training learn beyond the teacher's mistakes?

**Question:** Can the successful inference mechanism supply useful training targets to an assembler?

Only start after E4 and E5 show useful inference on development objects. Create pseudo-poses for the 120-object unlabeled training pool using the frozen selected teacher. Log every accepted/rejected object and confidence. Keep oracle results in a separate namespace excluded by the training loader. Reapply fresh independent SE(3) transformations to training fragments and transform pseudo-labels consistently; these augmentations do not turn incorrect pseudo-labels into true labels.

Train one fixed architecture with identical initialization policy, three seeds, point budgets and update counts. Compare geometry-consistency-only training, training on all pseudo-labels, training on filtered pseudo-labels, and training on equally many randomly accepted pseudo-labels. The random-count control distinguishes filtering quality from reduced data volume. Add the frozen teacher as an inference baseline. Where data counts differ, report unique object exposure and repetitions as well as equal optimizer steps.

Stop at one round first. A second round is conditional and includes retaining the previous teacher or an independent view/point partition for evaluation; repeated agreement with the same teacher is not independent verification. Select checkpoints using a predetermined schedule or fragment-only monitor for the strict track.

**Measurements:** held-out assembly success, damage rate, pseudo-label quality versus acceptance coverage (evaluator only), label drift over rounds, and inference cost. A student that matches its teacher while being faster is useful distillation; claim improved accuracy only if it exceeds the teacher and geometry-only learner with source-paired evidence.

**Gate:** at least five percentage points higher held-out success than the geometry-only learner with a positive paired 95% interval, stable across three seeds, and lower pseudo-label error than the random-count control at matched coverage. Record whether this is accuracy gain, speed gain, both, or neither.

### E7 — Does the result survive realistic ambiguity and domain shift?

**Question:** Where does the method stop working?

Evaluate the frozen system by changing one factor at a time: two/three/five pieces; point noise 0/0.25/0.5/1% of the input-only scale; 0/25/50% point removal; one missing fragment where at least two remain; fracture-surface erosion; altered sampling density; and real scans. Describe occlusion and material-loss models separately. Report missing pieces by their area/volume when known: removing one tiny chip differs from losing half the object.

For missing fragments, evaluate observed-piece poses independently from missing-shape completion. Disconnected contact graphs and symmetric objects may admit multiple solutions. Return multiple candidates/abstention rather than score an invented unique pose as certain.

A category-transfer claim needs a category excluded from all task-specific training/development, with one outer holdout protocol frozen in advance. The initial three-category split alone is same-category generalization. Real scans without pose references support contact/consistency and expert plausibility checks, not quantitative ground-truth accuracy claims. Obtain a separate reference assembly or pre-break scan for real quantitative evaluation where practical.

## Common metrics and statistical rules

1. **Rigid geometry success:** on a fixed evaluation sample, require every fragment's unsquared symmetric mean nearest-neighbor distance to its reference placement to be at most 0.01 times the declared input-only scale. The metric is the average of the two directional means, not a sum of squared distances. Report a threshold curve at 0.005, 0.01 and 0.02; tiny pieces also need size-relative and rotation diagnostics. Anchor gauge is fixed consistently; do not independently align each predicted part to ground truth for scoring.
2. **Pose errors:** geodesic SO(3) rotation error and translation error, with valid object/part symmetries declared. Preserve fragment identity. An arbitrary global rotation is gauge, not assembly failure; arbitrary part swaps are not automatically equivalent.
3. **Shape metrics:** evaluate the assembled observed fragments and the generated exterior separately. Whole-object Chamfer can hide misplaced pieces. Complete-exterior fitting excludes true fracture faces in evaluator metrics, not by silently passing those labels to inference.
4. **Contacts and collision:** evaluate true interface fit independently when labels exist. Do not reward fitting only the solver's self-selected easy contacts. For watertight solids use an explicit volumetric intersection estimate and numerical tolerance. For open surfaces label contact/collision proxies and their limitations.
5. **Failures:** include all requested cases in success/failure denominators, including generation failures, no candidate, empty mesh, failed alignment and abstention. Conditional pose/shape means must state how many returned outputs they cover. Do not insert identity transforms as fake successes.
6. **Selection:** the reported result is the deployed input-only selected candidate. Best-of-K using ground truth is an oracle ceiling reported in another column. Keep generation, pose search and refinement budgets visible.
7. **Statistics:** average patterns and repeats within each original source, then average sources. Bootstrap paired source IDs (2,000 resamples) for confidence intervals. Views, fracture patterns and random seeds are not independent source objects. Prespecify the main B2–B0 comparison and label all remaining comparisons exploratory unless multiplicity is handled.
8. **Go/no-go thresholds:** the numerical thresholds above are suggested project decisions, not measured results or universal physical tolerances. Freeze them before the locked evaluation. If they are changed after seeing development truth, disclose the supervision; if changed after final evaluation, require a new untouched evaluation set.

## Minimal first batch and diagnostic interpretation

The first commitment is E0 plus the twelve-object E1 screen and E3 oracle utility test. Train no assembler yet. Budget up to 192 edited images, 132 image-to-3D reconstructions in E2, and a logged fixed set of local/global pose solves. Stop at the relevant gate before spending on all categories, seeds, views and architectures. Measure three representative jobs before estimating GPU-hours; no runtime claim is made here.

| Observed outcome | Interpretation and next action |
|---|---|
| Rendering/back-projection fails | Fix camera/scale/I/O before testing models. |
| Image models create convincing but incompatible objects | Improve conditioning/masks or retain multiple hypotheses; do not create pseudo-labels yet. |
| True complete image → 3D is poor | The reconstruction bridge is a bottleneck; evaluate the second 3D model. |
| True shape fails to improve pose estimation | Fix correspondences, candidate coverage or the pose objective. Better image generation cannot fix this alone. |
| Generated shape helps only with oracle registration or oracle candidate choice | The deployment alignment/selection problem remains unsolved. |
| Generic/wrong template helps equally | No demonstrated object-specific completion benefit. |
| Reranking helps, refinement harms | Keep reranking; redesign the refinement objective and trust region. |
| Inference helps, self-training does not | Report an inference method; the unsupervised-learning claim remains unsupported. |
| Only known-category or bottle cases work | Narrow the claim to that supported setting. |

## Run artifacts required for future execution

Each experiment writes a run manifest with protocol hash; code/checkpoint revisions and hashes; source split; supervision track; allowed inputs; input/view/camera/mask hashes; all prompts and seeds; model settings; point/candidate budgets; scale and coordinate transforms; image and 3D hypothesis IDs; input-only selection scores; gate decisions; final rigid matrices; failure status; and compute/memory/storage totals.

Keep `inputs`, `generated`, `predictions`, `pseudo_labels`, and `evaluator_only` separate, with oracle flags enforced by future loaders. Preserve all four screen generations so results cannot be curated after the fact. No result values belong in the planning manifest. The implementation should produce a per-source table and a failure gallery in addition to aggregate scores.

## References

- [GenPC paper](https://arxiv.org/abs/2502.19896) and [released implementation](https://github.com/GenPC-Team/GenPC).
- [Qwen-Image-Edit-2509 model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2509).
- [SDXL inpainting model card](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1).
- [SD 1.5 inpainting checkpoint](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting).
- [ControlNet SD 1.5 depth checkpoint](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth).
- [InstantMesh implementation](https://github.com/TencentARC/InstantMesh).
- [TRELLIS.2 implementation](https://github.com/microsoft/TRELLIS.2).
- [Breaking Bad dataset](https://breaking-bad-dataset.github.io/).
- [Jigsaw++ methods and oracle correspondence caveat](https://arxiv.org/html/2410.11816v2).
- [CRAG joint generation/assembly method](https://arxiv.org/html/2602.22629v3).

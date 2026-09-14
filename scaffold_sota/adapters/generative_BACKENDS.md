# Native generative recipient adapters

These adapters instantiate released model classes from the supplied pinned source directory. CPU contract tests verify data boundaries and control routing; native CUDA forward/backward, training quality and A5000 memory are not yet verified. Keep each model in a fresh process/environment because several upstream projects use conflicting top-level Python package names.

| Backend | Supported stages | Feature checkpoint | Scope |
|---|---|---|---|
| `garf` | `pretrain`, `assembly` | Assembly requires a study GARF pretrain checkpoint | Actual FracSeg/PointTransformerV3 and native SE(3) flow denoiser, scheduler and objective |
| `puzzlefusion_pp` | `pretrain`, `assembly` | Assembly requires a study PF++ VQVAE pretrain checkpoint | Actual native VQVAE and **denoiser-only** pose diffusion; no verifier/agglomeration claim |
| `diffassemble` | `assembly` | No separate pretrained component | Released `spatial_diffusion_3d_test_double_diffusion.py`, VN-DGCNN and native graph diffusion/loss |
| `gpat` | Standalone semantic `assembly` | Native GPAT can start fresh | Native target segmentation and bounding-box fitting; requires separate semantic/surface inputs, excluded from the bottle runner |

GARF/PF++ prerequisite pretraining accepts `condition=native` (or B0 at the backend boundary) and rejects supplied priors. Assembly accepts `native`, B0, B1, B2, B3, B4. Native/B0 have no conditioning branch. Other branches share a zero-gated `conditioner` and retain all original native state-dictionary names. `initialize_from_native` permits missing `conditioner.*` keys only. The first native transformer layer receives the residual context at every denoising step; DiffAssemble receives it after its feature-combination MLP and before graph mixing. GARF's direct `forward_sdpa` route is handled explicitly because it bypasses PyTorch module hooks.

The runner's checkpoint metadata must identify `model_name`, `stage`, and `model`. GARF pretrain state uses `feature_extractor.*`; PF++ pretrain uses `autoencoder.*`. The complete resulting assembly state embeds the frozen feature weights. Required feature checkpoints remain explicit lineage inputs even when loading an assembly checkpoint.

## Input adaptation and output frame

Bottle inputs are unbatched normalized, independently centered `points[F,N,3]`, `fragment_mask[F]`, and an observed `anchor_index`. All prediction code reads deployable keys only. Labels are accessed exclusively by `loss`. Active fragment indices are preserved; padding is returned as identity/zero only for inactive slots. An invalid active native quaternion fails with no pose arrays.

GARF and PF++ divide each observed fragment by its own maximum absolute coordinate and pass this scale separately. Native pose translations stay in the original shared normalized observation units. Sampling fixes the chosen observed anchor at identity; final returned poses are re-expressed relative to the predicted anchor. DiffAssemble receives shared-normalized fragment coordinates directly and is similarly re-expressed after native sampling. These adaptations replace native assembled-coordinate preprocessing with the study's declared input-only frame.

GARF additionally needs normals. If `sample.normals` exists it must contain finite, nonzero, aligned observed normals. Otherwise local PCA with 24 neighbors estimates normals from each input fragment, with a deterministic radial sign choice. This is an approximate XYZ-only adaptation applied to every condition; it does not use intact mesh normals, fracture labels or GT poses. It can differ from the original paper's mesh-normal input and must be reported. Fracture pretraining uses `fracture_labels` aligned with the sampled points; assembly preserves the native flow objective.

GARF keeps its extractor parameters and BatchNorm statistics frozen in assembly. The native FracSeg path uses internal CUDA fp16 autocast, so GARF CUDA training requires an enabled GradScaler (`train.amp=true`); `requires_amp=True` exposes this requirement. Geometry in sampling returns to float32 between steps. Both `encoder_flash` (default true) and `denoiser_flash` (default false) are explicit options. Disabling denoiser FlashAttention does not disable the backbone or remove the native package import requirement.

PF++ preserves native frozen VQVAE parameters and native train/eval handling of its BatchNorm buffers. `variant=denoiser_only` is required. Its VQVAE reconstruction layout and DiffAssemble's loss code hard-code 1000 points; configure **all paired conditions** to exactly 1000 points per fragment. No hidden truncation or resampling occurs inside these backends. The retained geometry remains rigid.

DiffAssemble uses native defaults: 300 diffusion steps, DDIM stride 10, START_X prediction, 4 graph layers and VN-DGCNN. The default `noise_weight=0` with identity initial rotations makes its nominal seeds deterministic; do not claim eight distinct samples merely from eight repeated calls. Changing noise weight is a separately recorded candidate-generation policy. The 6D variant is explicitly unsupported here. A private import-only namespace bypasses the release's missing, unrelated `backbone_vist` module; no upstream files, equations or network classes are replaced.

## GPAT surface-target protocol

This backend is separately callable with `dataset_track=partnet`; the bottle StudyDataset is unsupported. The caller supplies native preprocessed semantic `points[max_parts,1000,3]`, contiguous valid parts followed by padding, `fragment_mask`, `anchor_index`, and `dataset_track='partnet'`. Parts and target must already share the native common scale. `max_parts` defaults to 20.

Every call requires `prior.surface_points[5000,3]`. Signed-distance queries, near-surface query centers, or an array of field values are not a target surface export. The exporter must materialize the predicted surface and sample it under the fixed target budget. Conditions are `native`/B0 or named `target_clean`, `target_predicted`, `target_degraded`, `target_wrong`, `target_generic`. The supplied geometry is never replaced with a clean target behind the caller's back. There are no B1-B4 additional-input adapters for GPAT.

With `use_dense_target=true`, the caller must also supply replacement `prior.surface_points_dense[100000,3]`. The adapter recomputes its nearest-neighbor map to the supplied sparse target, ignoring stale external maps. With the option false, native sparse-target fitting is used. `optimize=true` selects native CMA refinement and must match across target conditions. A narrow native bug fix initializes correspondence segments containing exactly 1000 target points; upstream otherwise leaves those slots uninitialized. Empty native target segments retain native fallback behavior and are counted in prediction diagnostics.

Training needs `target_segmentation[5000]`, `equivalence_classes[max_parts]`, and `target_surface_sha256` in the sample. The digest is SHA256 of the supplied target's contiguous little-endian float32 coordinates. This binds assignment labels to the exact replacement surface; clean-target labels cannot silently accompany a different predicted target. Loss uses the released GPAT cross entropy and equivalence reassignment implementation.

GPAT output is marked `coordinate_frame='provided_target'` and preserves that native target frame, without bottle-anchor alignment. Use semantic equivalence-aware evaluation in a separate track. This study has not implemented the semantic dataset reader or implicit-surface exporter. The original GPAT Python 3.6 environment also needs a declared compatible environment for the modern study runner before runtime verification.

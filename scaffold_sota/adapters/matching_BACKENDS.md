# Matching and direct-regression adapter contract

These adapters construct the pinned native networks. No substitute assembly
network is available from a production `build()` function. Full native execution
requires the corresponding compiled CUDA dependencies; passing CPU contract
tests does not establish that a native model trains, fits this GPU, or produces
accurate poses.

Each module provides `build(source_root, options, device)`. The returned module
has `loss(sample, prior=None)` and `predict(sample, prior=None, seed=0)` methods.
`loss` returns a differentiable scalar under the `loss` key. `predict` consumes
only `points`, `fragment_mask`, `anchor_index` and the optional prior, and returns
standard column-action rotation matrices and translations relative to the input
anchor: `x_aligned = x @ R.T + t`. Failed inference returns `rotations=None` and
`translations=None`, together with a reason. Valid fragments are packed in their
original order; no padded fragment is exported.

The caller controls train/eval mode, optimizer, schedule, source copy, output
paths and serialization. `native.*` state keys hold the actual network weights;
the only additional parameter prefix is `conditioning.*`. Use independent
processes for different native source trees because upstream projects reuse
top-level Python package names. Bytecode creation is disabled before native
imports to preserve the read-only source boundary.

| Adapter | Supported conditions | Native components and explicit study policy |
|---|---|---|
| Jigsaw | native, B0, B1, B2, B3, B4 | Native encoder/attention/classifier/affinity/Sinkhorn/Hungarian, matching objectives, RANSAC and global alignment. GT pivot alignment is disabled. Disconnected graphs of edges with at least three matches fail explicitly instead of exporting native identity/translation-only fallback. |
| CCS | native, B0, B1, B2, B3, B4 | Native WXTransformer and shared-memory correlator, pose head and geometric losses. Twenty padded native slots retain top-10 attention dimensions; only valid fragment features/poses are used. Per-object transient memory resets prevent cross-object state leakage. One sample, zero pose-head noise; no GT min-of-N selection. |
| PMTR | native, B0, B1, B2, B3, B4 | Native KPConv preprocessing, FPN, coarse and fine matching, training objective and pair registration. A pre-hook conditions coarse superpoint features. Prediction omits the unconditional GT bookkeeping in native `forward_pass`, invoking the same input-derived modules directly. |
| CMNet | native, B0, B1, B2 | Native equivariant feature extractor, local bases, invariant shape/occupancy descriptors, channel attention, optimal transport and weighted fitting. B2 uses a rotation-invariant radial-distance field summary in channel attention. B3/B4 reject explicitly pending a principled pose-aware spatial design. |

Common options are `condition="native"`, `token_dim=128`, `heads=4`,
`num_tokens=512`. Field records contain `query_xyz[Q,3]`, `distance[Q]`,
`log_scale[Q]`, and `valid[Q]`; B2/B3 ignore uncertainty, B4 requires it. B1
uses an identically sized learned null bank and does not inspect prior content.
The field encoder sees detached inputs. B2 pools all spatial encodings before
exposing the repeated global descriptor to the recipient; it exposes no localized
positions. All branch conditions allocate the same parameters. The residual gate
initializes at zero, reproducing native features while allowing its gate to learn.

Additional options and deviations to record in every comparison:

- Jigsaw: `matching_loss_weight=1.0`, `rigid_loss_weight=0.0`, optional native
  `encoder` override. The standalone pilot starts joint classification/matching
  immediately; it does not run the upstream epoch callbacks that progressively
  enable matching and rigid losses. Use identical explicit weights across arms.
- CCS: `native_max_parts=20`, which must accommodate its native top-k value.
  Native padded-slot loss normalization remains intact. The scaffold branch is
  masked out at padded slots before the correlator.
- PMTR: `fine_matcher="pmt"`, `cpconv_radius=0.05`,
  `subsampling_radius=0.01`, `lr=0.001`; at least 128 points per valid fragment.
  Training selects a random source and the strongest-overlap target using
  training-only geometry, retaining the full object's scaffold. The upstream
  Lightning logging block is disabled; the independent runner owns logging.
- CMNet: `lr=0.001`. The same native source/strongest-overlap target training
  policy uses its 0.018 contact radius. Its B2 radial summary is a deliberately
  narrower global-context experiment, not equivalent to spatial B3/B4.

PMTR and CMNet retain their different native graph scores/edge selection and
use the pinned `estimate_poses_given_rot` helper with GTSAM Shonan rotation
synchronization. Study inference initializes Shonan from predicted anchor-pair
rotations rather than unseeded GTSAM randomness, identically across conditions.
The native direct-anchor fallback is retained and explicitly labelled in
diagnostics. Native exports use `R^-1(x+t)`; the adapter composes this into the
common matrix/translation convention. These are declared study inference
policies, not claims of bit-for-bit native benchmark reproduction.

CPU tests in `tests/test_matching_contracts.py` verify the conditioning controls,
gradient isolation, state branching, coordinate algebra, input-only dispatch,
critical-point failure policy and CMNet context invariance. Native-boundary spies
are test fixtures only. Real forward/backward, all compiled operators, full
prediction and memory/runtime preflight remain mandatory in each native GPU
environment before producing an experiment result.

# Data for the v2 fragment assembly pilot

The supported first experiment uses ShapeNet bottle meshes (category `02876657`)
and complete sets of two or three complementary pieces. Fractures are deterministic
procedural height-field cuts, **not physically simulated breakage**. No existing
training checkpoint is used to prepare data or initialize the new pipeline.

Run commands from the `pt_reg` repository directory. Install the dependencies in
`requirements.txt`, including `trimesh`, `manifold3d`, `rtree`, and
`huggingface_hub`. The model's PyTorch installation on the training host must
support its RTX A5000; preparation itself runs on CPU.

## Acquire only the bottle category

First check access, the current archive size, and local storage:

```bash
python -m reassembly acquire --config configs/reassembly_v2.yaml --managed-root "$HOME/reassembly_v2" --output "$HOME/reassembly_v2/acquisition" --dry-run
```

Then download the category archive:

```bash
python -m reassembly acquire --config configs/reassembly_v2.yaml --managed-root "$HOME/reassembly_v2" --output "$HOME/reassembly_v2/acquisition"
```

Only `02876657.zip` from the official
[ShapeNet/ShapeNetCore repository](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)
is requested. The downloader checks metadata and exact bytes first, pins the
resolved revision, streams to a partial file, and validates the ZIP before making
the completed archive available. It never downloads the entire ShapeNet release.
The report records the revision, size, and downloaded-file SHA-256. Account tokens
are neither reported nor forwarded to a different download host.

The implementation-host access probe returned **HTTP 401**. To acquire data on
the training host, use an authorized Hugging Face account, accept the dataset's
access terms, and authenticate there (`hf auth login`). Re-run the dry-run command
to confirm access. This is an account/data-access requirement; the program cannot
grant access. An already obtained category ZIP or extracted mesh directory can be
prepared directly without using the downloader.

`acquisition.json` records failures as well as successful access checks. A failed
download keeps its `.zip.part` file for inspection; the downloader does not silently
overwrite either a partial or mismatched completed archive.

## Prepare a bounded geometry pilot

```bash
python -m reassembly prepare --config configs/reassembly_v2.yaml --managed-root "$HOME/reassembly_v2" --source "$HOME/reassembly_v2/acquisition/02876657.zip" --output "$HOME/reassembly_v2/prepared"
```

Replace `--source` with an existing ZIP or mesh-directory path when appropriate.
The default preparation inspects at most 100 source meshes and attempts eight
patterns per source, with at most 20 candidate cuts per requested pattern. These
limits are YAML fields under `data`; they are not download or extraction limits.
A pre-existing prepared `manifest.json` is never overwritten: choose a new output
directory for a different preparation configuration.

The ZIP is read one object at a time without extracting textures or the full
category tree. When a category tree contains several ShapeNet representations,
`models/model_normalized.obj` is preferred so one object is not counted repeatedly.
Explicit directory inputs are useful for synthetic correctness fixtures; production
pilot inputs must still be the requested bottle category.

Cleanup merges coincident vertex seams before removing duplicate/degenerate
triangles and unused vertices. This ordering handles ShapeNet's double-sided OBJ
faces stored with separate indices. Existing faces on a closed, connected surface
are then oriented consistently outward; the removed/reoriented face counts are
recorded in source metadata. No face is added or surface moved by orientation
repair. A source must then be connected, watertight, consistently wound,
and have positive volume. Invalid objects are rejected with a reason. The code does
not fill holes or substantial cavities, replace geometry with a convex hull, or
keep only the largest component. A normalized copy of each accepted original mesh
is retained, together with the centering/scale needed to recover its source units.

Each cutter is a closed solid with a smooth triangulated height-field boundary.
Manifold's intersection and difference produce the two complementary pieces.
Three-piece cases split the largest current piece once more. Face provenance
survives both Boolean operations: original surface faces have interface ID `-1`,
while each cut receives a nonnegative interface ID shared by its mating surfaces.
Every piece is sampled independently, so there are no deliberately duplicated
points across a mating interface.

Every accepted pattern satisfies:

- Exactly two or three connected, watertight, consistently wound pieces.
- Source volume preserved and pairwise interior overlap below the configured
  relative tolerance (`2e-5` by default).
- A connected contact graph and nonempty original/fracture surface provenance.
- Minimum piece-volume fractions of 25%, 15%, and 10% for easy, intermediate,
  and hard examples; easy two-piece cases also remain below 75% per piece.

Easy cuts favor central, broad, mildly curved contacts. Intermediate cuts introduce
three pieces and greater shape variation. Hard cuts favor peripheral contacts,
stronger curvature, and small XYZ observation noise. Their measured contact areas,
volume fractions, cutter parameters, and acceptance diagnostics are recorded.
Planar and symmetric cuts are explicit controls, not assumed to be easy examples.

## Splits, targets, and training samples

Candidate source IDs are allocated deterministically in 80/10/10 proportions to
train/validation/test **before generating any fractures**. Geometry rejection may
change the eventual accepted split proportions; reports expose actual counts.
Every fracture of a source stays in that source's partition. An additional radial
cut family is created only for test sources and exposed through `cut_holdout`.
It is never included in training or validation.

The output format is version 2:

- `manifest.json`: source assignments, pattern paths, content hashes, a dataset
  fingerprint, preparation settings, actual split counts, rejection reasons, and
  the `learning_ready` result.
- `sources/<id>.npz`: original source mesh, intact surface samples, near-surface
  and surrounding-space SDF queries, signed distances, and sampling masks. These
  targets are stored **once per source**, shared by all its fracture patterns.
- `patterns/<id>.npz`: independent fragment XYZ reservoirs, per-point fracture
  labels and interface IDs, plus provenance/cutter/quality metadata.

`reassembly.data.verify_manifest` checks source and pattern SHA-256 hashes,
recomputes the fingerprint from preparation settings and split assignments, and
rejects referenced paths outside the prepared directory. Preflight performs this
complete bounded-memory scan once; ordinary minibatch reads do not rehash every
source on each access.

Signed distance is negative inside the original solid and positive outside,
including within preserved cavities. Near-surface and surrounding-space queries
each supply half of training queries. SDF evaluation is chunked during preparation.
The true retained source mesh is available only to evaluation's GT-scaffold control;
it is never an inference input.

`FractureDataset` applies independent uniform SO(3) rotations, translations, and
point permutations at runtime. Fragment centroids are removed separately and all
fragments share one scale: the sum of their centered bounding radii. The greatest
RMS-radius fragment is the reference, with index order breaking exact ties. All
canonical points and source supervision are transformed into this observed
reference frame. Ground-truth transforms follow `x_aligned = x @ R.T + t`; the
reference transform is identity. These normalized transforms are composed with
original centroids and the common scale when exporting inference matrices.

Training adds intermediate examples at 30% of the update budget and hard examples
at 65%, while retaining earlier bands. Complete fragments are never dropped.
Evaluation and fixed-overfit samples are reproducible independent of traversal
order and current training step. All arrays remain padded to three fragments,
with an explicit fragment mask; a complete two-piece example is not interpreted
as missing data.

## Yield and resource gates

Preparation always writes its yield/rejection report. Fewer than 30 accepted
source objects sets `learning_ready=false` and returns a nonzero CLI status; this
does not fabricate held-out learning results. Inspect the rejection reasons before
obtaining different sources or revising preparation. Tiny geometry fixtures can
override counts for tests, but the supported pilot defaults remain 100 candidates
and a minimum of 30 accepted sources.

The managed root defaults to `~/reassembly_v2`, never `/data`. The CLI checks
resolved filesystem locations and counts managed acquisition, preparation, and
run artifacts together. Defaults cap them at 40 GiB and preserve at least 50 GiB
free. Use `--managed-root` to choose a suitable location on the root filesystem;
do not assume a home directory is on the root volume without checking `df -h`.

After successful preparation, profile the full model and backward pass on the
A5000 using this exact prepared dataset:

```bash
python -m reassembly preflight --config configs/reassembly_v2.yaml --managed-root "$HOME/reassembly_v2" --manifest "$HOME/reassembly_v2/prepared/manifest.json" --output "$HOME/reassembly_v2/preflight" --device cuda
```

The preflight must satisfy the strict GPU-memory limit before the fixed-overfit
and bounded learned-stage pilots. Successful geometry preparation alone does not
establish that the learned scaffold improves assembly.

## Breaking Bad transfer boundary

`reassembly.prepare.select_complete_breaking_bad_sets(raw_root)` provides a
read-only inventory of already decompressed Breaking Bad fracture directories.
It selects whole sets containing exactly two or three piece meshes and skips
larger sets. It never truncates a larger set to fit the model's fragment count.

This inventory is not yet a v2 training adapter: Breaking Bad's intact-source SDF
and fracture/interface supervision need explicit validation before conversion.
The default `prepare` command generates the specified controlled bottle experiment.
The legacy Breaking Bad loader and separate SOTA reproduction packages are not
used by this workflow.

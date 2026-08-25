# Data acquisition and 20 GiB protocol

Set `SOTA_DATA_ROOT` to a persistent Linux path and use a separate temporary scratch volume for archive extraction. Credentials and raw archives must not be stored in this repository.

## Breaking Bad

Download the public compressed data and official split files:

```bash
python -m sota_repro.download_breaking_bad --output-root "$SOTA_SCRATCH/breaking_bad_archives" --subset both
```

Extract the archives into scratch, clone the official Breaking Bad data repository, and create the source manifest. The first planning pass may use more than 20 GiB of scratch because the manifest needs the authors' decompressed meshes to count 2--20-piece patterns exactly. It does not retain that expansion.

```bash
python -m sota_repro data plan \
  --breaking-bad-root "$SOTA_SCRATCH/breaking_bad" \
  --partnet-gpat-root "$SOTA_SCRATCH/gpat/dataset/partnet" \
  --output "$SOTA_SCRATCH/common_v1_source_manifest.json"
python -m sota_repro data materialize \
  --manifest "$SOTA_SCRATCH/common_v1_source_manifest.json" \
  --output "$SOTA_DATA_ROOT"
```

For the retained corpus, do **not** copy a fully decompressed Breaking Bad tree. Instead, the materializer stages just the object IDs in the manifest as links under a disposable directory, invokes the authors' `decompress.py` category-by-category, verifies each content hash, copies only selected patterns, and removes the stage directory.

```bash
git clone https://github.com/Breaking-Bad-Dataset/Breaking-Bad-Dataset.github.io.git "$SOTA_SCRATCH/breaking-bad-official"
python -m sota_repro data breaking-bad-materialize \
  --manifest "$SOTA_SCRATCH/common_v1_source_manifest.json" \
  --compressed-root "$SOTA_SCRATCH/breaking_bad" \
  --official-repository "$SOTA_SCRATCH/breaking-bad-official" \
  --output-root "$SOTA_DATA_ROOT" \
  --scratch-root "$SOTA_SCRATCH/staging" \
  --python /opt/conda/envs/breaking-bad/bin/python
```

`--python` must be the authors' Python 3.8 decompressor environment (NumPy, SciPy, tqdm, libigl, and gpytoolbox). The official decompressor supports `--subset everyday --category CATEGORY`; the suite supplies one category at a time. Once `data materialize` has copied the selected PartNet records and verified the 20 GiB cap, remove `$SOTA_SCRATCH/breaking_bad` archives and temporary raw expansion. The source manifest reports exact object IDs and pattern IDs, source content hashes, split, byte count, and preprocessing version; retain it with results.

`breaking-bad-materialize` writes the verified selected-only manifest to `$SOTA_DATA_ROOT/common_v1_manifest.json`; do not replace it with the pre-materialization source manifest.

## PartNet / GPAT

PartNet is gated. Accept its license using the required ShapeNet account, authenticate with Hugging Face, download the PartNet v0 annotation archive into scratch, and run the official GPAT preprocessor there. Follow `../docs/PARTNET_GPAT_VM.md` for the compatible Python 3.6/Torch 1.10/PyTorch3D environment.

After preprocessing, pass the generated `dataset/partnet` directory to `sota_repro data plan`. The selected GPAT point clouds retain `parts.npy`, `target.npy`, `poses.npy`, labels, and equivalence classes. Delete the raw archive and unselected preprocessed tree once materialization verifies the 20 GiB retained-data cap.

### Breaking Bad-only option (no PartNet archive)

If the gated PartNet v0 archive cannot be staged, create a manifest for the seven Breaking Bad models only. This deliberately excludes GPAT and is labelled `common_v1_breaking_bad_only` in the output so it cannot be mistaken for the full two-track protocol:

```bash
python -m sota_repro data plan \
  --breaking-bad-root "$SOTA_SCRATCH/breaking_bad" \
  --without-partnet \
  --output "$SOTA_SCRATCH/breaking_bad_only_source_manifest.json"
```

## Native views

`python -m sota_repro data view --model MODEL --data-root "$SOTA_DATA_ROOT" --scratch-root "$SOTA_SCRATCH/views"` creates symlink-based source layouts one model at a time. Delete the scratch view after a run; the retained corpus itself is never duplicated.

```bash
python -m sota_repro data clean-view --model MODEL --scratch-root "$SOTA_SCRATCH/views"
```

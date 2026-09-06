# Reduced-data SOTA assembly suite

This package evaluates source-backed fractured-object and semantic-part assembly models on one deterministic retained corpus capped at 20 GiB. It is designed for a Linux CUDA VM; the local Windows workspace is for orchestration and tests only.

```bash
cd pt_reg
python -m sota_repro bootstrap
python -m sota_repro data plan --breaking-bad-root "$SOTA_SCRATCH/breaking_bad" --partnet-gpat-root "$SOTA_SCRATCH/gpat/dataset/partnet" --output "$SOTA_SCRATCH/common_v1_source_manifest.json"
python -m sota_repro data materialize --manifest "$SOTA_SCRATCH/common_v1_source_manifest.json" --output "$SOTA_DATA_ROOT"
python -m sota_repro setup --model cmnet
python -m sota_repro train --model cmnet --data-root "$SOTA_DATA_ROOT" --run-root "$SOTA_RUN_ROOT"
python -m sota_repro test --model cmnet --data-root "$SOTA_DATA_ROOT" --run-root "$SOTA_RUN_ROOT" --checkpoint /path/to/model.ckpt
python -m sota_repro test --model cmnet --track breaking_bad_artifact --data-root "$SOTA_DATA_ROOT" --run-root "$SOTA_RUN_ROOT" --checkpoint /path/to/model.ckpt
```

Each `models/<model>/` directory contains a source clone created by `bootstrap`, non-invasive configuration/output folders, and direct `setup_env.sh`, `smoke.sh`, `train.sh`, and `test.sh` wrappers. See [data instructions](docs/DATA.md) before downloading data.

Without the gated PartNet archive, add `--without-partnet` to `data plan` and run the seven Breaking Bad models only; GPAT is unavailable for that Breaking Bad-only protocol.

For DiffAssemble dependency failures, use the [pinned environment recovery guide](models/diffassemble/README.md).
Its setup now creates `sota-diffassemble-repro-v1`; its smoke performs native batch checks, not just `--help`.
This audited DiffAssemble entry supports everyday validation only, not artifact evaluation or automatic common prediction export.

Native data views are created in scratch per model and removed automatically after model execution. Each train, test, or smoke command runs from a per-run source copy under its immutable run directory, so native log/checkpoint/cache behavior cannot alter the pinned `upstream/` clone. Use `--keep-native-view` only when debugging a native loader.

See [the audited model list](docs/MODELS.md) for model-specific native entry points and checkpoint guidance. After native evaluation, normalize its transform export before scoring:

```bash
models/cmnet/export_predictions.sh --track breaking_bad_everyday --input native_predictions.json --output predictions.jsonl
python -m sota_repro evaluate --predictions predictions.jsonl --output evaluation.json
```

The generated result report intentionally labels metrics as **reduced-subset comparisons**, rather than claiming paper-level reproduction.

PuzzleFusion++ uses two learned stages at inference; its test command requires both `--checkpoint DENOISER.ckpt` and `--verifier-checkpoint VERIFIER.ckpt`.

Use `--track breaking_bad_artifact` for a model's artifact zero-shot protocol and `--track partnet_gpat` for its semantic protocol. Checkpoint resume is passed through for DiffAssemble, CCS, PMTR, GARF, and CMNet via `--resume`; Jigsaw, PuzzleFusion++, and GPAT retain their upstream configuration-specific resume mechanisms.

For a model-specific prepared representation (for example, GARF's HDF5), pass `--native-data "$SOTA_DATA_ROOT/garf/breaking_bad_vol.hdf5"`; the common retained corpus remains the source of record.

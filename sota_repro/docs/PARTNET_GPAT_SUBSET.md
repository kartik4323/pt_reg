# Gated PartNet subset guide

1. Obtain PartNet v0 access using the official portal/ShapeNet account.
2. Download the `data_v0_chunk.*` archive and PartNet metadata into temporary storage only.
3. Clone the pinned GPAT source with `python -m sota_repro bootstrap --model gpat` and run its official `dataset/preprocess.py` in a GPAT environment.
4. Run `sota_repro data plan` with the preprocessed GPAT directory. It keeps only chair, lamp, faucet, table, and display samples with 2–20 parts, preserving the official split and `eq_class.npy` data.
5. Materialize the selected samples, confirm `common_v1_manifest.json` is within its 4 GiB PartNet allocation, then remove the temporary raw archive and unselected preprocessed data.

The retained data is intentionally insufficient to claim the full GPAT paper result; it supports a deterministic, comparable reduced-data experiment.

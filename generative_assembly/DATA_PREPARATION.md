# Prepare fragment data for the generative assembly study

Run these commands from the `pt_reg` repository root. `import-v2` uses NumPy/SciPy and supports the existing Python 3.8 preparation environment; it does not require Torch, Diffusers or a new model installation. Use the separate Python 3.10+ study environment and its core/model requirements for the full experiments. The Git repository contains code, not the prepared datasets or model weights.

If import reports `AttributeError: 'PosixPath' object has no attribute 'is_relative_to'`, update with `git pull --ff-only` and rerun the same import command. The path guards now use the Python 3.8-compatible `relative_to` operation, retaining escape checks. In the previous version, this particular error occurred before any fragment output was written, so no data cleanup or regeneration is needed.

## Which data is used?

- `import-v2` reuses the original prepared point reservoirs from the reassembly/repair/scaffold study. It does not generate new fracture shapes or require old trained checkpoints.
- It writes a separate dataset, applies independently sampled rigid poses with a fixed seed, and preserves source/pattern IDs and source-level split membership. These are fixed new poses of existing fragments, not an exact replay of another pipeline's per-epoch pose augmentations.
- The public input contains only observed fragment points and optional input normals. Complete objects, poses and true fracture labels are placed under `evaluator_only/` for scoring and explicitly marked oracle controls; normal inference and student training do not load them.
- `val` becomes `dev`, and `cut_holdout` joins `test` while its original split and cut family remain in each case's metadata. Test and cut-holdout patterns from one source remain correlated.
- `demo` generates two-piece box partitions only for software smoke checks. The initial retained validation used six such sources. It did not use ShapeNet bottles or run pretrained generation.

## 1. Reuse the existing Linux server dataset

The previous bottle workflow commonly used `$REASSEMBLY_ROOT/bottles498/prepared/manifest.json`. Check that directory on the server, or discover the actual path with the existing standard-library tool:

```bash
python -m scaffold_sota find-data --search-root "$HOME"
```

Use the original prepared manifest, together with its `sources/` and `patterns/` directories. A diagnostic/results archive cannot replace it. After identifying the correct manifest:

```bash
# Set this to the path reported by discovery if yours differs.
GA_SOURCE_MANIFEST="${REASSEMBLY_ROOT:-$HOME/reassembly_v2}/bottles498/prepared/manifest.json"
GA_DATA_ROOT="$HOME/generative_assembly/bottles498-inputs-v1"
test -f "$GA_SOURCE_MANIFEST" || { echo 'Set GA_SOURCE_MANIFEST to the actual prepared manifest'; exit 1; }
python -m generative_assembly import-v2 --input "$GA_SOURCE_MANIFEST" --out "$GA_DATA_ROOT" --seed 42
```

The importer verifies all referenced asset hashes and rejects a source crossing training/development/test splits. It refuses to overwrite an existing `dataset.json`; reuse the resulting path for later runs instead of re-importing.

To inspect what was prepared:

```bash
python -c 'import collections,json,sys; d=json.load(open(sys.argv[1])); print("patterns:",dict(collections.Counter(c["split"] for c in d["cases"]))); print("sources:",{s:len({c["source_id"] for c in d["cases"] if c["split"]==s}) for s in ("train","dev","test","real")}); print("provenance:",d["provenance"])' "$GA_DATA_ROOT/dataset.json"
```

Pass `$GA_DATA_ROOT/dataset.json` to the `--dataset` argument in the [experiment run guide](RUNBOOK.md). Use a separate `--root` for generated images, reconstruction outputs, training checkpoints and results. Existing original files are read only.

## 2. Reuse the verified local Windows dataset

On this machine, the existing prepared dataset is at `D:/reassembly_v2/bottles498/prepared/manifest.json`. From PowerShell:

```powershell
python -m generative_assembly import-v2 --input 'D:/reassembly_v2/bottles498/prepared/manifest.json' --out 'D:/generative_assembly/bottles498-inputs-v1' --seed 42
```

This dataset has 73 accepted sources and 546 patterns: 418 train, 48 development, and 80 combined test/cut-holdout patterns. Its original fingerprint is `9e6a9677a1688b579ffe1ab205fddd0e4fa0815ea48332b9c9bfc6417d29b0d1`. The historical VM dataset used fingerprint `b7f2b84a20964276c894300a7ece4d8f97f232903bace0f14131535594462043`; matching counts do not make those byte-identical datasets. For comparison with VM runs, import the actual VM dataset and retain its recorded fingerprint.

## 3. Prepare from an existing bottle archive only when no prepared dataset exists

If you already have prepared data, skip this section. Regeneration can change patterns/fingerprints and does not reproduce another study's exact data.

The repository's existing preparer generates complementary fractures from the source meshes; use its expanded bottle configuration and a fresh managed directory:

```bash
GA_PREP_ROOT="$HOME/generative_assembly/source-preparation-v1"
GA_BOTTLE_ARCHIVE="$HOME/reassembly_v2/sources/02876657.zip"
python -m reassembly prepare --config configs/reassembly_v2_bottles498.yaml --managed-root "$GA_PREP_ROOT" --source "$GA_BOTTLE_ARCHIVE" --output "$GA_PREP_ROOT/prepared"
python -m generative_assembly import-v2 --input "$GA_PREP_ROOT/prepared/manifest.json" --out "$HOME/generative_assembly/bottles498-inputs-v1" --seed 42
```

Run these one at a time and stop if preparation fails. Preparation needs the original reassembly geometry dependencies and sufficient free space; follow [REASSEMBLY_DATA.md](../docs/REASSEMBLY_DATA.md) and [REASSEMBLY_SERVER.md](../docs/REASSEMBLY_SERVER.md) for environment setup and source acquisition. Use the archive you already obtained; no new download is required when it is available.

## 4. Import your own broken pieces or Breaking Bad fragments

Create the JSON specification shown in section 2 of [RUNBOOK.md](RUNBOOK.md), then run:

```bash
python -m generative_assembly import-spec --input /absolute/path/import-spec.json --out /absolute/path/ga-fragments --seed 42
```

Use `canonical_fragments:false` for pieces scanned in unknown poses: the importer preserves those observed coordinates. Use `true` only for fragments already in a shared correct assembly frame that should receive random poses. Complete meshes and reference poses are optional for real inference; accuracy remains unscored when no references are available.

## 5. Generate smoke data only

```bash
python -m generative_assembly demo --out /absolute/path/ga-smoke-data --sources 8
```

This produces box partitions and is not a fracture benchmark. Historical bottle test sources have already been examined in other studies, so their reused results are exploratory; use a fresh source-object holdout for confirmatory research claims.

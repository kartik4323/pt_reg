# Source-backed model registry

`models.lock.yaml` is the executable source of truth. The following models are included because a maintained official repository was available at the recorded audit revision.

| Folder | Primary protocol | Official entry points | Checkpoint guidance |
|---|---|---|---|
| `jigsaw` | Breaking Bad | `train_matching.py`, `eval_matching.py` | Official README Google Drive link |
| `diffassemble` | Breaking Bad | `puzzle_diff/train_3d.py` | Train from source; preserve native Lightning checkpoints |
| `ccs` | Breaking Bad, PartNet | `scripts/train.py`, `scripts/test.py` | Supply `--weight` from a run or author checkpoint |
| `pmtr` | Breaking Bad | `main.py --mpa`, `test_mpa.py` | Official README Google Drive link |
| `puzzlefusion_pp` | Breaking Bad | VQ-VAE, denoiser, verifier, inference stages | Supply both the denoiser (`--checkpoint`) and verifier (`--verifier-checkpoint`) checkpoints |
| `garf` | Breaking Bad | `train.py`, `eval.py` | Model zoo in the official README; the lock records its exact Python 3.12.3 / CUDA 12.8 stack |
| `cmnet` | Breaking Bad | `main.py --mpa`, `multi_part_assembly.py` | Official README Google Drive link |
| `gpat` | PartNet | `learning/gpat/run.py`, `learning/assembler.py` | Official README Google Drive link |

`jigsaw_pp` and `fragmentdiff` remain deliberate exclusions: their papers are recorded in the lockfile, but no maintained official implementation was verified during the audit. They can be promoted later by adding a repository URL, full commit SHA, environment specification, commands, and adapter contract.

Every official source remains detached and read-only in `models/<name>/upstream`. Put local scripts/configuration in the containing model folder; `bootstrap` rejects an upstream tree with a non-empty `git status --porcelain`.

## Model-specific data notes

- **Jigsaw, DiffAssemble, PMTR, and CMNet:** the launcher creates filtered, selected-only Breaking Bad split and pattern lists in the disposable native view. PMTR and CMNet receive that view through `--datapath`; Jigsaw and DiffAssemble receive the layout they expect through a link inside the per-run source copy.
- **PuzzleFusion++:** training first generates point-cloud caches from the selected mesh view, then removes those scratch caches when the run completes. Its verifier and full inference also need the authors' matching/verifier archives documented in the pinned upstream `docs/data_preparation.md`; stage them in the disposable native view as `matching_data/` and `verifier_data/` after selecting the same IDs. The denoiser/VQ-VAE stages need no retained cache.
- **GARF:** build the source-native Breaking Bad HDF5 cache in the disposable model view; use its `data.data_root` Hydra override. The exact CUDA 12.8 `uv` dependency set is pinned in the lockfile.
- **CCS:** its PartNet reader uses the paper's `shape_data`/contact-point preprocessing, rather than GPAT's `parts.npy` directory layout. Run the CCS upstream PartNet preprocessor in scratch and retain only the IDs in `common_v1_manifest.json`; preserve the conversion log beside the run manifest.
- **GPAT:** the launcher links the selected GPAT preprocessor output to its required `dataset/partnet` path and preserves equivalence classes for the shared evaluator.

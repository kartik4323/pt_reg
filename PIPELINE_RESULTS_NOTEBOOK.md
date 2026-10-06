# E0–E7 results notebook

Open `pipeline_stage_results.ipynb` from the repository root or `pt_reg/`.
Both copies use `pt_reg/pipeline_results_viewer.py`. This helper deliberately
lives outside `generative_assembly/`, so adding it does not change the package
hash used to resume existing experiments.

Set `RUN_PATHS` in the first code cell to individual run directories or parent
groups containing runs. You can include scaling, imagination, E4–E7, and nested
parallel shard roots together. Alternatively set `GA_RESULTS_ROOT`, or set
`GA_ROOT` and the notebook will inspect `$GA_ROOT/runs`.

Required packages: `numpy`, `pandas`, `matplotlib`, `Pillow`. Interactive 3D uses
`plotly`; the case browser uses `ipywidgets`. Set `INTERACTIVE_3D=False` for
Matplotlib views or use the manual case cell if widgets are unavailable.
The notebook does not install packages automatically.

The case browser exposes input RGB/depth/masks, every saved completion,
raw/aligned reconstructed point clouds, initial/final fragment assemblies,
gate diagnostics, training metadata, and corruption results. Images are
embedded; Plotly JS is embedded for offline output. Generated priors are shown
separately from colored measured fragments. Large clouds are deterministically
subsampled only for visualization.

Metrics come from saved `evaluation/<split>/metrics.jsonl` and `summary.json`.
Running jobs can be browsed before evaluation, but are not given invented
scores. A lock is displayed as **locked / liveness unknown**. Use **Refresh
files** or rerun the loading cell after new outputs arrive. The notebook does
not call inference, evaluator commands, training, unlock, or export.

Oracle jobs and synthetic smoke runs are excluded from research plots by
default. Toggle them deliberately for diagnostics. Reference geometry is
loaded only when **Show evaluator reference** is enabled or the manual cell
uses `show_reference=True`. When downloaded artifacts need a different dataset
path, add `DATASETS = {"/local/run/path": "/local/data/dataset.json"}`.

Status and outcome tables count observed jobs only; missing/unstarted jobs
cannot be inferred without a study request manifest. Failures and unavailable
pose outputs stay visible. Reference-free real scans remain unscored. Saved
paired comparisons retain the evaluator's original denominators; these are
displayed as saved evidence, not recomputed confirmation gates.

The E7 rotation preview uses base observations because saved predictions have
already been mapped back from augmented coordinates. Its augmented template
overlay is omitted to avoid combining incompatible frames. The existing
erosion condition is identified as a point-removal proxy, not physical erosion.

E6 checkpoint files are never deserialized. Training progress comes from
learning-curve JSONL and training JSON. Pseudo-label acceptance is shown
alongside evaluator-only label quality where available.

The repository contains an optional software-validation fixture at
`deliverables/generative_fragment_assembly_experiments/implementation_validation/run`.
Point `RUN_PATHS` there and set `INCLUDE_SMOKE=True` to try all stage views;
those outputs are **not research results**.

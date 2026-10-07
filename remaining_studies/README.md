# Parallel E4–E7 studies

These runners extend the existing `generative_assembly` engine without modifying
its Python files or the code hash of existing imagination/scaling studies.
Every condition/shard has its own immutable run root. All research runs use the
full eligible dev split; test data are never used for architecture selection.

## Launch on the GPU server

Use a separate checkout on the server while the existing studies continue.
Keep existing run directories and source code snapshots. Activate an environment
with the engine's existing dependencies. Use the locked config that specifies
the image/InstantMesh environment paths.

```bash
git clone https://github.com/kartik4323/pt_reg.git "$HOME/pt_reg-e4-e7"
cd "$HOME/pt_reg-e4-e7"
# Activate your existing assembly Python environment before continuing.
export GA_ROOT=/home/kpandey/generative_assembly
export GA_CONFIG="$GA_ROOT/configs/sd15-v100-locked.json"
export GA_DATA="$GA_ROOT/bottles498-inputs-v1/dataset.json"
export GA_RUN_GROUP="$GA_ROOT/runs/remaining-e4-e7-v1"
export GA_GPUS="1 2"  # replace with GPUs free on your server
export GA_EXCLUDE_GPUS="0"  # e.g. GPU running imagination
export GA_CPU_WORKERS=2
export GA_SOURCE_RUNS="$GA_ROOT/runs/imagination-v2-YYYYMMDD/exterior"

bash run_remaining_studies.sh --dry-run
bash run_remaining_studies.sh
```

Keep this shell's environment settings for status/resume. The suite command
already schedules E4/E5/E7 work in parallel and starts E6 after its gates pass;
do not launch the individual stage wrappers into this same suite root.

In another shell with the same Python environment and `GA_RUN_GROUP`:

```bash
cd "$HOME/pt_reg-e4-e7"
python -m remaining_studies status --root "$GA_RUN_GROUP"
```

After the suite stops, aggregate its saved results and start the notebook:

```bash
python -m remaining_studies aggregate --root "$GA_RUN_GROUP"
python -m pip install jupyterlab numpy pandas matplotlib Pillow plotly ipywidgets
export GA_RESULTS_ROOT="$GA_RUN_GROUP"
python -m jupyter lab pipeline_stage_results.ipynb --no-browser --ip=127.0.0.1 --port=8888
```

From your local computer, forward the notebook port with
`ssh -N -L 8888:127.0.0.1:8888 USER@SERVER`, then open the localhost URL/token
printed by Jupyter. In the notebook use **Run All**. The first cell reads
`GA_RESULTS_ROOT`; to include existing imagination and scaling studies too,
set `RUN_PATHS` to their run groups plus `GA_RUN_GROUP`.

Source-run paths can be individual runs or parent groups. `GA_SOURCE_RUNS` uses
space-separated paths; use the Python CLI for paths containing spaces. GPUs
must be explicitly assigned, unique, and unoccupied at launch. The scheduler
never stops existing GPU jobs. Resource locks coordinate simultaneous suites
on the same host. `GA_RESOURCE_LOCK_DIR` can select a shared local lock folder.

On a shared server, `--allow-shared-gpu` explicitly permits existing compute
processes on the selected GPUs (it does not infer availability from memory or
utilization). For example, with GPU 0 selected:

```bash
export GA_GPUS="0"
export GA_EXCLUDE_GPUS=""
bash run_remaining_studies.sh --allow-shared-gpu --dry-run
bash run_remaining_studies.sh --allow-shared-gpu
```

This retains suite resource leases and records sharing in the immutable plan,
task metadata and `gpu_preflight.json`. B5 outputs remain available but are
marked invalid for isolated timing comparisons and cannot claim a compute
match. Use the same flag on resume. Changing sharing policy requires a new root.

```bash
python -m remaining_studies plan --base "$GA_CONFIG" --dataset "$GA_DATA" \
  --root "$GA_RUN_GROUP" --gpus 1 2 --exclude-gpus 0 --cpu-workers 2 \
  --source-runs /path/to/run1 /path/to/run2
python -m remaining_studies status --root "$GA_RUN_GROUP"
python -m remaining_studies aggregate --root "$GA_RUN_GROUP"
```

`plan` / `--dry-run` do not create a study. They print source counts, unavailable
conditions, dependency edges, per-task budgets and generation/reconstruction
upper bounds. Runtime is unknown until real jobs have been measured.
Compatible historical E1/E2 measurements (at least three jobs) provide rough
work-hour estimates; these are not parallel wall-time guarantees and require
checking hardware compatibility. Category-informed versus generic prompting
is inherited from the supplied config and stays recorded in each study identity.
Use `--python /path/to/python` for the orchestration worker interpreter;
`images.python` and `reconstruction.python` remain the isolated model workers.

Stage runners are `run_stage4_study.sh`, `run_gating_study.sh`,
`run_self_training_study.sh`, and `run_robustness_study.sh`. E4/E5/E6 runners
include their prerequisite teacher/gating work. E7 CPU baselines can begin
alongside generation. The E4 wrapper accepts the historical `--full` flag;
it now always uses all eligible development cases.

## What runs

- E4 uses N=4096, shared candidate banks, B0–B6, F/A and SD 1.5 with/without
  depth. One-factor sweeps cover template weights 0/.03/.3/1, refinement
  0/25/100, and alignment starts 8/32. Missing priors return an explicitly
  recorded geometry fallback. B3 is same-category; wrong-category is separately
  labelled and unavailable when there is only one category.
- E5 uses nested seeds 11/23/37/51 at K=1/2/4, always/weak/gated policies,
  held-out exterior selection versus normalized exterior-plus-contact scoring,
  and privileged distractor/aspect/thickness diagnostics. Missing hypotheses
  fall back to B0 and incomplete K budgets remain recorded.
- E6 waits for useful E4 and E5 development evidence. It freezes the teacher,
  uses its exact gated predictions, records ancestry, and trains geometry/all/
  filtered/random-count PointNets for 2000 updates at seeds 101/202/303.
  Three seeds run in separate GPU workers. Zero filtered labels is a failure.
  A smoke configuration exercises this path without pretending the gates passed.
- E7 has clean, noise .0025/.005/.01, dropout .25/.5, erosion proxy .1,
  rotations and eligible missing-piece trials, three repeats. Density 256 and
  2048 is varied separately from corruption; the baseline is 512 points.
  All generated conditions rerender/regenerate/reconstruct corrupted input.
  Students are added after E6. Actual 3/5-piece patterns and real records are
  used only when present; unlabeled real accuracy stays unscored.

B5 runs in an exclusive scheduler window. Its target is primary-model/input
E1/E2 work across seeds. Existing config caps bound additional search; capped
or incomplete targets are explicitly unmatched. This isolates timing from other
suite tasks, not unrelated host CPU workloads: use reserved server resources
for interpretable wall-time comparisons.

## Artifacts, reuse, and resume

```text
plan.json                           immutable suite DAG / budgets / code hashes
configs/<task>.json                 resolved task configs
logs/<task>.log                     worker output
suite_status.json                   task states / active tasks / lock location
runs/<task>/                       ordinary study.json + jobs + evaluation
runs/<task>/reuse.json              verified imports / incompatibility reasons
runs/<task>/requests.json           requested cases before execution
runs/<task>/task_result.json        task outcome and evaluation hashes
analysis/<condition>__<split>/      per_source.csv / metrics.jsonl / REPORT.md
analysis/per_case.csv               all condition-labelled case evidence
analysis/development_gate.json      E4/E5 readiness for training
```

Each run also has a failure CSV/gallery and all standard stage artifacts.
Reference-derived controls never enter training. Imported outputs must be
completed, unlocked, hash verified, code compatible, input compatible and have
compatible parents. A proposer change invalidates A renders. Alignment changes
reuse compatible images but rerun reconstruction/alignment; old meshes are not
silently treated as newly aligned. Rejecting an import is logged, not hidden.

```bash
# Use exactly the same arguments/root to resume. Failed work needs explicit retry.
bash run_remaining_studies.sh --retry-failed
# Optional import before launching a stopped suite:
python -m remaining_studies import --root "$GA_RUN_GROUP" \
  --task baseline-generate-000 --source-runs /path/to/completed/run
```

Changed configuration, code, source-run list, dataset, or task partitioning
requires a new root. Inspect lock host/PID before removing a stale lock:
`python -m remaining_studies unlock --lock /exact/path/suite.lock
--confirmed-process-dead`. Engine `running.lock` jobs require the engine's
corresponding explicit unlock. Locks are never removed automatically.
Superseded outcomes from explicit retries remain archived, while current
per-case comparisons use the latest outcome for each requested arm. Source
aliases and identical cross-split observations are rejected; near-duplicate
geometry still requires the dataset's source-identity audit.

## Results notebook

Open `pipeline_stage_results.ipynb`, set `RUN_PATHS` to the suite directory, and run
all cells. Nested worker roots are discovered automatically. Use the stage
browser for images, raw/aligned surfaces, final poses, contacts, gate choices,
student checkpoints, and corruption outputs. Oracle and smoke rows are hidden
by default. Point `DATASETS` at the server dataset for reference visualization.

Statistics average repeats/patterns within original sources and use 2000 paired
source bootstrap resamples. Failures remain in requested deployable outcomes.
Per-condition reports avoid pooling architectures, duplicated cached jobs or
training seeds as independent sources. Comparisons are exploratory on dev;
inconclusive intervals remain inconclusive. Locked confirmation uses the engine's
separate freeze/test workflow after architecture choices are final.

For CPU software validation, pass `generative_assembly/configs/smoke.json` and
a separately generated `python -m generative_assembly demo` dataset. Smoke
outputs are never research evidence or production priors.

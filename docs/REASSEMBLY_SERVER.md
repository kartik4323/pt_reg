# Run reassembly v2 on a Linux GPU server

Repository: https://github.com/kartik4323/pt_reg

This guide uses an RTX A5000 (24 GB), Python 3.10, and fresh v2 checkpoints. It covers setup, data acquisition, preflight, three-stage training, evaluation, resume and inference. Run the blocks in order on the server in Bash.

**Current data gate:** the first 100 ShapeNet bottle candidates yielded **14 accepted sources and 102 fracture patterns**. Held-out learning requires **30 accepted sources**. Repeating those same candidates on a server will not solve the geometry yield problem. Setup and data preparation are ready to run; the training commands below are for a dataset that passes the gate after source-pool reassessment. Do not lower `min_sources`, fill cavities, or change rejection rules just to start training. The original 100-source pilot cap remains in the configuration.

## 1. Clone and create a fresh environment

Use a Linux x86-64 GPU VM with Python 3.10, Git, and an NVIDIA driver compatible with the chosen PyTorch CUDA wheel. `nvidia-smi` must show the GPU inside the VM. GPU passthrough/driver configuration is a VM-provider prerequisite.

```bash
git clone --branch main https://github.com/kartik4323/pt_reg.git
cd pt_reg
git rev-parse HEAD
nvidia-smi

python3.10 -m venv "$HOME/venvs/ptreg-v2"
source "$HOME/venvs/ptreg-v2/bin/activate"
python -m pip install --upgrade pip
python -m pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir -r requirements.reassembly-v2.txt
python -m pip check
```

If you already cloned the repository and have no local changes, use `git pull --ff-only origin main` in that checkout instead of cloning again. If `python3.10` or its venv module is unavailable, create a Python 3.10 environment with your server's environment manager before continuing.

The CUDA 12.6 wheel command is listed in [PyTorch's installation instructions](https://pytorch.org/get-started/previous-versions/#v2110). Follow [NVIDIA's compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) for the host driver. This workflow uses standard Torch operators and does not require compiling custom CUDA extensions or installing the separate SOTA environments.

Check actual CUDA execution:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA is unavailable; fix VM GPU/driver access before training'
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', torch.cuda.get_device_properties(0).total_memory / 1024**3)
x = torch.randn(128, 128, device='cuda')
y = x @ x.T
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print('CUDA execution passed')
PY

python -m unittest discover -s tests -p 'test_reassembly*.py' -v
```

For long runs, start `tmux new -s ptreg-v2`, then activate the environment and enter the repository in that session. Detach with Ctrl+B then D; reconnect with `tmux attach -t ptreg-v2`. A dropped SSH connection will then leave training running.

## 2. Configure paths and authenticate

The example uses your home filesystem, which should have at least 50 GiB free throughout the run. Your earlier `/data` filesystem was nearly full; do not use it by default. If home resolves to an unsuitable filesystem, choose another persistent location with enough space. Keep managed source data, preparation, logs and runs under one `ROOT`; storage is capped at 40 GiB.

```bash
# Re-run this block inside each new terminal/tmux session.
source "$HOME/venvs/ptreg-v2/bin/activate"
cd "$HOME/pt_reg"  # Adjust if you cloned elsewhere.
export ROOT="$HOME/reassembly_v2"
export BASE="$PWD/configs/reassembly_v2.yaml"
export DATA="$ROOT/prepared/manifest.json"
export PF="$ROOT/preflight/preflight.json"
export CFG="$ROOT/preflight/config.resolved.json"
export OMP_NUM_THREADS=8
mkdir -p "$ROOT/logs"
df -h "$ROOT"
```

Authenticate with the account that has approved access to ShapeNet. The following prompt hides the token and stores it in Hugging Face's user cache, outside the repository. It does not add Hugging Face credentials to Git:

```bash
python -c 'from getpass import getpass; from huggingface_hub import login; login(token=getpass("Hugging Face read token: "), add_to_git_credential=False)'
```

Use your token at the prompt; no credential belongs in a tracked script, configuration or command example. This Python login works with the pinned Hub client. Authentication is separate from GitHub authentication used to clone a private repository.

## 3. Acquire, prepare, and inspect the geometry gate

```bash
python -m reassembly acquire --config "$BASE" --managed-root "$ROOT" --dry-run
python -m reassembly acquire --config "$BASE" --managed-root "$ROOT"

set -o pipefail
python -u -m reassembly prepare --config "$BASE" --managed-root "$ROOT" \
  --source "$ROOT/sources/02876657.zip" --output "$ROOT/prepared" \
  2>&1 | tee "$ROOT/logs/prepare.log"
```

An existing bottle archive can be passed directly to `--source`; another download is unnecessary. Preparation refuses to overwrite an existing prepared manifest. Use a fresh output directory for a revised source selection and update `DATA` accordingly.

Exit code 2 is expected when yield is insufficient. Read the report:

```bash
python - "$DATA" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
r = m['report']
print('Accepted sources:', r['accepted_sources'])
print('Accepted patterns:', r['accepted_patterns'])
print('Learning ready:', r['learning_ready'])
if not r['learning_ready']:
    raise SystemExit('STOP: reassess the source pool before training; at least 30 accepted sources are required.')
PY
```

**Stop here for the current 14-source dataset.** The remaining commands document the measured training sequence once preparation passes. New source data changes the dataset fingerprint and requires fresh preflight and training runs.

## 4. Profile all three stages on the A5000

```bash
set -o pipefail
python -u -m reassembly preflight --config "$BASE" --managed-root "$ROOT" \
  --manifest "$DATA" --device cuda --output "$ROOT/preflight" \
  2>&1 | tee "$ROOT/logs/preflight.log"
```

The report must have `status: passed` and peak reserved memory **below 20 GiB**. The profiler retries batch 1 with more accumulation if batch 2 exceeds the cap; point resolution stays unchanged. Use `CFG`, the generated resolved config, for all later commands. A CPU preflight cannot authorize CUDA training. Do not use local Windows preflight reports on the server.

```bash
python - "$PF" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print('Status:', r['status'], 'fallback:', r.get('fallback_applied'))
for attempt in r['attempts']:
    memory = attempt.get('max_reserved_bytes')
    print('Batch:', attempt['batch_size'], 'reserved GiB:', None if memory is None else memory/1024**3)
assert r['status'] == 'passed', 'Stop: CUDA preflight did not pass'
PY
```

## 5. Fit the fixed 16 patterns first

Run this block only after the geometry and CUDA gates pass. The subshell stops immediately if any command fails. Each stage is bounded to 2,000 updates.

```bash
(
set -euo pipefail
for STAGE in 1 2 3; do
  INIT=()
  if [ "$STAGE" -gt 1 ]; then
    PREV=$((STAGE - 1))
    INIT=(--initialize-from "$ROOT/overfit/s$PREV/best.pt")
  fi
  python -u -m reassembly train --config "$CFG" --manifest "$DATA" --device cuda \
    --preflight-report "$PF" --overfit --stage "$STAGE" \
    --run-dir "$ROOT/overfit/s$STAGE" "${INIT[@]}" \
    2>&1 | tee "$ROOT/logs/overfit-s$STAGE.log"
done
python -u -m reassembly evaluate --config "$CFG" --manifest "$DATA" --device cuda \
  --overfit --checkpoint "$ROOT/overfit/s3/best.pt" --output "$ROOT/overfit/eval" \
  2>&1 | tee "$ROOT/logs/overfit-eval.log"
python - "$ROOT/overfit/eval/evaluation.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(r['summary'])
assert r['fixed_fit_passed'], 'Stop: the 16-pattern assembly check has not passed'
PY
)
```

The field alone is not the criterion: all 16 rigid assemblies must pass. No overfit checkpoint can initialize a held-out pilot. A failed check needs investigation before the next section.

## 6. Train matched fresh pilots

This runs separate fresh `predicted` and `contact_only` experiments with equal budgets. Stage 1 learns geometry; stage 2 learns the scaffold while retaining geometry supervision; stage 3 freezes the encoder/field and trains the matcher. It does not backpropagate through the numerical solver.

```bash
(
set -euo pipefail
for CONDITION in predicted contact_only; do
  RUN="$ROOT/pilot/$CONDITION"
  for STAGE in 1 2 3; do
    INIT=()
    if [ "$STAGE" -gt 1 ]; then
      PREV=$((STAGE - 1))
      INIT=(--initialize-from "$RUN/s$PREV/best.pt")
    fi
    python -u -m reassembly train --config "$CFG" --manifest "$DATA" --device cuda \
      --preflight-report "$PF" --overfit-report "$ROOT/overfit/eval/evaluation.json" \
      --condition "$CONDITION" --stage "$STAGE" --run-dir "$RUN/s$STAGE" "${INIT[@]}" \
      2>&1 | tee "$ROOT/logs/$CONDITION-s$STAGE.log"
  done
done
)
```

Run directories must be new unless you explicitly resume. Only `best.pt` and `latest.pt` are retained. Stage handoff requires a completed preceding stage; resume an interrupted stage before proceeding.

## 7. Evaluate all controls and unseen cut families

```bash
(
set -euo pipefail
for SPLIT in test cut_holdout; do
  for CONDITION in contact_only predicted gt perturbed; do
    MODEL=predicted
    if [ "$CONDITION" = contact_only ]; then MODEL=contact_only; fi
    python -u -m reassembly evaluate --config "$CFG" --manifest "$DATA" --device cuda \
      --split "$SPLIT" --condition "$CONDITION" \
      --checkpoint "$ROOT/pilot/$MODEL/s3/best.pt" \
      --output "$ROOT/eval/$SPLIT/$CONDITION" \
      2>&1 | tee "$ROOT/logs/eval-$SPLIT-$CONDITION.log"
  done
done
)
```

GT is an evaluation-only signed-field oracle. Perturbed fields test resistance to misleading priors. Neither condition changes input fragments or supplies GT contact correspondences to the solver.

Create one decision report for each split:

```bash
(
set -euo pipefail
for SPLIT in test cut_holdout; do
  INPUTS=(--input "$(dirname "$DATA")/preparation_report.json" --input "$PF"
          --input "$ROOT/overfit/eval/evaluation.json")
  for MODEL in predicted contact_only; do
    for STAGE in 1 2 3; do
      INPUTS+=(--input "$ROOT/pilot/$MODEL/s$STAGE/training_report.json")
    done
  done
  for CONDITION in contact_only predicted gt perturbed; do
    INPUTS+=(--input "$ROOT/eval/$SPLIT/$CONDITION/evaluation.json")
  done
  python -m reassembly report --config "$CFG" "${INPUTS[@]}" \
    --output "$ROOT/pilot_report_$SPLIT.json"
done
)
```

Report geometric assembly success, per-part/whole Chamfer, contact residuals, failures, difficulty bands and actual memory/throughput. Improved reconstruction alone is insufficient. Reporting never starts a longer run. Keep both split reports and resolved configs with any exported results.

## 8. Monitor, resume, and infer

In another SSH terminal:

```bash
watch -n 2 nvidia-smi
# Or inspect the current log:
tail -f "$HOME/reassembly_v2/logs/predicted-s3.log"
```

Resume an interrupted stage with its explicit checkpoint and the same run directory/configuration:

```bash
python -u -m reassembly train --config "$CFG" --manifest "$DATA" --device cuda \
  --preflight-report "$PF" --condition predicted --stage 3 \
  --run-dir "$ROOT/pilot/predicted/s3" --resume "$ROOT/pilot/predicted/s3/latest.pt" \
  2>&1 | tee -a "$ROOT/logs/predicted-s3-resume.log"
```

For an interrupted fixed overfit stage, add `--overfit` and use its `overfit/sN` directory. Completed budgets are not resumed automatically. Changing data, architecture, loss settings or batch settings requires a fresh experiment and matching preflight.

XYZ-only inference accepts two or three `N×3` NumPy arrays in consistent units:

```bash
python -m reassembly infer --config "$CFG" --device cuda \
  --checkpoint "$ROOT/pilot/predicted/s3/best.pt" \
  --fragment /path/to/fragment_a.npy --fragment /path/to/fragment_b.npy \
  --output "$ROOT/inference/example_001" --save-scaffold
```

For three pieces, add a third `--fragment`. Use a fresh output directory. `result.json` reports status, confidence and transforms; `assembly.npz` contains aligned original-resolution fragments when a solve exists. Apply matrices as `x_aligned = x @ R.T + t`; the reference transform is identity. Failed solves return null transforms; low-confidence or failed results exit with code 2 and must not be treated as successful assemblies.

For architectural details and metric conventions, see [the main runbook](REASSEMBLY_V2.md). For the measured local geometry yield, see [verification results](REASSEMBLY_VALIDATION.md).

# Run reassembly v2 on a Linux GPU server

Repository: https://github.com/kartik4323/pt_reg

This guide uses an RTX A5000 (24 GB), Python 3.10, and fresh v2 checkpoints. It covers setup, data acquisition, preflight, three-stage training, evaluation, resume and inference. Run the blocks in order on the server in Bash.

**Expanded source pool:** the first 100 ShapeNet bottle candidates yielded **14 accepted sources and 102 fracture patterns**. A subsequent audit of all **498** bottles found **73** sources passing the same geometry validation. This guide now uses `configs/reassembly_v2_bottles498.yaml`, which explicitly expands the source pool and pins the audited archive revision. Preparation must still retain at least **30 sources with valid complementary fractures** before training. The original 100-source configuration remains available for reproducing the initial pilot.

Expanded preparation has now completed locally: **73 sources and 546 complementary patterns**, with content hashes verified and `learning_ready: true`. Preparation took about **236 seconds on CPU**. GPU memory, overfit success and learned assembly performance still need to be measured on the VM.

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
export WORK="$ROOT/bottles498"
export REASSEMBLY_ROOT="$ROOT"
export BASE="$PWD/configs/reassembly_v2_bottles498.yaml"
export DATA="$WORK/prepared/manifest.json"
export PF="$WORK/preflight/preflight.json"
export CFG="$WORK/preflight/config.resolved.json"
export OMP_NUM_THREADS=8
mkdir -p "$WORK/logs"
df -h "$ROOT"
```

Authenticate with the account that has approved access to ShapeNet. The following prompt hides the token and stores it in Hugging Face's user cache, outside the repository. It does not add Hugging Face credentials to Git:

```bash
python -c 'from getpass import getpass; from huggingface_hub import login; login(token=getpass("Hugging Face read token: "), add_to_git_credential=False)'
```

Use your token at the prompt; no credential belongs in a tracked script, configuration or command example. This Python login works with the pinned Hub client. Authentication is separate from GitHub authentication used to clone a private repository.

The following runner performs acquisition/preparation, CUDA preflight, the fixed overfit check, both fresh training conditions, eight evaluations and the two final reports. It stops at the first failed command or acceptance gate and saves console output in `$WORK/logs`, alongside package versions and the Git commit. All training is bounded to 2,000 updates per stage; reporting does not launch a longer run.

```bash
bash scripts/run_reassembly_v2_pilot.sh all
```

For separate invocations, execute the stages in this order, checking each exit status before continuing:

```bash
bash scripts/run_reassembly_v2_pilot.sh prepare
bash scripts/run_reassembly_v2_pilot.sh preflight
bash scripts/run_reassembly_v2_pilot.sh overfit
bash scripts/run_reassembly_v2_pilot.sh train
bash scripts/run_reassembly_v2_pilot.sh evaluate
bash scripts/run_reassembly_v2_pilot.sh report
```

Choose either the single `all` invocation or the separate stages. Existing runs are preserved and are never automatically resumed. Sections 3–8 below document the underlying CLI calls and explicit resume procedure; they do not need to be repeated after a successful runner invocation. A failed gate is a result to inspect, not a reason to lower the thresholds.

## 3. Acquire, prepare, and inspect the geometry gate

```bash
python -m reassembly acquire --config "$BASE" --managed-root "$ROOT" --dry-run
python -m reassembly acquire --config "$BASE" --managed-root "$ROOT"

set -o pipefail
python -u -m reassembly prepare --config "$BASE" --managed-root "$ROOT" \
  --source "$ROOT/sources/02876657.zip" --output "$WORK/prepared" \
  2>&1 | tee "$WORK/logs/prepare.log"
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

Continue only when expanded preparation reports `learning_ready: true`. New source selection changes the dataset fingerprint and requires fresh preflight and training runs. Source-validation yield alone is insufficient; complementary fracture generation must pass too.

## 4. Profile all three stages on the A5000

```bash
set -o pipefail
python -u -m reassembly preflight --config "$BASE" --managed-root "$ROOT" \
  --manifest "$DATA" --device cuda --output "$WORK/preflight" \
  2>&1 | tee "$WORK/logs/preflight.log"
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
    INIT=(--initialize-from "$WORK/overfit/s$PREV/best.pt")
  fi
  python -u -m reassembly train --config "$CFG" --manifest "$DATA" --device cuda \
    --preflight-report "$PF" --overfit --stage "$STAGE" \
    --run-dir "$WORK/overfit/s$STAGE" "${INIT[@]}" \
    2>&1 | tee "$WORK/logs/overfit-s$STAGE.log"
done
python -u -m reassembly evaluate --config "$CFG" --manifest "$DATA" --device cuda \
  --overfit --checkpoint "$WORK/overfit/s3/best.pt" --output "$WORK/overfit/eval" \
  2>&1 | tee "$WORK/logs/overfit-eval.log"
python - "$WORK/overfit/eval/evaluation.json" <<'PY'
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
  RUN="$WORK/pilot/$CONDITION"
  for STAGE in 1 2 3; do
    INIT=()
    if [ "$STAGE" -gt 1 ]; then
      PREV=$((STAGE - 1))
      INIT=(--initialize-from "$RUN/s$PREV/best.pt")
    fi
    python -u -m reassembly train --config "$CFG" --manifest "$DATA" --device cuda \
      --preflight-report "$PF" --overfit-report "$WORK/overfit/eval/evaluation.json" \
      --condition "$CONDITION" --stage "$STAGE" --run-dir "$RUN/s$STAGE" "${INIT[@]}" \
      2>&1 | tee "$WORK/logs/$CONDITION-s$STAGE.log"
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
      --checkpoint "$WORK/pilot/$MODEL/s3/best.pt" \
      --output "$WORK/eval/$SPLIT/$CONDITION" \
      2>&1 | tee "$WORK/logs/eval-$SPLIT-$CONDITION.log"
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
          --input "$WORK/overfit/eval/evaluation.json")
  for MODEL in predicted contact_only; do
    for STAGE in 1 2 3; do
      INPUTS+=(--input "$WORK/pilot/$MODEL/s$STAGE/training_report.json")
    done
  done
  for CONDITION in contact_only predicted gt perturbed; do
    INPUTS+=(--input "$WORK/eval/$SPLIT/$CONDITION/evaluation.json")
  done
  python -m reassembly report --config "$CFG" "${INPUTS[@]}" \
    --output "$WORK/pilot_report_$SPLIT.json"
done
)
```

Report geometric assembly success, per-part/whole Chamfer, contact residuals, failures, difficulty bands and actual memory/throughput. Improved reconstruction alone is insufficient. Reporting never starts a longer run. Keep both split reports and resolved configs with any exported results.

## 8. Monitor, resume, and infer

### Restart after the 16-pattern assembly check failed

The September 11 contact fix changes matching selection, supervision, and confidence (`coarse-scaffold-reassembly-v2.1-local-contacts`). It requires **fresh stage 1/2/3 weights and a new CUDA preflight**. Old checkpoints are intentionally rejected. Keep the existing preparation; downloading or fracturing again is unnecessary. The 0.01 success tolerance and the all-16 acceptance gate are unchanged. See [the diagnosis and training review](REASSEMBLY_OVERFIT_REVIEW.md).

With `ptreg-v2` activated, run this on the reported VM:

```bash
(
set -euo pipefail
cd /home/kpandey/satellite/pt_reg
git pull --ff-only origin main
export REASSEMBLY_ROOT="/home/kpandey/reassembly_v2"
WORK="$REASSEMBLY_ROOT/bottles498"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
for NAME in overfit preflight logs; do
  if [ -d "$WORK/$NAME" ]; then
    mv -T -- "$WORK/$NAME" "$WORK/$NAME.before-contact-fix-$STAMP"
  fi
done
bash scripts/run_reassembly_v2_pilot.sh preflight
bash scripts/run_reassembly_v2_pilot.sh overfit
)
```

The renamed folders preserve the previous results; the new logs do not mix old and new attempts. The resource guard still counts the preserved files. Do not rerun `all`, which would try to prepare into the existing dataset directory. If overfit passes, continue the bounded pilot:

```bash
(
set -euo pipefail
export REASSEMBLY_ROOT="/home/kpandey/reassembly_v2"
bash scripts/run_reassembly_v2_pilot.sh train
bash scripts/run_reassembly_v2_pilot.sh evaluate
bash scripts/run_reassembly_v2_pilot.sh report
)
```

If it fails, inspect `overfit/eval/evaluation.json` and the new `sample_diagnostic` lines in `logs/overfit-eval.log`. They include per-part errors, contact support, pair candidate counts, confidence, and pre/post-refinement residuals. `matching_mass` and `matching_localization` are now separate training metrics; the new composite loss is not numerically comparable to the old mass-only objective.

An optional supervision-only diagnostic can test the representation ceiling on this exact prepared dataset without training:

```bash
python -m reassembly.contact_audit \
  --config configs/reassembly_v2_bottles498.yaml \
  --manifest "$REASSEMBLY_ROOT/bottles498/prepared/manifest.json" \
  --managed-root "$REASSEMBLY_ROOT" \
  --output "$REASSEMBLY_ROOT/bottles498/contact_ceiling_audit.json"
```

This audit supplies GT surface labels and positions to construct ideal matches. Its results **never** authorize training or count as learned assembly performance.

### Monitoring and explicit resume

In another SSH terminal:

```bash
watch -n 2 nvidia-smi
# Or inspect the current log:
tail -f "$HOME/reassembly_v2/bottles498/logs/predicted-s3.log"
```

Resume an interrupted stage with its explicit checkpoint and the same run directory/configuration:

```bash
python -u -m reassembly train --config "$CFG" --manifest "$DATA" --device cuda \
  --preflight-report "$PF" --condition predicted --stage 3 \
  --run-dir "$WORK/pilot/predicted/s3" --resume "$WORK/pilot/predicted/s3/latest.pt" \
  2>&1 | tee -a "$WORK/logs/predicted-s3-resume.log"
```

For an interrupted fixed overfit stage, add `--overfit` and use its `overfit/sN` directory. Completed budgets are not resumed automatically. Changing data, architecture, loss settings or batch settings requires a fresh experiment and matching preflight.

If an older checkout stopped with `Missing/non-finite gradients at stage 1, update 16`, update the code and repeat CUDA preflight. The revised training loop uses the current GradScaler API, keeps sensitive contact operations in float32, and retries overflowed accumulated batches with a lower scale. Retries preserve sample selection/random views and do not count as successful optimizer updates. Up to eight retries are allowed per update; persistent non-finite gradients, missing gradients and non-finite losses still stop training. This follows [PyTorch's AMP gradient-scaling and clipping procedure](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html).

For the reported early failure (before the first checkpoint at update 100), preserve the failed run and old preflight, then start a fresh overfit run using the existing prepared data:

```bash
(
set -euo pipefail
git pull --ff-only origin main
export REASSEMBLY_ROOT="${REASSEMBLY_ROOT:-$HOME/reassembly_v2}"
WORK="$REASSEMBLY_ROOT/bottles498"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mv -T -- "$WORK/overfit" "$WORK/overfit.failed-$STAMP"
mv -T -- "$WORK/preflight" "$WORK/preflight.before-amp-fix-$STAMP"
bash scripts/run_reassembly_v2_pilot.sh preflight
bash scripts/run_reassembly_v2_pilot.sh overfit
)
```

Use the managed root from the failed run's log if it differs from the default. Keep the Python environment active and run this from the repository directory. Prepared geometry is reused. Older preflight signatures are intentionally rejected because numerical operations and their GPU memory use changed.

`numerics.jsonl` records overflow attempts, scales, affected parameter names, sampled pattern IDs and loss components. Validation history includes gradient norm, AMP scale and cumulative retry count. A persistent numerical failure writes `failure.json` and marks `training_report.json` as failed. After the fixed overfit gate passes, continue with the runner's `train`, `evaluate` and `report` commands.

XYZ-only inference accepts two or three `N×3` NumPy arrays in consistent units:

```bash
python -m reassembly infer --config "$CFG" --device cuda \
  --checkpoint "$WORK/pilot/predicted/s3/best.pt" \
  --fragment /path/to/fragment_a.npy --fragment /path/to/fragment_b.npy \
  --output "$WORK/inference/example_001" --save-scaffold
```

For three pieces, add a third `--fragment`. Use a fresh output directory. `result.json` reports status, confidence and transforms; `assembly.npz` contains aligned original-resolution fragments when a solve exists. Apply matrices as `x_aligned = x @ R.T + t`; the reference transform is identity. Failed solves return null transforms; low-confidence or failed results exit with code 2 and must not be treated as successful assemblies.

For architectural details and metric conventions, see [the main runbook](REASSEMBLY_V2.md). For the measured local geometry yield, see [verification results](REASSEMBLY_VALIDATION.md).

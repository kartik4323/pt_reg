#!/usr/bin/env bash
# Fresh, bounded GPU pilot. Activate the documented Python environment first.
set -Eeuo pipefail

usage() {
  echo 'Usage: bash scripts/run_reassembly_v2_pilot.sh {prepare|preflight|overfit|train|evaluate|report|all}'
  echo 'Set REASSEMBLY_ROOT to the managed storage root (default: ~/reassembly_v2).'
  echo 'Outputs: $REASSEMBLY_ROOT/bottles498. Existing runs are never automatically resumed.'
}
ACTION="${1:-help}"
case "$ACTION" in
  help|-h|--help) usage; exit 0 ;;
  prepare|preflight|overfit|train|evaluate|report|all) ;;
  *) usage >&2; exit 2 ;;
esac
if [ "$#" -ne 1 ]; then usage >&2; exit 2; fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
PYTHON_BIN="${REASSEMBLY_PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
ROOT="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve().as_posix())' "${REASSEMBLY_ROOT:-$HOME/reassembly_v2}")"
WORK="$ROOT/bottles498"
BASE="$REPO_ROOT/configs/reassembly_v2_bottles498.yaml"
DATA="$WORK/prepared/manifest.json"
PF="$WORK/preflight/preflight.json"
CFG="$WORK/preflight/config.resolved.json"
CURRENT_STEP=setup
trap 'rc=$?; echo "Stopped during $CURRENT_STEP (exit $rc). Inspect $WORK/logs. Later steps were not run." >&2; exit "$rc"' ERR

# Resolve and check the actual filesystem before creating managed artifacts.
"$PYTHON_BIN" - "$BASE" "$ROOT" <<'PY'
import sys
from reassembly.config import load_config
from reassembly.resources import make_guard
cfg = load_config(sys.argv[1])
cfg['resources']['managed_root'] = sys.argv[2]
print(make_guard(cfg).check())
PY
mkdir -p "$WORK/logs"

run_logged() {
  CURRENT_STEP="$1"
  shift
  echo "Running $CURRENT_STEP"
  "$@" 2>&1 | tee -a "$WORK/logs/$CURRENT_STEP.log"
}

check_geometry() {
  run_logged geometry-check "$PYTHON_BIN" - "$DATA" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding='utf-8'))
r = m['report']
count = len({p['source_id'] for p in m['patterns']})
print('Accepted sources:', count, 'patterns:', len(m['patterns']))
if not r['learning_ready'] or count < max(30, r['minimum_sources']):
    raise SystemExit('STOP: preparation must retain at least 30 sources with valid fractures.')
PY
}

check_overfit() {
  run_logged overfit-check "$PYTHON_BIN" - "$WORK/overfit/eval/evaluation.json" "$DATA" "$CFG" <<'PY'
import json, sys
from pathlib import Path
from reassembly.cli import require_overfit
from reassembly.config import load_config
require_overfit(Path(sys.argv[1]), Path(sys.argv[2]), load_config(sys.argv[3]))
print(json.load(open(sys.argv[1], encoding='utf-8'))['summary'])
print('Fixed 16-pattern assembly gate passed.')
PY
}

prepare() {
  if [ -d "$WORK/prepared" ] && [ -n "$(ls -A "$WORK/prepared")" ]; then
    echo "Preparation output already exists: $WORK/prepared. Continue with preflight for a completed dataset." >&2
    return 2
  fi
  run_logged acquire-check "$PYTHON_BIN" -u -m reassembly acquire --config "$BASE" --managed-root "$ROOT" --dry-run
  run_logged acquire "$PYTHON_BIN" -u -m reassembly acquire --config "$BASE" --managed-root "$ROOT"
  run_logged prepare "$PYTHON_BIN" -u -m reassembly prepare --config "$BASE" --managed-root "$ROOT" \
    --source "$ROOT/sources/02876657.zip" --output "$WORK/prepared"
  check_geometry
}

preflight() {
  check_geometry
  run_logged preflight "$PYTHON_BIN" -u -m reassembly preflight --config "$BASE" --managed-root "$ROOT" \
    --manifest "$DATA" --device cuda --output "$WORK/preflight"
}

overfit() {
  check_geometry
  local stage prev
  local -a init
  for stage in 1 2 3; do
    init=()
    if [ "$stage" -gt 1 ]; then
      prev=$((stage - 1)); init=(--initialize-from "$WORK/overfit/s$prev/best.pt")
    fi
    run_logged "overfit-s$stage" "$PYTHON_BIN" -u -m reassembly train --config "$CFG" --managed-root "$ROOT" \
      --manifest "$DATA" --device cuda --preflight-report "$PF" --overfit --stage "$stage" \
      --run-dir "$WORK/overfit/s$stage" "${init[@]}"
  done
  run_logged overfit-eval "$PYTHON_BIN" -u -m reassembly evaluate --config "$CFG" --managed-root "$ROOT" \
    --manifest "$DATA" --device cuda --overfit --checkpoint "$WORK/overfit/s3/best.pt" --output "$WORK/overfit/eval"
  check_overfit
}

train() {
  check_geometry
  check_overfit
  local condition stage prev run
  local -a init
  for condition in predicted contact_only; do
    run="$WORK/pilot/$condition"
    for stage in 1 2 3; do
      init=()
      if [ "$stage" -gt 1 ]; then
        prev=$((stage - 1)); init=(--initialize-from "$run/s$prev/best.pt")
      fi
      run_logged "$condition-s$stage" "$PYTHON_BIN" -u -m reassembly train --config "$CFG" --managed-root "$ROOT" \
        --manifest "$DATA" --device cuda --preflight-report "$PF" --overfit-report "$WORK/overfit/eval/evaluation.json" \
        --condition "$condition" --stage "$stage" --run-dir "$run/s$stage" "${init[@]}"
    done
  done
}

evaluate() {
  local split condition model
  for split in test cut_holdout; do
    for condition in contact_only predicted gt perturbed; do
      model=predicted
      if [ "$condition" = contact_only ]; then model=contact_only; fi
      run_logged "eval-$split-$condition" "$PYTHON_BIN" -u -m reassembly evaluate --config "$CFG" --managed-root "$ROOT" \
        --manifest "$DATA" --device cuda --split "$split" --condition "$condition" \
        --checkpoint "$WORK/pilot/$model/s3/best.pt" --output "$WORK/eval/$split/$condition"
    done
  done
}

report() {
  local split model stage condition
  local -a inputs
  for split in test cut_holdout; do
    inputs=(--input "$WORK/prepared/preparation_report.json" --input "$PF" --input "$WORK/overfit/eval/evaluation.json")
    for model in predicted contact_only; do
      for stage in 1 2 3; do inputs+=(--input "$WORK/pilot/$model/s$stage/training_report.json"); done
    done
    for condition in contact_only predicted gt perturbed; do
      inputs+=(--input "$WORK/eval/$split/$condition/evaluation.json")
    done
    run_logged "report-$split" "$PYTHON_BIN" -u -m reassembly report --config "$CFG" --managed-root "$ROOT" \
      "${inputs[@]}" --output "$WORK/pilot_report_$split.json"
  done
}

# Package names/versions only: no credentials, environment dump or dependency URLs.
run_logged environment "$PYTHON_BIN" - <<'PY'
import importlib.metadata, json, platform, subprocess, sys
import torch
print(json.dumps(dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
    cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    packages={p.metadata['Name']: p.version for p in importlib.metadata.distributions() if p.metadata['Name']},
    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()), indent=2))
PY
if [ "$ACTION" = all ]; then
  for step in prepare preflight overfit train evaluate report; do CURRENT_STEP="$step"; "$step"; done
else
  CURRENT_STEP="$ACTION"; "$ACTION"
fi
echo "Completed $ACTION. Artifacts: $WORK"

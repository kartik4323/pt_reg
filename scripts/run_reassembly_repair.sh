#!/usr/bin/env bash
# Explicit repair experiment phases. No implicit full-training or test launch.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PHASE="${1:-}"
case "$PHASE" in
  field|comparisons|replicate|scaffold|final-test|bundle) ;;
  *) echo "Usage: bash scripts/run_reassembly_repair.sh {field|comparisons|replicate|scaffold|final-test|bundle}" >&2; exit 2 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
REASSEMBLY_ROOT="${REASSEMBLY_ROOT:-${HOME}/reassembly_v2}"
REPAIR_WORK="${REPAIR_WORK:-${REASSEMBLY_ROOT}/repair_v3}"
FIELD_DIAGNOSTICS="${FIELD_DIAGNOSTICS:-${REASSEMBLY_ROOT}/field_diagnostics/repair_v3}"
REPAIR_MANIFEST="${REPAIR_MANIFEST:-${REASSEMBLY_ROOT}/bottles498/prepared/manifest.json}"
REPAIR_CONFIG="${REPAIR_CONFIG:-configs/reassembly_repair_bottles498.yaml}"
REPAIR_DEVICE="${REPAIR_DEVICE:-cuda:0}"
export REASSEMBLY_ROOT REPAIR_WORK FIELD_DIAGNOSTICS REPAIR_MANIFEST

# Resolve paths and check the actual volumes before creating managed logs.
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from reassembly.resources import ResourceGuard
root = Path(os.environ['REASSEMBLY_ROOT']).expanduser().resolve()
work = Path(os.environ['REPAIR_WORK']).expanduser().resolve()
field = Path(os.environ['FIELD_DIAGNOSTICS']).expanduser().resolve()
prepared = Path(os.environ['REPAIR_MANIFEST']).expanduser().resolve().parent
for path in (work, field):
    if root not in path.parents or path == prepared or prepared in path.parents or path in prepared.parents:
        raise SystemExit('Use dedicated repair/field output directories under REASSEMBLY_ROOT, outside prepared data.')
ResourceGuard([root, prepared], cap_gib=40, min_free_gib=50).check()
PY

# Keep logs outside the worker output; the worker requires a fresh directory.
mkdir -p "${REASSEMBLY_ROOT}/repair_logs"
LOG="${REASSEMBLY_ROOT}/repair_logs/${PHASE}-$(date -u +%Y%m%dT%H%M%S).log"
trap 'status=$?; echo "Stopped in ${PHASE} (exit ${status}). Inspect ${LOG} and the phase JSON reports. A failed learning gate is not a successful experiment." >&2; exit "$status"' ERR

if [[ "$PHASE" == field ]]; then
  command=("$PYTHON_BIN" -m diagnostics.reassembly_field --managed-root "$REASSEMBLY_ROOT" --output "$FIELD_DIAGNOSTICS" --device "$REPAIR_DEVICE")
  if [[ -n "${DIAGNOSTIC_BUNDLE:-}" ]]; then command+=(--source-diagnostics "$DIAGNOSTIC_BUNDLE"); fi
  if [[ -f "${FIELD_DIAGNOSTICS}/invocation.json" ]]; then command+=(--resume); fi
elif [[ "$PHASE" == bundle ]]; then
  command=("$PYTHON_BIN" -m reassembly.repair bundle --managed-root "$REASSEMBLY_ROOT" --output "$REPAIR_WORK" --config "$REPAIR_CONFIG")
else
  command=("$PYTHON_BIN" -m reassembly.repair run --phase "$PHASE" --managed-root "$REASSEMBLY_ROOT"
    --manifest "$REPAIR_MANIFEST" --output "$REPAIR_WORK" --device "$REPAIR_DEVICE"
    --config "$REPAIR_CONFIG" --field-diagnostic-report "${FIELD_DIAGNOSTICS}/summary.json" --resume)
fi
printf 'Running %s; log: %s\n' "$PHASE" "$LOG"
"${command[@]}" 2>&1 | tee "$LOG"

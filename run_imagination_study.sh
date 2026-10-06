#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$SCRIPT_DIR"
[[ -d "$SCRIPT_DIR/generative_assembly" ]] || PACKAGE_ROOT="$SCRIPT_DIR/pt_reg"
export PYTHONPATH="$PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
GA_ROOT="${GA_ROOT:-/home/kpandey/generative_assembly}"
DATASET="${DATASET:-$GA_ROOT/bottles498-inputs-v1/dataset_2parts.json}"
BASE_CONFIG="${BASE_CONFIG:-$GA_ROOT/configs/sd15-v100-locked.json}"
IMAGES_PYTHON="${IMAGES_PYTHON:-$GA_ROOT/envs/images/bin/python}"
INSTANTMESH_PYTHON="${INSTANTMESH_PYTHON:-$GA_ROOT/envs/instantmesh/bin/python}"
RUN_GROUP="${RUN_GROUP:-$GA_ROOT/runs/imagination-v2-$(date -u +%Y%m%dT%H%M%SZ)}"
LIMIT="${LIMIT:-3}"
read -r -a PROFILES <<< "${PROFILES:-legacy_cleanup clean completion exterior}"
read -r -a STAGES <<< "${STAGES:-E0 E1 E2}"
for path in "$IMAGES_PYTHON" "$INSTANTMESH_PYTHON" "$BASE_CONFIG" "$DATASET"; do
  [[ -f "$path" ]] || { printf 'Missing required file: %s\n' "$path" >&2; exit 1; }
done
mkdir -p "$RUN_GROUP/configs"
printf 'Run group: %s\nProfiles: %s\nCase limit: %s\nStages: %s\n' "$RUN_GROUP" "${PROFILES[*]}" "$LIMIT" "${STAGES[*]}"
"$IMAGES_PYTHON" -m generative_assembly.imagination_study \
  --base "$BASE_CONFIG" --out "$RUN_GROUP/configs" --profiles "${PROFILES[@]}" \
  --images-python "$IMAGES_PYTHON" --reconstruction-python "$INSTANTMESH_PYTHON"
had_failure=0
for profile in "${PROFILES[@]}"; do
  run="$RUN_GROUP/$profile"
  cfg="$RUN_GROUP/configs/$profile.json"
  locked="$RUN_GROUP/configs/$profile-locked.json"
  if [[ ! -f "$locked" ]]; then
    if ! "$IMAGES_PYTHON" -m generative_assembly lock-models --config "$cfg" --out "$locked"; then
      printf 'Model locking failed for %s.\n' "$profile" >&2
      had_failure=1
      continue
    fi
  fi
  mkdir -p "$run"
  "$IMAGES_PYTHON" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); value=dict(dataset=str(Path(sys.argv[2]).resolve()),profile=sys.argv[3]); old=json.loads(p.read_text()) if p.exists() else value; assert old==value, "Resume dataset/profile changed"; p.write_text(json.dumps(value,indent=2))' "$run/experiment.json" "$DATASET" "$profile"
  for stage in "${STAGES[@]}"; do
    printf '\nProfile %s / %s\n' "$profile" "$stage"
    retry_args=()
    [[ "${RETRY_FAILED:-0}" == 1 ]] && retry_args+=(--retry-failed)
    if ! "$IMAGES_PYTHON" -m generative_assembly run \
      --config "$locked" --dataset "$DATASET" --root "$run" --split dev \
      --limit "$LIMIT" --stages "$stage" --allow-failures "${retry_args[@]}" 2>&1 | tee "$RUN_GROUP/$profile-$stage.log"; then
      had_failure=1
      printf 'Stage error retained in %s-%s.log\n' "$profile" "$stage" >&2
    fi
  done
  "$IMAGES_PYTHON" -m generative_assembly status --config "$locked" --dataset "$DATASET" --root "$run"
done
"$IMAGES_PYTHON" -m generative_assembly.imagination_report --root "$RUN_GROUP" --dataset "$DATASET"
printf '\nResults: %s\nOpen imagination_state_analysis.ipynb and set RUN_GROUP to this path.\n' "$RUN_GROUP"
[[ "$had_failure" == 0 ]] || printf 'Some commands failed; inspect logs and the notebook failure table.\n' >&2

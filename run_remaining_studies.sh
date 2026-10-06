#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
GA_ROOT="${GA_ROOT:-/home/kpandey/generative_assembly}"
GA_CONFIG="${GA_CONFIG:-$GA_ROOT/configs/sd15-v100-locked.json}"
GA_DATA="${GA_DATA:-$GA_ROOT/bottles498-inputs-v1/dataset.json}"
GA_RUN_GROUP="${GA_RUN_GROUP:-$GA_ROOT/runs/remaining-$(date -u +%Y%m%dT%H%M%SZ)}"
read -r -a ga_gpus <<< "${GA_GPUS:-}"
read -r -a ga_sources <<< "${GA_SOURCE_RUNS:-}"
read -r -a ga_excluded <<< "${GA_EXCLUDE_GPUS:-}"
exec "${GA_PYTHON:-python}" -m remaining_studies launch \
  --base "$GA_CONFIG" --dataset "$GA_DATA" --root "$GA_RUN_GROUP" \
  --gpus "${ga_gpus[@]}" --exclude-gpus "${ga_excluded[@]}" \
  --cpu-workers "${GA_CPU_WORKERS:-2}" --threads "${GA_THREADS:-1}" \
  --source-runs "${ga_sources[@]}" "$@"

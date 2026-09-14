#!/usr/bin/env bash
set -euo pipefail
# Invoke from pt_reg. Stages remain individually runnable/resumable.
if [ "$#" -lt 4 ]; then
  echo 'Usage: bash generative_assembly/run_all.sh CONFIG DATASET ROOT dev|train|test|robustness|real|smoke'
  exit 2
fi
ga_config="$1"
ga_dataset="$2"
ga_root="$3"
ga_phase="$4"
ga=(python -m generative_assembly)
args=(--config "$ga_config" --dataset "$ga_dataset" --root "$ga_root")
case "$ga_phase" in
  dev)
    "${ga[@]}" run "${args[@]}" --split dev --stages E0 E1 E2 E3 E4 E5
    "${ga[@]}" evaluate "${args[@]}" --split dev
    ;;
  train)
    "${ga[@]}" run "${args[@]}" --split train --stages E0 E1 E2 E4 E5
    "${ga[@]}" train "${args[@]}"
    "${ga[@]}" evaluate "${args[@]}" --split train
    "${ga[@]}" run "${args[@]}" --split dev --stages E6
    "${ga[@]}" evaluate "${args[@]}" --split dev
    ;;
  test)
    "${ga[@]}" freeze "${args[@]}"
    "${ga[@]}" run "${args[@]}" --split test --stages E0 E1 E2 E3 E4 E5 E6
    "${ga[@]}" evaluate "${args[@]}" --split test
    "${ga[@]}" verify "${args[@]}"
    ;;
  robustness)
    "${ga[@]}" run "${args[@]}" --split test --stages E7
    "${ga[@]}" evaluate "${args[@]}" --split test
    ;;
  real)
    "${ga[@]}" run "${args[@]}" --split real --stages E0 E1 E2 E4 E5 E6 E7
    "${ga[@]}" evaluate "${args[@]}" --split real
    ;;
  smoke)
    bash "$0" "$ga_config" "$ga_dataset" "$ga_root" dev
    bash "$0" "$ga_config" "$ga_dataset" "$ga_root" train
    bash "$0" "$ga_config" "$ga_dataset" "$ga_root" test
    bash "$0" "$ga_config" "$ga_dataset" "$ga_root" robustness
    ;;
  *) echo "Unknown phase: $ga_phase"; exit 2 ;;
esac

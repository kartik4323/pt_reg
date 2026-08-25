#!/usr/bin/env bash
set -euo pipefail
MODEL="$1"
ACTION="$2"
shift 2
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m sota_repro.model_entry --model "$MODEL" --action "$ACTION" "$@"

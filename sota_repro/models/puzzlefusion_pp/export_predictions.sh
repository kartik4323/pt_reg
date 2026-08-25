#!/usr/bin/env bash
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"; export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"; exec "${PYTHON:-python3}" -m sota_repro.adapters --model puzzlefusion_pp "$@"

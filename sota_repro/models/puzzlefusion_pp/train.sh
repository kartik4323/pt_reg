#!/usr/bin/env bash
exec "$(dirname "$0")/../../run_model.sh" puzzlefusion_pp train "$@"

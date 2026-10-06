#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--full" ]]; then shift; fi
exec bash "$HERE/run_remaining_studies.sh" --studies E4 "$@"

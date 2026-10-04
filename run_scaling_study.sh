#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_scaling_study.sh
#
# Runs the E0+E3 geometry search at multiple candidate counts
# on the SAME full dev split (48 objects) each time.
#
# Usage:
#   chmod +x run_scaling_study.sh
#   ./run_scaling_study.sh
#
# Results land in: $GA_ROOT/runs/scaling-study-<N>/evaluation/dev/metrics.jsonl
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── Check required env vars ──────────────────────────────────────────────────
: "${GA_ROOT:?Need GA_ROOT set}"
: "${GA_CONFIG:?Need GA_CONFIG set}"
: "${GA_DATA:?Need GA_DATA set}"

# ── Candidate counts to sweep (log-spaced for a smooth curve) ────────────────
CANDIDATE_COUNTS=(16 32 64 128 256 512 1024 2048 4096)

CONFIG_DIR="$GA_ROOT/configs/scaling-study"
mkdir -p "$CONFIG_DIR"

echo "============================================================"
echo " Scaling Study — Candidate Sweep"
echo " Base config : $GA_CONFIG"
echo " Dataset     : $GA_DATA"
echo " Counts      : ${CANDIDATE_COUNTS[*]}"
echo "============================================================"

for N in "${CANDIDATE_COUNTS[@]}"; do
    CONFIG_PATH="$CONFIG_DIR/scaling-n${N}.json"
    RUN_DIR="$GA_ROOT/runs/scaling-study-n${N}"

    echo ""
    echo "────────────────────────────────────────────────"
    echo " N = $N candidates  →  $RUN_DIR"
    echo "────────────────────────────────────────────────"

    # 1. Write config for this candidate count
    python3 - "$GA_CONFIG" "$CONFIG_PATH" "$N" <<'PY'
import json, sys
from pathlib import Path

src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = json.loads(Path(src).read_text())
cfg["solver"]["candidates"]           = n
cfg["solver"]["patches"]              = 64
cfg["solver"]["alignment_starts"]     = min(32, n)
cfg["solver"]["alignment_iterations"] = 20
cfg["solver"]["refine_evaluations"]   = 100
Path(dst).write_text(json.dumps(cfg, indent=2) + "\n")
print(f"  Wrote config: candidates={n}")
PY

    # 2. Run E0 + E3 on the FULL dev split (no --limit flag)
    python -m generative_assembly run \
        --config  "$CONFIG_PATH" \
        --dataset "$GA_DATA" \
        --root    "$RUN_DIR" \
        --split   dev \
        --stages  E0 E3

    # 3. Evaluate
    python -m generative_assembly evaluate \
        --config  "$CONFIG_PATH" \
        --dataset "$GA_DATA" \
        --root    "$RUN_DIR" \
        --split   dev

    echo "  ✅ Done — metrics at $RUN_DIR/evaluation/dev/metrics.jsonl"
done

echo ""
echo "============================================================"
echo " All runs complete!"
echo " Now open the notebook and run the scaling study cells."
echo "============================================================"

#!/bin/bash
set -e

GA_ROOT="/home/kpandey/generative_assembly"
GA_DATA_2PARTS="$GA_ROOT/bottles498-inputs-v1/dataset_2parts.json"
RUN_DIR="$GA_ROOT/runs/imagination-study"
BASE_CONFIG="$GA_ROOT/configs/sd15-v100-locked.json"
PATCHED_CONFIG="$GA_ROOT/configs/imagination-study-patched.json"

IMAGES_PYTHON="$GA_ROOT/envs/images/bin/python"
INSTANTMESH_PYTHON="$GA_ROOT/envs/instantmesh/bin/python"

echo "============================================================"
echo " Imagination Stage Study (E0, E1, E2)"
echo " Base config      : $BASE_CONFIG"
echo " Images python    : $IMAGES_PYTHON"
echo " InstantMesh py   : $INSTANTMESH_PYTHON"
echo " Dataset          : $GA_DATA_2PARTS"
echo " Run dir          : $RUN_DIR"
echo "============================================================"

# 1. Verify env pythons exist
if [ ! -f "$IMAGES_PYTHON" ]; then
    echo "❌ ERROR: Images env Python not found at $IMAGES_PYTHON"
    exit 1
fi
if [ ! -f "$INSTANTMESH_PYTHON" ]; then
    echo "❌ ERROR: InstantMesh env Python not found at $INSTANTMESH_PYTHON"
    exit 1
fi
echo "✅ Both Python envs found"

# Load .env if present (e.g. GEMINI_API_KEY)
if [ -f ~/satellite/pt_reg/.env ]; then
    export $(grep -v '^#' ~/satellite/pt_reg/.env | xargs)
    echo "✅ Loaded environment variables from .env"
fi

# 2. Patch the config — inject correct python paths for images + reconstruction
echo ""
echo "🔧 Patching config with correct Python env paths & experiment arms..."
python3 - <<PY
import json
from pathlib import Path

cfg = json.loads(Path("$BASE_CONFIG").read_text())

# Inject the images env python
if "images" not in cfg:
    cfg["images"] = {}
cfg["images"]["python"] = "$IMAGES_PYTHON"

# Inject the instantmesh env python
if "reconstruction" not in cfg:
    cfg["reconstruction"] = {}
cfg["reconstruction"]["python"] = "$INSTANTMESH_PYTHON"

# Compare imagination models in experiment:
# sd15_depth (with ControlNet) vs sd15 (pure inpainting without depth blocker)
cfg["image_models"] = ["sd15_depth", "sd15"]
cfg["primary_model"] = "sd15"
cfg["prompt_variant"] = "short"
cfg["prompt_short"] = "A smooth intact object, complete and undamaged, neutral gray CAD surface, isolated on seamless solid white background, single object, centered, no shadows."
cfg["bg_obj_threshold"] = 140

Path("$PATCHED_CONFIG").write_text(json.dumps(cfg, indent=2))
print(f"  images.python         = {cfg['images']['python']}")
print(f"  reconstruction.python = {cfg['reconstruction']['python']}")
print(f"  image_models          = {cfg['image_models']}")
print(f"  bg_obj_threshold      = {cfg['bg_obj_threshold']}")
print("✅ Patched config written to $PATCHED_CONFIG")
PY

# 3. Prepare the 2-part filtered dataset
echo ""
echo "🔍 Preparing 2-part dataset..."
python3 ~/satellite/pt_reg/prepare_2part_dataset.py

# 4. Remove old run to avoid 'Run identity changed' error
if [ -d "$RUN_DIR" ]; then
    echo ""
    echo "🗑️  Removing old run dir to avoid identity mismatch..."
    rm -rf "$RUN_DIR"
fi

# 5. Run stages one by one for clearer error reporting
echo ""
echo "🚀 Running E0 (Render)..."
python -m generative_assembly run \
    --config  "$PATCHED_CONFIG" \
    --dataset "$GA_DATA_2PARTS" \
    --root    "$RUN_DIR" \
    --split   dev \
    --stages  E0

echo ""
echo "🎨 Running E1 (Stable Diffusion image generation)..."
python -m generative_assembly run \
    --config  "$PATCHED_CONFIG" \
    --dataset "$GA_DATA_2PARTS" \
    --root    "$RUN_DIR" \
    --split   dev \
    --stages  E1

echo ""
echo "🧊 Running E2 (InstantMesh 3D reconstruction)..."
python -m generative_assembly run \
    --config  "$PATCHED_CONFIG" \
    --dataset "$GA_DATA_2PARTS" \
    --root    "$RUN_DIR" \
    --split   dev \
    --stages  E2

echo ""
echo "============================================================"
echo "✅ All stages complete!"
echo "   Now open: imagination_stage_analysis.ipynb"
echo "============================================================"

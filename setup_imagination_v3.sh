#!/usr/bin/env bash
set -euo pipefail
# Separate new environments; never upgrade the historical image/InstantMesh environments.
GA_ROOT="${GA_ROOT:-/home/kpandey/generative_assembly}"
PYTHON311="${PYTHON311:-python3.11}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON311" -c 'import sys; assert sys.version_info[:2]==(3,11), "Use Python 3.11 for isolated v3 environments"'
for name in images-v3 matte-v3; do
  [[ -f "$GA_ROOT/envs/$name/bin/python" ]] || "$PYTHON311" -m venv "$GA_ROOT/envs/$name"
done
"$GA_ROOT/envs/images-v3/bin/python" -m pip install 'torch==2.5.1' 'torchvision==0.20.1' --index-url https://download.pytorch.org/whl/cu121
"$GA_ROOT/envs/images-v3/bin/python" -m pip install \
  'diffusers==0.36.0' 'transformers==4.57.1' 'accelerate==1.11.0' \
  'numpy==1.26.4' 'scipy==1.13.1' 'Pillow==10.4.0' 'trimesh==4.4.9' \
  'omegaconf==2.3.0' 'psutil==6.1.0' 'safetensors==0.6.2' 'sentencepiece==0.2.1' 'rtree==1.3.0' \
  'huggingface-hub==0.36.0' 'pandas==2.2.3' 'plotly==5.24.1' 'ipykernel==6.29.5' 'nbconvert==7.16.6'
"$GA_ROOT/envs/matte-v3/bin/python" -m pip install 'rembg[cpu]==2.0.67' 'numpy==1.26.4' 'Pillow==10.4.0' 'scipy==1.13.1' 'onnxruntime==1.20.1'
mkdir -p "$GA_ROOT/environment-locks"
"$GA_ROOT/envs/images-v3/bin/python" -m pip freeze > "$GA_ROOT/environment-locks/images-v3.txt"
"$GA_ROOT/envs/matte-v3/bin/python" -m pip freeze > "$GA_ROOT/environment-locks/matte-v3.txt"
"$GA_ROOT/envs/images-v3/bin/python" -c 'import torch; from diffusers import QwenImageEditPlusPipeline,AutoPipelineForInpainting; print("CUDA:",torch.cuda.is_available(),"BF16:",torch.cuda.is_bf16_supported())'
"$GA_ROOT/envs/matte-v3/bin/python" -m generative_assembly.study_cycle lock-matte --out "$GA_ROOT/models/matte-v3"
"$GA_ROOT/envs/images-v3/bin/python" -m ipykernel install --user --name imagination-v3 --display-name 'Imagination v3 (isolated)'
printf '\nReady. Choose the Imagination v3 kernel; start with PHASE=pilot.\n'

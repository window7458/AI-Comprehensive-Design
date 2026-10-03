#!/usr/bin/env bash
# One-time environment setup on the Mondrian AI GPU server (any Linux + NVIDIA GPU container works).
#   bash scripts/setup_mondrian.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== NVIDIA driver =="
nvidia-smi || { echo "[x] nvidia-smi failed: no GPU attached to this container / driver missing"; exit 1; }

# Keep the server's own CUDA build of torch. Install one only if torch is missing or CPU-only.
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  DRV=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
  # pick the newest wheel the driver supports
  if   [ "${DRV:-0}" -ge 128 ]; then IDX=cu128
  elif [ "${DRV:-0}" -ge 126 ]; then IDX=cu126
  elif [ "${DRV:-0}" -ge 124 ]; then IDX=cu124
  else IDX=cu121; fi
  echo "[!] torch missing or CPU-only -> installing torch for $IDX (driver supports CUDA ${DRV:-?})"
  pip install --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/$IDX"
fi
pip install -r requirements.txt

# torch, CUDA, GPU + one AMP fwd/bwd of the distillation / fusion models on the GPU
python tools/check_gpu.py

# Pre-download the COCO-pretrained student so training does not stall on first use.
python -c "from ultralytics import YOLO; YOLO('yolo11s-seg.pt')"

if [ -z "${HF_TOKEN:-}" ]; then
  echo
  echo "[!] HF_TOKEN is not set. DINOv3 (B2/B3) is a gated model:"
  echo "    1) accept the licence at https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m"
  echo "    2) export HF_TOKEN=hf_xxx   (add it to ~/.bashrc to keep it)"
fi
echo "next: python tools/check_teachers.py   (downloads SigLIP2 / DINOv3 / RADIO and runs them on the GPU)"

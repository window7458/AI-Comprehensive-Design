#!/usr/bin/env bash
# One-time environment setup on the Mondrian AI GPU server (any Linux + CUDA container works).
#   bash scripts/setup_mondrian.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || {
  echo "PyTorch not found -> installing CUDA 12.x wheels"; pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124; }
pip install -r requirements.txt

# Pre-download the COCO-pretrained student so training does not stall on first use.
python -c "from ultralytics import YOLO; YOLO('yolo11s-seg.pt')"

if [ -z "${HF_TOKEN:-}" ]; then
  echo
  echo "[!] HF_TOKEN is not set. DINOv3 (B2/B3) is a gated model:"
  echo "    1) accept the licence at https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m"
  echo "    2) export HF_TOKEN=hf_xxx   (add it to ~/.bashrc to keep it)"
fi
nvidia-smi || true

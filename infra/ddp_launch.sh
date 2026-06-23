#!/usr/bin/env bash
# ============================================================
#  IndicLLM-Bharat-V1 — Multi-GPU DDP Launch Script
#  Scales from 1 GPU (g5.xlarge) to 8 GPUs (p4d.24xlarge)
#  using PyTorch DistributedDataParallel (DDP).
#
#  Usage (single node, 4 GPUs):
#    bash infra/ddp_launch.sh --gpus 4 --config configs/gpt2-350m.yaml
#
#  Usage (single GPU, fallback):
#    bash infra/ddp_launch.sh --gpus 1 --config configs/gpt2-124m.yaml
# ============================================================
set -euo pipefail

GPUS=1
CONFIG="configs/gpt2-124m.yaml"
PORT=29500

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)   GPUS="$2";   shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --port)   PORT="$2";   shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

echo "╔══════════════════════════════════════╗"
echo "║  IndicLLM DDP Training Launch        ║"
echo "║  GPUs    : ${GPUS}                         ║"
echo "║  Config  : ${CONFIG}   ║"
echo "╚══════════════════════════════════════╝"

if [ "$GPUS" -eq 1 ]; then
  echo "Single GPU mode — running directly"
  python train/pretrain.py --config "$CONFIG"
else
  echo "Multi-GPU mode — launching $GPUS processes via torchrun"
  torchrun \
    --standalone \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    train/pretrain_ddp.py \
    --config "$CONFIG"
fi

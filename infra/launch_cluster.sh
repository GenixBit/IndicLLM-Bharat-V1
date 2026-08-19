#!/usr/bin/env bash
# ==============================================================================
#  IndicLLM-Bharat-V1 — Sovereign Multi-GPU & Cluster Pretraining Launcher
#  Supports 350M, 1B, 3B, 7B, and 10B with DDP, FSDP, and DeepSpeed ZeRO-3
#
#  Usage Examples:
#    bash infra/launch_cluster.sh --model 1b --gpus 8 --backend fsdp
#    bash infra/launch_cluster.sh --model 10b --gpus 8 --backend deepspeed
#    bash infra/launch_cluster.sh --model 350m --gpus 4 --backend ddp
#    bash infra/launch_cluster.sh --model 1b --device mps --steps 50  # Apple Silicon
# ==============================================================================
set -euo pipefail

MODEL="1b"
GPUS=1
BACKEND="ddp"
STEPS=1000
PORT=29500
DEVICE="auto"
OUTPUT_DIR="checkpoints/bharat_cluster"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)    MODEL="$2";    shift 2 ;;
    --gpus)     GPUS="$2";     shift 2 ;;
    --backend)  BACKEND="$2";  shift 2 ;;
    --steps)    STEPS="$2";    shift 2 ;;
    --port)     PORT="$2";     shift 2 ;;
    --device)   DEVICE="$2";   shift 2 ;;
    --output)   OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG_FILE="configs/models/bharat-${MODEL}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "⚠️ Config $CONFIG_FILE not found! Falling back to bharat-1b.yaml"
  CONFIG_FILE="configs/models/bharat-1b.yaml"
fi

MODEL_UPPER=$(echo "$MODEL" | tr '[:lower:]' '[:upper:]')
BACKEND_UPPER=$(echo "$BACKEND" | tr '[:lower:]' '[:upper:]')

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  🇮🇳 IndicLLM-Bharat Sovereign Multi-GPU Cluster Pretrainer       ║"
echo "║  Model Tier : Bharat-${MODEL_UPPER}                                  ║"
echo "║  Config     : ${CONFIG_FILE}                                     ║"
echo "║  GPUs       : ${GPUS}                                            ║"
echo "║  Backend    : ${BACKEND_UPPER}                                       ║"
echo "║  Steps      : ${STEPS}                                           ║"
echo "║  Output     : ${OUTPUT_DIR}                                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# 1. Ensure data shards exist
if [[ ! -d "data/binary_shards" ]] || [[ -z "$(ls -A data/binary_shards 2>/dev/null)" ]]; then
  echo "==> Preparing binary data shards..."
  PYTHONPATH=. python3 scripts/prepare_mixture_shards.py
fi

# 2. Execution depending on backend and GPU count
if [[ "$GPUS" -eq 1 ]] || [[ "$DEVICE" == "mps" ]] || [[ "$DEVICE" == "cpu" ]]; then
  echo "==> Launching single-process pretrainer on device: ${DEVICE}..."
  PYTHONPATH=. python3 train/pretrain_bharat.py \
    --config "$CONFIG_FILE" \
    --max-steps "$STEPS" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE"
else
  echo "==> Launching multi-GPU distributed pretrainer across ${GPUS} GPUs..."
  torchrun \
    --standalone \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    train/pretrain_bharat.py \
    --config "$CONFIG_FILE" \
    --max-steps "$STEPS" \
    --output-dir "$OUTPUT_DIR" \
    --distributed
fi

echo "==> Pretraining job finished successfully."

#!/usr/bin/env bash
# Local sanity check: nanoGPT on Shakespeare (~5 min on M2)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate

NANO="${ROOT}/vendor/nanoGPT"
if [[ ! -d "$NANO" ]]; then
  bash scripts/setup.sh
fi

cd "$NANO"

# Prepare tiny dataset if missing
if [[ ! -f data/shakespeare_char/train.bin ]]; then
  python data/shakespeare_char/prepare.py
fi

# Short training run (MPS on Apple Silicon, CPU fallback)
python train.py config/train_shakespeare_char.py \
  --max_iters=500 \
  --eval_interval=100 \
  --log_interval=10 \
  --device=mps \
  --compile=False \
  || python train.py config/train_shakespeare_char.py \
  --max_iters=500 \
  --eval_interval=100 \
  --log_interval=10 \
  --device=cpu \
  --compile=False

echo ""
echo "Local sanity check passed. nanoGPT trained on Shakespeare for 500 iterations."
echo "Next: python data/prepare_data.py --subset sample-10BT --max-docs 100"

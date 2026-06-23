#!/usr/bin/env bash
# Environment setup for llm-lab
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "Python 3.11+ required."
  exit 1
fi

echo "Using $PYTHON ($($PYTHON --version))"

if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# PyTorch: CUDA on Linux cloud, MPS on Mac
if [[ "$(uname)" == "Darwin" ]]; then
  pip install torch torchvision torchaudio
else
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

pip install -r requirements.txt

# Clone nanoGPT for local sanity runs
if [[ ! -d vendor/nanoGPT ]]; then
  git clone --depth 1 https://github.com/karpathy/nanoGPT.git vendor/nanoGPT
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example — add WANDB_API_KEY when ready."
fi

echo "Setup complete. Activate with: source .venv/bin/activate"

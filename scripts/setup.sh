#!/usr/bin/env bash
# IndicLLM-Bharat — Developer & Training Environment Setup
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
  echo "Python 3.11+ is required."
  exit 1
fi

echo "=========================================================="
echo "  🇮🇳 IndicLLM-Bharat — Environment Setup"
echo "  Python: $PYTHON ($($PYTHON --version))"
echo "=========================================================="

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment at .venv..."
  "$PYTHON" -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# PyTorch: CUDA on Linux/cloud, MPS/CPU on macOS
if [[ "$(uname)" == "Darwin" ]]; then
  echo "Installing PyTorch for macOS (Apple Silicon MPS / CPU)..."
  pip install torch torchvision torchaudio
else
  echo "Installing PyTorch for Linux (CUDA 12.1)..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

echo "Installing IndicLLM-Bharat in editable development mode..."
pip install -e ".[dev]"

# Optional: clone nanoGPT for legacy comparisons
if [[ ! -d vendor/nanoGPT ]]; then
  echo "Cloning legacy nanoGPT vendor repository..."
  git clone --depth 1 https://github.com/karpathy/nanoGPT.git vendor/nanoGPT || true
fi

if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    cp .env.example .env
    echo "Created .env from .env.example."
  fi
fi

echo ""
echo "=========================================================="
echo "  ✅ Environment setup complete!"
echo "  Activate environment with: source .venv/bin/activate"
echo "  Run sanity check with    : python scripts/sanity_check.py --model bharat"
echo "=========================================================="

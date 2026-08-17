#!/usr/bin/env bash
# IndicLLM-Bharat — Local Fast Architecture Sanity Check
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

PYTHON=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "Python not found on PATH."
  exit 1
fi

echo "=========================================================="
echo "  🇮🇳 Running IndicLLM-Bharat Native Architecture Sanity Check"
echo "=========================================================="

"$PYTHON" scripts/sanity_check.py --model bharat --device auto --max-iters 10

echo ""
echo "=========================================================="
echo "  ✅ Native Bharat architecture sanity check passed!"
echo "  Next: python scripts/run_pipeline.py --config configs/pipeline/bharat-350m-e2e.yaml --dry-run"
echo "=========================================================="

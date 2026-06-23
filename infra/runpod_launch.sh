#!/usr/bin/env bash
# Launch pretraining on RunPod cloud GPUs.
# Prints setup checklist and generates a bootstrap script for the remote pod.
#
# Usage (local machine):
#   bash infra/runpod_launch.sh                          # 124M default
#   CONFIG=configs/gpt2-350m.yaml bash infra/runpod_launch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${CONFIG:-configs/gpt2-124m.yaml}"
REPO_URL="${REPO_URL:-}"

# ── Resolve model name and cost estimate from config ──────────────────────────
MODEL_NAME="$(python3 -c "import yaml,sys; c=yaml.safe_load(open('${CONFIG}')); print(c.get('name','model'))" 2>/dev/null || basename "${CONFIG}" .yaml)"
NUM_GPUS="$(python3 -c "import yaml,sys; c=yaml.safe_load(open('${CONFIG}')); print(c.get('cloud',{}).get('num_gpus',1))" 2>/dev/null || echo 1)"
GPU_TYPE="$(python3 -c "import yaml,sys; c=yaml.safe_load(open('${CONFIG}')); print(c.get('cloud',{}).get('gpu','A100-80GB'))" 2>/dev/null || echo 'A100-80GB')"
EST_COST="$(python3 -c "import yaml,sys; c=yaml.safe_load(open('${CONFIG}')); print(c.get('cloud',{}).get('estimated_cost_usd','?'))" 2>/dev/null || echo '?')"
EST_HRS="$(python3 -c "import yaml,sys; c=yaml.safe_load(open('${CONFIG}')); print(c.get('cloud',{}).get('estimated_hours','?'))" 2>/dev/null || echo '?')"

cat <<EOF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RunPod launch: ${MODEL_NAME}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Config   : ${CONFIG}
  GPUs     : ${NUM_GPUS}× ${GPU_TYPE}
  Est. time: ${EST_HRS} h  |  Est. cost: \$${EST_COST}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Create a pod
  → https://www.runpod.io/console/gpu-cloud
  Template : "RunPod PyTorch 2.x" (has CUDA + Python pre-installed)
  GPU      : ${NUM_GPUS}× ${GPU_TYPE}
  Volume   : mount a Network Volume at /workspace (persists across pod restarts)
  Disk     : 80 GB container disk minimum

STEP 2 — Open pod terminal (via RunPod web UI or SSH)

STEP 3 — Clone repo and run bootstrap
EOF

if [[ -n "$REPO_URL" ]]; then
  echo "  git clone ${REPO_URL} /workspace/llm-lab && cd /workspace/llm-lab"
else
  echo "  git clone <your-repo-url> /workspace/llm-lab && cd /workspace/llm-lab"
  echo "  (set REPO_URL env var to embed your URL here)"
fi

cat <<EOF

  # Set secrets before running
  export WANDB_API_KEY=<your-key>
  export HF_TOKEN=<your-key>   # if using gated HF datasets

  bash infra/runpod_bootstrap.sh

STEP 4 — Monitor
  # Tail logs (in a second terminal)
  tail -f /workspace/llm-lab/checkpoints/train.log

  # W&B dashboard
  https://wandb.ai/\${WANDB_ENTITY:-your-entity}/llm-lab

STEP 5 — Stop the pod when done (not just pause — stop to stop billing)
  Checkpoints on the Network Volume survive; container disk is wiped.

COST TIPS
  • Use "Spot" pods for ~50% savings; they can be preempted — checkpoint frequently
  • eval_interval in your config controls checkpoint frequency; default is every 500 iters
  • Use STOP (not TERMINATE) to keep your volume intact
  • RunPod charges by the second once a pod is running

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF

# ── Generate bootstrap script for the remote pod ─────────────────────────────
BOOT="${ROOT}/infra/runpod_bootstrap.sh"
cat > "$BOOT" <<SCRIPT
#!/usr/bin/env bash
# Run this inside the RunPod pod after cloning the repo.
set -euo pipefail
cd "\$(dirname "\$0")/.."

echo "==> Setting up environment..."
bash scripts/setup.sh

echo "==> Preparing data..."
MAX_DOCS="\${MAX_DOCS:-}"
if [[ -n "\$MAX_DOCS" ]]; then
  python data/prepare_data.py --subset sample-10BT --max-docs "\$MAX_DOCS"
else
  python data/prepare_data.py --subset sample-10BT
fi

echo "==> Starting pretraining..."
CONFIG="\${CONFIG:-${CONFIG}}"
python train/pretrain.py --config "\$CONFIG" 2>&1 | tee checkpoints/train.log

echo "==> Training complete."
echo "    Checkpoints saved to: checkpoints/"
echo "    If using a Network Volume, they are already persisted."
SCRIPT
chmod +x "$BOOT"
echo "Wrote bootstrap script → infra/runpod_bootstrap.sh"

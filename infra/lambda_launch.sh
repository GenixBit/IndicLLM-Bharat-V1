#!/usr/bin/env bash
# Launch pretraining on Lambda Labs cloud GPUs.
# Prints setup checklist and generates a bootstrap script for the remote instance.
#
# Usage (local machine):
#   bash infra/lambda_launch.sh                          # 124M default
#   CONFIG=configs/gpt2-350m.yaml bash infra/lambda_launch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${CONFIG:-configs/gpt2-124m.yaml}"
REPO_URL="${REPO_URL:-}"   # set to your git remote, e.g. git@github.com:you/llm-lab.git

# ── Resolve model name and cost estimate from config ──────────────────────────
MODEL_NAME="$(python3 -c "import yaml,sys; c=yaml.safe_load(open('${CONFIG}')); print(c.get('name','model'))" 2>/dev/null || basename "${CONFIG}" .yaml)"
NUM_GPUS="$(python3 -c "import yaml,sys; c=yaml.safe_load(open('${CONFIG}')); print(c.get('cloud',{}).get('num_gpus',1))" 2>/dev/null || echo 1)"
GPU_TYPE="$(python3 -c "import yaml,sys; c=yaml.safe_load(open('${CONFIG}')); print(c.get('cloud',{}).get('gpu','A100-80GB'))" 2>/dev/null || echo 'A100-80GB')"
EST_COST="$(python3 -c "import yaml,sys; c=yaml.safe_load(open('${CONFIG}')); print(c.get('cloud',{}).get('estimated_cost_usd','?'))" 2>/dev/null || echo '?')"
EST_HRS="$(python3 -c "import yaml,sys; c=yaml.safe_load(open('${CONFIG}')); print(c.get('cloud',{}).get('estimated_hours','?'))" 2>/dev/null || echo '?')"

cat <<EOF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Lambda Labs launch: ${MODEL_NAME}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Config   : ${CONFIG}
  GPUs     : ${NUM_GPUS}× ${GPU_TYPE}
  Est. time: ${EST_HRS} h  |  Est. cost: \$${EST_COST}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Reserve a GPU instance
  → https://cloud.lambdalabs.com/instances
  GPU: ${GPU_TYPE}  (prefer 1-Click Cluster for multi-node)
  Storage: attach a persistent filesystem at /home/ubuntu/llm-lab-fs
           (avoids losing checkpoints when instance terminates)

STEP 2 — SSH into the instance
  lambda ssh <instance-name>           # if using Lambda CLI
  ssh ubuntu@<instance-ip>             # raw SSH

STEP 3 — Clone repo and run bootstrap
EOF

if [[ -n "$REPO_URL" ]]; then
  echo "  git clone ${REPO_URL} && cd llm-lab"
else
  echo "  git clone <your-repo-url> && cd llm-lab"
  echo "  (set REPO_URL env var to embed your URL here)"
fi

cat <<EOF

  bash infra/lambda_bootstrap.sh

STEP 4 — Monitor
  # Tail logs
  tail -f checkpoints/${MODEL_NAME}/train.log

  # W&B dashboard (if WANDB_API_KEY is set)
  https://wandb.ai/\${WANDB_ENTITY:-your-entity}/llm-lab

STEP 5 — Download checkpoints BEFORE terminating
  rsync -avz ubuntu@<ip>:~/llm-lab/checkpoints/ ./checkpoints/
  # or copy to Lambda persistent storage so they survive instance deletion

COST TIPS
  • Lambda spot instances are ~30–40% cheaper; checkpoint every eval_interval
  • Use a persistent filesystem (\$0.20/GB/month) for checkpoints — cheaper
    than re-downloading if instance crashes
  • Kill idle instances; Lambda charges by the second

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF

# ── Generate bootstrap script for the remote pod ─────────────────────────────
BOOT="${ROOT}/infra/lambda_bootstrap.sh"
cat > "$BOOT" <<SCRIPT
#!/usr/bin/env bash
# Run this on the Lambda Labs instance after cloning the repo.
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

echo "==> Training complete. Copy checkpoints before terminating instance:"
echo "    rsync -avz ubuntu@<ip>:~/llm-lab/checkpoints/ ./checkpoints/"
SCRIPT
chmod +x "$BOOT"
echo "Wrote bootstrap script → infra/lambda_bootstrap.sh"
echo "Upload it with: scp infra/lambda_bootstrap.sh ubuntu@<ip>:~/llm-lab/infra/"

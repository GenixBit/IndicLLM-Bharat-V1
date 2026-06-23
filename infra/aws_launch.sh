#!/usr/bin/env bash
# ============================================================
#  IndicLLM-Bharat-V1 — AWS GPU Instance Launcher
#  Launches a g5.xlarge (A10G 24GB) spot instance on AWS,
#  sets up the full training environment, and starts the
#  124M parameter pretrain run.
#
#  Usage:
#    bash infra/aws_launch.sh
#    CONFIG=configs/gpt2-350m.yaml bash infra/aws_launch.sh
#
#  Requirements: aws CLI configured (aws configure)
# ============================================================
set -euo pipefail

# ── Config ───────────────────────────────────────────────────
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
# AZ auto-selected by AWS for best spot capacity
AMI_ID="ami-012ba162b9cd2729c"                  # DL PyTorch 2.7 Ubuntu 22.04
SPOT_MAX_PRICE="${SPOT_MAX_PRICE:-0.80}"        # max $/hr for spot (on-demand = $1.006)
KEY_NAME="indicllm-key"
KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"
SG_NAME="indicllm-sg"
INSTANCE_NAME="indicllm-124m-train"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${CONFIG:-configs/gpt2-124m.yaml}"
REPO_URL="https://github.com/GenixBit/IndicLLM-Bharat-V1.git"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
S3_BUCKET="indicllm-checkpoints-${ACCOUNT_ID}"
VOLUME_SIZE=100   # GB root EBS

# Colours
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   IndicLLM-Bharat-V1 — AWS GPU Launch               ║"
echo "║   Instance : ${INSTANCE_TYPE}  (A10G 24GB GPU)             ║"
echo "║   AMI      : ${AMI_ID}                    ║"
echo "║   Region   : ${REGION} (auto AZ)                      ║"
echo "║   Spot max : \$${SPOT_MAX_PRICE}/hr  (on-demand = \$1.006/hr)    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Step 1: Key Pair ─────────────────────────────────────────
echo -e "${GREEN}[1/7] Key pair...${NC}"
if [[ ! -f "$KEY_PATH" ]]; then
  echo "  Creating key pair '${KEY_NAME}'..."
  aws ec2 create-key-pair \
    --key-name "$KEY_NAME" \
    --region "$REGION" \
    --query "KeyMaterial" \
    --output text > "$KEY_PATH"
  chmod 400 "$KEY_PATH"
  echo "  ✅ Saved to ${KEY_PATH}"
else
  echo "  ✅ Key already exists at ${KEY_PATH}"
  # Ensure it exists in AWS too
  aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" \
    --output text > /dev/null 2>&1 || {
    echo "  Re-importing public key to AWS..."
    aws ec2 import-key-pair \
      --key-name "$KEY_NAME" \
      --public-key-material "$(ssh-keygen -y -f "$KEY_PATH" | base64)" \
      --region "$REGION" > /dev/null
  }
fi

# ── Step 2: Security Group ───────────────────────────────────
echo -e "${GREEN}[2/7] Security group...${NC}"
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" \
  --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text)

SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters "Name=group-name,Values=${SG_NAME}" \
  --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "None")

if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "IndicLLM training - SSH access" \
    --vpc-id "$VPC_ID" \
    --region "$REGION" \
    --query "GroupId" --output text)
  # SSH from anywhere (restrict to your IP in production)
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp --port 22 --cidr 0.0.0.0/0 \
    --region "$REGION" > /dev/null
  # Jupyter (optional)
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp --port 8888 --cidr 0.0.0.0/0 \
    --region "$REGION" > /dev/null
  echo "  ✅ Created security group ${SG_ID}"
else
  echo "  ✅ Reusing security group ${SG_ID}"
fi

# ── Step 3: S3 Bucket for Checkpoints ───────────────────────
echo -e "${GREEN}[3/7] S3 checkpoint bucket...${NC}"
if aws s3 ls "s3://${S3_BUCKET}" --region "$REGION" > /dev/null 2>&1; then
  echo "  ✅ Bucket already exists: s3://${S3_BUCKET}"
else
  aws s3 mb "s3://${S3_BUCKET}" --region "$REGION"
  aws s3api put-bucket-versioning \
    --bucket "$S3_BUCKET" \
    --versioning-configuration Status=Enabled \
    --region "$REGION"
  echo "  ✅ Created s3://${S3_BUCKET}"
fi

# ── Step 4: User-data bootstrap script ──────────────────────
echo -e "${GREEN}[4/7] Generating user-data bootstrap...${NC}"
USER_DATA=$(cat <<USERDATA
#!/bin/bash
set -e
exec > /home/ubuntu/bootstrap.log 2>&1

echo "==> Bootstrap started at \$(date)"

# Wait for nvidia driver
for i in {1..10}; do
  nvidia-smi && break || sleep 10
done

cd /home/ubuntu
git clone ${REPO_URL} IndicLLM-Bharat-V1
cd IndicLLM-Bharat-V1

# Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install torch>=2.2.0 transformers datasets tokenizers tqdm numpy \
            wandb pyyaml peft trl accelerate safetensors huggingface-hub \
            fastapi uvicorn pydantic python-dotenv --quiet

echo "==> Deps installed"

# Data pipeline
python data/prepare_data.py --subset sample-10BT --max-docs 50000 \
  --out-dir data/shards

echo "==> Data ready — starting training"

# Training
nohup python train/pretrain.py \
  --config configs/gpt2-124m.yaml \
  2>&1 | tee checkpoints/gpt2-124m/train.log &

echo "==> Training started (PID \$!)"
echo "==> Check logs: tail -f ~/IndicLLM-Bharat-V1/checkpoints/gpt2-124m/train.log"
USERDATA
)

# ── Step 5: Launch Spot Instance ────────────────────────────
echo -e "${GREEN}[5/7] Launching spot instance...${NC}"

# On-demand launch (spot quota exceeded on this account)
# To enable spot: request quota increase at AWS Service Quotas for "Running On-Demand G instances"
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_SIZE},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":false}}]" \
  --user-data "$USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}},{Key=Project,Value=IndicLLM-Bharat-V1}]" \
  --region "$REGION" \
  --query "Instances[0].InstanceId" \
  --output text)

echo "  ✅ Instance launched: ${INSTANCE_ID}"

# ── Step 6: Wait for public IP ──────────────────────────────
echo -e "${GREEN}[6/7] Waiting for instance to get public IP...${NC}"
echo "  (This usually takes 30–60 seconds)"
sleep 15

for i in {1..20}; do
  PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text 2>/dev/null || echo "")
  if [[ -n "$PUBLIC_IP" && "$PUBLIC_IP" != "None" ]]; then
    break
  fi
  echo "  Waiting... ($i/20)"
  sleep 10
done

# Save state file for teardown script
# Resolve actual AZ used
ACTUAL_AZ=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query "Reservations[0].Instances[0].Placement.AvailabilityZone" \
  --output text 2>/dev/null || echo "unknown")
echo "  AZ assigned: ${ACTUAL_AZ}"

STATE_FILE="${ROOT}/infra/.aws_instance_state"
cat > "$STATE_FILE" <<STATE
INSTANCE_ID=${INSTANCE_ID}
PUBLIC_IP=${PUBLIC_IP}
KEY_PATH=${KEY_PATH}
REGION=${REGION}
S3_BUCKET=${S3_BUCKET}
LAUNCHED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
CONFIG=${CONFIG}
STATE
echo "  ✅ State saved to ${STATE_FILE}"

# ── Step 7: Print access instructions ───────────────────────
echo -e "${GREEN}[7/7] Done!${NC}"
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════╗"
echo "║   🚀 IndicLLM AWS Training Instance is LIVE!         ║"
echo "╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}━━━ Instance Info ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  Instance ID : ${INSTANCE_ID}"
echo "  Public IP   : ${PUBLIC_IP}"
echo "  Instance    : ${INSTANCE_TYPE} (A10G 24GB GPU)"
echo "  Cost        : ~\$0.48/hr spot (max \$${SPOT_MAX_PRICE}/hr)"
echo "  Region      : ${REGION} / ${AZ}"
echo "  S3 Bucket   : s3://${S3_BUCKET}"
echo ""
echo -e "${YELLOW}━━━ How to Access ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  # 1. SSH into the GPU server:"
echo "  ssh -i ${KEY_PATH} ubuntu@${PUBLIC_IP}"
echo ""
echo "  # 2. Wait ~5 mins for bootstrap then watch training logs:"
echo "  ssh -i ${KEY_PATH} ubuntu@${PUBLIC_IP} 'tail -f ~/IndicLLM-Bharat-V1/checkpoints/gpt2-124m/train.log'"
echo ""
echo "  # 3. Check GPU usage:"
echo "  ssh -i ${KEY_PATH} ubuntu@${PUBLIC_IP} 'watch -n1 nvidia-smi'"
echo ""
echo "  # 4. Bootstrap log (setup progress):"
echo "  ssh -i ${KEY_PATH} ubuntu@${PUBLIC_IP} 'tail -f ~/bootstrap.log'"
echo ""
echo "  # 5. Download checkpoints when done:"
echo "  rsync -avz -e 'ssh -i ${KEY_PATH}' ubuntu@${PUBLIC_IP}:~/IndicLLM-Bharat-V1/checkpoints/ ./checkpoints/"
echo ""
echo "  # 6. Stop instance when done (preserves EBS, stops billing):"
echo "  bash infra/aws_teardown.sh"
echo ""
echo -e "${YELLOW}━━━ Training Details ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  Config     : ${CONFIG}"
echo "  Model      : GPT-2 124M (12L / 768d / 12H)"
echo "  Dataset    : FineWeb-Edu 50k docs (51M tokens)"
echo "  Est. time  : 8–12 hours on A10G"
echo "  Est. cost  : ~\$5–10 total (spot)"
echo ""
echo -e "${YELLOW}━━━ AWS Console Link ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  https://console.aws.amazon.com/ec2/home?region=${REGION}#Instances:instanceId=${INSTANCE_ID}"
echo ""

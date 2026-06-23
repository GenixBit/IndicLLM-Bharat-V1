#!/usr/bin/env bash
# ============================================================
#  IndicLLM-Bharat-V1 — AWS Teardown Script
#  Syncs checkpoints to S3, then stops the instance.
#  Run this when training is done or when you want to pause.
#
#  Usage:
#    bash infra/aws_teardown.sh          # stop instance (preserves EBS)
#    bash infra/aws_teardown.sh --terminate   # DESTROY everything
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE_FILE="${ROOT}/infra/.aws_instance_state"
TERMINATE="${1:-}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

if [[ ! -f "$STATE_FILE" ]]; then
  echo -e "${RED}No state file found at ${STATE_FILE}${NC}"
  echo "Run  bash infra/aws_launch.sh  first to create an instance."
  exit 1
fi

source "$STATE_FILE"

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   IndicLLM-Bharat-V1 — AWS Teardown                 ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "  Instance : ${INSTANCE_ID}"
echo "  IP       : ${PUBLIC_IP}"
echo "  Launched : ${LAUNCHED_AT}"
echo "  S3       : s3://${S3_BUCKET}"
echo ""

# ── Sync checkpoints to S3 ───────────────────────────────────
echo -e "${GREEN}[1/3] Syncing checkpoints to S3...${NC}"
if ssh -i "${KEY_PATH}" -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
     ubuntu@"${PUBLIC_IP}" "ls ~/IndicLLM-Bharat-V1/checkpoints/" > /dev/null 2>&1; then
  ssh -i "${KEY_PATH}" -o StrictHostKeyChecking=no ubuntu@"${PUBLIC_IP}" \
    "aws s3 sync ~/IndicLLM-Bharat-V1/checkpoints/ s3://${S3_BUCKET}/checkpoints/ --region ${REGION}"
  echo "  ✅ Checkpoints synced to s3://${S3_BUCKET}/checkpoints/"
else
  echo -e "  ${YELLOW}Could not SSH — skipping remote S3 sync${NC}"
fi

# Also pull checkpoints locally
echo -e "${GREEN}[2/3] Pulling checkpoints locally...${NC}"
rsync -avz --ignore-errors \
  -e "ssh -i ${KEY_PATH} -o StrictHostKeyChecking=no" \
  ubuntu@"${PUBLIC_IP}":~/IndicLLM-Bharat-V1/checkpoints/ \
  "${ROOT}/checkpoints/" 2>/dev/null || echo "  (Could not rsync — instance may be stopping)"

# ── Stop or terminate ────────────────────────────────────────
echo -e "${GREEN}[3/3] Stopping instance...${NC}"
if [[ "$TERMINATE" == "--terminate" ]]; then
  echo -e "  ${RED}TERMINATING instance ${INSTANCE_ID} (EBS will be deleted!)${NC}"
  read -r -p "  Are you sure? (yes/no): " confirm
  if [[ "$confirm" == "yes" ]]; then
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
    rm -f "$STATE_FILE"
    echo "  ✅ Terminated."
  else
    echo "  Cancelled."
  fi
else
  aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" > /dev/null
  echo "  ✅ Instance STOPPED (EBS preserved, billing paused)"
  echo ""
  echo "  To restart later:"
  echo "    aws ec2 start-instances --instance-ids ${INSTANCE_ID} --region ${REGION}"
  echo "  To fully terminate (delete EBS too):"
  echo "    bash infra/aws_teardown.sh --terminate"
fi
echo ""
echo "  S3 checkpoints: https://s3.console.aws.amazon.com/s3/buckets/${S3_BUCKET}"

#!/usr/bin/env bash
# Rare-Link AI Step Functions deploy
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-dev}"
STACK_NAME="rare-link-pipeline-${STAGE}"
REGION="ap-northeast-2"

cd "$HERE"

echo "============================================================"
echo " Rare-Link AI Step Functions 배포"
echo " Stage : ${STAGE}"
echo " Stack : ${STACK_NAME}"
echo "============================================================"

echo "[1] sam build"
sam build

echo "[2] sam deploy"
sam deploy \
  --region "$REGION" \
  --stack-name "$STACK_NAME" \
  --parameter-overrides "Stage=${STAGE}" \
  --capabilities CAPABILITY_IAM \
  --no-confirm-changeset \
  --resolve-s3 \
  --no-fail-on-empty-changeset \
  --tags project=pre-2-2team

echo
echo "[3] outputs:"
aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query 'Stacks[0].Outputs' \
  --output table

echo
echo "[Done] 테스트 실행 예:"
echo "  aws stepfunctions start-execution \\"
echo "    --state-machine-arn arn:aws:states:${REGION}:\$AWS_ACCOUNT_ID:stateMachine:${STACK_NAME} \\"
echo "    --input '{\"session_id\":\"<UUID>\",\"patient_id\":\"<id>\"}' \\"
echo "    --region ${REGION}"

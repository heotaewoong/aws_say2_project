#!/bin/bash
# deploy.sh — Phase 5 SAM 배포
# 작성자: AWS SAY2기 권미라
# 작성일: 2026-05-12
# 근거: Phase 3 deploy.sh 패턴 동일 적용

set -e

STAGE="${1:-dev}"
REGION="${2:-ap-northeast-2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Phase 5 SAM 배포 ==="
echo "  Stage:  $STAGE"
echo "  Region: $REGION"
echo ""

# VPC 설정 (허태웅 확인 후 채우기)
# 현재는 환경변수에서 읽음
VPC_SUBNET_IDS="${PHASE5_VPC_SUBNET_IDS:-}"
VPC_SG_IDS="${PHASE5_VPC_SG_IDS:-}"

if [ -z "$VPC_SUBNET_IDS" ] || [ -z "$VPC_SG_IDS" ]; then
    echo "⚠️  VPC 설정 없음 — Aurora 연결 불가"
    echo "   PHASE5_VPC_SUBNET_IDS, PHASE5_VPC_SG_IDS 환경변수 설정 필요"
    echo "   (허태웅에게 확인)"
    echo ""
fi

# 1. Layer 빌드
echo "[1/3] Lambda Layer 빌드..."
cd "$SCRIPT_DIR/layer"
bash build_layer.sh
cd "$SCRIPT_DIR"

# 2. SAM 빌드
echo ""
echo "[2/3] SAM 빌드..."
sam build \
    --template-file template.yaml \
    --build-dir .aws-sam/build

# 3. SAM 배포
echo ""
echo "[3/3] SAM 배포..."
sam deploy \
    --template-file .aws-sam/build/template.yaml \
    --stack-name "phase5-lr-scorer-$STAGE" \
    --region "$REGION" \
    --capabilities CAPABILITY_IAM \
    --no-confirm-changeset \
    --parameter-overrides \
        Stage="$STAGE" \
        VpcSubnetIds="${VPC_SUBNET_IDS}" \
        VpcSecurityGroupIds="${VPC_SG_IDS}"

echo ""
echo "✅ Phase 5 배포 완료"
echo ""

# 배포 결과 출력
aws cloudformation describe-stacks \
    --stack-name "phase5-lr-scorer-$STAGE" \
    --region "$REGION" \
    --query "Stacks[0].Outputs" \
    --output table

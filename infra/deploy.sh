#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# RAG 파이프라인 전체 AWS 배포 자동화 스크립트
# (00-simple-deploy.yaml 스택 + Lambda 컨테이너 이미지)
#
# 사용법:
#   bash infra/deploy.sh            # 전체 배포
#   bash infra/deploy.sh destroy    # 전체 삭제
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ── 설정 ──────────────────────────────────────────────────
REGION=${AWS_DEFAULT_REGION:-ap-northeast-2}
PROJECT=rare-link-simple
STACK_NAME=${PROJECT}-stack
REPO_NAME=${PROJECT}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest

# ── 삭제 모드 ─────────────────────────────────────────────
if [[ "${1:-deploy}" == "destroy" ]]; then
    echo "🗑️  스택 삭제 중..."
    aws cloudformation delete-stack --stack-name ${STACK_NAME} --region ${REGION}
    aws cloudformation wait stack-delete-complete --stack-name ${STACK_NAME} --region ${REGION}
    aws ecr delete-repository --repository-name ${REPO_NAME} --force --region ${REGION} 2>/dev/null || true
    echo "✅ 전체 삭제 완료"
    exit 0
fi

# ── 1단계: ECR 레포지토리 생성 ─────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/4] ECR 레포지토리 생성"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${REGION} >/dev/null 2>&1 || \
    aws ecr create-repository \
        --repository-name ${REPO_NAME} \
        --region ${REGION} \
        --tags Key=project,Value=pre-ecr-2-2-team

# ── 2단계: Docker 이미지 빌드 + 푸시 ─────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/4] Lambda 컨테이너 이미지 빌드 + 푸시"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
aws ecr get-login-password --region ${REGION} | \
    docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

docker build --platform linux/amd64 -t ${REPO_NAME} -f infra/lambda/Dockerfile .
docker tag ${REPO_NAME}:latest ${ECR_URI}
docker push ${ECR_URI}

# ── 3단계: CloudFormation 배포 ────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/4] CloudFormation 스택 배포"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
aws cloudformation deploy \
    --stack-name ${STACK_NAME} \
    --template-file infra/cloudformation/00-simple-deploy.yaml \
    --parameter-overrides \
        ProjectName=${PROJECT} \
        EcrImageUri=${ECR_URI} \
    --capabilities CAPABILITY_IAM \
    --region ${REGION}

# ── 4단계: Output 출력 ────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/4] 배포 완료 — 엔드포인트 확인"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME} \
    --query "Stacks[0].Outputs" \
    --output table \
    --region ${REGION}

echo ""
echo "✅ 배포 완료!"
echo ""
echo "🧪 테스트 방법:"
echo "   API_URL=\$(aws cloudformation describe-stacks --stack-name ${STACK_NAME} --query \"Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue\" --output text)"
echo "   BUCKET=\$(aws cloudformation describe-stacks --stack-name ${STACK_NAME} --query \"Stacks[0].Outputs[?OutputKey=='XrayBucketName'].OutputValue\" --output text)"
echo ""
echo "   # X-ray 업로드"
echo "   aws s3 cp /tmp/test_xray.jpg s3://\$BUCKET/uploads/test.jpg"
echo ""
echo "   # API 호출"
echo "   curl -X POST \$API_URL -H 'Content-Type: application/json' -d '{"
echo "     \"xray_s3_key\": \"uploads/test.jpg\","
echo "     \"patient_info\": {\"age\": 40, \"sex\": \"F\"},"
echo "     \"symptom_text\": \"호흡곤란, 흉통\","
echo "     \"lab_results\": {\"WBC\": 12.5, \"SpO2\": 92}"
echo "   }'"

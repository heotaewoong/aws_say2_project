#!/bin/bash
# build_layer.sh — Phase 5 Lambda Layer 빌드
# 작성자: AWS SAY2기 권미라
# 작성일: 2026-05-12
# 근거: Phase 3 layer/build_layer.sh 패턴 동일 적용

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE5_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$PHASE5_DIR")")")"
LAYER_OUT="$SCRIPT_DIR/python"

echo "=== Phase 5 Lambda Layer 빌드 ==="
echo "  프로젝트 루트: $PROJECT_ROOT"

# 기존 빌드 정리
rm -rf "$LAYER_OUT"
mkdir -p "$LAYER_OUT"

# 1. psycopg2-binary (Aurora PostgreSQL 연결)
echo "[1/3] psycopg2-binary 설치..."
pip install psycopg2-binary \
    --target "$LAYER_OUT" \
    --platform manylinux2014_x86_64 \
    --python-version 3.11 \
    --only-binary=:all: \
    --quiet

# 2. PyYAML (YAML KB 로드)
echo "[2/3] PyYAML 설치..."
pip install pyyaml \
    --target "$LAYER_OUT" \
    --platform manylinux2014_x86_64 \
    --python-version 3.11 \
    --only-binary=:all: \
    --quiet

# 3. YAML KB 파일 복사 (S3 fallback용 로컬 번들)
echo "[3/3] 희귀질환 KB 복사..."
YAML_SRC="$PROJECT_ROOT/data/rare_disease_profiles_v3_1.yaml"
if [ -f "$YAML_SRC" ]; then
    cp "$YAML_SRC" "$LAYER_OUT/"
    echo "  KB 복사 완료: $(du -h "$LAYER_OUT/rare_disease_profiles_v3_1.yaml" | cut -f1)"
else
    echo "  ⚠️  YAML 파일 없음: $YAML_SRC"
    echo "     Lambda 환경에서 S3에서 로드합니다."
fi

echo ""
echo "✅ Layer 빌드 완료"
echo "  출력: $LAYER_OUT"
echo "  크기: $(du -sh "$LAYER_OUT" | cut -f1)"

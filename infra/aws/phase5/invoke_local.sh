#!/bin/bash
# invoke_local.sh — Phase 5 로컬 테스트
# 작성자: AWS SAY2기 권미라
# 작성일: 2026-05-12
# 주의: DB 연결 없이 로컬 테스트 시 DB_MOCK=true 환경변수 설정 필요

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Phase 5 로컬 테스트 ==="

# sam local invoke
sam local invoke Phase5LRScorer \
    --template-file "$SCRIPT_DIR/template.yaml" \
    --event "$SCRIPT_DIR/events/sample_event.json" \
    --env-vars "$SCRIPT_DIR/events/local_env.json" \
    --region ap-northeast-2


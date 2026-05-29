# 대화 히스토리 — Rare-Link AI 에러 처리 분석 세션

> 세션 날짜: 2026-05-17
> 사용자: sungsoo.park.bio@gmail.com
> 작업 디렉토리: `C:\Users\tjdtn\Documents\rare-link-ai-frontend`
> 참조: [ERROR_HANDLING_REPORT.md](./ERROR_HANDLING_REPORT.md)

이 파일은 향후 새로운 Claude 세션에서 작업 맥락을 빠르게 복원할 수 있도록 대화의 흐름과 결정사항을 정리한 것입니다.

---

## 1. 세션 목표

S3 버킷 `say2-2team-bucket`의 Rare-Link AI 파이프라인(Phase 1~5 + RAG) 에서 에러 처리 방식을 통일하기. 특히 **Phase 3/4/5의 에러를 DB에 저장**할 수 있도록 설계.

---

## 2. 작업 흐름 (시간순)

### 단계 1: S3 → 로컬 클론
- `aws s3 sync` 로 7개 폴더 다운로드: `Phase_1`~`Phase_5`, `RAG`, `lung_dx`
- 추후 `frontend`, `deploy` 추가
- Phase_5 는 작업 중 새 파일 6개 추가됨 → 재 sync로 반영
- `Frontend` (대문자 1.6 MiB) vs `frontend` (소문자 11.1 MiB) 중 **소문자 선택**

### 단계 2: 에러 처리 현황 분석
- **Phase 1, 2**: `_log_error()` 메서드 있음. 로컬 `logs/*.json` 파일에만 저장. DB 연동 ❌
- **RAG**: `_log_error()` (파일) + `_mark_session_failed()` (DB UPDATE) 둘 다 있음. `diagnosis_session.error_message` 컬럼에 업데이트
- **Phase 3, 4**: Lambda. `logger.exception()` 으로 CloudWatch에만 기록. DB 연동 ❌
- **Phase 5**: RAG 래퍼 Lambda. 내부에서 RAG의 `_mark_session_failed` 호출됨. 단 cold-start 실패 시 DB 누락

### 단계 3: 1차 보고서 작성 → 저장
- ERROR_HANDLING_REPORT.md 생성
- Phase 3/4/5 핸들러에 `_get_db_conn()`, `_mark_session_failed()`, `_set_session_running()` 헬퍼 추가하는 패치안 포함
- RAG 패턴(`diagnosis_session` UPDATE) 그대로 따르는 방향

### 단계 4: "왜" 와 "after" 추가 요청
- 보고서에 §1.5 (5가지 문제점, RAG 패턴을 따르는 이유) 추가
- §6 (Before/After 5개 시나리오 + 잠재적 부작용) 추가

### 단계 5: 사용자가 "DB에 에러 전체 저장 원함" 의향 표명
- 폴링 vs DB 저장 개념 정리 (별개)
- 3가지 옵션 제안:
  - A: RAG 패턴 유지 (요약만)
  - B: 새 `phase_error_log` 테이블 신설 ← 추천
  - C: `diagnosis_session`에 JSONB 컬럼 추가

### 단계 6: 사용자 "이미 에러 컬럼 만든 걸로 기억" 발언 → 검증
- S3 `database/` 폴더 발견
- `4-layer-schema-ddl-v1.sql`, `4-layer-schema-ddl-v1.1.sql`, **`system-log-schema-ddl.sql`** 다운로드
- **🎯 `phase_execution_log` 테이블이 이미 완벽하게 설계되어 있음을 확인**
  - `error_code`, `error_message`, `error_stacktrace`, `error_category` 컬럼 모두 존재
  - View: `recent_errors`, `phase_success_rates_24h` 미리 정의됨
- DDL은 Aurora에 배포 완료 (`4-layer-db-team-guide.md`: "스키마 생성 완료 ✅")
- 그러나 **현재 코드 어디서도 `phase_execution_log` INSERT 안 함** (grep으로 확인)
- → 권장사항을 "옵션 B(신규 테이블)" → "옵션 B'(기존 테이블 활용)" 로 변경

### 단계 7: 최종 보고서 업데이트 + 본 히스토리 파일 작성

---

## 3. 최종 결정사항

### 채택된 접근
- **`phase_execution_log` 테이블 활용** (이미 존재, INSERT만 추가)
- Phase 1~5 공통 헬퍼 `_record_phase_log()` 작성
- 기존 `diagnosis_session.error_message` 업데이트도 병행 (세션 단위 상태 / 단계 단위 디테일)

### 보류된 결정 (다음 세션에서 진행)
1. **검증 스크립트 실행** — `check_phase_log.py` 만들어 실제 DB에 테이블 있는지 + 행 수 확인 (psycopg2 + VPC 접근 필요)
2. **Phase 3 핸들러 실제 패치** — 보고서대로 코드 변경 + SAM 배포
3. **Phase 4, 5 동일 작업**
4. **Lambda Layer `build_layer.sh`에 `psycopg2-binary` 추가** (Phase 3, 4)
5. **호출자(Step Functions/프론트)가 `session_id` 함께 전달하도록 수정**
6. **CloudWatch 보존 정책 확인** (`aws logs describe-log-groups`)

---

## 4. 주요 파일 위치

```
s3_clone/
├── ERROR_HANDLING_REPORT.md                          ← 메인 보고서
├── CONVERSATION_HISTORY.md                           ← 이 파일
├── database/
│   ├── 4-layer-db-team-guide.md                      ← 박성수 작성, DB 설계 가이드
│   ├── 4-layer-schema-ddl-v1.sql                     ← 기본 스키마
│   └── system-log-schema-ddl.sql                     ← ⭐ phase_execution_log DDL
├── scripts/
│   └── 4-layer-schema-ddl-v1.1.sql                   ← 최신 통합 DDL
├── Phase_1/symptom_llm_4.py                          ← 파일 로그만
├── Phase_2/soo_net_5.py                              ← 파일 로그만
├── Phase_3/infra/aws/phase3/lambda/handler.py        ← 수정 대상 ★
├── Phase_4/infra/aws/phase4/lambda/handler.py        ← 수정 대상 ★
├── Phase_5/infra/aws/phase5/lambda/handler.py        ← 안전망 추가 ★
├── RAG/rag_llm_3.py                                  ← 참조 표준 (DB 연동 패턴)
├── RAG/check_schema.py                               ← 스키마 확인 스크립트
└── RAG/check_sessions.py                             ← diagnosis_session 조회 스크립트
```

---

## 5. DB 연결 정보 (코드에서 추출)

```python
DB_HOST       = "patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com"
DB_NAME       = "soopul"
DB_USER       = "app_user"
DB_SECRET_ID  = "soopul/aurora/app-user"   # Secrets Manager
SCHEMA        = "soopulai"
REGION        = "ap-northeast-2"
```

---

## 6. 향후 세션 시작 시 빠른 컨텍스트 복원 명령

새로운 Claude 세션에서 이 작업을 이어가려면:

```
"s3_clone/CONVERSATION_HISTORY.md 와 s3_clone/ERROR_HANDLING_REPORT.md를 읽고
이전 세션에서 진행한 phase_execution_log 통합 작업을 이어서 도와줘.
Phase 3 핸들러부터 실제 패치 적용 부탁해."
```

또는 메모리 시스템에 저장된 프로젝트 메모리를 참조하면 자동으로 맥락이 복원됩니다.

---

## 7. 참고: 사용자의 작업 선호

- 변경 전 항상 옵션 비교를 통해 의사결정 — 단답형 답변보다 명확한 선택지 제시 선호
- "친절하고 자세한 보고서 형식" 요청 → 단순 요약보다 시나리오 기반 설명 선호
- 한국어 응답
- Windows 환경 (PowerShell 5.1 / Bash 둘 다 사용)

---

## 8. 추가 세션 (2026-05-18 오후 ~ 저녁)

### 8.1 SAM 배포 + Lambda 신규 생성
- Phase 3, 4 신규 Lambda (CloudFormation `phase3-scorer-dev`, `phase4-verifier-dev`) — 우리가 처음 만듦
- Phase 5 RAG (`phase5-rag-dev`) UPDATE
- 9개 IAM 정책 임시 부착 (fhir-ec2-role):
  - AWSLambda_FullAccess, AmazonAPIGatewayAdministrator, AWSCloudFormationFullAccess,
    IAMFullAccess, AmazonS3FullAccess, CloudWatchFullAccess, CloudWatchLogsFullAccess,
    AmazonSSMReadOnlyAccess, AWSStepFunctionsFullAccess
- 신규 IAM Role 만듦: `api-gateway-cloudwatch-logs-role` (API Gateway → CloudWatch 푸시)

### 8.2 박성수님 정체 확인
- 사용자 = 박성수님 본인 (Frontend + Backend 모두 직접 수행)
- CLAUDE.md 의 역할 분담 (박성수=Frontend Lead) outdated

### 8.3 Phase 5 LR Lambda 신규 디자인 + 구현 (8개 파일)
- 권미라님 v4 DDL (`phase5_rare_disease_listing`) 발견 (2026-05-14 작성)
- LIRICAL LR 스코어링 (Robinson 2020) 기반
- 신규 Lambda 분리: `phase5-lr-dev` (RAG `phase5-rag-dev` 와 별도)
- 모듈: handler.py + lr_engine.py + db_reader.py
- DDL v4 Aurora 적용 (master 권한, top_lr_score 등 5컬럼 + 인덱스 2개)
- VARCHAR(16) → TEXT ALTER (StringDataRightTruncation 해결)

### 8.4 VPC 설정 적용 (4개 template.yaml)
- VPC: `vpc-06dd0ad1f2335ea74`
- Subnets: `subnet-02eed659772bac6aa`, `subnet-09ba10cf625d73da0` (multi-AZ)
- SecurityGroup: `sg-03b9bc5d95699b797` (fhir-ec2-sg, RDS 5432 ingress 허용된 유일 SG)
- `AWSLambdaVPCAccessExecutionRole` 추가

### 8.5 lung_dx data path fix
- `paths.py`: `PROJECT_ROOT = parents[2]` → Lambda 에서 `/opt/python` → `/opt/python/data`
- Phase 4 build_layer.sh: data layer 별도 안 만들고 `deps-build/python/data/` 에 yaml 복사
- Phase 3 도 동일 패턴 추가

### 8.6 Bedrock 모델 ID
- `rag_llm_3.py:1331` model_id: `anthropic.claude-3-5-sonnet-20241022-v2:0` → `us.anthropic.claude-3-5-sonnet-20241022-v2:0` (inference profile)
- Phase 4 의 BEDROCK_MODEL_ID 도 동일 이슈 가능 (Phase 4 ASL mode=mock 으로 우회)

### 8.7 Step Functions State Machine 배포
- 위치: `s3_clone/infra/aws/stepfunctions/`
  - `state_machine.asl.json`
  - `template.yaml`
  - `deploy.sh`
- ARN: `arn:aws:states:ap-northeast-2:666803869796:stateMachine:rare-link-pipeline-dev`
- Stack: `rare-link-pipeline-dev` (CREATE_COMPLETE → UPDATE_COMPLETE)
- Role: `rare-link-pipeline-dev-RareLinkPipelineRole-SIzfE8QLzuNv`
- Logs: `/aws/vendedlogs/states/rare-link-pipeline-dev`

### 8.8 ASL 변경 이력
**v1 (초기):** `Parallel(Phase1, Phase2)` → `Parallel(Phase3→Phase4, Phase5)` → `RAG`
- 결과: Phase 3 + Phase 5 동시 cold start → Phase 5 LR timeout (5분)
- Phase 4 mode=real → Bedrock hang

**v2 (개선):** `Parallel(Phase1, Phase2)` → `Phase3` → `Parallel(Phase4, Phase5)` → `RAG`
- Phase 4: `mode: "mock"` (Bedrock 우회)
- Phase 5: Phase 3 끝난 후 시작 (동시성 회피)
- 재배포 + 재테스트 진행 중 (이 메시지 시점)

### 8.9 SFN 첫 execution 결과 (v1)
- 총 5분 30초, status: SUCCEEDED (Catch+Pass 우회)
- Phase 2: ✅ 0.9초
- Phase 3: ✅ 16초 (cold start)
- Phase 5 LR: ❌ 5분 timeout (단독 invoke 70초 SUCCESS — SFN 동시성 문제)
- Phase 4: ❌ 5분 timeout (mode=real, Bedrock hang)
- RAG: ✅ 12초

### 8.10 작업한 추가 파일들 (s3_clone/)
- `Phase_5/infra/aws/phase5-lr/` (8 파일 신규)
- `infra/aws/stepfunctions/` (3 파일 신규)
- `alter_phase5.sql`, `alter_varchar.sql` (DB 마이그레이션)
- `RAG/check_phase_log.py`, `RAG/apply_phase_log_ddl.py` (검증 스크립트)
- `BUILD_SUMMARY_2026-05-18.md` (작업 보고서)

### 8.11 다음 작업 (Backlog)
1. SFN v2 재테스트 결과 확인 (진행 중)
2. Phase 4 Bedrock 모델 ID inference profile 변경 (mode=real 복구 시)
3. FastAPI 구현 (start_execution + describe_execution)
4. Phase 1 Lambda 신규 배포 (현재 Pass state)
5. Phase 2 vision Lambda 에 phase_execution_log 로깅 추가
6. 임시 IAM 정책 9개 정리 (보안)

---

## 9. 빠른 컨텍스트 복원 (다음 세션)

```
s3_clone/BUILD_SUMMARY_2026-05-18.md 와
s3_clone/CONVERSATION_HISTORY.md 읽고
Rare-Link AI Step Functions v2 ASL 재배포 후 작업 이어서 진행해줘.
```

또는 메모리 자동 로드. error_handling_status_rare_link_ai.md 가 최신.


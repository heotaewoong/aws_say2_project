# Rare-Link AI Phase 3~5 통합 작업 — 최종 정리

> 작성일: 2026-05-18
> 작업자: 박성수 (= 사용자) + Claude
> 작업 범위: Phase 3, 4, 5 (RAG), Phase 5 (LR) Lambda 통합 + `phase_execution_log` 로깅

---

## 1. 작업 흐름 (시간순)

| 단계 | 작업 | 결과물 |
|---|---|---|
| 1 | S3 → 로컬 클론 (`Phase_1`~`Phase_5`, `RAG`, `lung_dx`, `frontend`, `deploy`, `database`, `scripts`) | `s3_clone/` |
| 2 | 에러 처리 현황 분석 (Phase 1, 2: 파일 / RAG: DB / Phase 3, 4: CloudWatch 만) | `ERROR_HANDLING_REPORT.md` |
| 3 | `phase_execution_log` 테이블 발견 (이미 Aurora 적용, 미사용) | `database/system-log-schema-ddl.sql` |
| 4 | 8개 파일 패치 (Phase 3, 4, 5 handler + build_layer + template) | S3 업로드 완료 |
| 5 | 7개 폴더 백업 | `s3://say2-2team-bucket/backup/2026-05-18-pre-phase-log/` |
| 6 | EC2 (`i-0f3f223fd40217b12`) 에서 SAM 설치 + 배포 (Phase 3, 4 신규 / Phase 5 update) | CloudFormation 3개 스택 |
| 7 | invoke 테스트 → 500 에러 발견 → VPC 미설정 + Lambda 파일 경로 문제 | 진단 완료 |
| 8 | 박성수님(=사용자) 본인 확인 → 권미라님 새 Phase 5 LR 디자인 공유 받음 | 디자인 변경 |
| 9 | Phase 5 LR Lambda 신규 디자인 + 구현 (handler + lr_engine + db_reader + IaC) | `phase5-lr/` 폴더 |
| 10 | VPC 설정 (subnet + fhir-ec2-sg) 4개 template.yaml 에 추가 | template 갱신 |
| 11 | EC2 sync + 4개 SAM 배포 (Phase 3, 4, 5-RAG, 5-LR) | **현재 진행 중** |

---

## 2. 현재 인프라 상태

### 2.1 Lambda 함수 (4개)

| Lambda | 역할 | API URL | 상태 |
|---|---|---|---|
| `phase3-scorer-dev` | Multimodal Weighted Scoring | `o2d4h10npl.execute-api.../dev/score` | VPC 추가 재배포 진행 중 |
| `phase4-verifier-dev` | LLM Verification (Bedrock) | `szdem2hni4.execute-api.../dev/verify` | VPC 추가 재배포 진행 중 |
| `phase5-rag-dev` | Hybrid Dual RAG → 진단 보고서 | `1nvbtwc3wd.execute-api.../dev/run` | VPC 추가 재배포 진행 중 |
| `phase5-lr-dev` (신규) | LIRICAL LR scoring → 희귀질환 리스팅 | 신규 발급 예정 | 첫 배포 진행 중 |

### 2.2 DB 테이블 (Aurora `soopulai`)

| 테이블 | 역할 | INSERT 권한 |
|---|---|---|
| `phase_execution_log` | 모든 Phase 의 성공/실패/스택트레이스 통합 로그 | ✅ `app_user` |
| `diagnosis_session` | 세션 상태 (`status`, `error_message`, `current_phase`) | ✅ `app_user` (UPDATE) |
| `phase5_rare_disease_listing` | Phase 5 LR 결과 (희귀질환 리스팅) | (확인 필요) |
| `final_report` | RAG 보고서 | (기존 사용 중) |

### 2.3 VPC 구성 (이번 작업으로 적용됨)

| 항목 | 값 |
|---|---|
| VPC | `vpc-06dd0ad1f2335ea74` |
| Lambda Subnets | `subnet-02eed659772bac6aa` (AZ 2a), `subnet-09ba10cf625d73da0` (AZ 2c) |
| Security Group | `sg-03b9bc5d95699b797` (fhir-ec2-sg — RDS 5432 ingress 허용된 유일 SG) |
| NAT Gateway | `nat-0197c0716d3c8493d` (Secrets Manager 호출용) |

### 2.4 IAM (fhir-ec2-role 에 임시 부착된 정책)

```
AWSLambda_FullAccess, AmazonAPIGatewayAdministrator,
AWSCloudFormationFullAccess, IAMFullAccess, AmazonS3FullAccess,
CloudWatchFullAccess, CloudWatchLogsFullAccess
```

**보안 정리 권장:** 배포 작업 종료 후 위 7개 detach. 향후 deploy 시 다시 부착.

신규 IAM Role:
- `api-gateway-cloudwatch-logs-role` (API Gateway → CloudWatch 로그 푸시용, account-wide 1회 설정)

---

## 3. 변경/신규 파일 (S3 업로드 완료)

### 3.1 Phase 3 (3 파일)
- `Phase_3/infra/aws/phase3/lambda/handler.py` — phase_execution_log 로깅
- `Phase_3/infra/aws/phase3/layer/build_layer.sh` — `psycopg2-binary` 추가
- `Phase_3/infra/aws/phase3/template.yaml` — Secrets Manager 정책 + VpcConfig

### 3.2 Phase 4 (3 파일)
- 동일 패턴 (handler / build_layer / template)

### 3.3 Phase 5 RAG (2 파일)
- `Phase_5/infra/aws/phase5/lambda/handler.py` — phase_execution_log 로깅 + 안전망
- `Phase_5/infra/aws/phase5/template.yaml` — VpcConfig

### 3.4 Phase 5 LR 신규 (8 파일)
- `Phase_5/infra/aws/phase5-lr/lambda/handler.py` — Lambda 진입점
- `Phase_5/infra/aws/phase5-lr/lambda/lr_engine.py` — LIRICAL LR 계산
- `Phase_5/infra/aws/phase5-lr/lambda/db_reader.py` — phase1/2 + lab raw 읽기
- `Phase_5/infra/aws/phase5-lr/lambda/requirements.txt`
- `Phase_5/infra/aws/phase5-lr/layer/build_layer.sh`
- `Phase_5/infra/aws/phase5-lr/template.yaml`
- `Phase_5/infra/aws/phase5-lr/deploy.sh`
- `Phase_5/infra/aws/phase5-lr/events/sample_event.json`

### 3.5 검증/적용 스크립트 (RAG/)
- `RAG/check_phase_log.py` — 테이블 + 컬럼 검증
- `RAG/apply_phase_log_ddl.py` — DDL 적용 (DBA 권한 필요)

### 3.6 문서 (s3_clone/ 로컬)
- `ERROR_HANDLING_REPORT.md` — 처음 분석 보고서
- `CONVERSATION_HISTORY.md` — 작업 흐름
- `BUILD_SUMMARY_2026-05-18.md` — 이 파일

---

## 4. 호출 페이로드 정리 (FastAPI/Step Functions 작성용)

### POST API Gateway (HTTP 호출)
| Phase | URL | Body |
|---|---|---|
| 3 Score | `https://o2d4h10npl.execute-api.../dev/score` | `{session_id, patient_id, patient_lab_findings, ...}` |
| 4 Verify | `https://szdem2hni4.execute-api.../dev/verify` | `{session_id, patient_id, mode, phase3_ranking, matched_hp_ids}` |
| 5 RAG Run | `https://1nvbtwc3wd.execute-api.../dev/run` | `{session_id}` |
| 5 LR Run | (배포 후 발급) | `{session_id}` |

### Lambda direct invoke (Step Functions)
```json
{
  "Type": "Task",
  "Resource": "arn:aws:states:::lambda:invoke",
  "Parameters": {
    "FunctionName": "phase3-scorer-dev",
    "Payload": {"session_id.$": "$.session_id", "patient_id.$": "$.patient_id", ...}
  }
}
```

Lambda ARN:
- `arn:aws:lambda:ap-northeast-2:666803869796:function:phase3-scorer-dev`
- `arn:aws:lambda:ap-northeast-2:666803869796:function:phase4-verifier-dev`
- `arn:aws:lambda:ap-northeast-2:666803869796:function:phase5-rag-dev`
- `arn:aws:lambda:ap-northeast-2:666803869796:function:phase5-lr-dev` (예정)

---

## 5. 핵심 디자인 결정 (요약)

| 결정 | 근거 |
|---|---|
| Phase 3, 4, 5 모두 `phase_execution_log` 통합 로깅 | 단일 테이블 = 단일 운영 SQL |
| Phase 5 LR 은 **별도 Lambda** (`phase5-lr-dev`) | RAG (보고서) ↔ LR (스코어링) 역할 다름 |
| Phase 4 비연결 확정 (권미라 v4 DDL) | `input_phase4_top_orphas` DEPRECATED |
| Lab → HPO 변환 = best-effort | 매핑 없으면 skip (`LAB_HPO_MAP` 현재 empty) |
| modality 매핑 단순 | phase1=symptoms, phase2=radiology, lab=lab |
| LR threshold = 5.0 | `listed_diseases` 포함 기준 (LR_pipeline_v2.docx) |
| VPC subnet 2개 (multi-AZ) + fhir-ec2-sg | RDS 사설 IP 접근 + ENI 부족 시 HA |

---

## 5.5 후속 작업 — Step Functions + 추가 통합

### Step Functions 배포 완료
- ARN: `arn:aws:states:ap-northeast-2:666803869796:stateMachine:rare-link-pipeline-dev`
- 파일: `s3_clone/infra/aws/stepfunctions/`
- ASL v2 (재배포):
  ```
  Parallel(Phase1, Phase2)  →  Phase3  →  Parallel(Phase4, Phase5)  →  RAG
  ```

### SFN execution v1 결과 (총 5분 30초, status=SUCCEEDED)
- ✅ Phase 2 (0.9s), Phase 3 (16s), RAG (12s)
- ❌ Phase 5 LR (5분 timeout) — 단독 invoke 70초 SUCCESS → SFN 동시성 문제
- ❌ Phase 4 (5분 timeout, mode=real) — Bedrock hang

### v2 fix (재배포)
- Phase 3 sequential (Phase 5 와 동시 cold start 회피)
- Phase 4 mode → `"mock"` (Bedrock hang 우회)
- v2 재테스트 진행 중

### DB v4 적용
- `phase5_rare_disease_listing` ALTER (5컬럼 + 인덱스 2개 추가)
- `rare_db_ver`/`rare_db_source`: VARCHAR(16/64) → TEXT
- master user (`postgres`/`soopul/aurora/master`)로 직접 적용

### 임시 부착 IAM 정책 (fhir-ec2-role, 9개 — 정리 필요)
```
AWSLambda_FullAccess, AmazonAPIGatewayAdministrator,
AWSCloudFormationFullAccess, IAMFullAccess, AmazonS3FullAccess,
CloudWatchFullAccess, CloudWatchLogsFullAccess,
AmazonSSMReadOnlyAccess, AWSStepFunctionsFullAccess
```

---

## 6. 추후 작업 (Backlog)

- [ ] `LAB_HPO_MAP` 채우기 (`Phase_5/infra/aws/phase5-lr/lambda/db_reader.py:LAB_HPO_MAP`)
- [ ] `recent_errors`, `phase_success_rates_24h` 뷰 생성 (DBA 권한 필요)
- [ ] Phase 1, 2 (스크립트) 도 `phase_execution_log` 로깅 추가
- [ ] 권미라님 새 LR handler.py 가 별도로 올라오면 비교/머지
- [ ] `phenotype.hpoa` 33MB 의 Layer 포함 여부 검토
- [ ] Secrets Manager VPC Endpoint 추가 (NAT 우회, latency 개선)
- [ ] 임시 부착된 IAM 정책 7개 detach (보안)

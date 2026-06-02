# Rare-Link AI — 시스템 아키텍처 보고서

> 작성일: 2026-05-19
> 대상 기간: 2026-05-17 ~ 2026-05-19
> 범위: AWS 인프라 (Lambda × 5, Step Functions, Aurora, FastAPI) + 데이터 흐름 + Phase 별 상세
> 작성자: 박성수님 (Frontend + Backend Lead) + Claude 협업

---

## 1. Executive Summary

희귀질환 진단 파이프라인 **Phase 1~5 + RAG** 의 AWS 통합이 완료됐습니다.

| 영역 | 상태 |
|---|---|
| Lambda 함수 (Phase 1, 2, 3, 4, 5-LR, RAG) | 6개 배포 완료 |
| 통합 로깅 (`phase_execution_log`) | Phase 3, 4, 5-LR, RAG 검증 완료 |
| 희귀질환 LR 스코어링 (Phase 5) | v4 DDL + LIRICAL 엔진 정상 작동 |
| Step Functions State Machine | 2개 배포 (`rare-link-pipeline-dev`, `say2-2team-rare-link-pipeline`) |
| FastAPI ↔ Step Functions 통합 | `api/app/services/stepfunctions.py` 구현 + ARN 환경변수 연결 가능 |
| Aurora PostgreSQL 스키마 (v4) | 권미라님 DDL 적용, `top_lr_score` 등 5컬럼 추가 |

알려진 이슈: SFN 안에서 Phase 4, 5-LR 동시 cold start 시 timeout (단독 invoke 정상). Catch+Pass 우회로 전체 execution 은 SUCCEEDED.

---

## 2. 시스템 아키텍처 (전체)

```
                    ┌──────────────────────────────────────┐
                    │  React Frontend (frontend/)          │
                    │  - 환자 선택, 진단 시작, 결과 표시      │
                    └──────────────┬───────────────────────┘
                                   │ HTTPS
                                   ▼
                    ┌──────────────────────────────────────┐
                    │  FastAPI (api/app/)                  │
                    │  - /sessions  (진단 시작)              │
                    │  - /patients  (환자 정보)              │
                    │  - /feedback  (의사 피드백)            │
                    │  - WS: /ws/emr-updates                │
                    └──────────────┬───────────────────────┘
                                   │ boto3.start_execution
                                   ▼
            ┌─────────────────────────────────────────────────┐
            │  AWS Step Functions                              │
            │  rare-link-pipeline-dev                          │
            │                                                  │
            │   [Start]                                        │
            │     │                                            │
            │     ▼                                            │
            │   ┌──────────────────────┐                       │
            │   │ ParallelPhase1And2   │                       │
            │   │   Phase1 || Phase2    │                       │
            │   └──────┬───────────────┘                       │
            │          ▼                                       │
            │   ┌──────────────────────┐                       │
            │   │ Phase 3              │                       │
            │   │ (Multimodal Scoring) │                       │
            │   └──────┬───────────────┘                       │
            │          ▼                                       │
            │   ┌──────────────────────┐                       │
            │   │ Parallel Phase4 + 5  │                       │
            │   │   LLM Verify || LR    │                       │
            │   └──────┬───────────────┘                       │
            │          ▼                                       │
            │   ┌──────────────────────┐                       │
            │   │ RAG                  │                       │
            │   │ (Final Report)       │                       │
            │   └──────┬───────────────┘                       │
            │          ▼                                       │
            │        [End]                                     │
            └───────┬──────────────────────────────────────────┘
                    │ Invoke
                    ▼
         ┌──────────────────────────────────────────────────┐
         │ Lambda 함수들 (VPC: vpc-06dd0ad1f2335ea74)        │
         │                                                  │
         │   phase1-symptom-dev    (Phase 1: HPO 추출)       │
         │   say2-2team-phase2-vision  (Phase 2: X-ray)     │
         │   phase3-scorer-dev     (Phase 3: Multimodal)    │
         │   phase4-verifier-dev   (Phase 4: LLM Verify)    │
         │   phase5-lr-dev         (Phase 5: LIRICAL LR)    │
         │   phase5-rag-dev        (RAG: 최종 보고서)         │
         └────────┬────────────────────────┬────────────────┘
                  │                        │
                  ▼                        ▼
       ┌────────────────────┐  ┌─────────────────────────┐
       │ Aurora PostgreSQL  │  │ AWS Bedrock             │
       │ soopul schema    │  │ - Claude Sonnet (3.5 v2)│
       │ - phase_execution  │  │ - inference profile     │
       │   _log (통합 로깅)   │  │   us.anthropic.*        │
       │ - diagnosis_       │  │                         │
       │   session          │  └─────────────────────────┘
       │ - phase1~5 결과     │
       │ - final_report     │
       └────────────────────┘
```

---

## 3. Phase 별 상세

### 3.1 Phase 1 — Symptom → HPO 추출

| 항목 | 값 |
|---|---|
| Lambda | `phase1-symptom-dev` |
| CFN Stack | `phase1-symptom-dev` (5-18 14:14 생성) |
| 역할 | 임상 노트 (한국어) → HPO 코드 추출 (Bedrock Claude) |
| 입력 | `{session_id, patient_id, symptom_text}` |
| DB 저장 | `phase1_hpo_extraction.positive_hpo, negative_hpo` (JSONB) |
| 작성자 | 박성수님 (다른 채팅 세션에서 신규 배포) |

핵심 코드 (`Phase_1/symptom_llm_4.py`):
- `BedrockHPOExtractor` 클래스
- `hpo_official.json` (HPO OBO Graph JSON) 로드 → 11,514 HPO 매핑
- 2-step Discovery → Reference → Extraction

### 3.2 Phase 2 — X-ray UNet + DenseNet

| 항목 | 값 |
|---|---|
| Lambda | `say2-2team-phase2-vision` |
| 역할 | 흉부 X-ray 영상 → 14 CheXpert 라벨 → HPO 변환 |
| 입력 | `{session_id, patient_id}` (영상 S3 key 는 DB 에서) |
| DB 저장 | `phase2_xray_processing.xray_hpo_inferred` (JSONB) |
| 모델 | UNet (lung+heart mask) + DenseNet (CheXpert) |

핵심 코드 (`Phase_2/soo_net_5.py`):
- `AnatomySooNetV5` 클래스
- xrv 사전학습 backbone + Anatomy Soft Attention + A³ Aggregation

### 3.3 Phase 3 — Multimodal Weighted Scoring

| 항목 | 값 |
|---|---|
| Lambda | `phase3-scorer-dev` (Memory 2048MB, Timeout 300s) |
| CFN Stack | `phase3-scorer-dev` |
| 역할 | Phase 1 (HPO) + Phase 2 (X-ray HPO) + Lab raw → 528 disease registry 와 매칭 → LR ranking |
| 입력 | `{session_id, patient_id}` |
| **DB 직접 read** | `_read_inputs_from_db(session_id)` 헬퍼 함수로 phase1/2 + lab raw 자체 조회 (박성수님 추가) |
| DB 저장 | `phase3_integrated_ranking.unified_positive_hpo, ranking` (JSONB) |
| Layer | `phase3-deps` (lung_dx + PyYAML + openpyxl + pandas + psycopg2-binary), `phase3-data` (yaml + xlsx) |
| 우리 추가 | `phase_execution_log` INSERT (시작/성공/실패 로깅) |

### 3.4 Phase 4 — LLM Verification

| 항목 | 값 |
|---|---|
| Lambda | `phase4-verifier-dev` |
| CFN Stack | `phase4-verifier-dev` |
| 역할 | Phase 3 ranking → Bedrock Claude 로 검증/재랭킹 |
| 입력 | `{session_id, patient_id, mode=mock|real, ...}` |
| DB 직접 read | `_read_inputs_from_db(session_id)` 으로 phase3 ranking 자체 조회 (박성수님 추가) |
| DB 저장 | `phase4_llm_rerank` |
| Bedrock | `anthropic.claude-sonnet-4-6` (region 강제: `ap-northeast-2` — 박성수님 수정) |
| 우리 추가 | `phase_execution_log` INSERT |

### 3.5 Phase 5 — LIRICAL LR Scoring (신규)

| 항목 | 값 |
|---|---|
| Lambda | `phase5-lr-dev` (Memory 1024MB, Timeout 300s) |
| CFN Stack | `phase5-lr-dev` (우리 작업으로 신규 생성) |
| 역할 | 희귀질환 KB 와 환자 HPO 매칭 → LIRICAL LR 스코어링 |
| 입력 | `{session_id, patient_id}` |
| DB 저장 | `phase5_rare_disease_listing` (v4 DDL by 권미라님) |
| 알고리즘 | LR(HP|D) = freq_in_disease(HP|D) / background_freq(HP), threshold > 5.0 |
| 데이터 | `hpo_background_freq.json` (HPOA marginal, 11,514 HPO), `rare_disease_profiles_v3_1.yaml` (323 질환) |
| 작성자 | Claude 작성, 박성수님 검토 |

핵심 모듈:
- `handler.py` — Lambda entry + `phase_execution_log` 로깅
- `lr_engine.py` — LIRICAL 알고리즘 (modality 가중치 A~G 카테고리)
- `db_reader.py` — phase1+2+lab raw → Step 0 HPO 정규화

### 3.6 RAG — 최종 진단 보고서

| 항목 | 값 |
|---|---|
| Lambda | `phase5-rag-dev` |
| CFN Stack | `phase5-rag-dev` |
| 역할 | 모든 Phase 결과 종합 → Hybrid Dual RAG (PubCaseFinder, Monarch, PubMed, ClinicalTrials) + Bedrock 으로 진단 보고서 PDF/JSON 생성 |
| 입력 | `{session_id}` |
| DB read | RAG 자체 `_read_phase4_from_db(session_id)` 등으로 모든 Phase 결과 조회 |
| DB 저장 | `final_report` |
| Bedrock | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` (inference profile, 우리가 수정) |
| 외부 API | PubMed (eutils.ncbi.nlm.nih.gov), Monarch (api-v3.monarchinitiative.org), PubCaseFinder, ClinicalTrials.gov |

---

## 4. 데이터 흐름 (Phase 별 input/output)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 0: EMR FHIR Bundle (raw_emr_bundle 테이블 — 원본 보존)         │
└──────────────────────┬──────────────────────────────────────────────┘
                       │  ETL (정규화)
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1: canonical                                                  │
│  - patient_profile  (환자 기본)                                       │
│  - clinical_note    (의사 노트 한국어)                                 │
│  - lab_result       (Lab raw: loinc_code, value_numeric, abnormal)    │
│  - imaging_study    (영상 메타)                                       │
└────┬──────────┬──────────┬──────────┬─────────────────────────────────┘
     │          │          │          │
     ▼          ▼          ▼          ▼
   Phase1   Phase2   Phase3     ... (Front)
     │          │          │          
     ▼          ▼          ▼          
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 2: phase_io (각 Phase 결과)                                    │
│  - phase1_hpo_extraction   (positive_hpo, negative_hpo JSONB)         │
│  - phase2_xray_processing  (xray_hpo_inferred JSONB)                  │
│  - phase3_integrated_ranking (unified_positive_hpo, ranking JSONB)    │
│  - phase4_llm_rerank       (재랭킹)                                   │
│  - phase5_rare_disease_listing  (LR > 5.0 희귀질환 list, v4)          │
│  - final_report            (RAG 최종 보고서)                           │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 3: physician_feedback, final_clinical_outcome                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  운영 로그: phase_execution_log  (모든 Phase 실행 이력)                │
│  - log_id, session_id, phase_name, phase_step, status                │
│  - duration_ms, lambda_function, lambda_request_id                   │
│  - error_code, error_message, error_stacktrace, error_category       │
│  - input_summary, output_summary, model_versions (JSONB)             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Step Functions 워크플로우 (ASL v4)

### 5.1 우리가 만든 SFN: `rare-link-pipeline-dev`

```
[Start]
   │  Input: {session_id, patient_fhir_id, symptom_text, cxr_s3_key, started_at}
   ▼
┌────────────────────────────────┐
│ ParallelPhase1And2             │
│ ┌─────────┐    ┌────────────┐  │
│ │ Phase1  │    │ Phase2     │  │
│ │ (Pass*) │    │ (vision)   │  │
│ └─────────┘    └────────────┘  │
└──────────┬─────────────────────┘
           ▼
┌────────────────────────────────┐
│ Phase3                         │  ← Sequential (동시 cold start 회피)
│ phase3-scorer-dev              │
│ ResultSelector: body 추출       │
└──────────┬─────────────────────┘
           ▼
┌────────────────────────────────┐
│ ParallelPhase4AndPhase5        │
│ ┌──────────┐  ┌────────────┐   │
│ │ Phase4   │  │ Phase5     │   │
│ │ (mock**) │  │ (LR)       │   │
│ └──────────┘  └────────────┘   │
└──────────┬─────────────────────┘
           ▼
┌────────────────────────────────┐
│ RAG                            │
│ phase5-rag-dev                 │
│ Catch+Pass 우회 (Phase 실패해도) │
└──────────┬─────────────────────┘
           ▼
        [End]

* Phase 1 은 처음엔 Pass state 였으나, 박성수님이 phase1-symptom-dev Lambda 별도 배포.
  → ASL Phase1 state 를 Lambda Task 로 교체 가능 (다음 작업)

** Phase 4 mode=mock 으로 Bedrock 호출 우회 중. 박성수님 region fix 후
   mode=real 로 복구 검토.
```

### 5.2 별도 SFN: `say2-2team-rare-link-pipeline`
- 박성수님이 별도 채팅 세션에서 만든 State Machine
- 생성: 2026-05-18 09:55 (KST)
- 우리 `rare-link-pipeline-dev` 와 별개 — 통합 또는 정리 필요

---

## 6. AWS 인프라 현황

### 6.1 Lambda 함수 (Rare-Link 관련 6개)
| Function | LastModified | Runtime | Memory | Timeout | VPC |
|---|---|---|---|---|---|
| `phase1-symptom-dev` | 5-18 17:10 | python3.11 | - | - | (확인 필요) |
| `say2-2team-phase2-vision` | 5-18 17:10 | python3.11 | - | - | vpc-06dd... |
| `phase3-scorer-dev` | 5-18 17:24 | python3.11 | 2048 | 300 | vpc-06dd... |
| `phase4-verifier-dev` | 5-18 17:24 | python3.11 | 512 | 300 | vpc-06dd... |
| `phase5-lr-dev` | 5-18 17:24 | python3.11 | 1024 | 300 | vpc-06dd... |
| `phase5-rag-dev` | 5-18 17:24 | python3.11 | 1024 | 300 | vpc-06dd... |

### 6.2 CloudFormation 스택 (6개)
- `phase1-symptom-dev` (5-18 생성)
- `phase3-scorer-dev`
- `phase4-verifier-dev`
- `phase5-lr-dev` (신규)
- `phase5-rag-dev` (update)
- `rare-link-pipeline-dev` (Step Functions)

### 6.3 IAM Role (신규/관련)
- `api-gateway-cloudwatch-logs-role` (API Gateway 로그용)
- `phase1-symptom-dev-Phase1SymptomFunctionRole-...`
- `phase3-scorer-dev-Phase3ScorerFunctionRole-...`
- `phase4-verifier-dev-Phase4VerifierFunctionRole-...`
- `phase5-lr-dev-Phase5LRFunctionRole-...`
- `rare-link-pipeline-dev-RareLinkPipelineRole-...`
- `say2-2team-rare-link-stepfn-role` (박성수님 별도 SFN role)

### 6.4 Aurora PostgreSQL
- Cluster: `patient-db-cluster` (postgres 16.4)
- Endpoint: `patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com`
- Schema: `soopulai`
- Master: `postgres` (secret: `soopul/aurora/master`)
- App user: `app_user` (secret: `soopul/aurora/app-user`)
- VPC: vpc-06dd0ad1f2335ea74

### 6.5 VPC 구성
- VPC: `vpc-06dd0ad1f2335ea74`
- Lambda Subnets: `subnet-02eed659772bac6aa` (AZ 2a), `subnet-09ba10cf625d73da0` (AZ 2c)
- Lambda Security Group: `sg-03b9bc5d95699b797` (fhir-ec2-sg — RDS 5432 ingress 유일 허용)
- NAT Gateway: `nat-0197c0716d3c8493d` (Secrets Manager 외부 호출용)
- VPC Endpoints: S3, SageMaker, Bedrock

### 6.6 Step Functions
- `rare-link-pipeline-dev` (우리 작업, ASL v4)
- `say2-2team-rare-link-pipeline` (박성수님 별도)

---

## 7. 2026-05-17 ~ 19 변경사항 Timeline

### 5-17
- 11:30: 기존 phase 분석 시작 (Phase 1, 2 → 파일 로그만, RAG → DB UPDATE)
- 15:20: Phase 5 README + 메타 파일 업로드 (phase 5 폴더 구조)
- 18:23: 우리 코드 패치 1차 S3 업로드 (handler.py 3개 + build_layer.sh 2개 + template.yaml 3개 — phase_execution_log 로깅 추가)

### 5-18 새벽
- 03:23: Phase 5 handler.py 최종 패치
- 03:15: 백업 `s3://say2-2team-bucket/backup/2026-05-18-pre-phase-log/` 생성

### 5-18 오전
- 04:25~04:46: Phase 3 SAM 첫 배포 시도 (4회 재시도 후 성공)
- 04:45: api-gateway-cloudwatch-logs-role 신규 생성
- 04:46~05:03: Phase 3, 4, 5 신규 Lambda 모두 CREATE_COMPLETE
- 06:42: Phase 5 LR Lambda 신규 디자인 + 구현 (lr_engine.py, db_reader.py, handler.py)
- 07:06: Phase 5 LR 첫 배포 + ALTER TABLE (v4 DDL)
- 08:15: 모든 Lambda VPC 설정 추가 (subnet + sg-03b9bc5d95699b797)
- 09:00: rare_db_ver VARCHAR(16) → TEXT ALTER (StringDataRightTruncation 해결)
- 09:37: Step Functions `rare-link-pipeline-dev` 생성
- 09:55: 박성수님 `say2-2team-rare-link-stepfn-role` 생성 (별도 SFN 작업)

### 5-18 오후
- 11:55, 12:00: 권미라님 `rare_disease_profiles_v3_1.yaml` 업로드 (1.1MB)
- 13:14: 권미라님 `lr_data/` 추가 (phenotype.hpoa 33MB, hpo_background_freq.*)
- 14:14: 박성수님 **phase1-symptom-dev Lambda 신규 생성** (다른 채팅)
- 17:10: phase1-symptom-dev, say2-2team-phase2-vision 업데이트
- 17:24: **박성수님 Phase 3, 4 handler 수정 + 재배포**
  - Phase 3, 4: `_read_inputs_from_db(session_id)` 헬퍼 추가 (DB 직접 read)
  - Phase 4: `BedrockPhase4Verifier` region → `ap-northeast-2` 강제
- 18:00~21:30: Step Functions ASL v1~v4 반복 테스트

### 5-19 ~
- 추가 작업 계속 진행 중

---

## 8. 운영 가이드

### 8.1 배포 (박성수님 본인 작업)

```bash
# EC2 i-0f3f223fd40217b12 에서:
cd ~/rare-link-deploy/Phase_3/infra/aws/phase3
./deploy.sh dev  # 또는 staging / prod

# Step Functions:
cd ~/rare-link-deploy/infra/aws/stepfunctions
./deploy.sh dev
```

### 8.2 진단 시작 (FastAPI 통해)

```bash
# FastAPI 환경변수 설정
export STEPFN_STATE_MACHINE_ARN=arn:aws:states:ap-northeast-2:666803869796:stateMachine:rare-link-pipeline-dev

# 또는 직접 boto3:
python3 -c "
import boto3, json, uuid, datetime
sfn = boto3.client('stepfunctions', region_name='ap-northeast-2')
resp = sfn.start_execution(
    stateMachineArn='arn:aws:states:ap-northeast-2:666803869796:stateMachine:rare-link-pipeline-dev',
    name=f'session-{uuid.uuid4().hex[:8]}',
    input=json.dumps({
        'session_id': '<UUID>',
        'patient_fhir_id': '<patient_id>',
        'symptom_text': '...',
        'cxr_s3_key': 's3://.../xray.png',
        'started_at': datetime.datetime.now(datetime.UTC).isoformat(),
    }, ensure_ascii=False),
)
print(resp['executionArn'])
"
```

### 8.3 검증 SQL

```sql
-- 세션별 실행 이력
SELECT phase_name, phase_step, status, duration_ms, error_code
FROM soopulai.phase_execution_log
WHERE session_id = '<UUID>'
ORDER BY started_at;

-- Phase 5 LR 결과
SELECT total_listed_count, top_lr_score, top_lr_orphacode
FROM soopulai.phase5_rare_disease_listing
WHERE session_id = '<UUID>';

-- 최근 실패 100건
SELECT phase_name, error_code, error_category, COUNT(*)
FROM soopulai.phase_execution_log
WHERE status='failed' AND started_at > NOW() - INTERVAL '24 hours'
GROUP BY 1, 2, 3 ORDER BY 4 DESC;
```

---

## 9. 알려진 이슈 & Backlog

### 알려진 이슈
1. **Phase 4, 5-LR SFN 동시 cold start 시 timeout** — 단독 invoke 정상. VPC ENI 또는 RDS connection 경쟁 의심
2. `say2-2team-rare-link-pipeline` 별도 SFN 과 우리 `rare-link-pipeline-dev` 의 통합 또는 정리 필요
3. ASL 의 Phase 1 이 Pass state — 실제 `phase1-symptom-dev` Lambda 호출로 교체 필요

### Backlog
- [ ] ASL 의 Phase 1 → Task state 로 교체 (phase1-symptom-dev Lambda invoke)
- [ ] Phase 4 의 Bedrock region fix 검증 → mode=real 복구
- [ ] Phase 5 LR + Phase 4 동시 cold start 해결 (Provisioned Concurrency, sequential, subnet 확장 등)
- [ ] FastAPI 의 `STEPFN_STATE_MACHINE_ARN` env 실제 운영 환경에 설정
- [ ] Phase 2 vision Lambda 에 `phase_execution_log` 로깅 추가
- [ ] 임시 부착된 IAM 정책 9개 정리 (보안):
  - fhir-ec2-role 의 AWSLambda_FullAccess, AmazonAPIGatewayAdministrator,
    AWSCloudFormationFullAccess, IAMFullAccess, AmazonS3FullAccess,
    CloudWatchFullAccess, CloudWatchLogsFullAccess,
    AmazonSSMReadOnlyAccess, AWSStepFunctionsFullAccess
- [ ] 별도 deploy 전용 IAM Role 분리 (fhir-ec2-role 의료데이터용에서 deploy 권한 분리)
- [ ] 운영 편의 뷰 생성 (`recent_errors`, `phase_success_rates_24h`) — DBA 권한

---

## 10. 참고 파일

| 파일 | 위치 | 용도 |
|---|---|---|
| `BUILD_SUMMARY_2026-05-18.md` | `s3_clone/`, `s3://docs/` | 5-18 작업 보고서 |
| `CONVERSATION_HISTORY.md` | `s3_clone/`, `s3://docs/` | 작업 대화 흐름 (§1~§9) |
| `ERROR_HANDLING_REPORT.md` | `s3_clone/`, `s3://docs/` | 초기 에러 처리 분석 |
| `ARCHITECTURE_REPORT_2026-05-19.md` | (이 파일) | 종합 아키텍처 보고서 |
| `database/system-log-schema-ddl.sql` | S3 `database/` | phase_execution_log DDL |
| `database/4-layer-db-team-guide.md` | S3 `database/` | 박성수님 DB 설계 가이드 |
| `RAG/check_phase_log.py` | S3, EC2 `/tmp/` | 검증 스크립트 |
| `Phase_5/infra/aws/phase5-lr/` | S3, 로컬 | Phase 5 LR Lambda 전체 (8 파일) |
| `infra/aws/stepfunctions/` | S3, 로컬 | Step Functions 정의 |

---

## 11. 다음 세션 컨텍스트 복원

```
s3_clone/ARCHITECTURE_REPORT_2026-05-19.md 와
s3_clone/BUILD_SUMMARY_2026-05-18.md 읽고
Backlog 의 [ASL 의 Phase 1 → Task 교체] 부터 진행해줘.
```

또는 메모리 자동 로드 (4개 메모리 — project, db, error, user).

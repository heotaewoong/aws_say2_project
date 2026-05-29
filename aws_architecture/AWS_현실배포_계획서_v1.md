# AWS 현실 배포 계획서 (v1)

> **목적**: `architecture_final.html`의 25개+ 서비스 중 **프리프로젝트 시연에 필수인 것만** 추려서 2~3일 안에 실제 배포 가능한 축소안.
> **전제**: 크레딧 한정, 팀원 분업, 2AZ HA는 데모에서 불필요.

---

## 1. 최종안 vs 현실안 비교표

| 계층 | architecture_final.html (이상) | 현실 배포안 (팀 2~3일) | 생략 사유 |
|------|--------------------------------|------------------------|-----------|
| DNS/CDN | Route 53 + CloudFront | S3 정적 호스팅 또는 `vite dev` | 커스텀 도메인 없음 |
| WAF/보안 | WAF + GuardDuty + CloudTrail | 기본 VPC + SG만 | 데모에 불필요 |
| 인증 | Cognito User Pool | 없음 (공개 데모) | 프리프로젝트 = 팀 내부만 접근 |
| API 게이트웨이 | API Gateway + Cognito Authorizer | API Gateway (AuthorizationType=NONE) | 그대로 |
| 오케스트레이션 | Step Functions Parallel State | **Lambda 단일 호출** 또는 Python `ThreadPoolExecutor` | Lambda 하나로 파이프라인 전체 처리 가능 |
| AZ 구성 | Multi-AZ (a/c) 전부 이중화 | **Single-AZ** 1개 (ap-northeast-2a만) | 데모에서 HA 증명 불필요 |
| NAT Gateway | ×2 HA | **×1** (시간당 $0.045 절약) | Single-AZ면 1개 충분 |
| VPC Endpoint | 6종 (Bedrock/S3/DynamoDB/SageMaker/Logs/STS) | **4종** (S3, DynamoDB, Bedrock, SageMaker Runtime) | Logs/STS는 NAT 경유 허용 |
| X-ray 추론 | SageMaker Endpoint ml.g4dn.xlarge (GPU) | **Lambda 컨테이너 이미지** (CPU, 3GB 메모리) | 시간당 $1.2 → 요청당 $0.00002 |
| LLM 추출 | Bedrock Haiku (Phase 1) | 그대로 | Bedrock은 종량제 |
| LLM 소견서 | Bedrock Sonnet 3.5 (Phase 5) | 그대로 | Bedrock은 종량제 |
| 희귀 스코어링 | Lambda (Phase 5 LIRICAL) | 같은 Lambda 함수 내부 로직 | 별도 Lambda 분리 불필요 |
| 일반 스코어링 | Lambda (Phase 4) | 같은 Lambda 함수 내부 로직 | 통합 |
| 벡터 DB | RDS pgvector (528 질환) | **로컬 CSV** (orphadata_weighted.csv, 4335 질환) | 현재 RAG가 파일 기반으로 잘 돌아감 |
| FHIR | EC2 + ALB + Aurora | **없음** | EMR 연동은 "추후 연동 계층" 도식만 |
| 캐시 | ElastiCache Redis | **DynamoDB** TTL 또는 Lambda `/tmp` | Redis 관리 부담 |
| 결과 저장 | DynamoDB | 그대로 | |
| 원본 저장 | S3 (X-ray/Reports) | 그대로 | |
| 모니터링 | CloudWatch + SNS 알림 | **CloudWatch Logs만** | SNS 설정 시간 소비 |

**결과**: 25개 → **10개** 서비스로 축소

---

## 2. 현실 배포 구성도

```
  사용자 (브라우저)
      │
      ▼
┌──────────────────┐
│  API Gateway     │ (REST, AuthorizationType=NONE)
│  POST /diagnose  │
└────────┬─────────┘
         │
         ▼
    VPC (10.0.0.0/16) — Single-AZ (ap-northeast-2a)
   ┌──────────────────────────────────────────────────┐
   │                                                   │
   │  Public Subnet           NAT Gateway ×1           │
   │       │                                           │
   │       ▼                                           │
   │  Private Subnet                                   │
   │   ┌────────────────────────────────────────┐     │
   │   │  Lambda (rag-pipeline)                  │     │
   │   │  - 컨테이너 이미지 (PyTorch + boto3)     │     │
   │   │  - 3GB 메모리, 5분 Timeout             │     │
   │   │  - rag_pipeline.py 통째로 탑재          │     │
   │   │                                         │     │
   │   │  내부 작업:                             │     │
   │   │    ① SooNet 추론 (CPU)                  │     │
   │   │    ② Bedrock Haiku 호출 (HPO)          │     │
   │   │    ③ Lab Rule-based HPO                │     │
   │   │    ④ LIRICAL + General 스코어링         │     │
   │   │    ⑤ 5개 외부 API 병렬                 │     │
   │   │    ⑥ Bedrock Sonnet 소견서              │     │
   │   └──────────┬─────────────────────────────┘     │
   │              │                                     │
   │   VPC Endpoint (Interface)                        │
   │    ├─ Bedrock Runtime                             │
   │    └─ SageMaker Runtime (안 써도 OK)              │
   │                                                    │
   │   VPC Endpoint (Gateway)                          │
   │    ├─ S3                                           │
   │    └─ DynamoDB                                    │
   │                                                    │
   └───────────────────┬───────────────────────────────┘
                       │ (NAT 경유)
                       ▼
              외부 API (PubMed, ClinicalTrials,
                       PubCaseFinder, Monarch, Orphanet)

   저장소:
     S3 (rare-link-phase2-xxx)
       ├─ uploads/  (X-ray 원본)
       └─ results/  (소견서 JSON)
     DynamoDB (rare-link-results)
       └─ diagnosis_id, created_at, report
```

---

## 3. 팀 분업 계획 (2~3일 기준)

### Day 1 — 인프라 기반 (내 담당)

| 시간 | 작업 | 확인 방법 |
|------|------|-----------|
| 오전 | `01-network.yaml` 배포 (Single-AZ 버전) | `aws ec2 describe-vpcs` |
| 오후 | `02-phase2-lambda.yaml` 배포 (SageMaker → Lambda 변경) | Lambda 콘솔에서 `invoke` 성공 |

### Day 2 — Lambda 컨테이너 이미지 빌드 + 테스트

| 시간 | 작업 | 확인 방법 |
|------|------|-----------|
| 오전 | Docker build → ECR push | `aws ecr describe-images` |
| 오후 | Lambda Test → CloudWatch Logs | JSON 소견서 출력 확인 |

### Day 3 — 통합 + 시연

| 시간 | 작업 |
|------|------|
| 오전 | API Gateway → Lambda 연결, S3 presigned URL 업로드 |
| 오후 | 프론트엔드(`frontend/`)에서 E2E 호출 |

---

## 4. 크레딧 예산

| 시나리오 | 비용 | 기간 |
|---------|------|------|
| 구축 중 (NAT + Lambda 대기) | $1/일 | Day 1~3 |
| 시연 (10번 호출) | $0.10 | 시연 당일 |
| **총 예상** | **$4~5** | 3일 전체 |

> **SageMaker 엔드포인트는 쓰지 않음** → 가장 큰 비용 (시간당 $1.2) 제거.
> Lambda 컨테이너 이미지는 **요청당 과금**이라 유휴 비용 0원.

---

## 5. Phase 2만 SageMaker로 가는 옵션 (선택)

팀원이 "GPU 필요하다"고 주장하면 아래 옵션 추가 가능:

| 방식 | 비용 | 구현 난이도 |
|------|------|------------|
| Lambda CPU 추론 (기본) | 요청당 $0.00002 | 낮음 (컨테이너 이미지만) |
| SageMaker Serverless Inference | 요청당 $0.0004 + cold start | 중간 |
| SageMaker Endpoint ml.m5.large | 시간당 $0.115 (24h=$2.76) | 중간 |
| SageMaker Endpoint ml.g4dn.xlarge (GPU) | 시간당 $1.20 (24h=$28.80) | 중간 |

> **권장**: Lambda CPU 추론 (100장/분 처리 가능, 시연 충분).
> CPU에서 DenseNet-121 한 장 추론은 ~500ms. 수용 가능.

---

## 6. 단순화된 CloudFormation 템플릿 (Day 1~2 배포용)

현재 `infra/cloudformation/` 에 있는 2개 스택 수정 계획:

### `01-network.yaml` 수정 (Single-AZ)

- Public Subnet: A만 (C 제거)
- Private Subnet: 1개만 (Phase 1·2·3 통합)
- NAT Gateway: 1개
- VPC Endpoint: Bedrock, S3, DynamoDB만

→ **리소스 수 50% 감소**, 배포 시간 5분 이내

### `02-phase2-lambda.yaml` (신규)

- SageMaker 전부 제거
- Lambda 컨테이너 이미지 + ECR 리포지토리
- Lambda에 `rag_pipeline.py` 전체 탑재
- API Gateway REST 추가 (`POST /diagnose`)

→ 기존 `02-phase2-xray.yaml`은 **GPU 필요 시에만** 사용

---

## 7. 시연 시나리오

1. 시연자: 프론트엔드에서 X-ray 업로드 버튼 클릭
2. 프론트: S3 presigned PUT URL 요청 → 이미지 업로드
3. 프론트: API Gateway `POST /diagnose` 호출 (S3 key 전달)
4. Lambda: 파이프라인 5단계 실행 (~30초)
5. 프론트: 결과 JSON을 카드 형태로 렌더링 (Top3 질환 + 소견서)
6. (옵션) DynamoDB에서 diagnosis_id로 결과 재조회 시연

---

## 8. 리스크 & 대비책

| 리스크 | 대비책 |
|--------|--------|
| Lambda cold start 10초+ | Provisioned Concurrency 1개 (시연 직전 활성화) |
| PubCaseFinder 502 장애 | 로컬 폴백 이미 구현됨 |
| Bedrock 리전 권한 | ap-northeast-2는 Sonnet 3.5 지원 확인 완료 |
| Docker 이미지 크기 (PyTorch 2GB+) | Lambda 컨테이너는 10GB까지 허용 |
| 크레딧 소진 | 시연 끝나면 `cloudformation delete-stack` 즉시 |

---

## 9. 다음 액션 (내가 바로 할 일)

1. [ ] `01-network.yaml` Single-AZ 버전 (`01-network-simple.yaml`) 생성
2. [ ] `02-phase2-lambda.yaml` 신규 작성 (SageMaker 대체)
3. [ ] `infra/lambda/` Dockerfile 복원 (이전 버전 + 수정)
4. [ ] 배포 스크립트 (`deploy.sh`) 작성
5. [ ] `infra/README.md` 현실안 반영

이 파일(`AWS_현실배포_계획서_v1.md`)이 **공식 기준**이고, 이후 구현은 이 계획을 따름.

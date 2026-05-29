# AWS 인프라 현황 (담당: 허태웅)

> **계정**: 666803869796 | **리전**: ap-northeast-2 (서울)  
> **최종 업데이트**: 2026-05-21  
> 리소스 ID/ARN 상세 → [`resource_ids.md`](./resource_ids.md)  
> 전체 배포 절차 → [`AWS_콘솔_직접배포_가이드.md`](./AWS_콘솔_직접배포_가이드.md)

---

## 📋 전체 구축 현황

| 단계 | 서비스 | 리소스 이름 | 상태 | 완료일 |
|------|--------|------------|------|--------|
| 1 | VPC | `say2-2team` (10.0.0.0/24) | ✅ | - |
| 2 | Subnet × 4 | public, private-1/2/3 | ✅ | - |
| 3 | Security Group × 3 | lambda, sagemaker, vpce | ✅ | - |
| 4 | IGW + NAT + Route Table | `say2-2team-igw/nat/rt` | ✅ | - |
| 5 | S3 폴더 | `say2-2team-bucket/Phase_2/` | ✅ | - |
| 6 | IAM Role × 2 | sagemaker-role, lambda-role | ✅ | 2026-05-08 |
| 7 | 모델 S3 업로드 | `model.tar.gz` (146MB) | ✅ | 2026-05-08 |
| 8 | SageMaker Endpoint | `say2-2team-soonet-endpoint` | ⏸️ 발표 직전 생성 | 2026-05-08 |
| 9 | Lambda 함수 | `say2-2team-phase2-vision` | ✅ | 2026-05-08 |
| 10 | E2E 테스트 | Lambda → SageMaker → HPO | ⬜ | - |
| 11 | Cognito User Pool | `say2-2team-user-pool` | ⬜ 보안 | - |
| 12 | Secrets Manager | `say2-2team/phase2-config` | ⬜ 보안 | - |
| 13 | KMS | `alias/say2-2team-data-key` | ✅ | 2026-05-12 |
| 14 | CloudTrail | `say2-2team-audit-trail` | ✅ | 2026-05-12 |
| 15 | GuardDuty + Security Hub | Detector + Hub | ⚠️ 권한 없음 | - |
| 16 | S3 보안 강화 | 퍼블릭 차단 + 버전관리 + 수명주기 | ✅ | 2026-05-12 |
| 17 | API Gateway + Cognito Authorizer | `say2-2team-diagnose-api` | ⬜ 보안 | - |
| 18 | Aurora DB | `patient-db-cluster` 이미 존재, `rarelinkai.final_report` 사용 | ✅ (팀 공유) | - |
| 19 | Aurora VPC SG 수정 | Lambda SG(`sg-03e64fdde60d52a6c`) → Aurora SG(`sg-019a357627f1594db`) 5432 개방 | ✅ | 2026-05-13 |
| 20 | SNS 알림 | `say2-2team-alerts` | ✅ | 2026-05-12 |
| 21 | AWS Budgets | CloudWatch Billing 알람으로 대체 | ✅ | 2026-05-12 |
| 22 | Detective + Inspector | 심층 보안 | ⬜ 보안 | - |
| 23 | 프론트엔드 S3 업로드 | `say2-2team-bucket/frontend/` | ✅ | 2026-05-12 |
| 24 | CloudFront 배포 | `say2-2team-cf-distribution` (`E2ZHONIV05TX9D`) | ⚠️ IAM 권한 필요 | - |
| 25 | WAF WebACL (Global) | `say2-2team-waf` (us-east-1, Essentials) | ⚠️ IAM 권한 필요 | - |
| 26 | Route 53 DNS | - (커스텀 도메인 없음 — 스킵) | ➖ | - |
| 27 | EC2 IAM 권한 추가 | `fhir-ec2-role` Bedrock + S3 PutObject | ⬜ | - |
| 27.1 | EC2에서 rag_llm_3.py 실행 | SSM 접속 → pip 설치 → 실행 | ⬜ | - |
| 28 | SQS (FHIR 요청 큐) | - | ⬜ | - |
| 29 | ElastiCache (Redis) — HPO 버퍼 | - | ⬜ | - |
| 30 | Step Functions — 진단 오케스트레이터 | - | ⬜ | - |
| 31 | EventBridge — MLOps 트리거 | - | ⬜ | - |
| 32 | ECR — 컨테이너 이미지 저장소 | - | ⬜ | - |
| 33 | AWS Backup — 데이터 백업 정책 | - | ⬜ | - |
| 34 | Aurora (RAG/FHIR 백엔드 DB) | `patient-db-cluster` (팀 공유, rarelinkai 스키마) | ✅ available | - |

---

## 🏗️ 서비스 아키텍처 (현재 구축 기준)

```
┌──────────────────── VPC (say2-2team, 10.0.0.0/24) ──────────────────┐
│                                                                      │
│  ┌─── Public Subnet (10.0.0.0/28) ──────────────────────────────┐   │
│  │  🌍 Internet Gateway (say2-2team-igw)                         │   │
│  │  🌍 NAT Gateway (say2-2team-nat)                              │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─── Private Subnet ① (10.0.0.128/28) ─────────────────────────┐   │
│  │  λ Lambda: say2-2team-phase2-vision (Python 3.11, 1GB, 5min)  │   │
│  │     ↓ VPC Endpoint 경유                                        │   │
│  │  🤖 SageMaker: say2-2team-soonet-endpoint (ml.m5.large)       │   │
│  │     SooNet DenseNet-121 → 14개 질환 확률 + HPO 코드            │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─── Private Subnet ② (10.0.0.16/28) ──────────────────────────┐   │
│  │  (Phase 4·5 예약)                                              │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─── Private Subnet ③ (10.0.0.32/28) ──────────────────────────┐   │
│  │  (RAG & Report 예약)                                           │   │
│  └───────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
         │
         ▼ (S3 Gateway Endpoint)
┌─────────────────────────────────────────────────────────────────────┐
│  S3: say2-2team-bucket                                              │
│  ├── Phase_2/uploads/     ← X-ray 이미지 입력                       │
│  ├── Phase_2/results/     ← 진단 결과 JSON                          │
│  ├── Phase_2/models/      ← model.tar.gz (146MB, SooNet+UNet)       │
│  ├── guardduty-findings/  ← GuardDuty 탐지 결과                     │
│  ├── config-logs/         ← AWS Config 변경 이력                    │
│  └── cloudtrail-logs/     ← API 감사 로그                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔐 보안 레이어 구성

```
요청 흐름:
외부 → [API Gateway + Cognito JWT ⬜] → Lambda → SageMaker → S3

보안 레이어:
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: 네트워크                                                │
│   • Private Subnet 배치 (Lambda, SageMaker 인터넷 노출 없음)     │
│   • Security Group 최소 권한 (Lambda→SageMaker 443만 허용)       │
│   • S3 퍼블릭 접근 차단 (4개 항목 모두 활성화)                   │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: 암호화                                                  │
│   • S3 SSE-KMS (alias/say2-2team-data-key)                      │
│   • CloudTrail 로그 KMS 암호화 (alias/say2-2team-cloudtrail-key) │
│   • Bucket Key 활성화 (KMS 호출 99% 절감)                        │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: 감사 & 탐지                                             │
│   • CloudTrail: 모든 API 호출 + S3 데이터 이벤트 기록            │
│   • GuardDuty: 비정상 API 호출, 자격증명 탈취 시도 탐지          │
│   • Security Hub: CIS + AWS FSBP 표준 준수 점수                  │
│   • AWS Config: 리소스 변경 이력 추적                            │
├─────────────────────────────────────────────────────────────────┤
│ Layer 4: 비밀 관리                                               │
│   • Secrets Manager: SageMaker endpoint, S3 경로 등 설정값      │
│   • IAM 최소 권한: Lambda는 Phase_2/* + 자기 endpoint만 접근     │
├─────────────────────────────────────────────────────────────────┤
│ Layer 5: 모니터링 & 알람                                         │
│   • Lambda 에러 ≥3회/5분 → SNS 알림                             │
│   • SageMaker 지연 ≥30초 → SNS 알림                             │
│   • S3 버킷 삭제 시도 → SNS 알림                                 │
│   • 전체 비용 ≥$80/일 → SNS 알림 (us-east-1)                    │
│   • SageMaker 비용 ≥$10/일 → SNS 알림 (us-east-1)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 파일 구조

```
infra/
├── README.md                      ← 이 파일 (현황 요약)
├── resource_ids.md                ← 리소스 ID/ARN 모음
├── AWS_콘솔_직접배포_가이드.md     ← 단계별 배포 절차 (Step 1~22)
├── cloudformation/
│   └── 01-network.yaml            ← VPC/Subnet/SG CloudFormation 템플릿
└── lambda/
    └── phase2/
        └── phase2_handler.py      ← Lambda 핸들러 (S3→SageMaker→HPO)
```

---

## 🚀 발표 직전 체크리스트

```bash
export AWS_ACCESS_KEY_ID=AKIAZWQE3TBSFI4FMM57
export AWS_DEFAULT_REGION=ap-northeast-2

# 1. SageMaker Endpoint 생성 (5~10분 소요)
aws sagemaker create-model --model-name say2-2team-soonet-model \
  --execution-role-arn arn:aws:iam::666803869796:role/say2-2team-sagemaker-role \
  --primary-container '{
    "Image": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:2.0.0-cpu-py310-ubuntu20.04-sagemaker",
    "ModelDataUrl": "s3://say2-2team-bucket/Phase_2/models/soonet/model.tar.gz",
    "Environment": {"SAGEMAKER_PROGRAM": "inference.py", "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code"}
  }' --vpc-config "Subnets=subnet-02eed659772bac6aa,SecurityGroupIds=sg-03e64fdde60d52a6c"

aws sagemaker create-endpoint-config --endpoint-config-name say2-2team-soonet-config \
  --production-variants '[{"VariantName":"primary","ModelName":"say2-2team-soonet-model","InitialInstanceCount":1,"InstanceType":"ml.m5.large"}]'

aws sagemaker create-endpoint --endpoint-name say2-2team-soonet-endpoint \
  --endpoint-config-name say2-2team-soonet-config

# 2. 상태 확인 (InService 될 때까지)
aws sagemaker describe-endpoint --endpoint-name say2-2team-soonet-endpoint \
  --query "EndpointStatus"

# 3. E2E 테스트
aws lambda invoke --function-name say2-2team-phase2-vision \
  --payload '{"xray_s3_key":"Phase_2/uploads/test_mimic.jpg","threshold":0.3}' \
  --cli-binary-format raw-in-base64-out /tmp/result.json
```

---

## 🗑️ 발표 후 즉시 삭제 (과금 방지)

```bash
# SageMaker 삭제 (즉시 과금 중단)
aws sagemaker delete-endpoint --endpoint-name say2-2team-soonet-endpoint
aws sagemaker delete-endpoint-config --endpoint-config-name say2-2team-soonet-config
aws sagemaker delete-model --model-name say2-2team-soonet-model

# 보안 서비스 비활성화 (월 $5~10 절감)
aws guardduty delete-detector --detector-id 94cf0cf8ca5c1d4657736fbde575830a
aws securityhub disable-security-hub
aws configservice stop-configuration-recorder --configuration-recorder-name say2-2team-config-recorder
```

---

## 💰 현재 과금 중인 리소스

| 리소스 | 시간당 | 비고 |
|--------|--------|------|
| GuardDuty | ~$0.004 | ⚠️ 권한 없음 — 미구성 |
| Security Hub | ~$0.001 | ⚠️ 권한 없음 — 미구성 |
| AWS Config | ~$0.003 | 리소스 수 기반 |
| CloudTrail (데이터 이벤트) | $0.10/100,000건 | 트래픽 따라 |
| SNS | 무료 (이메일 1,000건/월) | ✅ 2026-05-12 완료 |
| S3 (저장) | $0.023/GB/월 | 현재 ~200MB |
| **SageMaker Endpoint** | **$0.115/h** | **⚠️ 발표 직전에만 생성** |
| CloudFront | $0.0085/10,000건 | ⚠️ IAM 권한 필요 — 미구성 |
| WAF | $5/WebACL/월 + $1/백만 요청 | ⚠️ IAM 권한 필요 — 미구성 |
| ACM (SSL/TLS) | 무료 (퍼블릭 인증서) | ⬜ 커스텀 도메인 있을 경우 생성 |

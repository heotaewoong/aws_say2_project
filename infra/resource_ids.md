# say2-2team AWS 리소스 ID 모음

> **용도**: 배포된 AWS 리소스의 ID/ARN을 한 곳에서 관리  
> **계정**: 666803869796  
> **리전**: ap-northeast-2 (서울)  
> **최종 업데이트**: 2026-05-21

---

## ⚠️ 보안 주의사항

| 항목 | 저장 위치 | 이유 |
|------|----------|------|
| KMS Key ID | ✅ 이 파일 (공개 가능) | Key ID는 식별자일 뿐, 권한은 IAM Role이 제어 |
| AWS Access Key / Secret Key | ❌ `.env` 또는 AWS CLI 설정 | 절대 코드/문서에 하드코딩 금지 |
| Cognito App Client Secret | ❌ Secrets Manager | 런타임에 동적 로드 |
| NCBI API Key | ❌ Secrets Manager | 런타임에 동적 로드 |

---

## 🌐 VPC / 네트워크

| 항목 | 값 | 상태 |
|------|-----|------|
| VPC ID | `vpc-06dd0ad1f2335ea74` | ✅ |
| VPC 이름 | `say2-2team` | ✅ |
| CIDR | `10.0.0.0/24` | ✅ |

### Subnet

| 이름 | Subnet ID | CIDR | AZ | 상태 |
|------|-----------|------|-----|------|
| `say2-2team-public` | `subnet-0468cec99c4805a07` | 10.0.0.0/28 | ap-northeast-2a | ✅ |
| `say2-2team-private-1` | `subnet-02eed659772bac6aa` | 10.0.0.128/28 | ap-northeast-2a | ✅ |
| `say2-2team-private-2` | `subnet-08f8d0eaa597b4f04` | 10.0.0.16/28 | ap-northeast-2a | ✅ |
| `say2-2team-private-3` | `subnet-099966af5fc9c2090` | 10.0.0.32/28 | ap-northeast-2a | ✅ |

### Security Group

| 이름 | SG ID | 역할 | 상태 |
|------|-------|------|------|
| `say2-2team-sg-lambda` | `sg-08d35c498d8886a98` | Lambda 실행 | ✅ |
| `say2-2team-sg-sagemaker` | `sg-03e64fdde60d52a6c` | SageMaker Endpoint | ✅ |
| `say2-2team-sg-vpce` | `sg-0cf817a0115fa94bd` | VPC Endpoint ENI | ✅ |
| `fhir-ec2-sg` | `sg-03b9bc5d95699b797` | 2-2team-fhir-ec2 (RAG 실행 EC2) | ✅ |

**VPCE SG 인바운드 규칙 추가 내역 (2026-05-13)**

| Rule ID | 프로토콜 | 포트 | 소스 SG | 용도 |
|---------|---------|------|---------|------|
| `sgr-0cf29720f0be98328` | TCP | 443 | `sg-03b9bc5d95699b797` (fhir-ec2-sg) | EC2 → Bedrock VPCE 연결 허용 |

---

## 🪣 S3

| 항목 | 값 | 상태 |
|------|-----|------|
| 버킷 이름 | `say2-2team-bucket` | ✅ |
| Phase 2 prefix | `Phase_2/` | ✅ |
| 모델 경로 | `s3://say2-2team-bucket/Phase_2/models/soonet/model.tar.gz` (146MB) | ✅ |
| **RAG 스크립트** | `s3://say2-2team-bucket/RAG/rag_llm_3.py` | ✅ 2026-05-14 |
| **RAG 최종 보고서 PDF** | `s3://say2-2team-bucket/final_reports/{session_id}/report.pdf` | ✅ 2026-05-14 |
| 프론트엔드 | `s3://say2-2team-bucket/frontend/` | ✅ |
| 암호화 | SSE-KMS (`alias/say2-2team-data-key`) + Bucket Key | ✅ |
| 퍼블릭 접근 차단 | 4개 항목 모두 활성화 | ✅ |
| 버전 관리 | Enabled | ✅ |
| 수명 주기 | `say2-2team-model-lifecycle` (30일→Standard-IA, 비현재 7일→Glacier IR) | ✅ |

---

## 🔑 IAM Role

| Role 이름 | ARN | 상태 |
|-----------|-----|------|
| `say2-2team-sagemaker-role` | `arn:aws:iam::666803869796:role/say2-2team-sagemaker-role` | ✅ |
| `say2-2team-lambda-role` | `arn:aws:iam::666803869796:role/say2-2team-lambda-role` | ✅ |
| `say2-2team-config-role` | `arn:aws:iam::666803869796:role/say2-2team-config-role` | ✅ |

### Lambda Role 인라인 정책

| 정책 이름 | 권한 |
|----------|------|
| `Phase2Access` | S3 GetObject/PutObject (Phase_2/*), SageMaker InvokeEndpoint |
| `SecretsManagerAccess` | GetSecretValue (say2-2team/*) |
| `KMSAccess` | Decrypt, GenerateDataKey (say2-2team-data-key) |

---

## 🤖 SageMaker ✅ InService (2026-05-08)

| 항목 | 값 | 상태 |
|------|-----|------|
| Model 이름 | `say2-2team-soonet-model` | ✅ |
| Endpoint Config | `say2-2team-soonet-config` | ✅ |
| Endpoint 이름 | `say2-2team-soonet-endpoint` | ✅ **InService** |
| Instance Type | `ml.m5.large` (CPU, $0.115/h) | ✅ |
| 컨테이너 이미지 | `763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:2.0.0-cpu-py310-ubuntu20.04-sagemaker` | ✅ |
| 생성일 | 2026-05-08 | ✅ |

> ⚠️ **발표 종료 후 반드시 삭제** — $0.115/h 과금 중 (`aws sagemaker delete-endpoint --endpoint-name say2-2team-soonet-endpoint`)

---

## λ Lambda

| 항목 | 값 | 상태 |
|------|-----|------|
| 함수 이름 | `say2-2team-phase2-vision` | ✅ |
| Runtime | Python 3.11 | ✅ |
| Timeout | 5분 | ✅ |
| Memory | 1024 MB | ✅ |
| VPC | `say2-2team` / `say2-2team-private-1` | ✅ |

---

## 🔐 KMS

| 항목 | 값 | 상태 |
|------|-----|------|
| Key ID | `4a1be264-1ccf-4a2c-a937-3e6847a751d5` | ✅ |
| Key ARN | `arn:aws:kms:ap-northeast-2:666803869796:key/4a1be264-1ccf-4a2c-a937-3e6847a751d5` | ✅ |
| Alias | `alias/say2-2team-data-key` | ✅ |
| 용도 | S3 SSE-KMS (X-ray 이미지, 모델 가중치), DynamoDB 암호화 | ✅ |
| 태그 | `project = pre-kms-2-2-team` | ✅ |
| CloudTrail KMS | `alias/say2-2team-cloudtrail-key` (별도 키) | ✅ |

---

## 🔒 Secrets Manager

| Secret 이름 | 포맷 | 저장 값 | 용도 | 상태 |
|------------|------|--------|------|------|
| `say2-2team/phase2-config` | JSON | `sagemaker_endpoint`, `s3_bucket`, `s3_prefix`, `xray_threshold` | Phase 2 Lambda 설정 | ✅ |
| `rare-link-ai/aurora/app-user` | plain text | 비밀번호 (app_user 계정) | RAG → Aurora 연결 | ✅ |
| `rare-link-ai/aurora/master` | plain text | 비밀번호 (postgres 계정) | Aurora 관리 | ✅ |
| `rare-link-ai/aurora/hapi-user` | plain text | 비밀번호 (hapi_user 계정) | HAPI FHIR 서버 | ✅ |

> ⚠️ `rare-link-ai/aurora/*` 시크릿은 **plain text(비밀번호만)** 저장 방식 — `json.loads()` 불가, `SecretString` 직접 사용

| 태그 | `project = pre-secretsmanager-2-2-team` | ✅ |

---

## 📋 CloudTrail

| 항목 | 값 | 상태 |
|------|-----|------|
| Trail 이름 | `say2-2team-audit-trail` | ✅ |
| Trail ARN | `arn:aws:cloudtrail:ap-northeast-2:666803869796:trail/say2-2team-audit-trail` | ✅ |
| S3 버킷 | `aws-cloudtrail-logs-666803869796-9e80af9d` | ✅ |
| CloudWatch Log Group | `say2-2team-cloudtrail-logs` | ✅ |
| 이벤트 | Management + S3 Data events (say2-2team-bucket) | ✅ |
| 태그 | `project = pre-cloudtrail-2-2-team` | ✅ |

---

## 🛡️ GuardDuty

| 항목 | 값 | 상태 |
|------|-----|------|
| Detector ID | `94cf0cf8ca5c1d4657736fbde575830a` | ⚠️ 권한 없음 — 미구성 |
| Status | `ENABLED` | ⚠️ 권한 없음 |
| Finding 발행 주기 | 15분 | ⚠️ 미구성 |
| S3 내보내기 | `say2-2team-bucket` (Status: PUBLISHING) | ⚠️ 미구성 |
| Destination ID | `a4cf0cfca7d256a4f47868e15a2252b2` | ⚠️ 미구성 |

---

## 🔍 Security Hub

| 항목 | 값 | 상태 |
|------|-----|------|
| Hub ARN | `arn:aws:securityhub:ap-northeast-2:666803869796:hub/default` | ⚠️ 권한 없음 — 미구성 |
| 구독 시각 | - | ⚠️ 미구성 |
| 활성 표준 | - | ⚠️ 미구성 |
| GuardDuty 연동 | - | ⚠️ 미구성 |

---

## ⚙️ AWS Config

| 항목 | 값 | 상태 |
|------|-----|------|
| Recorder 이름 | `say2-2team-config-recorder` | ✅ |
| Recording | `true` (SUCCESS) | ✅ |
| Delivery Channel | `say2-2team-config-delivery` → `say2-2team-bucket/config-logs` | ✅ |
| IAM Role | `say2-2team-config-role` | ✅ |

---

## 🔔 SNS

| Topic 이름 | ARN | 리전 | 용도 | 상태 |
|-----------|-----|------|------|------|
| `say2-2team-alerts` | `arn:aws:sns:ap-northeast-2:666803869796:say2-2team-alerts` | ap-northeast-2 | Lambda/SageMaker/S3 보안 알람 | ✅ |
| `say2-2team-billing-alerts` | `arn:aws:sns:us-east-1:666803869796:say2-2team-billing-alerts` | us-east-1 | 비용 초과 알람 (Billing 메트릭 전용) | ✅ |

---

## 📊 CloudWatch 알람

| 알람 이름 | 메트릭 | 임계값 | SNS | 리전 | 상태 |
|----------|--------|--------|-----|------|------|
| `say2-2team-lambda-error-alarm` | Lambda Errors | ≥3회/5분 | say2-2team-alerts | ap-northeast-2 | ✅ |
| `say2-2team-sagemaker-latency-alarm` | SageMaker ModelLatency | ≥30초 | say2-2team-alerts | ap-northeast-2 | ✅ |
| `say2-2team-s3-delete-alarm` | S3DeleteBucketCount | ≥1회 | say2-2team-alerts | ap-northeast-2 | ✅ |
| `say2-2team-cost-80usd-alarm` | EstimatedCharges (전체) | ≥$80/일 | say2-2team-billing-alerts | us-east-1 | ✅ |
| `say2-2team-sagemaker-cost-10usd-alarm` | EstimatedCharges (SageMaker) | ≥$10/일 | say2-2team-billing-alerts | us-east-1 | ✅ |

---

## 🌐 CloudFront + WAF ✅ 2026-05-13

### 프론트엔드 빌드 현황

| 항목 | 값 | 상태 |
|------|-----|------|
| 빌드 도구 | Vite 5 (React 18 + Tailwind) | ✅ |
| S3 업로드 경로 | `s3://say2-2team-bucket/frontend/` | ✅ 2026-05-12 |
| index.html | `cache-control: no-cache` | ✅ |
| JS/CSS assets | `cache-control: max-age=31536000,immutable` | ✅ |

### CloudFront 배포 정보 ✅ 완료 (2026-05-13)

| 항목 | 값 | 상태 |
|------|-----|------|
| Distribution 이름 | `say2-2team-cf-distribution` | ⚠️ IAM 권한 필요 |
| **Distribution ID** | **`E2ZHONIV05TX9D`** | ⚠️ IAM 권한 필요 |
| **CloudFront URL** | **`https://d300v14l8u0wx7.cloudfront.net`** | ⚠️ IAM 권한 필요 |
| ARN | `arn:aws:cloudfront::666803869796:distribution/E2ZHONIV05TX9D` | ⚠️ IAM 권한 필요 |
| Origin | `say2-2team-bucket.s3.ap-northeast-2.amazonaws.com/frontend` | ⚠️ IAM 권한 필요 |
| Private S3 access (OAC) | - | ⚠️ IAM 권한 필요 |
| Default root object | `index.html` | ⚠️ IAM 권한 필요 |
| Error pages | 403/404 → `/index.html` (200) | ⚠️ IAM 권한 필요 |
| S3 Bucket policy (OAC) | - | ⚠️ IAM 권한 필요 |
| 태그 | `project = pre-cloudfront-2-2-team` | ⚠️ IAM 권한 필요 |

### WAF WebACL ✅ 완료 (2026-05-13)

| 항목 | 값 | 상태 |
|------|-----|------|
| WebACL 이름 | `say2-2team-waf` | ⚠️ IAM 권한 필요 |
| **WebACL ARN** | **`arn:aws:wafv2:us-east-1:666803869796:global/webacl/say2-2team-waf/6884d8c7-f2b3-4916-b513-40ed0b9bbd12`** | ⚠️ IAM 권한 필요 |
| 리전 | `us-east-1` (CloudFront 전용 글로벌) | ⚠️ IAM 권한 필요 |
| 보호 수준 | Essentials (Layer7 DDoS, IP blocklist, SQLi, XSS, Known bad inputs) | ⚠️ IAM 권한 필요 |
| 연결 배포 | `E2ZHONIV05TX9D` (say2-2team-cf-distribution) | ⚠️ IAM 권한 필요 |
| 태그 | `project = pre-waf-2-2-team` | ⚠️ IAM 권한 필요 |

---

## 🔒 Cognito ✅ 2026-05-21

| 항목 | 값 | 상태 |
|------|-----|------|
| User Pool 이름 | `say2-2team-rare-link-pool` | ✅ |
| User Pool ID | `ap-northeast-2_CMtZTRCTa` | ✅ |
| User Pool ARN | `arn:aws:cognito-idp:ap-northeast-2:666803869796:userpool/ap-northeast-2_CMtZTRCTa` | ✅ |
| App Client ID | `1280u1fg8gbvt1g21sv8dn4246` | ✅ |
| 현재 유저 수 | 5명 | ✅ |
| 테스트 계정 | `doctor-test@say2team.com` | ⬜ 생성 필요 |

---

## 🗄️ DynamoDB (Step 18 완료 후 기입)

| 테이블 이름 | ARN | 상태 |
|------------|-----|------|
| `say2-2team-diagnosis-history` | _(생성 후 기입)_ | ⬜ |
| `say2-2team-rare-case-collection` | _(생성 후 기입)_ | ⬜ |

---

## 📡 API Gateway (Step 17 완료 후 기입)

| 항목 | 값 |
|------|-----|
| API 이름 | `say2-2team-diagnose-api` |
| Invoke URL | _(배포 후 기입)_ |
| Stage | `prod` |

---

## 💰 비용 알람 요약 (Step 21)

| 알람 | 임계값 | 알림 채널 |
|------|--------|---------|
| 전체 비용 | $80/일 초과 시 | say2-2team-billing-alerts SNS (us-east-1) |
| SageMaker 비용 | $10/일 초과 시 | say2-2team-billing-alerts SNS (us-east-1) |

> ⚠️ **AWS Budgets 권한 없음** — CloudWatch Billing 알람으로 대체 (동일 효과)  
> Billing 메트릭은 AWS 특성상 **us-east-1**에서만 조회 가능

---

## 🌐 ACM (Step 23 완료 후 기입)

| 항목 | 값 | 상태 |
|------|-----|------|
| 인증서 ARN | _(생성 후 기입)_ | ⬜ |
| 도메인 | _(커스텀 도메인 확정 후 기입)_ | ⬜ |
| 리전 | `us-east-1` (CloudFront 연동 필수) | ⬜ |

---

## 🌍 Route 53 (Step 26 — 커스텀 도메인 있을 경우)

| 항목 | 값 | 상태 |
|------|-----|------|
| Hosted Zone ID | _(커스텀 도메인 확정 후 기입)_ | ⬜ |
| 도메인 | _(확정 후 기입)_ | ⬜ |

---

## 🖥️ EC2 — RAG 실행 환경 ✅ 2026-05-13

| 항목 | 값 | 상태 |
|------|-----|------|
| Instance 이름 | `2-2team-fhir-ec2` | ✅ |
| Instance ID | `i-0f3f223fd40217b12` | ✅ **running** |
| Instance Type | `t3.large` | ✅ |
| OS | Ubuntu 25.04 (Noble) | ✅ |
| Security Group | `fhir-ec2-sg` (`sg-03b9bc5d95699b797`) | ✅ |
| 접속 방식 | SSM Session Manager (키 없이) | ✅ |
| 용도 | `rag_llm_3.py` 실행 — Bedrock 추론 + PDF 생성 + Aurora 저장 | ✅ |
| IAM Role | `fhir-ec2-role` (Bedrock·S3·SecretsManager 권한 포함) | ✅ |

**RAG 실행 명령어 (SSM)**
```bash
aws ssm start-session --target i-0f3f223fd40217b12 --region ap-northeast-2
# EC2 내부:
aws s3 cp s3://say2-2team-bucket/RAG/rag_llm_3.py . --region ap-northeast-2
python3 rag_llm_3.py
```

## ⚖️ ALB + EC2 — HAPI FHIR 서버 (Step 27 완료 후 기입)

| 항목 | 값 | 상태 |
|------|-----|------|
| ALB ARN | _(생성 후 기입)_ | ⬜ |
| EC2 Instance ID | _(생성 후 기입)_ | ⬜ |
| 용도 | HAPI FHIR 서버 (EMR 연동) | ⬜ |

---

## 📨 SQS — FHIR 요청 큐 (Step 28 완료 후 기입)

| 항목 | 값 | 상태 |
|------|-----|------|
| Queue URL | _(생성 후 기입)_ | ⬜ |
| Queue ARN | _(생성 후 기입)_ | ⬜ |
| 용도 | FHIR 요청 큐 (EMR 연동) | ⬜ |

---

## 🗄️ ElastiCache Redis — HPO 버퍼 (Step 29 완료 후 기입)

| 항목 | 값 | 상태 |
|------|-----|------|
| Cluster ID | _(생성 후 기입)_ | ⬜ |
| Endpoint | _(생성 후 기입)_ | ⬜ |
| 용도 | HPO 버퍼 (Phase 4·5) | ⬜ |

---

## 🔄 Step Functions — 진단 오케스트레이터 (Step 30 완료 후 기입)

| 항목 | 값 | 상태 |
|------|-----|------|
| State Machine ARN | _(생성 후 기입)_ | ⬜ |
| 이름 | _(생성 후 기입)_ | ⬜ |
| 용도 | 진단 파이프라인 오케스트레이션 (Phase 4·5) | ⬜ |

---

## 📅 EventBridge — MLOps 트리거 (Step 31 완료 후 기입)

| 항목 | 값 | 상태 |
|------|-----|------|
| Rule ARN | _(생성 후 기입)_ | ⬜ |
| 이름 | _(생성 후 기입)_ | ⬜ |
| 용도 | MLOps 자동화 트리거 | ⬜ |

---

## 📦 ECR — 컨테이너 이미지 저장소 (Step 32 완료 후 기입)

| 항목 | 값 | 상태 |
|------|-----|------|
| Repository URI | _(생성 후 기입)_ | ⬜ |
| 이름 | _(생성 후 기입)_ | ⬜ |
| 용도 | MLOps 컨테이너 이미지 저장 | ⬜ |

---

## 💾 AWS Backup — 데이터 백업 정책 (Step 33 완료 후 기입)

| 항목 | 값 | 상태 |
|------|-----|------|
| Backup Plan ID | _(생성 후 기입)_ | ⬜ |
| Vault 이름 | _(생성 후 기입)_ | ⬜ |
| 용도 | S3 / DynamoDB / Aurora 백업 | ⬜ |

---

## 🗃️ Aurora — RAG/FHIR 백엔드 DB ✅ (팀 공유 리소스)

| 항목 | 값 | 상태 |
|------|-----|------|
| Cluster 식별자 | `patient-db-cluster` | ✅ available |
| Cluster ARN | `arn:aws:rds:ap-northeast-2:666803869796:cluster:patient-db-cluster` | ✅ |
| Writer Endpoint | `patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com` | ✅ |
| Reader Endpoint | `patient-db-cluster.cluster-ro-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com` | ✅ |
| Engine | Aurora PostgreSQL 16.11 | ✅ |
| Port | 5432 | ✅ |
| Master User | `postgres` | ✅ |
| RAG App User | `app_user` (비밀번호: `rare-link-ai/aurora/app-user` Secret) | ✅ |
| RAG 스키마 | `rarelinkai` | ✅ |
| RAG 사용 테이블 | `final_report`, `diagnosis_session`, `patient_profile`, `raw_emr_bundle`, `rag_api_cache` | ✅ |
| Aurora SG | `sg-019a357627f1594db` (Lambda SG 허용 포트 5432) | ✅ |

**Aurora FK 체인 (INSERT 순서)**
```
raw_emr_bundle → patient_profile → diagnosis_session → final_report
```

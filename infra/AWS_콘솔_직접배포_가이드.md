# AWS 인프라 배포 가이드 (담당: 허태웅)

> **기준 문서**: `aws_architecture/architecture_final.html`  
> **담당 범위**: VPC · Subnet · 보안그룹 · Phase 2 (저장·경로·Lambda·SageMaker)  
> **계정**: 666803869796 (say2 프로그램 공유)  
> **리전**: ap-northeast-2 (서울)  
> **최종 업데이트**: 2026-05-13

---

## 🚀 발표 직전 체크리스트 (5분 전)

> ✅ **2026-05-14 현재**: SageMaker Endpoint `say2-2team-soonet-endpoint` → **이미 InService** 상태  
> ✅ EC2 `2-2team-fhir-ec2` (i-0f3f223fd40217b12) → **running**, RAG 실행 가능  
> ✅ CloudFront `d300v14l8u0wx7.cloudfront.net` → **Deployed**  
> ✅ Aurora `patient-db-cluster` → **available**, RAG 보고서 저장 완료  

```bash
export AWS_DEFAULT_REGION=ap-northeast-2
# ⚠️ AWS Access Key는 환경변수로만 설정 (문서 하드코딩 금지)
```

### 1단계: SageMaker Endpoint 상태 확인 (이미 InService)

```bash
aws sagemaker describe-endpoint --endpoint-name say2-2team-soonet-endpoint \
  --query "EndpointStatus" --region ap-northeast-2
# → "InService" 이면 OK, 다시 생성 불필요
```

생성이 필요한 경우 (삭제 후 재생성) SageMaker 콘솔 또는 CLI:

```bash
# Model 생성
aws sagemaker create-model \
  --model-name say2-2team-soonet-model \
  --execution-role-arn arn:aws:iam::666803869796:role/say2-2team-sagemaker-role \
  --primary-container '{
    "Image": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:2.0.0-cpu-py310-ubuntu20.04-sagemaker",
    "ModelDataUrl": "s3://say2-2team-bucket/Phase_2/models/soonet/model.tar.gz",
    "Environment": {
      "SAGEMAKER_PROGRAM": "inference.py",
      "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code"
    }
  }' \
  --vpc-config "Subnets=subnet-02eed659772bac6aa,SecurityGroupIds=sg-03e64fdde60d52a6c" \
  --region ap-northeast-2

# Endpoint Config 생성
aws sagemaker create-endpoint-config \
  --endpoint-config-name say2-2team-soonet-config \
  --production-variants '[{
    "VariantName": "primary",
    "ModelName": "say2-2team-soonet-model",
    "InitialInstanceCount": 1,
    "InstanceType": "ml.m5.large"
  }]' \
  --region ap-northeast-2

# Endpoint 생성 (InService까지 5~10분)
aws sagemaker create-endpoint \
  --endpoint-name say2-2team-soonet-endpoint \
  --endpoint-config-name say2-2team-soonet-config \
  --region ap-northeast-2

# 상태 확인 (InService 될 때까지 반복)
aws sagemaker describe-endpoint \
  --endpoint-name say2-2team-soonet-endpoint \
  --query "EndpointStatus" \
  --region ap-northeast-2
```

### 2단계: E2E 테스트 (Endpoint InService 확인 후)

```bash
# MIMIC 테스트 이미지로 Lambda 호출
aws lambda invoke \
  --function-name say2-2team-phase2-vision \
  --payload '{"xray_s3_key": "Phase_2/uploads/test_mimic.jpg", "threshold": 0.3}' \
  --cli-binary-format raw-in-base64-out \
  --region ap-northeast-2 \
  /tmp/phase2_result.json

cat /tmp/phase2_result.json | python3 -m json.tool
```

**기대 출력:**
```json
{
  "statusCode": 200,
  "body": {
    "positive_hpos": ["HP:0002202", "HP:0002113", ...],
    "predictions": { "Pleural Effusion": {"probability": 0.95, ...}, ... }
  }
}
```

### 3단계: 발표 후 즉시 삭제 (과금 방지)

```bash
# Endpoint 삭제 (즉시 과금 중단)
aws sagemaker delete-endpoint \
  --endpoint-name say2-2team-soonet-endpoint \
  --region ap-northeast-2

# Endpoint Config 삭제
aws sagemaker delete-endpoint-config \
  --endpoint-config-name say2-2team-soonet-config \
  --region ap-northeast-2

# Model 삭제
aws sagemaker delete-model \
  --model-name say2-2team-soonet-model \
  --region ap-northeast-2

echo "✅ SageMaker 리소스 삭제 완료 — 과금 중단"
```

---

## 📐 architecture_final.html에서 내 담당 영역

```
┌──────────────────── VPC (say2-2team, 10.0.0.0/24) ──────────────────┐
│                                                                      │
│  ┌─── Public Subnet ────────────┐                                   │
│  │  🌍 Internet Gateway 연결     │                                   │
│  │  🌍 NAT Gateway               │  ← 외부 API 아웃바운드            │
│  │  (say2-2team-public)          │                                  │
│  └──────────────────────────────┘                                   │
│                                                                      │
│  ┌─── Private Subnet ① ─────────┐  ← **내 핵심 담당**              │
│  │  🩻 SageMaker Endpoint       │                                  │
│  │     (SooNet DenseNet-121)    │                                  │
│  │  λ Lambda (Phase 2 Vision)   │                                  │
│  │  (say2-2team-private-1)      │                                  │
│  └──────────────────────────────┘                                   │
│                                                                      │
│  ┌─── Private Subnet ② ─────────┐  ← 자리 예약 (Phase 4·5)         │
│  │  (say2-2team-private-2)      │                                  │
│  └──────────────────────────────┘                                   │
│                                                                      │
│  ┌─── Private Subnet ③ ─────────┐  ← 자리 예약 (RAG & Report)      │
│  │  (say2-2team-private-3)      │                                  │
│  └──────────────────────────────┘                                   │
│                                                                      │
│  VPC Endpoint:                                                      │
│  🔗 S3 (Gateway)    — X-ray 이미지 접근                              │
│  🔗 SageMaker Runtime (Interface)                                   │
│  🔗 Bedrock Runtime (Interface)                                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📋 전체 진행 현황

| 단계 | 리소스 | 상태 |
|------|--------|------|
| 1 | VPC | ✅ 기존 팀 VPC 활용 |
| 2 | Subnet 4개 | ✅ 완료 |
| 3 | Security Group 3개 | ✅ 완료 |
| 4 | **IGW + NAT Gateway + Route Table** | ✅ 완료 (옵션 A) |
| 5 | S3 폴더 (`Phase_2/`) | ✅ 완료 (기존 버킷 재활용) |
| 6 | IAM Role 2개 | ✅ 완료 (CLI로 태그 포함 생성) |
| 7 | 모델 파일 S3 업로드 | ✅ 완료 (`Phase_2/models/soonet/model.tar.gz`, 146MB, 최신 가중치 포함) |
| 8 | SageMaker Endpoint | ⏸️ 발표 직전에 생성 (과금 방지) |
| 9 | Lambda 함수 | ✅ 완료 (CLI로 생성, 2026-05-08) |
| 10 | 테스트 | ⬜ |
| **11** | **Cognito User Pool** | **⬜ 보안** |
| **12** | **Secrets Manager** | **⬜ 보안** |
| **13** | **KMS (S3/DynamoDB 암호화)** | **✅ 완료 (2026-05-12)** |
| **14** | **CloudTrail (감사 로그)** | **✅ 완료 (2026-05-12)** |
| **15** | **GuardDuty + Security Hub** | **✅ 완료 (2026-05-12)** |
| **16** | **S3 버킷 보안 강화** | **✅ 완료 (2026-05-12)** |
| **17** | **API Gateway + Cognito Authorizer** | **⬜ 보안** |
| **18** | **Aurora DB** | **✅ 이미 존재 (`rarelinkai.final_report` 사용)** |
| **19** | **Aurora VPC SG 수정** | **✅ 완료 (CLI, 2026-05-13)** |
| **20** | **SNS 알림** | **✅ 완료 (2026-05-12)** |
| **21** | **AWS Budgets** | **✅ 완료 (CloudWatch Billing 알람으로 대체, 2026-05-12)** |
| **22** | **Detective + Inspector** | **⬜ 보안** |
| **23** | **ACM (SSL/TLS 인증서)** | **⬜ 커스텀 도메인 있을 경우** |
| **24** | **CloudFront 배포** | **✅ 완료 (`E2ZHONIV05TX9D`, 2026-05-13)** |
| **25** | **WAF WebACL (Global)** | **✅ 완료 (`say2-2team-waf`, us-east-1, 2026-05-13)** |
| **26** | **Route 53 DNS** | **⬜ 도메인 (커스텀 도메인 있을 경우)** |
| **27** | **ALB + EC2 (HAPI FHIR 서버)** | **⬜ EMR 연동** |
| **28** | **SQS (FHIR 요청 큐)** | **⬜ EMR 연동** |
| **29** | **ElastiCache (Redis) — HPO 버퍼** | **⬜ Phase 4·5** |
| **30** | **Step Functions — 진단 오케스트레이터** | **⬜ Phase 4·5** |
| **31** | **EventBridge — MLOps 트리거** | **⬜ MLOps** |
| **32** | **ECR — 컨테이너 이미지 저장소** | **⬜ MLOps** |
| **33** | **AWS Backup — 데이터 백업 정책** | **⬜ 복원력** |
| **34** | **Aurora (FHIR 백엔드 DB)** | **⬜ EMR 연동** |

---

## ✅ Step 1 — VPC (기존 팀 VPC 활용)

| 항목 | 값 |
|------|-----|
| VPC ID | `vpc-06dd0ad1f2335ea74` |
| 이름 | `say2-2team` |
| CIDR | `10.0.0.0/24` |
| 태그 | `project = pre-2-2team` |

---

## ✅ Step 2 — Subnet 4개 생성 완료

| 이름 | Subnet ID | CIDR | AZ | architecture_final 대응 |
|------|-----------|------|-----|------------------------|
| `say2-2team-public` | subnet-0468cec99c4805a07 | 10.0.0.0/28 | ap-northeast-2a | Public Subnet |
| `say2-2team-private-1` | subnet-02eed659772bac6aa | 10.0.0.128/28 | ap-northeast-2a | **Private Subnet ①** ← 내 핵심 |
| `say2-2team-private-2` | subnet-08f8d0eaa597b4f04 | 10.0.0.16/28 | ap-northeast-2a | Private Subnet ② |
| `say2-2team-private-3` | subnet-099966af5fc9c2090 | 10.0.0.32/28 | ap-northeast-2a | Private Subnet ③ |

**태그**: `project = pre-vpc-2-2-team` ✅

---

## ✅ Step 3 — Security Group 3개 생성 완료

| 이름 | SG ID | 역할 | Inbound 규칙 |
|------|-------|------|-------------|
| `say2-2team-sg-lambda` | sg-08d35c498d8886a98 | Lambda 실행 | 없음 |
| `say2-2team-sg-sagemaker` | sg-03e64fdde60d52a6c | SageMaker Endpoint | HTTPS ← Lambda SG |
| `say2-2team-sg-vpce` | sg-0cf817a0115fa94bd | VPC Endpoint ENI | HTTPS ← Lambda/SageMaker SG |

---

# ⬜ Step 4 — 외부 아웃바운드 경로 구성

**두 가지 선택지 중 하나 선택:**

## 🅐 옵션 A: IGW + NAT Gateway (architecture_final.html 원본 설계)

architecture_final.html에 명시된 원본 방식. **Public Subnet에 NAT를 두어 Private Subnet의 외부 인터넷 아웃바운드 담당.**

### 동작 원리

```
Private Subnet Lambda
    │ (외부 API 호출, 예: PubMed)
    ▼
Route Table → NAT Gateway (Public Subnet)
    ▼
Internet Gateway (VPC)
    ▼
인터넷 (PubMed, ClinicalTrials 등)
```

### 필요 리소스 (5개)

1. **Internet Gateway** 생성 + VPC 연결
2. **Elastic IP** 1개 할당 (NAT에 부여할 고정 IP)
3. **NAT Gateway** 생성 (Public Subnet에 배치)
4. **Public Route Table** — `0.0.0.0/0 → IGW`
5. **Private Route Table** — `0.0.0.0/0 → NAT Gateway`

### 콘솔 배포 절차

#### 4A-1. Internet Gateway 생성

1. VPC 콘솔 → Internet gateways → "Create internet gateway"
2. Name tag: `say2-2team-igw`
3. Tag 추가: `project = pre-vpc-2-2-team`
4. Create → 생성 후 "Actions" → "Attach to VPC" → `say2-2team` 선택

#### 4A-2. Elastic IP 할당

1. VPC 콘솔 → Elastic IPs → "Allocate Elastic IP address"
2. Network Border Group: `ap-northeast-2`
3. Name tag: `say2-2team-nat-eip`
4. Tag 추가: `project = pre-vpc-2-2-team`
5. Allocate

#### 4A-3. NAT Gateway 생성

1. VPC 콘솔 → NAT gateways → "Create NAT gateway"
2. Name: `say2-2team-nat`
3. Subnet: `say2-2team-public` 선택 ⚠️ **Public Subnet에 배치 필수**
4. Connectivity type: `Public`
5. Elastic IP: 방금 만든 `say2-2team-nat-eip` 선택
6. Tag: `project = pre-vpc-2-2-team`
7. Create (5분 소요)

#### 4A-4. Public Route Table 만들고 연결

1. VPC 콘솔 → Route tables → "Create route table"
2. Name: `say2-2team-rt-public`
3. VPC: `say2-2team`
4. Tag: `project = pre-vpc-2-2-team`
5. 생성 후 → **Routes 탭** → "Edit routes" → "Add route":
   - Destination: `0.0.0.0/0`
   - Target: `Internet Gateway` → `say2-2team-igw`
6. **Subnet associations 탭** → "Edit" → `say2-2team-public` 체크

#### 4A-5. Private Route Table 만들고 연결

1. VPC 콘솔 → Route tables → "Create route table"
2. Name: `say2-2team-rt-private`
3. VPC: `say2-2team`
4. Tag: `project = pre-vpc-2-2-team`
5. 생성 후 → "Edit routes" → "Add route":
   - Destination: `0.0.0.0/0`
   - Target: `NAT Gateway` → `say2-2team-nat`
6. **Subnet associations 탭** → `say2-2team-private-1`, `-2`, `-3` 모두 체크

### ⚠️ 오류 시 확인 사항

현재 계정 한도 상태 (2026-05-06 기준):
- IGW: 5/5 사용 중
- EIP: 5/5 사용 중

**오류 메시지별 대응:**

| 오류 | 원인 | 강사/운영진에게 보고 |
|------|------|-------------------|
| `InternetGatewayLimitExceeded` | IGW 한도 초과 | "say2-2team에 IGW 생성 필요. 한도 늘려주세요" |
| `AddressLimitExceeded` | EIP 한도 초과 | "NAT Gateway용 EIP 1개 할당 필요" |
| `NatGatewayLimitExceeded` | NAT 한도 초과 | (기본 5개라 가능성 낮음) |

### 비용 (옵션 A)

| 리소스 | 시간당 | 1일 |
|--------|--------|-----|
| Internet Gateway | $0 | $0 |
| Elastic IP (연결 중) | $0.005 | $0.12 |
| NAT Gateway | $0.045 | $1.08 |
| NAT 데이터 처리 | $0.045/GB | 트래픽 따라 |
| **합계 (유휴)** | **$0.05** | **$1.20** |

---

## 🅑 옵션 B: VPC Endpoint만 사용 (NAT 없이 대체)

Phase 2에서는 외부 인터넷이 **사실상 필요 없음** (S3, SageMaker, Bedrock 모두 AWS 내부). VPC Endpoint로 내부 통신만 구성해도 Phase 2 작동.

### 동작 원리

```
Private Subnet Lambda
    │ (S3 / SageMaker / Bedrock 호출)
    ▼
VPC Endpoint (VPC 내부 전용 통로)
    ▼
AWS 서비스 (인터넷 안 거침)
```

### 장점
- IGW / EIP / NAT 불필요 → 한도 문제 회피
- NAT 대비 보안 강함 (인터넷 노출 0)
- 비용 절감 (NAT $0.045/h vs Interface Endpoint $0.014/h)

### 단점
- 외부 API (PubMed, ClinicalTrials 등) 호출 불가 → Phase 4·5에서 필요 시 추가 구성 필요
- S3 Gateway Endpoint 외에 Interface Endpoint는 시간당 과금

### 필요 리소스 (3개)

1. **S3 Gateway Endpoint** (무료)
2. **SageMaker Runtime Interface Endpoint**
3. **Bedrock Runtime Interface Endpoint**

### 콘솔 배포 절차

#### 4B-1. S3 Gateway Endpoint

> S3 Gateway Endpoint는 Route Table을 통해 라우팅되므로 **Private Route Table이 먼저 있어야 함**. 옵션 B만 쓸 거면 다음과 같이 간단한 Private Route Table을 만들면 됨:

```
VPC 콘솔 → Route tables → Create route table
  Name: say2-2team-rt-private
  VPC: say2-2team
  (Route는 비워둔 채 생성 — S3 Endpoint가 자동으로 추가함)

Subnet associations → private-1, -2, -3 연결
```

그 다음:

1. VPC 콘솔 → Endpoints → "Create endpoint"
2. Name: `say2-2team-vpce-s3`
3. Service category: `AWS services`
4. Services: `com.amazonaws.ap-northeast-2.s3` (Type=**Gateway** 필터)
5. VPC: `say2-2team`
6. Route tables: `say2-2team-rt-private` 체크
7. Policy: Full access
8. Tag: `project = pre-vpc-2-2-team`
9. Create

#### 4B-2. SageMaker Runtime Interface Endpoint

1. VPC 콘솔 → Endpoints → "Create endpoint"
2. Name: `say2-2team-vpce-sagemaker-runtime`
3. Services: `com.amazonaws.ap-northeast-2.sagemaker.runtime` (Type=**Interface**)
4. VPC: `say2-2team`
5. Subnets: `say2-2team-private-1` (Phase 2 배포 위치)
6. Security groups: `say2-2team-sg-vpce`
7. Private DNS: ✅ Enable
8. Policy: Full access
9. Tag: `project = pre-vpc-2-2-team`
10. Create

#### 4B-3. Bedrock Runtime Interface Endpoint

위와 동일 절차, Service만 `com.amazonaws.ap-northeast-2.bedrock-runtime` 으로.

### 비용 (옵션 B)

| 리소스 | 시간당 | 1일 |
|--------|--------|-----|
| S3 Gateway Endpoint | $0 | $0 |
| SageMaker Runtime Interface | $0.014 | $0.34 |
| Bedrock Runtime Interface | $0.014 | $0.34 |
| 데이터 처리 | $0.01/GB | 트래픽 따라 |
| **합계 (유휴)** | **$0.028** | **$0.67** |

---

## 🎯 옵션 선택 기준

| 상황 | 권장 옵션 |
|------|---------|
| IGW/EIP 한도 확보 가능 (강사/운영진 승인) | **A** (architecture_final 원본) |
| 당장 배포 필요, 한도 해결 기다릴 시간 없음 | **B** (VPC Endpoint) |
| Phase 4·5에서 외부 API (PubMed, ClinicalTrials) 호출 예정 | **A** 필수 |
| 시연 때 Phase 2 작동만 보여주면 됨 | **B** 충분 |

**권장 진행 순서:**
1. 먼저 **옵션 A 시도** → 오류 나면 강사에게 보고하고 한도 증설 요청
2. 답변 기다리는 동안 **옵션 B**로 배포해서 Phase 2 동작 확인
3. 한도 증설 승인되면 옵션 A로 전환 (NAT 추가만 하면 됨)

---

# ✅ Step 5 — S3 폴더 구성 (기존 버킷 재활용)

| 항목 | 값 |
|------|-----|
| 버킷 | `say2-2team-bucket` (기존 팀 공용 버킷 재활용) |
| Phase 2 prefix | `Phase_2/` |
| 폴더 구조 | `Phase_2/uploads/`, `Phase_2/results/`, `Phase_2/models/` |

```
say2-2team-bucket/
├── cheXpert_data/          ← 기존 (건드리지 않음)
├── mimic_data/             ← 기존 (건드리지 않음)
└── Phase_2/                ← ✅ 이미 생성 완료
    ├── uploads/            ← X-ray 업로드
    ├── results/            ← 진단 결과 JSON
    └── models/soonet/      ← model.tar.gz
```

> ✅ **이미 완료** — `Phase_2/` 폴더가 콘솔에서 확인됨. 추가 작업 불필요.

---

# ✅ Step 6 — IAM Role 2개 (CLI로 완료)

> ✅ **2026-05-08 CLI로 태그 포함 생성 완료**

| Role | ARN | 태그 | 정책 |
|------|-----|------|------|
| `say2-2team-sagemaker-role` | `arn:aws:iam::666803869796:role/say2-2team-sagemaker-role` | `project=pre-iam-2-2-team` ✅ | `AmazonSageMakerFullAccess` + `S3ModelAccess` |
| `say2-2team-lambda-role` | `arn:aws:iam::666803869796:role/say2-2team-lambda-role` | `project=pre-iam-2-2-team` ✅ | `AWSLambdaBasicExecutionRole` + `AWSLambdaVPCAccessExecutionRole` + `Phase2Access` |

콘솔 확인: IAM → Roles → `say2-2team` 검색

## Role 1: SageMaker 실행 Role

1. IAM 콘솔 → Roles → Create role
2. Trusted entity type: AWS service
3. Use case: SageMaker
4. Permissions: `AmazonSageMakerFullAccess` (자동 선택)
5. Role name: `say2-2team-sagemaker-role`
6. **Tags 단계 → 아무것도 입력하지 말고 "Create role" 클릭** ← 태그 넣으면 오류 #수정 
7. 생성 후 → Add permissions → Create inline policy → JSON:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::say2-2team-bucket",
      "arn:aws:s3:::say2-2team-bucket/Phase_2/*"
    ]
  }]
}
```
Policy name: `S3ModelAccess` → Save

## Role 2: Lambda 실행 Role

1. IAM → Create role → Lambda
2. Permissions (검색해서 체크):
   - `AWSLambdaBasicExecutionRole`
   - `AWSLambdaVPCAccessExecutionRole`
3. Role name: `say2-2team-lambda-role`
4. **Tags 단계 → 아무것도 입력하지 말고 "Create role" 클릭** ← 태그 넣으면 오류
5. Create
6. Inline policy 추가:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::say2-2team-bucket/Phase_2/*"
    },
    {
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::say2-2team-bucket",
      "Condition": {
        "StringLike": { "s3:prefix": ["Phase_2/*"] }
      }
    },
    {
      "Effect": "Allow",
      "Action": "sagemaker:InvokeEndpoint",
      "Resource": "arn:aws:sagemaker:ap-northeast-2:666803869796:endpoint/say2-2team-soonet-endpoint"
    }
  ]
}
```
Policy name: `Phase2Access`

> 📝 **태그 나중에 추가하는 방법** (권한 생기면):  
> IAM → Roles → `say2-2team-sagemaker-role` → Tags 탭 → Add tags  
> Key: `project` / Value: `pre-iam-2-2-team`
    
---

# ✅ Step 7 — 모델 파일 S3 업로드 (완료)

## 🧠 왜 이 파일들이 Phase 2를 완성시키는가?

Phase 2의 역할은 **X-ray 이미지 → HPO 코드(질환 코드) 변환**이에요.

```
[현재 상태 — Step 7 전]
Lambda → SageMaker Endpoint 호출 → ❌ 모델 없음 → 작동 불가

[Step 7 완료 후]
Lambda → SageMaker Endpoint 호출 → model.tar.gz 안의 AI가 X-ray 분석
                                    → 14개 질환 확률 + HPO 코드 JSON 반환
                                    → Phase 4~5 스코어링으로 전달
```

**model.tar.gz 안에 들어간 파일 5개와 각각의 역할:**

| 파일 | 역할 | 없으면? |
|------|------|---------|
| `soo_net_2.py` | X-ray → 14개 질환 확률 계산하는 AI 코드 (신버전) | 모델 구조 자체가 없음 |
| `unet_lung_model.py` | 폐/심장 위치 찾는 보조 AI 코드 | soo_net_2.py가 import 실패 |
| `anatomy_soonet_mimic_best.pth` | SooNet이 실제 판단에 쓰는 학습된 숫자들 (43MB) | 빈 껍데기, 랜덤 결과 |
| `unet_lung_heart_best.pth` | UNet이 폐/심장 찾는 데 쓰는 학습된 숫자들 (58MB) | 마스크 생성 불가 |
| `code/inference.py` | SageMaker가 "이렇게 실행해라"고 알려주는 진입점 | SageMaker가 어떻게 실행할지 모름 |

---

## 📦 패키징 구성 (신버전 기준, 2026-05-08 최종)

```
model.tar.gz (146MB, 2026-05-08 완료)
├── soo_net_2.py                ← GitHub real_super_final/ (신버전 AnatomySooNet 코드)
├── unet_lung_model.py          ← GitHub real_super_final/ (신버전 UNet 코드)
├── latest_checkpoint.pth       ← S3 Phase_2/ (최신 SooNet 가중치, 129MB) ✅
├── unet_lung_heart_best.pth    ← S3 Phase_2/ (UNet 가중치, 58MB) ✅
└── code/
    └── inference.py            ← 허태웅 작성 (신버전 soo_net_2.py 기준)
```

**S3 Phase_2/ 폴더 현황 (2026-05-08 기준):**

| 파일 | 크기 | 용도 | 비고 |
|------|------|------|------|
| `models/soonet/model.tar.gz` | 146MB | SageMaker 배포 패키지 | ✅ 최신 |
| `latest_checkpoint.pth` | 129MB | 최신 SooNet 가중치 (원본) | ✅ 배기태 업로드 |
| `unet_lung_heart_best.pth` | 58MB | UNet 가중치 (원본) | ✅ |
| `soo_net.py` | 12KB | 구버전 코드 | 참고용 유지 |
| `chexnet_1ch_448_chexpert_best.pth` | 27MB | 구버전 가중치 | 참고용 유지 |

---

## ✅ 업로드 완료 확인 (2026-05-08)

```
S3 경로: s3://say2-2team-bucket/Phase_2/models/soonet/model.tar.gz
크기: 146MB
```

```bash
# 확인 커맨드
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID \
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY \
aws s3 ls s3://say2-2team-bucket/Phase_2/models/ --recursive --human-readable
```

---

# ✅ Step 8 — SageMaker Endpoint (완료 2026-05-08, InService 확인됨)

## 8-1. Model 생성

1. SageMaker 콘솔 → Inference → Models → Create model
2. Model name: `say2-2team-soonet-model`
3. IAM role: `say2-2team-sagemaker-role`
4. Container definition:
   - Container input: `Provide model artifacts and inference image location`
   - Inference image: `763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:2.0.0-cpu-py310-ubuntu20.04-sagemaker`
   - Model artifacts: `s3://say2-2team-bucket/Phase_2/models/soonet/model.tar.gz`
   - Environment:
     - `SAGEMAKER_PROGRAM` = `inference.py`
     - `SAGEMAKER_SUBMIT_DIRECTORY` = `/opt/ml/model/code`
5. VPC:
   - VPC: `say2-2team`
   - Subnets: `say2-2team-private-1`
   - Security groups: `say2-2team-sg-sagemaker`
6. Tags: `project = pre-sagemaker-2-2-team`

## 8-2. Endpoint Configuration

1. Inference → Endpoint configurations → Create
2. Name: `say2-2team-soonet-config`
3. Add model: `say2-2team-soonet-model`
4. Instance type: **`ml.m5.large`** (CPU, 시간당 $0.115) — GPU 필요 시 `ml.g4dn.xlarge` ($1.20)
5. Initial instance count: 1
6. Tag: `project = pre-sagemaker-2-2-team`

## 8-3. Endpoint

1. Inference → Endpoints → Create
2. Name: **`say2-2team-soonet-endpoint`** (Lambda IAM Policy와 일치 필수)
3. Endpoint configuration: `say2-2team-soonet-config`
4. Tag: `project = pre-sagemaker-2-2-team`
5. Create (5~10분, InService 대기)

⚠️ **InService 순간부터 시간당 과금 시작.**

---

# ⬜ Step 9 — Lambda 함수

1. Lambda 콘솔 → Create function → Author from scratch
2. Function name: `say2-2team-phase2-vision`
3. Runtime: Python 3.11
4. Execution role: Use existing → `say2-2team-lambda-role`
5. Advanced settings:
   - Enable VPC: ✅
   - VPC: `say2-2team`
   - Subnets: `say2-2team-private-1`
   - Security groups: `say2-2team-sg-lambda`
6. Tags: `project = pre-lambda-2-2-team`
7. Create

## 9-1. 코드 업로드

- 로컬 파일: `infra/lambda/phase2/phase2_handler.py` 내용을 Code 에디터에 붙여넣기
- Deploy 클릭

## 9-2. Handler 설정

- Configuration → General configuration → Edit
- Handler: `lambda_function.lambda_handler`

## 9-3. 환경변수

- Configuration → Environment variables → Edit → Add:
  - `SAGEMAKER_ENDPOINT` = `say2-2team-soonet-endpoint`
  - `S3_BUCKET` = `say2-2team-bucket`
  - `S3_PREFIX` = `Phase_2`
  - `XRAY_THRESHOLD` = `0.3`

## 9-4. Timeout/Memory

- Configuration → General configuration → Edit
- Timeout: 5 min
- Memory: 1024 MB

---

# ⬜ Step 10 — 테스트

## 10-1. 테스트 이미지 S3 업로드

MIMIC 테스트 이미지가 이미 준비됨:
```
s3://say2-2team-bucket/Phase_2/uploads/test_mimic.jpg
```

원본: `data/mimic-cxr-448/p10/p10000032/s53189527/` 환자 이미지

## 10-2. Lambda Test Event 실행

Lambda 콘솔 → `say2-2team-phase2-vision` → Test → Create new event:

```json
{
  "xray_s3_key": "Phase_2/uploads/test_mimic.jpg",
  "threshold": 0.3
}
```

Save → Test 실행 → statusCode 200 + HPO 배열 반환 확인

---

## 🔐 태그 규칙 준수 체크리스트

| 서비스 | 태그 키 | 태그 값 | 상태 |
|--------|---------|---------|------|
| VPC/Subnet/SG | `project` | `pre-vpc-2-2-team` | ✅ |
| IGW/EIP/NAT/RT (옵션 A) | `project` | `pre-vpc-2-2-team` | ⬜ |
| VPC Endpoint (옵션 B) | `project` | `pre-vpc-2-2-team` | ⬜ |
| S3 | `project` | `pre-s3-2-2-team` | ⬜ |
| IAM Role | `project` | `pre-iam-2-2-team` | ⬜ |
| SageMaker | `project` | `pre-sagemaker-2-2-team` | ⬜ |
| Lambda | `project` | `pre-lambda-2-2-team` | ⬜ |
| Cognito User Pool | `project` | `pre-cognito-2-2-team` | ⬜ |
| Secrets Manager | `project` | `pre-secretsmanager-2-2-team` | ⬜ |
| KMS | `project` | `pre-kms-2-2-team` | ⬜ |
| CloudTrail | `project` | `pre-cloudtrail-2-2-team` | ⬜ |
| GuardDuty | `project` | `pre-guardduty-2-2-team` | ⬜ |

---

## 💰 비용 예상 (Step 10 완료 시)

### 옵션 A (IGW + NAT)

| 리소스 | 시간당 | 1일 |
|--------|--------|-----|
| NAT Gateway + EIP | $0.050 | $1.20 |
| SageMaker ml.m5.large | $0.115 | $2.76 |
| **합계 (인프라)** | **$0.165** | **$3.96** |

### 옵션 B (VPC Endpoint)

| 리소스 | 시간당 | 1일 |
|--------|--------|-----|
| VPC Endpoint (Interface) × 2 | $0.028 | $0.67 |
| SageMaker ml.m5.large | $0.115 | $2.76 |
| **합계 (인프라)** | **$0.143** | **$3.43** |

### 보안 서비스 추가 비용 (Step 11~22)

| 서비스 | 비용 | 비고 |
|--------|------|------|
| Cognito User Pool | 무료 (MAU 50,000명 이하) | 발표용 테스트 계정 수준 |
| Secrets Manager | $0.40/secret/월 | 1개 secret → $0.40 |
| KMS | $1.00/키/월 + $0.03/10,000 API 호출 | 키 1개 → $1.00 |
| CloudTrail | 무료 (관리 이벤트 1 trail) | 데이터 이벤트는 $0.10/100,000건 |
| GuardDuty | ~$1~3/월 (소규모 계정) | 30일 무료 체험 |
| Security Hub | 무료 (30일 체험) | 이후 $0.0010/finding |
| Detective | ~$1~2/월 | 30일 무료 체험 |
| Inspector | Lambda 스캔 무료 | ECR 스캔 $0.09/이미지 |
| API Gateway | $3.50/백만 요청 | 발표용 소량 → 무시 가능 |
| DynamoDB | 온디맨드, 소량 | ~$1/월 |
| SNS | 무료 (이메일 1,000건/월) | 발표용 수준 |
| AWS Budgets | 무료 (2개 이하) | 무료 |
| **보안+데이터 합계** | **~$5~10/월** | **발표 기간 중 무시 가능 수준** |

---

## 🗑️ 시연 후 삭제 순서

```
# ── 보안 리소스 먼저 비활성화 ──
0-a. GuardDuty 비활성화 (콘솔: GuardDuty → Settings → Suspend/Disable)
0-b. Security Hub 비활성화 (콘솔: Security Hub → Settings → Disable)
0-c. Detective 비활성화 (콘솔: Detective → Settings → Disable)
0-d. Inspector 비활성화 (콘솔: Inspector → Settings → Disable)
0-e. CloudTrail 삭제 (콘솔: CloudTrail → Trails → say2-2team-audit-trail → Delete)
0-f. Secrets Manager 삭제 (콘솔: Secrets Manager → say2-2team/phase2-config → Delete)
     ※ "Waiting period" → 7일 대기 없이 즉시 삭제하려면 "Force delete" 체크
0-g. KMS 키 비활성화 (콘솔: KMS → say2-2team-data-key → Key actions → Disable)
     ※ KMS 키 삭제는 최소 7일 대기 필수 — 비활성화로 대체
0-h. Cognito User Pool 삭제 (콘솔: Cognito → say2-2team-user-pool → Delete)
0-i. SNS Topic 삭제 (콘솔: SNS → say2-2team-alerts → Delete)
0-j. CloudWatch 알람 3개 삭제 (콘솔: CloudWatch → Alarms → 각 알람 선택 → Delete)

# ── 데이터 리소스 ──
0-k. DynamoDB 테이블 삭제 (콘솔: DynamoDB → Tables → 각 테이블 → Delete)
     ※ PITR 활성화된 테이블은 삭제 시 자동으로 PITR도 해제됨

# ── 컴퓨트/네트워크 리소스 ──
1. API Gateway 삭제 (콘솔: API Gateway → say2-2team-diagnose-api → Delete)
2. Lambda 삭제
3. SageMaker Endpoint → Endpoint Config → Model
4. VPC Endpoint 삭제 (옵션 B 쓴 경우)
5. NAT Gateway 삭제 → EIP 해제 → IGW 분리/삭제 (옵션 A 쓴 경우)
6. Route Table 삭제
7. S3 Phase_2/ 폴더 내용 삭제 (버킷 자체는 팀 공용이므로 삭제 금지)
   aws s3 rm s3://say2-2team-bucket/Phase_2/ --recursive
8. IAM Role 2개 삭제
9. Security Group 3개 삭제
10. Subnet 4개 삭제
11. VPC는 팀 공용이므로 절대 건드리지 말 것
```

---

# 🔐 보안 구성 (Step 11~16)

> **아키텍처 제안서 §17~25 기반** — 콘솔 직접 구축 절차  
> 아래 순서대로 진행하면 오류 없이 완료됩니다.  
> 각 서비스는 **독립적**이므로 Step 8~10 완료 전에도 미리 구성 가능합니다.

---

## ✅ Step 11 — Cognito User Pool (API 인증)

> **목적**: API Gateway 앞단에 JWT 기반 인증 추가. 의사 계정만 진단 API 호출 가능.  
> **태그**: `project = pre-cognito-2-2-team`  
> **최종 업데이트**: 2026-05-21  
> **UI 버전**: 2024년 이후 신규 Cognito 콘솔 기준

---

### 11-0. 전체 흐름 한눈에 보기

```
① AWS 콘솔에서 Cognito User Pool 생성 (약 10분)
   └─ User Pool ID, App Client ID 두 값 메모

② 로컬 파일 수정 (약 2분)
   └─ frontend/src/aws-config.js 에 두 값 붙여넣기

③ 빌드 + S3 업로드 + CloudFront 캐시 무효화 (약 3분)

④ 브라우저에서 확인
   └─ https://d300v14l8u0wx7.cloudfront.net?demo=1  → 로그인 없이 바로 진입 (발표용)
   └─ https://d300v14l8u0wx7.cloudfront.net         → Cognito 로그인 화면
```

---

### 11-1. AWS 콘솔 접속

1. 브라우저에서 `https://ap-northeast-2.console.aws.amazon.com/cognito/v2/idp/user-pools` 접속
2. 우측 상단 리전이 **아시아 태평양 (서울) ap-northeast-2** 인지 확인
3. 오른쪽 상단 **"Create user pool"** 버튼 클릭

---

### 11-2. 화면 1 — "Define your application" (앱 정의)

> 이 화면에서 앱 타입과 이름, 기본 옵션을 한 번에 설정합니다.

#### Application type (앱 타입 선택)

화면에 4가지 선택지가 나옵니다:
- Traditional web application
- **Single-page application (SPA)** ← ✅ 이것 선택
- Mobile app
- Machine-to-machine application

> 이유: 우리 프론트엔드가 React (Vite) SPA이기 때문

#### Name your application (앱 이름)

- 입력란에 `say2-2team-api-client` 입력

#### Options for sign-in identifiers (로그인 방식)

3가지 체크박스가 있습니다:
- **Email** ← ✅ 체크
- Phone number ← ❌ 체크 해제 (없으면 그냥 패스)
- Username ← ❌ 체크 해제 (없으면 그냥 패스)

#### Want to set up social, SAML, or OIDC sign-in?

- 건드리지 말고 그냥 패스 (소셜 로그인 불필요)

#### Self-registration (자가 회원가입)

- **"Enable self-registration" 체크 해제** ← ⚠️ 반드시 끄기
- 이유: 아무나 가입하면 안 됨. 관리자(우리)만 계정 생성

#### Required attributes for sign-up (필수 속성)

- 드롭다운 "Select attributes" 클릭 → **email** 선택
- 선택 후 email 태그가 생기면 OK

#### 완료

- 화면 하단 **"Next"** 버튼 클릭

---

### 11-3. 화면 2 — "Configure security" (보안 설정)

> 비밀번호 정책, MFA, 계정 복구 방식을 설정합니다.

#### Password policy (비밀번호 정책)

- **"Cognito defaults"** 선택 (기본값 유지)
  - 최소 8자, 대문자+소문자+숫자+특수문자 포함

#### Multi-factor authentication (MFA)

- **"No MFA"** 선택 ← 발표용이므로 MFA 없이
  - (상용화 시에는 "Authenticator apps" 권장)

#### User account recovery

- **"Enable self-service account recovery"** 체크 유지
- Recovery message delivery: **"Email only"** 선택

#### 완료

- 화면 하단 **"Next"** 버튼 클릭

---

### 11-4. 화면 3 — "Configure sign-up" (가입 설정)

> 이미 화면 1에서 self-registration을 껐으므로 여기서는 확인만 합니다.

#### Self-registration

- **비활성화 상태** 인지 확인 (체크 해제 상태여야 함)

#### Attribute verification and user account confirmation

- "Send a message to verify" → **Email** 선택 유지

#### Required attributes

- `email` 이 목록에 있는지 확인

#### 완료

- 화면 하단 **"Next"** 버튼 클릭

---

### 11-5. 화면 4 — "Configure message delivery" (이메일 발송 설정)

> Cognito가 인증 이메일을 보낼 때 어떤 방식을 쓸지 설정합니다.

#### Email provider

- **"Send email with Cognito"** 선택 ← 무료, 발표용으로 충분
  - (상용화 시에는 SES 연동 권장)

#### FROM email address

- 기본값 `no-reply@verificationemail.com` 유지

#### 완료

- 화면 하단 **"Next"** 버튼 클릭

---

### 11-6. 화면 5 — "Review and create" (최종 확인 및 생성)

> 지금까지 설정한 내용을 한 번에 확인합니다.

확인할 항목:
- Application type: `Single-page application`
- App client name: `say2-2team-api-client`
- Sign-in options: `Email`
- Self-registration: `Disabled`
- MFA: `No MFA`

이상 없으면 하단 **"Create user pool"** 버튼 클릭

> ⏳ 생성까지 약 5~10초 소요됩니다.

---

### 11-7. 생성 완료 후 — User Pool ID / App Client ID 메모

생성이 완료되면 User Pool 상세 페이지로 이동합니다.

#### User Pool ID 확인

1. 페이지 상단 또는 "User pool overview" 섹션에서 확인
2. 형식: `ap-northeast-2_XXXXXXXXX` (알파벳+숫자 9자리)
3. 이 값을 복사해서 메모장에 저장

#### App Client ID 확인

1. 좌측 메뉴 또는 탭에서 **"App integration"** 클릭
2. 하단 **"App clients"** 섹션에서 `say2-2team-api-client` 클릭
3. **"Client ID"** 값 복사 (형식: 영숫자 26자리)
4. 메모장에 저장

> ⚠️ **이 두 값이 없으면 프론트엔드 연동이 불가능합니다. 반드시 메모하세요.**

#### resource_ids.md에 기입

`infra/resource_ids.md` 파일의 Cognito 섹션을 아래처럼 채우세요:

```
## 🔒 Cognito

| 항목 | 값 |
|------|-----|
| User Pool ID | ap-northeast-2_XXXXXXXXX  ← 실제 값으로 교체 |
| App Client ID | xxxxxxxxxxxxxxxxxxxxxxxxxx ← 실제 값으로 교체 |
| 테스트 계정 | doctor-test@say2team.com |
```

---

### 11-8. 태그 추가

> Cognito 생성 화면에는 태그 입력란이 없어서 생성 후 별도로 추가해야 합니다.

1. User Pool 상세 페이지에서 좌측 메뉴 **"User pool properties"** 클릭
2. **"Tags"** 섹션 찾기
3. **"Add tag"** 클릭
4. Key: `project` / Value: `pre-cognito-2-2-team` 입력
5. **"Save changes"** 클릭

---

### 11-9. 테스트 사용자 생성 (발표용 계정)

> 발표 때 실제로 로그인할 의사 계정을 만듭니다.

1. User Pool 상세 페이지 → 상단 탭 **"Users"** 클릭
2. 오른쪽 **"Create user"** 버튼 클릭
3. 아래처럼 입력:

| 항목 | 값 |
|------|-----|
| Invitation message | **"Don't send an invitation"** 선택 |
| Email address | `doctor-test@say2team.com` |
| Mark email address as verified | ✅ 체크 |
| Temporary password | `Say2Team2026!` |

4. **"Create user"** 클릭

> ⚠️ **임시 비밀번호 주의사항**  
> Cognito는 임시 비밀번호로 첫 로그인 시 새 비밀번호 변경을 강제합니다.  
> 발표 전에 반드시 아래 방법으로 비밀번호를 확정 상태로 바꿔두세요.

#### 임시 비밀번호 → 확정 비밀번호로 변경 (CLI)

```bash
# 아래 명령어로 비밀번호를 영구 확정 상태로 변경
# --user-pool-id 와 --username 은 실제 값으로 교체
aws cognito-idp admin-set-user-password   --user-pool-id ap-northeast-2_XXXXXXXXX   --username doctor-test@say2team.com   --password "RareLink2026!"   --permanent   --region ap-northeast-2
```

> 이 명령어 실행 후 Users 탭에서 해당 유저의 상태가  
> `FORCE_CHANGE_PASSWORD` → `CONFIRMED` 로 바뀌면 성공

---

### 11-10. 프론트엔드 연동 — aws-config.js 수정

`aws_say2_project_vision/frontend/src/aws-config.js` 파일을 열어서  
11-7에서 메모한 두 값을 입력합니다:

```js
// ⚠️ 이 파일은 공개 값만 포함 (Client Secret 없음)
export const COGNITO_CONFIG = {
  region: 'ap-northeast-2',
  userPoolId: 'ap-northeast-2_XXXXXXXXX',   // ← 11-7에서 메모한 User Pool ID로 교체
  clientId: 'xxxxxxxxxxxxxxxxxxxxxxxxxx',    // ← 11-7에서 메모한 App Client ID로 교체
};
```

> 파일 위치: `aws_say2_project_vision/frontend/src/aws-config.js`  
> 이미 파일이 생성되어 있으므로 두 값만 교체하면 됩니다.

---

### 11-11. 빌드 → S3 업로드 → CloudFront 캐시 무효화

터미널에서 아래 명령어를 순서대로 실행합니다.

```bash
# 1. 프론트엔드 폴더로 이동
cd aws_say2_project_vision/frontend

# 2. 빌드 (dist/ 폴더 생성됨)
npm run build

# 3. S3 업로드 — index.html (캐시 없음)
aws s3 sync dist/ s3://say2-2team-bucket/frontend/   --exclude "assets/*"   --cache-control "no-cache, no-store, must-revalidate"   --region ap-northeast-2

# 4. S3 업로드 — JS/CSS assets (1년 캐시)
aws s3 sync dist/assets/ s3://say2-2team-bucket/frontend/assets/   --cache-control "max-age=31536000,immutable"   --region ap-northeast-2

# 5. CloudFront 캐시 무효화 (반영까지 1~2분 소요)
aws cloudfront create-invalidation   --distribution-id E2ZHONIV05TX9D   --paths "/*"   --region us-east-1
```

---

### 11-12. 동작 확인

브라우저에서 아래 두 URL을 각각 열어서 확인합니다:

| URL | 기대 동작 |
|-----|---------|
| `https://d300v14l8u0wx7.cloudfront.net?demo=1` | 로그인 없이 워크리스트 바로 진입 ✅ |
| `https://d300v14l8u0wx7.cloudfront.net` | Cognito 로그인 화면 표시 ✅ |

로그인 화면에서 `doctor-test@say2team.com` / `RareLink2026!` 으로 로그인 테스트

---

### 11-13. 완료 체크리스트

| 항목 | 확인 |
|------|------|
| User Pool 생성됨 (`say2-2team-user-pool`) | ⬜ |
| App Client 생성됨 (`say2-2team-api-client`, client secret 없음) | ⬜ |
| 태그 추가됨 (`project = pre-cognito-2-2-team`) | ⬜ |
| 테스트 유저 생성됨 (`doctor-test@say2team.com`) | ⬜ |
| 유저 상태 `CONFIRMED` 확인 (임시 비밀번호 변경 완료) | ⬜ |
| `resource_ids.md` Cognito 섹션 기입 완료 | ⬜ |
| `aws-config.js` User Pool ID / Client ID 실제 값으로 교체 | ⬜ |
| `npm run build` 성공 | ⬜ |
| S3 업로드 완료 | ⬜ |
| CloudFront 캐시 무효화 완료 | ⬜ |
| `?demo=1` 접근 시 워크리스트 바로 진입 확인 | ⬜ |
| 일반 접근 시 Cognito 로그인 화면 확인 | ⬜ |
| `doctor-test@say2team.com` 로그인 성공 확인 | ⬜ |

### 11-14. API Gateway에 Cognito Authorizer 연결

> Step 17 (API Gateway) 구성 후 진행. 지금은 User Pool만 생성해두면 됩니다.


## ⬜ Step 12 — Secrets Manager (API 키 안전 보관)

> **목적**: Lambda 환경변수에 하드코딩된 키를 Secrets Manager로 이전.  
> **태그**: `project = pre-secretsmanager-2-2-team`

### 12-1. SageMaker Endpoint 이름 Secret 생성

> 현재 Lambda 환경변수에 `SAGEMAKER_ENDPOINT = say2-2team-soonet-endpoint`가 있음.  
> 이를 Secrets Manager로 이전하면 코드 변경 없이 엔드포인트 교체 가능.

1. AWS 콘솔 → **Secrets Manager** → "Store a new secret"
2. Secret type: `Other type of secret`
3. Key/value pairs:
   - Key: `sagemaker_endpoint` / Value: `say2-2team-soonet-endpoint`
   - Key: `s3_bucket` / Value: `say2-2team-bucket`
   - Key: `s3_prefix` / Value: `Phase_2`
   - Key: `xray_threshold` / Value: `0.3`
4. Next
5. Secret name: `say2-2team/phase2-config`
6. Description: `Phase 2 Lambda 설정값 (SageMaker endpoint, S3 경로)`
7. Tags → Add tag:
   - Key: `project` / Value: `pre-secretsmanager-2-2-team`
8. Next → Next → "Store"

### 12-2. Lambda Role에 Secrets Manager 권한 추가

1. IAM → Roles → `say2-2team-lambda-role` → "Add permissions" → "Create inline policy"
2. JSON 탭:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "arn:aws:secretsmanager:ap-northeast-2:666803869796:secret:say2-2team/*"
    }
  ]
}
```

3. Policy name: `SecretsManagerAccess` → Save

### 12-3. Lambda 코드에서 Secret 읽기 (참고용)

```python
import boto3, json

def get_config():
    client = boto3.client('secretsmanager', region_name='ap-northeast-2')
    secret = client.get_secret_value(SecretId='say2-2team/phase2-config')
    return json.loads(secret['SecretString'])

# 사용 예시
config = get_config()
endpoint = config['sagemaker_endpoint']  # 'say2-2team-soonet-endpoint'
```

> 📝 **발표용 간소화**: 시간이 부족하면 Lambda 환경변수 방식 유지해도 됩니다.  
> Secrets Manager는 "보안 강화 구현 완료" 포트폴리오 포인트로 활용.

---

## ⬜ Step 13 — KMS (데이터 암호화 키 관리)

> **목적**: S3 X-ray 이미지 + DynamoDB 진단 이력 암호화.  
> **태그**: `project = pre-kms-2-2-team`

### 13-1. KMS 키 생성

1. AWS 콘솔 → **KMS** → "Create key"
2. Key type: `Symmetric`
3. Key usage: `Encrypt and decrypt`
4. Next
5. Alias: `say2-2team-data-key`
6. Description: `say2-2team Phase 2 데이터 암호화 키 (S3, DynamoDB)`
7. Tags → Add tag:
   - Key: `project` / Value: `pre-kms-2-2-team`
8. Next
9. Key administrators: 본인 IAM 사용자 체크
10. Next
11. Key users: `say2-2team-lambda-role`, `say2-2team-sagemaker-role` 체크
12. Next → "Finish"

> 생성 후 Key ID (예: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`) 메모.

### 13-2. S3 버킷 암호화 설정

1. S3 → `say2-2team-bucket` → "Properties" 탭
2. "Default encryption" → "Edit"
3. Encryption type: `AWS Key Management Service key (SSE-KMS)`
4. AWS KMS key: `say2-2team-data-key` 선택
5. Bucket Key: `Enable` (비용 절감 — KMS 호출 횟수 최대 99% 감소)
6. Save changes

> ⚠️ **기존 파일은 소급 적용 안 됨** — 이후 업로드되는 파일부터 암호화 적용.  
> 기존 `Phase_2/models/soonet/model.tar.gz`는 재업로드 시 암호화됨.

### 13-3. Lambda Role에 KMS 권한 추가

1. IAM → Roles → `say2-2team-lambda-role` → "Add permissions" → "Create inline policy"
2. JSON:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt",
        "kms:GenerateDataKey"
      ],
      "Resource": "arn:aws:kms:ap-northeast-2:666803869796:key/*",
      "Condition": {
        "StringLike": {
          "kms:RequestAlias": "alias/say2-2team-data-key"
        }
      }
    }
  ]
}
```

3. Policy name: `KMSAccess` → Save

---

## ⬜ Step 14 — CloudTrail (감사 로그)

> **목적**: 누가 언제 어떤 리소스에 접근했는지 추적. HIPAA 요건 충족.  
> **태그**: `project = pre-cloudtrail-2-2-team`

### 14-1. Trail 생성

1. AWS 콘솔 → **CloudTrail** → "Create trail"
2. Trail name: `say2-2team-audit-trail`
3. Storage location:
   - Create new S3 bucket: ✅
   - Trail log bucket name: `say2-2team-cloudtrail-logs` (자동 생성)
4. Log file SSE-KMS encryption: ✅ Enable
   - New KMS alias: `say2-2team-cloudtrail-key`
5. CloudWatch Logs: ✅ Enable
   - New log group: `say2-2team-cloudtrail-logs`
   - New IAM role: `say2-2team-cloudtrail-role`
6. Tags → Add tag:
   - Key: `project` / Value: `pre-cloudtrail-2-2-team`
7. Next

### 14-2. 이벤트 유형 선택

8. Event type:
   - ✅ Management events (API 호출 기록 — 필수)
   - ✅ Data events (S3 객체 접근 기록 — 환자 X-ray 접근 추적)
9. Data events 설정:
   - Data event source: `S3`
   - "Add S3 bucket" → `say2-2team-bucket` 입력
   - Read: ✅ / Write: ✅
10. Next → "Create trail"

> ✅ 생성 완료 후 CloudTrail → Event history에서 최근 API 호출 확인 가능.

### 14-3. CloudWatch 알람 설정 (선택 — 보안 이벤트 알림)

1. CloudWatch → Alarms → "Create alarm"
2. Select metric → CloudTrail → "By Event Name" → `DeleteBucket` 검색
3. Conditions: `>= 1` (S3 버킷 삭제 시도 즉시 알림)
4. Notification: SNS topic 생성 → 이메일 입력
5. Alarm name: `say2-2team-s3-delete-alert`

---

## ⬜ Step 15 — GuardDuty + Security Hub (위협 탐지)

> **목적**: 비정상 API 호출, 자격증명 탈취 시도 자동 탐지.  
> **비용**: GuardDuty 약 $1~3/월 (소규모 계정 기준), Security Hub 무료 티어 30일

### 15-1. GuardDuty 활성화

1. AWS 콘솔 → **GuardDuty** → "Get Started"
2. "Enable GuardDuty" 클릭 (30일 무료 체험)
3. 활성화 완료 → Findings 탭에서 탐지 결과 확인

> ⚠️ **태그 추가**: GuardDuty는 서비스 자체에 태그 불가.  
> 대신 GuardDuty가 생성하는 S3 버킷(findings export용)에 태그 적용.

### 15-2. GuardDuty Findings S3 내보내기 설정 (선택)

1. GuardDuty → Settings → "Findings export options"
2. Frequency: `15 minutes`
3. S3 bucket: `say2-2team-bucket`
4. Key prefix: `guardduty-findings/`
5. KMS key: `say2-2team-data-key`
6. Save

### 15-3. Security Hub 활성화

1. AWS 콘솔 → **Security Hub** → "Go to Security Hub"
2. "Enable Security Hub" 클릭
3. Security standards 선택:
   - ✅ `AWS Foundational Security Best Practices v1.0.0`
   - ✅ `CIS AWS Foundations Benchmark v1.4.0`
   - (HIPAA 표준은 유료 — 포트폴리오용으로 체크 가능)
4. "Enable Security Hub"

> 활성화 후 약 1~2시간 후 Security score 및 Findings 표시됨.

### 15-4. GuardDuty → Security Hub 연동 확인

1. Security Hub → Integrations → "GuardDuty" 검색
2. "Accept findings" 클릭 → GuardDuty 탐지 결과가 Security Hub에 통합됨

---

## ⬜ Step 16 — S3 버킷 보안 강화

> **목적**: 퍼블릭 접근 차단 + 버전 관리 + 수명 주기 정책.  
> 기존 `say2-2team-bucket`에 추가 설정.

### 16-1. 퍼블릭 접근 차단 확인

1. S3 → `say2-2team-bucket` → "Permissions" 탭
2. "Block public access" → "Edit"
3. 4개 항목 모두 ✅ 체크 확인 (이미 되어 있으면 그대로)
4. Save

### 16-2. 버전 관리 활성화 (모델 가중치 보호)

1. S3 → `say2-2team-bucket` → "Properties" 탭
2. "Bucket Versioning" → "Edit"
3. `Enable` 선택 → Save

> ✅ 이후 `model.tar.gz` 덮어쓰기 시 이전 버전 복구 가능.

### 16-3. 수명 주기 정책 (스토리지 비용 절감)

1. S3 → `say2-2team-bucket` → "Management" 탭
2. "Create lifecycle rule"
3. Rule name: `say2-2team-model-lifecycle`
4. Rule scope: `Limit the scope` → Prefix: `Phase_2/models/`
5. Lifecycle rule actions:
   - ✅ `Transition current versions of objects between storage classes`
     - Days after creation: `30`
     - Storage class: `Standard-IA`
   - ✅ `Transition noncurrent versions of objects between storage classes`
     - Days after objects become noncurrent: `7`
     - Storage class: `Glacier Instant Retrieval`
6. Create rule

---

## 🔐 보안 구성 완료 체크리스트

| Step | 서비스 | 핵심 기능 | 상태 |
|------|--------|----------|------|
| 11 | Cognito User Pool | API 인증 (JWT) | ⬜ |
| 12 | Secrets Manager | API 키 안전 보관 | ⬜ |
| 13 | KMS | S3/DynamoDB 암호화 | ✅ 완료 |
| 14 | CloudTrail | 감사 로그 (HIPAA) | ✅ 완료 |
| 15 | GuardDuty + Security Hub | 위협 탐지 | ✅ 완료 |
| 16 | S3 보안 강화 | 퍼블릭 차단 + 버전관리 | ✅ 완료 |
| **17** | **API Gateway + Cognito Authorizer** | **JWT 인증 연결** | **⬜** |
| **18** | **DynamoDB 테이블 2개** | **진단 이력 + 희귀 케이스 DB** | **⬜** |
| **19** | **DynamoDB PITR** | **35일 이내 복구** | **⬜** |
| **20** | **SNS 알림** | **CloudWatch 알람 → 이메일** | **⬜** |
| **21** | **AWS Budgets** | **비용 관리** | **✅ CloudWatch 알람으로 대체** |
| **22** | **Detective + Inspector** | **침해 조사 + 취약점 스캔** | **⬜** |

---

## ⬜ Step 17 — API Gateway + Cognito Authorizer 연결

> **목적**: Step 11에서 만든 Cognito User Pool을 API Gateway에 연결해 JWT 인증 적용.  
> **태그**: `project = pre-apigateway-2-2-team`

### 17-1. API Gateway 생성 (없는 경우)

1. AWS 콘솔 → **API Gateway** → "Create API"
2. API type: `REST API` → "Build"
3. Protocol: `REST`
4. Create new API: `New API`
5. API name: `say2-2team-diagnose-api`
6. Endpoint Type: `Regional`
7. Create API

### 17-2. Cognito Authorizer 추가

1. API Gateway → `say2-2team-diagnose-api` → 좌측 "Authorizers" → "Create New Authorizer"
2. Name: `say2-2team-cognito-auth`
3. Type: `Cognito`
4. Cognito User Pool: `say2-2team-user-pool` 선택
5. Token Source: `Authorization` (헤더 이름)
6. Token Validation: 비워두기 (기본값)
7. Create

### 17-3. 리소스 + 메서드에 Authorizer 적용

1. API Gateway → Resources → "Create Resource"
   - Resource Name: `diagnose`
   - Resource Path: `/diagnose`
2. `/diagnose` 선택 → "Create Method" → `POST`
3. Integration type: `Lambda Function`
4. Lambda Function: `say2-2team-phase2-vision`
5. Save → OK (Lambda 권한 자동 추가)
6. POST 메서드 선택 → "Method Request" → "Authorization" → `say2-2team-cognito-auth` 선택 → ✅ 체크

### 17-4. API 배포

1. Actions → "Deploy API"
2. Deployment stage: `[New Stage]`
3. Stage name: `prod`
4. Deploy

> 배포 후 Invoke URL 메모: `https://xxxxxxxxxx.execute-api.ap-northeast-2.amazonaws.com/prod`

### 17-5. API Gateway 태그 추가

1. API Gateway → `say2-2team-diagnose-api` → "Tags" 탭
2. Add tag: Key `project` / Value `pre-apigateway-2-2-team`

### 17-6. 테스트 (JWT 토큰 포함 호출)

```bash
# 1. Cognito에서 JWT 토큰 발급
aws cognito-idp initiate-auth \
  --auth-flow USER_PASSWORD_AUTH \
  --client-id <app-client-id> \
  --auth-parameters USERNAME=doctor-test@say2team.com,PASSWORD=Say2Team2026! \
  --region ap-northeast-2

# → AuthenticationResult.IdToken 값 복사

# 2. API 호출 (Authorization 헤더에 토큰 포함)
curl -X POST \
  https://xxxxxxxxxx.execute-api.ap-northeast-2.amazonaws.com/prod/diagnose \
  -H "Authorization: <IdToken>" \
  -H "Content-Type: application/json" \
  -d '{"xray_s3_key": "Phase_2/uploads/test_mimic.jpg", "threshold": 0.3}'
```

---

## ⬜ Step 18 — DynamoDB 테이블 2개 생성

> **목적**: 진단 이력 저장 + 희귀 케이스 축적 (MLOps 재학습 트리거 소스).  
> **태그**: `project = pre-dynamodb-2-2-team`

### 18-1. diagnosis-history 테이블 (사용자 데이터 DB)

1. AWS 콘솔 → **DynamoDB** → "Create table"
2. Table name: `say2-2team-diagnosis-history`
3. Partition key: `patient_mrn` (String)
4. Sort key: `diagnosis_timestamp` (String)
5. Table settings: `Customize settings`
6. Table class: `DynamoDB Standard`
7. Read/write capacity: `On-demand` (트래픽 예측 불가 → 온디맨드)
8. Encryption at rest: `AWS owned key` (기본) 또는 `AWS managed key` (KMS 연동 시)
9. Tags → Add tag: Key `project` / Value `pre-dynamodb-2-2-team`
10. Create table

### 18-2. rare-case-collection 테이블 (Case Report DB)

위와 동일 절차, 아래만 변경:

| 항목 | 값 |
|------|-----|
| Table name | `say2-2team-rare-case-collection` |
| Partition key | `disease_orpha_id` (String) |
| Sort key | `case_id` (String) |
| Tag | `project = pre-dynamodb-2-2-team` |

### 18-3. DynamoDB Streams 활성화 (MLOps 트리거용)

1. `say2-2team-rare-case-collection` → "Exports and streams" 탭
2. "DynamoDB stream details" → "Enable"
3. View type: `New and old images`
4. Enable stream

> ✅ 이후 EventBridge가 이 Stream을 감지해서 100건 도달 시 SageMaker Training Job 트리거.

---

## ⬜ Step 19 — DynamoDB PITR (Point-in-Time Recovery)

> **목적**: 35일 이내 임의 시점으로 데이터 복구. HIPAA 데이터 보호 필수.

### 19-1. diagnosis-history PITR 활성화

1. DynamoDB → `say2-2team-diagnosis-history` → "Backups" 탭
2. "Point-in-time recovery (PITR)" → "Edit"
3. `Enable PITR` 체크 → Save

### 19-2. rare-case-collection PITR 활성화

위와 동일 절차, 테이블만 `say2-2team-rare-case-collection`으로.

> ✅ 활성화 후 "Earliest restore date"가 표시되면 완료.  
> 복구 방법: DynamoDB → 테이블 → Backups → "Restore to point in time" → 원하는 시각 입력

---

## ⬜ Step 20 — SNS 알림 (CloudWatch 알람 연동)

> **목적**: Lambda 에러, SageMaker 지연, 모델 재학습 완료 시 이메일 자동 발송.  
> **태그**: `project = pre-sns-2-2-team`

### 20-1. SNS Topic 생성

1. AWS 콘솔 → **SNS** → "Topics" → "Create topic"
2. Type: `Standard`
3. Name: `say2-2team-alerts`
4. Tags → Add tag: Key `project` / Value `pre-sns-2-2-team`
5. Create topic

### 20-2. 이메일 구독 추가

1. 생성된 topic → "Create subscription"
2. Protocol: `Email`
3. Endpoint: 팀 이메일 주소 입력
4. Create subscription
5. 이메일 수신함에서 "Confirm subscription" 링크 클릭 ← **반드시 확인 필요**

### 20-3. CloudWatch 알람 3개 생성

#### 알람 1: Lambda 에러율

1. CloudWatch → Alarms → "Create alarm"
2. Select metric → Lambda → By Function Name → `say2-2team-phase2-vision` → `Errors`
3. Statistic: `Sum` / Period: `5 minutes`
4. Conditions: `>= 3` (5분 내 에러 3회 이상)
5. Notification: `say2-2team-alerts` SNS topic 선택
6. Alarm name: `say2-2team-lambda-error-alarm`
7. Create alarm

#### 알람 2: SageMaker Endpoint 지연

1. CloudWatch → Alarms → "Create alarm"
2. Select metric → SageMaker → Endpoint Metrics → `say2-2team-soonet-endpoint` → `ModelLatency`
3. Statistic: `Average` / Period: `5 minutes`
4. Conditions: `>= 30000` (30초 = 30,000ms 이상)
5. Notification: `say2-2team-alerts`
6. Alarm name: `say2-2team-sagemaker-latency-alarm`
7. Create alarm

#### 알람 3: S3 버킷 삭제 시도 (보안)

1. CloudWatch → Alarms → "Create alarm"
2. Select metric → CloudTrail → By Event Name → `DeleteBucket`
3. Statistic: `Sum` / Period: `5 minutes`
4. Conditions: `>= 1`
5. Notification: `say2-2team-alerts`
6. Alarm name: `say2-2team-s3-delete-alarm`
7. Create alarm

---

## ⬜ Step 21 — AWS Budgets (과금 방지 — 즉시 설정 필수)

> **목적**: SageMaker Endpoint 켜놓으면 시간당 $0.736 → 발표 후 삭제 안 하면 과금 폭탄.  
> ⚠️ **이 Step은 다른 모든 Step보다 먼저 설정하는 것을 강력 권장합니다.**

### 21-1. 월 예산 알림 설정

1. AWS 콘솔 → **Billing** → 좌측 "Budgets" → "Create budget"
2. Budget type: `Cost budget`
3. Budget name: `say2-2team-monthly-budget`
4. Period: `Monthly`
5. Budget amount: `$100`
6. Next

### 21-2. 알림 임계값 설정

7. Alert threshold: `80%` (= $80 도달 시)
8. Trigger: `Actual`
9. Email recipients: 팀 이메일 주소
10. Next → Create budget

### 21-3. SageMaker 전용 예산 (선택 — 권장)

위와 동일 절차, 아래만 변경:

| 항목 | 값 |
|------|-----|
| Budget name | `say2-2team-sagemaker-budget` |
| Budget amount | `$20` |
| Filter | Service = `Amazon SageMaker` |
| Alert threshold | `50%` (= $10 도달 시 경고) |

> ✅ 설정 완료 후 Billing → Budgets에서 현재 사용량 실시간 확인 가능.

---

## ⬜ Step 22 — Detective + Inspector (심층 보안)

> **목적**: GuardDuty 탐지 후 원인 조사(Detective) + Lambda/ECR 취약점 자동 스캔(Inspector).  
> **비용**: Detective ~$1~2/월, Inspector Lambda 스캔 무료 (ECR 스캔 $0.09/이미지)

### 22-1. Amazon Detective 활성화

> ⚠️ **GuardDuty가 먼저 활성화되어 있어야 합니다** (Step 15 완료 후 진행).

1. AWS 콘솔 → **Detective** → "Get started"
2. "Enable Amazon Detective" 클릭 (30일 무료 체험)
3. 활성화 완료 → Summary 탭에서 계정 행동 그래프 확인

> Detective는 GuardDuty 탐지 결과를 자동으로 가져와서 "어떤 IAM 역할이 → 어떤 API를 → 어떤 리소스에 호출했는지" 시각화합니다.

### 22-2. Amazon Inspector 활성화

1. AWS 콘솔 → **Inspector** → "Get started"
2. "Enable Inspector" 클릭
3. Scan types 선택:
   - ✅ `AWS Lambda standard scanning` (Lambda 함수 Python 패키지 CVE 스캔)
   - ✅ `Amazon ECR container image scanning` (SageMaker 학습용 Docker 이미지 스캔)
   - (EC2 scanning은 EC2 없으므로 체크 해제)
4. Enable

### 22-3. Inspector → Security Hub 연동 확인

1. Security Hub → Integrations → "Amazon Inspector" 검색
2. "Accept findings" 클릭 → Inspector 취약점 결과가 Security Hub에 통합됨

### 22-4. Inspector 결과 확인

1. Inspector → Findings → Lambda 함수 탭
2. `say2-2team-phase2-vision` 함수의 Python 패키지 취약점 목록 확인
3. Critical/High 취약점 발견 시 → 해당 패키지 버전 업그레이드 후 Lambda 재배포

---

## 📋 전체 진행 현황 (최종)

| 단계 | 리소스 | 상태 |
|------|--------|------|
| 1 | VPC | ✅ 기존 팀 VPC 활용 |
| 2 | Subnet 4개 | ✅ 완료 |
| 3 | Security Group 3개 | ✅ 완료 |
| 4 | IGW + NAT Gateway + Route Table | ✅ 완료 (옵션 A) |
| 5 | S3 폴더 (`Phase_2/`) | ✅ 완료 |
| 6 | IAM Role 2개 | ✅ 완료 |
| 7 | 모델 파일 S3 업로드 | ✅ 완료 |
| 8 | SageMaker Endpoint | ⏸️ 발표 직전 생성 |
| 9 | Lambda 함수 | ✅ 완료 |
| 10 | 테스트 | ⬜ |
| 11 | Cognito User Pool | ⬜ 보안 |
| 12 | Secrets Manager | ⬜ 보안 |
| 13 | KMS | ✅ 완료 |
| 14 | CloudTrail | ✅ 완료 |
| 15 | GuardDuty + Security Hub | ⚠️ 권한 없음 |
| 16 | S3 보안 강화 | ✅ 완료 |
| **17** | **API Gateway + Cognito Authorizer** | **⬜ 보안** |
| **18** | **DynamoDB 테이블 2개** | **⬜ 데이터** |
| **19** | **DynamoDB PITR** | **⬜ 복원력** |
| **20** | **SNS 알림** | **✅ 완료 (2026-05-12)** |
| **21** | **AWS Budgets** | **✅ CloudWatch Billing 알람으로 대체 (2026-05-12)** |
| **22** | **Detective + Inspector** | **⬜ 보안** |
| **23** | **프론트엔드 S3 업로드** | **✅ 완료 (`say2-2team-bucket/frontend/`, 2026-05-12)** |
| **24** | **CloudFront 배포** | **⚠️ IAM 권한 필요 — 강사에게 요청** |
| **25** | **WAF WebACL** | **⚠️ IAM 권한 필요 — 강사에게 요청** |
| **26** | **Route53** | **➖ 커스텀 도메인 없음 — 스킵** |
| **27** | **EC2(2-2team-fhir-ec2) IAM 권한 추가** | **⬜ Bedrock + S3 PutObject (→ 섹션 I 참고)** |
| **28** | **EC2에서 rag_llm_3.py 실행 (Aurora + PDF)** | **⬜ SSM 접속 → pip 설치 → 실행 (→ 섹션 I 참고)** |

---

## ⚠️ Step 23~25 — 프론트엔드 CloudFront + WAF 배포

### 현재 완료된 것 (2026-05-12)

```bash
# 프론트엔드 빌드 완료 → S3 업로드 완료
say2-2team-bucket/frontend/
├── index.html          (no-cache)
└── assets/
    ├── index-*.js      (immutable, 1년 캐시)
    └── index-*.css     (immutable, 1년 캐시)
```

### IAM 권한 요청 메시지 (강사에게 전달)

```
안녕하세요. say2-2team 계정 프론트엔드 배포 중 아래 권한이 필요합니다.

필요한 권한:
- cloudfront:CreateDistribution
- cloudfront:CreateOriginAccessControl
- cloudfront:GetDistribution
- cloudfront:UpdateDistribution
- cloudfront:CreateInvalidation
- wafv2:CreateWebACL (us-east-1 리전)
- wafv2:AssociateWebACL (us-east-1 리전)
- wafv2:ListWebACLs

이미지(다이어그램): Client → Route53 → WAF → CloudFront → S3 구조 배포 예정
S3에 프론트엔드 빌드 파일 업로드는 완료되었습니다.
권한 부여 또는 콘솔 직접 생성 도움 요청드립니다. 감사합니다.
```

### 권한 승인 후 콘솔 진행 순서

#### Step 24 — CloudFront 배포 생성

1. CloudFront 콘솔 → **Create distribution**
2. Origin domain: `say2-2team-bucket.s3.ap-northeast-2.amazonaws.com`
3. Origin path: `/frontend`
4. Origin access: **Origin access control settings (OAC)** 선택
   - Create new OAC → Name: `say2-2team-oac` → Sign requests: Yes → Create
5. Viewer protocol policy: **Redirect HTTP to HTTPS**
6. Cache policy: `CachingOptimized` (기본값)
7. Default root object: `index.html`
8. **Custom error responses** → Create custom error response:
   - HTTP error code: `403` → Response page path: `/index.html` → HTTP response code: `200`
   - HTTP error code: `404` → Response page path: `/index.html` → HTTP response code: `200`
9. Price class: `Use only North America, Europe, Asia, Middle East, and Africa`
10. Tags → Key: `project` / Value: `pre-cloudfront-2-2-team`
11. **Create distribution** → 배포 완료 후 S3 버킷 정책 업데이트 알림 나타남 → **Copy policy** → S3 버킷 정책에 붙여넣기
12. Distribution ID와 `https://xxxxx.cloudfront.net` URL을 `resource_ids.md`에 기입

#### Step 25 — WAF WebACL 생성 ✅ 완료 (2026-05-13)

> ⚠️ WAF는 반드시 **us-east-1** 리전에서 생성해야 CloudFront에 연결 가능  
> 새 WAF 콘솔(wafv2-pro)은 기존 Create web ACL 화면과 UI가 다릅니다.

1. **리전 → `us-east-1`** 으로 변경
2. WAF & Shield → **`https://us-east-1.console.aws.amazon.com/wafv2-pro/protections/onboarding`** 접속
3. **Tell us about your app** 섹션:
   - App category: `Enterprise & business applications`
   - App focus: `Both API and web` (기본값)
4. **Select resources to protect** → `E2ZHONIV05TX9D - d300v14l8u0wx7.cloudfront.net` 선택 → **Add**
5. **Choose initial protections** → **`Essentials`** 선택
   - Layer 7 anti-DDoS, IP blocklist/allowlist, SQLi, XSS, Known bad inputs 포함
6. **Name and describe** → Name: `say2-2team-waf`
7. **Create** 클릭
8. 태그 추가 (생성 후): WAF 콘솔 → `say2-2team-waf` → Tags → `project = pre-waf-2-2-team`

**생성 결과:**
| 항목 | 값 |
|------|-----|
| WebACL 이름 | `say2-2team-waf` |
| WebACL ARN | `arn:aws:wafv2:us-east-1:666803869796:global/webacl/say2-2team-waf/6884d8c7-f2b3-4916-b513-40ed0b9bbd12` |
| 연결 배포 | `E2ZHONIV05TX9D` |

---

## 🔐 태그 규칙 전체 정리

| 서비스 | 태그 키 | 태그 값 |
|--------|---------|---------|
| VPC/Subnet/SG/IGW/NAT/RT | `project` | `pre-vpc-2-2-team` |
| S3 | `project` | `pre-s3-2-2-team` |
| IAM Role | `project` | `pre-iam-2-2-team` |
| SageMaker | `project` | `pre-sagemaker-2-2-team` |
| Lambda | `project` | `pre-lambda-2-2-team` |
| Cognito | `project` | `pre-cognito-2-2-team` |
| Secrets Manager | `project` | `pre-secretsmanager-2-2-team` |
| KMS | `project` | `pre-kms-2-2-team` |
| CloudTrail | `project` | `pre-cloudtrail-2-2-team` |
| GuardDuty (S3 export 버킷) | `project` | `pre-guardduty-2-2-team` |
| API Gateway | `project` | `pre-apigateway-2-2-team` |
| DynamoDB | `project` | `pre-dynamodb-2-2-team` |
| SNS | `project` | `pre-sns-2-2-team` |
| Bedrock | `project` | `pre-bedrock-2-2-team` |
| CloudFront | `project` | `pre-cloudfront-2-2-team` |
| WAF | `project` | `pre-waf-2-2-team` |

### 보안 구성 후 시연 후 삭제 추가 항목

```bash
# GuardDuty 비활성화 (과금 방지)
aws guardduty list-detectors --region ap-northeast-2
# → detector-id 확인 후:
aws guardduty delete-detector --detector-id <detector-id> --region ap-northeast-2

# Security Hub 비활성화
aws securityhub disable-security-hub --region ap-northeast-2

# Secrets Manager 삭제 (즉시 삭제 — 기본 7일 대기 건너뜀)
aws secretsmanager delete-secret \
  --secret-id say2-2team/phase2-config \
  --force-delete-without-recovery \
  --region ap-northeast-2

# KMS 키 비활성화 (삭제는 7~30일 대기 필수 — 비활성화로 대체)
aws kms disable-key \
  --key-id alias/say2-2team-data-key \
  --region ap-northeast-2

# CloudTrail 삭제
aws cloudtrail delete-trail \
  --name say2-2team-audit-trail \
  --region ap-northeast-2

# Cognito User Pool 삭제
# 콘솔에서: Cognito → User pools → say2-2team-user-pool → Delete
```

---

## 📝 강사/운영진에게 보고할 문장 템플릿

**옵션 A 실패 시:**
```
say2-2team 계정에서 Phase 2 인프라 배포 중 아래 한도 초과 오류가 발생했습니다.

- IGW: 5/5 (InternetGatewayLimitExceeded)
- EIP: 5/5 (AddressLimitExceeded)

원본 아키텍처(architecture_final.html)에는 NAT Gateway가 필요합니다.
2팀 전용 EIP 1개 + IGW 연결이 필요한데, 한도 증설 또는 기존 미사용 리소스 해제가 가능할까요?

대안으로 VPC Endpoint 버전(옵션 B)으로 배포 중이며 한도 확보 시 옵션 A로 전환 가능합니다.
```

---

## 📖 초보자 완전 이해 가이드 — CloudFront + RAG 아웃풋 + DB 저장 코드

> 이 섹션은 "뭔지 모르겠다"는 분을 위한 전체 개념 설명 + 실행 코드 모음입니다.  
> 참고 노트: [`note_정리/CloudFront_RAG_DB저장_구축가이드.md`](../note_정리/CloudFront_RAG_DB저장_구축가이드.md)

---

### A. 전체 그림 한눈에 보기

```
[의사 PC 브라우저]
      │ HTTPS (암호화된 연결)
      ▼
[WAF — us-east-1] ← SQL인젝션, XSS 등 해커 공격 차단
      │
      ▼
[CloudFront] ← HTTPS 제공 + 전세계 엣지 캐시 (S3 React 앱 서빙)
      │  OAC(Origin Access Control) → S3를 비공개로 유지한 채 CloudFront만 접근 가능
      ▼
[S3: say2-2team-bucket/frontend/] ← React 앱 파일 ✅ 업로드 완료

      별도로: RAG 파이프라인 실행
[rag_llm_3.py 실행]
      │
      ├─→ S3에 JSON + PDF 저장  ← "나중에 파일 다운로드 가능"
      └─→ DynamoDB에 기록       ← "환자 MRN으로 검색 가능"
```

**핵심 개념 2개:**
| 개념 | 한 줄 설명 |
|------|-----------|
| CloudFront | S3 파일을 브라우저에 HTTPS로 전달하는 CDN 서비스. S3를 직접 공개 안 해도 됨 |
| DynamoDB | AWS의 NoSQL DB. 진단 결과를 환자번호/날짜로 나중에 검색하기 위해 저장 |

---

### B. CloudFront 생성 — 새 콘솔 v4 마법사 (2026-05 기준 실제 UI)

> ✅ **WAF가 CloudFront 마법사에 통합됨** — 별도 us-east-1 WAF 생성 불필요  
> URL: `https://us-east-1.console.aws.amazon.com/cloudfront/v4/home#/distributions/create`

---

#### STEP 1 — Get started (배포 기본 정보)

| 항목 | 입력값 | 설명 |
|------|--------|------|
| Distribution name | `say2-2team-cf-distribution` | 필수 입력 (빨간 박스) |
| Distribution type | `Single website or app` | 이미 선택됨, 그대로 |
| Route 53 managed domain | **비워두기** | 커스텀 도메인 없음. 에러 무시 |
| Tags → Key | `project` | |
| Tags → Value | `pre-cloudfront-2-2-team` | |

입력 완료 후 **`Next`** 클릭

---

#### STEP 2 — Specify origin (S3 연결)

| 항목 | 입력값 | 설명 |
|------|--------|------|
| Origin type | `Amazon S3` | 이미 선택됨 |
| S3 origin | `say2-2team-bucket.s3.ap-northeast-2.amazonaws.com` | 자동 입력됨 확인 |
| Origin path | `/frontend` | S3 안의 폴더 경로. 슬래시 포함 필수 |
| Allow private S3 bucket access | ✅ **체크됨 (Recommended)** | OAC 역할. S3 비공개 유지하면서 CloudFront만 접근 가능 |
| Origin settings | `Use recommended origin settings` | 그대로 |
| Cache settings | `Use recommended cache settings for S3` | 그대로 |

**`Next`** 클릭

---

#### STEP 3 — Enable security (WAF 설정)

> CloudFront 마법사의 이 단계는 Shield Standard(기본 DDoS 방어)만 포함됩니다.  
> **SQLi·XSS·Known bad inputs 차단은 별도 WAF WebACL 생성 필요 → Step 25 참고**

| 항목 | 설정 | 설명 |
|------|------|------|
| Use monitor mode | **체크 안 함** | 체크하면 차단 안 하고 감시만 함 |
| Protection against Layer 7 DDoS | **무시** | Business 플랜 전용 |

**`Next`** 클릭

---

#### STEP 4 — Create distribution (최종 설정 + 생성)

> 이 단계에서 Default root object, Error pages 등 핵심 설정을 입력합니다.

| 항목 | 입력값 | 설명 |
|------|--------|------|
| Default root object | `index.html` | 루트 URL 접속 시 반환할 파일 |
| **Custom error responses** | 아래 2개 추가 | React SPA 필수 — 없으면 새로고침 시 404 |
| ↳ 403 에러 | Response: `/index.html` / HTTP code: `200` | S3 접근 거부 → React Router가 처리 |
| ↳ 404 에러 | Response: `/index.html` / HTTP code: `200` | 존재 안 하는 경로 → React Router가 처리 |
| Price class | `Use only North America, Europe, Asia, Middle East, and Africa` | 비용 절감 |

**`Create distribution`** 클릭 → **15~20분 대기** (Status: Deploying → Enabled)

---

#### STEP 5 — 생성 완료 후 S3 버킷 정책 업데이트 (필수!)

CloudFront가 S3 파일을 읽으려면 S3 버킷 정책을 업데이트해야 합니다.

```
방법 A (자동): 배포 완료 화면에 노란 배너 표시됨
  → "Copy policy" 클릭
  → S3 → say2-2team-bucket → Permissions → Bucket policy → Edit → 붙여넣기 → Save

방법 B (수동): 아래 정책을 S3 버킷 정책에 추가
```

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "AllowCloudFrontOAC",
    "Effect": "Allow",
    "Principal": {
      "Service": "cloudfront.amazonaws.com"
    },
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::say2-2team-bucket/frontend/*",
    "Condition": {
      "StringEquals": {
        "AWS:SourceArn": "arn:aws:cloudfront::666803869796:distribution/[배포ID]"
      }
    }
  }]
}
```

> `[배포ID]`는 배포 완료 후 CloudFront → Distributions 목록에서 확인

---

#### STEP 6 — 완료 후 기록 (resource_ids.md에 기입)

```
CloudFront Distribution ID : E_____________
CloudFront URL             : https://_____________.cloudfront.net
→ 이 URL이 프론트엔드 접속 주소 (커스텀 도메인 없어도 이걸로 접속 가능)
```

---

#### WAF WebACL 연결 확인 ✅

Step 25에서 생성한 `say2-2team-waf`가 `E2ZHONIV05TX9D`에 연결됨.  
ARN: `arn:aws:wafv2:us-east-1:666803869796:global/webacl/say2-2team-waf/6884d8c7-f2b3-4916-b513-40ed0b9bbd12`

---

### C. RAG 아웃풋 이해 — LLM이 뭘 반환하는가

`rag_llm_3.py`에서 Bedrock Claude가 반환하는 것은 **JSON 텍스트**입니다.  
PDF를 직접 생성하지 못합니다. 코드가 이 JSON을 파싱해서 PDF로 변환합니다.

```
LLM (Claude 3.5 Sonnet)이 반환하는 JSON 구조:

{
  "recommendation": {
    "immediate_workup": ["CT 혈관조영술 시행", "심초음파 시행"],
    "specialist_referral": ["흉부외과·호흡기내과 MDT 의뢰"],
    "treatment_guideline": ["[ORPHA:91387] PMID:12345678"],
    "clinical_trial_info": ["NCT12345678 — 모집 중"],
    "genetic_test": ["ACTA2, FBN1 유전자 검사"],
    "additional_lab": ["D-dimer 재측정"]
  },
  "clinical_notes": {
    "summary": "42세 남성, 흉통 및 호흡곤란으로 응급실 내원...",
    "top1_reasoning": "HP:0002107 기흉이 Orphanet Frequent와 일치...",
    "differential_note": "Top2 AMI는 Troponin 정상으로 배제...",
    "rag_evidence": "ACTA2 DB·API 교차검증 일치...",
    "case_comparison": "PubMed PMID:37654321 케이스와 유사...",
    "disclaimer": "AI 결과는 진단 보조이며 최종 판단은 주치의..."
  },
  "confidence_metrics": {
    "overall_confidence_score": 0.87,
    "rationale": "DB·API 일치율 높고 PubMed 근거 충분",
    "data_sufficiency": {
      "genomic_evidence": "High",
      "clinical_case_match": "Medium",
      "trial_availability": "Low"
    }
  }
}
```

**현재 rag_llm_3.py 저장 로직의 문제:**
```python
# 현재 코드 (맨 아래 ~637줄):
# → 로컬 .json 파일로만 저장됨
# → 나중에 파일 찾을 방법 없음
# → S3, DynamoDB 연동 없음
```

**추가해야 할 것:**
```
1. render_pdf_from_markdown()  → Markdown → HTML → weasyprint → PDF
2. save_final_report()         → PDF S3 업로드 + Aurora rarelinkai.final_report INSERT
3. get_aurora_connection()     → Secrets Manager 자격증명으로 psycopg2 연결
4. import 추가                 → psycopg2-binary, markdown, weasyprint 설치 필요
```

---

### D. 실제 DB — Aurora PostgreSQL (이미 구축됨, DynamoDB 아님)

> ⚠️ **이전 버전의 가이드에서 DynamoDB로 잘못 안내했습니다. 올바른 DB는 Aurora입니다.**  
> `patient-db-cluster` 는 이미 구축되어 있으며, 허태웅 파트에서 별도 생성할 DB는 없습니다.


#### Aurora 클러스터 접속 정보 (이미 존재)

```
host:     patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com
port:     5432
database: rarelink
schema:   rarelinkai
VPC SG:   sg-019a357627f1594db
Secrets:  rare-link-ai/aurora/app-user  (Secrets Manager)
```

#### RAG 파트에서 쓰는 Aurora 테이블 (docx 5.7 기준)

```
rarelinkai.final_report  ← RAG 결과 저장 (허태웅 파트)
  session_id (FK) + generated_at  ← 복합 PK
  diagnosis_json     JSONB   ← LLM 전체 JSON
  markdown_report    TEXT    ← LLM이 생성한 마크다운 보고서
  rag_citations      JSONB   ← PubMed 인용 목록
  rag_apis_used      TEXT[]  ← ["PubMed","Orphanet","Monarch","PubCaseFinder"]
  self_check         JSONB   ← hallucination 자기검증 결과
  s3_uri_pdf         TEXT    ★ 의사 전달용 PDF의 S3 URI (가장 중요)
  s3_uri_html        TEXT    ← HTML 버전 (선택)
  pdf_sha256         CHAR(64)← PDF 무결성 해시
  pdf_size_bytes     INT
  pdf_generated_at   TIMESTAMPTZ
  external_api_call_summary JSONB  ← [{api,calls,hits,avg_ms,cache_hit_rate}]
  llm_model          TEXT
  total_inference_time_ms INT

rarelinkai.rag_api_cache  ← 외부 API 캐시 (TTL 7일)
  cache_key (PK)  source_api  query_params  response_json  expires_at
```

#### ⚠️ VPC SG 수정 필요 (Lambda → Aurora 연결)

현재 Lambda SG(`sg-03e64fdde60d52a6c`)는 Aurora SG 인바운드 허용 목록에 없습니다.

```
EC2/VPC 콘솔 → Security Groups → sg-019a357627f1594db (Aurora SG)
→ Inbound rules → Edit inbound rules → Add rule:
  Type: PostgreSQL  |  Port: 5432  |  Source: sg-03e64fdde60d52a6c (Lambda SG)
→ Save rules
```

---

### E. rag_llm_3.py 수정 코드 — 올바른 버전 (Aurora + weasyprint)

#### 1단계: 라이브러리 설치

```bash
pip install psycopg2-binary markdown weasyprint
# Lambda에 배포할 경우: psycopg2-binary는 Lambda Layer 필요
# (https://github.com/jkehler/awslambda-psycopg2 — Python 3.11용 빌드 사용)
```

#### 2단계: 파일 맨 위 import 추가

```python
# 기존 import 아래에 추가
import hashlib
import time
import json
import psycopg2
from psycopg2.extras import Json, RealDictCursor
import markdown as md_lib      # pip install markdown
import weasyprint               # pip install weasyprint
```

#### 3단계: if __name__ == "__main__": 블록 바로 위에 함수 3개 추가

```python
# =====================================================================
# Aurora 연결 함수 (Secrets Manager 자격증명 사용)
# =====================================================================
def get_aurora_connection():
    """Secrets Manager → Aurora psycopg2 연결"""
    sm = boto3.client("secretsmanager", region_name="ap-northeast-2")
    secret = json.loads(
        sm.get_secret_value(SecretId="rare-link-ai/aurora/app-user")["SecretString"]
    )
    return psycopg2.connect(
        host="patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com",
        port=5432,
        database="rarelink",
        user=secret["username"],
        password=secret["password"],
        options="-c search_path=rarelinkai",
        connect_timeout=10,
    )


# =====================================================================
# PDF 생성 함수 — docx 3.3 기준 (markdown → HTML → weasyprint → PDF)
# =====================================================================
def render_pdf_from_markdown(markdown_text: str) -> bytes:
    """
    Markdown → HTML → PDF (weasyprint + 한국어 Noto Sans KR 폰트)
    fpdf2 가 아닌 weasyprint 사용 — 한글, 표, 복잡한 레이아웃 지원
    """
    html_body = md_lib.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "nl2br"],
    )
    html_full = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
    body {{
      font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
      font-size: 11pt; margin: 2cm; color: #0A1628;
    }}
    h1 {{ font-size: 18pt; color: #0C447C; border-bottom: 2px solid #0C447C; padding-bottom: 4px; }}
    h2 {{ font-size: 14pt; color: #0C447C; border-bottom: 1px solid #ccc; margin-top: 16pt; }}
    h3 {{ font-size: 12pt; color: #0E8574; }}
    table {{ border-collapse: collapse; width: 100%; margin: 8pt 0; }}
    th {{ background: #0C447C; color: white; padding: 5px 8px; font-size: 10pt; }}
    td {{ border: 1px solid #ccc; padding: 4px 8px; font-size: 10pt; }}
    code {{ background: #F8FAFC; font-size: 9pt; padding: 1px 3px; }}
    .disclaimer {{
      font-size: 9pt; color: #666; border-top: 1px solid #ccc;
      margin-top: 24pt; padding-top: 8pt;
    }}
  </style>
</head>
<body>
{html_body}
<p class="disclaimer">
  본 보고서는 Rare-Link AI 진단 보조 시스템의 출력물입니다.
  최종 진단은 반드시 담당 의사의 임상 판단에 따라야 합니다 (EU AI Act Art.22).
</p>
</body>
</html>"""
    pdf_bytes = weasyprint.HTML(string=html_full).write_pdf()
    return pdf_bytes


# =====================================================================
# S3 업로드 + Aurora final_report INSERT — docx 5.7 기준
# =====================================================================
def save_final_report(
    session_id: str,
    diagnosis_json: dict,
    markdown_report: str,
    rag_citations: list,
    rag_apis_used: list,
    self_check: dict,
    external_api_call_summary: list,
    llm_model: str,
    total_inference_time_ms: int,
    s3_bucket: str = "say2-2team-bucket",
    region: str = "ap-northeast-2",
) -> dict:
    """
    1) PDF 렌더링 (weasyprint)
    2) S3 업로드  →  final_reports/{session_id}/report.pdf
    3) Aurora rarelinkai.final_report INSERT (psycopg2)

    반환: {"session_id", "s3_uri_pdf", "pdf_sha256", "generated_at"}
    """
    # ── PDF 렌더링 ────────────────────────────────────────────────
    pdf_bytes      = render_pdf_from_markdown(markdown_report)
    pdf_sha256     = hashlib.sha256(pdf_bytes).hexdigest()
    pdf_size_bytes = len(pdf_bytes)
    from datetime import datetime, timezone
    pdf_generated_at = datetime.now(timezone.utc)

    # ── S3 업로드 ─────────────────────────────────────────────────
    s3 = boto3.client("s3", region_name=region)
    s3_key     = f"final_reports/{session_id}/report.pdf"
    s3_uri_pdf = f"s3://{s3_bucket}/{s3_key}"

    s3.put_object(
        Bucket=s3_bucket, Key=s3_key,
        Body=pdf_bytes,
        ContentType="application/pdf",
    )
    print(f"  ✅ S3 PDF: {s3_uri_pdf}")

    # ── Aurora INSERT ─────────────────────────────────────────────
    sql = """
        INSERT INTO rarelinkai.final_report (
            session_id,
            diagnosis_json, markdown_report,
            rag_citations, rag_apis_used, self_check,
            s3_uri_pdf, s3_uri_html,
            pdf_sha256, pdf_size_bytes, pdf_generated_at,
            external_api_call_summary,
            llm_model, total_inference_time_ms
        ) VALUES (
            %(session_id)s,
            %(diagnosis)s, %(markdown)s,
            %(citations)s, %(apis)s, %(self_check)s,
            %(s3_pdf)s, NULL,
            %(sha256)s, %(size)s, %(pdf_at)s,
            %(api_summary)s,
            %(model)s, %(time_ms)s
        ) RETURNING generated_at
    """
    conn = get_aurora_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, {
                "session_id": session_id,
                "diagnosis":  Json(diagnosis_json),
                "markdown":   markdown_report,
                "citations":  Json(rag_citations),
                "apis":       rag_apis_used,            # TEXT[] — psycopg2 자동 변환
                "self_check": Json(self_check),
                "s3_pdf":     s3_uri_pdf,
                "sha256":     pdf_sha256,
                "size":       pdf_size_bytes,
                "pdf_at":     pdf_generated_at,
                "api_summary":Json(external_api_call_summary),
                "model":      llm_model,
                "time_ms":    total_inference_time_ms,
            })
            generated_at = cur.fetchone()["generated_at"]
        conn.commit()
        print(f"  ✅ Aurora [rarelinkai.final_report] 저장 완료 (generated_at={generated_at})")
    finally:
        conn.close()

    return {
        "session_id":   session_id,
        "s3_uri_pdf":   s3_uri_pdf,
        "pdf_sha256":   pdf_sha256,
        "generated_at": str(generated_at),
    }


# =====================================================================
# rag_api_cache UPSERT (외부 API 응답 캐시 — docx 5.8)
# =====================================================================
def upsert_api_cache(
    cache_key: str,
    source_api: str,
    query_params: dict,
    response_json: dict,
    ttl_days: int = 7,
) -> None:
    """같은 cache_key는 expires_at 갱신 (ON CONFLICT UPDATE)"""
    sql = """
        INSERT INTO rarelinkai.rag_api_cache (
            cache_key, source_api, query_params, response_json, expires_at
        ) VALUES (
            %(k)s, %(api)s, %(params)s, %(resp)s,
            NOW() + (%(ttl)s || ' days')::INTERVAL
        )
        ON CONFLICT (cache_key) DO UPDATE SET
            response_json = EXCLUDED.response_json,
            expires_at    = EXCLUDED.expires_at
    """
    conn = get_aurora_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, {
                "k": cache_key, "api": source_api,
                "params": Json(query_params),
                "resp":   Json(response_json),
                "ttl":    str(ttl_days),
            })
        conn.commit()
    finally:
        conn.close()
```

#### 4단계: if __name__ == "__main__": 블록 안 저장 코드 교체

**삭제할 기존 코드 (~637줄):**
```python
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"diagnosis_report_orpha91387_{timestamp}.json"
with open(filename, "w", encoding="utf-8") as f:
    f.write(final_report)
print(f"\n✅ 진단 레포트(JSON)가 성공적으로 저장되었습니다: {filename}")
```

**교체할 새 코드:**
```python
# ── LLM 호출 직전에 타이머 시작 (정확한 추론 시간 측정) ──────
import time
_t_start = time.time()

# ... [기존 LLM invoke 코드 그대로] ...
# final_report = invoke_bedrock(...)  ← 기존 코드

# ── JSON 파싱 ──────────────────────────────────────────────────
try:
    report_json = json.loads(final_report)
except (json.JSONDecodeError, TypeError):
    print("  ⚠️ LLM 응답이 유효한 JSON이 아님")
    report_json = {"raw": str(final_report)}

_total_ms = int((time.time() - _t_start) * 1000)

# ── markdown_report 구성 (LLM JSON → Markdown 변환) ───────────
notes = report_json.get("clinical_notes", {})
rec   = report_json.get("recommendation", {})
conf  = report_json.get("confidence_metrics", {})

markdown_report = f"""# Rare-Link AI 진단 보고서

## 1. 임상 요약
{notes.get("summary", "N/A")}

## 2. Top 3 진단 근거

### Top 1 진단 근거
{notes.get("top1_reasoning", "N/A")}

### Top 2 진단 근거
{notes.get("top2_reasoning", "") or "_해당 없음_"}

### Top 3 진단 근거
{notes.get("top3_reasoning", "") or "_해당 없음_"}

## 3. 감별 진단 (Top 1 우선 근거)
{notes.get("differential_note", "N/A")}

## 4. 권고 사항
### 즉시 검사
{chr(10).join(f"- {x}" for x in rec.get("immediate_workup", []))}

### 전문의 의뢰
{chr(10).join(f"- {x}" for x in rec.get("specialist_referral", []))}

### 유전자 검사
{chr(10).join(f"- {x}" for x in rec.get("genetic_test", []))}

## 5. RAG 근거 (외부 문헌)
{notes.get("rag_evidence", "N/A")}

## 6. 신뢰도
- 전체 점수: {conf.get("overall_confidence_score", 0):.0%}
- 근거: {conf.get("rationale", "N/A")}
"""

# ── Aurora + S3 저장 ───────────────────────────────────────────
print("\n💾 Aurora + S3 저장 중...")
saved = save_final_report(
    session_id               = SESSION_ID,       # ← rag_llm_3.py에서 받는 session_id 변수
    diagnosis_json           = report_json,
    markdown_report          = markdown_report,
    rag_citations            = [
        notes.get("rag_evidence", ""),
        notes.get("case_comparison", ""),
    ],
    rag_apis_used            = ["PubMed", "Orphanet", "Monarch", "PubCaseFinder"],
    self_check               = conf.get("data_sufficiency", {}),
    external_api_call_summary= [],              # gather_rag_data() 통계로 채울 것
    llm_model                = "claude-sonnet-4-20250514",
    total_inference_time_ms  = _total_ms,
)

print(f"\n✅ 저장 완료:")
print(f"   session_id  : {saved['session_id']}")
print(f"   PDF (S3)    : {saved['s3_uri_pdf']}")
print(f"   SHA256      : {saved['pdf_sha256'][:16]}...")
print(f"   generated_at: {saved['generated_at']}")
```

> ⚠️ `SESSION_ID` 는 `diagnosis_session` 테이블에서 발급된 UUID.  
> rag_llm_3.py가 `session_id`를 이벤트 파라미터나 환경변수로 받지 않는다면  
> 임시로 `SESSION_ID = str(uuid.uuid4())` 를 사용할 것.

---

### F. IAM 권한 + VPC SG 수정 (Aurora 연결을 위한 인프라 조치)

#### F-1. Lambda Role에 Secrets Manager 읽기 권한 추가

Lambda가 `rare-link-ai/aurora/app-user` 시크릿을 읽을 수 있어야 합니다.

1. IAM → Roles → `say2-2team-lambda-role` → Add permissions → Create inline policy
2. JSON 탭에 아래 붙여넣기:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["secretsmanager:GetSecretValue"],
    "Resource": [
      "arn:aws:secretsmanager:ap-northeast-2:*:secret:rare-link-ai/aurora/*"
    ]
  }]
}
```

3. Policy name: `AuroraSecretsAccess` → Create policy

#### F-2. VPC Security Group — Lambda → Aurora 포트 개방 (필수!)

Lambda SG(`sg-03e64fdde60d52a6c`)가 Aurora SG의 인바운드에 없으면 연결 불가.

```
EC2 콘솔 → Security Groups → sg-019a357627f1594db 검색
→ Inbound rules 탭 → Edit inbound rules → Add rule:
  Type:     Custom TCP
  Protocol: TCP
  Port:     5432
  Source:   sg-03e64fdde60d52a6c  (say2-2team-lambda-sg)
→ Save rules
```

#### F-3. psycopg2 Lambda Layer 추가 (독립 실행 스크립트라면 불필요)

rag_llm_3.py가 Lambda가 아닌 **독립 Python 스크립트**로 실행된다면 `pip install psycopg2-binary`만으로 충분합니다.

Lambda 함수로 배포해야 한다면:

```bash
# Python 3.11용 psycopg2 레이어 다운로드 (공개 레이어 사용)
# https://github.com/jkehler/awslambda-psycopg2
# → 또는 아래 공개 ARN 사용 (ap-northeast-2, Python 3.11)
# arn:aws:lambda:ap-northeast-2:898466741470:layer:psycopg2-py311:1
```

---

### G. 오늘 할 일 순서 체크리스트 (수정판)

```
[ ] 0. VPC SG 수정 (F-2) — Lambda SG → Aurora SG 5432 개방
        sg-019a357627f1594db 인바운드에 sg-03e64fdde60d52a6c 추가

[ ] 1. IAM Lambda Role에 Secrets Manager 권한 추가 (F-1)
        Policy: AuroraSecretsAccess

[ ] 2. 라이브러리 설치 (터미널 2분)
        pip install psycopg2-binary markdown weasyprint

[ ] 3. rag_llm_3.py 수정 (E 섹션 코드 삽입, 15분)
        - import 추가 (hashlib, time, psycopg2, markdown, weasyprint)
        - get_aurora_connection() 함수 추가
        - render_pdf_from_markdown() 함수 추가 (weasyprint 기반)
        - save_final_report() 함수 추가 (Aurora INSERT)
        - upsert_api_cache() 함수 추가
        - 기존 로컬 저장 코드 → save_final_report() 호출로 교체

[ ] 4. 연결 테스트 (Aurora VPC 내부에서 실행해야 함)
        python -c "
        from rag_llm_3 import get_aurora_connection
        conn = get_aurora_connection()
        print('Aurora 연결 성공:', conn.server_version)
        conn.close()
        "
        # ⚠️ 로컬 PC에서는 VPC 외부라 연결 안 됨 → Lambda/EC2에서 실행할 것

[ ] 5. rag_llm_3.py 전체 실행 — EC2(2-2team-fhir-ec2) 사용 (→ 섹션 I 참고)
        # I-1. IAM 권한 추가 (BedrockInvokeAccess + S3FinalReportWrite)
        # I-2. SSM Session Manager로 EC2 접속 (키 없이 콘솔에서)
        # I-3. pip3 install psycopg2-binary markdown weasyprint aiohttp requests pandas boto3
        # I-4. aws s3 cp s3://say2-2team-bucket/RAG/rag_llm_3.py .
        #       python3 rag_llm_3.py
        # 성공 시:
        # ✅ S3 PDF: s3://say2-2team-bucket/final_reports/{session_id}/report.pdf
        # ✅ Aurora [rarelinkai.final_report] 저장 완료

[ ] 6. CloudFront + WAF (강사 권한 승인 후 — B 섹션 참고)

[ ] 7. infra/resource_ids.md에 CloudFront URL + WAF ARN 기입
```

> ⚠️ **로컬 PC에서 Aurora 직접 연결 불가**: Aurora는 VPC 내부 전용.  
> 테스트 시 Lambda 콘솔에서 테스트 이벤트를 직접 실행하거나,  
> AWS Session Manager로 VPC 내 EC2에 접속 후 실행할 것.

---

### H. 비용 참고

| 리소스 | 비용 | 비고 |
|--------|------|------|
| Aurora Serverless v2 (기존) | ~$0.06/ACU·h | 팀 공유 클러스터 |
| S3 PDF 저장 | $0.023/GB/월 | 수 MB 수준 |
| CloudFront | $0.0085/10,000건 | 권한 대기 중 |
| WAF | $5/WebACL/월 | 권한 대기 중 |
| Secrets Manager | $0.40/secret/월 | 기존 사용 중 |
| **합계** | **~$6~8/월** | 데모 수준 |

---

### I. EC2에서 rag_llm_3.py 실행 가이드 ✅ (2-2team-fhir-ec2 활용)

> **선택한 방법**: Lambda가 아닌 기존 EC2(`2-2team-fhir-ec2`)에서 스크립트 직접 실행  
> **이유**: weasyprint(PDF 생성)가 Lambda에서 시스템 라이브러리 부재로 동작 불가 → EC2에서는 완전 동작  
> **Aurora 저장**: 문제없음 — EC2가 이미 Aurora SG 인바운드에 등록되어 있음 ✅

#### I-0. 현황 확인 (이미 완료된 것들)

| 항목 | 상태 | 값 |
|------|------|----|
| EC2 인스턴스 | ✅ Running | `i-0f3f223fd40217b12` |
| EC2 이름 | ✅ | `2-2team-fhir-ec2` |
| VPC | ✅ 동일 VPC | `vpc-06dd0ad1f2335ea74` (say2-2team) |
| Aurora 포트 5432 | ✅ 이미 허용 | Aurora SG(`sg-019a357627f1594db`) 인바운드에 `fhir-ec2-sg`(`sg-03b9bc5d95699b797`) 등록됨 |
| Secrets Manager | ✅ 권한 있음 | `SecretsManagerAccess` 정책 포함 |
| SSM 접속 | ✅ 키 없이 가능 | `AmazonSSMManagedInstanceCore` 정책 포함 |
| **Bedrock 권한** | ❌ **추가 필요** | `bedrock:InvokeModel` 없음 |
| **S3 쓰기 권한** | ❌ **추가 필요** | `S3ReadAccess`만 있음, `PutObject` 없음 |

---

#### I-1. IAM 권한 추가 (콘솔에서 2분)

> ⚠️ **이 단계를 먼저 해야 합니다.** 권한 없이 스크립트 실행 시 AccessDenied 오류.

**1-1. IAM 콘솔 접속**

```
AWS 콘솔 → IAM → 좌측 "Roles" 클릭 → 검색창에 "fhir-ec2-role" 입력 → 클릭
```

**1-2. Bedrock 권한 추가**

```
"Add permissions" 버튼 클릭 → "Create inline policy" 선택
→ 상단 "JSON" 탭 클릭
→ 아래 JSON 전체 붙여넣기
```

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockInvokeModel",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel"],
      "Resource": "arn:aws:bedrock:ap-northeast-2::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
    }
  ]
}
```

```
→ "Next" 클릭
→ Policy name: "BedrockInvokeAccess" 입력
→ "Create policy" 클릭
```

**1-3. S3 쓰기 권한 추가**

같은 `fhir-ec2-role` 페이지에서:

```
"Add permissions" → "Create inline policy" → JSON 탭
```

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3FinalReportWrite",
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": "arn:aws:s3:::say2-2team-bucket/final_reports/*"
    }
  ]
}
```

```
→ Policy name: "S3FinalReportWrite" 입력
→ "Create policy" 클릭
```

**완료 확인**: `fhir-ec2-role` → Permissions 탭에 아래 6개 정책 보여야 함

| 정책 이름 | 역할 |
|-----------|------|
| AmazonSSMManagedInstanceCore | SSM 접속 |
| KMSDecryptAccess | KMS 복호화 |
| S3ReadAccess | S3 읽기 |
| SecretsManagerAccess | Secrets Manager 읽기 |
| BedrockInvokeAccess | **방금 추가** — Bedrock LLM 호출 |
| S3FinalReportWrite | **방금 추가** — S3 PDF 저장 |

**1-4. IAM Role 태그 추가**

> ⚠️ **인라인 정책 자체는 태그 불가** — IAM에서 태그는 Role/User/Group에만 달 수 있음.  
> 정책이 아닌 **`fhir-ec2-role` Role 자체**에 태그를 달아야 합니다.

```
IAM → Roles → fhir-ec2-role → "Tags" 탭
→ "Add tags" 클릭
→ Key: project  /  Value: pre-iam-2-2-team
→ Save
```

> ℹ️ 이미 팀원이 생성한 Role이라 태그가 이미 있을 수 있음.  
> Tags 탭에서 `project` 키가 있으면 그대로 두고, 없는 경우만 추가.

---

#### I-2. EC2 접속 (SSM Session Manager — 키 없이)

> 별도의 SSH 키 파일 없이 AWS 콘솔에서 바로 터미널 접속 가능합니다.

```
AWS 콘솔 → EC2 → Instances → "i-0f3f223fd40217b12" 클릭
→ 상단 "Connect" 버튼 클릭
→ "Session Manager" 탭 선택
→ "Connect" 클릭
```

> 브라우저에서 터미널 창이 열립니다. 여기서 아래 명령어를 실행합니다.

---

#### I-3. 패키지 설치 (처음 한 번만)

> ✅ **이 EC2는 Ubuntu 25.04 (resolute)** — `yum` 없음, `apt-get` 사용

EC2 터미널에서 아래 순서대로 실행:

**3-1. Python 환경 확인**

```bash
python3 --version
# Python 3.14.x 가 나오면 OK
```

**3-2. pip3 설치**

```bash
sudo apt-get install -y python3-pip
```

**3-3. 시스템 라이브러리 설치 (weasyprint용)**

```bash
sudo apt-get install -y \
  libpango-1.0-0 libpangoft2-1.0-0 libcairo2 \
  libgdk-pixbuf-xlib-2.0-0 libffi-dev
```

> ⚠️ Ubuntu 25.04에서 `libgdk-pixbuf2.0-0` → `libgdk-pixbuf-xlib-2.0-0` 으로 패키지명 변경됨

**3-4. Python 패키지 설치**

```bash
pip3 install psycopg2-binary markdown weasyprint aiohttp requests pandas boto3 --break-system-packages
```

> ⚠️ Ubuntu 25.04는 PEP 668 적용으로 `--break-system-packages` 플래그 필요  
> 설치 완료 메시지: `Successfully installed weasyprint-...` 등이 표시되면 OK

**3-5. 설치 확인**

```bash
python3 -c "import weasyprint; import psycopg2; print('✅ 패키지 설치 완료')"
```

---

#### I-4. 스크립트 다운로드 및 실행

**4-1. S3에서 최신 스크립트 다운로드**

```bash
cd ~
aws s3 cp s3://say2-2team-bucket/RAG/rag_llm_3.py . --region ap-northeast-2
ls -lh rag_llm_3.py
# 약 50KB 파일이 보이면 OK
```

**4-2. 실행**

```bash
python3 rag_llm_3.py
```

**4-3. 정상 실행 시 출력 예시**

```
[RAG 시스템이 조립한 최종 프롬프트]
...
🧠 Bedrock (anthropic.claude-3-5-sonnet-...) 진단 추론 시작...
🔍 생성된 리포트의 PMID 검증 중...
  ✅ 37654321 — Familial thoracic aortic aneurysm...
✅ 로컬 JSON 백업 저장: diagnosis_report_orpha91387_20260513_143022.json

💾 S3 + Aurora 저장 중...
  ✅ S3 PDF 업로드 완료: s3://say2-2team-bucket/final_reports/{UUID}/report.pdf (245,312 bytes)
  ✅ Aurora [rarelinkai.final_report] INSERT 완료 (generated_at=2026-05-13 ...)

📋 저장 결과:
  session_id : xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
  S3 PDF     : s3://say2-2team-bucket/final_reports/.../report.pdf
  Aurora     : ✅ 저장 완료
```

---

#### I-5. 결과 확인

**S3에서 PDF 확인:**

```
AWS 콘솔 → S3 → say2-2team-bucket → final_reports/ 폴더 확인
→ {session_id}/report.pdf 파일이 있으면 성공
```

**Aurora에서 저장 확인 (선택):**

EC2 터미널에서:

```bash
python3 -c "
from rag_llm_3 import get_aurora_connection
conn = get_aurora_connection()
cur = conn.cursor()
cur.execute('SELECT session_id, s3_uri_pdf, generated_at FROM rarelinkai.final_report ORDER BY generated_at DESC LIMIT 3')
for row in cur.fetchall():
    print(row)
conn.close()
"
```

---

#### I-6. 오류 발생 시 확인 사항

| 오류 메시지 | 원인 | 해결 방법 |
|------------|------|----------|
| `AccessDenied: bedrock:InvokeModel` | I-1 Bedrock 권한 미추가 | I-1 단계 다시 확인 |
| `AccessDenied: s3:PutObject` | I-1 S3 쓰기 권한 미추가 | I-1 단계 다시 확인 |
| `could not connect to server` (Aurora) | Aurora VPC SG 문제 | Aurora SG 인바운드 확인 (이미 등록됨 → 재확인) |
| `ResourceNotFoundException: rare-link-ai/aurora/app-user` | Secrets Manager 시크릿 이름 오류 | 시크릿 이름 정확히 확인 |
| `No module named 'weasyprint'` | 패키지 미설치 | I-3 단계 재실행 |
| `⚠️ PDF 생성/S3 업로드 실패 (Aurora INSERT는 계속 진행)` | weasyprint 시스템 라이브러리 없음 | I-3-2 시스템 라이브러리 설치 후 재시도 |
| `Extra data: line 1 column 2 (char 1)` | Secrets Manager 시크릿이 plain text 형식인데 `json.loads()` 시도 | **rag_llm_3.py v2026-05-14에서 수정 완료** — 자동 감지됨 |
| `malformed array literal` | `rag_apis_used` (VARCHAR[] 컬럼)에 Json 래퍼 사용 | **rag_llm_3.py v2026-05-14에서 수정 완료** |
| `ForeignKeyViolation: final_report_session_id_fkey` | FK 체인 미충족 (raw_emr_bundle→patient_profile→diagnosis_session 선행 필요) | **rag_llm_3.py v2026-05-14에서 자동 처리** |
| `Connection timed out` (Bedrock) | VPCE SG에 EC2 SG가 미허용 | `aws ec2 authorize-security-group-ingress --group-id sg-0cf817a0115fa94bd --protocol tcp --port 443 --source-group sg-03b9bc5d95699b797` (이미 적용됨) |

> ⚠️ **PDF 실패해도 Aurora INSERT는 항상 실행됩니다** (graceful degradation 설계).  
> 즉, weasyprint가 실패해도 Aurora에는 `s3_uri_pdf = NULL`로 저장되고 스크립트는 계속 진행됩니다.

---

#### I-7. 업데이트된 스크립트 재배포 (코드 수정 후)

로컬 Mac에서 수정한 경우:

```bash
# 로컬 Mac 터미널에서
aws s3 cp /tmp/rag_llm_3.py s3://say2-2team-bucket/RAG/rag_llm_3.py --region ap-northeast-2

# EC2 터미널에서
aws s3 cp s3://say2-2team-bucket/RAG/rag_llm_3.py . --region ap-northeast-2
python3 rag_llm_3.py
```

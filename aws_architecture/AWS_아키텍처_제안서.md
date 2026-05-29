# Rare-Link AI — AWS 아키텍처 제안서

작성일: 2026-04-27 (최초) / 2026-05-04 (회의 후 화이트보드 반영)
근거: 팀 회의 녹음 (2026-04-27) + 화이트보드 정리 (2026-05-04) + 현행 코드베이스 분석

---

## 1. 핵심 설계 원칙

회의에서 멘토(이희찬)님이 강조한 3가지 핵심:

1. **EMR 대체가 아닌 브릿징** — 기존 EMR 업체와 경쟁하지 않고, API 게이트웨이를 통해 어떤 EMR이든 연동 가능한 구조
2. **포트폴리오용 AWS 풀스택 + 현실용 온프레미스 듀얼 아키텍처** — AWS 서비스를 최대한 활용한 그림 + 온프레미스 배포 가능한 구조 병행
3. **모델 자동 갱신 파이프라인** — 새 환자 데이터 → 희귀질환 DB 축적 → 배치 재학습 → 모델 자동 교체

---

## 2. 전체 아키텍처 (AWS 클라우드 버전) — 2026-05-04 회의 반영

화이트보드 정리 결과 **5-Phase 파이프라인**으로 재정의됨. 입력 에이전트(기존 1A/1B/1C)가
독립 Phase 1·2·3로 분리되고, 검증 단계(Phase 4)와 희귀질환 분리 단계(Phase 5)가 명시됨.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        외부 연동 계층                                    │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │ EMR 시스템 │  │ SMART on FHIR│  │ 사전 문진    │                      │
│  │ (BESTCare │  │  Sandbox     │  │ (향후 과제)  │                      │
│  └─────┬─────┘  └──────┬───────┘  └──────┬───────┘                      │
│        └───────────────┼─────────────────┘                              │
│                        ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │   API Gateway (FHIR R4) + WAF + CloudFront                      │    │
│  └─────────────────────────┬───────────────────────────────────────┘    │
└────────────────────────────┼────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Public Subnet — SQS (요청 큐잉, 트래픽 폭주 흡수)                         │
└────────────────────────────┬────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Private Subnet ① — Phase 1·2·3 (1차 Output 생성)                        │
│  ┌──────────────────────────┐  ┌──────────────────────────┐             │
│  │ [Phase 1] Symptom        │  │ Lambda → Bedrock Haiku    │             │
│  │  (LLM-Bedrock)           │  │ → HPO(JSON) + 근거 정리    │             │
│  │ 환자 기본정보·증상(문진)    │  │                          │             │
│  └──────────────────────────┘  └──────────────────────────┘             │
│  ┌──────────────────────────┐  ┌──────────────────────────┐             │
│  │ [Phase 2] X-ray (SooNet) │  │ SageMaker Endpoint        │             │
│  │  ml.g4dn.xlarge GPU      │  │ → HPO%(JSON, 14-class)   │             │
│  └──────────────────────────┘  └──────────────────────────┘             │
│  ┌──────────────────────────┐  ┌──────────────────────────┐             │
│  │ [Phase 3] Multimodal-    │  │ Lambda (Lab Rules)        │             │
│  │  Scoring (lab + micro)   │  │ → HPO + 추가 data         │             │
│  └──────────────────────────┘  └──────────────────────────┘             │
│                          ↓                                              │
│                   🌀 1차 Output (HPO 통합 + 가중치 적용)                   │
└─────────────────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Private Subnet ② — 가중치 적용 ~ 2차 Output (검증·희귀 분리)               │
│  ┌──────────────────────────────────────────────────┐                   │
│  │ 일반 Ranking (Lambda: 가중치·규칙 기반 점수 계산)    │                   │
│  └──────────────────────────────────────────────────┘                   │
│                          ↓                                              │
│  ┌──────────────────────────────────────────────────┐                   │
│  │ [Phase 4] 검증 → 일반 Ranking (확정)              │                   │
│  │  Lambda: cross-check (LIRICAL ↔ 일반 score ratio) │                   │
│  └──────────────────────────────────────────────────┘                   │
│                          ↓                                              │
│  ┌──────────────────────────────────────────────────┐                   │
│  │ [Phase 5] (가중치·LR + Threshold) → 희귀 Listing   │                   │
│  │  Lambda(LLM): Orphanet LR 기반 임계치 통과한 희귀질환만 │              │
│  └──────────────────────────────────────────────────┘                   │
│                          ↓                                              │
│                   🌀 2차 Output (희귀 Listing + 일반 Ranking 확정)         │
└─────────────────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Private Subnet ③ — RAG → 최종 보고서 DB화                                │
│  API List(외부 지식) → Bedrock(LLM) RAG → 최종 보고서                     │
│  ↓                                                                       │
│  사용자 데이터 DB (DynamoDB) │ Case Report DB (rare-case-collection)      │
│  Aurora 환자맥 DB (FHIR R4 형태)│ 백업 DB (S3 + AWS Backup)                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 번호 변경 매핑 (기존 ↔ 회의 후)

| 회의 후 (화이트보드) | 기존 문서 | 비고 |
|----------------------|-----------|------|
| **Phase 1** Symptom (LLM-Bedrock) | Phase 1B (NLP Agent) | 환자 기본정보·문진 텍스트 → HPO |
| **Phase 2** X-ray (SooNet, SageMaker) | Phase 1A (Vision Agent) | X-ray → 14-class HPO% |
| **Phase 3** Multimodal-Scoring | Phase 1C (Lab) + Phase 2 집계 | lab + micro 데이터 → HPO + 추가 가중치 |
| **1차 Output → 일반 Ranking** | Phase 2A (LIRICAL) + 2B (일반) | 통합 점수 산출 |
| **Phase 4** 검증 | (신규) | LIRICAL ↔ 일반 score ratio 교차 검증 |
| **Phase 5** 희귀 Listing (LR+Threshold) | Phase 3 (RAG 트리거) 일부 | Orphanet 기반 LR 임계치 통과만 |
| **2차 Output → RAG → 최종 보고서** | Phase 4 (RAG) + Phase 5 (저장) | 외부 API + Bedrock Sonnet |

---

## 3. AWS 서비스 매핑 (상세)

### 3-1. 프론트엔드 배포

| 서비스 | 용도 | 비고 |
|--------|------|------|
| **CloudFront** | CDN + HTTPS | React 빌드 정적 파일 배포 |
| **S3** (정적 호스팅) | React 앱 호스팅 | `rare-link-ai-frontend` 버킷 |
| **Route 53** | 도메인 관리 | (선택) 커스텀 도메인 |

배포 흐름:
```
React Build (dist/) → S3 버킷 → CloudFront → 사용자 브라우저
```

### 3-2. API 계층

| 서비스 | 용도 | 비고 |
|--------|------|------|
| **API Gateway** | REST API 엔드포인트 | FHIR R4 호환 인터페이스 |
| **Lambda** (오케스트레이터) | 4-Phase 파이프라인 조율 | Python 3.12, 타임아웃 900초 (Bedrock + PubMed 포함 시 300초 부족) |
| **Lambda** (Phase 2/3) | HPO-LR 스코어링 | 메모리 1024MB |

API 엔드포인트 설계:
```
POST /api/v1/diagnose          — 전체 진단 파이프라인
POST /api/v1/xray/analyze      — X-ray 단독 분석
POST /api/v1/rare/screen       — 희귀질환 스크리닝 단독
GET  /api/v1/patient/{mrn}     — 환자 진단 이력 조회
GET  /api/v1/health            — 헬스체크
```

### 3-3. AI/ML 계층

| 서비스 | 용도 | 비고 |
|--------|------|------|
| **SageMaker Endpoint** | SooNet (DenseNet-121) 추론 | `ml.g4dn.xlarge`, 실시간 |
| **SageMaker Training** | 모델 학습/재학습 | `ml.g4dn.xlarge` |
| **Bedrock** (Claude Haiku) | 임상 텍스트 → HPO 추출 | `ap-northeast-2` |
| **Bedrock** (Claude Sonnet) | 진단 리포트 생성 | `ap-northeast-2` |

### 3-4. 데이터 계층 (회의 후 4-DB 구조로 명확화)

화이트보드 결정: **DB를 역할별로 4종 분리** — 사용자 데이터 / Case Report / 환자맥 / 백업

| 서비스 | 용도 | 비고 |
|--------|------|------|
| **DynamoDB** (사용자 데이터 DB) | 진단 이력 + 메타데이터 | `diagnosis-history`, 온디맨드 |
| **DynamoDB** (Case Report DB) | 희귀질환 RAG 결과 누적 | `rare-case-collection` (MLOps 트리거 소스) |
| **Aurora PostgreSQL** (환자맥 DB) | **FHIR R4 형태 환자 영구 DB** | Multi-AZ, EMR 연동용 (회의 신규 추가) |
| **S3** (백업 DB + 데이터) | X-ray, 모델 가중치, 학습 데이터, 백업 | `say2-2team-bucket` + AWS Backup 통합 |
| **로컬 파일** (Orphanet XML) | 희귀질환-HPO 매핑 사전 | `en_product4.xml` → CSV 변환 |

**왜 Aurora를 추가했나** (회의 핵심): DynamoDB(NoSQL)는 FHIR R4의 관계형 리소스
(Patient ↔ Observation ↔ DiagnosticReport ↔ ImagingStudy)를 표현하기 어려움.
Aurora PostgreSQL로 환자 영구 DB를 분리해서 EMR 연동성과 ACID 트랜잭션을 확보.

DynamoDB 테이블 설계:
```
[diagnosis-history] — 사용자 데이터 DB
  PK: patient_mrn · SK: diagnosis_timestamp
  Attributes: case_id, top_diseases, confidence, phase3_triggered, report_url

[rare-case-collection] — Case Report DB
  PK: disease_orpha_id · SK: case_id
  Attributes: hpo_codes, lab_summary, xray_findings, confirmed_diagnosis, collected_at
```

Aurora 테이블 설계 (FHIR R4 호환):
```
[patient]              — Patient 리소스 (MRN, 인구통계)
[observation]          — Observation (Lab, Vital Signs)
[diagnostic_report]    — 우리 진단 결과 (EMR로 전달)
[imaging_study]        — X-ray DICOM 참조 (S3 URL)
[case_history]         — 환자별 진단 이력 통합 뷰 (View)
```

### 3-4-1. EC2 FHIR 서버 (회의 신규 추가)

화이트보드: `FHIR → EC2`, `SQS → ALB`

| 서비스 | 용도 | 비고 |
|--------|------|------|
| **EC2** (FHIR Server) | HAPI FHIR 서버 호스팅 | t3.medium, Multi-AZ Auto Scaling Group |
| **ALB** | FHIR 트래픽 분산 | EC2 앞단, SSL 종료 |
| **SQS** | FHIR 요청 비동기 큐잉 | EMR 연동 트래픽 폭주 흡수 |

**왜 EC2가 필요한가**: HAPI FHIR(오픈소스 FHIR 서버)는 Java Stateful 애플리케이션이라
Lambda(Stateless)로 호스팅 불가. EC2 + ALB + Auto Scaling으로 표준 배포.
Aurora를 백엔드 DB로 사용 → FHIR 리소스를 그대로 저장/조회.

### 3-4-2. Subnet 분배 (회의 후 명확화)

| Subnet | 역할 | 배치 서비스 | 비고 |
|--------|------|------------|------|
| **Public** | 진입점, 큐잉 | NAT GW (×2 AZ), SQS, ALB | 외부 인터넷과 직접 통신 |
| **Private ①** | Phase 1·2·3 (1차 Output) | Lambda(Symptom), SageMaker(X-ray), Lambda(Multimodal) | 입력 처리 격리 |
| **Private ②** | 가중치~2차 Output | Lambda(검증), Lambda(LLM 희귀 listing) | 핵심 로직 격리 |
| **Private ③** | RAG → 최종 보고서 DB화 | Lambda(RAG), DynamoDB, Aurora, RDS pgvector | 데이터 저장 격리 |
| **MLOps Subnet** | 자동 재학습 | EventBridge, SageMaker Training, ECR | MLOps 격리 |

**Subnet을 단계별로 나눈 이유**: 보안 그룹을 단계별로 따로 적용해
Phase 1 Lambda가 직접 DB에 접근 못 하도록 계층 분리. 또 한 Subnet 장애가
다른 단계로 번지지 않게 하는 **Blast Radius 격리**.

### 3-5. RAG / 외부 API 연동

| 서비스/API | 용도 | 호출 방식 | 키 필요 |
|------------|------|-----------|---------|
| **PubCaseFinder API** | 희귀질환 유사 케이스 검색 (LIRICAL 교차검증) | Lambda → HTTPS | 불필요 (비영리 공개 API, 간헐적 다운 주의) |
| **PubMed E-utilities** | 최신 논문 검색 | Lambda → HTTPS | 불필요 (선택: NCBI API Key로 10 req/s 향상) |
| **Monarch Initiative** | HPO 코드 → 증상명 변환 (가독성 향상) | Lambda → HTTPS | 불필요 — **구현 완료** (`rag/monarch_fetcher.py`) |
| **ClinicalTrials.gov v2** | 희귀질환 임상시험 정보 | Lambda → HTTPS | 불필요 — **구현 완료** (`rag/clinicaltrials_fetcher.py`) |
| **Bedrock Knowledge Base** | RAG 벡터 검색 (선택) | Bedrock API | AWS 자격증명 |

---

## 4. 모델 자동 갱신 파이프라인 (CI/CD for ML)

회의에서 멘토님이 강조한 핵심 시나리오:
> "새 환자 데이터가 들어왔을 때, 희귀질환 DB에 축적 → 일정량 도달 시 자동 재학습 → 모델 갱신"

```
┌──────────────────────────────────────────────────────────────────┐
│                    모델 자동 갱신 파이프라인                        │
│                                                                    │
│  ① 새 케이스 수집                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐         │
│  │ 진단 결과 │───▶│  Lambda      │───▶│  DynamoDB        │         │
│  │ (확진 후) │    │  (수집 트리거)│    │  rare-case-      │         │
│  └──────────┘    └──────────────┘    │  collection      │         │
│                                       └────────┬─────────┘         │
│                                                │                   │
│  ② 배치 트리거 (일정량 도달 시)                    │                   │
│                                                ▼                   │
│                                       ┌──────────────────┐         │
│                                       │  EventBridge     │         │
│                                       │  (스케줄/조건)    │         │
│                                       └────────┬─────────┘         │
│                                                │                   │
│  ③ 재학습                                       ▼                   │
│                                       ┌──────────────────┐         │
│                                       │  SageMaker       │         │
│                                       │  Training Job    │         │
│                                       │  (G4DN.16XL)     │         │
│                                       └────────┬─────────┘         │
│                                                │                   │
│  ④ 모델 교체                                    ▼                   │
│                                       ┌──────────────────┐         │
│                                       │  SageMaker       │         │
│                                       │  Endpoint Update │         │
│                                       │  (Blue/Green)    │         │
│                                       └──────────────────┘         │
└──────────────────────────────────────────────────────────────────┘
```

구현 포인트:
- EventBridge Rule: `rare-case-collection` 테이블 아이템 수가 임계값(예: 100건) 도달 시 트리거
- SageMaker Training Job: S3에서 기존 데이터 + 새 데이터 로드 → 재학습
- Endpoint Update: Blue/Green 배포로 무중단 모델 교체

---

## 5. 온프레미스 버전 (현실 배포용)

회의 핵심: "포트폴리오는 AWS 풀스택, 현실은 온프레미스도 가능해야 한다"

```
┌─────────────────────────────────────────────────┐
│              온프레미스 서버 (병원 내부)            │
│                                                   │
│  ┌───────────┐  ┌───────────────────────────┐    │
│  │  Nginx    │  │  FastAPI (uvicorn)         │    │
│  │  Reverse  │──│  4-Phase Pipeline          │    │
│  │  Proxy    │  │  + SooNet Local Inference  │    │
│  └───────────┘  └─────────────┬─────────────┘    │
│                               │                   │
│  ┌───────────┐  ┌─────────────▼─────────────┐    │
│  │PostgreSQL │  │  모델 가중치 (로컬 파일)     │    │
│  │ (환자 이력)│  │  chexnet_unet_crop_best.pth│    │
│  └───────────┘  └───────────────────────────┘    │
│                                                   │
│  ┌───────────────────────────────────────────┐    │
│  │  FHIR Gateway (SMART on FHIR Client)      │    │
│  │  ↔ 병원 EMR 시스템 연동                     │    │
│  └───────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

현행 코드가 이미 환경변수 기반 백엔드 전환을 지원하므로:
- `LUNG_DX_XRAY_BACKEND=local` → GPU 서버에서 직접 추론
- `LUNG_DX_REPORT_BACKEND=template` → Bedrock 없이 로컬 템플릿
- PostgreSQL 또는 SQLite로 진단 이력 저장

---

## 6. 듀얼 아키텍처 비교표

| 구성 요소 | AWS 클라우드 버전 | 온프레미스 버전 |
|-----------|------------------|----------------|
| 프론트엔드 | S3 + CloudFront | Nginx 정적 호스팅 |
| API | API Gateway + Lambda | FastAPI (uvicorn) |
| X-ray 추론 | SageMaker Endpoint | 로컬 GPU (PyTorch) |
| 리포트 생성 | Bedrock Claude | 로컬 템플릿 |
| 환자 DB | DynamoDB | PostgreSQL |
| 이미지 저장 | S3 | 로컬 파일시스템 |
| 모델 재학습 | SageMaker Training | 로컬 GPU 학습 |
| EMR 연동 | API Gateway FHIR | FHIR Gateway |
| 모니터링 | CloudWatch | Prometheus + Grafana |

---

## 7. EMR 브릿징 전략

회의 핵심: "EMR을 대체하지 않고, API로 브릿징한다"

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  기존 EMR    │     │  Rare-Link AI    │     │  외부 API    │
│  (BESTCare,  │◀───▶│  API Gateway     │◀───▶│  PubMed,     │
│   의료정보,  │     │  (FHIR R4 호환)  │     │  PubCase-    │
│   이지케어)  │     │                  │     │  Finder 등   │
└──────────────┘     └──────────────────┘     └──────────────┘
       │                      │
       │    SMART on FHIR     │
       │    Patient Context   │
       └──────────────────────┘
```

SMART on FHIR 연동 포인트:
- **Patient** 리소스: 환자 기본 정보 (MRN, 나이, 성별)
- **Observation** 리소스: Lab 결과, Vital Signs
- **DiagnosticReport** 리소스: 우리 진단 결과를 EMR에 전달
- **ImagingStudy** 리소스: X-ray DICOM 참조

FHIR 클라이언트 라이브러리: [SMART Health IT](https://github.com/smart-on-fhir/client-js) (오픈소스)

---

## 8. 구현 우선순위 및 일정 제안

### Phase A: 이번 주 (04/27 ~ 05/01) — 핵심 파이프라인 완성

| 작업 | 담당 | AWS 서비스 |
|------|------|-----------|
| SooNet 최종 학습 (2~3만장 검증 → 풀데이터) | 배기태 | SageMaker Training (G4DN.16XL) |
| SageMaker Endpoint 배포 | 배기태/허태웅 | SageMaker Endpoint |
| API 확정 + Lambda 오케스트레이터 | 허태웅 | Lambda + API Gateway |
| 프론트엔드 진단 워크스페이스 | 박성수 | — |
| RAG API 확정 (PubCaseFinder + PubMed) | 공동 | — |

### Phase B: 다음 주 (05/04 ~ 05/10) — AWS 인프라 확장

| 작업 | 담당 | AWS 서비스 |
|------|------|-----------|
| DynamoDB 테이블 생성 (진단 이력 + 희귀 케이스) | 허태웅 | DynamoDB |
| S3 + CloudFront 프론트엔드 배포 | 박성수 | S3, CloudFront |
| Bedrock 연동 (HPO 추출 + 리포트) | 공동 | Bedrock |
| SageMaker ↔ Lambda 연결 테스트 | 배기태 | SageMaker, Lambda |

### Phase C: 05/11 ~ 05/17 — 통합 + 시나리오

| 작업 | 담당 | AWS 서비스 |
|------|------|-----------|
| 모델 자동 갱신 파이프라인 (최소 구현) | 배기태 | EventBridge, SageMaker |
| 전체 E2E 시나리오 테스트 | 전원 | 전체 |
| 아키텍처 다이어그램 최종 정리 | 공동 | — |

### Phase D: 05/18 ~ 05/24 — 발표 준비

| 작업 | 비고 |
|------|------|
| 시나리오 시연 녹화 | 새 환자 → 진단 → 희귀질환 발견 → DB 축적 → 모델 갱신 |
| 포트폴리오 아키텍처 다이어그램 | AWS 풀스택 + 온프레미스 듀얼 |
| 발표 자료 | 05/27 최종 발표 |

---

## 9. 비용 추정 (월간, 개발/테스트 기준)

| 서비스 | 사양 | 예상 비용 |
|--------|------|----------|
| SageMaker Endpoint | ml.g4dn.xlarge, 온디맨드 | ~$0.736/hr (필요 시만 켜기) |
| SageMaker Training | ml.g4dn.16xlarge | ~$7.34/hr (학습 시만) |
| Lambda | 1024MB, 월 1만 호출 | ~$2 |
| API Gateway | 월 1만 요청 | ~$0.035 |
| DynamoDB | 온디맨드, 소량 | ~$1 |
| S3 | 50GB | ~$1.15 |
| CloudFront | 소량 트래픽 | ~$1 |
| Bedrock (Haiku) | 월 1000 호출 | ~$5 |
| Bedrock (Sonnet) | 월 500 호출 | ~$15 |

팁: SageMaker Endpoint는 테스트 시에만 켜고, 발표 전에 미리 warm-up 해두세요.

---

## 10. 아키텍처 다이어그램 (발표용 전체 그림)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Rare-Link AI — AWS Architecture                     │
│                                                                                   │
│   ┌─────────┐                                                                    │
│   │ 의사    │──▶ CloudFront + S3 (React Frontend)                                │
│   │ (브라우저)│                    │                                               │
│   └─────────┘                    ▼                                               │
│                          API Gateway (FHIR R4)                                    │
│                                  │                                               │
│                    ┌─────────────┼─────────────┐                                 │
│                    ▼             ▼             ▼                                  │
│              ┌──────────┐ ┌──────────┐ ┌──────────┐                              │
│              │ Lambda   │ │ Lambda   │ │ Lambda   │                              │
│              │ Phase1   │ │ Phase2/3 │ │ Phase4   │                              │
│              │ 오케스트 │ │ HPO-LR   │ │ 리포트   │                              │
│              └────┬─────┘ └────┬─────┘ └────┬─────┘                              │
│                   │            │            │                                     │
│         ┌─────────▼──┐   ┌────▼────┐  ┌────▼────────┐                           │
│         │ SageMaker  │   │DynamoDB │  │  Bedrock    │                            │
│         │ Endpoint   │   │(진단이력)│  │  Claude     │                            │
│         │ (SooNet)   │   │(희귀DB) │  │  Sonnet     │                            │
│         └────────────┘   └─────────┘  └─────────────┘                            │
│                                                                                   │
│   ┌───────────────────── 모델 자동 갱신 ──────────────────────┐                   │
│   │                                                            │                  │
│   │  DynamoDB Stream → EventBridge → SageMaker Training        │                  │
│   │                                  → Endpoint Update         │                  │
│   └────────────────────────────────────────────────────────────┘                  │
│                                                                                   │
│   ┌───────────────────── 외부 API 연동 ──────────────────────┐                    │
│   │  PubCaseFinder │ PubMed │ ClinicalTrials.gov │ Monarch Initiative │
│   └──────────────────────────────────────────────────────────┘                    │
│                                                                                   │
│   ┌───────────────────── EMR 브릿징 ─────────────────────────┐                    │
│   │  SMART on FHIR ↔ API Gateway ↔ 기존 EMR 시스템           │                    │
│   └──────────────────────────────────────────────────────────┘                    │
│                                                                                   │
│   ┌───────────────────── 데이터 레이크 ──────────────────────┐                    │
│   │  S3 (이미지/모델) │ DynamoDB (메타) │ Redshift (분석용)   │                    │
│   └──────────────────────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. 회의에서 나온 추가 아이디어 (향후 과제)

### 11-1. 사전 문진 시스템 (LLM 기반 대화형)
- 환자가 내원 전 링크를 통해 증상 입력
- 챗봇 형태로 대화하며 HPO 코드 자동 추출
- 70대 이하: 텍스트 기반, 80대 이상: 보호자 지원 또는 음성 입력
- 이 데이터가 Phase 2 입력으로 직접 활용 → 진단 정확도 향상
- "없는 증상"으로 소거법 적용 가능 (의사 차팅에는 없는 정보)

### 11-2. 데이터 레이크 구축
- S3: 이미지, 모델 가중치 (비정형)
- DynamoDB: 환자 메타데이터, 진단 이력 (반정형)
- Redshift: 장기 분석용 데이터 웨어하우스 (포트폴리오 그림용)
- 현실적으로는 DynamoDB + S3만으로 충분

### 11-3. PubCaseFinder API 리스크 관리
- DBCLS(일본 생명과학통합데이터베이스센터) 운영, 비영리 연구용
- 회원가입 없이 무료 API 제공 중 → 트래픽 급증 시 차단 가능성
- **현재 간헐적 502 Bad Gateway 확인됨** (2026-04-27 기준) — 서버 불안정
- 대응: 질환 리스트 기반 사전 캐싱 + PubMed / Monarch 크로스체크로 폴백
- 상용화 시에는 별도 협의 필요

### 11-4. HPO 코드 검증 체계
- CheXpert 14개 라벨 → HPO 코드 매핑은 Monarch Initiative API로 검증
- `soo_net.py` hpo_map: Monarch API로 검증 완료 (2026-04-27)
  - 수정: HP:0002095→HP:0100750, HP:0002111→HP:0100598, HP:0034251→HP:0034501, HP:0025000→HP:0033608
- HPO 코드 추가/변경 시 반드시 `https://api.monarchinitiative.org/v3/api/entity/{HPO_ID}` 로 재확인

---

## 12. AWS 아키텍처 초보자용 설명

### AWS가 뭔가요?

AWS(Amazon Web Services)는 아마존이 운영하는 클라우드 서비스예요.
쉽게 말하면 **"인터넷으로 빌려 쓰는 컴퓨터 + 저장소 + 소프트웨어"** 묶음이에요.

우리가 직접 서버를 사서 병원에 설치하는 대신, AWS에서 필요한 만큼만 빌려서 쓰는 거예요.
쓴 만큼만 돈 내고, 안 쓸 때는 꺼두면 돼요.

---

### 우리 시스템에서 각 AWS 서비스가 하는 일

#### 1단계: 의사가 브라우저에서 접속하면

```
의사가 주소 입력
    ↓
Route 53 — 전화번호부. "이 주소가 어디 있는지" 알려줌
    ↓
CloudFront — 전 세계 어디서 접속해도 빠르게 보여주는 CDN
    ↓
S3 (정적 호스팅) — React로 만든 웹 화면 파일이 저장된 곳
    ↓
의사 화면에 웹앱 표시
```

#### 2단계: 의사가 환자 데이터를 입력하면

```
API Gateway — 우리 시스템의 정문. 외부 요청을 받아서 안으로 전달
    ↓
WAF — 해킹 시도 차단 (방화벽)
Cognito — 로그인 확인 (인증)
    ↓
Lambda (오케스트레이터) — 각 단계를 순서대로 실행하는 지휘자
```

**Lambda가 뭔가요?**
Lambda는 "코드를 실행하는 서버인데, 서버를 직접 관리 안 해도 되는 것"이에요.
우리 `rag_pipeline.py`가 Lambda 위에서 실행된다고 생각하면 돼요.
요청이 올 때만 켜지고, 끝나면 자동으로 꺼져요. 그래서 비용이 저렴해요.

#### 3단계: 5-Phase 진단 파이프라인이 실행되는 곳 (회의 후 확정)

```
[Subnet ①] Phase 1·2·3 — 1차 Output 생성 (입력 → HPO 변환)

  Phase 1 — Symptom (LLM-Bedrock)
    환자 기본정보 + 증상(문진 텍스트) → Bedrock Claude Haiku
    → HPO 코드(JSON) + 근거 정리 (왜 이 HPO인지)

  Phase 2 — X-ray (SooNet, SageMaker)
    X-ray 이미지 → SageMaker Endpoint (DenseNet-121 GPU)
    → 14개 HPO 확률값(JSON)

  Phase 3 — Multimodal-Scoring (Lab + Micro)
    혈액검사 + 미생물 검사 → Lambda (Rule-based)
    → HPO + 추가 가중치 데이터

  → 1차 Output (HPO 통합 + 가중치 적용)

[Subnet ②] 가중치 ~ 2차 Output (검증·희귀 분리)

  일반 Ranking — 가중치 기반 점수로 폐질환 후보 정렬

  Phase 4 — 검증
    Lambda: LIRICAL(희귀) ↔ 일반 score ratio 교차 검증
    → 일반 Ranking (확정)

  Phase 5 — 희귀 Listing (LR + Threshold)
    Lambda(LLM): Orphanet LR 임계치 통과한 희귀질환만 추출
    → 2차 Output

[Subnet ③] RAG → 최종 보고서

  API List → Bedrock(LLM) RAG
    → PubCaseFinder, PubMed, ClinicalTrials.gov, Monarch Initiative
    → Bedrock Claude Sonnet으로 최종 진단 보조 리포트 생성
    → DB 저장 (사용자 데이터 / Case Report / Aurora 환자맥)
```

**왜 단계가 늘어났나** (회의 후 변경): 기존엔 "입력 3개 병렬 → 점수 → RAG"였는데,
임상 현실에서 "검증 단계(Phase 4)"와 "희귀질환만 분리하는 단계(Phase 5)"가 따로
필요하다는 멘토 피드백 반영. 일반 폐질환 진단과 희귀질환 진단을 명확히 분리.

#### 4단계: 데이터가 저장되는 곳 (회의 후 4-DB 구조)

```
사용자 데이터 DB (DynamoDB diagnosis-history)
    → 환자별 진단 이력 — "이 환자 지난번 결과는?" 조회

Case Report DB (DynamoDB rare-case-collection)
    → 희귀질환 RAG 결과 누적 — 100건 쌓이면 자동 재학습 트리거

Aurora 환자맥 DB (PostgreSQL Multi-AZ, FHIR R4 형태)
    → EMR과 연동 가능한 환자 영구 저장소
    → 회의 신규 추가: NoSQL(DynamoDB)로는 FHIR 관계 표현이 어려워서

백업 DB (S3 + AWS Backup)
    → X-ray 이미지, 모델 가중치(.pth), 학습 데이터, 일별 스냅샷

ElastiCache (Redis) — Phase 1·2·3 결과 임시 버퍼
    → Step Functions가 3개 에이전트 완료를 동기화하는 데 사용
```

⊕ 환자맥 DB + Case Report DB → **History 통합 뷰**: 의사가
"이 환자 케이스를 다시 리뷰해야 할 때" 두 DB를 조인해서 시간순 보여줌.

#### 4-1단계: EMR 연동 (FHIR Gateway)

```
병원 EMR (BESTCare 등) ──▶ ALB ──▶ EC2 (HAPI FHIR Server) ──▶ Aurora
                                       ↑
                                       │ SQS (트래픽 폭주 흡수)
                                       │
                          API Gateway ─┘
```

**EC2가 등장한 이유**: FHIR 서버는 Java Stateful 앱이라 Lambda로 못 돌림.
EC2 + Auto Scaling Group으로 띄우고, ALB가 트래픽을 분산.
SQS는 EMR이 한꺼번에 많이 호출할 때 큐로 흡수해서 EC2가 안 죽게 함.

#### 5단계: 모델이 자동으로 업데이트되는 과정

```
새 환자 진단 결과 → DynamoDB에 저장
    ↓ (100건 쌓이면)
EventBridge — 조건 충족 시 자동으로 다음 단계 실행하는 타이머/트리거
    ↓
SageMaker Training Job — 새 데이터로 SooNet 모델 재학습
    ↓
S3에 새 모델 가중치 저장
    ↓
SageMaker Endpoint 업데이트 — 무중단으로 새 모델로 교체 (Blue/Green)
```

#### 6단계: 외부 API들은 어디서 호출되나요?

Lambda 안에서 직접 HTTP 요청으로 호출해요.

```
Lambda
  ├── PubCaseFinder API → 희귀질환 매칭 (서버 장애 시 로컬 폴백)
  ├── PubMed API → 최신 논문 3편
  ├── ClinicalTrials.gov API → 모집 중 임상시험 3건
  └── Monarch Initiative API → HPO 코드 → 증상명 변환
```

#### 7단계: 모니터링

```
CloudWatch — 모든 Lambda, SageMaker, API Gateway 로그를 한 곳에서 봄
    → 에러 나면 알림 오게 설정 가능
    → 비용 얼마 쓰는지도 여기서 확인

AWS IAM — 누가 어떤 서비스에 접근할 수 있는지 권한 관리
    → Lambda가 S3에 접근하려면 IAM 권한 필요
    → 보안의 핵심
```

---

### 지금 코드 vs AWS 배포 — 뭐가 다른가요?

| 지금 코드 (로컬) | AWS 배포 시 |
|-----------------|-------------|
| `python rag_pipeline.py` 직접 실행 | Lambda가 자동으로 실행 |
| 로컬 파일에 결과 저장 | DynamoDB에 저장 |
| 터미널에 print() 출력 | CloudWatch에 로그 저장 |
| 인증 없음 | Cognito 로그인 필요 |
| 한 번에 하나씩 처리 | 여러 환자 동시 처리 가능 |
| 서버 항상 켜져 있어야 함 | Lambda는 요청 올 때만 켜짐 |
| ChromaDB 없음 (제거됨) | ElastiCache Redis로 캐싱 |

지금은 "내 컴퓨터에서 돌아가는 프로토타입"이고,
AWS 아키텍처는 "실제 병원에서 수천 명 환자를 처리할 수 있는 서비스"예요.

---

## 13. 태그 규칙 (필수)

모든 AWS 리소스 생성 시:
```
Tag: project = pre-{서비스명}-2-2-team
Region: ap-northeast-2
```

예시:
- SageMaker: `pre-sagemaker-2-2-team`
- Lambda: `pre-lambda-2-2-team`
- DynamoDB: `pre-dynamodb-2-2-team`
- S3: `pre-s3-2-2-team`

---

## 13-1. 회의 후 변경사항 요약 (2026-05-04 화이트보드)

| 항목 | 변경 전 | 변경 후 (회의 결정) | 이유 |
|------|---------|---------------------|------|
| **Phase 번호** | 1A·1B·1C → 2 → 3 → 4 → 5 | 1·2·3 → 1차 Out → 4(검증) → 5(희귀 분리) → 2차 Out | 임상 흐름 명확화 |
| **DB 구조** | DynamoDB 2개 + S3 | 사용자 + Case + Aurora(FHIR) + 백업 (4종) | EMR 연동 + ACID 필요 |
| **FHIR 서버** | API Gateway 직접 처리 | EC2(HAPI FHIR) + ALB + SQS | Stateful 서버 필요 |
| **Subnet 분배** | 단일 Private Subnet | Private ①·②·③ 단계별 분리 | Blast Radius 격리 |
| **검증 단계** | (없음) | Phase 4 cross-check | 일반 vs 희귀 ratio 검증 |
| **희귀 분리** | RAG 트리거 안에 혼재 | Phase 5 LR+Threshold 독립 | 임계치 명시화 |

**완성을 위한 To-Do 우선순위**:

1. **즉시** — Subnet ①·②·③ VPC 재구성 (CDK/Terraform 코드 작성)
2. **이번 주** — Aurora PostgreSQL 인스턴스 생성 + FHIR 스키마 설계
3. **다음 주** — EC2 HAPI FHIR 배포 (Auto Scaling, ALB 연동)
4. **5/11~** — Phase 4(검증) Lambda + Phase 5(LR threshold) Lambda 분리 구현
5. **5/18~** — DB 4종 통합 테스트 + History 뷰 구현
6. **발표 전** — 전체 E2E 시연 녹화 (5-Phase 흐름)

---

## 14. 요약

이 아키텍처는 두 가지 버전을 동시에 제공합니다:

1. **AWS 클라우드 버전** (포트폴리오): CloudFront → API Gateway → Lambda → SageMaker/Bedrock/DynamoDB 풀스택. 모델 자동 갱신 파이프라인 포함. 발표 시 "이 시스템은 AWS 환경에서 이렇게 확장 가능합니다"를 보여주는 그림.

2. **온프레미스 버전** (현실 배포): FastAPI + 로컬 GPU + PostgreSQL. 환경변수 전환만으로 동일 파이프라인 동작. "병원 내부에 설치해서 바로 쓸 수 있습니다"를 보여주는 실체.

핵심 차별점은 **EMR 브릿징 API Gateway**입니다. 어떤 EMR 시스템이든 FHIR R4 호환 API를 통해 연동 가능하므로, 기존 EMR 업체와 경쟁이 아닌 협력 관계를 만들 수 있습니다.

---

## 15. 아키텍처 구현을 위한 실행 전략 (How to Implement)

초기 프로토타입을 넘어 상용화 수준으로 AWS 인프라를 구축하려면 다음 단계로 접근해야 합니다.

### 15-1. IaC (Infrastructure as Code) 기반 인프라 구축

VPC, 프라이빗 서브넷, 보안 그룹, VPC 엔드포인트 등 복잡한 네트워크 환경을 수동으로 구성하면 재현성이 떨어집니다. AWS CDK나 Terraform을 사용하여 인프라를 코드로 정의하고 배포해야 합니다.

```
# AWS CDK 예시 구조
cdk/
├── app.py
├── stacks/
│   ├── vpc_stack.py          # VPC, 서브넷, NAT GW, VPC Endpoint
│   ├── api_stack.py          # API Gateway, Lambda, Cognito
│   ├── ml_stack.py           # SageMaker Endpoint, ECR
│   ├── data_stack.py         # DynamoDB, S3, ElastiCache
│   └── mlops_stack.py        # EventBridge, SageMaker Training
```

### 15-2. API 계층 및 EMR 연동 로직 구현

API Gateway를 통해 들어오는 요청을 처리할 때, SMART on FHIR 규격(Patient, Observation, DiagnosticReport 리소스 등)에 맞춰 데이터를 변환하는 인터페이스(어댑터) 계층을 Lambda에 구현해야 합니다.

```python
# Lambda FHIR 어댑터 예시
def fhir_to_pipeline_input(fhir_bundle: dict) -> dict:
    patient = fhir_bundle.get("Patient", {})
    observations = fhir_bundle.get("Observation", [])
    return {
        "symptom_text": extract_clinical_notes(observations),
        "lab_results":  extract_lab_values(observations),
        "patient_mrn":  patient.get("id"),
    }
```

### 15-3. 자동화된 MLOps 파이프라인 구축

DynamoDB `rare-case-collection` 테이블에 새 케이스가 축적되면 EventBridge가 이를 감지하여 SageMaker Training Job을 트리거하는 구조를 만들어야 합니다. 이 파이프라인은 최종적으로 다운타임 없이(Zero-downtime) Blue/Green 배포 방식으로 SageMaker Endpoint를 업데이트하도록 구성해야 합니다.

```
DynamoDB Stream → EventBridge Rule (100건 도달 시)
    → SageMaker Training Job (새 데이터 + 기존 데이터)
    → S3 모델 가중치 저장
    → SageMaker Endpoint Blue/Green 업데이트 (Downtime 0초)
```

---

## 16. 현재 아키텍처의 한계점 및 설계 개선안 (Critique & Polish)

논문 게재나 실제 의료 현장 도입을 위해서는 시스템의 안정성(Fault Tolerance)과 데이터 보안 측면에서 다음 사항들을 반드시 보완해야 합니다.

| 한계점 (Risk) | 원인 및 현상 | 설계 개선안 (Solution) |
|--------------|-------------|----------------------|
| **외부 API 장애 취약성** | PubCaseFinder API는 비영리 공개 API로 간헐적인 502 Bad Gateway 에러가 발생합니다. 외부 API가 멈추면 파이프라인 전체가 실패할 위험이 있습니다. | **Circuit Breaker 패턴 및 캐싱 도입**: 외부 API 호출 실패 시 즉시 로컬 Orphanet CSV 내부 검색 결과만으로 폴백(Fallback)하는 로직을 구현해야 합니다. 또한, ElastiCache(Redis)를 단순 HPO 버퍼가 아닌 API 응답 캐싱용으로도 적극 활용해야 합니다. |
| **네트워크 병목 및 지연** | Step Functions(오케스트레이터)는 VPC 외부에 위치하고, 작업을 수행하는 Lambda들은 VPC 프라이빗 서브넷 내부에 있습니다. 이로 인한 통신 오버헤드와 ENI(탄력적 네트워크 인터페이스) 콜드 스타트가 발생할 수 있습니다. | **Express Workflows 전환**: 응답 속도가 중요한 실시간 진단 API이므로, Step Functions를 Standard 대신 지연 시간이 짧은 Express Workflows로 구성하거나, 오케스트레이션 역할을 전담하는 VPC 내부의 마스터 Lambda를 두어 최적화해야 합니다. |
| **개인정보보호(HIPAA) 위반 위험** | MLOps 재학습을 위해 DynamoDB에 환자의 진단 결과와 MRN(환자 등록 번호)을 원본 그대로 축적하는 구조입니다. | **데이터 비식별화 파이프라인 추가**: 진단 이력을 `rare-case-collection` 테이블로 넘기기 전에, 환자 식별 정보를 익명화(De-identification) 처리하는 Lambda 함수를 파이프라인 중간에 반드시 추가해야 법적 문제를 방지할 수 있습니다. |
| **단순 벡터 검색의 한계** | RDS pgvector를 사용하여 희귀질환을 의미 기반(Vector)으로 검색합니다. 그러나 의료 도메인에서는 특정 유전자 기호나 HPO 코드가 정확히 일치해야 하는 경우가 많습니다. | **하이브리드 검색(Hybrid Search) 적용**: 벡터 유사도 검색과 키워드 정확도 일치(Exact Match) 검색을 결합하여, 두 점수를 혼합해 문서의 연관성을 평가하는 하이브리드 검색 방식으로 고도화해야 논문 수준의 검색 정확도(Retrieval Accuracy)를 입증할 수 있습니다. |


---

## 17. 보안 강화 설계 (Security Hardening)

### 17-1. 자격 인증 및 권한 부여

| 서비스 | 적용 여부 | 용도 |
|--------|----------|------|
| **AWS Secrets Manager** | ✅ 필수 | Bedrock API 키, NCBI API Key 등 Lambda 환경변수 대체. 런타임에 동적 로드 |
| **AWS KMS** | ✅ 필수 | S3 SSE-KMS (X-ray 이미지, 모델 가중치), DynamoDB 암호화 키 관리 |
| **MFA** | 운영 정책 | AWS 콘솔 접근 IAM 계정에 적용. 아키텍처 다이어그램 범위 외 |

Secrets Manager 적용 예시:
```python
# Lambda에서 런타임에 키 로드 (환경변수 하드코딩 대체)
import boto3
client = boto3.client('secretsmanager', region_name='ap-northeast-2')
secret = client.get_secret_value(SecretId='rare-link/ncbi-api-key')
```

### 17-2. VPC 보안 — Security Group 설계

Security Group만으로 충분 (NACL은 불필요). Security Group은 Stateful이라 응답 트래픽 자동 허용.

```
Lambda SG:
  Outbound: HTTPS(443) → 0.0.0.0/0  (NAT GW 경유 외부 API)
  Outbound: 443 → SageMaker Endpoint SG
  Outbound: 443 → ElastiCache SG (Redis)

SageMaker Endpoint SG:
  Inbound:  443 → Lambda SG만 허용
  Outbound: S3 VPC Endpoint만

ElastiCache SG:
  Inbound:  6379 → Lambda SG만 허용
```

### 17-3. 감사 로그 — CloudTrail

CloudTrail은 필수. 의료 데이터 접근 감사 로그 (누가 언제 어떤 환자 데이터에 접근했는지 추적). HIPAA 요건 충족.

- AWS Config는 이 프로젝트 규모에서 오버스펙. 상용화 단계에서 추가.

### 17-4. 탐지 제어

| 서비스 | 권장 | 이유 |
|--------|------|------|
| **GuardDuty** | ✅ 추가 | 비정상 API 호출, 자격증명 탈취 시도 탐지. 비용 저렴, 켜두기만 하면 됨 |
| **Security Hub** | ✅ 추가 | GuardDuty + CloudTrail 결과 통합 대시보드 |
| **Macie** | 선택 | S3 환자 데이터 PII 자동 탐지. 포트폴리오 그림용으로 언급 가능 |

### 17-5. 네트워크 보안

| 서비스 | 결론 |
|--------|------|
| **AWS WAF** | ✅ 이미 적용 (CloudFront + API Gateway 이중 보호) |
| **Shield Standard** | ✅ 자동 적용 (모든 AWS 계정 무료) |
| **Shield Advanced** | ❌ 불필요 (월 $3,000, 오버스펙) |
| **Firewall Manager** | ❌ 불필요 (단일 계정 프로젝트) |

### 17-6. 데이터 보호 — 암호화

저장 중 암호화(Encryption at Rest)와 전송 중 암호화(Encryption in Transit) 모두 적용 필수.

| 대상 | 방식 |
|------|------|
| S3 (X-ray, 모델 가중치) | SSE-KMS |
| DynamoDB (진단 이력) | AWS 관리형 키 암호화 활성화 |
| API Gateway → Lambda | TLS 1.2+ (기본 적용) |
| Lambda → SageMaker | TLS (VPC 내부 통신 강제) |
| Lambda → Bedrock | TLS (VPC PrivateLink) |

---

## 18. 컴퓨트 복원력 및 가용성

### 18-1. AWS Systems Manager

- **Parameter Store**: Secrets Manager 보완용 설정값 관리 (비용 무료 티어)
- **Session Manager**: EC2/SageMaker 접근 시 SSH 키 없이 접근 → 보안 향상

### 18-2. 가용성 — Multi-AZ

ALB/NLB는 현재 Lambda 기반 아키텍처에 불필요 (API Gateway가 자동 관리).

추가할 것:
- **DynamoDB Multi-AZ**: 기본 활성화 (온디맨드 모드)
- **ElastiCache Multi-AZ**: Redis 클러스터 모드 활성화
- **NAT Gateway 이중화**: AZ1 + AZ2 각각 배치 (이미 아키텍처에 반영됨)

### 18-3. 데이터 복원

| 서비스 | 적용 대상 | 효과 |
|--------|----------|------|
| **S3 Versioning** | 모델 가중치 버킷 | 실수로 덮어쓴 .pth 파일 복구 |
| **DynamoDB PITR** | diagnosis-history, rare-case-collection | 35일 이내 임의 시점 복구 |
| **AWS Backup** | S3 + DynamoDB 통합 | 중앙 백업 관리 |

S3 Lifecycle Policy (스토리지 비용 최적화):
```
현재 모델 가중치    → S3 Standard
이전 모델 버전      → S3 Standard-IA (30일 후 자동 전환, 30% 절감)
학습 데이터 아카이브 → S3 Glacier Instant Retrieval (90일 후, 재학습 시만 접근)
X-ray 이미지 (1년+) → S3 Glacier (장기 보관, 90% 절감)
```

---

## 19. 비동기 처리 — SQS / SNS

### 19-1. SQS 적용 포인트

현재 Lambda 오케스트레이터가 동기 방식으로 4-Phase 순차 실행 → 트래픽 몰릴 때 Lambda 타임아웃(900초) 위험.

```
진단 요청 → API Gateway → SQS Queue → Lambda (비동기 처리)
                                    ↓
                          SageMaker Training Job 트리거 (MLOps)
```

- SQS Dead Letter Queue(DLQ) 추가: 실패한 요청 재처리 가능

### 19-2. SNS 알림 연동

```
CloudWatch Alarm → SNS Topic → 이메일/Slack
  - Lambda 에러율 > 5%
  - SageMaker Endpoint 응답 지연 > 30초
  - DynamoDB 쓰기 실패
  - 모델 재학습 완료 알림
```

---

## 20. 재해 복구 전략

**권장: Pilot Light**

이 프로젝트 성격(포트폴리오 + 의료 보조 시스템)에 가장 적합.

| 전략 | RPO/RTO | 비용 | 적합성 |
|------|---------|------|--------|
| Backup & Restore | 시간 단위 | 최저 | ❌ 의료 시스템에 부적절 |
| **Pilot Light** | 분 단위 | 낮음 | ✅ 권장 |
| Warm Standby | 분 단위 | 중간 | △ 비용 2배 |
| Multi-Site Active/Active | ≈0 | 최고 | ❌ 오버스펙 |

Pilot Light 구성:
```
메인 리전: ap-northeast-2 (서울) — 풀 운영
파일럿:    ap-southeast-1 (싱가포르) — 핵심만 최소 실행
  - S3 크로스 리전 복제 (모델 가중치, X-ray)
  - DynamoDB Global Tables (진단 이력)
  - 재해 시: Lambda + API Gateway 스케일 업 → 수분 내 복구
```

---

## 21. SageMaker 인스턴스 및 S3 스토리지 클래스 상세

### 21-1. SageMaker 인스턴스 유형

| 용도 | 인스턴스 | 비용/hr | 선택 이유 |
|------|---------|---------|----------|
| 실시간 추론 (SooNet) | ml.g4dn.xlarge | $0.736 | T4 GPU, 비용/성능 균형 최적 |
| 모델 학습 (DenseNet-121) | ml.g4dn.16xlarge | $7.34 | 학습 시만 사용, 빠른 완료 |
| 배치 추론 (선택) | ml.g4dn.2xlarge | $1.47 | 비용 절감용 |

팁: Endpoint는 발표 전날 warm-up, 평소에는 꺼두기.

### 21-2. S3 스토리지 클래스

| 데이터 | 클래스 | 전환 시점 |
|--------|--------|----------|
| 현재 모델 가중치 | S3 Standard | — |
| 이전 모델 버전 | S3 Standard-IA | 30일 후 자동 |
| 학습 데이터 아카이브 | S3 Glacier Instant Retrieval | 90일 후 자동 |
| X-ray 이미지 (최근 1년) | S3 Standard | — |
| X-ray 이미지 (1년 이상) | S3 Glacier | 365일 후 자동 |

---

## 22. 데이터 분석 파이프라인 (포트폴리오 확장)

현재 프로젝트 규모에서 Kinesis, Glue, Redshift는 실제 구현 불필요. 포트폴리오 그림에만 언급.

| 서비스 | 실제 구현 | 포트폴리오 언급 | 이유 |
|--------|----------|----------------|------|
| **Glue** | ❌ | ✅ | DynamoDB → Redshift ETL용. 현재 데이터 규모 불필요 |
| **Redshift** | ❌ | ✅ | 장기 분석용 DW. DynamoDB + S3로 충분 |
| **QuickSight** | 선택 | ✅ | 진단 통계 대시보드. 데모용으로 구현 가능 |
| **Kinesis** | ❌ | ✅ | 실시간 스트리밍 불필요 (환자 1명씩 처리 구조) |

---

## 23. AWS 비용 관리

| 도구 | 용도 | 우선순위 |
|------|------|---------|
| **AWS Budgets** | 월 예산 설정 + 임계값 초과 시 이메일 알림 | ✅ 즉시 설정 (과금 폭탄 방지) |
| **Cost Explorer** | 서비스별 비용 분석, SageMaker 과금 추적 | ✅ 주기적 확인 |
| **Savings Plans** | SageMaker 1년 약정 시 최대 64% 절감 | 상용화 단계에서 고려 |

AWS Budgets 설정 권장값 (개발/테스트 기준):
```
월 예산: $100
알림 임계값: 80% ($80) → 이메일 발송
SageMaker Endpoint: 테스트 시에만 켜기 (시간당 $0.736)
```

---

## 24. 최종 우선순위 요약

### 반드시 추가 (보안/안정성 필수)
- Secrets Manager — API 키 관리
- KMS — S3/DynamoDB 암호화
- CloudTrail — 감사 로그
- VPC Endpoint (S3, DynamoDB Gateway + Bedrock Interface)
- DynamoDB PITR + S3 Versioning
- AWS Budgets — 과금 방지
- Security Group 명시적 설계

### 추가하면 완성도 향상 (포트폴리오 가치)
- GuardDuty + Security Hub
- SQS (비동기 파이프라인 + DLQ)
- SNS (CloudWatch 알람 연동)
- S3 Lifecycle Policy (스토리지 클래스 자동 전환)
- Pilot Light DR 전략 명시

### 포트폴리오 그림에만 언급 (실제 구현 불필요)
- Redshift + QuickSight + Glue
- Kinesis (향후 확장)
- Macie

### 넣지 않아도 됨
- Shield Advanced, Firewall Manager
- NACL (Security Group으로 충분)
- ALB/NLB (Lambda 기반 아키텍처)
- Transit Gateway, Direct Connect
- Multi-Site Active/Active DR


---

## 25. 보안 탐지 체인 및 RDS 고가용성 (추가 검토)

### 25-1. 보안 탐지 체인 전체 흐름

```
탐지: GuardDuty → 조사: Detective → 스캔: Inspector → 통합: Security Hub → 차단: WAF/Shield → 관리: Firewall Manager
```

| 서비스 | 이 프로젝트 적합성 | 결론 | 이유 |
|--------|-----------------|------|------|
| **GuardDuty** | ✅ 높음 | 적용 | 비정상 API 호출, 자격증명 탈취 탐지. 비용 저렴 |
| **Detective** | ✅ 중간 | 권장 | GuardDuty 탐지 후 침해 원인 심층 조사. 의료 데이터 침해 사고 분석에 유용 |
| **Inspector** | ✅ 중간 | 권장 | Lambda 함수 CVE 취약점 자동 스캔. 코드 의존성 취약점 탐지 |
| **Security Hub** | ✅ 높음 | 적용 | GuardDuty + CloudTrail + Inspector 결과 통합 대시보드 |
| **WAF** | ✅ 높음 | 이미 적용 | CloudFront + API Gateway 이중 보호 |
| **Shield Standard** | ✅ 자동 | 이미 적용 | 무료, 모든 계정 자동 적용 |
| **Firewall Manager** | ❌ 낮음 | 불필요 | 다중 계정/리전 통합 관리용. 단일 계정 프로젝트에 오버스펙 |

### 25-2. Detective vs GuardDuty 차이

- **GuardDuty**: "이상한 일이 일어나고 있다"를 탐지 (경보기)
- **Detective**: "왜, 어떻게 일어났는지"를 조사 (수사관)

GuardDuty가 "Lambda에서 비정상 S3 접근 감지"를 알리면, Detective가 해당 요청의 전체 경로(어떤 IAM 역할 → 어떤 API 호출 → 어떤 리소스 접근)를 시각화해서 보여줌.

의료 데이터 침해 사고 발생 시 원인 파악에 필수적. 포트폴리오에서 "보안 사고 대응 체계"를 보여주기 좋음.

### 25-3. Inspector 적용 범위

현재 아키텍처에서 Inspector가 스캔할 수 있는 대상:
- Lambda 함수: Python 패키지 의존성 CVE 취약점 (boto3, requests, numpy 등)
- ECR 이미지: SageMaker 학습용 Docker 이미지 취약점

EC2가 없어서 OS 레벨 스캔은 해당 없지만, Lambda + ECR 스캔만으로도 충분한 가치.

### 25-4. RDS Multi-AZ / Read Replica

현재 아키텍처에 RDS PostgreSQL(pgvector)이 이미 있으므로 이 설정은 필수급.

**Multi-AZ (필수)**
```
Primary RDS (AZ1, ap-northeast-2a)
    ↓ 동기 복제
Standby RDS (AZ2, ap-northeast-2c)
    → Primary 장애 시 자동 페일오버 (약 60~120초)
    → 진단 파이프라인 중단 없이 복구
```

**Read Replica (권장)**
```
Primary RDS → 쓰기 전용 (임베딩 업데이트, 새 질환 추가)
Read Replica → 읽기 전용 (RAG 벡터 검색 쿼리)
    → Phase 4 RAG Lambda가 Read Replica에만 접근
    → Primary 부하 분산 + 검색 응답 속도 향상
```

비용 참고:
- Multi-AZ: 단일 인스턴스 대비 약 2배 비용 (db.t3.medium 기준 ~$0.068/hr → ~$0.136/hr)
- Read Replica: 추가 인스턴스 비용 발생 (포트폴리오 발표 시에는 꺼두고 그림에만 표시 가능)

### 25-5. 최종 보안 탐지 체인 적용 결정

**HTML 다이어그램에 추가 (포트폴리오 완성도)**:
- RDS Multi-AZ + Read Replica → 이미 추가됨
- Detective → Security 상단 바에 추가됨
- Inspector → Security 상단 바에 추가됨

**실제 구현 우선순위**:
1. GuardDuty + Security Hub (즉시, 비용 저렴)
2. RDS Multi-AZ (RDS 생성 시 옵션 체크만 하면 됨)
3. Inspector (Lambda 배포 후 자동 스캔)
4. Detective (GuardDuty 활성화 후 추가)
5. Read Replica (발표 후 상용화 단계)

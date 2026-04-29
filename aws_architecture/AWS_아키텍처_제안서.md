# Rare-Link AI — AWS 아키텍처 제안서

작성일: 2026-04-27
근거: 팀 회의 녹음 (2026-04-27) + 현행 코드베이스 분석

---

## 1. 핵심 설계 원칙

회의에서 멘토(이희찬)님이 강조한 3가지 핵심:

1. **EMR 대체가 아닌 브릿징** — 기존 EMR 업체와 경쟁하지 않고, API 게이트웨이를 통해 어떤 EMR이든 연동 가능한 구조
2. **포트폴리오용 AWS 풀스택 + 현실용 온프레미스 듀얼 아키텍처** — AWS 서비스를 최대한 활용한 그림 + 온프레미스 배포 가능한 구조 병행
3. **모델 자동 갱신 파이프라인** — 새 환자 데이터 → 희귀질환 DB 축적 → 배치 재학습 → 모델 자동 교체

---

## 2. 전체 아키텍처 (AWS 클라우드 버전)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        외부 연동 계층                                    │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │ EMR 시스템 │  │ SMART on FHIR│  │ 사전 문진    │                      │
│  │ (BESTCare │  │  Sandbox     │  │ (향후 과제)  │                      │
│  │  등 3rd)  │  │              │  │              │                      │
│  └─────┬─────┘  └──────┬───────┘  └──────┬───────┘                      │
│        │               │                 │                              │
│        └───────────────┼─────────────────┘                              │
│                        ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              API Gateway (REST API)                              │    │
│  │         FHIR R4 호환 · API Key 인증 · Rate Limiting             │    │
│  └─────────────────────────┬───────────────────────────────────────┘    │
└────────────────────────────┼────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     컴퓨팅 계층 (진단 파이프라인)                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Lambda — 오케스트레이터                        │    │
│  │         환자 입력 수신 → 4-Phase 파이프라인 조율                   │    │
│  └──┬──────────┬──────────────┬──────────────┬─────────────────────┘    │
│     │          │              │              │                          │
│     ▼          ▼              ▼              ▼                          │
│  ┌──────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │Phase1│  │  Phase2   │  │  Phase3   │  │  Phase4   │                  │
│  │X-ray │  │Multi-modal│  │   Rare    │  │  Report   │                  │
│  │      │  │  Scoring  │  │ Screening │  │Generation │                  │
│  └──┬───┘  └──────────┘  └─────┬─────┘  └─────┬─────┘                  │
│     │                          │               │                        │
│     ▼                          ▼               ▼                        │
│  SageMaker                 Lambda           Bedrock                     │
│  Endpoint                  (HPO-LR          Claude                      │
│  (SooNet                    Engine)          Sonnet                      │
│   DenseNet-121)                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

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

### 3-4. 데이터 계층

| 서비스 | 용도 | 비고 |
|--------|------|------|
| **S3** (데이터) | X-ray 이미지, 모델 가중치, 학습 데이터 | `say2-2team-bucket` |
| **DynamoDB** | 환자 진단 이력, 메타데이터 | 온디맨드 과금 |
| **DynamoDB** | 희귀질환 케이스 축적 DB | 새 케이스 수집용 |
| **로컬 파일** (Orphanet XML) | 희귀질환-HPO 매핑 DB | `en_product4.xml` 사전 다운로드 필요 → `rag/knowledge_base.py` 실행으로 CSV 생성 |
| **로컬 ChromaDB** | PubMed 논문 벡터 캐시 | `rag/data/chroma_db/` — chromadb 패키지 필요 |

DynamoDB 테이블 설계:
```
[diagnosis-history]
  PK: patient_mrn
  SK: diagnosis_timestamp
  Attributes: case_id, top_diseases, confidence, phase3_triggered, report_url

[rare-case-collection]
  PK: disease_orpha_id
  SK: case_id
  Attributes: hpo_codes, lab_summary, xray_findings, confirmed_diagnosis, collected_at
```

### 3-5. RAG / 외부 API 연동

| 서비스/API | 용도 | 호출 방식 | 키 필요 |
|------------|------|-----------|---------|
| **PubCaseFinder API** | 희귀질환 유사 케이스 검색 (LIRICAL 교차검증) | Lambda → HTTPS | 불필요 (비영리 공개 API, 간헐적 다운 주의) |
| **PubMed E-utilities** | 최신 논문 검색 + ChromaDB 캐싱 | Lambda → HTTPS | 불필요 (선택: NCBI API Key로 10 req/s 향상) |
| **Monarch Initiative** | HPO 코드 검증 및 질환-유전자 연결 | Lambda → HTTPS | 불필요 |
| **ClinicalTrials.gov v2** | 희귀질환 임상시험 정보 | Lambda → HTTPS | 불필요 |
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
│   │  PubCaseFinder │ PubMed │ ClinicalTrials.gov (향후)      │                    │
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

## 12. 태그 규칙 (필수)

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

## 13. 요약

이 아키텍처는 두 가지 버전을 동시에 제공합니다:

1. **AWS 클라우드 버전** (포트폴리오): CloudFront → API Gateway → Lambda → SageMaker/Bedrock/DynamoDB 풀스택. 모델 자동 갱신 파이프라인 포함. 발표 시 "이 시스템은 AWS 환경에서 이렇게 확장 가능합니다"를 보여주는 그림.

2. **온프레미스 버전** (현실 배포): FastAPI + 로컬 GPU + PostgreSQL. 환경변수 전환만으로 동일 파이프라인 동작. "병원 내부에 설치해서 바로 쓸 수 있습니다"를 보여주는 실체.

핵심 차별점은 **EMR 브릿징 API Gateway**입니다. 어떤 EMR 시스템이든 FHIR R4 호환 API를 통해 연동 가능하므로, 기존 EMR 업체와 경쟁이 아닌 협력 관계를 만들 수 있습니다.

# S3 버킷 구조 가이드 — Soo-Pul 서비스 운영 기준

> 버킷: `say2-2team-bucket` (ap-northeast-2)  
> CloudFront: `https://d300v14l8u0wx7.cloudfront.net`  
> 이 폴더(`2_aws_say2_project_cloudfront`)는 S3 핵심 파일의 로컬 백업입니다.

---

## 1. CloudFront `?demo=1` 모드에 필요한 파일

> 이것만 있으면 AWS 없이도 GitHub Pages에서 전체 UI 시연 가능

```
s3://say2-2team-bucket/frontend/
├── index.html          메인 React SPA
├── app.html            SMART on FHIR 콜백 페이지
├── launch.html         SMART OAuth2 진입 페이지
├── assets/
│   ├── main-CS3Ig2pc.js    627KB · 전체 앱 번들
│   ├── main-D2K1u-GR.css   16KB  · Tailwind + IBM Plex 폰트
│   ├── browser-BR0-WZ3V.js 77KB  · fhirclient 라이브러리
│   └── ...
└── mock_fhir/
    ├── Patient/            합성환자 5명 (FHIR R4)
    ├── Observation/        HPO · Lab · Vital
    ├── Condition/          워킹 진단
    ├── ImagingStudy/       CXR 메타
    ├── DocumentReference/  한국어 임상노트
    ├── Endpoint/           CXR 이미지 URL
    ├── mock_results/       Phase 1~F 사전계산 결과 (progressive 렌더링)
    ├── analytics/          대시보드 KPI JSON 7종
    └── knowledge/rare_diseases.json   Orphanet 528종 희귀질환 DB
```

**복구 명령:**
```bash
aws s3 sync 2_aws_say2_project_cloudfront/frontend/ s3://say2-2team-bucket/frontend/
```

---

## 2. 실제 AI 추론에 필요한 파일 (full 모드)

### Phase 2 — SooNet CXR 추론 (SageMaker)

```
s3://say2-2team-bucket/
├── Phase_2/models/soonet/model.tar.gz   146MB · SageMaker 엔드포인트 모델 패키지
│                                         → SageMaker create_model() 에 사용
└── weights/models/
    ├── soonet_uones.pth    27MB · U-Ones 학습 전략 가중치
    ├── soonet_uignore.pth  27MB · U-Ignore 학습 전략 가중치
    ├── soonet_umixed.pth   27MB · U-Mixed 학습 전략 가중치
    └── (위 3종 중 하나가 model.tar.gz 안에 패키징되어 SageMaker에 배포됨)

weights/unet_lung_mask_ep10.pth   56.5MB · 폐 영역 마스킹 U-Net
                                   → Phase 2 전처리 단계에서 사용
```

### Phase 1 — 증상 텍스트 → HPO (Lambda + Bedrock Haiku)

```
Phase_1/
├── symptom_llm_4.py                   Lambda 핸들러 소스
├── hpo_official.json                  HPO 공식 온톨로지 (22MB)
└── infra/aws/phase1/
    ├── lambda/handler.py              Lambda 함수 진입점
    ├── template.yaml                  SAM 배포 템플릿
    └── deploy.sh                      배포 스크립트
```

### Phase 3 — Lab 수치 → HPO (Lambda, Rule-based)

```
Phase_3/infra/aws/phase3/
├── lambda/handler.py                  27KB · 규칙 기반 HPO 변환 로직
├── template.yaml
└── deploy.sh
```

### Phase 4 — 통합 랭킹 + LLM 검증 (Lambda)

```
Phase_4/infra/aws/phase4/
├── lambda/handler.py                  18KB · 감별진단 랭킹 로직
├── template.yaml
└── deploy.sh
```

### Phase 5 — LIRICAL LR 스코어링 + 소견서 생성 (Lambda + Bedrock Sonnet)

```
Phase_5/infra/aws/phase5-lr/
├── lambda/
│   ├── handler.py                     16KB · Phase 5 핸들러
│   ├── lr_engine.py                   8KB  · LIRICAL LR 계산 엔진
│   └── db_reader.py                   10KB · DynamoDB 조회
└── template.yaml

assets/
├── AppleGothic.ttf   14.5MB · 한국어 폰트 (소견서 PDF 생성)
└── ArialUnicode.ttf  22.2MB · 유니코드 폰트 (소견서 PDF 생성)
```

> **중요**: `assets/` 폰트 없으면 Phase 5 소견서 PDF 한국어 깨짐

### RAG 오케스트레이터 (Lambda + Bedrock)

```
RAG/infra/lambda/
├── rag_llm_3.py       75KB · RAG 핵심 로직 (PubCaseFinder · Monarch · PubMed)
├── handler.py         13KB · Lambda 진입점
└── requirements.txt

RAG/hpo_official.json       22MB · HPO 온톨로지
RAG/hpo_whitelist.json      1MB  · 필터링된 HPO 화이트리스트
```

### 오케스트레이션 (Step Functions)

```
infra/aws/stepfunctions/
├── state_machine.asl.json   Phase 1→2→3→4→5→RAG 연결 정의
├── template.yaml            SAM 배포 템플릿
└── deploy.sh
```

---

## 3. 백엔드 API

```
api/app/
├── main.py                FastAPI 앱 초기화
├── config.py              Cognito · API Gateway 설정값
├── deps.py                JWT 검증 의존성
└── routers/
    ├── patients.py        환자 목록 · 상세 조회
    ├── sessions.py        진단 세션 생성 · 조회
    ├── emr_updates.py     WebSocket /ws/emr-updates/ 실시간 진행상태
    ├── feedback.py        의사 피드백 저장
    └── admin.py           관리자 API
```

---

## 4. 데이터베이스

```
database/
└── (DynamoDB 테이블 스키마 · 마이그레이션)

주요 테이블:
- diagnosis_sessions     진단 세션 (TTL 24h 캐시)
- phase_logs             각 Phase 실행 로그
- doctor_feedback        의사 수정 피드백
```

---

## 5. S3 전체 폴더 현황

| 폴더 | 크기 | 이 폴더에 백업됨 | 용도 |
|---|---|---|---|
| `frontend/` | 2.8 MB | ✅ | CloudFront 서빙 파일 |
| `Phase_1~5/` | ~560 MB | ✅ | Lambda 소스 + SAM |
| `RAG/` | 23 MB | ✅ | RAG Lambda |
| `infra/` | 72 KB | ✅ | Step Functions |
| `api/` | 204 KB | ✅ | FastAPI 백엔드 |
| `docs/` | 140 KB | ✅ | 설계 문서 |
| `lung_dx/` | 250 KB | ✅ | 도메인 코드 |
| `deploy/` | 3.2 MB | ✅ | 배포 자동화 |
| `database/` | 3.4 MB | ✅ | DB 스키마 |
| `mock-emr/` | 455 KB | ✅ | EMR mock 데이터 |
| `output/` | 57 MB | ✅ (로그만, .pth 제외) | SageMaker 학습 결과 |
| `weights/` | 137 MB | ✅ (이번에 추가) | SooNet 추론 가중치 ⚠️ |
| `assets/` | 36.7 MB | ✅ (이번에 추가) | PDF 한국어 폰트 ⚠️ |
| `Phase_2/models/soonet/model.tar.gz` | 146 MB | ✅ | SageMaker 모델 패키지 |
| `models/` | 3.6 GB | ❌ | 과거 학습 아티팩트 (불필요) |
| `checkpoints/` | 1.1 GB | ❌ | 재학습용 체크포인트 |
| `code/` | 3 GB | ❌ | SageMaker 학습 컨테이너 |
| `SAM/` | 176 MB | ❌ | 패키징된 Lambda (소스에서 재빌드 가능) |
| `data/` | 11.6 GB+ | ❌ | MIMIC-CXR 학습 이미지 |
| `cheXpert_data/` | 수십 GB | ❌ | CheXpert 검증 이미지 |
| `csv/` | 32 MB | ❌ | 학습 CSV |
| `final_csv/` | 417 MB | ❌ | 평가 결과 CSV |

---

## 6. 서비스 복구 절차

### 6-1. demo=1 모드만 복구 (5분)

```bash
# S3에 프론트엔드 업로드
aws s3 sync frontend/ s3://say2-2team-bucket/frontend/
# CloudFront 캐시 무효화
aws cloudfront create-invalidation --distribution-id <ID> --paths "/*"
```

### 6-2. 실제 AI 추론 포함 전체 복구 (신규 AWS 계정 기준 1~2일)

```bash
# 1. 모델 가중치 업로드
aws s3 sync weights/ s3://say2-2team-bucket/weights/
aws s3 cp Phase_2/models/soonet/model.tar.gz s3://say2-2team-bucket/Phase_2/models/soonet/

# 2. 폰트 업로드
aws s3 sync assets/ s3://say2-2team-bucket/assets/

# 3. 각 Phase SAM 배포
bash Phase_1/infra/aws/phase1/deploy.sh
bash Phase_2/infra/aws/phase2/deploy.sh
bash Phase_3/infra/aws/phase3/deploy.sh
bash Phase_4/infra/aws/phase4/deploy.sh
bash Phase_5/infra/aws/phase5-lr/deploy.sh
bash RAG/infra/deploy.sh

# 4. Step Functions 배포
bash infra/aws/stepfunctions/deploy.sh

# 5. 프론트엔드 + CloudFront
aws s3 sync frontend/ s3://say2-2team-bucket/frontend/

# 전체 가이드: TEAM_DEPLOYMENT_GUIDE.md 참고
```

---

## 7. 환경변수 / 설정값 (코드에 없는 것)

서비스 재배포 시 아래 값들을 별도로 설정해야 합니다.  
현재 값은 AWS Parameter Store 또는 팀 내부 보관.

| 항목 | 위치 |
|---|---|
| Cognito User Pool ID | `api/app/config.py` → 환경변수 `COGNITO_USER_POOL_ID` |
| Cognito Client ID | `api/app/config.py` → 환경변수 `COGNITO_CLIENT_ID` |
| API Gateway URL | 프론트엔드 빌드 시 `VITE_API_BASE_URL` |
| CloudFront Distribution ID | 배포 스크립트 파라미터 |
| DynamoDB 테이블명 | Lambda 환경변수 `TABLE_NAME` |
| Bedrock 모델 ID | Lambda 환경변수 (ap-northeast-2 Haiku · Sonnet 3.5) |

---

*작성일: 2026-05-29 · SKKU AWS SAY 2기 2팀*

# Soo-Pul (수폴) — AI-Assisted Rare Pulmonary Disease Diagnosis

> **SKKU AWS SAY 2기 2팀** · 2026 Q1–Q2  
> 흉부 X-ray + 증상 텍스트 + 혈액검사 수치를 결합해  
> 528종 폐질환 감별진단 보조 소견서를 자동 생성하는 Clinical Decision Support 시스템

---

## Live Demo

| 환경 | URL | 비고 |
|---|---|---|
| **GitHub Pages** (AWS 없이 영구 동작) | [heotaewoong.github.io/aws_say2_project/?demo=1](https://heotaewoong.github.io/aws_say2_project/?demo=1) | 이 리포지토리 |
| **AWS CloudFront** (프로덕션) | [d300v14l8u0wx7.cloudfront.net/?demo=1](https://d300v14l8u0wx7.cloudfront.net/?demo=1) | S3 origin |

`?demo=1` 모드: 실제 AWS 추론 없이 mock FHIR JSON으로 전체 화면 시연 가능.  
AWS 계정 만료 후에도 이 GitHub Pages 링크는 정상 동작합니다.

---

## 시스템 개요

흉부 X-ray · 증상 · 혈액검사 3축 입력을 받아 5단계 파이프라인으로 처리:

```
입력 3축
┌──────────────┬────────────────────┬──────────────────┐
│  CXR 영상    │  증상 자유 텍스트   │  혈액검사 수치   │
└──────┬───────┴──────────┬──────────┴────────┬─────────┘
       │                  │                   │
  Phase 2             Phase 1             Phase 3
  SooNet 추론         Bedrock Haiku        Rule-based
  (DenseNet-121)      증상 → HPO ID        Lab → HPO ID
       │                  │                   │
       └──────────────────┴───────────────────┘
                          │  HPO 벡터 통합
              ┌───────────┴───────────┐
              │                       │
        일반 폐질환               희귀 폐질환
     가중치 스코어링           LIRICAL LR 계산
     (CheXpert 14종)          (Orphanet 528종)
              │                       │
              └───────────┬───────────┘
                          │  Phase 4  통합 랭킹
                    ┌─────┴──────┐
                    │  RAG 3단계  │
                    │  PubCaseFinder · Monarch · Orphanet
                    │  PubMed · ClinicalTrials
                    └─────┬──────┘
                          │  Phase 5
                 Bedrock Claude Sonnet 3.5
                 → JSON 감별진단 소견서
```

---

## 리포지토리 구조

이 리포지토리는 `s3://say2-2team-bucket` 에서 내려받은 내용입니다.

```
/
├── index.html              React SPA 진입점 (GitHub Pages 경로 수정됨)
├── app.html                SMART on FHIR 콜백
├── launch.html             SMART OAuth2 진입
├── .nojekyll               GitHub Pages Jekyll 비활성화
├── 404.html                SPA 라우팅 폴백
├── TEAM_DEPLOYMENT_GUIDE.md / .pdf   팀 배포 가이드 전체
│
├── assets/                 JS + CSS 빌드 번들
│   ├── main-CS3Ig2pc.js    627 KB · 전체 앱 (진단 워크스페이스 포함)
│   ├── main-D2K1u-GR.css   16 KB · Tailwind + IBM Plex 폰트
│   └── ...
│
├── mock_fhir/              정적 FHIR R4 mock 데이터 (demo=1 모드용)
│   ├── Patient/            합성환자 5명
│   ├── Observation/        HPO · Lab · Vital
│   ├── Condition/          워킹 진단
│   ├── ImagingStudy/       CXR 메타
│   ├── DocumentReference/  한국어 임상노트
│   ├── mock_results/       Phase 1–F 사전계산 진단 결과 (progressive 렌더링용)
│   ├── analytics/          대시보드 KPI 7종
│   └── knowledge/rare_diseases.json   Orphanet 528종 DB
│
├── Phase_1/                증상 텍스트 → HPO 변환 (Bedrock Haiku · Lambda)
│   ├── symptom_llm_4.py
│   ├── hpo_official.json   공식 HPO 온톨로지
│   └── infra/aws/phase1/   SAM 템플릿 · Lambda handler · deploy.sh
│
├── Phase_2/                CXR → 14개 레이블 분류 (SooNet · SageMaker)
│   └── infra/aws/phase2/   SAM 템플릿 · SageMaker inference
│
├── Phase_3/                Lab 수치 → HPO (Rule-based · Lambda)
│   └── infra/aws/phase3/   SAM 템플릿 · Lambda handler
│
├── Phase_4/                통합 랭킹 + LLM 검증 (Lambda)
│   └── infra/aws/phase4/   SAM 템플릿 · Lambda handler
│
├── Phase_5/                LIRICAL LR 스코어링 + 소견서 생성 (Lambda)
│   ├── infra/aws/phase5/   SAM 템플릿 · Lambda handler · lr_engine.py
│   └── infra/aws/phase5-lr/ LR 계산 분리판
│
├── RAG/                    RAG 파이프라인 오케스트레이터
│   ├── infra/lambda/rag_llm_3.py   75 KB · 핵심 RAG 로직
│   ├── hpo_official.json / hpo_whitelist.json
│   └── infra/template.yaml  SAM 배포 템플릿
│
├── infra/                  Step Functions + 공통 인프라
│   ├── aws/stepfunctions/state_machine.asl.json   오케스트레이션 정의
│   └── aws/stepfunctions/template.yaml
│
├── api/                    FastAPI 백엔드
│   ├── app/main.py
│   ├── app/routers/        patients · sessions · emr_updates · feedback
│   └── README.md
│
├── docs/                   설계 문서
│   ├── ARCHITECTURE_REPORT_2026-05-19.md
│   ├── BUILD_SUMMARY_2026-05-18.md
│   ├── ERROR_HANDLING_REPORT.md
│   └── pipeline_io_examples/   각 Phase 입출력 스키마 + SQL DDL
│
├── lung_dx/                도메인 지식 · 멀티모달 추론 모듈
├── deploy/                 배포 자동화 스크립트
├── database/               DB 스키마 · 마이그레이션
├── mock-emr/               EMR 연동 mock 데이터
└── output/                 SageMaker 학습 결과
    ├── chexnet-2team-v4/training_history.json
    └── chexnet-2team-v4/training.log
```

---

## AI 모델 — SooNet

DenseNet-121 기반 흉부 X-ray 분류 모델 (내부 개발).  
MIMIC-CXR 파인튜닝, CheXpert validation 50장 평가 결과:

| 질환 | AUROC |
|---|---|
| Pleural Effusion | **0.982** |
| Consolidation | **0.935** |
| Cardiomegaly | 0.855 |
| Edema | 0.843 |
| Lung Opacity | 0.834 |
| **평균 (14 레이블)** | **0.776** |

학습 이력: `output/chexnet-2team-v4/training_history.json`

---

## 프론트엔드 (React)

| 화면 | 내용 |
|---|---|
| 로그인 | SMART on FHIR SSO · Cognito 연동 |
| 외래 워크리스트 | 당일 환자 목록 · 프리뷰 드로어 |
| **진단 워크스페이스** | 3-Panel (입력 / CXR+SooNet / 감별진단) |
| **LR 막대 시각화** | 지지(녹) / 반박(적) · Robinson 2020 Fig.2 차용 |
| CXR 뷰어 | Grad-CAM overlay · PACS 스타일 확대 |
| 리포트 뷰어 | Bedrock 생성 소견서 |
| 유사 케이스 (RAG) | 벡터 검색 유사 증례 |
| 애널리틱스 | KPI 대시보드 (희귀탐지율 · AI 정확도 등) |

스택: React 18 + Vite + Tailwind CSS · IBM Plex Sans KR · fhirclient (SMART on FHIR v2.2)

### 데모 환자 5명 (mock_fhir/)

| MRN | 시나리오 | 핵심 감별 |
|---|---|---|
| 26-145982 | Community-acquired Pneumonia | Pneumonia 0.92 |
| 26-098234 | COPD Exacerbation | Emphysema |
| 26-204017 | IPF (희귀) | Idiopathic Pulmonary Fibrosis |
| 26-301102 | LAM (희귀) | Lymphangioleiomyomatosis |
| 26-415523 | 폐 종괴 (감별) | Lung Mass DDx |

---

## AWS 인프라

| 서비스 | 용도 |
|---|---|
| S3 | 정적 프론트엔드 + 학습 데이터 + CXR 이미지 |
| CloudFront | CDN + HTTPS (S3 OAC) |
| Cognito | 의사 계정 JWT 인증 |
| API Gateway | REST + WebSocket (`/ws/emr-updates/`) |
| Lambda (×5) | 각 Phase 독립 배포 · SAM 템플릿 |
| Step Functions | Phase 1→5 오케스트레이션 (`infra/aws/stepfunctions/`) |
| SageMaker | SooNet 추론 엔드포인트 (Phase 2) |
| Bedrock Haiku | Phase 1: 증상 → HPO |
| Bedrock Sonnet 3.5 | Phase 5 + RAG: JSON 소견서 생성 |
| DynamoDB | 진단 결과 캐시 (TTL 24h) |

배포: 각 Phase 폴더의 `infra/aws/phaseN/deploy.sh` 또는 `TEAM_DEPLOYMENT_GUIDE.md` 참고

---

## GitHub Pages vs CloudFront

| | GitHub Pages (이 리포) | CloudFront |
|---|---|---|
| 서빙 | GitHub CDN · 정적만 | AWS CDN + API Gateway + Lambda |
| 인증 | 없음 | Cognito JWT |
| AI 추론 | mock JSON | SageMaker + Bedrock 실추론 |
| WebSocket | 없음 | `/ws/emr-updates/` 실시간 |
| `?demo=1` 시 | 완전 동작 | 완전 동작 |
| AWS 종료 후 | **영구 동작** | 중단 |

---

## 팀

**SKKU AWS SAY (Solution Architect Young) 2기 2팀**

| 이름 | 역할 |
|---|---|
| 박성수 | Frontend Lead · UI/UX 설계 |
| 허태웅 | AI 모델 학습 · AWS 인프라 (VPC · Lambda · SageMaker) |
| 배기태 | AI 모델 학습 · AWS 인프라 |
| 권미라 | MIMIC-IV 데이터 추출 · 희귀질환 지식베이스 |
| 양희인 | MIMIC-IV 데이터 추출 · 희귀질환 지식베이스 |
| 이희찬 | AWS 멘토 |

---

## 주요 참고 문헌

- Robinson PN et al. *Am J Hum Genet* 2020;107:403-417 — LIRICAL LR paradigm
- Neri E et al. *Radiol Med* 2023;128:755-764 — Explainable AI in radiology
- Mandel JC et al. *JAMIA* 2016;23:899-908 — SMART on FHIR
- EU AI Act 2024/1680/EU Art. 22 — Human-in-the-loop

---

> 연구·교육용 프로토타입. SaMD 허가 전이며 임상 진단에 직접 사용 불가.

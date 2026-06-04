# 1_aws_say2_project — 최신 개발 버전

> SooNet V8 + 전체 AWS 파이프라인 통합 · SKKU AWS SAY 2기 2팀

이 디렉토리는 **가장 최신 개발 코드**입니다.  
배포 버전은 `2_aws_say2_project_cloudfront/`, 랜딩 페이지는 `3_soonet_demo/`를 참고하세요.

---

## 폴더 구조

| 폴더 | 역할 | 담당 |
|------|------|------|
| `Phase_1/` | 증상 텍스트 → HPO 코드 추출 (Bedrock Haiku Lambda) | 허태웅 |
| `Phase_2/` | SooNet V8 CXR 추론 + Grad-CAM (SageMaker) | 배기태 · 허태웅 |
| `Phase_3/` | Lab 수치 → HPO 통합 다중 질환 스코어링 | 박성수 |
| `Phase_4/` | LLM 감별진단 검증 Top-3 (Bedrock Sonnet) | 박성수 |
| `Phase_5/` | LIRICAL LR 희귀질환 528종 스크리닝 | 허태웅 |
| `RAG/` | Hybrid Dual RAG 최종 AI 소견서 생성 | 허태웅 · 권미라 |
| `api/` | FastAPI 백엔드 + Step Functions 연동 | 박성수 |
| `frontend/` | React 18 + Vite 프론트엔드 | 박성수 |
| `infra/` | CloudFormation IaC · Lambda · SageMaker 배포 스크립트 | 허태웅 |
| `database/` | Aurora PostgreSQL DDL (v4) · DynamoDB 스키마 | 권미라 |
| `mock-emr/` | SMART on FHIR 시뮬레이션 환자 데이터 | 양희인 |

---

## 최종 모델 성능 (SooNet V8)

| 지표 | 값 |
|------|:--:|
| **Macro AUROC** | **0.773** |
| 평균 Recall | 61.5% |
| 최고 AUROC | 0.851 (No Finding) |
| 최고 Recall | 88.5% (Lung Opacity) |

전체 14종 지표: 루트 `README.md` 참고

S3 모델 경로: `s3://say2-2team-bucket/Phase_2/soonet_v8_phase4_best.pth`

---

## 팀

| 역할 | 이름 | 담당 |
|:----:|:----:|------|
| 🎯 Frontend & Backend Lead | 박성수 (팀장) | React UI · SMART on FHIR · Lambda/Step Functions |
| 🧠 Model · RAG · Arch | 허태웅 | SooNet V8 학습 · AWS 아키텍처 · RAG · 최종 발표 |
| 🧠 Model & Data | 배기태 | SooNet 모델 훈련 · 데이터 전처리 |
| 📊 Data | 양희인 | MIMIC-IV 데이터 수집 · 전처리 |
| 🗃 RAG & DB | 권미라 | RAG 파이프라인 · Aurora DDL · 데이터 분석 |
| 🏫 AWS Mentor | 이희찬 | AWS AINATION 멘토링 |

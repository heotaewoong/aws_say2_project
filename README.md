# Rare-Link AI — 폐질환 진단 보조 시스템

> 흉부 X-ray + 혈액검사 + 증상 텍스트를 종합하여  
> AWS Bedrock 기반 RAG 파이프라인으로 진단 보조 소견서(JSON)를 생성합니다.

**상태** (2026-05-05 기준)
- RAG 파이프라인 로컬 실행 검증 ✅ (MIMIC 환자 1명, 샘플 X-ray)
- SooNet 모델 성능 CheXpert 50장: 평균 AUROC **0.78** · 핵심 질환(흉막삼출·경화) **0.93+**
- AWS 인프라 CloudFormation 2개 스택 validate 통과 ✅
- 현실 배포안(Single-AZ + Lambda 단독) 문서화 완료

---

## 📁 프로젝트 구조

```
aws_say2_project_vision/
│
├── rag_pipeline.py           # 🚀 메인 — 5단계 RAG 오케스트레이터
├── soo_net.py                # X-ray 분류 (DenseNet-121 + U-Net crop)
├── requirements.txt
├── .env                      # AWS 키 (gitignore)
│
├── rag/                      # 🧠 RAG 컴포넌트
│   ├── bedrock_extractor.py      # Phase 1: 증상 → HPO (Bedrock Haiku)
│   ├── lab_rules.py              # Phase 3: Lab 수치 → HPO (Rule-based)
│   ├── lirical_scorer.py         # 희귀질환 LR 스코어링 (LIRICAL 방식)
│   ├── general_disease_scorer.py # 일반 폐질환 가중치 스코어링
│   ├── pubcasefinder.py          # API ①: HPO → 희귀질환 후보 (DBCLS)
│   ├── orphanet_fetcher.py       # API ②: OrphaCode → 유전자/역학 (로컬 XML)
│   ├── monarch_fetcher.py        # API ③: 인과 유전자 + HPO 이름 (Monarch)
│   ├── pubmed_fetcher.py         # API ④: 질환명 → 케이스리포트 (NCBI)
│   ├── clinicaltrials_fetcher.py # API ⑤: 모집 중 임상시험 (NIH)
│   ├── ragas_eval.py             # RAG 검증 (PMID 환각 체크)
│   ├── knowledge_base.py         # (확장 예정) pgvector 연동
│   └── valid/                    # 검증 스크립트
│       ├── run_rag_test.py           # MIMIC 다중 환자 E2E 테스트
│       ├── fetch_mimic_patient.py    # S3에서 환자 데이터 수집
│       ├── eval_proper.py            # RAG 정량 검증 (AUROC + 환각 체크)
│       └── RAG_통합_검증보고서.md
│
├── scripts/                  # 🔧 학습/평가/전처리
│   ├── train/
│   │   ├── train_soo_net_local.py
│   │   ├── train_chexpert.py
│   │   ├── unet_train.py
│   │   └── sagemaker/             # SageMaker 학습 스크립트
│   ├── eval/
│   │   ├── eval_soonet_local.py       # 로컬 성능 평가 (AUROC/F1)
│   │   ├── eval_valid_compare.py       # U-Ones/U-Ignore/Mixed 비교
│   │   └── eval_soo_net_sagemaker.py
│   └── preprocess/
│       ├── preprocess_mimic.py
│       ├── resize_mimic_to_s3.py
│       └── check_mimic_mapping.py
│
├── infra/                    # ☁️ AWS 배포
│   ├── README.md                      # 배포 가이드
│   ├── deploy.sh                      # 🚀 원클릭 배포 (ECR push + CFN)
│   ├── cloudformation/
│   │   ├── 00-simple-deploy.yaml          # 🎯 현실 배포 (Lambda 단독)
│   │   ├── 01-network.yaml                # Multi-AZ 완전판 (HA)
│   │   └── 02-phase2-xray.yaml            # SageMaker GPU 분리판
│   ├── lambda/
│   │   ├── Dockerfile                 # Lambda 컨테이너 이미지
│   │   ├── app.py                     # RAG 전체 핸들러
│   │   └── phase2/
│   │       └── phase2_handler.py      # Phase 2 분리판 (SageMaker 경유)
│   └── sagemaker/
│       └── inference.py               # model.tar.gz 추론 엔트리포인트
│
├── aws_architecture/
│   ├── architecture_final.html            # 완전판 아키텍처 도식
│   ├── AWS_아키텍처_제안서.md
│   └── AWS_현실배포_계획서_v1.md          # 🆕 2~3일 배포 축소안
│
├── note_정리/
│   ├── rag/
│   │   ├── RAG_구현_보고서_v2.md          # 🚀 팀 공용 구현 보고서
│   │   ├── RAG_프롬프트_API_정리서.md
│   │   └── 최종프롬프트_API시스템_확정문서_v1.docx
│   └── (학습 실험 결과)
│
├── data/                     # Orphanet XML + CSV (4,335 희귀질환)
├── model/                    # 학습된 가중치 (.pth, gitignore)
├── frontend/                 # React UI (Vite)
├── cam_results/              # Grad-CAM 시각화 (5개 질환)
│
├── archive/
│   └── legacy_pipeline/      # 🗂️ 예전 4-Phase 구현 (참고용, 미사용)
│
├── CLAUDE.md                 # Claude Code 행동 규칙
├── CLEANUP_PLAN.md           # 폴더 정리 이력
└── README.md                 # (이 파일)
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
pip install -r requirements.txt

# .env 작성 (gitignore됨)
cat > .env <<EOF
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_DEFAULT_REGION=ap-northeast-2
EOF
```

### 2. 로컬 RAG 파이프라인 실행

```bash
# 테스트 X-ray 다운로드
aws s3 cp s3://say2-2team-bucket/cheXpert_data/valid_only/patient64541/study1/view1_frontal.jpg /tmp/test_xray.jpg

# 5단계 파이프라인 실행
python rag_pipeline.py
```

출력: JSON 소견서 (`recommendation` + `clinical_notes` 구조)

### 3. SooNet 성능 평가

```bash
python scripts/eval/eval_soonet_local.py --samples 50
```

### 4. MIMIC 다중 환자 검증

```bash
python rag/valid/fetch_mimic_patient.py
python rag/valid/run_rag_test.py
```

---

## ☁️ AWS 배포

**현실 배포안** (2~3일 내 가능, 자세한 내용 `aws_architecture/AWS_현실배포_계획서_v1.md`):

```bash
# 원클릭 배포 (ECR push + CloudFormation stack)
bash infra/deploy.sh

# 삭제
bash infra/deploy.sh destroy
```

**완전판** (Multi-AZ + SageMaker GPU, 크레딧 넉넉할 때):

```bash
# 2개 스택 순차 배포
aws cloudformation deploy --stack-name rare-link-network \
  --template-file infra/cloudformation/01-network.yaml \
  --capabilities CAPABILITY_NAMED_IAM

aws cloudformation deploy --stack-name rare-link-phase2 \
  --template-file infra/cloudformation/02-phase2-xray.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

자세한 내용: `infra/README.md`

---

## 🧩 파이프라인 5단계

```
환자 데이터 (X-ray + 증상 + Lab)
         │
         ▼
  ① Phase 1~3 — HPO 변환
         │ Bedrock Haiku  (증상 → HPO)
         │ SooNet         (X-ray → HPO)
         │ Rule-based     (Lab → HPO)
         ▼
  ② 스코어링 — 이중 트랙
         │ 일반: rank_general_diseases()  → Top 10
         │ 희귀: LIRICAL LR 계산          → 리스팅
         ▼
  ③ Phase 4 — Top 3 통합
         │ 희귀 우선 배치, 일반으로 나머지 채움
         ▼
  ④ RAG 트리거 — 3단계 API 호출
         │ [1] PubCaseFinder (HPO → 후보 질환)
         │ [2] Monarch + Orphanet (병렬 메타데이터)
         │ [3] PubMed + ClinicalTrials (병렬 근거)
         ▼
  ⑤ Bedrock Claude Sonnet 3.5 → JSON 소견서
```

상세 흐름: `note_정리/rag/RAG_구현_보고서_v2.md`

---

## 📊 검증 결과

### SooNet 모델 (CheXpert valid 50장)

| 질환 | AUROC | 판정 |
|------|-------|------|
| Pleural Effusion | 0.9824 | 매우 우수 |
| Consolidation | 0.9352 | 매우 우수 |
| Cardiomegaly | 0.8552 | 우수 |
| Edema | 0.8434 | 우수 |
| Lung Opacity | 0.8342 | 우수 |
| **평균** | **0.7762** | |

### RAG 파이프라인 (MIMIC 환자 + 샘플)

- 5단계 E2E 통과 ✅
- JSON 스키마 검증 통과 ✅
- PMID 환각 0건 ✅
- Bedrock Sonnet 3.5 JSON 출력 정상 ✅

자세한 내용: `rag/valid/RAG_통합_검증보고서.md`

---

## 🏗️ AWS 아키텍처

| 문서 | 내용 |
|------|------|
| `aws_architecture/architecture_final.html` | 25개 서비스 완전판 (이상) |
| `aws_architecture/AWS_현실배포_계획서_v1.md` | 10개 서비스 축소판 (2~3일) |
| `infra/cloudformation/00-simple-deploy.yaml` | 원클릭 배포 템플릿 |

담당 범위 (허태웅):
- VPC / Subnet / 보안 전체
- Phase 2 (X-ray → HPO) 저장·경로·Lambda·SageMaker

---

## 🤝 팀

**SKKU AWS SAY 2기 2팀** · 프리프로젝트 (2026 Q1~Q2)

- 권미라, 배기태, 허태웅

---

## 📄 License

Internal use only.

# 폴더 정리 계획서

> **목적**: 가독성 향상 + legacy 파일 제거 + 팀원이 한눈에 구조 파악
> **원칙**: 파괴적 삭제 최소화, 의심스러운 것은 `archive/`로 격리

---

## 🗑️ 안전하게 삭제 가능 (즉시)

`.gitignore`에 있거나 아무 데서도 import 안 됨:

| 경로 | 사유 |
|------|------|
| `__pycache__/` | Python 캐시 (자동 재생성) |
| `rag/__pycache__/` | 동일 |
| `.DS_Store` | macOS 메타 |
| `eval_soonet_result.csv` | 개별 실험 결과 (`scripts/eval/` 생성물) |
| `test_xray.jpg` | 테스트 입력 (S3에서 받아 쓰는 게 맞음) |
| `note_정리/rag/~$프롬프트_API시스템_확정문서_v1.docx` | Word 임시 락 파일 |

## 📦 archive로 이동 (legacy 코드)

현재 파이프라인(`rag_pipeline.py`) 대신 예전에 썼던 구 파일. 혹시 나중에 참고용으로 남김:

| 파일 | 대체된 곳 |
|------|----------|
| `main.py` | `rag_pipeline.py` (5단계 통합) |
| `app.py` | `infra/lambda/app.py` (Lambda 핸들러) |
| `inference_engine.py` | `rag/lirical_scorer.py` + `rag/general_disease_scorer.py` |
| `knowledge_base.py` | `rag/orphanet_fetcher.py` + `data/orphadata_weighted.csv` |
| `extractor.py` | `rag/bedrock_extractor.py` |
| `reporter.py` | `rag_pipeline.step5_generate_report()` |
| `lab_genomic_agent.py` | `rag/lab_rules.py` |
| `vision_engine.py` | `soo_net.py` (SooNet이 현재 프로덕션) |

→ `archive/legacy_pipeline/` 폴더로 이동 (삭제 X, 히스토리 보존)

## 🔀 구조 재배치

```
중복된 sagemaker/ 폴더가 2개 존재:
  - sagemaker/                       (구 학습 스크립트)
  - scripts/train/train_soo_net_sagemaker.py
```

→ `sagemaker/` 전체를 `scripts/train/sagemaker/`로 이동

```
빈 폴더:
  - preprocess/  (resize_mimic_to_s3.py 1개만)
```

→ `scripts/preprocess/`로 통합

## ✅ 유지

| 폴더/파일 | 역할 |
|-----------|------|
| `rag_pipeline.py` | **메인 엔트리** (실행 검증 완료) |
| `soo_net.py` | X-ray SooNet 엔진 |
| `rag/` | RAG 5개 API + 스코어링 |
| `scripts/` | 학습/평가/전처리 |
| `infra/` | AWS 배포 템플릿 |
| `model/` | 학습된 가중치 (.pth) |
| `data/` | Orphanet XML + CSV (파이프라인 실행 필수) |
| `frontend/` | React UI |
| `aws_architecture/` | 아키텍처 문서 |
| `note_정리/` | 회의록 + 구현 보고서 |
| `cam_results/` | Grad-CAM 시각화 (발표 자료) |

---

## 최종 목표 구조

```
aws_say2_project_vision/
│
├── README.md                          ← 진입점 (업데이트됨)
├── CLAUDE.md                          ← Claude Code 규칙
├── requirements.txt
├── .env                               ← (gitignored)
├── .gitignore
│
├── rag_pipeline.py                    ← 🚀 메인 실행 파일
├── soo_net.py                         ← X-ray 모델
│
├── rag/                               ← RAG 컴포넌트
│   ├── __init__.py
│   ├── bedrock_extractor.py
│   ├── lab_rules.py
│   ├── lirical_scorer.py
│   ├── general_disease_scorer.py
│   ├── pubcasefinder.py
│   ├── orphanet_fetcher.py
│   ├── monarch_fetcher.py
│   ├── pubmed_fetcher.py
│   ├── clinicaltrials_fetcher.py
│   ├── ragas_eval.py
│   ├── knowledge_base.py              ← 남겨둠 (미사용이지만 확장 예정)
│   └── valid/                         ← 검증 스크립트
│       ├── run_rag_test.py
│       ├── fetch_mimic_patient.py
│       ├── eval_proper.py
│       └── RAG_통합_검증보고서.md
│
├── scripts/                           ← 유틸 스크립트
│   ├── train/                         ← 모델 학습
│   │   ├── train_soo_net_local.py
│   │   ├── train_chexpert.py
│   │   ├── unet_train.py
│   │   └── sagemaker/                 ← (이동) SageMaker 학습
│   │       ├── run_sagemaker.py
│   │       ├── run_sagemaker-2class.py
│   │       ├── run_sagemaker-3class.py
│   │       ├── train.py
│   │       ├── train-2class.py
│   │       ├── train-3class.py
│   │       └── train_soo_net_sagemaker.py
│   ├── eval/                          ← 성능 평가
│   │   ├── eval_soonet_local.py
│   │   ├── eval_soo_net_cloud.py
│   │   ├── eval_soo_net_sagemaker.py
│   │   └── eval_valid_compare.py
│   └── preprocess/                    ← (이동) 전처리
│       ├── preprocess_mimic.py
│       ├── check_mimic_mapping.py
│       └── resize_mimic_to_s3.py
│
├── infra/                             ← AWS 배포
│   ├── README.md                      ← 배포 가이드
│   ├── deploy.sh                      ← 원클릭 배포
│   ├── cloudformation/
│   │   ├── 00-simple-deploy.yaml      ← 🚀 현실 배포 1-스택
│   │   ├── 01-network.yaml            ← Multi-AZ 완전판 (선택)
│   │   └── 02-phase2-xray.yaml        ← SageMaker GPU 버전 (선택)
│   ├── lambda/
│   │   ├── Dockerfile                 ← Lambda 컨테이너
│   │   ├── app.py                     ← RAG 전체 핸들러
│   │   └── phase2/
│   │       └── phase2_handler.py      ← Phase 2 분리판 (SageMaker 쓸 때만)
│   └── sagemaker/
│       └── inference.py               ← SageMaker PyTorch 추론 스크립트
│
├── model/                             ← 가중치 (.pth, gitignore됨)
├── data/                              ← Orphanet XML/CSV
├── frontend/                          ← React UI
├── aws_architecture/                  ← 아키텍처 문서
│   ├── architecture_final.html        ← 완전판 도식
│   ├── AWS_아키텍처_제안서.md
│   └── AWS_현실배포_계획서_v1.md      ← 🆕 축소 배포안
├── note_정리/                         ← 회의록 + 보고서
│   ├── rag/
│   │   ├── RAG_구현_보고서_v2.md
│   │   ├── RAG_프롬프트_API_정리서.md
│   │   └── 최종프롬프트_API시스템_확정문서_v1.docx
│   └── ...
├── cam_results/                       ← Grad-CAM 결과
│
└── archive/                           ← 🆕 legacy (필요 시 참고)
    └── legacy_pipeline/
        ├── main.py
        ├── app.py
        ├── inference_engine.py
        ├── knowledge_base.py
        ├── extractor.py
        ├── reporter.py
        ├── lab_genomic_agent.py
        └── vision_engine.py
```

---

## 실행 순서 (지금부터 할 것)

1. `archive/legacy_pipeline/` 생성 후 legacy 8개 파일 이동
2. `sagemaker/` → `scripts/train/sagemaker/` 이동
3. `preprocess/` → `scripts/preprocess/` 통합
4. `__pycache__`, `.DS_Store`, 임시파일 삭제
5. README.md 전면 재작성

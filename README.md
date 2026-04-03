# Rare-Link AI

**멀티모달 희귀 폐 질환 진단 보조 시스템**

> 흉부 X-ray + MIMIC-IV 혈액검사 + 임상 소견 + 유전체 데이터를 종합하여,
> 환자에게 어떤 희귀 폐 질환이 있을 수 있는지 찾아주고 추가 검사를 추천하는 AI 시스템입니다.

---

## 프로젝트 개요 (초보자용)

### 이 프로젝트가 하는 일

```
환자 데이터 4가지 입력:
  1. 흉부 X-ray 이미지      → AI가 보고 "폐렴 85%, 흉막삼출 70%" 판별
  2. 혈액검사 결과 (Lab)     → "WBC 높음, CRP 높음" → 비정상 수치 추출
  3. 임상 소견 (퇴원 요약문)  → NLP로 증상 키워드 자동 추출
  4. 유전체 데이터 (참고용)   → 특정 유전자 변이 확인

         ↓ 모든 결과를 HPO 코드로 통합 ↓

  종합 분석 → 가능성 높은 희귀 폐질환 리스트업
         → 추가 검사 추천 (유전체검사 포함)
```

### 비유: "4명의 전문가가 협력하는 진료 시스템"

| 전문가 | 입력 데이터 | 모델/방법 | 출력 | 코드 |
|--------|------------|----------|------|------|
| 영상의학과 | Chest X-ray | CheXNet (DenseNet-121) | Pathology별 확률 | `vision_engine.py` |
| 진단검사의학과 | MIMIC-IV 혈액검사 | 규칙 기반 판정 | 정상범위 비교 (abnormal) | `lab_genomic_agent.py` |
| 내과 | 임상 소견 (퇴원 요약문) | LLM (Ollama/Gemini) | 예측 질병/증상 카테고리 | `extractor.py` |
| 유전학과 | 유전체 데이터 | 파이프라인 (진단 프로세스) | 변이 확인 + 검사 추천 | `lab_genomic_agent.py` |

---

## 핵심 파이프라인 (전체 흐름)

```
STEP 1: 흉부 X-ray 분석 (메인)
  └─ CheXNet → 14개 Pathology 확률 → HPO 코드 변환
                ↓
STEP 2: Pathology 기반 질병 필터링
  └─ 14개 소견 중 양성인 것만 필터링
  └─ 해당 Pathology를 포함하는 폐 관련 질환 리스트업
                ↓
STEP 3: 추가 데이터로 질병 범위 좁히기 (서브)
  ├─ 혈액검사 (Lab) → 비정상 수치 → HPO 코드
  └─ 임상 소견 (NLP) → 증상 키워드 → HPO 코드
                ↓
STEP 4: 종합 질병 리스트업
  └─ TF-IDF 기반 HPO 매칭 → Orphanet DB → 희귀질환 순위
                ↓
STEP 5: 추가 검사 추천
  └─ 리스트업된 질병 중 추가 검사 필수인 경우:
     "~한 질병의 가능성이 높으니 ~검사를 추가적으로 진행해보시는걸 추천드립니다."
  └─ 희귀질환 가능성 → 유전체검사 추천
```

---

## 프로젝트 파일 구조

```
mini_project/
│
│  -- 핵심 실행 파일 --
├── app.py                    # Streamlit 웹 대시보드 (브라우저 UI)
├── main.py                   # 터미널 배치 실행 스크립트
│
│  -- 파이프라인 모듈 --
├── knowledge_base.py         # Orphanet XML → CSV 변환 (지식베이스 구축)
├── extractor.py              # NLP: 퇴원요약문 → HPO 코드 추출
├── vision_engine.py          # Vision: CheXNet + Grad-CAM (14개 소견 판별)
├── lab_genomic_agent.py      # Lab/유전체: 혈액검사 수치 → HPO 코드
├── inference_engine.py       # 추론: HPO 코드 → 희귀질환 매칭 (TF-IDF)
├── reporter.py               # 리포트: 종합 결과 → 의사용 보고서
│
│  -- 모델 학습 --
├── train_chexpert.py         # CheXNet 학습 (CheXpert 데이터)
├── train_mimic_chexnet.py    # CheXNet 학습 (MIMIC-CXR 데이터)
├── sagemaker/
│   ├── train.py              # SageMaker용 학습 스크립트
│   └── run_sagemaker.py      # SageMaker 작업 실행기
│
│  -- 데이터 --
├── data/
│   ├── CheXpert-v1.0-small/         # CheXpert X-ray 이미지 + 라벨
│   │   ├── train.csv                #   훈련 라벨 (223,414장)
│   │   ├── valid.csv                #   검증 라벨 (234장)
│   │   ├── train/                   #   훈련 이미지
│   │   └── valid/                   #   검증 이미지
│   │
│   ├── mimic-iv/                    # MIMIC-IV 임상 데이터
│   │   ├── hosp/                    #   입원 데이터
│   │   │   ├── labevents.csv.gz     #     혈액검사 결과 (2.4GB)
│   │   │   ├── d_labitems.csv.gz    #     검사명 사전 (1,650항목)
│   │   │   ├── diagnoses_icd.csv.gz #     환자 진단코드 (636만건)
│   │   │   ├── d_icd_diagnoses.csv.gz #   ICD코드→병명 사전 (11만건)
│   │   │   ├── admissions.csv.gz    #     입원기록 (54만건)
│   │   │   └── patients.csv.gz     #     환자정보 (36만명)
│   │   └── icu/                     #   ICU 데이터
│   │       ├── chartevents.csv.gz   #     활력징후 (SpO2, 호흡수) (3.3GB)
│   │       ├── d_items.csv.gz       #     측정항목 사전 (4,095항목)
│   │       └── icustays.csv.gz      #     ICU 입실기록 (9.4만건)
│   │
│   ├── mimic-iv-note/               # MIMIC-IV 퇴원 요약문
│   │   ├── discharge.csv            #   퇴원 요약문 (NLP 입력)
│   │   └── radiology.csv            #   영상의학 소견서
│   │
│   ├── mimic-cxr-2.0.0-chexpert.csv # MIMIC-CXR 라벨 (CheXpert 형식, 22만건)
│   ├── en_product4.xml              # Orphanet 희귀질환 DB (45MB)
│   ├── orphadata_weighted.csv       # 변환된 지식베이스 (8.7MB)
│   └── person3_bacteria_13.jpeg     # 테스트용 X-ray 이미지
│
│  -- 모델 가중치 --
├── models/
│   ├── chexnet_mimic_best.pth       # CheXNet 가중치 (CheXpert 10에포크 학습 완료)
│   └── normal_link_v2_ep50.pth      # NormalLink AE 가중치
│
│  -- 결과물 --
├── cam_results/                     # Grad-CAM 히트맵 이미지
│   ├── 정상_No_Finding.png
│   ├── 심장비대_Cardiomegaly.png
│   ├── 기흉_Pneumothorax.png
│   ├── 폐렴_Pneumonia.png
│   └── 흉막삼출_Pleural_Effusion.png
│
├── note_정리/                       # 회의 노트
└── README.md
```

---

## 핵심 용어 사전

| 용어 | 설명 | 비유 |
|------|------|------|
| **HPO** | 증상을 표준 코드로 변환 (예: 호흡곤란 = HP:0002094) | 증상 바코드 |
| **CheXNet** | X-ray에서 14가지 소견 탐지 AI (DenseNet-121) | X-ray 판독 AI |
| **Grad-CAM** | AI가 이미지 어디를 보고 판단했는지 히트맵 시각화 | AI의 시선 추적기 |
| **Orphanet** | 6,000+개 희귀질환 DB (유럽) | 희귀질환 백과사전 |
| **TF-IDF** | 희귀 증상에 높은 가중치 부여하는 점수 방식 | 희귀할수록 중요! |
| **MIMIC-IV** | MIT 익명화 실제 환자 데이터 (36만명) | 연구용 병원 데이터 |
| **Pathology** | X-ray에서 발견된 이상 소견 (14가지) | AI 판독 결과 |
| **ICD-10** | 국제 질병 분류 코드 (예: J12.9 = 바이러스성 폐렴) | 질병 분류 번호 |

---

## CheXpert 14개 라벨 & HPO 매핑

| # | 소견 (CheXpert) | 한국어 | HPO 코드 | 설명 |
|---|----------------|--------|----------|------|
| 0 | Atelectasis | 무기폐 | HP:0002095 | 폐가 부분적으로 꺼진 상태 |
| 1 | Cardiomegaly | 심장비대 | HP:0001640 | 심장이 비정상적으로 커진 상태 |
| 2 | Consolidation | 폐경화 | HP:0002113 | 폐 조직이 밀집하여 딱딱해진 상태 |
| 3 | Edema | 폐부종 | HP:0002111 | 폐에 액체가 고인 상태 |
| 4 | Enlarged Cardiomediastinum | 종격동확장 | HP:0001640 | 심장 주변 공간이 넓어진 상태 |
| 5 | Fracture | 골절 | HP:0020110 | 뼈가 부러진 상태 |
| 6 | Lung Lesion | 폐병변 | HP:0002088 | 폐에 병적인 부위가 발생 |
| 7 | Lung Opacity | 폐혼탁 | HP:0002113 | 폐가 흐린 상태 |
| 8 | No Finding | 정상 | - | 이상 소견 없음 |
| 9 | Pleural Effusion | 흉막삼출 | HP:0002202 | 폐를 싸고 있는 막에 액체가 고임 |
| 10 | Pleural Other | 기타흉막 | HP:0002103 | 흉막의 다른 이상 |
| 11 | Pneumonia | 폐렴 | HP:0002090 | 폐에 염증/감염 |
| 12 | Pneumothorax | 기흉 | HP:0002107 | 폐가 터져서 공기가 빠져나감 |
| 13 | Support Devices | 의료기기 | - | 관, 튜브 등 의료 기구 확인 |

---

## MIMIC-IV 데이터 스키마 (실제 검증 완료)

### 테이블 관계도

```
patients (subject_id) ──────────────────────────────┐
  │                                                  │
  ├── admissions (subject_id → hadm_id)              │
  │     ├── diagnoses_icd (hadm_id)                  │
  │     │     └─ JOIN d_icd_diagnoses (icd_code)     │
  │     └── labevents (hadm_id)                      │
  │           └─ JOIN d_labitems (itemid)             │
  │                                                  │
  └── icustays (subject_id → hadm_id → stay_id)      │
        └── chartevents (stay_id)                    │
              └─ JOIN d_items (itemid)               │
```

### hosp/ 핵심 테이블

| 테이블 | 행 수 | 핵심 컬럼 | 용도 |
|--------|-------|----------|------|
| patients | 364,627명 | subject_id, gender, anchor_age, dod | 환자 기본정보 |
| admissions | 546,028건 | subject_id, hadm_id, admittime, discharge_location | 입원기록 |
| labevents | 수억 건 | subject_id, hadm_id, itemid, valuenum, flag | 혈액검사 결과 |
| d_labitems | 1,650항목 | itemid, label, fluid, category | 검사명 사전 |
| diagnoses_icd | 6,364,520건 | subject_id, hadm_id, seq_num, icd_code, icd_version | 환자 진단코드 |
| d_icd_diagnoses | 112,107건 | icd_code, icd_version, long_title | ICD→병명 변환 |

### icu/ 핵심 테이블

| 테이블 | 행 수 | 핵심 컬럼 | 용도 |
|--------|-------|----------|------|
| chartevents | 수억 건 | subject_id, stay_id, itemid, valuenum, warning | 활력징후 (SpO2, HR) |
| d_items | 4,095항목 | itemid, label, category, unitname, lownormalvalue | 측정항목 사전 |
| icustays | 94,458건 | subject_id, hadm_id, stay_id, first_careunit, los | ICU 입실기록 |

---

## 현재 진행 상태 (2026.03.30)

| 항목 | 상태 | 비고 |
|------|------|------|
| CheXpert 데이터 다운로드 | ✅ 완료 | 223,414장 이미지 + 라벨 |
| CheXNet 로컬 학습 (10 에포크) | ✅ 완료 | mAUROC ~0.78, 개선 필요 |
| Grad-CAM 시각화 | ✅ 완료 | cam_results/ 폴더에 5개 이미지 |
| MIMIC-IV hosp 다운로드 | ✅ 완료 | S3에서 6개 파일 (labevents, diagnoses 등) |
| MIMIC-IV icu 다운로드 | ✅ 완료 | S3에서 3개 파일 (chartevents 등) |
| MIMIC-IV Note 다운로드 | ✅ 완료 | discharge.csv, radiology.csv |
| MIMIC-IV 스키마 정리 | ✅ 완료 | 9개 테이블 컬럼/샘플/행수 검증 |
| SageMaker 학습 스크립트 | ✅ 준비 완료 | sagemaker/train.py, run_sagemaker.py |
| SageMaker GPU 학습 | ❌ 미진행 | S3 업로드 + 실행 필요 |
| MIMIC-IV 전처리 | ❌ 미진행 | 폐질환 환자 필터링 + Lab 추출 |
| Ollama 설치 | ❌ 미진행 | NLP/리포트 생성에 필요 |
| 전체 파이프라인 통합 테스트 | ❌ 미진행 | 모든 모듈 연결 |

---

## 다음 해야 할 일 (단계별)

### Phase 1: SageMaker로 모델 학습 (성능 향상)

```bash
# 1. CheXpert 데이터를 S3에 업로드
aws s3 sync data/CheXpert-v1.0-small/ s3://your-bucket/chexpert/

# 2. SageMaker 학습 시작
cd sagemaker/
python run_sagemaker.py

# 3. 학습 완료 후 가중치 다운로드
aws s3 cp s3://your-bucket/output/model.pth models/chexnet_best.pth
```

### Phase 2: MIMIC-IV 데이터 전처리

```bash
# 1. 압축 해제 (필요한 파일만)
cd data/mimic-iv/hosp/
gunzip d_labitems.csv.gz d_icd_diagnoses.csv.gz diagnoses_icd.csv.gz patients.csv.gz

# 2. 폐질환 환자 필터링 + 혈액검사 추출
python explore_mimic.py
```

**전처리 목표:**
- diagnoses_icd에서 폐질환 ICD코드 환자 필터링
- 해당 환자의 labevents 혈액검사 수치 추출
- lab_genomic_agent.py에 입력할 수 있는 형태로 변환

### Phase 3: Ollama 설치 + NLP 연결

```bash
brew install ollama
ollama serve          # 터미널 1
ollama pull llama3.1  # 터미널 2
```

### Phase 4: 전체 파이프라인 통합 테스트

```bash
streamlit run app.py

# 브라우저에서 http://localhost:8501 접속
# X-ray 업로드 → 14개 소견 → Lab + NLP → 희귀질환 매칭 → 보고서
```

---

## 팀 역할 분담 (26.03.18 회의 기준)

| 역할 | 담당 | 주요 작업 |
|------|------|----------|
| 전체 파이프라인 | 팀 전체 | 흐름 확정 + 통합 테스트 |
| CheXNet 학습/평가 | - | CheXpert finetuning + SageMaker |
| MIMIC-IV 전처리 | 권미라, 양희인, 허태웅 | 스키마 정리 + 데이터 추출 |
| NLP (임상 소견) | - | 퇴원요약문 → HPO 추출 |
| 유전체 파이프라인 | - | 유전자 변이 확인 + 검사 추천 |

---

## 최종 목표 산출물

```
입력: 환자의 흉부 X-ray + (선택) 혈액검사/임상소견

출력:
  1. X-ray에서 발견된 이상 소견 + Grad-CAM 히트맵
  2. 가능성 높은 폐 관련 질병 리스트 (HPO 매칭 기반)
  3. 추가 검사 추천:
     "~한 질병의 가능성이 높으니 ~검사를 추가적으로 진행해보시는걸 추천드립니다."
  4. 희귀질환 가능성이 있는 경우:
     → 유전체검사 추천 제시
```

---

## 향후 확장 계획

- 뇌 관련 MRI 분석 파이프라인 추가 (동일 파이프라인 구조 유지 가능한지 검토)
- MIMIC-IV ICU 데이터로 중증도 계층화
- 환자별 치사율/가능성 예측 모델

---

## 설치 및 실행

```bash
# 라이브러리 설치
pip install streamlit torch torchvision pandas numpy pillow opencv-python scikit-learn ollama

# 지식베이스 생성 (최초 1회)
python knowledge_base.py

# 웹 UI 실행
streamlit run app.py

# 또는 터미널 배치 실행
python main.py
```

---

## 참여자

권미라, 박성수, 배기태, 양희인, 허태웅
